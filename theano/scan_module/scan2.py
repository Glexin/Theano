"""
This module provides the Scan Op.

Scanning is a general form of recurrence, which can be used for looping.
The idea is that you *scan* a function along some input sequence, producing
an output at each time-step that can be seen (but not modified) by the
function at the next time-step. (Technically, the function can see the
previous K  time-steps of your outputs and L time steps (from past and
future) of your inputs.

So for example, ``sum()`` could be computed by scanning the ``z+x_i``
function over a list, given an initial state of ``z=0``.

Special cases:

* A *reduce* operation can be performed by using only the last
  output of a ``scan``.
* A *map* operation can be performed by applying a function that
  ignores previous steps of the outputs.

Often a for-loop or while-loop can be expressed as a ``scan()`` operation,
and ``scan`` is the closest that theano comes to looping. The advantages
of using ``scan`` over `for` loops in python (amongs other) are:

* it allows the number of iterations to be part of the symbolic graph
* it allows computing gradients through the for loop
* there exist a bunch of optimizations that help re-write your loop
such that less memory is used and that it runs faster
* it ensures that data is not copied from host to gpu and gpu to
host at each step

The Scan Op should typically be used by calling any of the following
functions: ``scan()``, ``map()``, ``reduce()``, ``foldl()``,
``foldr()``.

There is a lot of variables in this code prefixed with heads below:
    ii : Inner input, input of inner function of scan op
    io : Inner output, output of inner function of scan op
    oi : Outer input, input of scan op
    oo : Outer output, output of scan op
    scfn :
    stfn :

"""
from __future__ import absolute_import, print_function, division
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "Frederic Bastien "
               "James Bergstra "
               "Pascal Lamblin "
               "Sihang Gao ")
__copyright__ = "(c) 2010, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"


import logging
import numpy as np
import warnings
from collections import OrderedDict

from theano.compat import ifilter, izip
from six import iteritems, integer_types
from theano.compile import SharedVariable, function
from theano import compile
from theano import gof
from theano.tensor import opt
from theano import tensor
from theano import config
from theano.updates import OrderedUpdates
from theano.compile import ops


from theano.scan_module import scan_op
from theano.scan_module import scan_utils
from theano.scan_module.scan_utils import safe_new, traverse

# Logging function for sending warning or info
_logger = logging.getLogger('theano.scan_module.scan')

def scan(fn,
         sequences=None,
         outputs_info=None,
         non_sequences=None,
         n_steps=None,
         truncate_gradient=-1,
         go_backwards=False,
         mode=None,
         name=None,
         profile=False,
         allow_gc=None,
         strict=False,
         return_list=False):
    """
    This function constructs and applies a Scan op to the provided
    arguments.

    Parameters
    ----------
    fn
        ``fn`` is a function that describes the operations involved in one
        step of ``scan``. ``fn`` should construct variables describing the
        output of one iteration step. It should expect as input theano
        variables representing all the slices of the input sequences
        and previous values of the outputs, as well as all other arguments
        given to scan as ``non_sequences``. The order in which scan passes
        these variables to ``fn``  is the following :

        * all time slices of the first sequence
        * all time slices of the second sequence
        * ...
        * all time slices of the last sequence
        * all past slices of the first output
        * all past slices of the second otuput
        * ...
        * all past slices of the last output
        * all other arguments (the list given as `non_sequences` to
            scan)

        The order of the sequences is the same as the one in the list
        `sequences` given to scan. The order of the outputs is the same
        as the order of ``outputs_info``. For any sequence or output the
        order of the time slices is the same as the one in which they have
        been given as taps. For example if one writes the following :

        .. code-block:: python

            scan(fn, sequences = [ dict(input= Sequence1, taps = [-3,2,-1])
                                 , Sequence2
                                 , dict(input =  Sequence3, taps = 3) ]
                   , outputs_info = [ dict(initial =  Output1, taps = [-3,-5])
                                    , dict(initial = Output2, taps = None)
                                    , Output3 ]
                   , non_sequences = [ Argument1, Argument2])

        ``fn`` should expect the following arguments in this given order:

        #. ``Sequence1[t-3]``
        #. ``Sequence1[t+2]``
        #. ``Sequence1[t-1]``
        #. ``Sequence2[t]``
        #. ``Sequence3[t+3]``
        #. ``Output1[t-3]``
        #. ``Output1[t-5]``
        #. ``Output3[t-1]``
        #. ``Argument1``
        #. ``Argument2``

        The list of ``non_sequences`` can also contain shared variables
        used in the function, though ``scan`` is able to figure those
        out on its own so they can be skipped. For the clarity of the
        code we recommend though to provide them to scan. To some extend
        ``scan`` can also figure out other ``non sequences`` (not shared)
        even if not passed to scan (but used by `fn`). A simple example of
        this would be :

        .. code-block:: python

            import theano.tensor as TT
            W   = TT.matrix()
            W_2 = W**2
            def f(x):
                return TT.dot(x,W_2)

        The function is expected to return two things. One is a list of
        outputs ordered in the same order as ``outputs_info``, with the
        difference that there should be only one output variable per
        output initial state (even if no tap value is used). Secondly
        `fn` should return an update dictionary (that tells how to
        update any shared variable after each iteration step). The
        dictionary can optionally be given as a list of tuples. There is
        no constraint on the order of these two list, ``fn`` can return
        either ``(outputs_list, update_dictionary)`` or
        ``(update_dictionary, outputs_list)`` or just one of the two (in
        case the other is empty).

        To use ``scan`` as a while loop, the user needs to change the
        function ``fn`` such that also a stopping condition is returned.
        To do so, he/she needs to wrap the condition in an ``until`` class.
        The condition should be returned as a third element, for example:

        .. code-block:: python

            ...
            return [y1_t, y2_t], {x:x+1}, theano.scan_module.until(x < 50)

        Note that a number of steps (considered in here as the maximum
        number of steps ) is still required even though a condition is
        passed (and it is used to allocate memory if needed). = {}):

    sequences
        ``sequences`` is the list of Theano variables or dictionaries
        describing the sequences ``scan`` has to iterate over. If a
        sequence is given as wrapped in a dictionary, then a set of optional
        information can be provided about the sequence. The dictionary
        should have the following keys:

        * ``input`` (*mandatory*) -- Theano variable representing the
          sequence.

        * ``taps`` -- Temporal taps of the sequence required by ``fn``.
          They are provided as a list of integers, where a value ``tap``
          impiles that at iteration step ``t`` scan will pass to ``fn``
          the slice ``t+tap``. Default value is ``[0]``

        Any Theano variable in the list ``sequences`` is automatically
        wrapped into a dictionary where ``taps`` is set to ``[0]``

    outputs_info
        ``outputs_info`` is the list of Theano variables or dictionaries
        describing the initial state of the outputs computed
        recurrently. When this initial states are given as dictionary
        optional information can be provided about the output corresponding
        to these initial states. The dictionary should have the following
        keys:

        * ``initial`` -- Theano variable that represents the initial
          state of a given output. In case the output is not computed
          recursively (think of a map) and does not require an initial
          state this field can be skipped. Given that (only) the previous
          time step of the output is used by ``fn``, the initial state
          **should have the same shape** as the output and **should not
          involve a downcast** of the data type of the output. If multiple
          time taps are used, the initial state should have one extra
          dimension that should cover all the possible taps. For example
          if we use ``-5``, ``-2`` and ``-1`` as past taps, at step 0,
          ``fn`` will require (by an abuse of notation) ``output[-5]``,
          ``output[-2]`` and ``output[-1]``. This will be given by
          the initial state, which in this case should have the shape
          (5,)+output.shape. If this variable containing the initial
          state is called ``init_y`` then ``init_y[0]`` *corresponds to*
          ``output[-5]``. ``init_y[1]`` *correponds to* ``output[-4]``,
          ``init_y[2]`` corresponds to ``output[-3]``, ``init_y[3]``
          coresponds to ``output[-2]``, ``init_y[4]`` corresponds to
          ``output[-1]``. While this order might seem strange, it comes
          natural from splitting an array at a given point. Assume that
          we have a array ``x``, and we choose ``tap`` to be time step
          ``0``. Then our initial state would be ``x[:tap]``, while the
          output will be ``x[tap:]``. Looking at this split, elements in
          ``x[:tap]`` are ordered exactly like those in ``init_y``.
        * ``taps`` -- Temporal taps of the output that will be pass to
          ``fn``. They are provided as a list of *negative* integers,
          where a value ``tap`` implies that at iteration step ``t`` scan
          will pass to ``fn`` the slice ``t+tap``.

        ``scan`` will follow this logic if partial information is given:

        * If an output is not wrapped in a dictionary, ``scan`` will wrap
          it in one assuming that you use only the last step of the output
          (i.e. it makes your tap value list equal to [-1]).
        * If you wrap an output in a dictionary and you do not provide any
          taps but you provide an initial state it will assume that you are
          using only a tap value of -1.
        * If you wrap an output in a dictionary but you do not provide any
          initial state, it assumes that you are not using any form of
          taps.
        * If you provide a ``None`` instead of a variable or a empty
          dictionary ``scan`` assumes that you will not use any taps for
          this output (like for example in case of a map)

        If ``outputs_info`` is an empty list or None, ``scan`` assumes
        that no tap is used for any of the outputs. If information is
        provided just for a subset of the outputs an exception is
        raised (because there is no convention on how scan should map
        the provided information to the outputs of ``fn``)

    non_sequences
        ``non_sequences`` is the list of arguments that are passed to
        ``fn`` at each steps. One can opt to exclude variable
        used in ``fn`` from this list as long as they are part of the
        computational graph, though for clarity we encourage not to do so.

    n_steps
        ``n_steps`` is the number of steps to iterate given as an int
        or Theano scalar. If any of the input sequences do not have
        enough elements, scan will raise an error. If the *value is 0* the
        outputs will have *0 rows*. If n_steps is not provided, ``scan`` will
        figure out the amount of steps it should run given its input
        sequences. ``n_steps`` < 0 is not supported anymore.

    truncate_gradient
        ``truncate_gradient`` is the number of steps to use in truncated
        BPTT.  If you compute gradients through a scan op, they are
        computed using backpropagation through time. By providing a
        different value then -1, you choose to use truncated BPTT instead
        of classical BPTT, where you go for only ``truncate_gradient``
        number of steps back in time.

    go_backwards
        ``go_backwards`` is a flag indicating if ``scan`` should go
        backwards through the sequences. If you think of each sequence
        as indexed by time, making this flag True would mean that
        ``scan`` goes back in time, namely that for any sequence it
        starts from the end and goes towards 0.

    name
        When profiling ``scan``, it is crucial to provide a name for any
        instance of ``scan``. The profiler will produce an overall
        profile of your code as well as profiles for the computation of
        one step of each instance of ``scan``. The ``name`` of the instance
        appears in those profiles and can greatly help to disambiguate
        information.

    mode
        It is recommended to leave this argument to None, especially
        when profiling ``scan`` (otherwise the results are not going to
        be accurate). If you prefer the computations of one step of
        ``scan`` to be done differently then the entire function, you
        can use this parameter to describe how the computations in this
        loop are done (see ``theano.function`` for details about
        possible values and their meaning).

    profile
        Flag or string. If true, or different from the empty string, a
        profile object will be created and attached to the inner graph of
        scan. In case ``profile`` is True, the profile object will have the
        name of the scan instance, otherwise it will have the passed string.
        Profile object collect (and print) information only when running the
        inner graph with the new cvm linker ( with default modes,
        other linkers this argument is useless)

    allow_gc
        Set the value of allow gc for the internal graph of scan.  If
        set to None, this will use the value of config.scan.allow_gc.

        The full scan behavior related to allocation is determined by
        this value and the Theano flag allow_gc. If the flag allow_gc
        is True (default) and this scan parameter allow_gc is False
        (default), then we let scan allocate all intermediate memory
        on the first iteration, those are not garbage collected them
        during that first iteration (this is determined by the scan
        allow_gc). This speed up allocation of the following
        iteration. But we free all those temp allocation at the end of
        all iterations (this is what the Theano flag allow_gc mean).

        If you use cnmem and this scan is on GPU, the speed up from
        the scan allow_gc is small. If you are missing memory, disable
        the scan allow_gc could help you run graph that request much
        memory.

    strict
        If true, all the shared variables used in ``fn`` must be provided as a
        part of ``non_sequences`` or ``sequences``.

    return_list
        If True, will always return a list, even if there is only 1 output.

    Returns
    -------
    tuple
        Tuple of the form (outputs, updates); ``outputs`` is either a
        Theano variable or a list of Theano variables representing the
        outputs of ``scan`` (in the same order as in ``outputs_info``).
        ``updates`` is a subclass of dictionary specifying the update rules for
        all shared variables used in scan.
        This dictionary should be passed to ``theano.function`` when you compile
        your function. The change compared to a normal dictionary is that we
        validate that keys are SharedVariable and addition of those dictionary
        are validated to be consistent.

    """
    # General observation : this code is executed only once, at creation
    # of the computational graph, so we don't yet need to be smart about
    # anything (to speed things up)

    ##
    # Step 1. Check, unify and extract infomations from parameters.
    ##
    scf_name = name

    def wrap_into_list(x):
        """
        Wrap the input into a list if it is not already a list.

        """
        if x is None:
            return []
        elif not isinstance(x, (list, tuple)):
            return [x]
        else:
            return list(x)

    # Unify sequences.
    seqs = wrap_into_list(sequences)
    # wrap sequences in a dictionary if they are not already dictionaries
    for i, seq_info in enumerate(seqs):
        if not isinstance(seq_info, dict):
            seqs[i] = OrderedDict([('input', seq_info), ('taps', [0])])
        elif seq_info.get('taps', None) is not None:
            seq_info['taps'] = wrap_into_list(seq_info['taps'])
        else:
            # seqs dictionary does not have the ``taps`` key
            seq_info['taps'] = [0]

    # Unify ``outputs_info``
    outs_info = wrap_into_list(outputs_info)
    # wrap outputs info in a dictionary if they are not already in one
    for i, out_info in enumerate(outs_info):
        if out_info is not None:
            if isinstance(out_info, dict):
                # DEPRECATED :
                if out_info.get('return_steps', None) is not None:
                    raise ValueError(
                            "Using `return_steps` has been deprecated. "
                            "Simply select the entries you need using a "
                            "subtensor. Scan will optimize memory "
                            "consumption, so do not worry about that.")
                # END

            if not isinstance(out_info, dict):
                # by default any output has a tap value of -1
                outs_info[i] = OrderedDict([('initial', out_info),
                                            ('taps', [-1])])
            elif (out_info.get('initial', None) is None and
                    out_info.get('taps', None) is not None):
                # ^ no initial state but taps provided
                raise ValueError(('If you are using slices of an output '
                                  'you need to provide a initial state '
                                  'for it'), out_info)
            elif (out_info.get('initial', None) is not None and
                  out_info.get('taps', None) is None):
                # ^ initial state but taps not provided
                if 'taps' in out_info:
                    # ^ explicitly provided a None for taps
                    _logger.warning('Output %s ( index %d) has a initial '
                            'state but taps is explicitly set to None ',
                            getattr(out_info['initial'], 'name', 'None'),
                            i)
                out_info['taps'] = [-1]
            elif out_info.get('taps', None) is not None:
                # Check that taps are valid (< 0 and all dfferent)
                taps = out_info['taps']
                if len(taps) > len(set(taps)):
                    raise ValueError(('All the taps must be different in '
                                      ' `outputs_info`'), out_info)
                for t in taps:
                    if t >= 0:
                        raise ValueError(('All the tap values must be '
                                          'smaller than 0.'), out_info)
            else:
                # Both no inital state and taps.
                # In cases that outputs are not used as inputs for 'fn'
                pass
        else:
            # if a None is provided as the output info we replace it
            # with an empty OrdereDict() to simplify handling
            # In cases that outputs are not used as inputs for 'fn'
            outs_info[i] = OrderedDict()

    # Unify non_sequences
    non_seqs = wrap_into_list(non_sequences)
    # Make sure we get rid of numpy arrays or ints or anything like that
    # passed as inputs to scan
    for i, non_seq in enumerate(non_seqs):
        if not isinstance(non_seq, gof.Variable):
            non_seqs[i] = tensor.as_tensor_variable(non_seq)
        else:
            non_seqs[i] = non_seq

    # Handle n_steps
    # Check n_steps is an int
    if (hasattr(n_steps, 'dtype') and
        str(n_steps.dtype) not in tensor.integer_dtypes):
        raise ValueError(' n_steps must be an int. dtype provided '
                         'is %s' % n_steps.dtype)

    # Try to extract real value of n_steps before compilation.
    # TODO add one step optimization explanation.
    n_fixed_steps = None
    if isinstance(n_steps, (float, integer_types)):
        n_fixed_steps = int(n_steps)
    else:
        try:
            n_fixed_steps = opt.get_scalar_constant_value(n_steps)
        except tensor.basic.NotScalarConstantError:
            n_fixed_steps = None

    ##
    # Step 2. Find hidden variables(non sequences)
    ##

    # TODO combine docs in scan_new487-491
    svm = scan_utils.ScanVarMap()
    ois = svm.create_list(svm.OI_TYPE, listen=True)
    iis = svm.create_list(svm.II_TYPE, listen=True)
    ios = svm.create_list(svm.IO_TYPE, listen=True)
    # original dynamic shareds in oos order.
    oos_ori_dyshared = svm.create_list(svm.OO_TYPE, listen=True)
    oos_t0_idx = svm.create_list(svm.OO_TYPE, listen=True)
    # TODO add osi (one step input) doc
    # TODO make sure 'direct' is fit with STFNI_TYPE
    osis = svm.create_list(svm.STFNI_TYPE, listen=True)
    for i, seq_info in enumerate(seqs):
        taps = seq_info["taps"]
        seq_input = seq_info['input']
        seq_var = tensor.as_tensor_variable(seq_input)
        meta = dict(name="seq%d"%i,
                    tag=svm.SEQ_COMBO_TAG,
                    n_in=len(taps),
                    scfn_rank=i)
        abst_idx = svm.add_var(meta)
        # TODO test mintap maxtap bug
        mintap = np.min(taps)
        maxtap = np.min(taps)
        # TODO modify doc below
        # We cut the sequence such that seq[i] to correspond to
        # seq[i-tap]. For the purposes of cutting the sequences, we
        # need to pretend tap 0 is used to avoid cutting the sequences
        # too long if the taps are all lower or all higher than 0.
        if mintap > 0:
            t0_idx = 0
        else:
            t0_idx = -mintap
        # ``tn1_offset`` is offset of t=last_step relative to seq[-1].
        if maxtap < 0:
            tn1_offset = 0
        else:
            tn1_offset = -maxtap
        # TODO multi tap seq lead to mitmot grad?
        for tap_i, tap in enumerate(taps):
            tap_t0_idx = t0_idx + tap

            ## osi_slice
            osi_slice = seq_input[tap_t0_idx]
            svm.set_list_by_entry(svm.STFNI_TYPE, osis, abst_idx, tap_i,
                    osi_slice)

            ## TODO ii_slice doc
            seq_slice = seq_var[tap_t0_idx]
            ii_slice = seq_slice.type()
            # Try to transfer test_value to the new variable
            if config.compute_test_value != 'off':
                try:
                    ii_slice.tag.test_value = gof.Op._get_test_value(
                        seq_slice)
                except AttributeError as e:
                    if config.compute_test_value != 'ignore':
                        # No need to print a warning or raise an error now,
                        # it will be done when fn will be called.
                        _logger.info(('Cannot compute test value for '
                            'the inner function of scan, input value '
                            'missing %s'), e)
            # Add names to slices for debugging and pretty printing ..
            # that is if the input already has a name
            if getattr(seq_input, 'name', None) is not None:
                if tap > 0:
                    nw_name = seq_input.name + '[t+%d]' % tap
                elif tap == 0:
                    nw_name = seq_input.name + '[t]'
                else:
                    nw_name = seq_input.name + '[t%d]' % tap
                ii_slice.name = nw_name
            svm.set_list_by_entry(svm.II_TYPE, iis, abst_idx, tap_i, ii_slice)

            ## oi_seq doc
            start = tap + t0_idx
            end = tap + tn1_offset
            if end == 0:
                oi_seq = seq_input[start:]
                if getattr(seq_input, 'name', None) is not None:
                    oi_seq.name = seq_input.name + "[%d:]" % start
            else:
                oi_seq = seq_input[start:end]
                if getattr(seq_input, 'name', None) is not None:
                    oi_seq.name = seq_input.name + "[%d:%d]" % (start, end)
            if go_backwards:
                oi_seq = oi_seq[::-1]
            svm.set_list_by_entry(svm.OI_TYPE, ois, abst_idx, tap_i, oi_seq)

    # Inference ``actual_n_steps`` from ``n_steps`` and ``seqs``.
    if not scan_utils.isNaN_or_Inf_or_None(n_steps):
        # If the user has provided the number of steps, do that
        # regardless ( and raise an error later if the sequences are
        # not long enough )
        actual_n_steps = tensor.as_tensor(n_steps)
    else:
        oi_seqs = svm.select(svm.OI_TYPE, svm.SEQ_COMBO_TAG, ois)
        min_seq_len = None
        for oi_seq in oi_seqs:
            seq_len = oi_seq.shape[0]
            if min_seq_len is None:
                min_seq_len = seq_len
            else:
                min_seq_len = tensor.minimum(min_seq_len, seq_len)
        if min_seq_len is None:
            # ^ No information about the number of steps
            raise ValueError('No information about the number of steps '
                            'provided. Either provide a value for '
                            'n_steps argument of scan or provide an input '
                            'sequence')
        actual_n_steps = min_seq_len
    abst_meta = dict(name='n_steps',
                     tag=svm.N_STEP_COMBO_TAG)
    abst_idx = svm.add_var(abst_meta)
    svm.set_list_by_entry(svm.OI_TYPE, ois, abst_idx, 0, actual_n_steps)

    # Since we've added all sequences now we need to level them up based on
    # n_steps or their different shapes
    # Operations below will raise an error if seq is too short for steps
    # while evaluating.
    nw_oi_seqs = [seq_info[:actual_n_steps] for seq_info in oi_seqs]
    svm.set_list_by_tag(ois, svm.OI_TYPE, svm.SEQ_COMBO_TAG, nw_oi_seqs)

    ## Handle ``output_infos``
    # go through outputs picking up time slices as needed

    def add_nitsot(scfn_rank):
        abst_meta = dict(name="out_%d_nitsot"%scfn_rank,
                         tag=svm.NITSOT_COMBO_TAG,
                         scfn_rank=scfn_rank)
        abst_idx = svm.add_var(abst_meta)
        # We set entry of nitsot with actual_n_steps for later memory
        # allocation.
        svm.set_list_by_entry(svm.OI_TYPE, ois, abst_idx, 0, actual_n_steps)

    sitsot_taps = []
    mitsot_taps = []
    for i, out_info in enumerate(outs_info):
        taps = out_info.get('taps', None)
        # Note that our convention dictates that if an output uses
        # just the previous time step, as a initial state we will only
        # provide a tensor of the same dimension as one time step; This
        # makes code much cleaner for those who do not use taps. Otherwise
        # they would always had to shape_padleft the initial state ..
        # which is ugly
        if taps == [-1]: # sitsot
            sitsot_taps.append(taps)
            abst_meta = dict(name="out_%d_sitsot"%i, tag=svm.SITSOT_COMBO_TAG,
                scfn_rank=i)
            abst_idx = svm.add_var(abst_meta)
            svm.set_list_by_entry(svm.OO_TYPE, oos_t0_idx, abst_idx, 0, 1)
            actual_arg = out_info['initial']
            if not isinstance(actual_arg, tensor.Variable):
                actual_arg = tensor.as_tensor_variable(actual_arg)
            arg = safe_new(actual_arg)
            if isinstance(arg, tensor.Constant):
                # safe new returns a clone of the constants, but that is not
                # what we need for initial states
                arg = arg.type()

            # Try to transfer test_value to the new variable
            if config.compute_test_value != 'off':
                try:
                    arg.tag.test_value = gof.Op._get_test_value(actual_arg)
                except AttributeError as e:
                    if config.compute_test_value != 'ignore':
                        # No need to print a warning or raise an error now,
                        # it will be done when fn will be called.
                        _logger.info(('Cannot compute test value for the '
                            'inner function of scan, input value missing %s'),
                                     e)

            if getattr(out_info['initial'], 'name', None) is not None:
                arg.name = out_info['initial'].name + '[t-1]'

            # TODO doc
            # We need now to allocate space for storing the output and copy
            # the initial state over. We do this using the expand function
            # defined in scan utils
            sitsot_oi = scan_utils.expand_empty(
                tensor.unbroadcast(
                    tensor.shape_padleft(actual_arg), 0),
                actual_n_steps)
            svm.set_list_by_entry(svm.OI_TYPE, ois, abst_idx, 0, sitsot_oi)
            svm.set_list_by_entry(svm.II_TYPE, iis, abst_idx, 0, arg)
            svm.set_list_by_entry(svm.STFNI_TYPE, osis, abst_idx, 0, actual_arg)
        elif taps: # mitsot
            mitsot_taps.append(taps)
            abst_meta = dict(name="out_%d_mitsot"%i, tag=svm.MITSOT_COMBO_TAG,
                n_in=len(taps), scfn_rank=i)
            abst_idx = svm.add_var(abst_meta)
            # TODO tap = 0 error
            if np.any(np.array(taps) > 0):
                # Make sure we do not have requests for future values of a
                # sequence we can not provide such values
                raise ValueError('Can not use future taps of outputs',
                                    out_info)
            init_out = out_info['initial']
            # go through the taps
            t0_idx = -np.min(taps)
            svm.set_list_by_entry(svm.OO_TYPE, oos_t0_idx, abst_idx, 0, t0_idx)
            # Sequence
            # TODO check initial length == initlen
            # TODO is expand_empty gradable to initial
            # TODO one var in ois for mitsot??? If yes ,why diff with
            # seq.
            mitsot_oi = scan_utils.expand_empty(init_out[:t0_idx],
                                        actual_n_steps)
            svm.set_list_by_entry(svm.OI_TYPE, ois, abst_idx, 0, mitsot_oi)
            init_out_var = tensor.as_tensor_variable(init_out)
            for tap_i, tap in enumerate(taps):
                tap_t0_idx = t0_idx + tap
                osi_slice = init_out[tap_t0_idx]
                _ii_slice = init_out_var[tap_t0_idx]
                ii_slice = _ii_slice.type()

                # Try to transfer test_value to the new variable
                if config.compute_test_value != 'off':
                    try:
                        ii_slice.tag.test_value = gof.Op._get_test_value(
                                _ii_slice)
                    except AttributeError as e:
                        if config.compute_test_value != 'ignore':
                            # No need to print a warning or raise an
                            # error now, it will be done when fn will be
                            # called.
                            _logger.info(('Cannnot compute test value for '
                                'the inner function of scan, input value '
                                'missing. %s'), e)

                # give it a name or debugging and pretty printing
                if getattr(init_out, 'name', None) is not None:
                    assert tap < 0
                    ii_slice.name = init_out.name + '[t%d]' % tap
                svm.set_list_by_entry(svm.II_TYPE, iis, abst_idx, tap_i, ii_slice)
                svm.set_list_by_entry(svm.STFNI_TYPE, osis, abst_idx, tap_i, osi_slice)
        else: # nitsot
            # NOTE: there is another case, in which we don't want to provide
            #      any previous value of the output to the inner function
            #      (i.e. a map); in that case we do not have to to anything
            add_nitsot(i)

    if n_fixed_steps in [1, -1]:
        stfn_args = osis
    else:
        stfn_args = svm.map(svm.II_TYPE, svm.STFNI_TYPE, iis)

    # when we apply the lambda expression we get a mixture of update
    # rules and outputs that needs to be seperated
    condition, outputs, updates = \
            scan_utils.get_updates_and_outputs(fn(*stfn_args))
    ios = svm.map(svm.SCFNI_OUTPUT_TYPE, svm.IO_TYPE, outputs)

    ##
    # Step 3. Check if we actually need scan and remove it if we don't
    ##
    if n_fixed_steps in [1, -1]:
        # We don't need to use the scan op anymore, so we can just
        # return the outputs and updates we have
        if condition is not None:
            _logger.warning(('When the number of steps is fixed and '
                    'equal to 1, the provided stopping condition, ',
                    str(condition), ' is ignored'))

            for pos, inner_out in enumerate(outputs):
                # TODO
                # We need to see if we need to pad our sequences with
                # an unbroadcastable dimension; case excample : we
                # return an output for which we want all intermediate.
                # If n_steps is 1 then, if we return the output as given
                # by inner function this will represent only a slice and
                # it will have one dimension less.
                if isinstance(init_out.type, tensor.TensorType):
                    outputs[pos] = tensor.unbroadcast(
                            tensor.shape_padleft(inner_out), 0)
                    if return_list is not True and len(outputs) == 1:
                        outputs = outputs[0]
        return (outputs, updates)
    if condition:
        as_while = True
    else:
        as_while = False

    ##
    # TODO Create dummy graph to find hidden inputs.
    ##

    # We can now compile a dummy function just to see what shared variable
    # we have and what are their update rules (note that the user has
    # the option not to pass the shared variable to scan, so we need to
    # pick them manually and add them to scan)
    # make the compilation as fast as possible by not applying any
    # optimization or conversion to C [ note this region is not important
    # for performance so we can do stuff as unoptimal as we wish ]

    # extract still missing inputs (there still might be so) and add them
    # as non sequences at the end of our args
    fake_nonseqs = [x.type() for x in non_seqs] # make nonseqs independent
    all_outs = list(outputs)
    if condition is not None:
        all_outs.append(condition)
    # TODO add update hidden test
    update_outs = []
    for k, v in updates.items():
        update_outs.append(v)
    all_outs += update_outs
    # some vars can be put in 'fn' implicitly, they may be inputs
    fake_outputs = scan_utils.clone(all_outs,
                                    replace=OrderedDict(izip(non_seqs,
                                                             fake_nonseqs)))
    # some vars can be put in 'fn' implicitly, they may be inputs
    # or depend on some inputs, we add them in.
    all_inputs = ifilter(
        lambda x: (isinstance(x, gof.Variable) and
                   not isinstance(x, SharedVariable) and
                   not isinstance(x, gof.Constant)),
        gof.graph.inputs(fake_outputs))
    extra_inputs = [x for x in all_inputs if x not in stfn_args + fake_nonseqs]

    # add only the non-shared variables and non-constants to the
    # arguments of the dummy function [ a function should not get shared
    # variables or constants as input ]
    dummy_args = [arg for arg in stfn_args
            if (not isinstance(arg, SharedVariable) and
                not isinstance(arg, tensor.Constant))]
    dummy_args += extra_inputs
    dummy_outs = outputs
    # Perform a try-except to provide a meaningful error message to the
    # user if inputs of the inner function are missing.
    try:
        dummy_f = function(dummy_args,
                           dummy_outs,
                           updates=updates,
                           mode=compile.mode.Mode(linker='py',
                                                  optimizer=None),
                           on_unused_input='ignore',
                           profile=False)
    except gof.fg.MissingInputError as err:
        msg = ("\nPlease pass this variable to the scan's inner "
               "function. Do not forget to also pass it to the "
               "`non_sequences` attribute of scan.")
        raise gof.fg.MissingInputError(err.args[0] + msg)

    ##
    # Step 5. Re-arange inputs of scan into more strict order
    ##

    # Step 5.0 Check the outputs of the dummy function to see if they
    # match with ``outputs_info`` parameter

    # if the number of outputs to the function does not match the number of
    # assumed outputs until now (provided by the user) there can be
    # only one explanation: No information is provided for any of the
    # outputs (i.e. we are dealing with a map)
    n_dummy_outs = len(dummy_f.maker.outputs)
    if as_while:
        n_dummy_outs -= 1
    n_dummy_outs -= len(update_outs)
    if len(outs_info) != 0:
        if n_dummy_outs != len(outs_info):
            raise ValueError('Please provide None as outputs_info for '
                            'any output that does not feed back into '
                            'scan (i.e. it behaves like a map) ')
    else:
        outs_info = [OrderedDict() for x in xrange(n_dummy_outs)]
        for scfn_rank in range(n_dummy_outs):
            add_nitsot(scfn_rank)

    # Step 5.3 Outputs that correspond to update rules of shared variables
    n_ext_dys = 0
    # TODO
    givens = {}
    for input in dummy_f.maker.expanded_inputs:
        original_var = input.variable
        if isinstance(original_var, SharedVariable) and input.update:
            # dynamic shared variable
            new_var = safe_new(original_var)
            if getattr(original_var, 'name', None) is not None:
                new_var.name = original_var.name + "_copy"
            # ops.expandable_types are list of types which is capable to
            # add extra dimemsion, till when I'm writing this code,
            # it only contains TensorType
            if isinstance(new_var.type, ops.expandable_types):
                # expandable dynamic shared(sitsot shared)
                abst_meta = dict(name="ext_dynamic_shard_%d"%n_ext_dys,
                                 tag=svm.SITSOT_SHARED_COMBO_TAG)
                abst_idx = svm.add_var(abst_meta)
                svm.set_list_by_entry(svm.II_TYPE, iis, abst_idx, 0, new_var)
                expanded = scan_utils.expand_empty(
                    tensor.unbroadcast(
                        tensor.shape_padleft(original_var), 0),
                    actual_n_steps)
                svm.set_list_by_entry(svm.OI_TYPE, ois, abst_idx, 0, expanded)
                update_var = tensor.as_tensor_variable(input.update)
                svm.set_list_by_entry(svm.IO_TYPE, ios, abst_idx, 0, update_var)
                svm.set_list_by_entry(svm.OO_TYPE, oos_ori_dyshared, abst_idx, 0, original_var)
                givens[original_var] = new_var
            else:
                # unexpandable dynamic shared
                name="unexpandable_dynamic_shard_%d"%n_ext_dys
                abst_meta = dict(name=name,
                                 tag=svm.DYNAMIC_SHARED_COMBO_TAG)
                abst_idx = svm.add_var(abst_meta)
                svm.set_list_by_entry(svm.II_TYPE, iis, abst_idx, 0, new_var)
                svm.set_list_by_entry(svm.OI_TYPE, ois, abst_idx, 0, original_var)
                svm.set_list_by_entry(svm.IO_TYPE, ios, abst_idx, 0, input.update)
                givens[original_var] = new_var # givens[oi] = ii


    # handle explicit non_seq inputs
    for scfn_rank, non_seq in enumerate(non_seqs):
        if (not isinstance(non_seq, SharedVariable) and
            not isinstance(non_seq, tensor.Constant)):
            name = "nonseq_%d_input" % scfn_rank
            abst_meta = dict(name=name,
                             tag=svm.NS_INPUT_COMBO_TAG,
                             scfn_rank=scfn_rank)
            abst_idx = svm.add_var(abst_meta)
            ns_input_ii = safe_new(non_seq, '_copy')
            svm.set_list_by_entry(svm.II_TYPE, iis, abst_idx, 0, ns_input_ii)
            svm.set_list_by_entry(svm.OI_TYPE, ois, abst_idx, 0, non_seq)
            givens[non_seq] = ns_input_ii

    # handle hidden non_seq inputs
    for i, _input in enumerate(extra_inputs):
        name = "hidden_nonseq_input_%d" % i
        abst_meta = dict(name=name,
                            tag=svm.HIDDEN_NS_INPUT_COMBO_TAG)
        abst_idx = svm.add_var(abst_meta)
        input_ii = safe_new(_input, '_copy')
        svm.set_list_by_entry(svm.II_TYPE, iis, abst_idx, 0, input_ii)
        svm.set_list_by_entry(svm.OI_TYPE, ois, abst_idx, 0, _input)
        givens[_input] = input_ii

    # handle static shared vars
    static_shareds_info = [arg for arg in dummy_f.maker.expanded_inputs
                      if isinstance(arg, SharedVariable) and not arg.update]

    # handle explicit static shared vars
    static_shared_vars = [var_info.variable for var_info in static_shareds_info]
    static_shared_vars = set(static_shared_vars)
    for scfn_rank, non_seq in enumerate(non_seqs):
        if non_seq in static_shared_vars:
            name = "nonseq_%d_static_shared" % scfn_rank
            abst_meta = dict(name=name,
                             tag=svm.SHARED_COMBO_TAG,
                             scfn_rank=scfn_rank)
            abst_idx = svm.add_var(abst_meta)
            svm.set_list_by_entry(svm.OI_TYPE, ois, abst_idx, 0, non_seq)
            non_seq_ii = safe_new(non_seq, '_copy')
            svm.set_list_by_entry(svm.II_TYPE, iis, abst_idx, 0, non_seq_ii)
            givens[non_seq] = non_seq_ii

    # handle hidden static shared vars
    if not strict:
        # TODO : What does ``strict`` mean, why does it not applied on
        # inputs or other nonseqs.

        # In strict mode, shared vars used not in ``non_sequences`` will be
        # omit, will cause exception in function compiling later.
        non_seq_set = set(non_seqs)
        for i, static_shared_var in enumerate(static_shared_vars):
            name = "hidden_static_shared_%d" % i
            abst_meta = dict(name=name,
                             tag=svm.HIDDEN_SHARED_COMBO_TAG)
            abst_idx = svm.add_var(abst_meta)
            svm.set_list_by_entry(svm.OI_TYPE, ois, abst_idx, 0, static_shared_var)
            ii_var = safe_new(static_shared_var, '_copy')
            svm.set_list_by_entry(svm.II_TYPE, iis, abst_idx, 0, ii_var)
            givens[static_shared_var] = ii_var

    # handle condition var
    if condition is not None:
        abst_meta = dict(name="condition",
                         tag=svm.CONDITION_COMBO_TAG)
        abst_idx = svm.add_var(abst_meta)
        svm.set_list_by_entry(svm.OI_TYPE, ois, abst_idx, 0, condition)

    # gpuarray is imported here, instead of being imported on top of
    # the file because that would force on the user some dependencies that
    # we might do not want to. Currently we are working on removing the
    # dependencies on sandbox code completely.
    from theano import gpuarray
    if gpuarray.pygpu_activated:
        # very often we end up in this situation when we want to replace
        # oi_var with ii_var, where oi_var is a GPU variable and ii_var is
        # TensorType. This is because shared variables are put on GPU
        # right away >:|,
        new_givens = OrderedDict()

        for oi_var, ii_var in givens.items():
            if (isinstance(oi_var.type, gpuarray.GpuArrayType) and
                isinstance(ii_var.type, tensor.TensorType)):
                for io_var in ios:
                    # TODO what does that do.
                    new_givens = traverse(io_var, oi_var, ii_var, new_givens)
            else:
                new_givens[oi_var] = ii_var
    else:
        new_givens = givens

    print("\nDEBUG")
    print(ios)
    print(new_givens)
    new_ios = scan_utils.clone(ios, replace=new_givens)

    #* Create the Scan Op
    tap_array = mitsot_taps + sitsot_taps
    if allow_gc is None:
        allow_gc = config.scan.allow_gc
    info = OrderedDict()

    # TODO convert tap_array to ii type
    info['tap_array'] = tap_array
    info['n_seqs'] = len(svm.select_by_tag(svm.OI_TYPE, svm.SEQ_COMBO_TAG))
    info['n_mit_mot'] = 0
    info['n_mit_mot_outs'] = 0
    info['mit_mot_out_slices'] = []
    info['n_mit_sot'] = len(svm.select_by_tag(svm.SCFNI_OUTPUT_TYPE, svm.MITSOT_COMBO_TAG))
    info['n_sit_sot'] = len(svm.select_by_tag(svm.II_TYPE, svm.SITSOT_COMBO_TAG))
    info['n_shared_outs'] = len(svm.select_by_tag(svm.OI_TYPE, svm.DYNAMIC_SHARED_COMBO_TAG))
    info['n_nit_sot'] = len(svm.select_by_tag(svm.OO_TYPE, svm.NITSOT_COMBO_TAG))
    info['truncate_gradient'] = truncate_gradient
    info['name'] = scf_name
    info['mode'] = mode
    info['destroy_map'] = OrderedDict()
    info['gpua'] = False
    info['as_while'] = as_while
    info['profile'] = profile
    info['allow_gc'] = allow_gc
    info['strict'] = strict

    local_op = scan_op.Scan(iis, new_ios, info)

    ##
    # Compute the outputs using the scan op
    ##

    new_ois = []
    for arg in ois:
        try:
            arg = tensor.as_tensor_variable(arg)
        except TypeError:
            # This happens for RandomStates for e.g. but it is a good
            # way to make sure all inputs are tensors.
            pass
        new_ois.append(arg)
    oos = local_op(*new_ois)
    if type(oos) not in (list, tuple):
        oos = [oos]

    ##
    # Collect final scan outputs and update rules.
    ##


    # Remove initials of outputs.
    for i, t0_idx, var in enumerate(zip(oos_t0_idx, oos)):
        if t0_idx != None and t0_idx != 0:
            oos[i] = var[t0_idx:]
    # collect outputs
    scan_outs = svm.map(svm.OO_TYPE, svm.SCFNO_TYPE, oos)
    # normalize
    if len(scan_outs) == 0:
        scan_outs = None
    elif return_list is not True and len(scan_outs) == 1:
        scan_outs = scan_outs[0]

    # extract dynamic shared updates
    update_map = OrderedDict()
    update_pairs = zip(oos_ori_dyshared, oos)
    update_pairs = svm.select_by_tag(svm.OO_TYPE, svm.SITSOT_SHARED_COMBO_TAG, update_pairs)
    for ori_shared, out_seq in update_pairs:
        update_map[ori_shared] = out_seq[-1]

    return scan_outs, update_pairs
