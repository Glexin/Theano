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
          They are provided as a list of integers, where a value ``k``
          impiles that at iteration step ``t`` scan will pass to ``fn``
          the slice ``t+k``. Default value is ``[0]``

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
          we have a array ``x``, and we choose ``k`` to be time step
          ``0``. Then our initial state would be ``x[:k]``, while the
          output will be ``x[k:]``. Looking at this split, elements in
          ``x[:k]`` are ordered exactly like those in ``init_y``.
        * ``taps`` -- Temporal taps of the output that will be pass to
          ``fn``. They are provided as a list of *negative* integers,
          where a value ``k`` implies that at iteration step ``t`` scan
          will pass to ``fn`` the slice ``t+k``.

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
    for i, seq in enumerate(seqs):
        if not isinstance(seq, dict):
            seqs[i] = OrderedDict([('input', seq), ('taps', [0])])
        elif seq.get('taps', None) is not None:
            seq['taps'] = wrap_into_list(seq['taps'])
        else:
            # seqs dictionary does not have the ``taps`` key
            seq['taps'] = [0]

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
    for i, non_seq in non_sequences:
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
    # TODO add osi
    # TODO make sure 'direct' is fit with STFNI_TYPE
    osis = svm.create_list(svm.STFNI_TYPE, listen=True)
    for i, seq in enumerate(seqs):
        taps = seqs["taps"]
        seq_val = seq['input']
        seq_var = tensor.as_tensor_variable(seq_val)
        meta = dict(name="seq%d"%i,
                    tag=svm.SEQ_COMBO_TAG,
                    n_in=len(taps),
                    scfn_rank=i)
        svm.add_var(meta)
        # TODO test mintap maxtap bug
        mintap = np.min(taps)
        maxtap = np.min(taps)
        # TODO modify doc below
        # We cut the sequence such that seq[i] to correspond to
        # seq[i-k]. For the purposes of cutting the sequences, we
        # need to pretend tap 0 is used to avoid cutting the sequences
        # too long if the taps are all lower or all higher than 0.
        if mintap > 0:
            t0_idx = 0
        else:
            t0_idx = -mintap
        if maxtap < 0:
            end_offset = 0
        else:
            end_offset = -maxtap
        # TODO multi tap seq lead to mitmot grad?
        for tap in taps:
            tap_t0_idx = t0_idx + tap
            os_slice = seq_val[tap_t0_idx]

            # TODO ii_slice doc
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
            if getattr(seq_val, 'name', None) is not None:
                if k > 0:
                    nw_name = seq['input'].name + '[t+%d]' % k
                elif k == 0:
                    nw_name = seq['input'].name + '[t]'
                else:
                    nw_name = seq['input'].name + '[t%d]' % k
                ii_slice.name = nw_name

            # oi_seq doc
            maxtap_proxy = max(maxtap, 0)
            mintap_proxy = min(initlen, 0)
            start = (k - mintap_proxy)
            if k == maxtap_proxy:
                nw_seq = seq['input'][start:]
            else:
                end = -(maxtap_proxy - k)
                nw_seq = seq['input'][start:end]

            if go_backwards:
                nw_seq = nw_seq[::-1]

            oi_seqs.append(nw_seq)
            ii_seqs_slices.append(ii_slice)
            seq_direct_cal_slices.append(direct_cal_slice)
            n_seqs_oi += 1
