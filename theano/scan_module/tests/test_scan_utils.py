from __future__ import absolute_import, print_function, division
import itertools
import unittest
import numpy as np
import theano
from theano import tensor
from theano.scan_module.scan_utils import equal_computations, map_variables,\
    ScanVarMap
from theano.tensor.type_other import NoneConst

def test_var_map():
    rd = np.random.RandomState(2018)
    n_group = 10
    _max = 10
    for group_i in range(n_group):
        n_seqs = rd.randint(_max)
        n_mitsots = rd.randint(_max)
        n_sitsots = rd.randint(_max)
        n_nitsots = rd.randint(_max)
        n_shareds = rd.randint(_max)
        n_dynamic_shareds = rd.randint(_max)
        n_sitsot_shareds = rd.randint(_max)
        n_ns_inputs = rd.randint(_max)
        n_consts = rd.randint(_max)
        n_hid_ns_inputs = rd.randint(_max)

        n_step_vars = [dict(tag=ScanVarMap.N_STEP_COMBO_TAG, name="n_steps")]
        condi_vars = [dict(tag=ScanVarMap.CONDITION_COMBO_TAG, name="condition")]
        # seqs
        seqs = []
        for seq_i in range(n_seqs):
            seq_info = dict(tag=ScanVarMap.SEQ_COMBO_TAG,
                            n_in=rd.randint(_max),
                            scfn_rank=seq_i,
                            name="seq%02d"%seq_i)
            seqs.append(seq_info)
        # scfn outputs
        n_scfni_outs = n_mitsots + n_sitsots + n_nitsots
        scfni_outs_order = rd.choice(n_scfni_outs, n_scfni_outs, replace=False)
        order_i = 0
        # mitsot
        mitsots = []
        for i in range(n_mitsots):
            rank = scfni_outs_order[order_i]
            order_i += 1
            meta = dict(tag=ScanVarMap.MITSOT_COMBO_TAG,
                            n_in=rd.randint(_max),
                            scfn_rank=rank,
                            name="out_%02d_mitsot"%rank)
            mitsots.append(meta)
        # sitsot
        sitsots = []
        for i in range(n_sitsots):
            rank = scfni_outs_order[order_i]
            order_i += 1
            meta = dict(tag=ScanVarMap.SITSOT_COMBO_TAG,
                            scfn_rank=rank,
                            name="out_%02d_sitsot"%rank)
            sitsots.append(meta)
        # sitsot
        nitsots = []
        for i in range(n_nitsots):
            rank = scfni_outs_order[order_i]
            order_i += 1
            meta = dict(tag=ScanVarMap.NITSOT_COMBO_TAG,
                            scfn_rank=rank,
                            name="out_%02d_nitsot"%rank)
            nitsots.append(meta)
        assert order_i == n_scfni_outs
        # nonseqs
        n_scfni_nss = n_shareds + n_dynamic_shareds + n_sitsot_shareds +\
            n_ns_inputs + n_consts
        scfni_nss_order = rd.choice(n_scfni_nss, n_scfni_nss, replace=False)
        order_i = 0
        # consts
        consts = []
        for i in range(n_consts):
            rank = scfni_nss_order[order_i]
            order_i += 1
            meta = dict(tag=ScanVarMap.CONST_COMBO_TAG,
                            scfn_rank=rank,
                            name="const_%02d"%rank)
            consts.append(meta)
        # shareds
        shareds = []
        for i in range(n_shareds):
            rank = scfni_nss_order[order_i]
            order_i += 1
            meta = dict(tag=ScanVarMap.SHARED_COMBO_TAG,
                            scfn_rank=rank,
                            name="nonseq_%02d_shared"%rank)
            shareds.append(meta)
        # dynamic shareds
        dyshareds = []
        for i in range(n_dynamic_shareds):
            rank = scfni_nss_order[order_i]
            order_i += 1
            meta = dict(tag=ScanVarMap.DYNAMIC_SHARED_COMBO_TAG,
                            scfn_rank=rank,
                            name="nonseq_%02d_dynamic_shared"%rank)
            dyshareds.append(meta)
        # sitsot shareds
        sitsot_shareds = []
        for i in range(n_sitsot_shareds):
            rank = scfni_nss_order[order_i]
            order_i += 1
            meta = dict(tag=ScanVarMap.SITSOT_SHARED_COMBO_TAG,
                            scfn_rank=rank,
                            name="nonseq_%02d_sitsot_shared"%rank)
            sitsot_shareds.append(meta)
        # ns inputs
        ns_inputs = []
        for i in range(n_ns_inputs):
            rank = scfni_nss_order[order_i]
            order_i += 1
            meta = dict(tag=ScanVarMap.NS_INPUT_COMBO_TAG,
                            scfn_rank=rank,
                            name="nonseq_%02d_ns_input"%rank)
            ns_inputs.append(meta)
        assert order_i == n_scfni_nss
        # hidden ns inputs
        hid_ns_inputs = []
        for i in range(n_hid_ns_inputs):
            # add rank to make all hid_ns_input var in same order
            # for test.
            meta = dict(tag=ScanVarMap.HIDDEN_NS_INPUT_COMBO_TAG,
                    scfn_rank=i,
                    name="hid_ns_input%02d"%i)
            hid_ns_inputs.append(meta)

        all_vars = n_step_vars + condi_vars + seqs + mitsots + sitsots + \
                nitsots + shareds + dyshareds + sitsot_shareds + ns_inputs + \
                hid_ns_inputs + consts

        iis = seqs + mitsots + sitsots + sitsot_shareds + dyshareds + shareds\
            + ns_inputs + hid_ns_inputs
        ios = condi_vars + mitsots + sitsots + nitsots + dyshareds + \
            sitsot_shareds
        ois = n_step_vars + seqs + mitsots + sitsots + shareds + dyshareds +\
            sitsot_shareds + ns_inputs + hid_ns_inputs + nitsots
        print("ois %d." % len(ois))
        oos = mitsots + sitsots + nitsots + sitsot_shareds
        scfni_seqs = seqs
        scfni_nss = shareds + dyshareds + sitsot_shareds + consts + ns_inputs
        scfnos = mitsots + sitsots + nitsots
        stfnis = seqs + mitsots + sitsots + nitsots + scfni_nss
        n_iis = sum([meta.get("n_in", 1) for meta in iis])
        n_ios = sum([meta.get("n_out", 1) for meta in ios])
        n_ois = sum([meta.get("n_in", 1) for meta in ois])
        n_oos = sum([meta.get("n_out", 1) for meta in oos])
        n_scfni_seqs = len(scfni_seqs)
        n_scfnos = len(scfnos)
        n_stfnis = sum([meta.get("n_in", 1) for meta in stfnis])

        print("---Show all vars----")
        for var in all_vars:
            print(var["name"])
        print("")

        def get_var_names_by_set(svm, set_type):
            var_metas = svm.get_abst_metas_by_set(set_type)
            names = [meta["name"] for meta in var_metas]
            return names

        def compare(lsvm, rsvm):
            for set_type in ScanVarMap.SET_TYPES:
                lnames = get_var_names_by_set(lsvm, set_type)
                rnames = get_var_names_by_set(rsvm, set_type)
                assert lnames == rnames

        _svm = None
        for i in range(_max):
            rd_all_vars = [v for v in all_vars]
            rd.shuffle(rd_all_vars)

            svm = ScanVarMap()
            for var in rd_all_vars:
                svm.add_var(var)

            if _svm == None:
                _svm = svm
                assert _svm.var_set_size(ScanVarMap.II_TYPE) == n_iis
                assert _svm.var_set_size(ScanVarMap.IO_TYPE) == n_ios
                assert _svm.var_set_size(ScanVarMap.OI_TYPE) == n_ois
                assert _svm.var_set_size(ScanVarMap.OO_TYPE) == n_oos
                assert _svm.var_set_size(ScanVarMap.SCFNI_SEQ_TYPE) == n_scfni_seqs
                assert _svm.var_set_size(ScanVarMap.SCFNI_OUTPUT_TYPE) == n_scfni_outs
                assert _svm.var_set_size(ScanVarMap.SCFNI_NONSEQ_TYPE) == n_scfni_nss
                assert _svm.var_set_size(ScanVarMap.SCFNO_TYPE) == n_scfnos
                assert _svm.var_set_size(ScanVarMap.STFNI_TYPE) == n_stfnis
            else:
                compare(svm, _svm)
            print("case %d %d ok." % (group_i,i))

def test_equal_compuations():
    # This was a bug report by a Theano user.
    c = NoneConst
    assert equal_computations([c], [c])
    m = theano.tensor.matrix()
    max_argmax1 = theano.tensor.max_and_argmax(m)
    max_argmax2 = theano.tensor.max_and_argmax(m)
    assert equal_computations(max_argmax1, max_argmax2)


#################
# map_variables #
#################

class TestMapVariables(unittest.TestCase):
    @staticmethod
    def replacer(graph):
        return getattr(graph.tag, "replacement", graph)

    def test_leaf(self):
        a = tensor.scalar("a")
        b = tensor.scalar("b")
        c = tensor.scalar("c")

        b.tag.replacement = c

        u = a + b
        v, = map_variables(self.replacer, [u])

        assert u.owner.inputs == [a, b]
        assert v.owner.inputs == [a, c]

    def test_leaf_inside_scan(self):
        x = tensor.vector('x')
        y = tensor.scalar('y')
        z = tensor.scalar('z')

        y.tag.replacement = z

        s, _ = theano.scan(lambda x: x * y, sequences=x)
        s2, = map_variables(self.replacer, [s])

        f = theano.function([x, y, z], [s, s2])
        rval = f(x=np.array([1, 2, 3], dtype=np.float32), y=1, z=2)
        assert np.array_equal(rval, [[1, 2, 3], [2, 4, 6]])

    def test_scan(self):
        x = tensor.vector('x')

        # we will insert a subgraph involving these variables into the inner
        # graph of scan. since they were not previously in the inner graph,
        # they are like non_sequences to scan(). scan() infers these and
        # imports them into the inner graph properly, and map_variables()
        # should do this as well.
        outer = tensor.scalar("outer")
        shared = theano.shared(
            np.array(1., dtype=theano.config.floatX),
            name="shared")
        constant = tensor.constant(1, name="constant")

        # z will equal 1 so multiplying by it doesn't change any values
        z = outer * (shared + constant)

        def step(x, a):
            r = a + x
            r.tag.replacement = z * (a - x)
            return r

        s, _ = theano.scan(step, sequences=x,
                           outputs_info=[np.array(0.)])
        # ensure z is owned by the outer graph so map_variables() will need to
        # jump through additional hoops to placate FunctionGraph.
        t = z * s
        s2, = map_variables(self.replacer, [t])
        t2 = z * s2

        f = theano.function([x, outer], [t, t2])
        rval = f(x=np.array([1, 2, 3], dtype=np.float32), outer=0.5)
        assert np.array_equal(rval, [[1, 3, 6], [-1, -3, -6]])

    def test_scan_with_shared_update(self):
        x = tensor.vector('x')

        # counts how many times its value is used
        counter = theano.shared(0, name="shared")
        counter.update = counter + 1

        def step(x, a):
            r = a + x
            # introducing a shared variable with an update into the
            # inner graph is unsupported and the code must crash rather
            # than silently produce the wrong result.
            r.tag.replacement = counter * (a - x)
            return r

        s, _ = theano.scan(step, sequences=x,
                           outputs_info=[np.array(0.)])
        self.assertRaises(NotImplementedError,
                          map_variables, self.replacer, [s])

    def test_scan_with_shared_update2(self):
        x = tensor.vector('x')

        # counts how many times its value is used
        counter = theano.shared(0, name="shared")
        counter.update = counter + 1

        def step(x, a):
            r = a + x
            # introducing a shared variable with an update into the
            # inner graph is unsupported and the code must crash rather
            # than silently produce the wrong result.
            r.tag.replacement = counter * (a - x)
            # the shared variable was already present, but the
            # replacement changes the number of times it is used,
            # which would have to change the updates, which is
            # unsupported.
            return r + counter

        s, _ = theano.scan(step, sequences=x,
                           outputs_info=[np.array(0.)])
        self.assertRaises(NotImplementedError,
                          map_variables, self.replacer, [s])

    def test_opfromgraph(self):
        # as with the scan tests above, insert foreign inputs into the
        # inner graph.
        outer = tensor.scalar("outer")
        shared = theano.shared(
            np.array(1., dtype=theano.config.floatX),
            name="shared")
        constant = tensor.constant(1., name="constant")
        z = outer * (shared + constant)

        # construct the inner graph
        a = tensor.scalar()
        b = tensor.scalar()
        r = a + b
        r.tag.replacement = z * (a - b)

        # construct the outer graph
        c = tensor.scalar()
        d = tensor.scalar()
        u = theano.OpFromGraph([a, b], [r])(c, d)
        t = z * u
        v, = map_variables(self.replacer, [t])
        t2 = z * v

        f = theano.function([c, d, outer], [t, t2])
        for m, n in itertools.combinations(range(10), 2):
            assert f(m, n, outer=0.5) == [m + n, m - n]

        # test that the unsupported case of replacement with a shared
        # variable with updates crashes
        shared.update = shared + 1
        self.assertRaises(NotImplementedError,
                          map_variables, self.replacer, [t])


