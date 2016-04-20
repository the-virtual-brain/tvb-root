
"""
Tests for the experiment Numba simulator component implementations.

.. moduleauthor:: Marmaduke Woodman <mmwoodman@gmail.com>

"""

import unittest
import numpy
import numpy.testing

try:
    import numba
    from numba import cuda
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    CUDA_SIM = True

from tvb.tests.library.base_testcase import BaseTestCase

if HAVE_NUMBA:
    from tvb.simulator._numba.coupling import (cu_simple_cfun, next_pow_of_2, cu_linear_cfe_post, cu_linear_cfe_pre,
                                               cu_delay_cfun, _cu_mod_pow_2, cu_tanh_cfe_pre, cu_sigm_cfe_post,
                                               cu_kura_cfe_pre)
    from tvb.simulator._numba.util import CUDA_SIM

from tvb.simulator import coupling as py_coupling

def skip_if_no_numba(f):
    return unittest.skipIf(not HAVE_NUMBA, "Numba unavailable")(f)

class CudaBaseCase(BaseTestCase):

    # Implementations and tests written for a 1D block & 1D grid layout.
    # Results in 4 threads under simulator.
    block_dim, grid_dim = ((2, ), (2, )) if CUDA_SIM else ((16, ), (2, ))
    n_thread = block_dim[0] * grid_dim[0]

    def jit_and_run(self, *args):
        "Convenience decorator for compiling and running a kernel."
        def _(kernel):
            kernel = cuda.jit(kernel)
            kernel[self.block_dim, self.grid_dim](*args)
        return _


class TestSimpleCfun(CudaBaseCase):

    def _run_n_node(self, n_node):
        weights = numpy.random.randn(n_node, n_node).astype('f')
        state = numpy.random.randn(n_node, 2, self.n_thread).astype('f')
        out = numpy.zeros((n_node, self.n_thread)).astype('f')
        offset = 0.5
        cf = cu_simple_cfun(offset, 1)
        @self.jit_and_run(out, weights, state)
        def kernel(out, weights, state):
            t = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
            for i in range(weights.shape[0]):
                out[i, t] = cf(weights, state, i, t)
        expected = weights.dot(state[:, 1] + offset)
        ok = numpy.allclose(expected, out, 1e-4, 1e-5)
        se = numpy.sqrt((expected - out)**2)
        numpy.testing.assert_allclose(out, expected, 1e-4, 1e-5)

    @skip_if_no_numba
    def test_cf(self):
        for n_node in [3, 5, 10, 50]:
            self._run_n_node(n_node)


class TestUtils(CudaBaseCase):

    @skip_if_no_numba
    def test_pow_2(self):
        for n, e in ((5, 8), (32, 64), (63, 64)):
            self.assertEqual(e, next_pow_of_2(n))

    @skip_if_no_numba
    def test_mod_pow_2(self):
        n = 32
        i = numpy.r_[:self.n_thread].astype(numpy.int32)
        out = (i * 0).astype(numpy.int32)
        @self.jit_and_run(out, i, n)
        def kernel(out, i, n):
            t = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
            out[t] = _cu_mod_pow_2(i[t], n)
        numpy.testing.assert_equal(out, i % n)


class TestCfunExpr(CudaBaseCase):

    @skip_if_no_numba
    def test_linear_post(self):
        slope, intercept = 0.2, 0.5
        out = numpy.zeros((self.n_thread, ), 'f')
        state = numpy.random.rand(self.n_thread).astype('f')
        post = cu_linear_cfe_post(slope, intercept)

        @self.jit_and_run(out, state)
        def kernel(out, state):
            t = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
            out[t] = post(state[t])

        numpy.testing.assert_allclose(out, state*slope + intercept, 1e-4, 1e-5)

    @skip_if_no_numba
    def test_linear_pre(self):
        ai, aj, intercept = -0.2, 0.3, 0.25
        out = numpy.zeros((self.n_thread,), 'f')
        xj, xi = numpy.random.rand(2, self.n_thread).astype('f')
        pre = cu_linear_cfe_pre(ai, aj, intercept)

        @self.jit_and_run(out, xi, xj)
        def kernel(out, xi, xj):
            t = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
            out[t] = pre(xi[t], xj[t])

        numpy.testing.assert_allclose(out, ai * xi + aj * xj + intercept, 1e-4, 1e-5)

    @skip_if_no_numba
    def test_tanh(self):
        py_cf = py_coupling.HyperbolicTangent()
        cu_cf = cu_tanh_cfe_pre(py_cf.a[0], py_cf.b[0], py_cf.midpoint[0], py_cf.sigma[0])
        out = numpy.zeros((self.n_thread,), 'f')
        xj, xi = numpy.random.rand(2, self.n_thread).astype('f')

        @self.jit_and_run(out, xi, xj)
        def kernel(out, xi, xj):
            t = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
            out[t] = cu_cf(xi[t], xj[t])

        numpy.testing.assert_allclose(out, py_cf.pre(xi, xj))

    @skip_if_no_numba
    def test_sigm(self):
        py_cf = py_coupling.Sigmoidal()
        cu_cf = cu_sigm_cfe_post(py_cf.cmin[0], py_cf.cmax[0], py_cf.midpoint[0], py_cf.a[0], py_cf.sigma[0])
        out = numpy.zeros((self.n_thread,), 'f')
        gx = numpy.random.rand(self.n_thread).astype('f')

        @self.jit_and_run(out, gx)
        def kernel(out, gx):
            t = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
            out[t] = cu_cf(gx[t])

        numpy.testing.assert_allclose(out, py_cf.post(gx), 1e-4, 1e-5)

    @skip_if_no_numba
    def test_kura(self):
        py_cf = py_coupling.Kuramoto()
        cu_cf = cu_kura_cfe_pre()
        out = numpy.zeros((self.n_thread,), 'f')
        xj, xi = numpy.random.rand(2, self.n_thread).astype('f')

        @self.jit_and_run(out, xi, xj)
        def kernel(out, xi, xj):
            t = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
            out[t] = cu_cf(xi[t], xj[t])

        numpy.testing.assert_allclose(out, py_cf.pre(xi, xj))


class TestDcfun(CudaBaseCase):

    def _do_for_params(self, horizon):

        # setup test data
        numpy.random.seed(42)
        n_step, n_node, n_cvar, n_svar, n_thread = 100, 5, 2, 4, self.n_thread
        cvars = numpy.random.randint(0, n_svar, n_cvar).astype(numpy.int32)
        out = numpy.zeros((n_step, n_node, n_cvar, n_thread), numpy.float32)
        delays = numpy.random.randint(0, horizon - 2, (n_node, n_node)).astype(numpy.int32)
        weights = numpy.random.randn(n_node, n_node).astype(numpy.float32)
        state = numpy.random.randn(n_step, n_node, n_svar, n_thread).astype(numpy.float32)
        buf = numpy.zeros((n_node, horizon, n_cvar, n_thread), numpy.float32)
        # debugging
        delayed_step = numpy.zeros_like(delays)

        # setup cu functions
        pre = cu_linear_cfe_pre(0.0, 1.0, 0.0)
        post = cu_linear_cfe_post(1.0, 0.0)
        dcf = cu_delay_cfun(horizon, pre, post, n_cvar, self.block_dim[0])

        # run it
        @self.jit_and_run(out, delays, weights, state, cvars, buf)#,delayed_step)
        def kernel(out, delays, weights, state, cvars, buf):#, delayed_step):
            i_thread = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
            for step in range(state.shape[0]):
                for i_post in range(state.shape[1]):
                    dcf(out, delays, weights, state, i_post, i_thread, step, cvars, buf)#,delayed_step)

        # ensure buffer is updating correctly
        buf_state = numpy.roll(state[:, :, cvars][-horizon:].transpose((1, 0, 2, 3)), n_step, axis=1)
        numpy.testing.assert_allclose(buf, buf_state)

        # ensure buffer time indexing is correct
        # numpy.testing.assert_equal(delayed_step, (n_step - 1 - delays + horizon) % horizon)

        # replay
        nodes = numpy.tile(numpy.r_[:n_node], (n_node, 1))
        for step in range(horizon + 3, n_step):
            delayed_state = state[:, :, cvars][step - delays, nodes] # (n_node, n_node, n_cvar, n_thread)
            afferent = (weights.reshape((n_node, n_node, 1, 1)) * delayed_state).sum(axis=1) # (n_node, n_cvar, n_thread)
            err = numpy.abs(afferent - out[step])
            numpy.testing.assert_allclose(afferent, out[step], 1e-5, 1e-6)

    @skip_if_no_numba
    @unittest.skipIf(CUDA_SIM, "https://github.com/numba/numba/issues/1837")
    def test_dcfun_horizons(self):
        self.assertRaises(ValueError, self._do_for_params, 13)
        for horizon in (16, 32, 64):
            self._do_for_params(horizon)



def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestSimpleCfun))
    test_suite.addTest(unittest.makeSuite(TestCfunExpr))
    test_suite.addTest(unittest.makeSuite(TestDcfun))
    test_suite.addTest(unittest.makeSuite(TestUtils))
    return test_suite


if __name__ == '__main__':
    unittest.main()