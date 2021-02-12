"""
Tests for the CUDA backend.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import unittest

from tvb.simulator.backend.cu import CuBackend, pycuda_available
from tvb.simulator.coupling import Sigmoidal, Linear
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin

from .backendtestbase import BaseTestSimODE, BaseTestCoupling, BaseTestDfun


class TestCUSimODE(BaseTestSimODE):

    @unittest.skipUnless(pycuda_available, 'requires working PyCUDA')
    def test_cu_mpr(self):
        "Test generated CUDA kernel directly from Simulator instance."
        sim, state, t, y = self._create_sim(inhom_mmpr=True)
        template = '<%include file="cu-sim-ode.mako"/>'
        kernel = CuBackend().build_cu_func(template, dict(sim=sim, pi=np.pi))
        dX = state.copy()
        weights = sim.connectivity.weights.T.copy().astype('f')
        parmat = sim.model.spatial_parameter_matrix.astype('f')
        yh = np.empty((len(t),)+state.shape, 'f')
        kernel(
            In(state), In(weights), Out(yh), In(parmat),
            grid=(1,1), block=(128,1,1))
        self._check_match(y, yh)


class TestCUCoupling(BaseTestCoupling):

    def _test_cu_cfun(self, cfun):
        "Test CUDA cfun template."
        class sim:  # dummy
            model = MontbrioPazoRoxin()
            coupling = cfun
        template = '''
<%include file="cu-coupling.mako"/>
__global__ void kernel(float *state, float *weights, float *cX) {
    coupling(threadIdx.x, ${n_node}, cX, weights, state);
}
'''
        content = dict(n_node=128, sim=sim)
        kernel = self._build_cu_func(template, content)
        state = np.random.rand(2, content['n_node']).astype('f')
        weights = np.random.randn(state.shape[1], state.shape[1]).astype('f')
        cX = np.empty_like(state)
        kernel(In(state), In(weights), Out(cX), 
            grid=(1,1), block=(content['n_node'],1,1))
        expected = self._eval_cfun_no_delay(sim.coupling, weights, state)
        np.testing.assert_allclose(cX, expected, 1e-5, 1e-6)

    @unittest.skipUnless(pycuda_available, 'requires working PyCUDA')
    def test_cu_linear(self):
        self._test_cu_cfun(Linear())

    @unittest.skipUnless(pycuda_available, 'requires working PyCUDA')
    def test_cu_sigmoidal(self):
        self._test_cu_cfun(Sigmoidal())


class TestCUDfun(BaseTestDfun):

    def _test_cu_model(self, model_):
        "Test CUDA model dfuns."
        class sim:  # dummy
            model = model_
        template = '''

#define M_PI_F 3.14159265358979f

<%include file="cu-dfuns.mako"/>
__global__ void kernel(float *dX, float *state, float *cX, float *parmat) {
    dfuns(threadIdx.x, ${n_node}, dX, state, cX, parmat);
}
'''
        content = dict(n_node=128, sim=sim)
        kernel = self._build_cu_func(template, content, print_source=True)
        dX, state, cX = np.random.rand(3, 2, content['n_node']).astype('f')
        parmat = sim.model.spatial_parameter_matrix.astype('f')
        if parmat.size == 0:
            parmat = np.zeros((1,),'f') # dummy
        kernel(Out(dX), In(state), In(cX), In(parmat), 
            grid=(1,1), block=(content['n_node'],1,1))
        expected = sim.model.dfun(state, cX)
        np.testing.assert_allclose(dX, expected, 1e-5, 1e-6)

    @unittest.skipUnless(pycuda_available, 'requires working PyCUDA')
    def test_cu_mpr_symmetric(self):
        self._test_cu_model(self._prep_model())

    @unittest.skipUnless(pycuda_available, 'requires working PyCUDA')
    def test_cu_mpr_spatial1(self):
        "Test MPR w/ 1 spatial parameter."
        self._test_cu_model(self._prep_model(1))

    @unittest.skipUnless(pycuda_available, 'requires working PyCUDA')
    def test_cu_mpr_spatial2(self):
        "Test MPR w/ 2 spatial parameters."
        self._test_cu_model(self._prep_model(2))
