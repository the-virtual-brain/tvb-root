"""
Tests for the Numba backend.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import numpy as np

from tvb.simulator.coupling import Sigmoidal, Linear
from tvb.simulator.backend.nb import NbBackend

from .backendtestbase import BaseTestCoupling, BaseTestDfun


class TestNbCoupling(BaseTestCoupling):

    def _test_cfun(self, cfun):
        "Test a Python cfun template."
        sim = self._prep_sim(cfun)
        # prep & invoke kernel
        template = f'''import numpy as np
<%include file="nb-coupling.mako"/>
'''
        kernel = NbBackend().build_py_func(template, dict(sim=sim, np=np), 
            name='coupling', print_source=True)
        fill = np.r_[:sim.history.buffer.size]
        fill = np.reshape(fill, sim.history.buffer.shape[:-1])
        sim.history.buffer[..., 0] = fill
        sim.current_state[:] = fill[0,:,:,None]
        buf = sim.history.buffer[...,0]
        # kernel has history in reverse order except 1st element 🤕
        rbuf = np.concatenate((buf[0:1], buf[1:][::-1]), axis=0)
        state = np.transpose(rbuf, (1, 0, 2)).astype('f')
        weights = sim.connectivity.weights.astype('f')
        cX = np.zeros_like(state[:,0])
        kernel(cX, weights, state, sim.connectivity.idelays)
        # do comparison
        (t, y), = sim.run()
        np.testing.assert_allclose(cX, y[0,:,:,0], 1e-5, 1e-6)

    def test_nb_linear(self): self._test_cfun(Linear())
    def test_nb_sigmoidal(self): self._test_cfun(Sigmoidal())


    def _test_cfun2(self, cfun):
        "This variant tests per node cfun kernel."
        sim = self._prep_sim(cfun)
        # prep & invoke kernel
        template = '''import numpy as np
<%include file="nb-coupling2.mako"/>
def kernel(t, cx, weights, state, di):
    for i in range(weights.shape[0]):
% for cvar, cterm in zip(sim.model.cvar, sim.model.coupling_terms):
        cx[${loop.index},i] = cx_${cterm}(t, i, weights, state, di)
% endfor
'''
        kernel = NbBackend().build_py_func(template, dict(sim=sim, np=np), 
            print_source=True)
        fill = np.r_[:sim.history.buffer.size]
        fill = np.reshape(fill, sim.history.buffer.shape[:-1])
        sim.history.buffer[..., 0] = fill
        sim.current_state[:] = fill[0,:,:,None]
        # [current_state, oldest, next oldest, ..., current - 1]
        buf = sim.history.buffer[...,0] # (horizon, svar, node)
        horizon, nsvar, nnode = buf.shape
        # switch to Jan's history layout here, time increasing with index
        buf2 = np.zeros((nsvar, nnode, horizon + 10))
        buf2[:,:,:horizon-1] = np.transpose(buf[1:], (1,2,0))
        buf2[:,:, horizon-1] = buf[0]
        weights = sim.connectivity.weights.astype('f')
        cX = np.zeros((nsvar,nnode))
        kernel(horizon-1, cX, weights, buf2, sim.connectivity.idelays)
        # do comparison
        (t, y), = sim.run()
        np.testing.assert_allclose(cX, y[0,:,:,0], 1e-5, 1e-6)

    def test_nb_linear2(self): self._test_cfun2(Linear())
    def test_nb_sigmoidal2(self): self._test_cfun2(Sigmoidal())


class TestNbDfun(BaseTestDfun):

    def _test_dfun(self, model_):
        "Test a Numba cfun template."
        class sim:  # dummy sim
            model = model_
        template = '''import numpy as np
<%include file="nb-dfuns.mako"/>
def kernel(dx, state, cx, parmat):
    for i in range(state.shape[1]):
% for svar in sim.model.state_variables:
        dx[${loop.index},i] = dx_${svar}(state[:,i], cx[:,i], 
            parmat[:,i] if parmat.size else None)
% endfor
'''
        kernel = NbBackend().build_py_func(template, dict(sim=sim, np=np),
            print_source=True)
        cX = np.random.rand(2, 128) / 10.0
        dX = np.zeros_like(cX)
        state = np.random.rand(2, 128)
        parmat = sim.model.spatial_parameter_matrix
        kernel(dX, state, cX, parmat)
        drh, dVh = dX
        dr, dV = sim.model.dfun(state, cX)
        np.testing.assert_allclose(drh, dr)
        np.testing.assert_allclose(dVh, dV)

    def test_py_mpr_symmetric(self):
        "Test symmetric MPR model"
        self._test_dfun(self._prep_model())

    def test_py_mpr_spatial1(self):
        "Test MPR w/ 1 spatial parameter."
        self._test_dfun(self._prep_model(1))

    def test_py_mpr_spatial2(self):
        "Test MPR w/ 2 spatial parameters."
        self._test_dfun(self._prep_model(2))

