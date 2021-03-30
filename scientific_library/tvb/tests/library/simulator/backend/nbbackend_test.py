"""
Tests for the Numba backend.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import numpy as np

from tvb.simulator.coupling import Sigmoidal, Linear
from tvb.simulator.backend.nb import NbBackend

from .backendtestbase import BaseTestCoupling


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
        template = f'''import numpy as np
<%include file="nb-coupling2.py.mako"/>
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
        for i in range(weights.shape[0]):
            kernel(i, cX[:,i], weights, state, sim.connectivity.idelays)
        # do comparison
        (t, y), = sim.run()
        np.testing.assert_allclose(cX, y[0,:,:,0], 1e-5, 1e-6)

    def test_nb_linear2(self): self._test_cfun2(Linear())
    def test_nb_sigmoidal2(self): self._test_cfun2(Sigmoidal())
