"""
Tests for the NumPy backend.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import numpy as np

from tvb.simulator.backend.np import NpBackend
from tvb.simulator.coupling import Sigmoidal, Linear
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.integrators import (EulerDeterministic, EulerStochastic,
    HeunDeterministic, HeunStochastic)

from .backendtestbase import (BaseTestSimODE, BaseTestCoupling, BaseTestDfun,
    BaseTestIntegrate)


class TestNpSimODE(BaseTestSimODE):

    def test_np_mpr(self):
        sim, state, t, y = self._create_sim(inhom_mmpr=True)
        template = '<%include file="np-sim-ode.mako"/>'
        kernel = NpBackend().build_py_func(template, dict(sim=sim), print_source=True)
        dX = state.copy()
        weights = sim.connectivity.weights.copy()
        yh = np.empty((len(t),)+state.shape)
        parmat = sim.model.spatial_parameter_matrix
        self.assertEqual(parmat.shape[0], 1)
        self.assertEqual(parmat.shape[1], weights.shape[1])
        kernel(state, weights, yh, parmat)
        self._check_match(y, yh)


class TestNpCoupling(BaseTestCoupling):

    def _test_py_cfun(self, mode, cfun):
        "Test a Python cfun template."
        class sim:  # dummy
            model = MontbrioPazoRoxin()
            coupling = cfun
        template = f'''import numpy as np
<%include file="{mode}-coupling.mako"/>
'''
        kernel = NpBackend().build_py_func(template, dict(sim=sim), name='coupling',
            print_source=True)
        state = np.random.rand(2, 128).astype('f')
        weights = np.random.randn(state.shape[1], state.shape[1]).astype('f')
        cX = np.zeros_like(state)
        kernel(cX, weights.T, state)
        expected = self._eval_cfun_no_delay(sim.coupling, weights, state)
        np.testing.assert_allclose(cX, expected, 1e-5, 1e-6)

    def test_nb_linear(self): self._test_py_cfun('nb', Linear())
    def test_nb_sigmoidal(self): self._test_py_cfun('nb', Sigmoidal())

    def test_np_linear(self): self._test_py_cfun('np', Linear())
    def test_np_sigmoidal(self): self._test_py_cfun('np', Sigmoidal())


class TestNpDfun(BaseTestDfun):

    def _test_py_model(self, model_):
        "Test a Python cfun template."
        class sim:  # dummy sim
            model = model_
        template = '''import numpy as np
<%include file="np-dfuns.mako"/>
'''
        kernel = NpBackend().build_py_func(template, dict(sim=sim), name='dfuns',
                    print_source=True)
        state, cX = np.random.rand(2, 2, 128)
        dX = np.zeros_like(state)
        parmat = sim.model.spatial_parameter_matrix
        kernel(dX, state, cX, parmat)
        np.testing.assert_allclose(dX, sim.model.dfun(state, cX))

    def test_py_mpr_symmetric(self):
        "Test symmetric MPR model"
        self._test_py_model(self._prep_model())

    def test_py_mpr_spatial1(self):
        "Test MPR w/ 1 spatial parameter."
        self._test_py_model(self._prep_model(1))

    def test_py_mpr_spatial2(self):
        "Test MPR w/ 2 spatial parameters."
        self._test_py_model(self._prep_model(2))


class TestNpIntegrate(BaseTestIntegrate):

    def _test_cfun(self, weights, state): weights.dot(state.T).T
    def _test_dfun(self, state, cX): -state*cX**2/state.shape[1]
    def _test_integrator(self, integrator_):
        class sim:
            integrator = integrator_
        sim.integrator.configure()
        template = '''
import numpy as np
def coupling(cX, weights, state): cX[:] = weights.dot(state.T).T
def dfuns(dX, state, cX, parmat): dX[:] = -state*cX**2/state.shape[1]
<%include file="np-integrate.mako" />
'''
        integrate = NpBackend().build_py_func(template, dict(sim=sim),
            name='integrate')
        state = np.random.randn(2, 64)
        weights = np.random.randn(64, 64)
        parmat = np.zeros(0)
        dX = np.zeros((integrator_.n_dx, 2, 64))
        cX = np.zeros((2, 64))
        integrate(state, weights, parmat, dX, cX)

    def test_euler_deterministic(self): self._test_integrator(EulerDeterministic())
    def test_euler_stochast(self): self._test_integrator(EulerDeterministic())
    def test_heun_deterministic(self): self._test_integrator(HeunDeterministic())
    def test_heun_stochastic(self): self._test_integrator(HeunDeterministic())
