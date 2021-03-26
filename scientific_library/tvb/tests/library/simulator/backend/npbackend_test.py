"""
Tests for the NumPy backend.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import numpy as np

from tvb.simulator.backend.np import NpBackend
from tvb.simulator.coupling import Sigmoidal, Linear
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.integrators import (EulerDeterministic, EulerStochastic,
    HeunDeterministic, HeunStochastic, IntegratorStochastic, 
    RungeKutta4thOrderDeterministic, Identity, IdentityStochastic,
    VODEStochastic)
from tvb.simulator.noise import Additive, Multiplicative
from tvb.datatypes.connectivity import Connectivity

from .backendtestbase import (BaseTestSim, BaseTestCoupling, BaseTestDfun,
    BaseTestIntegrate)


class TestNpSim(BaseTestSim):

    def _test_mpr(self, integrator):
        sim, state, t, y = self._create_sim(
            integrator,
            inhom_mmpr=True)
        template = '<%include file="np-sim.mako"/>'
        content = dict(sim=sim, np=np)
        kernel = NpBackend().build_py_func(template, content, print_source=True)
        dX = state.copy()
        n_svar, n_node = state.shape
        self.assertEqual(sim.connectivity.horizon, 1)  # for now
        state = state.reshape((n_svar, sim.connectivity.horizon, n_node))
        weights = sim.connectivity.weights.copy()
        yh = np.empty((len(t),)+state.shape)
        parmat = sim.model.spatial_parameter_matrix
        self.assertEqual(parmat.shape[0], 1)
        self.assertEqual(parmat.shape[1], weights.shape[1])
        np.random.seed(42)
        args = state, weights, yh, parmat
        if isinstance(integrator, IntegratorStochastic):
            args = args + (integrator.noise.nsig,)
        kernel(*args)
        self._check_match(y, yh)

    def _test_mvar(self, integrator):
        pass # TODO

    def _test_integrator(self, Integrator):
        dt = 0.01
        if issubclass(Integrator, IntegratorStochastic):
            integrator = Integrator(dt=dt, noise=Additive(nsig=np.r_[dt]))
            integrator.noise.dt = integrator.dt
        else:
            integrator = Integrator(dt=dt)
        if isinstance(integrator, (Identity, IdentityStochastic)):
            self._test_mvar(integrator)
        else:
            self._test_mpr(integrator)

    # TODO move to BaseTestSim to avoid duplicating all the methods

    def test_euler(self): self._test_integrator(EulerDeterministic)
    def test_eulers(self): self._test_integrator(EulerStochastic)
    def test_heun(self): self._test_integrator(HeunDeterministic)
    def test_heuns(self): self._test_integrator(HeunStochastic)
    def test_rk4(self): self._test_integrator(RungeKutta4thOrderDeterministic)
    def test_id(self): self._test_integrator(Identity)
    def test_ids(self): self._test_integrator(IdentityStochastic)

    def test_scipy_int_notimpl(self):
        with self.assertRaises(NotImplementedError):
            self._test_integrator(VODEStochastic)

    def test_multnoise_notimpl(self):
        dt = 0.01
        integrator = HeunStochastic(dt=dt, noise=Multiplicative(nsig=np.r_[dt]))
        with self.assertRaises(NotImplementedError):
            self._test_mpr(integrator)

class TestNpCoupling(BaseTestCoupling):

    def _test_cfun(self, mode, cfun):
        "Test a Python cfun template."
        sim = self._prep_sim(cfun)
        # prep & invoke kernel
        template = f'''import numpy as np
<%include file="{mode}-coupling.mako"/>
'''
        kernel = NpBackend().build_py_func(template, dict(sim=sim, np=np), 
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
        kernel(cX, weights, state, sim.connectivity.delay_indices)
        # do comparison
        (t, y), = sim.run()
        np.testing.assert_allclose(cX, y[0,:,:,0], 1e-5, 1e-6)

    # def test_nb_linear(self): self._test_cfun('nb', Linear())
    # def test_nb_sigmoidal(self): self._test_cfun('nb', Sigmoidal())

    def test_np_linear(self): self._test_cfun('np', Linear())
    def test_np_sigmoidal(self): self._test_cfun('np', Sigmoidal())


class TestNpDfun(BaseTestDfun):

    def _test_dfun(self, model_):
        "Test a Python cfun template."
        class sim:  # dummy sim
            model = model_
        template = '''import numpy as np
<%include file="np-dfuns.mako"/>
'''
        kernel = NpBackend().build_py_func(template, dict(sim=sim, np=np),
            name='dfuns', print_source=True)
        cX = np.random.rand(2, 128)
        dX = np.zeros_like(cX)
        state = np.random.rand(2, 1, 128)
        parmat = sim.model.spatial_parameter_matrix
        kernel(dX, state, cX, parmat)
        np.testing.assert_allclose(dX, sim.model.dfun(state, cX))

    def test_py_mpr_symmetric(self):
        "Test symmetric MPR model"
        self._test_dfun(self._prep_model())

    def test_py_mpr_spatial1(self):
        "Test MPR w/ 1 spatial parameter."
        self._test_dfun(self._prep_model(1))

    def test_py_mpr_spatial2(self):
        "Test MPR w/ 2 spatial parameters."
        self._test_dfun(self._prep_model(2))


class TestNpIntegrate(BaseTestIntegrate):

    def _test_dfun(self, state, cX, lc):
        return -state*cX**2/state.shape[1]

    def _eval_cg(self, integrator_, state, weights_):
        class sim:
            integrator = integrator_
            connectivity = Connectivity.from_file()
            class model:
                state_variables = 'foo', 'bar'
        sim.connectivity.speed = np.r_[np.inf]
        sim.connectivity.configure()
        sim.integrator.configure()
        sim.connectivity.set_idelays(sim.integrator.dt)
        template = '''
import numpy as np
def coupling(cX, weights, state): cX[:] = weights.dot(state[:,0].T).T
def dfuns(dX, state, cX, parmat):
    d = -state*cX**2/state.shape[1]
    dX[:] = d
<%include file="np-integrate.mako" />
'''
        integrate = NpBackend().build_py_func(template, dict(sim=sim, np=np),
            name='integrate', print_source=True)
        parmat = np.zeros(0)
        dX = np.zeros((integrator_.n_dx,)+state[:,0].shape)
        cX = np.zeros_like(state[:,0])
        np.random.seed(42)
        args = state, weights_, parmat, dX, cX
        if isinstance(sim.integrator, IntegratorStochastic):
            args = args + (sim.integrator.noise.nsig, )
        integrate(*args)
        return state

    def _test_integrator(self, Integrator):
        if issubclass(Integrator, IntegratorStochastic):
            integrator = Integrator(dt=0.1, noise=Additive(nsig=np.r_[0.01]))
            integrator.noise.dt = integrator.dt
        else:
            integrator = Integrator(dt=0.1)
        state = np.random.randn(2, 1, 64)
        weights = np.random.randn(64, 64)
        cx = weights.dot(state[:,0].T).T
        assert cx.shape == (2, 64)
        expected = integrator.scheme(state[:,0], self._test_dfun, cx, 0, 0)
        actual = state.copy()
        np.random.seed(42)
        self._eval_cg(integrator, actual, weights)
        np.testing.assert_allclose(actual[:,0], expected)

    def test_euler(self): self._test_integrator(EulerDeterministic)
    def test_eulers(self): self._test_integrator(EulerStochastic)
    def test_heun(self): self._test_integrator(HeunDeterministic)
    def test_heuns(self): self._test_integrator(HeunStochastic)
    def test_rk4(self): self._test_integrator(RungeKutta4thOrderDeterministic)
    def test_id(self): self._test_integrator(Identity)
    def test_ids(self): self._test_integrator(IdentityStochastic)


# TODO monitor support

# TODO surface support

# TODO stimulus support

# TODO bounds/clamp support

