# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
Tests for the pytensor backend.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import numpy as np
import pytensor
from pytensor import tensor as pyt

from tvb.simulator.backend.theano import TheanoBackend
from tvb.simulator.coupling import Sigmoidal, Linear, Difference
from tvb.simulator.integrators import (
    EulerDeterministic, EulerStochastic,
    HeunDeterministic, HeunStochastic,
    IntegratorStochastic, RungeKutta4thOrderDeterministic,
    Identity, IdentityStochastic,
    VODEStochastic)
from tvb.simulator.noise import Additive, Multiplicative
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.models.oscillator import Generic2dOscillator

from .backendtestbase import BaseTestDfun, BaseTestCoupling, BaseTestIntegrate, BaseTestSim


class TestPytensorSim(BaseTestSim):

    def _test_mpr(self, integrator, delays=False):
        sim, state_numpy, t, y = self._create_sim(
            integrator,
            inhom_mmpr=True,
            delays=delays
        )
        template = '<%include file="theano-sim.py.mako"/>'
        content = dict(sim=sim, mparams={}, cparams={})
        kernel = TheanoBackend().build_py_func(template, content, print_source=True)

        state = pyt.as_tensor_variable(state_numpy, name="state")
        dX = state.copy()
        n_svar, _, n_node = state.eval().shape

        if not delays:
            self.assertEqual(sim.connectivity.horizon, 1)  # for now

        state = state.reshape((n_svar, sim.connectivity.horizon, n_node))

        weights_numpy = sim.connectivity.weights.copy()
        weights = pyt.as_tensor_variable(weights_numpy, name="weights")

        yh_numpy = np.zeros((len(t),) + state.eval()[:, 0].shape)
        yh = pyt.as_tensor_variable(yh_numpy, name="yh")

        parmat = sim.model.spatial_parameter_matrix
        self.assertEqual(parmat.shape[0], 1)
        self.assertEqual(parmat.shape[1], weights.eval().shape[1])
        np.random.seed(42)

        args = state, weights, yh, parmat
        if isinstance(integrator, IntegratorStochastic):
            args = args + (integrator.noise.nsig,)
        if delays:
            args = args + (sim.connectivity.delay_indices,)

        yh = kernel(*args)
        self._check_match(y, yh.eval())

    def _test_mvar(self, integrator):
        pass  # TODO

    def _test_osc(self, integrator, delays=False):
        sim, state_numpy, t, y = self._create_osc_sim(
            integrator,
            delays=delays
        )
        template = '<%include file="theano-sim.py.mako"/>'
        content = dict(sim=sim, np=np, theano=pytensor, tt=pyt)
        kernel = TheanoBackend().build_py_func(template, content, print_source=True)

        state = pyt.as_tensor_variable(state_numpy, name="state")
        dX = state.copy()
        n_svar, _, n_node = state.eval().shape

        if not delays:
            self.assertEqual(sim.connectivity.horizon, 1)  # for now

        state = state.reshape((n_svar, sim.connectivity.horizon, n_node))

        weights_numpy = sim.connectivity.weights.copy()
        weights = pyt.as_tensor_variable(weights_numpy, name="weights")

        yh_numpy = np.zeros((len(t),) + state.eval()[:, 0].shape)
        yh = pyt.as_tensor_variable(yh_numpy, name="yh")

        parmat = sim.model.spatial_parameter_matrix
        self.assertEqual(parmat.shape[0], 0)
        np.random.seed(42)

        args = state, weights, yh, parmat
        if isinstance(integrator, IntegratorStochastic):
            args = args + (integrator.noise.nsig,)
        if delays:
            args = args + (sim.connectivity.delay_indices,)

        yh = kernel(*args)
        self._check_match(y, yh.eval()[:, sim.model.cvar, :])

    def _test_integrator(self, Integrator, delays=False):
        dt = 0.01
        if issubclass(Integrator, IntegratorStochastic):
            integrator = Integrator(dt=dt, noise=Additive(nsig=np.r_[dt]))
            integrator.noise.dt = integrator.dt
        else:
            integrator = Integrator(dt=dt)
        if isinstance(integrator, (Identity, IdentityStochastic)):
            self._test_mvar(integrator, delays=delays)
        else:
            # self._test_mpr(integrator, delays=delays)
            self._test_osc(integrator, delays=delays)

    # TODO move to BaseTestSim to avoid duplicating all the methods

    def test_euler(self):
        self._test_integrator(EulerDeterministic)

    def test_eulers(self):
        self._test_integrator(EulerStochastic)

    def test_heun(self):
        self._test_integrator(HeunDeterministic)

    def test_heuns(self):
        self._test_integrator(HeunStochastic)

    def test_rk4(self):
        self._test_integrator(RungeKutta4thOrderDeterministic)

    def test_deuler(self):
        self._test_integrator(EulerDeterministic, delays=True)

    def test_deulers(self):
        self._test_integrator(EulerStochastic, delays=True)

    def test_dheun(self):
        self._test_integrator(HeunDeterministic, delays=True)

    def test_dheuns(self):
        self._test_integrator(HeunStochastic, delays=True)

    def test_drk4(self):
        self._test_integrator(RungeKutta4thOrderDeterministic, delays=True)


class TestPytensorCoupling(BaseTestCoupling):

    def _test_cfun(self, cfun, **cparams):
        """Test a Python cfun template."""

        sim = self._prep_sim(cfun)

        # prep & invoke kernel
        template = f'''
        import numpy as np
        import pytensor
        from pytensor import tensor as pyt
        <%include file="theano-coupling.py.mako"/>
        '''
        kernel = TheanoBackend().build_py_func(template, dict(sim=sim, cparams=cparams), name='coupling', print_source=True)

        fill = np.r_[:sim.history.buffer.size]
        fill = np.reshape(fill, sim.history.buffer.shape[:-1])
        sim.history.buffer[..., 0] = fill
        sim.current_state[:] = fill[0, :, :, None]
        buf = sim.history.buffer[..., 0]
        # kernel has history in reverse order except 1st element 🤕
        rbuf = np.concatenate((buf[0:1], buf[1:][::-1]), axis=0)

        state_numpy = np.transpose(rbuf, (1, 0, 2)).astype('f')
        state = pyt.as_tensor_variable(state_numpy, name="state")

        weights_numpy = sim.connectivity.weights.astype('f')
        weights = pyt.as_tensor_variable(weights_numpy, name="weights")

        cX_numpy = np.zeros_like(state_numpy[:, 0])
        cX = pyt.as_tensor_variable(cX_numpy, name="cX")

        cX = kernel(cX, weights, state, sim.connectivity.delay_indices)
        # do comparison
        (t, y), = sim.run()
        np.testing.assert_allclose(cX.eval(), y[0, :, :, 0], 1e-5, 1e-6)

    def test_linear(self):
        self._test_cfun(Linear())

    def test_difference(self):
        self._test_cfun(Difference())


class TestPytensorDfun(BaseTestDfun):

    def _test_dfun(self, model_, **mparams):
        """Test a Python dfun template."""

        class sim:  # dummy sim
            model = model_

        template = '''
        import numpy as np
        import pytensor
        from pytensor import tensor as pyt
        <%include file="theano-dfuns.py.mako"/>
        '''
        kernel = TheanoBackend().build_py_func(template, dict(sim=sim, mparams=mparams), name="dfuns", print_source=True)

        cX_numpy = np.random.rand(2, 128, 1)
        cX = pyt.as_tensor_variable(cX_numpy, name="cX")

        dX = pyt.zeros(shape=(2, 128, 1))

        state_numpy = np.random.rand(2, 128, 1)
        state = pyt.as_tensor_variable(state_numpy, name="state")

        parmat = sim.model.spatial_parameter_matrix
        dX = kernel(dX, state, cX, parmat)
        np.testing.assert_allclose(dX.eval(),
                                   sim.model.dfun(state_numpy, cX_numpy))

    def test_oscillator(self):
        """Test Generic2dOscillator model"""
        oscillator_model = Generic2dOscillator()
        self._test_dfun(oscillator_model)

    def test_py_mpr_symmetric(self):
        """Test symmetric MPR model"""
        self._test_dfun(self._prep_model())


class TestPytensorIntegrate(BaseTestIntegrate):

    def _test_dfun(self, state, cX, lc):
        return -state * cX ** 2 / state.shape[1]

    def _eval_cg(self, integrator_, state, weights_):
        class sim:
            integrator = integrator_
            connectivity = Connectivity.from_file()

            class model:
                state_variables = "foo", "bar"

        sim.connectivity.speed = np.r_[np.inf]
        sim.connectivity.configure()
        sim.integrator.configure()
        sim.connectivity.set_idelays(sim.integrator.dt)
        template = '''
import numpy as np
import pytensor
from pytensor import tensor as pyt
def coupling(cX, weights, state): 
    cX = pyt.set_subtensor(cX[:], weights.dot(state[:,0].T).T)
    return cX
def dfuns(dX, state, cX, parmat):
    dX = pyt.set_subtensor(dX[:], -state*cX**2/state.shape[1])
    return dX
<%include file="theano-integrate.py.mako" />
'''
        integrate = TheanoBackend().build_py_func(template, dict(sim=sim), name='integrate', print_source=True)

        parmat = pyt.zeros(0)
        dX = pyt.zeros(shape=(integrator_.n_dx,) + state[:, 0].eval().shape)
        cX = pyt.zeros_like(state[:, 0])
        np.random.seed(42)
        args = state, weights_, parmat, dX, cX

        if isinstance(sim.integrator, IntegratorStochastic):
            # dynamic noise
            # if isinstance(sim.integrator.noise, Additive):
            #     n_node = sim.connectivity.weights.shape[0]
            #     n_svar = len(sim.model.state_variables)
            #     D = tt.sqrt(2 * sim.integrator.noise.nsig)
            #     dWt = np.random.randn(n_svar, n_node)
            #     dWt = tt.as_tensor_variable(dWt)
            #     dyn_noise = tt.sqrt(sim.integrator.dt) * D * dWt
            # else:
            #     raise NotImplementedError
            # args = args + (dyn_noise, )
            args = args + (sim.integrator.noise.nsig,)
        state = integrate(*args)
        return state

    def _test_integrator(self, Integrator):
        if issubclass(Integrator, IntegratorStochastic):
            integrator = Integrator(dt=0.1, noise=Additive(nsig=np.r_[0.01]))
            integrator.noise.dt = integrator.dt
        else:
            integrator = Integrator(dt=0.1)
        nn = 76
        state_numpy = np.random.randn(2, 1, nn)
        state = pyt.as_tensor_variable(state_numpy, name="state")

        weights_numpy = np.random.randn(nn, nn)
        weights = pyt.as_tensor_variable(weights_numpy, name="weights")

        cx_numpy = weights_numpy.dot(state_numpy[:, 0].T).T
        cx = weights.dot(state[:, 0].T).T

        assert cx_numpy.shape == (2, nn)
        expected = integrator.scheme(state_numpy[:, 0], self._test_dfun, cx_numpy, 0, 0)
        # actual = state
        np.random.seed(42)
        actual = self._eval_cg(integrator, state, weights)
        np.testing.assert_allclose(actual[:, 0].eval(), expected)

    def test_euler(self):
        self._test_integrator(EulerDeterministic)

    def test_eulers(self):
        self._test_integrator(EulerStochastic)

    def test_heun(self):
        self._test_integrator(HeunDeterministic)

    def test_heuns(self):
        self._test_integrator(HeunStochastic)

    def test_rk4(self):
        self._test_integrator(RungeKutta4thOrderDeterministic)

    def test_id(self):
        self._test_integrator(Identity)

    def test_ids(self):
        self._test_integrator(IdentityStochastic)
