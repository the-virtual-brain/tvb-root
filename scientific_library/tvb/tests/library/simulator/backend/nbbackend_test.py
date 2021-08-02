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
Tests for the Numba backend.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import unittest
import numpy as np

from tvb.simulator.coupling import Sigmoidal, Linear
from tvb.simulator.noise import Additive, Multiplicative
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.integrators import (EulerDeterministic, EulerStochastic,
    HeunDeterministic, HeunStochastic, IntegratorStochastic, 
    RungeKutta4thOrderDeterministic, Identity, IdentityStochastic,
    VODEStochastic)
from tvb.simulator.backend.nb import NbBackend

from .backendtestbase import (BaseTestCoupling, BaseTestDfun,
    BaseTestIntegrate, BaseTestSim)


class TestNbCoupling(BaseTestCoupling):

    def _test_cfun(self, cfun):
        "Test a Python cfun template."
        sim = self._prep_sim(cfun)
        # prep & invoke kernel
        template = f'''import numpy as np
<%include file="nb-coupling.py.mako"/>
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
<%include file="nb-coupling2.py.mako"/>
def kernel(t, cx, weights, state, di):
    for i in range(weights.shape[0]):
% for cvar, cterm in zip(sim.model.cvar, sim.model.coupling_terms):
        cx[${loop.index},i] = cx_${cterm}(t, i, weights, state[${cvar}], di)
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
<%
    svars = ', '.join(sim.model.state_variables)
    cvars = ', '.join(sim.model.coupling_terms)
%>
<%include file="nb-dfuns.py.mako"/>
def kernel(dx, state, cx, parmat):
    for i in range(state.shape[1]):
        ${svars} = state[:, i]
% for cvar in sim.model.coupling_terms:
        ${cvar} = cx[${loop.index}, i]
% endfor
% for svar in sim.model.state_variables:
        dx[${loop.index},i] = dx_${svar}(${svars}, ${cvars},
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


class TestNbIntegrate(BaseTestIntegrate):

    def _test_dfun(self, state, cX, lc):
        return -state*cX**2/state.shape[1]

    def _eval_cg(self, integrator_, state, weights_):
        class sim:
            integrator = integrator_
            connectivity = Connectivity.from_file()
            model = MontbrioPazoRoxin()
        sim.connectivity.speed = np.r_[np.inf]
        sim.connectivity.configure()
        sim.integrator.configure()
        sim.connectivity.set_idelays(sim.integrator.dt)
        template = '''
import numpy as np
import numba as nb
cx_Coupling_Term_r = nb.njit(lambda t, i, w, r: np.dot(w[i],r[:,t]))
cx_Coupling_Term_V = nb.njit(lambda t, i, w, V: np.dot(w[i],V[:,t]))
dx_r = nb.njit(lambda r,V,Coupling_Term_r,Coupling_Term_V,parmat: -r*Coupling_Term_r**2/76)
dx_V = nb.njit(lambda r,V,Coupling_Term_r,Coupling_Term_V,parmat: -V*Coupling_Term_V**2/76)
<%include file="nb-integrate.py.mako" />
'''
        integrate = NbBackend().build_py_func(template, dict(sim=sim, np=np),
            name='integrate', print_source=True)
        parmat = np.zeros(0)
        np.random.seed(42)
        args = state, weights_, parmat
        if isinstance(sim.integrator, IntegratorStochastic):
            print(sim.integrator.noise.nsig.shape)
            args = args + (sim.integrator.noise.nsig, )
        integrate(0, *args)
        return state

    def _test_integrator(self, Integrator):
        if issubclass(Integrator, IntegratorStochastic):
            integrator = Integrator(dt=0.1, noise=Additive(nsig=np.r_[0.01]))
            integrator.noise.dt = integrator.dt
        else:
            integrator = Integrator(dt=0.1)
        nn = 76
        state = np.random.randn(2, nn, 2)
        weights = np.random.randn(nn, nn)
        cx = weights.dot(state[...,0].T).T
        assert cx.shape == (2, nn)
        expected = integrator.scheme(state[...,0], self._test_dfun, cx, 0, 0)
        actual = state.copy()
        np.random.seed(42)
        actual[...,1] = np.random.randn(*actual[...,1].shape)
        self._eval_cg(integrator, actual, weights)
        np.testing.assert_allclose(actual[...,1], expected)

    def test_euler(self): self._test_integrator(EulerDeterministic)
    def test_eulers(self): self._test_integrator(EulerStochastic)
    def test_heun(self): self._test_integrator(HeunDeterministic)
    def test_heuns(self): self._test_integrator(HeunStochastic)
    def test_rk4(self): self._test_integrator(RungeKutta4thOrderDeterministic)
    def test_id(self): self._test_integrator(Identity)
    def test_ids(self): self._test_integrator(IdentityStochastic)


class TestNbSim(BaseTestSim):

    def _test_mpr(self, integrator, delays=False):
        sim = self._create_sim(
            integrator,
            inhom_mmpr=True,
            delays=delays,
            run_sim=False
        )
        template = '<%include file="nb-sim.py.mako"/>'
        content = dict(sim=sim, np=np, debug_nojit=True)
        kernel = NbBackend().build_py_func(
                template, content, print_source=True, name='run_sim',
                )
        np.random.seed(42)
        state = kernel(sim)  # (nsvar, nnode, horizon + nstep)
        yh = np.transpose(state[:,:,sim.connectivity.horizon:], (2,0,1))
        (_, y), = sim.run()
        self._check_match(y, yh)

    def _test_mvar(self, integrator):
        pass # TODO

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
            self._test_mpr(integrator, delays=delays)

    def test_euler(self): self._test_integrator(EulerDeterministic)
    def test_eulers(self): self._test_integrator(EulerStochastic)
    def test_heun(self): self._test_integrator(HeunDeterministic)
    def test_heuns(self): self._test_integrator(HeunStochastic)
    def test_rk4(self): self._test_integrator(RungeKutta4thOrderDeterministic)

    @unittest.skip('TODO')
    def test_id(self): self._test_integrator(Identity)

    @unittest.skip('TODO')
    def test_ids(self): self._test_integrator(IdentityStochastic)

    def test_deuler(self): self._test_integrator(EulerDeterministic,
                                                 delays=True)

    def test_deulers(self): self._test_integrator(EulerStochastic,
                                                  delays=True)

    def test_dheun(self): self._test_integrator(HeunDeterministic,
                                                delays=True)

    def test_dheuns(self): self._test_integrator(HeunStochastic,
                                                 delays=True)

    def test_drk4(self): self._test_integrator(RungeKutta4thOrderDeterministic,
                                               delays=True)
