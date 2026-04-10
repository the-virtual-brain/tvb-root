# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
Base classes for backend tests.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import unittest
import numpy as np

from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.models.linear import Linear as LinearModel
from tvb.simulator.models.k_ion_exchange import KIonEx
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.integrators import (EulerDeterministic, IntegratorStochastic,
    Identity)
from tvb.simulator.monitors import Raw
from tvb.simulator.simulator import Simulator


class BaseTestSim(unittest.TestCase):
    "Integration tests of ODE cases against TVB builtins."

    # ------------------------------------------------------------------
    # KIonEx-specific valid initial condition (scalar per state variable).
    # Chosen so that K_o = K_o0 - beta*DKi + Kg > 0 (required for log).
    # beta = w_i/w_o = 3.0, K_o = 4.8 - 3*(-2) + 0.5 = 11.3 > 0
    # Na_i = 16 + 2 = 18 > 0,  Na_o = 138 - 3*2 = 132 > 0
    _KIONEX_IC = np.array([0.3, -55.0, 0.4, -2.0, 0.5])  # x, V, n, DKi, Kg

    def _set_kionex_valid_ic(self, sim):
        """Overwrite sim history buffer with a physically valid KIonEx state."""
        ic = self._KIONEX_IC
        # history buffer stores only coupling variables: shape (horizon, n_cvar, nnode, nmode)
        for k, cv in enumerate(sim.model.cvar):
            sim.history.buffer[:, k, :, :] = ic[cv]
        # current_state stores all state variables: shape (nsvar, nnode, nmode)
        sim.current_state[:] = ic[:, None, None]


    def _create_sim(self, integrator=None, inhom_mmpr=False, delays=False,
            run_sim=True):
        mpr = MontbrioPazoRoxin()
        conn = Connectivity.from_file()
        if inhom_mmpr:
            dispersion = 1 + np.random.randn(conn.weights.shape[0])*0.1
            mpr = MontbrioPazoRoxin(eta=mpr.eta*dispersion)
        conn.speed = np.r_[3.0 if delays else np.inf]
        if integrator is None:
            dt = 0.01
            integrator = EulerDeterministic(dt=dt)
        else:
            dt = integrator.dt
        sim = Simulator(connectivity=conn, model=mpr, integrator=integrator, 
            monitors=[Raw()],
            simulation_length=0.1)  # 10 steps
        sim.configure()
        if not delays:
            self.assertTrue((conn.idelays == 0).all())
        buf = sim.history.buffer[...,0]
        # kernel has history in reverse order except 1st element 🤕
        rbuf = np.concatenate((buf[0:1], buf[1:][::-1]), axis=0)
        state = np.transpose(rbuf, (1, 0, 2)).astype('f')
        self.assertEqual(state.shape[0], 2)
        self.assertEqual(state.shape[2], conn.weights.shape[0])
        if isinstance(sim.integrator, IntegratorStochastic):
            sim.integrator.noise.reset_random_stream()
        if run_sim:
            (t,y), = sim.run()
            return sim, state, t, y
        else:
            return sim

    def _create_sim_kionex(self, integrator=None, delays=False, run_sim=True):
        """Create a KIonEx simulator with a physically valid initial condition."""
        conn = Connectivity.from_file()
        conn.speed = np.r_[3.0 if delays else np.inf]
        if integrator is None:
            integrator = EulerDeterministic(dt=0.01)
        sim = Simulator(
            connectivity=conn,
            model=KIonEx(),
            integrator=integrator,
            monitors=[Raw()],
            simulation_length=0.1,  # 10 steps
        )
        sim.configure()
        self._set_kionex_valid_ic(sim)
        if not delays:
            self.assertTrue((conn.idelays == 0).all())
        if isinstance(sim.integrator, IntegratorStochastic):
            sim.integrator.noise.reset_random_stream()
        if run_sim:
            (t, y), = sim.run()
            return sim, t, y
        else:
            return sim

    def _check_match(self, expected, actual):
        # check we don't have numerical errors
        self.assertTrue(np.isfinite(actual).all())
        # check tolerances
        maxtol = np.max(np.abs(actual[0,0] - expected[0,:,:,0]))
        print('maxtol 1st step:', maxtol)
        for t in range(1, len(actual)):
            print(t, 'tol:', np.max(np.abs(actual[t] - expected[t,:,:,0])))
            np.testing.assert_allclose(actual[t, :],
                                       expected[t, :, :, 0], 2e-5*t*2, 1e-5*t*2)


class BaseTestCoupling(unittest.TestCase):
    "Unit tests for coupling function implementations."    

    def _eval_cfun_no_delay(self, cfun, weights, X):
        nsvar, nnode = X.shape
        x_i, x_j = X.reshape((nsvar, 1, nnode)), X.reshape((nsvar, nnode, 1))
        gx = (weights * cfun.pre(x_i+x_j*0, x_j+x_i*0)).sum(axis=1)
        return cfun.post(gx)

    def _prep_sim(self, coupling) -> Simulator:
        "Prepare simulator for testing a coupling function."
        con = Connectivity.from_file()
        con.weights[:] = 1.0
        # con = Connectivity(
        #     region_labels=np.array(['']),
        #     weights=con.weights[:5][:,:5],
        #     tract_lengths=con.tract_lengths[:5][:,:5],
        #     speed=np.array([10.0]),
        #     centres=np.array([0.0]))
        sim = Simulator(
            connectivity=con,
            model=LinearModel(gamma=np.r_[0.0]),
            coupling=coupling,
            integrator=Identity(dt=1.0),
            monitors=[Raw()],
            simulation_length=0.5
            )
        sim.configure()
        return sim


class BaseTestDfun(unittest.TestCase):
    "Unit tests for dfun evaluation implementations."

    def _prep_model(self, n_spatial=0):
        model = MontbrioPazoRoxin()
        if n_spatial > 0:
            model.eta = model.eta * (1 - np.r_[:0.1:128j])
        if n_spatial > 1:
            model.J = model.J * (1 - np.r_[:0.1:128j])
        if n_spatial > 2:
            raise NotImplemented
        self.assertEqual(len(model.spatial_parameter_matrix), n_spatial)
        return model

    def _prep_kionex_model(self):
        return KIonEx()

    def _make_kionex_valid_state(self, nnode=128, seed=42):
        """Return (state, cx) with physically valid KIonEx values.

        Constraints for log-safety:
          K_i = K_i0 + DKi > 0  →  DKi > -130  (always satisfied)
          K_o = K_o0 - beta*DKi + Kg > 0  →  Kg > 3*DKi - 4.8
          Na_i = Na_i0 + (-DKi) > 0  →  DKi < 130  (always satisfied)
          Na_o = Na_o0 + beta*DKi > 0  →  DKi > -46  (satisfied for DKi in [-3,0])
        With DKi in [-3, 0] and Kg in [0, 1]: K_o in [4.8, 13.8] > 0.
        """
        rng = np.random.default_rng(seed)
        n = nnode
        state = np.zeros((5, n))
        state[0] = rng.uniform(0.05, 1.0, n)        # x  >= 0
        state[1] = rng.uniform(-75.0, -30.0, n)     # V  (mV)
        state[2] = rng.uniform(0.05, 0.90, n)       # n  in [0,1]
        state[3] = rng.uniform(-3.0, -0.1, n)       # DKi
        state[4] = rng.uniform(0.0, 1.0, n)         # Kg  (ensures K_o > 0)
        cx = rng.uniform(0.0, 0.5, (1, n))          # Coupling_Term
        return state, cx


class BaseTestIntegrate(unittest.TestCase):
    pass
