"""
Base classes for backend tests.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import unittest
import numpy as np

from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.coupling import Sigmoidal, Linear
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.integrators import EulerDeterministic
from tvb.simulator.monitors import Raw
from tvb.simulator.simulator import Simulator


class BaseTestSimODE(unittest.TestCase):
    "Integration tests of ODE cases against TVB builtins."

    def _create_sim(self, inhom_mmpr=False):
        mpr = MontbrioPazoRoxin()
        conn = Connectivity.from_file()
        if inhom_mmpr:
            dispersion = 1 + np.random.randn(conn.weights.shape[0])*0.1
            mpr = MontbrioPazoRoxin(eta=mpr.eta*dispersion)
        conn.speed = np.r_[np.inf]
        dt = 0.01
        integrator = EulerDeterministic(dt=dt)
        sim = Simulator(connectivity=conn, model=mpr, integrator=integrator, 
            monitors=[Raw()],
            simulation_length=0.1)  # 10 steps
        sim.configure()
        self.assertTrue((conn.idelays == 0).all())
        state = sim.current_state.copy()[:,:,0].astype('f')
        self.assertEqual(state.shape[0], 2)
        self.assertEqual(state.shape[1], conn.weights.shape[0])
        (t,y), = sim.run()
        return sim, state, t, y

    def _check_match(self, expected, actual):
        # check we don't have numerical errors
        self.assertTrue(np.isfinite(actual).all())
        # check tolerances
        maxtol = np.max(np.abs(actual[0] - expected[0,:,:,0]))
        for t in range(1, len(actual)):
            print(t, 'tol:', np.max(np.abs(actual[t] - expected[t,:,:,0])))
            np.testing.assert_allclose(actual[t], expected[t, :, :, 0], 2e-5*t*2, 1e-5*t*2)


class BaseTestCoupling(unittest.TestCase):
    "Unit tests for coupling function implementations."    

    def _eval_cfun_no_delay(self, cfun, weights, X):
        nsvar, nnode = X.shape
        x_i, x_j = X.reshape((nsvar, 1, nnode)), X.reshape((nsvar, nnode, 1))
        gx = (weights * cfun.pre(x_i+x_j*0, x_j+x_i*0)).sum(axis=1)
        return cfun.post(gx)


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

