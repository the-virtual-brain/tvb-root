"""
Base classes for backend tests.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import unittest
import numpy as np

from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.models.linear import Linear as LinearModel
from tvb.simulator.coupling import Sigmoidal, Linear
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.integrators import (EulerDeterministic, IntegratorStochastic,
    Identity)
from tvb.simulator.monitors import Raw
from tvb.simulator.simulator import Simulator


class BaseTestSim(unittest.TestCase):
    "Integration tests of ODE cases against TVB builtins."


    def _create_sim(self, integrator=None, inhom_mmpr=False):
        mpr = MontbrioPazoRoxin()
        conn = Connectivity.from_file()
        if inhom_mmpr:
            dispersion = 1 + np.random.randn(conn.weights.shape[0])*0.1
            mpr = MontbrioPazoRoxin(eta=mpr.eta*dispersion)
        conn.speed = np.r_[np.inf]
        if integrator is None:
            dt = 0.01
            integrator = EulerDeterministic(dt=dt)
        else:
            dt = integrator.dt
        sim = Simulator(connectivity=conn, model=mpr, integrator=integrator, 
            monitors=[Raw()],
            simulation_length=0.1)  # 10 steps
        sim.configure()
        self.assertTrue((conn.idelays == 0).all())
        state = sim.current_state.copy()[:,:,0].astype('f')
        self.assertEqual(state.shape[0], 2)
        self.assertEqual(state.shape[1], conn.weights.shape[0])
        if isinstance(sim.integrator, IntegratorStochastic):
            sim.integrator.noise.reset_random_stream()
        (t,y), = sim.run()
        return sim, state, t, y

    def _check_match(self, expected, actual):
        # check we don't have numerical errors
        self.assertTrue(np.isfinite(actual).all())
        # check tolerances
        maxtol = np.max(np.abs(actual[0] - expected[0,:,:,0]))
        print('maxtol 1st step:', maxtol)
        for t in range(1, len(actual)):
            print(t, 'tol:', np.max(np.abs(actual[t] - expected[t,:,:,0])))
            np.testing.assert_allclose(actual[t], expected[t, :, :, 0], 2e-5*t*2, 1e-5*t*2)


class BaseTestCoupling(unittest.TestCase):
    "Unit tests for coupling function implementations."    

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


class BaseTestIntegrate(unittest.TestCase):
    pass
