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
Base classes for backend tests.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import unittest
import numpy as np

from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.models.linear import Linear as LinearModel
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.integrators import (EulerDeterministic, IntegratorStochastic,
    Identity)
from tvb.simulator.monitors import Raw
from tvb.simulator.simulator import Simulator


class BaseTestSim(unittest.TestCase):
    "Integration tests of ODE cases against TVB builtins."


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


class BaseTestIntegrate(unittest.TestCase):
    pass
