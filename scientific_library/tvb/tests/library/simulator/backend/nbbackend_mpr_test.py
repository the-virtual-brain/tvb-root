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

import numpy as np
import scipy.sparse as ss

from .backendtestbase import BaseTestSim
from tvb.simulator.backend.nb_mpr import NbMPRBackend
from tvb.simulator.lab import *


class TestNbSim(BaseTestSim):

    def _get_run_sim(self, print_source=False):
        template = '<%include file="nb-montbrio.py.mako"/>'
        content = dict(foo='bar')

        return NbMPRBackend().build_py_func(template, content, name='run_sim', print_source=print_source)

    def _random_network(self, N=500, density=0.1, low=5, high=250, speed=np.inf):
        weights = ss.random(N,N, density=density, format='lil')
        weights.setdiag(0)
        weights = weights.tocsr()
        lengths = weights.copy()
        lengths.data[:] = np.random.uniform(low=low, high=high, size=lengths.data.shape[0])

        N = weights.shape[0]
        conn = connectivity.Connectivity(
                weights=weights.A,
                tract_lengths=lengths.A,
                region_labels=np.array( [f'roi_{i}' for i in range(N)]), 
                centres=np.zeros(N),
                speed=np.r_[speed]
        )
        return conn

    def test_import(self):
        run_sim = self._get_run_sim(print_source=True)

    def test_local_deterministic(self):
        run_sim = self._get_run_sim(print_source=True)
        dt = 0.01
        G = 0.

        sim = simulator.Simulator(
            model=models.MontbrioPazoRoxin(),
            coupling=coupling.Scaling(a=np.array([G])),
            connectivity=self._random_network(),
            conduction_speed=np.inf,
            monitors=[
                monitors.Raw()
            ],
            integrator=integrators.HeunStochastic( # matching integrator
                dt=dt, 
                noise=noise.Additive(
                    nsig=np.array([0.0, 0.0]),
                    noise_seed=42
                )
            )
        ).configure()

        r_pdq, V_pdq = run_sim(sim, nstep=1)

        # check initial conditions
        np.testing.assert_allclose(sim.current_state[0,:,0], r_pdq[:,0])
        np.testing.assert_allclose(sim.current_state[1,:,0], V_pdq[:,0])

        (raw_t, raw_d), = sim.run(simulation_length=1)
        r_tvb, V_tvb = raw_d[0,:,:,0]
        r_pdq = r_pdq[:,1] # we include initial state, TVB doesn't
        V_pdq = V_pdq[:,1]

        np.testing.assert_allclose(r_tvb, r_pdq, rtol=1e-5)
        np.testing.assert_allclose(V_tvb, V_pdq, rtol=1e-5)


    def test_local_stochastic(self):
        run_sim = self._get_run_sim(print_source=True)
        dt = 0.01
        G = 0.

        sim = simulator.Simulator(
            model=models.MontbrioPazoRoxin(),
            coupling=coupling.Scaling(a=np.array([G])),
            connectivity=self._random_network(),
            conduction_speed=np.inf,
            monitors=[
                monitors.Raw()
            ],
            integrator=integrators.HeunStochastic( 
                dt=dt, 
                noise=noise.Additive(
                    nsig=np.array([0.01, 0.02]),
                    noise_seed=42
                )
            )
        ).configure()

        sim_det = simulator.Simulator(
            model=models.MontbrioPazoRoxin(),
            coupling=coupling.Scaling(a=np.array([G])),
            connectivity=self._random_network(),
            conduction_speed=np.inf,
            monitors=[
                monitors.Raw()
            ],
            integrator=integrators.HeunDeterministic( 
                dt=dt, 
            )
        ).configure()

        r_pdq, V_pdq = run_sim(sim, nstep=200)

        # check initial conditions
        np.testing.assert_allclose(sim.current_state[0,:,0], r_pdq[:,0])
        np.testing.assert_allclose(sim.current_state[1,:,0], V_pdq[:,0])

        (raw_t, raw_d), = sim.run(simulation_length=1)
        (raw_t_det, raw_d_det), = sim_det.run(simulation_length=1)
        r_tvb, V_tvb = raw_d[0,:,:,0]
        r_tvb_det, V_tvb_det = raw_d_det[0,:,:,0]
        r_pdq = r_pdq[:,1] # we include initial state, TVB doesn't
        V_pdq = V_pdq[:,1]

        np.testing.assert_allclose(
                np.mean(r_tvb_det - r_tvb),
                np.mean(r_tvb_det - r_pdq),
                atol=1e-2 # zero mean doesn't play well with rtol
        )
        np.testing.assert_allclose(
                np.mean(V_tvb_det - V_tvb),
                np.mean(V_tvb_det - V_pdq),
                atol=1e-2
        )
        np.testing.assert_allclose(
                np.std(r_tvb_det - r_tvb), 
                np.std(r_tvb_det - r_pdq),
                rtol=1e-2
        )
        np.testing.assert_allclose(
                np.std(V_tvb_det - V_tvb),
                np.std(V_tvb_det - V_pdq),
                rtol=1e-2
        )

    def test_network_deterministic_nodelay(self):
        run_sim = self._get_run_sim(print_source=True)
        dt = 0.01
        G = 0.8

        sim = simulator.Simulator(
            model=models.MontbrioPazoRoxin(),
            coupling=coupling.Scaling(a=np.array([G])),
            connectivity=self._random_network(),
            conduction_speed=np.inf,
            monitors=[
                monitors.Raw()
            ],
            integrator=integrators.HeunStochastic( 
                dt=dt, 
                noise=noise.Additive(
                    nsig=np.array([0.0, 0.0]),
                    noise_seed=42
                )
            )
        ).configure()

        r_pdq, V_pdq = run_sim(sim, nstep=1)

        np.testing.assert_allclose(sim.current_state[0,:,0], r_pdq[:,0])
        np.testing.assert_allclose(sim.current_state[1,:,0], V_pdq[:,0])
        r_pdq = r_pdq[:,1] # we include initial state, TVB doesn't
        V_pdq = V_pdq[:,1]

        (raw_t, raw_d), = sim.run(simulation_length=1)
        r_tvb, V_tvb = raw_d[0,:,:,0]

        np.testing.assert_allclose(r_tvb, r_pdq, rtol=1e-4)
        np.testing.assert_allclose(V_tvb, V_pdq, rtol=1e-4)
        
    def test_network_deterministic_delay(self):
        run_sim = self._get_run_sim(print_source=True)
        dt = 0.01
        G = 0.
        speed=2.

        sim = simulator.Simulator(
            model=models.MontbrioPazoRoxin(),
            coupling=coupling.Scaling(a=np.array([G])),
            connectivity=self._random_network(speed=speed),
            conduction_speed=speed,
            monitors=[
                monitors.Raw()
            ],
            integrator=integrators.HeunStochastic( 
                dt=dt, 
                noise=noise.Additive(
                    nsig=np.array([0.0, 0.0]),
                    noise_seed=42
                )
            )
        ).configure()

        r_pdq, V_pdq = run_sim(sim, nstep=1)

        np.testing.assert_allclose(sim.current_state[0,:,0], r_pdq[:,sim.connectivity.horizon-1])
        np.testing.assert_allclose(sim.current_state[1,:,0], V_pdq[:,sim.connectivity.horizon-1])
        r_pdq = r_pdq[:,-1] # we include initial state, TVB doesn't
        V_pdq = V_pdq[:,-1]

        (raw_t, raw_d), = sim.run(simulation_length=1)
        r_tvb, V_tvb = raw_d[0,:,:,0]

        np.testing.assert_allclose(r_tvb, r_pdq, rtol=1e-4)
        np.testing.assert_allclose(V_tvb, V_pdq, rtol=1e-4)
