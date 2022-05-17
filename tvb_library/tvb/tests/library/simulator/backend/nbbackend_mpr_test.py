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
import numpy
import numpy as np
import scipy.sparse as ss

from .backendtestbase import BaseTestSim
from tvb.simulator.backend.nb_mpr import NbMPRBackend
from tvb.simulator.lab import *


class TestNbSim(BaseTestSim):

    #def get_run_sim_chunked(self, print_source=False):
    #    template = '<%include file="nb-montbrio.py.mako"/>'
    #    content = dict(foo='bar')

    #    return NbMPRBackend().build_py_func(template, content, name='run_sim_tavg_chunked', print_source=print_source)

    def _random_network(self, N=300, density=0.1, low=5, high=250, speed=np.inf):
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

    def test_local_deterministic(self):
        dt = 0.01
        G = 0.

        sim = simulator.Simulator(
            model=models.MontbrioPazoRoxin(),
            coupling=coupling.Linear(a=np.array([G])),
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

        (pdq_t, pdq_d), = NbMPRBackend().run_sim(sim, nstep=1)
        (raw_t, raw_d), = sim.run(simulation_length=1)

        np.testing.assert_allclose(raw_d[0,:], pdq_d[0,:], rtol=1e-5)

    def test_local_deterministic_spatial(self):
        dt = 0.01
        G = 0.
        N=300

        sim = simulator.Simulator(
            model=models.MontbrioPazoRoxin(
                eta=np.random.uniform(
                    models.MontbrioPazoRoxin.eta.domain.lo,
                    models.MontbrioPazoRoxin.eta.domain.hi,
                    size=N
                )
            ),
            coupling=coupling.Linear(a=np.array([G])),
            connectivity=self._random_network(N),
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

        (pdq_t, pdq_d), = NbMPRBackend().run_sim(sim, nstep=1)
        (raw_t, raw_d), = sim.run(simulation_length=1)

        np.testing.assert_allclose(raw_d[0,:], pdq_d[0,:], rtol=1e-5, atol=1e-3)


    def test_local_stochastic(self):
        #run_sim = self._get_run_sim(print_source=True)
        dt = 0.01
        G = 0.

        sim = simulator.Simulator(
            model=models.MontbrioPazoRoxin(),
            coupling=coupling.Linear(a=np.array([G])),
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
            coupling=coupling.Linear(a=np.array([G])),
            connectivity=self._random_network(),
            conduction_speed=np.inf,
            monitors=[
                monitors.Raw()
            ],
            integrator=integrators.HeunDeterministic( 
                dt=dt, 
            )
        ).configure()

        (pdq_t, pdq_d), = NbMPRBackend().run_sim(sim, nstep=200)

        (raw_t, raw_d), = sim.run(simulation_length=1)
        (raw_t_det, raw_d_det), = sim_det.run(simulation_length=1)
        r_tvb, V_tvb = raw_d[0,:,:,0]
        r_tvb_det, V_tvb_det = raw_d_det[0,:,:,0]
        r_pdq, V_pdq = pdq_d[0,:,:,0] 

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
        dt = 0.01
        G = 0.8

        sim = simulator.Simulator(
            model=models.MontbrioPazoRoxin(),
            coupling=coupling.Linear(a=np.array([G])),
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

        with numpy.errstate(all='ignore'):
            (pdq_t, pdq_d), = NbMPRBackend().run_sim(sim, nstep=1, compatibility_mode=True)
            r_pdq, V_pdq = pdq_d[0,:,:,0] 

            (raw_t, raw_d), = sim.run(simulation_length=1)
            r_tvb, V_tvb = raw_d[0, :, :, 0]

        np.testing.assert_allclose(r_tvb, r_pdq, rtol=1e-4)
        np.testing.assert_allclose(V_tvb, V_pdq, rtol=1e-4)
        
    def test_network_deterministic_delay(self):
        dt = 0.01
        G = 0.8
        speed=2.

        sim = simulator.Simulator(
            model=models.MontbrioPazoRoxin(),
            coupling=coupling.Linear(a=np.array([G])),
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

        (pdq_t, pdq_d), = NbMPRBackend().run_sim(sim, nstep=1, compatibility_mode=True)
        r_pdq, V_pdq = pdq_d[0,:,:,0] 

        (raw_t, raw_d), = sim.run(simulation_length=1)
        r_tvb, V_tvb = raw_d[0,:,:,0]

        np.testing.assert_allclose(r_tvb, r_pdq, rtol=1e-4)
        np.testing.assert_allclose(V_tvb, V_pdq, rtol=1e-4, atol=1e-3)

    def test_tavg_chunking(self):
        dt = 0.01
        G = 0.1
        speed=2.

        sim = simulator.Simulator(
            model=models.MontbrioPazoRoxin(),
            coupling=coupling.Linear(a=np.array([G])),
            connectivity=self._random_network(speed=speed),
            conduction_speed=speed,
            monitors=[
                monitors.TemporalAverage(period=1)
            ],
            integrator=integrators.HeunStochastic( 
                dt=dt, 
                noise=noise.Additive(
                    nsig=np.array([0.0, 0.0]),
                    noise_seed=42
                )
            )
        ).configure()

        (pdq_t, pdq_d), = NbMPRBackend().run_sim(sim, nstep=2000, chunksize=2000, compatibility_mode=True)
        r_pdq = pdq_d[:,0,:,0]
        V_pdq = pdq_d[:,1,:,0]
        (pdq_chu_t, pdq_chu_d), = NbMPRBackend().run_sim(sim, nstep=2000, chunksize=500, compatibility_mode=True)
        r_pdq_chu = pdq_d[:,0,:,0]
        V_pdq_chu = pdq_d[:,1,:,0]

        (raw_t, raw_d), = sim.run(simulation_length=20)
        r_tvb = raw_d[:,0,:,0]
        V_tvb = raw_d[:,1,:,0]

        np.testing.assert_allclose(pdq_t,raw_t)
        np.testing.assert_allclose(pdq_chu_t,raw_t)

        np.testing.assert_allclose(r_pdq, r_tvb, atol=1e-4, rtol=0.)
        np.testing.assert_allclose(V_pdq, V_tvb, atol=1e-4, rtol=0.)
        # think a bit about the tolerances...  TVB stores in floats, so that 
        # can accumulate. Might be a good idea to test agains history with 
        # double typed buffer.
        np.testing.assert_allclose(r_pdq_chu, r_tvb, atol=1e-4, rtol=0.)
        np.testing.assert_allclose(V_pdq_chu, V_tvb, atol=1e-4, rtol=0.)

    def test_stim(self):

        G = 0.525
        nsigma = 0.0
        conn_speed=2.

        conn = connectivity.Connectivity()
        conn.motif_all_to_all(number_of_regions=3)
        conn.speed = np.r_[conn_speed]
        conn.weights[:] = 0.
        conn.weights[0,1] = 1.
        conn.weights[1,2] = 1.
        conn.weights[2,0] = 1.
        conn.centres_spherical(number_of_regions=conn.number_of_regions)
        conn.create_region_labels(mode='alphabetic')

        conn.weights = conn.weights/np.max(conn.weights)
        np.fill_diagonal(conn.weights, 0.)
        conn.configure()

        weighting = np.zeros((conn.number_of_regions, ))
        weighting[[1]] = 5.0

        eqn_t = equations.PulseTrain(
            parameters={
                'onset': 50,
                'T': 10000.0 ,
                'tau': 10.0,
                'amp': 1.
            }
        )

        stimulus = patterns.StimuliRegion(
            temporal=eqn_t,
            connectivity=conn,
            weight=weighting)


        simulation_length = 150
        seed=42

        ic = np.zeros( (1, 2, 3, 1))
        ic[:,1,:,:] = -2.

        sim = simulator.Simulator(
            model=models.MontbrioPazoRoxin(
                eta   = np.r_[-5.0],
                J     = np.r_[15.],
                Delta = np.r_[1.],
            ),
            connectivity=conn,
            coupling=coupling.Linear(
                a=np.array([G])
            ),
            conduction_speed=conn_speed,
            integrator=integrators.HeunStochastic(
                dt=0.01,
                noise=noise.Additive(
                    nsig=np.array(
                        [nsigma,nsigma*2]
                    ),
                    noise_seed=seed)
            ),
            monitors=[
                monitors.Raw()
            ],
            initial_conditions=ic,
            stimulus=stimulus,
            simulation_length=simulation_length
        ).configure()

        (pdq_t, pdq_d), = NbMPRBackend().run_sim(sim, simulation_length=simulation_length, compatibility_mode=True)
        (raw_t, raw_d), = sim.run(simulation_length=simulation_length)

        np.testing.assert_allclose(raw_d, pdq_d, rtol=1e-4)


    def test_stim_chunked(self):
        G = 0.525
        nsigma = 0.0
        conn_speed=2.
        transient = 0

        conn = connectivity.Connectivity()
        conn.motif_all_to_all(number_of_regions=3)
        conn.speed = np.r_[conn_speed]
        conn.weights[:] = 0.
        conn.weights[0,1] = 1.
        conn.weights[1,2] = 1.
        conn.weights[2,0] = 1.
        conn.centres_spherical(number_of_regions=conn.number_of_regions)
        conn.create_region_labels(mode='alphabetic')

        conn.weights = conn.weights/np.max(conn.weights)
        np.fill_diagonal(conn.weights, 0.)
        conn.configure()

        weighting = np.zeros((conn.number_of_regions, ))
        weighting[[1]] = 5.0

        eqn_t = equations.PulseTrain(
            parameters={
                'onset': 50,
                'T': 10000.0 ,
                'tau': 10.0,
                'amp': 1.
            }
        )

        stimulus = patterns.StimuliRegion(
            temporal=eqn_t,
            connectivity=conn,
            weight=weighting)


        simulation_length = 150
        seed=42

        ic = np.zeros( (1, 2, 3, 1))
        ic[:,1,:,:] = -2.

        sim = simulator.Simulator(
            model=models.MontbrioPazoRoxin(
                eta   = np.r_[-5.0],
                J     = np.r_[15.],
                Delta = np.r_[1.],
            ),
            connectivity=conn,
            coupling=coupling.Linear(
                a=np.array([G])
            ),
            conduction_speed=conn_speed,
            integrator=integrators.HeunStochastic(
                dt=0.01,
                noise=noise.Additive(
                    nsig=np.array(
                        [nsigma,nsigma*2]
                    ),
                    noise_seed=seed)
            ),
            monitors=[
                monitors.TemporalAverage(period=1.0)
            ],
            initial_conditions=ic,
            stimulus=stimulus,
            simulation_length=simulation_length
        ).configure()


        (pdq_t, pdq_d), = NbMPRBackend().run_sim(sim, simulation_length=simulation_length, compatibility_mode=True, chunksize=5000)

        (tvb_t, tvb_d), = sim.run(simulation_length=simulation_length)

        np.testing.assert_allclose(tvb_d, pdq_d, rtol=1e-4)
