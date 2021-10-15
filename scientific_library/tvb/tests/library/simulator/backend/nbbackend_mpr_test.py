import numpy as np
import scipy.sparse as ss

from .backendtestbase import BaseTestSim
from tvb.simulator.backend.nb_mpr import NbMPRBackend
from tvb.simulator.lab import *


class TestNbSim(BaseTestSim):

    def _get_run_sim_chunked(self, print_source=False):
        template = '<%include file="nb-montbrio.py.mako"/>'
        content = dict(foo='bar')

        return NbMPRBackend().build_py_func(template, content, name='run_sim_tavg_chunked', print_source=print_source)

    def _random_network(self, N=300, density=0.1, low=5, high=250, speed=np.inf):
        weights = ss.random(N, N, density=density, format='lil')
        weights.setdiag(0)
        weights = weights.tocsr()
        lengths = weights.copy()
        lengths.data[:] = np.random.uniform(low=low, high=high, size=lengths.data.shape[0])

        N = weights.shape[0]
        conn = connectivity.Connectivity(
            weights=weights.A,
            tract_lengths=lengths.A,
            region_labels=np.array([f'roi_{i}' for i in range(N)]),
            centres=np.zeros(N),
            speed=np.r_[speed]
        )
        return conn

    def test_local_deterministic(self):
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
            integrator=integrators.HeunStochastic(  # matching integrator
                dt=dt,
                noise=noise.Additive(
                    nsig=np.array([0.0, 0.0]),
                    noise_seed=42
                )
            )
        ).configure()

        (pdq_t, pdq_d), = NbMPRBackend().run_sim(sim, nstep=1)
        (raw_t, raw_d), = sim.run(simulation_length=1)

        np.testing.assert_allclose(raw_d[0, :], pdq_d[0, :], rtol=1e-5)

    def test_local_stochastic(self):
        # run_sim = self._get_run_sim(print_source=True)
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

        (pdq_t, pdq_d), = NbMPRBackend().run_sim(sim, nstep=200)

        (raw_t, raw_d), = sim.run(simulation_length=1)
        (raw_t_det, raw_d_det), = sim_det.run(simulation_length=1)
        r_tvb, V_tvb = raw_d[0, :, :, 0]
        r_tvb_det, V_tvb_det = raw_d_det[0, :, :, 0]
        r_pdq, V_pdq = pdq_d[0, :, :, 0]

        np.testing.assert_allclose(
            np.mean(r_tvb_det - r_tvb),
            np.mean(r_tvb_det - r_pdq),
            atol=1e-2  # zero mean doesn't play well with rtol
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
        # run_sim = self._get_run_sim(print_source=True)
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

        (pdq_t, pdq_d), = NbMPRBackend().run_sim(sim, nstep=1, compatibility_mode=True)
        r_pdq, V_pdq = pdq_d[0, :, :, 0]

        (raw_t, raw_d), = sim.run(simulation_length=1)
        r_tvb, V_tvb = raw_d[0, :, :, 0]

        np.testing.assert_allclose(r_tvb, r_pdq, rtol=1e-4)
        np.testing.assert_allclose(V_tvb, V_pdq, rtol=1e-4)

    def test_network_deterministic_delay(self):
        dt = 0.01
        G = 0.8
        speed = 2.

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

        (pdq_t, pdq_d), = NbMPRBackend().run_sim(sim, nstep=1, compatibility_mode=True)
        r_pdq, V_pdq = pdq_d[0, :, :, 0]

        (raw_t, raw_d), = sim.run(simulation_length=1)
        r_tvb, V_tvb = raw_d[0, :, :, 0]

        np.testing.assert_allclose(r_tvb, r_pdq, rtol=1e-4)
        np.testing.assert_allclose(V_tvb, V_pdq, rtol=1e-4)

    def test_tavg_chunking(self):
        dt = 0.01
        G = 0.1
        speed = 2.

        sim = simulator.Simulator(
            model=models.MontbrioPazoRoxin(),
            coupling=coupling.Scaling(a=np.array([G])),
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
        r_pdq = pdq_d[:, 0, :, 0]
        V_pdq = pdq_d[:, 1, :, 0]
        (pdq_chu_t, pdq_chu_d), = NbMPRBackend().run_sim(sim, nstep=2000, chunksize=500, compatibility_mode=True)
        r_pdq_chu = pdq_d[:, 0, :, 0]
        V_pdq_chu = pdq_d[:, 1, :, 0]

        (raw_t, raw_d), = sim.run(simulation_length=20)
        r_tvb = raw_d[:, 0, :, 0]
        V_tvb = raw_d[:, 1, :, 0]

        np.testing.assert_allclose(pdq_t, raw_t)
        np.testing.assert_allclose(pdq_chu_t, raw_t)

        np.testing.assert_allclose(r_pdq, r_tvb, rtol=1e-3)
        np.testing.assert_allclose(V_pdq, V_tvb, rtol=1e-3)
        # think a bit about the tolerances...  TVB stores in floats, so that 
        # can accumulate. Might be a good idea to test agains history with 
        # double typed buffer.
        np.testing.assert_allclose(r_pdq_chu, r_tvb, rtol=1e-3)
        np.testing.assert_allclose(V_pdq_chu, V_tvb, rtol=1e-3)