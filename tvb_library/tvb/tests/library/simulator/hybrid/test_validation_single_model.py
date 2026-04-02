"""
Validate that hybrid TVB produces identical results to classic TVB for single model case.
"""

import numpy as np
from scipy import sparse as sp
from tvb.tests.library.simulator.hybrid.test_validation_base import ValidationTestBase
from tvb.simulator import simulator as classic_simulator
from tvb.simulator import coupling, integrators, monitors
from tvb.simulator.models import Generic2dOscillator, JansenRit, ReducedWongWang
from tvb.simulator.hybrid import Subnetwork, NetworkSet, Simulator, projection_utils
from tvb.datatypes.connectivity import Connectivity


class TestValidationSingleModel(ValidationTestBase):
    """Validate hybrid TVB produces same results as classic TVB for single model."""

    def test_generic_2d_oscillator(self):
        """Test Generic2dOscillator model equivalence."""
        dt = 0.1
        simulation_length = 100.0
        coupling_strength = 0.004

        conn_classic = Connectivity.from_file()
        conn_classic.configure()
        np.random.seed(42)

        model_classic = Generic2dOscillator()
        coupling_classic = coupling.Linear(a=np.array([coupling_strength]))
        integrator_classic = integrators.HeunDeterministic(dt=dt)
        monitor_classic = monitors.TemporalAverage(period=1.0)

        sim_classic = classic_simulator.Simulator()
        sim_classic.connectivity = conn_classic
        sim_classic.model = model_classic
        sim_classic.coupling = coupling_classic
        sim_classic.integrator = integrator_classic
        sim_classic.monitors = (monitor_classic,)
        sim_classic.configure()

        ((t_classic, y_classic),) = sim_classic.run(simulation_length=simulation_length)

        scheme = integrators.HeunDeterministic(dt=dt)
        subnet = Subnetwork(
            name="all_regions",
            model=Generic2dOscillator(),
            scheme=scheme,
            nnodes=conn_classic.number_of_regions,
        ).configure()

        weights_sp = sp.csr_matrix(conn_classic.weights)
        lengths_sp = sp.csr_matrix(conn_classic.tract_lengths)

        proj = projection_utils.create_intra_projection(
            subnet=subnet,
            source_cvar="V",
            target_cvar="V",
            weights=weights_sp,
            lengths=lengths_sp,
            scale=coupling_strength,
            cv=conn_classic.speed[0],
            dt=scheme.dt,
        )

        nets = NetworkSet(subnets=[subnet], projections=[proj])
        monitor_hybrid = monitors.TemporalAverage(period=1.0)
        sim_hybrid = Simulator(
            nets=nets, simulation_length=simulation_length, monitors=[monitor_hybrid]
        )
        sim_hybrid.configure()
        ((t_hybrid, y_hybrid),) = sim_hybrid.run()

        self.assertEqual(t_classic.shape, t_hybrid.shape)
        self.assertEqual(y_classic.shape, y_hybrid.shape)

        np.testing.assert_allclose(t_classic, t_hybrid, rtol=1e-6, atol=1e-8)

        for i in range(y_classic.shape[1]):
            np.testing.assert_allclose(
                y_classic[:, i, :, :],
                y_hybrid[:, i, :, :],
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"Mismatch in state variable {i}",
            )

        classic_mean, classic_std = y_classic.mean(), y_classic.std()
        hybrid_mean, hybrid_std = y_hybrid.mean(), y_hybrid.std()
        self.assertAlmostEqual(classic_mean, hybrid_mean, places=4)
        self.assertAlmostEqual(classic_std, hybrid_std, places=4)

    def test_jansen_rit_model(self):
        """Test JansenRit model equivalence."""
        dt = 0.1
        simulation_length = 100.0
        coupling_strength = 0.003

        conn_classic = Connectivity.from_file()
        conn_classic.configure()
        np.random.seed(42)

        model_classic = JansenRit()
        coupling_classic = coupling.Linear(a=np.array([coupling_strength]))
        integrator_classic = integrators.HeunDeterministic(dt=dt)
        monitor_classic = monitors.TemporalAverage(period=1.0)

        sim_classic = classic_simulator.Simulator()
        sim_classic.connectivity = conn_classic
        sim_classic.model = model_classic
        sim_classic.coupling = coupling_classic
        sim_classic.integrator = integrator_classic
        sim_classic.monitors = (monitor_classic,)
        sim_classic.configure()

        ((t_classic, y_classic),) = sim_classic.run(simulation_length=simulation_length)

        scheme = integrators.HeunDeterministic(dt=dt)
        subnet = Subnetwork(
            name="all_regions",
            model=JansenRit(),
            scheme=scheme,
            nnodes=conn_classic.number_of_regions,
        ).configure()

        weights_sp = sp.csr_matrix(conn_classic.weights)
        lengths_sp = sp.csr_matrix(conn_classic.tract_lengths)

        proj = projection_utils.create_intra_projection(
            subnet=subnet,
            source_cvar=["y1", "y2"],
            target_cvar=["y1", "y2"],
            weights=weights_sp,
            lengths=lengths_sp,
            scale=coupling_strength,
            cv=conn_classic.speed[0],
            dt=scheme.dt,
        )

        nets = NetworkSet(subnets=[subnet], projections=[proj])
        monitor_hybrid = monitors.TemporalAverage(period=1.0)
        sim_hybrid = Simulator(
            nets=nets, simulation_length=simulation_length, monitors=[monitor_hybrid]
        )
        sim_hybrid.configure()
        ((t_hybrid, y_hybrid),) = sim_hybrid.run()

        self.assertEqual(t_classic.shape, t_hybrid.shape)
        self.assertEqual(y_classic.shape, y_hybrid.shape)

        np.testing.assert_allclose(t_classic, t_hybrid, rtol=1e-6, atol=1e-8)

        for i in range(y_classic.shape[1]):
            np.testing.assert_allclose(
                y_classic[:, i, :, :],
                y_hybrid[:, i, :, :],
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"Mismatch in state variable {i}",
            )

        self.assert_statistics_equivalent(y_classic, y_hybrid, places=4)

    def test_reduced_wong_wang(self):
        """Test ReducedWongWang model equivalence."""
        dt = 0.1
        simulation_length = 100.0
        coupling_strength = 0.004

        conn_classic = Connectivity.from_file()
        conn_classic.configure()
        np.random.seed(42)

        model_classic = ReducedWongWang()
        coupling_classic = coupling.Linear(a=np.array([coupling_strength]))
        integrator_classic = integrators.HeunDeterministic(dt=dt)
        monitor_classic = monitors.TemporalAverage(period=1.0)

        sim_classic = classic_simulator.Simulator()
        sim_classic.connectivity = conn_classic
        sim_classic.model = model_classic
        sim_classic.coupling = coupling_classic
        sim_classic.integrator = integrator_classic
        sim_classic.monitors = (monitor_classic,)
        sim_classic.configure()

        ((t_classic, y_classic),) = sim_classic.run(simulation_length=simulation_length)

        scheme = integrators.HeunDeterministic(dt=dt)
        subnet = Subnetwork(
            name="all_regions",
            model=ReducedWongWang(),
            scheme=scheme,
            nnodes=conn_classic.number_of_regions,
        ).configure()

        weights_sp = sp.csr_matrix(conn_classic.weights)
        lengths_sp = sp.csr_matrix(conn_classic.tract_lengths)

        proj = projection_utils.create_intra_projection(
            subnet=subnet,
            source_cvar="S",
            target_cvar="S",
            weights=weights_sp,
            lengths=lengths_sp,
            scale=coupling_strength,
            cv=conn_classic.speed[0],
            dt=scheme.dt,
        )

        nets = NetworkSet(subnets=[subnet], projections=[proj])
        monitor_hybrid = monitors.TemporalAverage(period=1.0)
        sim_hybrid = Simulator(
            nets=nets, simulation_length=simulation_length, monitors=[monitor_hybrid]
        )
        sim_hybrid.configure()
        ((t_hybrid, y_hybrid),) = sim_hybrid.run()

        self.assertEqual(t_classic.shape, t_hybrid.shape)
        self.assertEqual(y_classic.shape, y_hybrid.shape)

        np.testing.assert_allclose(t_classic, t_hybrid, rtol=1e-6, atol=1e-8)

        for i in range(y_classic.shape[1]):
            np.testing.assert_allclose(
                y_classic[:, i, :, :],
                y_hybrid[:, i, :, :],
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"Mismatch in state variable {i}",
            )

        self.assert_statistics_equivalent(y_classic, y_hybrid, places=4)
