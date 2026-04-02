"""
Educational toy scenario demonstrating 2-node synchronization.
"""

import numpy as np
from scipy import sparse as sp
from tvb.tests.library.simulator.hybrid.test_validation_base import ValidationTestBase
from tvb.simulator.hybrid import Subnetwork, NetworkSet, Simulator, projection_utils
from tvb.simulator.models import Kuramoto
from tvb.simulator.integrators import HeunDeterministic
from tvb.datatypes.connectivity import Connectivity


class TestToySynchronization(ValidationTestBase):
    """Educational toy scenario: 2-node synchronization with Kuramoto model."""

    def test_two_node_synchronization(self):
        """Test that 2 Kuramoto oscillators synchronize with sufficient coupling."""
        weights = np.array([[0.0, 0.5], [0.5, 0.0]])
        lengths = np.array([[0.0, 5.0], [5.0, 0.0]])
        centres = np.array([[0, 0, 0], [10, 0, 0]])

        conn = Connectivity(
            weights=weights,
            tract_lengths=lengths,
            centres=centres,
            speed=np.array([5.0]),
            region_labels=["Node0", "Node1"],
        )
        conn.configure()

        scheme = HeunDeterministic(dt=0.1)
        model = Kuramoto(omega=np.array([1.0]))

        subnet = Subnetwork(
            name="kuramoto_network", model=model, scheme=scheme, nnodes=2
        ).configure()

        weights_sp = sp.csr_matrix(conn.weights)
        lengths_sp = sp.csr_matrix(conn.tract_lengths)

        proj = projection_utils.create_intra_projection(
            subnet=subnet,
            source_cvar="theta",
            target_cvar="theta",
            weights=weights_sp,
            lengths=lengths_sp,
            scale=1.0,
            cv=conn.speed[0],
            dt=scheme.dt,
        )

        nets = NetworkSet(subnets=[subnet], projections=[proj])
        sim = Simulator(nets=nets, simulation_length=100.0)
        sim.configure()
        ((t, y),) = sim.run()

        theta0 = y[:, 0, 0, 0]
        theta1 = y[:, 0, 1, 0]

        phase_diff = np.angle(np.exp(1j * (theta1 - theta0)))

        final_phase_diff = np.abs(phase_diff[-1])
        self.assertLess(
            final_phase_diff,
            0.1,
            f"Nodes did not synchronize. Final phase diff: {final_phase_diff}",
        )

        order_param = np.abs(np.mean(np.exp(1j * y[:, 0, :, 0]), axis=1))
        final_order = order_param[-1]
        self.assertGreater(
            final_order, 0.95, f"Order parameter too low. Final: {final_order}"
        )

        print(f"Final phase difference: {final_phase_diff:.4f} rad")
        print(f"Final order parameter: {final_order:.4f}")

    def test_synchronization_time(self):
        """Test measure time to synchronization for different coupling strengths."""
        dt = 0.1
        simulation_length = 200.0

        weights = np.array([[0.0, 0.3], [0.3, 0.0]])
        lengths = np.array([[0.0, 5.0], [5.0, 0.0]])
        centres = np.array([[0, 0, 0], [10, 0, 0]])

        conn = Connectivity(
            weights=weights,
            tract_lengths=lengths,
            centres=centres,
            speed=np.array([5.0]),
            region_labels=["Node0", "Node1"],
        )
        conn.configure()

        scheme = HeunDeterministic(dt=dt)
        model = Kuramoto(omega=np.array([1.0]))

        subnet = Subnetwork(
            name="kuramoto_network", model=model, scheme=scheme, nnodes=2
        ).configure()

        weights_sp = sp.csr_matrix(conn.weights)
        lengths_sp = sp.csr_matrix(conn.tract_lengths)

        proj = projection_utils.create_intra_projection(
            subnet=subnet,
            source_cvar="theta",
            target_cvar="theta",
            weights=weights_sp,
            lengths=lengths_sp,
            scale=1.0,
            cv=conn.speed[0],
            dt=scheme.dt,
        )

        nets = NetworkSet(subnets=[subnet], projections=[proj])
        sim = Simulator(nets=nets, simulation_length=simulation_length)
        sim.configure()
        ((t, y),) = sim.run()

        theta0 = y[:, 0, 0, 0]
        theta1 = y[:, 0, 1, 0]

        phase_diff = np.abs(np.angle(np.exp(1j * (theta1 - theta0))))

        sync_threshold = 0.1
        synced_indices = np.where(phase_diff < sync_threshold)[0]

        if len(synced_indices) > 0:
            sync_time = t[synced_indices[0]]
            print(f"Synchronization time: {sync_time:.2f} ms")
            self.assertLess(
                sync_time,
                simulation_length * 0.5,
                "Nodes should synchronize within half the simulation time",
            )
        else:
            final_diff = phase_diff[-1]
            print(
                f"Warning: No full synchronization achieved. Final diff: {final_diff:.4f} rad"
            )
            self.assertLess(
                final_diff,
                0.5,
                f"Nodes did not synchronize significantly. Final diff: {final_diff}",
            )

    def test_three_node_ring_synchronization(self):
        """Test synchronization of 3 oscillators in a ring topology."""
        n_nodes = 3
        dt = 0.1
        simulation_length = 100.0
        coupling_strength = 0.5

        weights = np.zeros((n_nodes, n_nodes))
        lengths = np.zeros((n_nodes, n_nodes))
        centres = np.zeros((n_nodes, 3))

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    weights[i, j] = coupling_strength
                    lengths[i, j] = 5.0
            centres[i, 0] = i * 10.0

        conn = Connectivity(
            weights=weights,
            tract_lengths=lengths,
            centres=centres,
            speed=np.array([5.0]),
            region_labels=[f"Node{i}" for i in range(n_nodes)],
        )
        conn.configure()

        scheme = HeunDeterministic(dt=dt)
        model = Kuramoto(omega=np.array([1.0]))

        subnet = Subnetwork(
            name="kuramoto_ring", model=model, scheme=scheme, nnodes=n_nodes
        ).configure()

        weights_sp = sp.csr_matrix(conn.weights)
        lengths_sp = sp.csr_matrix(conn.tract_lengths)

        proj = projection_utils.create_intra_projection(
            subnet=subnet,
            source_cvar="theta",
            target_cvar="theta",
            weights=weights_sp,
            lengths=lengths_sp,
            scale=1.0,
            cv=conn.speed[0],
            dt=scheme.dt,
        )

        nets = NetworkSet(subnets=[subnet], projections=[proj])
        sim = Simulator(nets=nets, simulation_length=simulation_length)
        sim.configure()
        ((t, y),) = sim.run()

        phases = y[:, 0, :, 0]
        order_param = np.abs(np.mean(np.exp(1j * phases), axis=1))
        final_order = order_param[-1]

        self.assertGreater(
            final_order,
            0.9,
            f"Ring network did not synchronize well. Final order: {final_order}",
        )

        print(f"3-node ring final order parameter: {final_order:.4f}")

    def test_effect_of_coupling_strength(self):
        """Test that synchronization depends on coupling strength."""
        dt = 0.1
        simulation_length = 50.0

        coupling_strengths = [0.1, 0.3, 0.5, 1.0]
        final_orders = []

        for strength in coupling_strengths:
            weights = np.array([[0.0, strength], [strength, 0.0]])
            lengths = np.array([[0.0, 5.0], [5.0, 0.0]])
            centres = np.array([[0, 0, 0], [10, 0, 0]])

            conn = Connectivity(
                weights=weights,
                tract_lengths=lengths,
                centres=centres,
                speed=np.array([5.0]),
                region_labels=["Node0", "Node1"],
            )
            conn.configure()

            scheme = HeunDeterministic(dt=dt)
            model = Kuramoto(omega=np.array([1.0]))

            subnet = Subnetwork(
                name="kuramoto_network", model=model, scheme=scheme, nnodes=2
            ).configure()

            weights_sp = sp.csr_matrix(conn.weights)
            lengths_sp = sp.csr_matrix(conn.tract_lengths)

            proj = projection_utils.create_intra_projection(
                subnet=subnet,
                source_cvar="theta",
                target_cvar="theta",
                weights=weights_sp,
                lengths=lengths_sp,
                scale=1.0,
                cv=conn.speed[0],
                dt=scheme.dt,
            )

            nets = NetworkSet(subnets=[subnet], projections=[proj])
            sim = Simulator(nets=nets, simulation_length=simulation_length)
            sim.configure()
            ((t, y),) = sim.run()

            phases = y[:, 0, :, 0]
            order_param = np.abs(np.mean(np.exp(1j * phases), axis=1))
            final_orders.append(order_param[-1])

        for i, (strength, order) in enumerate(zip(coupling_strengths, final_orders)):
            print(f"Coupling strength {strength}: final order = {order:.4f}")

        self.assertTrue(
            all(
                final_orders[i] <= final_orders[i + 1] + 0.05
                for i in range(len(final_orders) - 1)
            ),
            "Order parameter should increase with coupling strength",
        )
