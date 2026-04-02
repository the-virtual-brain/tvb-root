# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
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
Educational toy scenario: 2-node Kuramoto synchronisation.

Two Kuramoto phase oscillators are coupled via a symmetric intra-projection
with a short axonal delay (5 mm / 5 mm·ms⁻¹ = 1 ms).  With sufficient
coupling strength the two oscillators lock in phase, quantified by:

- **Phase difference** – ``|angle(exp(i*(θ₁−θ₀)))|`` should fall below 0.1 rad.
- **Order parameter** ``R`` – ``|mean(exp(i*θ))|`` should exceed 0.95.

These tests serve as a minimal sanity check that the hybrid simulator
reproduces the well-known synchronisation behaviour of the Kuramoto model.
"""

import numpy as np
from scipy import sparse as sp
from tvb.tests.library.simulator.hybrid.test_validation_base import ValidationTestBase
from tvb.simulator.hybrid import Subnetwork, NetworkSet, Simulator, projection_utils
from tvb.simulator.models import Kuramoto
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.monitors import TemporalAverage
from tvb.datatypes.connectivity import Connectivity


class TestToySynchronization(ValidationTestBase):
    """Two-node Kuramoto synchronisation demonstration.

    Physical interpretation: mutual excitatory coupling (weight = 0.5, or 0.3
    for the slower test) with a 1-ms axonal delay is expected to drive phase
    locking within 100–200 ms.  Tests assert synchronisation at the end of
    the simulation rather than during transient dynamics.
    """

    def test_two_node_synchronization(self):
        """Two strongly coupled Kuramoto oscillators must phase-lock within 100 ms.

        Coupling weight is 0.5 (strong) with a 1-ms axonal delay.  After
        100 ms the following criteria must hold:

        - Instantaneous phase difference ``|angle(exp(i*(θ₁−θ₀)))|`` < 0.1 rad.
        - Kuramoto order parameter ``R = |mean(exp(i*θ))|`` > 0.95.
        """
        weights = np.array([[0.0, 0.5], [0.5, 0.0]])
        lengths = np.array([[0.0, 5.0], [5.0, 0.0]])
        centres = np.array([[0, 0, 0], [10, 0, 0]])

        conn = Connectivity(
            weights=weights,
            tract_lengths=lengths,
            centres=centres,
            speed=np.array([5.0]),
            region_labels=np.array(["Node0", "Node1"]),
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
        sim = Simulator(
            nets=nets, simulation_length=100.0, monitors=[TemporalAverage(period=1.0)]
        )
        sim.configure()
        ((t, y),) = sim.run()

        theta0 = y[:, 0, 0, 0]
        theta1 = y[:, 0, 1, 0]

        phase_diff = np.angle(np.exp(1j * (theta1 - theta0)))

        final_phase_diff = np.abs(phase_diff[-1])
        assert final_phase_diff < 0.1, (
            f"Nodes did not synchronize. Final phase diff: {final_phase_diff}"
        )

        order_param = np.abs(np.mean(np.exp(1j * y[:, 0, :, 0]), axis=1))
        final_order = order_param[-1]
        assert final_order > 0.95, f"Order parameter too low. Final: {final_order}"

        print(f"Final phase difference: {final_phase_diff:.4f} rad")
        print(f"Final order parameter: {final_order:.4f}")

    def test_synchronization_time(self):
        """Synchronisation should be achieved within half the simulation length.

        Uses a reduced coupling weight (0.3) over a 200-ms run.  The test finds
        the first time step at which the phase difference drops below 0.1 rad
        and asserts it occurs before t = 100 ms.  If no full synchronisation is
        observed the test prints a diagnostic warning rather than failing,
        since this coupling strength may be marginal.
        """
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
            region_labels=np.array(["Node0", "Node1"]),
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
        sim = Simulator(nets=nets, simulation_length=simulation_length, monitors=[TemporalAverage(period=1.0)])
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
            assert sync_time < simulation_length * 0.5, (
                "Nodes should synchronize within half the simulation time"
            )
        else:
            final_diff = phase_diff[-1]
            print(
                f"Warning: No full synchronization achieved. Final diff: {final_diff:.4f} rad"
            )
            assert final_diff < 0.5, (
                f"Nodes did not synchronize significantly. Final diff: {final_diff}"
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
            region_labels=np.array(["Node0", "Node1"]),
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
        sim = Simulator(nets=nets, simulation_length=simulation_length, monitors=[TemporalAverage(period=1.0)])
        sim.configure()
        ((t, y),) = sim.run()

        phases = y[:, 0, :, 0]
        order_param = np.abs(np.mean(np.exp(1j * phases), axis=1))
        final_order = order_param[-1]

        assert final_order > 0.9, (
            f"Ring network did not synchronize well. Final order: {final_order}"
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
                region_labels=np.array(["Node0", "Node1"]),
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
            sim = Simulator(
                nets=nets,
                simulation_length=simulation_length,
                monitors=[TemporalAverage(period=1.0)],
            )
            sim.configure()
            ((t, y),) = sim.run()

            phases = y[:, 0, :, 0]
            order_param = np.abs(np.mean(np.exp(1j * phases), axis=1))
            final_orders.append(order_param[-1])

        for i, (strength, order) in enumerate(zip(coupling_strengths, final_orders)):
            print(f"Coupling strength {strength}: final order = {order:.4f}")

        assert all(
            final_orders[i] <= final_orders[i + 1] + 0.05
            for i in range(len(final_orders) - 1)
        ), "Order parameter should increase with coupling strength"
