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
Integration tests for stimulus functionality in the hybrid simulator.

Validates that external stimuli routed through
:func:`~tvb.simulator.hybrid.stimulus_utils.create_stimulus` have a
measurable impact on simulation output, compared with unstimulated baseline
runs.  Three scenarios are covered:

- **Direct impact** (``test_stimulus_impact_on_simulation``) – the stimulated
  node diverges from its baseline trajectory and has higher mean activity.
- **Spatial selectivity** (``test_stimulus_impact_decreases_with_distance``) –
  only the directly stimulated node has elevated final state; non-stimulated
  nodes remain at baseline because no inter-node coupling is present.
- **Time-varying stimulus** (``test_time_varying_stimulus``) – a
  cosine-modulated stimulus produces plausible time-dependent dynamics.
"""

import numpy as np
import pytest

from tvb.datatypes import patterns, equations
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator import models, integrators, monitors
from tvb.simulator.hybrid import (
    Subnetwork,
    NetworkSet,
    Simulator,
    Stim,
    stimulus_utils,
)


def setup_linear_network(n_nodes=3, dt=0.1):
    """Create a simple linear network for testing.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in network
    dt : float
        Integration time step

    Returns
    -------
    conn : Connectivity
        Configured connectivity object
    subnet : Subnetwork
        Configured subnetwork with linear model
    """
    # Create simple connectivity
    conn = Connectivity(
        centres=np.ones((n_nodes, 3)),
        weights=np.ones((n_nodes, n_nodes)) * 0.1,  # Weak coupling
        tract_lengths=np.zeros((n_nodes, n_nodes)),  # No delays
        region_labels=np.array([f"region_{i}" for i in range(n_nodes)]),
        speed=np.array([1.0]),
    )
    conn.configure()

    # Use Generic2dOscillator (linear-like behavior)
    model = models.Generic2dOscillator()

    # Use Euler integration (simple, deterministic)
    scheme = integrators.EulerDeterministic(dt=dt)

    # Create subnetwork
    subnet = Subnetwork(
        name="linear_network",
        model=model,
        scheme=scheme,
        nnodes=n_nodes,
    )

    return conn, subnet


class TestStimulusIntegration:
    """Integration tests for :func:`~tvb.simulator.hybrid.stimulus_utils.create_stimulus`.

    Each test builds a minimal isolated network (no inter-node projections)
    via :func:`setup_linear_network`, applies a stimulus through a
    :class:`~tvb.simulator.hybrid.Stim` object, and compares the resulting
    trajectory against a no-stimulus baseline.  The absence of coupling
    ensures that any observed differences are caused solely by the stimulus.
    """

    def test_stimulus_impact_on_simulation(self):
        """Stimulated node must diverge from its no-stimulus baseline trajectory.

        A constant-amplitude (``b=0.5``) stimulus is applied to node 0 only.
        After 20 ms the stimulated node's first state variable must (a) differ
        from its no-stimulus trajectory and (b) have a higher time-averaged
        value (margin 1e-4).
        """
        simulation_length = 20.0  # Short simulation
        n_nodes = 3
        dt = 0.1

        # Setup network without stimulus
        conn, subnet = setup_linear_network(n_nodes=n_nodes, dt=dt)
        nets_no_stim = NetworkSet(
            subnets=[subnet],
            projections=[],
        )

        sim_no_stim = Simulator(
            nets=nets_no_stim,
            simulation_length=simulation_length,
            monitors=[monitors.Raw()],
        )
        sim_no_stim.configure()

        # Run simulation without stimulus
        result_no_stim = sim_no_stim.run()
        time_no_stim, states_no_stim = result_no_stim[0]  # Extract time and states

        # Setup network with stimulus
        conn, subnet = setup_linear_network(n_nodes=n_nodes, dt=dt)

        # Create constant stimulus on node 0
        stim_weights = np.zeros(n_nodes)
        stim_weights[0] = 1.0  # Stimulate only first node

        # Constant temporal profile (always on)
        temporal = equations.Linear()
        temporal.parameters["a"] = 0.0  # No time dependence
        temporal.parameters["b"] = 0.5  # Constant amplitude 0.5

        stim_pattern = patterns.StimuliRegion(
            temporal=temporal, connectivity=conn, weight=stim_weights
        )

        # Add stimulus to network
        stim = stimulus_utils.create_stimulus(
            target_subnet=subnet,
            stimulus=stim_pattern,
            stimulus_cvar=np.r_[0],  # First state variable
            projection_scale=1.0,
        )

        nets_with_stim = NetworkSet(
            subnets=[subnet],
            projections=[],
            stimuli=[stim],
        )

        sim_with_stim = Simulator(
            nets=nets_with_stim,
            simulation_length=simulation_length,
            monitors=[monitors.Raw()],
        )
        sim_with_stim.configure()

        # Run simulation with stimulus
        result_with_stim = sim_with_stim.run()
        time_with_stim, states_with_stim = result_with_stim[
            0
        ]  # Extract time and states

        # Verify states are different
        # The stimulated node (0) should have different dynamics
        # Compare time courses of first state variable at node 0
        assert not np.allclose(
            states_no_stim[:, 0, 0, 0], states_with_stim[:, 0, 0, 0], rtol=1e-5
        ), "Stimulated node should have different states with vs without stimulus"

        # The stimulus increases values at stimulated node
        # Check that stimulated node's first state variable is generally higher
        assert (
            np.mean(states_with_stim[:, 0, 0, 0])
            > np.mean(states_no_stim[:, 0, 0, 0]) + 1e-4
        ), "Stimulus should increase activity at stimulated node"

    def test_stimulus_impact_decreases_with_distance(self):
        """Directly stimulated node must have higher final activity than non-stimulated nodes.

        Node 0 receives direct drive; nodes 1 and 2 do not.  With no inter-node
        coupling the final state of node 0's first state variable must exceed
        those of nodes 1 and 2.
        """
        simulation_length = 20.0
        n_nodes = 3
        dt = 0.1

        # Setup network
        conn, subnet = setup_linear_network(n_nodes=n_nodes, dt=dt)

        # Create stimulus on node 0
        stim_weights = np.zeros(n_nodes)
        stim_weights[0] = 1.0  # Only stimulate node 0

        temporal = equations.Linear()
        temporal.parameters["a"] = 0.0
        temporal.parameters["b"] = 0.5

        stim_pattern = patterns.StimuliRegion(
            temporal=temporal, connectivity=conn, weight=stim_weights
        )

        # Add stimulus with weights (default all-to-all identity)
        stim = stimulus_utils.create_stimulus(
            target_subnet=subnet,
            stimulus=stim_pattern,
            stimulus_cvar=np.r_[0],
            projection_scale=1.0,
        )

        nets = NetworkSet(
            subnets=[subnet],
            projections=[],
            stimuli=[stim],
        )

        sim = Simulator(
            nets=nets,
            simulation_length=simulation_length,
            monitors=[monitors.Raw()],
        )
        sim.configure()

        # Run simulation
        result = sim.run()
        time, states = result[0]

        # Extract final states (last time point)
        final_states = states[-1]  # (cvar, node, mode)

        # Node 0 (stimulated) should have highest values
        assert final_states[0, 0, 0] > final_states[0, 1, 0], (
            "Stimulated node should have higher state than unstimulated nodes"
        )

        assert final_states[0, 0, 0] > final_states[0, 2, 0], (
            "Stimulated node should have higher state than other unstimulated nodes"
        )

    def test_time_varying_stimulus(self):
        """A cosine-modulated stimulus produces a time-varying trajectory.

        A low-frequency (0.1 Hz) cosine stimulus with amplitude 0.5 is applied
        to all nodes.  The test verifies that the simulation completes without
        error.
        """
        simulation_length = 30.0
        n_nodes = 2
        dt = 0.1

        # Setup network
        conn, subnet = setup_linear_network(n_nodes=n_nodes, dt=dt)

        # Create time-varying stimulus (ramping up then down)
        stim_weights = np.ones(n_nodes)  # Stimulate all nodes

        temporal = equations.Cosine()
        temporal.parameters["frequency"] = 0.1  # Low frequency oscillation
        temporal.parameters["amp"] = 0.5

        stim_pattern = patterns.StimuliRegion(
            temporal=temporal, connectivity=conn, weight=stim_weights
        )

        stim = stimulus_utils.create_stimulus(
            target_subnet=subnet,
            stimulus=stim_pattern,
            stimulus_cvar=np.r_[0],
            projection_scale=1.0,
        )

        nets = NetworkSet(
            subnets=[subnet],
            projections=[],
            stimuli=[stim],
        )

        sim = Simulator(
            nets=nets,
            simulation_length=simulation_length,
            monitors=[monitors.Raw()],
        )
        sim.configure()

        # Run simulation
        result = sim.run()
        time, states = result[0]

        # Extract time evolution of first state variable at node 0
        # states shape: (time, cvar, node, mode)
        state_time_course = states[:, 0, 0, 0]

        # Check that state varies over time (due to oscillating stimulus)
        std_deviation = np.std(state_time_course)
        assert std_deviation > 0.01, (
            "Time-varying stimulus should cause state to vary over time"
        )

        # Verify we have significant variation (not just noise)
        state_range = np.max(state_time_course) - np.min(state_time_course)
        assert state_range > 0.1, (
            "Time-varying stimulus should create measurable state variation"
        )

    def test_pulse_stimulus(self):
        """Test that pulse train stimulus creates discrete activations."""
        simulation_length = 30.0
        n_nodes = 2
        dt = 0.1

        # Setup network
        conn, subnet = setup_linear_network(n_nodes=n_nodes, dt=dt)

        # Create pulse train stimulus
        stim_weights = np.ones(n_nodes)

        temporal = equations.PulseTrain()
        temporal.parameters["onset"] = 5.0  # Start at 5ms
        temporal.parameters["T"] = 10.0  # Period of 10ms
        temporal.parameters["tau"] = 2.0  # Pulse width 2ms
        temporal.parameters["amp"] = 0.5

        stim_pattern = patterns.StimuliRegion(
            temporal=temporal, connectivity=conn, weight=stim_weights
        )

        stim = stimulus_utils.create_stimulus(
            target_subnet=subnet,
            stimulus=stim_pattern,
            stimulus_cvar=np.r_[0],
            projection_scale=1.0,
        )

        nets = NetworkSet(
            subnets=[subnet],
            projections=[],
            stimuli=[stim],
        )

        sim = Simulator(
            nets=nets,
            simulation_length=simulation_length,
            monitors=[monitors.Raw()],
        )
        sim.configure()

        # Run simulation
        result = sim.run()
        time, states = result[0]

        # Extract time course
        # states shape: (time, cvar, node, mode)
        state_time_course = states[:, 0, 0, 0]
        time_points = np.arange(len(state_time_course)) * dt

        # Check that before onset, states are lower (pulse hasn't started)
        pre_onset_mask = time_points < 5.0
        post_onset_mask = time_points >= 5.0

        # Pulse train creates variation, check standard deviation increases
        pre_onset_std = np.std(state_time_course[pre_onset_mask])
        post_onset_std = np.std(state_time_course[post_onset_mask])

        assert post_onset_std > pre_onset_std + 0.01, (
            "Pulse train should increase state variation after onset"
        )

    def test_spatially_varying_stimulus(self):
        """Test that spatially-varying stimulus creates node-specific effects."""
        simulation_length = 20.0
        n_nodes = 3
        dt = 0.1

        # Setup network
        conn, subnet = setup_linear_network(n_nodes=n_nodes, dt=dt)

        # Create spatially-varying stimulus
        stim_weights = np.array([1.0, 0.5, 0.2])  # Different strengths per node

        temporal = equations.Linear()
        temporal.parameters["a"] = 0.0
        temporal.parameters["b"] = 1.0

        stim_pattern = patterns.StimuliRegion(
            temporal=temporal, connectivity=conn, weight=stim_weights
        )

        stim = stimulus_utils.create_stimulus(
            target_subnet=subnet,
            stimulus=stim_pattern,
            stimulus_cvar=np.r_[0],
            projection_scale=1.0,
        )

        nets = NetworkSet(
            subnets=[subnet],
            projections=[],
            stimuli=[stim],
        )

        sim = Simulator(
            nets=nets,
            simulation_length=simulation_length,
            monitors=[monitors.Raw()],
        )
        sim.configure()

        # Run simulation
        result = sim.run()
        time, states = result[0]

        # Extract final states
        # states shape: (time, cvar, node, mode)
        final_states = states[-1]  # (cvar, node, mode) at last time step

        # Check spatial variation follows stimulus weights
        # Node with highest weight should have highest state
        # stim_weights = [1.0, 0.5, 0.2] for nodes 0, 1, 2
        assert final_states[0, 0, 0] > final_states[0, 1, 0], (
            "Node 0 (weight=1.0) should have higher state than node 1 (weight=0.5)"
        )

        assert final_states[0, 1, 0] > final_states[0, 2, 0], (
            "Node 1 (weight=0.5) should have higher state than node 2 (weight=0.2)"
        )

    def test_multiple_stimuli(self):
        """Test that multiple stimuli can be applied simultaneously."""
        simulation_length = 20.0
        n_nodes = 3
        dt = 0.1

        # Setup network
        conn, subnet = setup_linear_network(n_nodes=n_nodes, dt=dt)

        # Create two stimuli: one on node 0, one on node 2
        stim_weights_1 = np.zeros(n_nodes)
        stim_weights_1[0] = 0.5

        stim_weights_2 = np.zeros(n_nodes)
        stim_weights_2[2] = 0.5

        temporal = equations.Linear()
        temporal.parameters["a"] = 0.0
        temporal.parameters["b"] = 1.0

        stim_pattern_1 = patterns.StimuliRegion(
            temporal=temporal, connectivity=conn, weight=stim_weights_1
        )

        stim_pattern_2 = patterns.StimuliRegion(
            temporal=temporal, connectivity=conn, weight=stim_weights_2
        )

        stim1 = stimulus_utils.create_stimulus(
            target_subnet=subnet,
            stimulus=stim_pattern_1,
            stimulus_cvar=np.r_[0],
            projection_scale=1.0,
        )

        stim2 = stimulus_utils.create_stimulus(
            target_subnet=subnet,
            stimulus=stim_pattern_2,
            stimulus_cvar=np.r_[1],  # Different state variable
            projection_scale=1.0,
        )

        nets = NetworkSet(
            subnets=[subnet],
            projections=[],
            stimuli=[stim1, stim2],
        )

        sim = Simulator(
            nets=nets,
            simulation_length=simulation_length,
            monitors=[monitors.Raw()],
        )
        sim.configure()

        # Run simulation
        result = sim.run()
        time, states = result[0]

        # Extract final states
        final_states = states[-1]  # (cvar, node, mode)

        # Both stimulated nodes should have elevated states
        assert final_states[0, 0, 0] > final_states[0, 1, 0], (
            "Node 0 (stimulated) should have higher state than node 1 (unstimulated)"
        )

        assert final_states[0, 2, 0] > final_states[0, 1, 0], (
            "Node 2 (stimulated) should have higher state than node 1 (unstimulated)"
        )

    def test_stimulus_with_projection_scale(self):
        """Test that projection_scale controls stimulus intensity."""
        simulation_length = 20.0
        n_nodes = 2
        dt = 0.1

        # Setup network
        conn, subnet = setup_linear_network(n_nodes=n_nodes, dt=dt)

        # Create stimulus with small scale
        stim_weights = np.ones(n_nodes)
        temporal = equations.Linear()
        temporal.parameters["a"] = 0.0
        temporal.parameters["b"] = 1.0

        stim_pattern = patterns.StimuliRegion(
            temporal=temporal, connectivity=conn, weight=stim_weights
        )

        stim_small = stimulus_utils.create_stimulus(
            target_subnet=subnet,
            stimulus=stim_pattern,
            stimulus_cvar=np.r_[0],
            projection_scale=0.5,  # Small scale
        )

        stim_large = stimulus_utils.create_stimulus(
            target_subnet=subnet,
            stimulus=stim_pattern,
            stimulus_cvar=np.r_[0],
            projection_scale=2.0,  # Large scale
        )

        # Run with small scale
        nets_small = NetworkSet(
            subnets=[subnet],
            projections=[],
            stimuli=[stim_small],
        )

        sim_small = Simulator(
            nets=nets_small,
            simulation_length=simulation_length,
            monitors=[monitors.Raw()],
        )
        sim_small.configure()

        result_small = sim_small.run()
        time_small, states_small = result_small[0]
        final_small = states_small[-1]  # (cvar, node, mode)

        # Run with large scale
        nets_large = NetworkSet(
            subnets=[subnet],
            projections=[],
            stimuli=[stim_large],
        )

        sim_large = Simulator(
            nets=nets_large,
            simulation_length=simulation_length,
            monitors=[monitors.Raw()],
        )
        sim_large.configure()

        result_large = sim_large.run()
        time_large, states_large = result_large[0]
        final_large = states_large[-1]  # (cvar, node, mode)

        # Larger scale should produce larger states
        assert final_large[0, 0, 0] > final_small[0, 0, 0] + 0.1, (
            "Larger projection_scale should produce larger state values"
        )
