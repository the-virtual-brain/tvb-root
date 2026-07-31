"""Regression coverage for stimulus coupling-slot and node selectivity."""

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose

from tvb.datatypes import equations
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.patterns import StimuliRegion
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.hybrid.intra_projection import IntraProjection
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.stimulus import Stim
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.integrators import EulerDeterministic
from tvb.simulator.models.epileptor import Epileptor


DT = 0.01
N_NODES = 3
N_STEPS = 6
TARGET_NODE = 1
AMPLITUDE = 0.4
RESPONSE_STATE = {0: 0, 1: 3}


def _make_network(target_slot, with_projection=False):
    model = Epileptor(
        variables_of_interest=Epileptor.state_variables,
        Kvf=np.array([1.0]),
        Kf=np.array([1.0]),
    )
    assert np.array_equal(model.cvar, [0, 3])
    subnet = Subnetwork(
        name="epileptor",
        model=model,
        scheme=EulerDeterministic(dt=DT),
        nnodes=N_NODES,
    )

    if target_slot is not None:
        connectivity = Connectivity(
            centres=np.zeros((N_NODES, 3)),
            weights=np.zeros((N_NODES, N_NODES)),
            tract_lengths=np.zeros((N_NODES, N_NODES)),
            region_labels=np.array(["left", "target", "right"]),
            speed=np.array([1.0]),
        )
        connectivity.configure()
        temporal = equations.Linear()
        temporal.parameters["a"] = 0.0
        temporal.parameters["b"] = AMPLITUDE
        spatial_weights = np.zeros(N_NODES)
        spatial_weights[TARGET_NODE] = 1.0
        pattern = StimuliRegion(
            temporal=temporal,
            connectivity=connectivity,
            weight=spatial_weights,
        )
        subnet.stimuli = [
            Stim(
                target=subnet,
                stimulus=pattern,
                target_cvar=np.array([target_slot], dtype=np.int_),
            )
        ]

    if with_projection:
        subnet.projections = [
            IntraProjection(
                source_cvar=np.array([0], dtype=np.int_),
                target_cvar=np.array([0], dtype=np.int_),
                weights=sp.csr_matrix((N_NODES, N_NODES), dtype=np.float64),
                lengths=sp.csr_matrix((N_NODES, N_NODES), dtype=np.float64),
                cv=1.0,
                dt=DT,
                cfun=Linear(),
            )
        ]

    subnet.configure(simulation_length=N_STEPS * DT)
    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()
    return network


def _initial_state():
    return np.array(
        [
            [[-1.50], [-1.55], [-1.60]],
            [[-10.0], [-10.2], [-10.4]],
            [[3.0], [3.1], [3.2]],
            [[-0.50], [-0.55], [-0.60]],
            [[0.01], [0.02], [0.03]],
            [[0.0], [0.1], [0.2]],
        ],
        dtype=np.float64,
    )


def _run_python(target_slot):
    network = _make_network(target_slot)
    subnet = network.subnets[0]
    states = network.zero_states(initial_states=[_initial_state()])
    network.init_projection_buffers(states)
    state_history = []
    coupling_history = []

    for step in range(1, N_STEPS + 1):
        coupling = network.cfun(step, states)[0]
        internal_coupling = subnet.cfun(step, states[0])
        for stimulus in subnet.stimuli:
            internal_coupling[stimulus.target_cvar] += stimulus.get_coupling(step)
        coupling_history.append((coupling + internal_coupling).copy())

        states = network.step(step, states)
        state_history.append(states[0].copy())

    return np.stack(state_history), np.stack(coupling_history)


def _run_numba(target_slot):
    network = _make_network(target_slot)
    result = NbHybridBackend().run_network(
        network,
        nstep=N_STEPS,
        chunk_size=1,
        initial_states=[_initial_state()],
    )[0]
    _times, state_history, coupling_history = result
    return state_history.astype(np.float64), coupling_history.astype(np.float64)


def _run_numba_prange(target_slot):
    network = _make_network(target_slot, with_projection=True)
    result = NbHybridBackend().sweep(
        network,
        params={"epileptor.intra.a": np.array([0.0, 0.0], dtype=np.float32)},
        nstep=N_STEPS,
        backend="cpu",
        n_workers=2,
        initial_states=[_initial_state()],
    )
    return (
        result.tavg["epileptor"][0].astype(np.float64),
        result.ctavg["epileptor"][0].astype(np.float64),
    )


def _assert_slot_and_node_selectivity(runs):
    baseline_state, baseline_coupling = runs[None]

    for target_slot in (0, 1):
        state, coupling = runs[target_slot]
        other_slot = 1 - target_slot
        response_state = RESPONSE_STATE[target_slot]
        other_response_state = RESPONSE_STATE[other_slot]

        # The requested slot responds at the selected node on every step.
        assert np.all(
            np.abs(
                coupling[:, target_slot, TARGET_NODE, :]
                - baseline_coupling[:, target_slot, TARGET_NODE, :]
            )
            > 0.1
        )
        assert (
            abs(
                state[0, response_state, TARGET_NODE, 0]
                - baseline_state[0, response_state, TARGET_NODE, 0]
            )
            > 0.003
        )

        # A one-slot stimulus must not broadcast onto the other coupling input.
        assert_allclose(
            coupling[:, other_slot],
            baseline_coupling[:, other_slot],
            rtol=0.0,
            atol=1e-7,
        )
        # Euler exposes the same invariant in state before populations mix it.
        assert_allclose(
            state[0, other_response_state, TARGET_NODE],
            baseline_state[0, other_response_state, TARGET_NODE],
            rtol=0.0,
            atol=1e-7,
        )

        non_target_nodes = [node for node in range(N_NODES) if node != TARGET_NODE]
        assert_allclose(
            coupling[:, target_slot, non_target_nodes],
            baseline_coupling[:, target_slot, non_target_nodes],
            rtol=0.0,
            atol=1e-7,
        )
        assert_allclose(
            state[:, :, non_target_nodes],
            baseline_state[:, :, non_target_nodes],
            rtol=0.0,
            atol=1e-7,
        )


def test_stimulus_targets_exactly_one_coupling_slot_and_node():
    python_runs = {slot: _run_python(slot) for slot in (None, 0, 1)}
    numba_runs = {slot: _run_numba(slot) for slot in (None, 0, 1)}

    _assert_slot_and_node_selectivity(python_runs)
    _assert_slot_and_node_selectivity(numba_runs)

    for slot in (None, 0, 1):
        python_state, python_coupling = python_runs[slot]
        numba_state, numba_coupling = numba_runs[slot]
        assert_allclose(numba_state, python_state, rtol=2e-5, atol=2e-6)
        assert_allclose(numba_coupling, python_coupling, rtol=2e-6, atol=2e-7)


def test_prange_stimulus_targets_exactly_one_coupling_slot_and_node():
    runs = {slot: _run_numba_prange(slot) for slot in (None, 0, 1)}
    _assert_slot_and_node_selectivity(runs)
