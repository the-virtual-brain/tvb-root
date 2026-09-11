"""Regressions for non-identity model.cvar handling in the hybrid backend."""

import numpy as np
import pytest
import scipy.sparse as sp

from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.hybrid.coupling import Difference, Kuramoto, Linear
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.hybrid.intra_projection import IntraProjection
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.integrators import EulerDeterministic
from tvb.simulator.models.jansen_rit import JansenRit


DT = 0.05
NSTEPS = 5
MAPPINGS = (
    pytest.param([2], [1], id="one-to-one"),
    pytest.param([2], [0, 1], id="one-to-many"),
    pytest.param([1, 2], [0], id="many-to-one"),
)


def _subnetwork(name, nnodes=2):
    model = JansenRit()
    model.configure()
    assert np.array_equal(model.cvar, [1, 2])
    subnet = Subnetwork(
        name=name,
        model=model,
        scheme=EulerDeterministic(dt=DT),
        nnodes=nnodes,
    )
    subnet.configure()
    return subnet


def _initial_state(model, nnodes, offset=0.0):
    """Distinct values make state indices 0, 1, and 2 non-interchangeable."""
    state = np.zeros((model.nvar, nnodes, model.number_of_modes), dtype=np.float64)
    node = np.arange(nnodes, dtype=np.float64)[:, None]
    state[0] = -0.7 + offset + 0.2 * node
    state[1] = 1.4 + offset + 0.3 * node
    state[2] = -1.1 - offset - 0.4 * node
    state[3] = 0.15 + 0.03 * node
    state[4] = -0.25 + 0.02 * node
    state[5] = 0.35 - 0.01 * node
    return state


def _weights(n_target, n_source):
    values = np.arange(1, n_target * n_source + 1, dtype=np.float64)
    values = values.reshape(n_target, n_source) / 3.0
    return sp.csr_matrix(values)


def _projection_kwargs(source_cvar, target_cvar, weights, cfun=None):
    return {
        "source_cvar": np.asarray(source_cvar, dtype=np.int_),
        # Targets index the coupling array, not the model state array.
        "target_cvar": np.asarray(target_cvar, dtype=np.int_),
        "weights": weights,
        "lengths": sp.csr_matrix(weights.shape, dtype=np.float64),
        "cv": 1.0,
        "dt": DT,
        "scale": 1.7,
        "cfun": cfun or Linear(a=np.array([1.3]), b=np.array([0.0])),
    }


def _python_trajectory(network, initial_states, nsteps=NSTEPS):
    states = network.zero_states(
        initial_states=[state.copy() for state in initial_states]
    )
    network.init_projection_buffers(states)
    trajectory = [[] for _ in network.subnets]
    for step in range(1, nsteps + 1):
        states = network.step(step, states)
        for output, state in zip(trajectory, states):
            output.append(np.asarray(state).copy())
    return [np.stack(output) for output in trajectory]


def _numba_trajectory(network, initial_states, nsteps=NSTEPS):
    """Use snapshots to retain every full state, including unobserved modes."""
    compiled = NbHybridBackend().compile(network)
    _, snapshot = compiled.run(
        1,
        chunk_size=1,
        initial_states=[state.copy() for state in initial_states],
        return_snapshot=True,
    )
    trajectory = [[state.copy()] for state in snapshot["states"]]
    for _ in range(1, nsteps):
        _, snapshot = compiled.resume(
            snapshot, 1, chunk_size=1, return_snapshot=True
        )
        for output, state in zip(trajectory, snapshot["states"]):
            output.append(state.copy())
    return [np.stack(output) for output in trajectory]


def _assert_trajectory_parity(network, initial_states):
    python = _python_trajectory(network, initial_states)
    numba = _numba_trajectory(network, initial_states)
    for subnet, expected, actual in zip(network.subnets, python, numba):
        assert expected.shape == actual.shape
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=4e-4,
            atol=3e-5,
            err_msg=f"full state trajectory differs for {subnet.name}",
        )


@pytest.mark.parametrize("source_cvar,target_cvar", MAPPINGS)
def test_intra_projection_maps_nonidentity_cvars(source_cvar, target_cvar):
    subnet = _subnetwork("jr")
    weights = _weights(subnet.nnodes, subnet.nnodes)
    subnet.projections = [
        IntraProjection(
            **_projection_kwargs(source_cvar, target_cvar, weights)
        )
    ]
    subnet.configure()
    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()

    _assert_trajectory_parity(
        network, [_initial_state(subnet.model, subnet.nnodes)]
    )


@pytest.mark.parametrize("source_cvar,target_cvar", MAPPINGS)
def test_inter_projection_maps_nonidentity_cvars(source_cvar, target_cvar):
    source = _subnetwork("source", nnodes=2)
    target = _subnetwork("target", nnodes=3)
    weights = _weights(target.nnodes, source.nnodes)
    projection = InterProjection(
        source=source,
        target=target,
        **_projection_kwargs(source_cvar, target_cvar, weights),
    )
    network = NetworkSet(subnets=[source, target], projections=[projection])
    network.configure()

    _assert_trajectory_parity(
        network,
        [
            _initial_state(source.model, source.nnodes, offset=0.25),
            _initial_state(target.model, target.nnodes, offset=-0.15),
        ],
    )


@pytest.mark.parametrize(
    "cfun,source_cvar,expected",
    (
        pytest.param(
            Difference(a=np.array([1.9])),
            [1, 2],
            lambda state: 1.9 * 1.7 * (state[1] - state[2]),
            id="difference-many-to-one",
        ),
        pytest.param(
            Kuramoto(a=np.array([2.1])),
            [2],
            lambda state: np.zeros_like(state[2]),
            id="kuramoto-one-to-one",
        ),
    ),
)
def test_target_state_lookup_maps_slot_through_model_cvar(
    cfun, source_cvar, expected
):
    """Target slot 1 denotes state model.cvar[1] (state index 2), not state 1."""
    subnet = _subnetwork("jr", nnodes=1)
    weights = sp.csr_matrix(np.ones((1, 1), dtype=np.float64))
    subnet.projections = [
        IntraProjection(
            **_projection_kwargs(source_cvar, [1], weights, cfun=cfun)
        )
    ]
    subnet.configure()
    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()
    initial = _initial_state(subnet.model, subnet.nnodes)

    states = network.zero_states(initial_states=[initial.copy()])
    network.init_projection_buffers(states)
    coupling = subnet.cfun(1, states[0])
    np.testing.assert_allclose(coupling[1], expected(initial), rtol=1e-7, atol=5e-7)

    _assert_trajectory_parity(network, [initial])
