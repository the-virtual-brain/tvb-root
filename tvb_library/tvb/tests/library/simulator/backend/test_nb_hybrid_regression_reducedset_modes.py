# -*- coding: utf-8 -*-
"""Multi-mode regressions for reduced-set models in the Numba hybrid backend."""

import numpy as np
import pytest
import scipy.sparse as sp

from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.hybrid.coupling import Scaling
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.hybrid.intra_projection import IntraProjection
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.integrators import EulerDeterministic
from tvb.simulator.models.stefanescu_jirsa import (
    ReducedSetFitzHughNagumo,
    ReducedSetHindmarshRose,
)


DT = 0.005
NSTEPS = 3
MODEL_CLASSES = (ReducedSetFitzHughNagumo, ReducedSetHindmarshRose)


def _model(model_class):
    # Keep this regression specific to projected coupling, not the models'
    # separate K-weighted internal mode mixing.
    return model_class(
        K11=np.array([0.0]),
        K12=np.array([0.0]),
        K21=np.array([0.0]),
    )


def _state(model, offset):
    """Deterministic state with independently varying nodes, variables, and modes."""
    state = np.empty((model.nvar, 2, model.number_of_modes), dtype=np.float64)
    mode_values = np.array([0.15, 0.55, 1.05])
    for var in range(model.nvar):
        for node in range(2):
            state[var, node] = mode_values + offset + 0.04 * var + 0.09 * node
    return state


def _projection_kwargs():
    weights = sp.csr_matrix(np.array([[0.0, 0.8], [0.45, 0.0]], dtype=np.float64))
    return {
        "source_cvar": np.array([0], dtype=np.int_),
        "target_cvar": np.array([0], dtype=np.int_),
        "weights": weights,
        "lengths": sp.csr_matrix(weights.shape, dtype=np.float64),
        "cv": 1.0,
        "dt": DT,
        "scale": 0.7,
        "cfun": Scaling(a=np.array([0.9])),
    }


def _build_case(model_class, projection_kind):
    if projection_kind == "intra":
        model = _model(model_class)
        projection = IntraProjection(**_projection_kwargs())
        target = Subnetwork(
            name="target",
            model=model,
            scheme=EulerDeterministic(dt=DT),
            nnodes=2,
            projections=[projection],
        )
        target.configure()
        network = NetworkSet(subnets=[target], projections=[])
        initial_states = [_state(model, 0.0)]
    else:
        source_model = _model(model_class)
        target_model = _model(model_class)
        source = Subnetwork(
            name="source",
            model=source_model,
            scheme=EulerDeterministic(dt=DT),
            nnodes=2,
        )
        target = Subnetwork(
            name="target",
            model=target_model,
            scheme=EulerDeterministic(dt=DT),
            nnodes=2,
        )
        source.configure()
        target.configure()
        projection = InterProjection(
            source=source,
            target=target,
            mode_map=np.array([[1, 0, 1], [0, 2, 0], [3, 0, 1]], dtype=np.int_),
            **_projection_kwargs(),
        )
        network = NetworkSet(subnets=[source, target], projections=[projection])
        initial_states = [_state(source_model, 0.0), _state(target_model, 0.23)]

    network.configure()
    return network, initial_states


def _total_initial_coupling(network, initial_states):
    states = network.States(*[state.copy() for state in initial_states])
    network.init_projection_buffers(states)
    external = network.cfun(1, states)
    return [
        ext + subnet.cfun(1, state)
        for subnet, state, ext in zip(network.subnets, states, external)
    ]


def _python_trajectory(network, initial_states):
    states = network.States(*[state.copy() for state in initial_states])
    network.init_projection_buffers(states)
    trajectory = [[] for _ in network.subnets]
    for step in range(1, NSTEPS + 1):
        states = network.step(step, states)
        for subnet_index, state in enumerate(states):
            trajectory[subnet_index].append(state.copy())
    return [np.stack(states, axis=0) for states in trajectory]


def _numba_trajectory(network, initial_states):
    compiled = NbHybridBackend().compile(network)
    trajectory = [[] for _ in network.subnets]
    # Prefix runs provide each full internal state through the snapshot API.
    for stop in range(1, NSTEPS + 1):
        _, snapshot = compiled.run(
            nstep=stop,
            chunk_size=1,
            initial_states=initial_states,
            return_snapshot=True,
        )
        for subnet_index, state in enumerate(snapshot["states"]):
            trajectory[subnet_index].append(state)
    return [np.stack(states, axis=0) for states in trajectory]


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_intra_analysis_exposes_all_source_modes(model_class):
    """Code generation must not silently compile an intra projection as mode 0 only."""
    network, _ = _build_case(model_class, "intra")
    analysis = NbHybridBackend()._analyse(network)

    assert len(analysis.intra_projections) == 1
    assert analysis.intra_projections[0].n_src_modes == 3
    assert analysis.intra_projections[0].n_tgt_modes == 3


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
@pytest.mark.parametrize("projection_kind", ("intra", "inter"))
def test_projected_coupling_requires_all_modes_and_combined_sum(
    model_class, projection_kind
):
    """The fixture distinguishes both mode-0-only and per-mode dfun inputs."""
    network, initial_states = _build_case(model_class, projection_kind)
    coupling = _total_initial_coupling(network, initial_states)[-1]

    assert coupling.shape == (2, 2, 3)
    assert np.max(np.abs(coupling[0, :, 1:])) > 0.05
    summed = coupling[0].sum(axis=-1, keepdims=True)
    assert np.max(np.abs(summed - coupling[0, :, :1])) > 0.05
    assert np.max(np.ptp(coupling[0], axis=-1)) > 0.05

    model = network.subnets[-1].model
    expected = model.dfun(initial_states[-1], coupling)
    wrong_per_mode = expected.copy()
    gain = float(model.tau[0]) if isinstance(model, ReducedSetFitzHughNagumo) else 1.0
    affected = (0, 2) if isinstance(model, ReducedSetFitzHughNagumo) else (0, 3)
    for state_variable in affected:
        wrong_per_mode[state_variable] += gain * (coupling[0] - summed)
    assert np.max(np.abs(expected - wrong_per_mode)) > 0.05


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_reduced_set_intra_projection_full_mode_trajectory_parity(model_class):
    network, initial_states = _build_case(model_class, "intra")
    python = _python_trajectory(network, initial_states)
    numba = _numba_trajectory(network, initial_states)

    np.testing.assert_allclose(
        numba[0],
        python[0],
        rtol=3e-4,
        atol=3e-5,
        err_msg="Numba intra trajectory lost modes or passed per-mode coupling to combined dfun",
    )


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_reduced_set_inter_projection_full_mode_trajectory_parity(model_class):
    network, initial_states = _build_case(model_class, "inter")
    python = _python_trajectory(network, initial_states)
    numba = _numba_trajectory(network, initial_states)

    for subnet_index, (numba_states, python_states) in enumerate(zip(numba, python)):
        np.testing.assert_allclose(
            numba_states,
            python_states,
            rtol=3e-4,
            atol=3e-5,
            err_msg=(
                f"Numba inter trajectory for subnet {subnet_index} lost modes or "
                "passed per-mode coupling to combined dfun"
            ),
        )
