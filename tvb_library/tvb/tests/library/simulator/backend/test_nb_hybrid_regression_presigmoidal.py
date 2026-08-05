# -*- coding: utf-8 -*-
"""Regression coverage for dynamic PreSigmoidal in the Numba backend."""

import numpy as np
import pytest
import scipy.sparse as sp

from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.hybrid.coupling import PreSigmoidal
from tvb.simulator.hybrid.intra_projection import IntraProjection
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.integrators import EulerDeterministic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin


DT = 0.01
WEIGHTS = np.array(
    [
        [0.10, 0.30, 0.20],
        [0.40, 0.15, 0.25],
        [0.20, 0.35, 0.10],
    ],
    dtype=np.float64,
)
INITIAL_STATE = np.array(
    [
        [[0.20], [0.55], [0.85]],
        [[-0.35], [0.10], [0.45]],
    ],
    dtype=np.float64,
)
PARAMS = {
    "H": np.array([0.73]),
    "Q": np.array([0.19]),
    "G": np.array([1.37]),
    "P": np.array([0.82]),
    "theta": np.array([0.31]),
    "dynamic": True,
}


def _make_network(global_threshold):
    model = MontbrioPazoRoxin()
    model.configure()
    subnetwork = Subnetwork(
        name="mpr",
        model=model,
        scheme=EulerDeterministic(dt=DT),
        nnodes=WEIGHTS.shape[0],
    )
    projection = IntraProjection(
        source_cvar=np.array([0, 1], dtype=np.int_),
        target_cvar=np.array([0], dtype=np.int_),
        weights=sp.csr_matrix(WEIGHTS),
        lengths=sp.csr_matrix(WEIGHTS.shape, dtype=np.float64),
        cv=1.0,
        dt=DT,
        scale=0.67,
        cfun=PreSigmoidal(globalT=global_threshold, **PARAMS),
    )
    subnetwork.projections = [projection]
    subnetwork.configure()
    network = NetworkSet(subnets=[subnetwork], projections=[])
    network.configure()
    return network, projection


def _expected_coupling(global_threshold):
    signal = INITIAL_STATE[0, :, 0]
    threshold = INITIAL_STATE[1, :, 0]
    if global_threshold:
        threshold = np.full_like(threshold, threshold.mean())
    transformed = PARAMS["H"][0] * (
        PARAMS["Q"][0]
        + np.tanh(PARAMS["G"][0] * (PARAMS["P"][0] * signal - threshold))
    )
    return 0.67 * WEIGHTS.dot(transformed)


def _python_coupling(global_threshold):
    network, projection = _make_network(global_threshold)
    states = network.States(INITIAL_STATE.copy())
    network.init_projection_buffers(states)
    coupling = network.zero_cvars()[0]
    projection.apply(coupling, t=1, n_modes=1, x_i=INITIAL_STATE)
    return coupling


def _run_python(global_threshold, nstep):
    network, _ = _make_network(global_threshold)
    state = network.States(INITIAL_STATE.copy())
    network.init_projection_buffers(state)
    trajectory = []
    for step in range(1, nstep + 1):
        state = network.step(step, state)
        trajectory.append(state[0].copy())
    return np.stack(trajectory)


def _run_numba(global_threshold, nstep):
    network, _ = _make_network(global_threshold)
    result = NbHybridBackend().run_network(
        network,
        nstep=nstep,
        chunk_size=1,
        initial_states=[INITIAL_STATE.copy()],
    )
    return result[0][1]


@pytest.mark.parametrize("global_threshold", [False, True], ids=["local", "global"])
def test_dynamic_presigmoidal_direct_coupling_collapses_two_cvars_once(global_threshold):
    coupling = _python_coupling(global_threshold)
    expected = _expected_coupling(global_threshold)

    assert coupling.shape == (2, WEIGHTS.shape[0], 1)
    np.testing.assert_allclose(coupling[0, :, 0], expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(coupling[1], 0.0)
    assert not np.allclose(coupling[0, :, 0], 2.0 * expected, rtol=1e-3, atol=1e-4), (
        "fixture cannot detect accumulation of the collapsed coupling once per source cvar"
    )


def test_dynamic_presigmoidal_fixture_is_sensitive_to_global_threshold_semantics():
    local = _python_coupling(False)[0, :, 0]
    global_ = _python_coupling(True)[0, :, 0]

    assert np.max(np.abs(local - global_)) > 1e-2, (
        "heterogeneous thresholds must distinguish globalT=True from local thresholds"
    )


@pytest.mark.parametrize("global_threshold", [False, True], ids=["local", "global"])
def test_dynamic_presigmoidal_single_step_matches_python(global_threshold):
    python = _run_python(global_threshold, nstep=1)
    numba = _run_numba(global_threshold, nstep=1)

    assert numba.shape == python.shape
    np.testing.assert_allclose(numba, python, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("global_threshold", [False, True], ids=["local", "global"])
def test_dynamic_presigmoidal_trajectory_matches_python(global_threshold):
    python = _run_python(global_threshold, nstep=12)
    numba = _run_numba(global_threshold, nstep=12)

    assert np.all(np.isfinite(python))
    assert np.all(np.isfinite(numba))
    np.testing.assert_allclose(numba, python, rtol=2e-4, atol=2e-5)


def test_dynamic_presigmoidal_global_threshold_changes_numba_trajectory():
    local = _run_numba(False, nstep=12)
    global_ = _run_numba(True, nstep=12)

    assert np.max(np.abs(local - global_)) > 1e-4, (
        "Numba must implement globalT=True rather than silently using local thresholds"
    )
