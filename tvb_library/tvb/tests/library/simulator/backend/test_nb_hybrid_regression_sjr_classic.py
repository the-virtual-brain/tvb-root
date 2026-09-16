# -*- coding: utf-8 -*-
"""Regression coverage for classic SigmoidalJansenRit Numba coupling."""

import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.hybrid.coupling import SigmoidalJansenRit
from tvb.simulator.hybrid.intra_projection import IntraProjection
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.jansen_rit import JansenRit


DT = 0.01
N_NODES = 4
TWO_SOURCE_CVARS = np.array([1, 2], dtype=np.int_)


def _coupling(a=1.7):
    return SigmoidalJansenRit(
        a=np.array([a]),
        cmin=np.array([0.15]),
        cmax=np.array([1.35]),
        r=np.array([0.8]),
        midpoint=np.array([1.0]),
        use_classic=1,
    )


def _network(a=1.7, source_cvar=TWO_SOURCE_CVARS):
    model = JansenRit()
    model.configure()
    subnet = Subnetwork(
        name="jr",
        model=model,
        scheme=HeunDeterministic(dt=DT),
        nnodes=N_NODES,
    )

    weights = np.array(
        [
            [0.0, 0.30, 0.18, 0.00],
            [0.22, 0.00, 0.27, 0.16],
            [0.14, 0.25, 0.00, 0.31],
            [0.29, 0.17, 0.23, 0.00],
        ],
        dtype=np.float64,
    )
    delay_steps = np.array(
        [
            [0, 1, 3, 0],
            [2, 0, 1, 4],
            [3, 2, 0, 1],
            [1, 4, 2, 0],
        ],
        dtype=np.float64,
    )
    projection = IntraProjection(
        source_cvar=np.array(source_cvar, dtype=np.int_),
        target_cvar=np.array([0], dtype=np.int_),
        weights=sp.csr_matrix(weights),
        lengths=sp.csr_matrix(delay_steps * DT),
        cv=1.0,
        dt=DT,
        scale=0.45,
        cfun=_coupling(a),
    )
    subnet.projections = [projection]
    subnet.configure()
    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()
    return network


def _initial_state():
    state = np.zeros((6, N_NODES, 1), dtype=np.float64)
    state[0, :, 0] = [0.02, 0.05, -0.01, 0.04]
    state[1, :, 0] = [2.0, 3.0, 4.5, 5.5]
    state[2, :, 0] = [2.8, 2.2, 2.0, 1.5]
    state[3, :, 0] = [0.10, -0.05, 0.08, -0.02]
    state[4, :, 0] = [0.20, -0.10, 0.15, 0.05]
    state[5, :, 0] = [-0.08, 0.12, -0.04, 0.09]
    return state


def _python_coupling(network, state, suppress_fallback_warning=False):
    states = network.States(state.copy())
    network.init_projection_buffers(states)
    context = warnings.catch_warnings() if suppress_fallback_warning else warnings.catch_warnings()
    with context:
        if suppress_fallback_warning:
            warnings.simplefilter("ignore", RuntimeWarning)
        return network.subnets[0].cfun(1, state.copy())


def _numba_coupling(network, state):
    _, _, ctavg = NbHybridBackend().run_network(
        network,
        nstep=1,
        chunk_size=1,
        initial_states=[state.copy()],
    )[0]
    return ctavg[0]


def _classic_reference(network, state, a):
    projection = network.subnets[0].projections[0]
    cfun = projection.cfun
    source_difference = state[1, :, 0] - state[2, :, 0]
    sigmoid = cfun.cmin[0] + (cfun.cmax[0] - cfun.cmin[0]) / (
        1.0 + np.exp(cfun.r[0] * (cfun.midpoint[0] - source_difference))
    )
    expected = projection.scale * a * projection.weights.dot(sigmoid)
    result = np.zeros((2, N_NODES, 1), dtype=np.float64)
    result[0, :, 0] = expected
    return result


def _python_trajectory(network, state, nstep):
    states = network.States(state.copy())
    network.init_projection_buffers(states)
    output = []
    for step in range(1, nstep + 1):
        states = network.step(step, states)
        observed = network.subnets[0].model.observe(states[0])
        output.append(observed.sum(axis=-1)[..., np.newaxis])
    return np.stack(output)


def _numba_trajectory(network, state, nstep):
    return NbHybridBackend().run_network(
        network,
        nstep=nstep,
        chunk_size=1,
        initial_states=[state.copy()],
    )[0][1]


def test_classic_two_cvar_collapse_is_accumulated_once():
    network = _network(a=1.0)
    state = _initial_state()
    reference = _classic_reference(network, state, a=1.0)
    python_coupling = _python_coupling(network, state)
    numba_coupling = _numba_coupling(network, state)

    np.testing.assert_allclose(python_coupling, reference, rtol=1e-8, atol=2e-9)
    assert np.max(np.abs(reference)) > 0.1
    assert np.max(np.abs(reference - 2.0 * reference)) > 0.1
    np.testing.assert_allclose(
        numba_coupling,
        reference,
        rtol=2e-5,
        atol=2e-6,
        err_msg="classic 2-to-1 collapse was accumulated more than once",
    )


def test_classic_nonunit_amplitude_is_not_omitted():
    state = _initial_state()
    unit = _numba_coupling(_network(a=1.0), state)
    amplified_network = _network(a=1.7)
    amplified = _numba_coupling(amplified_network, state)
    reference = _classic_reference(amplified_network, state, a=1.7)

    assert np.max(np.abs(reference - reference / 1.7)) > 0.05
    np.testing.assert_allclose(amplified, 1.7 * unit, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(amplified, reference, rtol=2e-5, atol=2e-6)


def test_classic_two_cvar_delayed_trajectory_matches_for_101_steps():
    state = _initial_state()
    python = _python_trajectory(_network(a=1.7), state, nstep=101)
    numba = _numba_trajectory(_network(a=1.7), state, nstep=101)

    assert np.all(np.isfinite(python))
    assert np.ptp(python[:, 1, :, 0]) > 0.1
    np.testing.assert_allclose(
        numba,
        python,
        rtol=2e-3,
        atol=2e-3,
        err_msg="delayed classic SigmoidalJansenRit trajectory diverged",
    )


def test_explicit_one_cvar_is_not_silently_expanded_to_classic_branch():
    state = _initial_state()
    network = _network(a=1.3, source_cvar=np.array([1], dtype=np.int_))
    cfun = network.subnets[0].projections[0].cfun
    one_cvar_input = state[[1]]
    two_cvar_input = state[[1, 2]]

    with pytest.warns(RuntimeWarning, match="Falling back to legacy"):
        legacy_reference = cfun.pre(one_cvar_input)
    classic_reference = cfun.pre(two_cvar_input)
    assert np.max(np.abs(legacy_reference - classic_reference)) > 0.1

    analysis = NbHybridBackend()._analyse(network)
    analysed = analysis.intra_projections[0].source_cvar
    np.testing.assert_array_equal(
        analysed,
        np.array([1], dtype=np.int32),
        err_msg="Numba changed an explicit one-cvar projection into a two-cvar branch",
    )

    python_coupling = _python_coupling(network, state, suppress_fallback_warning=True)
    numba_coupling = _numba_coupling(network, state)
    np.testing.assert_allclose(numba_coupling, python_coupling, rtol=2e-5, atol=2e-6)
