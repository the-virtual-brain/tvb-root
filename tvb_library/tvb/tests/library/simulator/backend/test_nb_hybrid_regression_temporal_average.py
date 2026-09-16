"""Regression coverage for Python TemporalAverage semantics in NbHybridBackend."""

import numpy as np
import pytest

from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.integrators import EulerDeterministic
from tvb.simulator.models.linear import Linear
from tvb.simulator.monitors import TemporalAverage


DT = 0.2
PERIOD_STEPS = 6
PERIOD = PERIOD_STEPS * DT
INITIAL_STATE = np.array([[[0.25], [0.75]]], dtype=np.float64)


def _network():
    model = Linear(gamma=np.array([-1.0]))
    model.configure()
    subnet = Subnetwork(
        name="linear",
        model=model,
        scheme=EulerDeterministic(dt=DT),
        nnodes=INITIAL_STATE.shape[1],
    )
    subnet.configure()
    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()
    return network


def _python_temporal_average(nstep):
    """Run the Python step loop and feed its observations to the real monitor."""
    network = _network()
    subnet = network.subnets[0]
    state = network.zero_states(initial_states=[INITIAL_STATE.copy()])
    network.init_projection_buffers(state)

    monitor = TemporalAverage(period=PERIOD)
    monitor._config_dt(DT)
    monitor.voi = np.arange(len(subnet.model.variables_of_interest))
    monitor._config_stock(
        monitor.voi.size,
        subnet.nnodes,
        subnet.model.number_of_modes,
    )

    times = []
    data = []
    observations = []
    for step in range(1, nstep + 1):
        state = network.step(step, state)
        observed = subnet.model.observe(np.asarray(state[0]))
        observed = observed.sum(axis=-1)[..., np.newaxis]
        observations.append(observed.copy())
        sample = monitor.sample(step, observed)
        if sample is not None:
            time, average = sample
            times.append(time)
            data.append(average.copy())

    return np.asarray(times), np.asarray(data), np.asarray(observations)


def _numba_temporal_average(nstep, chunk_size=None):
    results = NbHybridBackend().run_network(
        _network(),
        nstep=nstep,
        chunk_size=chunk_size,
        initial_states=[INITIAL_STATE.copy()],
        monitors=[TemporalAverage(period=PERIOD)],
    )
    return results[0][0]


def _assert_fixture_is_sensitive(times, data, observations):
    assert len(times) >= 2
    assert not np.allclose(data[0], data[1]), "successive averages must differ"
    assert not np.allclose(
        data[0], observations[PERIOD_STEPS - 1]
    ), "the average must not be indistinguishable from a window endpoint"
    assert not np.array_equal(
        times, times + 0.5 * DT
    ), "dt must expose the known half-step timestamp displacement"


def test_exact_timestamps_match_python_first_midpoint_and_multiple_periods():
    nstep = 3 * PERIOD_STEPS
    python_times, python_data, observations = _python_temporal_average(nstep)
    numba_times, _ = _numba_temporal_average(nstep)

    _assert_fixture_is_sensitive(python_times, python_data, observations)
    assert python_times[0] == (PERIOD_STEPS - PERIOD_STEPS / 2.0) * DT
    assert len(python_times) == 3
    np.testing.assert_array_equal(numba_times, python_times)


def test_data_matches_python_temporal_average_over_multiple_periods():
    nstep = 3 * PERIOD_STEPS
    python_times, python_data, observations = _python_temporal_average(nstep)
    _, numba_data = _numba_temporal_average(nstep)

    _assert_fixture_is_sensitive(python_times, python_data, observations)
    assert numba_data.shape == python_data.shape
    np.testing.assert_allclose(numba_data, python_data, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize(
    "chunk_size",
    [PERIOD_STEPS, PERIOD_STEPS // 3],
    ids=["period-sized-chunks", "period-spans-chunk-boundaries"],
)
def test_exact_parity_omits_non_divisible_tail_across_chunk_boundaries(chunk_size):
    nstep = 2 * PERIOD_STEPS + 2
    assert PERIOD_STEPS % chunk_size == 0
    assert nstep % PERIOD_STEPS != 0

    python_times, python_data, observations = _python_temporal_average(nstep)
    numba_times, numba_data = _numba_temporal_average(nstep, chunk_size=chunk_size)

    _assert_fixture_is_sensitive(python_times, python_data, observations)
    assert len(python_times) == nstep // PERIOD_STEPS
    np.testing.assert_array_equal(numba_times, python_times)
    np.testing.assert_allclose(numba_data, python_data, rtol=2e-6, atol=2e-7)
