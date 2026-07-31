"""Contract regressions for sequential and CPU-prange sweep monitors."""

import numpy as np
import pytest
import scipy.sparse as sp

from tvb.simulator.backend.nb_hybrid import NbHybridBackend, SweepResult
from tvb.simulator.hybrid.coupling import Scaling
from tvb.simulator.hybrid.intra_projection import IntraProjection
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.oscillator import Generic2dOscillator


DT = 0.1
NSTEP = 6
SWEEP_VALUES = np.array([0.2, 0.8], dtype=np.float32)
INITIAL_STATE = np.array(
    [
        [[-0.3], [0.1], [0.4]],
        [[0.2], [-0.1], [0.3]],
    ],
    dtype=np.float64,
)


def _network():
    model = Generic2dOscillator()
    model.configure()
    subnet = Subnetwork(
        name="tiny", model=model, scheme=HeunDeterministic(dt=DT), nnodes=3
    )
    weights = sp.csr_matrix(
        np.array(
            [
                [0.0, 0.3, 0.1],
                [0.2, 0.0, 0.4],
                [0.1, 0.2, 0.0],
            ],
            dtype=np.float64,
        )
    )
    subnet.projections = [
        IntraProjection(
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=weights,
            lengths=sp.csr_matrix(weights.shape, dtype=np.float64),
            cv=1.0,
            dt=DT,
            scale=1.0,
            cfun=Scaling(a=np.array([1.0])),
        )
    ]
    subnet.configure()
    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()
    return network


def _sweep(n_workers, **options):
    return NbHybridBackend().sweep(
        _network(),
        params={"coupling_scale": SWEEP_VALUES},
        nstep=NSTEP,
        backend="cpu",
        n_workers=n_workers,
        initial_states=[INITIAL_STATE.copy()],
        **options,
    )


def _assert_common_result_contract(sequential, parallel, expected_times):
    expected_times = np.asarray(expected_times, dtype=np.float64)
    n_samples = len(expected_times)
    assert isinstance(sequential, SweepResult)
    assert isinstance(parallel, SweepResult)
    assert sequential.backend == "cpu-seq"
    assert parallel.backend == "cpu-prange"
    np.testing.assert_array_equal(sequential.sweep_values[:, 0], SWEEP_VALUES)
    np.testing.assert_array_equal(parallel.sweep_values, sequential.sweep_values)

    sequential_data = sequential.tavg
    parallel_data = parallel.tavg
    assert set(sequential_data) == set(parallel_data) == {"tiny"}

    expected_shape = (len(SWEEP_VALUES), n_samples, 1, 3, 1)
    assert sequential.times.shape == parallel.times.shape == (n_samples,)
    assert sequential_data["tiny"].shape == expected_shape
    assert parallel_data["tiny"].shape == expected_shape
    np.testing.assert_allclose(parallel.times, sequential.times, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(sequential.times, expected_times, rtol=0.0, atol=1e-15)
    assert np.all(np.diff(sequential.times) > 0.0)
    np.testing.assert_allclose(
        parallel_data["tiny"], sequential_data["tiny"], rtol=2e-5, atol=2e-6
    )
    assert sequential.merged_tavg.shape == parallel.merged_tavg.shape == expected_shape
    np.testing.assert_allclose(
        sequential.merged_tavg, sequential_data["tiny"], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        parallel.merged_tavg, sequential.merged_tavg, rtol=2e-5, atol=2e-6
    )


def _assert_optional_monitor_data(sequential, parallel, field, n_samples):
    sequential_data = getattr(sequential, field)
    parallel_data = getattr(parallel, field)
    assert set(sequential_data) == set(parallel_data) == {"tiny"}
    expected_shape = (len(SWEEP_VALUES), n_samples, 1, 3, 1)
    assert sequential_data["tiny"].shape == expected_shape
    assert parallel_data["tiny"].shape == expected_shape
    np.testing.assert_allclose(
        parallel_data["tiny"], sequential_data["tiny"], rtol=2e-5, atol=2e-6
    )


@pytest.mark.parametrize(
    "options,expected_times,extra_field,extra_samples",
    [
        pytest.param(
            {"monitor": "tavg", "chunk_size": 2},
            (0.15, 0.35, 0.55),
            None,
            None,
            id="chunk-size",
        ),
        pytest.param(
            {"monitor": "raw", "chunk_size": 1},
            (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            "raw",
            NSTEP,
            id="monitor-raw",
        ),
        pytest.param(
            {"monitor": "subsample", "monitor_period": 2, "chunk_size": 1},
            (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            "raw",
            3,
            id="monitor-period",
        ),
    ],
)
def test_sequential_and_real_prange_share_monitor_sample_contract(
    options, expected_times, extra_field, extra_samples
):
    sequential = _sweep(1, **options)
    parallel = _sweep(2, **options)

    _assert_common_result_contract(sequential, parallel, expected_times)
    if extra_field is not None:
        _assert_optional_monitor_data(
            sequential, parallel, extra_field, extra_samples
        )
    assert not np.allclose(
        sequential.merged_tavg[0], sequential.merged_tavg[1], rtol=1e-7, atol=1e-7
    )


def test_sequential_and_real_prange_forward_bold_period_with_sample_axis():
    options = {"monitor": "tavg", "chunk_size": 2, "bold_period": 0.2}
    sequential = _sweep(1, **options)
    parallel = _sweep(2, **options)

    _assert_common_result_contract(sequential, parallel, (0.15, 0.35, 0.55))
    _assert_optional_monitor_data(sequential, parallel, "bold", 3)
