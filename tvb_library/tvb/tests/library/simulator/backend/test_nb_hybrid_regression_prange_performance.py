"""Warmed throughput regression for the CPU-prange sweep backend."""

import os
import time

import numba
import numpy as np
import pytest
import scipy.sparse as sp

from tvb.simulator.backend.nb_hybrid import NbHybridBackend, _COMPILED_FN_CACHE
from tvb.simulator.backend.nb_hybrid_sweep_cpu import _SWEEP_KERNEL_CACHE
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.hybrid.intra_projection import IntraProjection
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin


N_NODES = 32
N_STEPS = 250
N_SWEEPS = 64
REPEATS = 5
MIN_SPEEDUP_ENV = "TVB_PRANGE_MIN_SPEEDUP"


def _make_network():
    model = MontbrioPazoRoxin()
    model.configure()
    subnet = Subnetwork(
        name="ctx",
        model=model,
        scheme=HeunDeterministic(dt=0.01),
        nnodes=N_NODES,
    )
    rng = np.random.RandomState(2026)
    weights = rng.uniform(0.0, 0.02, (N_NODES, N_NODES))
    np.fill_diagonal(weights, 0.0)
    subnet.projections = [
        IntraProjection(
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=sp.csr_matrix(weights),
            lengths=sp.csr_matrix((N_NODES, N_NODES), dtype=np.float64),
            cv=1.0,
            dt=0.01,
            scale=0.5,
            cfun=Linear(a=np.array([0.03]), b=np.array([0.0])),
        )
    ]
    subnet.configure()
    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()
    initial_state = np.empty((model.nvar, N_NODES, model.number_of_modes))
    initial_state[0] = 0.1
    initial_state[1] = -2.0
    return network, [initial_state]


def _median_timing(run):
    durations = []
    results = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        results.append(run())
        durations.append(time.perf_counter() - started)
    return float(np.median(durations)), results


@pytest.mark.slow
def test_warmed_cpu_prange_median_throughput_and_dispatch():
    """Measure warmed kernels and optionally enforce a machine-specific gate.

    Set ``TVB_PRANGE_MIN_SPEEDUP`` on a dedicated performance runner to its
    calibrated minimum. Shared CI still checks real parallel dispatch without
    assuming that an oversubscribed or throttled runner must show a fixed gain.
    """
    configured_threads = numba.get_num_threads()
    assert configured_threads > 1, (
        "CPU-prange performance requires more than one configured Numba thread"
    )
    workers = min(4, configured_threads)
    assert workers > 1

    network, initial_states = _make_network()
    values = np.linspace(0.02, 0.06, N_SWEEPS, dtype=np.float32)
    params = {"ctx.intra.a": values}
    backend = NbHybridBackend()

    def run_sequential():
        return backend.sweep(
            network,
            params=params,
            nstep=N_STEPS,
            backend="cpu",
            n_workers=1,
            initial_states=initial_states,
        )

    def run_parallel():
        return backend.sweep(
            network,
            params=params,
            nstep=N_STEPS,
            backend="cpu",
            n_workers=workers,
            initial_states=initial_states,
        )

    # Full-size warmups force both JIT paths before either clock starts.
    warm_seq = run_sequential()
    warm_par = run_parallel()
    assert warm_seq.backend == "cpu-seq"
    assert warm_par.backend == "cpu-prange"
    np.testing.assert_allclose(
        warm_seq.merged_tavg,
        warm_par.merged_tavg,
        rtol=2e-5,
        atol=2e-6,
    )

    assert _SWEEP_KERNEL_CACHE
    sweep_cache_before = {key: id(kernel) for key, kernel in _SWEEP_KERNEL_CACHE.items()}
    compiled_cache_before = {key: id(kernel) for key, kernel in _COMPILED_FN_CACHE.items()}
    prange_kernel = next(reversed(_SWEEP_KERNEL_CACHE.values()))
    signatures_before = tuple(prange_kernel.signatures)
    assert prange_kernel.targetoptions.get("parallel") is True
    assert signatures_before, "the warmed prange dispatcher has no native signature"
    assert numba.threading_layer() in {"omp", "tbb", "workqueue"}
    assert numba.get_num_threads() > 1

    seq_seconds, seq_results = _median_timing(run_sequential)
    par_seconds, par_results = _median_timing(run_parallel)

    assert all(result.backend == "cpu-seq" for result in seq_results)
    assert all(result.backend == "cpu-prange" for result in par_results)
    assert {key: id(kernel) for key, kernel in _SWEEP_KERNEL_CACHE.items()} == sweep_cache_before
    assert {key: id(kernel) for key, kernel in _COMPILED_FN_CACHE.items()} == compiled_cache_before
    assert tuple(prange_kernel.signatures) == signatures_before
    np.testing.assert_allclose(
        seq_results[-1].merged_tavg,
        par_results[-1].merged_tavg,
        rtol=2e-5,
        atol=2e-6,
    )

    iterations = N_SWEEPS * N_STEPS
    seq_throughput = iterations / seq_seconds
    par_throughput = iterations / par_seconds
    speedup = par_throughput / seq_throughput
    assert np.isfinite(speedup) and speedup > 0.0
    print(
        "\n[CPU-prange warmed benchmark] "
        f"threads={numba.get_num_threads()} requested_workers={workers} "
        f"layer={numba.threading_layer()}\n"
        f"  cpu-seq:    {seq_throughput / 1000:.1f} kiter/s "
        f"(median {seq_seconds:.3f}s, n={REPEATS})\n"
        f"  cpu-prange: {par_throughput / 1000:.1f} kiter/s "
        f"(median {par_seconds:.3f}s, n={REPEATS})\n"
        f"  speedup:    {speedup:.2f}x"
    )

    configured_minimum = os.environ.get(MIN_SPEEDUP_ENV)
    if configured_minimum is not None:
        minimum = float(configured_minimum)
        assert minimum > 0.0, f"{MIN_SPEEDUP_ENV} must be positive"
        assert speedup >= minimum, (
            f"warmed CPU-prange speedup {speedup:.2f}x is below the calibrated "
            f"{MIN_SPEEDUP_ENV}={minimum:.2f}x gate"
        )
