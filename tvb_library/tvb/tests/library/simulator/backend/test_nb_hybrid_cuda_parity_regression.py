"""Focused regression coverage for CPU/CUDA hybrid sweep parity."""

import numpy as np
import pytest
import scipy.sparse as sp

from tvb.datatypes import equations, patterns
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.backend.nb_hybrid_cuda_sweep_backend import (
    NbHybridCUDASweepBackend,
    CompiledCUDASweepKernel,
    _CUDA_COMPILED_CACHE,
)
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.hybrid.intra_projection import IntraProjection
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.integrators import EulerDeterministic, EulerStochastic
from tvb.simulator.models.cerebellar_mf import CerebellarMF
from tvb.simulator.models.linear import Linear as LinearModel
from tvb.simulator.noise import Additive
from tvb.tests.library.simulator.backend.test_nb_hybrid_cuda_sweep import (
    _build_g2d_network,
    _build_rsfhn_network,
    _make_8node_connectivity,
)
from tvb.tests.library.simulator.backend.test_nb_hybrid_regression_presigmoidal import (
    INITIAL_STATE as PRESIGMOIDAL_INITIAL_STATE,
    PARAMS as PRESIGMOIDAL_PARAMS,
    _make_network as _make_presigmoidal_network,
)


DT = 0.1
NSTEPS = 7
SWEEP_VALUES = np.array([0.15, 0.75], dtype=np.float32)
CUDA_RESUME_TOTAL = 7
CUDA_RESUME_SPLIT = 3
CUDA_NOISE_SEED = 9173
DYNAMIC_PRESIGMOIDAL_NSTEPS = 12


@pytest.fixture(scope="module", autouse=True)
def require_cudasim():
    from numba import cuda

    if not cuda.is_available():
        pytest.skip("CUDA/CUDASIM is not available")
    _CUDA_COMPILED_CACHE.clear()


def _linear_network(*, stimulus=False):
    model = LinearModel(gamma=np.array([-0.25]))
    model.configure()
    subnet = Subnetwork(
        name="unit",
        model=model,
        scheme=EulerDeterministic(dt=DT),
        nnodes=2,
    )
    subnet.configure()
    intra = IntraProjection(
        source_cvar=np.array([0], dtype=np.int_),
        target_cvar=np.array([0], dtype=np.int_),
        weights=sp.eye(2, format="csr", dtype=np.float64),
        lengths=sp.csr_matrix((2, 2), dtype=np.float64),
        cv=1.0,
        dt=DT,
        scale=1.0,
        cfun=Linear(a=np.array([0.4]), b=np.array([0.0])),
    )
    subnet.projections = [intra]

    if stimulus:
        connectivity = Connectivity(
            centres=np.zeros((2, 3)),
            weights=np.zeros((2, 2)),
            tract_lengths=np.zeros((2, 2)),
            region_labels=np.array(["left", "right"]),
            speed=np.array([1.0]),
        )
        connectivity.configure()
        temporal = equations.Linear()
        temporal.parameters["a"] = 0.35
        temporal.parameters["b"] = 0.2
        pattern = patterns.StimuliRegion(
            temporal=temporal,
            connectivity=connectivity,
            weight=np.array([1.0, 0.25]),
        )
        subnet.add_stimulus(pattern, "x", projection_scale=0.6)

    subnet.configure(simulation_length=NSTEPS * DT)
    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()
    return network


def _initial_state():
    return np.array([[[0.2], [0.6]]], dtype=np.float32)


def _stochastic_linear_network():
    network = _linear_network()
    noise = Additive(nsig=np.array([0.3]))
    noise.noise_seed = CUDA_NOISE_SEED
    noise.random_stream = np.random.RandomState(CUDA_NOISE_SEED)
    noise.configure_white(DT)
    scheme = EulerStochastic(dt=DT, noise=noise)
    scheme.configure_boundaries(network.subnets[0].model)
    network.subnets[0].scheme = scheme
    network.configure()
    return network


def _reset_noise(network, seed=CUDA_NOISE_SEED):
    noise = network.subnets[0].scheme.noise
    noise.noise_seed = seed
    noise.random_stream = np.random.RandomState(seed)


def _compiled_linear(network):
    descriptor, values = NbHybridBackend()._resolve_named_params(
        network, {"unit.intra.b": np.array([0.3], dtype=np.float32)}
    )
    compiled = NbHybridCUDASweepBackend().compile_sweep(
        network, sweep_descriptor=descriptor
    )
    return compiled, values


def _cpu_trajectory_for_b(value):
    network = _linear_network()
    network.subnets[0].projections[0].cfun.b = np.array([value])
    return NbHybridBackend().run_network(
        network,
        nstep=NSTEPS,
        chunk_size=1,
        initial_states=[_initial_state()],
    )[0][1]


def _unified(backend, monitor="tavg", **kwargs):
    return NbHybridBackend().sweep(
        _linear_network(),
        params={"unit.intra.b": SWEEP_VALUES},
        nstep=NSTEPS,
        backend=backend,
        monitor=monitor,
        initial_states=[_initial_state()],
        **kwargs,
    )


def _dynamic_presigmoidal_cuda_and_cpu(global_threshold):
    cpu_network, _ = _make_presigmoidal_network(global_threshold)
    cpu_result = NbHybridBackend().run_network(
        cpu_network,
        nstep=DYNAMIC_PRESIGMOIDAL_NSTEPS,
        chunk_size=1,
        initial_states=[PRESIGMOIDAL_INITIAL_STATE.copy()],
    )[0]

    cuda_network, _ = _make_presigmoidal_network(global_threshold)
    descriptor, values = NbHybridBackend()._resolve_named_params(
        cuda_network,
        {"mpr.intra.H": PRESIGMOIDAL_PARAMS["H"].astype(np.float32)},
    )
    compiled = NbHybridCUDASweepBackend().compile_sweep(
        cuda_network, sweep_descriptor=descriptor
    )
    cuda_result = compiled.run(
        nstep=DYNAMIC_PRESIGMOIDAL_NSTEPS,
        sweep_values=values,
        initial_states=[PRESIGMOIDAL_INITIAL_STATE.copy()],
        monitor_type="raw",
    )
    return cuda_result["raw"][0][0], cuda_result["ctavg"][0][0], cpu_result


def test_direct_cuda_named_linear_b_matches_fresh_cpu_trajectories_and_means():
    network = _linear_network()
    descriptor, values = NbHybridBackend()._resolve_named_params(
        network, {"unit.intra.b": SWEEP_VALUES}
    )
    cuda = NbHybridCUDASweepBackend().run_sweep(
        network,
        sweep_values=values,
        sweep_descriptor=descriptor,
        nstep=NSTEPS,
        monitor_type="raw",
        initial_states=[_initial_state()],
    )
    cpu_trajectories = np.stack(
        [_cpu_trajectory_for_b(value) for value in SWEEP_VALUES]
    )

    assert not np.array_equal(cuda["raw"][0][0], cuda["raw"][0][1])
    np.testing.assert_array_equal(cuda["raw"][0], cpu_trajectories)
    np.testing.assert_array_equal(cuda["tavg"][0], cpu_trajectories.mean(axis=1))


@pytest.mark.parametrize("global_threshold", [False, True], ids=["local", "global"])
def test_cuda_regression_dynamic_presigmoidal_raw_and_ctavg_match_cpu(global_threshold):
    cuda_raw, cuda_ctavg, (_, cpu_raw, cpu_ctavg) = (
        _dynamic_presigmoidal_cuda_and_cpu(global_threshold)
    )

    np.testing.assert_allclose(cuda_raw, cpu_raw, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(
        cuda_ctavg, cpu_ctavg.mean(axis=0), rtol=2e-4, atol=2e-5
    )


def test_cuda_regression_dynamic_presigmoidal_fixture_distinguishes_global_threshold():
    local_raw, local_ctavg, (_, local_cpu_raw, local_cpu_ctavg) = (
        _dynamic_presigmoidal_cuda_and_cpu(False)
    )
    global_raw, global_ctavg, (_, global_cpu_raw, global_cpu_ctavg) = (
        _dynamic_presigmoidal_cuda_and_cpu(True)
    )

    assert np.max(np.abs(local_cpu_raw - global_cpu_raw)) > 1e-4
    assert np.max(np.abs(local_cpu_ctavg - global_cpu_ctavg)) > 1e-4
    assert np.max(np.abs(local_raw - global_raw)) > 1e-4
    assert np.max(np.abs(local_ctavg - global_ctavg)) > 1e-4


def test_cuda_regression_omitted_initial_states_use_subnetwork_zeros_like_cpu():
    network = _linear_network()
    network.subnets[0].projections[0].cfun.b = np.array([0.3])
    compiled, values = _compiled_linear(network)

    cuda = compiled.run(
        nstep=NSTEPS,
        sweep_values=values,
        monitor_type="raw",
    )
    cpu = NbHybridBackend().run_network(
        network,
        nstep=NSTEPS,
        chunk_size=1,
    )[0][1]

    np.testing.assert_array_equal(cuda["raw"][0][0], cpu)


def test_unified_cuda_tavg_has_cpu_sample_shape_times_and_values():
    cpu = _unified("cpu", chunk_size=3)
    cuda = _unified("cuda", chunk_size=3)

    assert cpu.tavg["unit"].shape == (2, 3, 1, 2, 1)
    assert cuda.tavg["unit"].shape == cpu.tavg["unit"].shape
    np.testing.assert_array_equal(cuda.times, cpu.times)
    np.testing.assert_array_equal(cuda.tavg["unit"], cpu.tavg["unit"])


def test_unified_cuda_ctavg_has_cpu_sample_shape_times_and_values():
    cpu = _unified("cpu", chunk_size=3)
    cuda = _unified("cuda", chunk_size=3)

    assert cuda.ctavg["unit"].shape == cpu.ctavg["unit"].shape
    np.testing.assert_array_equal(cuda.times, cpu.times)
    np.testing.assert_allclose(
        cuda.ctavg["unit"], cpu.ctavg["unit"], rtol=2e-6, atol=2e-7
    )


@pytest.mark.parametrize("monitor, period", [("raw", 1), ("subsample", 2)])
def test_unified_cuda_raw_and_subsample_values_and_timing_match_cpu(monitor, period):
    cpu = _unified("cpu", monitor=monitor, monitor_period=period)
    cuda = _unified("cuda", monitor=monitor, monitor_period=period)

    np.testing.assert_array_equal(cuda.raw["unit"], cpu.raw["unit"])
    np.testing.assert_array_equal(cuda.times, cpu.times)


def test_chunked_cuda_stimulus_equals_unchunked():
    network = _linear_network(stimulus=True)
    descriptor, values = NbHybridBackend()._resolve_named_params(
        network, {"unit.intra.b": np.array([0.3], dtype=np.float32)}
    )
    compiled = NbHybridCUDASweepBackend().compile_sweep(
        network, sweep_descriptor=descriptor
    )
    run_kwargs = dict(
        nstep=NSTEPS,
        sweep_values=values,
        initial_states=[_initial_state()],
        monitor_type="raw",
    )
    unchunked = compiled.run(chunk_size=NSTEPS, **run_kwargs)
    chunked = compiled.run(chunk_size=3, **run_kwargs)

    np.testing.assert_array_equal(chunked["raw"][0], unchunked["raw"][0])
    np.testing.assert_array_equal(chunked["tavg"][0], unchunked["tavg"][0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"monitor": "not-a-monitor"},
        {"chunk_size": 0},
        {"chunk_size": 1.5},
        {"monitor": "subsample", "monitor_period": 0},
        {"monitor": "subsample", "monitor_period": 1.5},
    ],
    ids=["monitor", "zero-chunk", "noninteger-chunk", "zero-period", "noninteger-period"],
)
def test_unified_cuda_rejects_invalid_monitor_chunk_and_period(kwargs):
    with pytest.raises(ValueError):
        _unified("cuda", **kwargs)


@pytest.mark.parametrize("backend", ["cdua", "gpu", ""])
def test_unified_sweep_rejects_unknown_backend(backend):
    with pytest.raises(ValueError, match="backend"):
        _unified(backend)


@pytest.mark.parametrize("nstep", [0, -1, 1.5, True])
def test_unified_sweep_rejects_invalid_nstep(nstep):
    with pytest.raises(ValueError, match="nstep"):
        NbHybridBackend().sweep(
            _linear_network(),
            params={"unit.intra.b": SWEEP_VALUES},
            nstep=nstep,
            backend="auto",
            initial_states=[_initial_state()],
        )


def test_auto_falls_back_to_cpu_for_cuda_unsupported_clamps():
    network = _linear_network()
    scheme = network.subnets[0].scheme
    scheme.clamped_state_variable_indices = np.array([0], dtype=np.int32)
    scheme.clamped_state_variable_values = np.array([0.125])

    result = NbHybridBackend().sweep(
        network,
        params={"unit.intra.b": SWEEP_VALUES},
        nstep=2,
        backend="auto",
        initial_states=[_initial_state()],
    )

    assert result.backend == "cpu-seq"


@pytest.mark.parametrize(
    "initial_state",
    [
        np.zeros((2, 2, 1), dtype=np.float32),
        np.zeros((1, 3, 1), dtype=np.float32),
        np.zeros((1, 2, 2), dtype=np.float32),
        np.zeros((2, 1, 3, 1), dtype=np.float32),
    ],
)
def test_low_level_cuda_validates_initial_state_topology(initial_state):
    compiled, values = _compiled_linear(_linear_network())

    with pytest.raises(ValueError, match="initial_states"):
        compiled.run(
            nstep=1,
            sweep_values=values,
            initial_states=[initial_state],
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"monitor_type": "not-a-monitor"},
        {"monitor_type": -1},
        {"monitor_type": 3},
        {"monitor_type": 1.5},
        {"nstep": 0},
        {"nstep": -1},
        {"nstep": 1.5},
        {"chunk_size": 0},
        {"chunk_size": -1},
        {"chunk_size": 1.5},
        {"monitor_type": "subsample", "monitor_period": 0},
        {"monitor_type": "subsample", "monitor_period": -1},
        {"monitor_type": "subsample", "monitor_period": 1.5},
        {"bold_period": 0},
        {"bold_period": -1.0},
    ],
    ids=[
        "unknown-monitor",
        "negative-monitor-number",
        "unknown-monitor-number",
        "noninteger-monitor-number",
        "zero-nstep",
        "negative-nstep",
        "noninteger-nstep",
        "zero-chunk",
        "negative-chunk",
        "noninteger-chunk",
        "zero-monitor-period",
        "negative-monitor-period",
        "noninteger-monitor-period",
        "zero-bold-period",
        "negative-bold-period",
    ],
)
def test_cuda_regression_low_level_run_rejects_invalid_semantic_arguments(kwargs):
    compiled, values = _compiled_linear(_linear_network())
    run_kwargs = {"nstep": 2, "sweep_values": values}
    run_kwargs.update(kwargs)

    with pytest.raises(ValueError):
        compiled.run(**run_kwargs)


def test_cuda_heterogeneous_subnet_merge_returns_none():
    weights, lengths = _make_8node_connectivity()
    subnetworks = NbHybridBackend()._analyse(
        _build_g2d_network(8, weights, lengths)
    ).subnetworks
    one_voi = np.zeros((2, 1, 8, 1), dtype=np.float32)
    two_vois = np.zeros((2, 2, 8, 1), dtype=np.float32)

    merged = CompiledCUDASweepKernel._merge_subnet_outputs(
        [one_voi, two_vois], subnetworks * 2
    )

    assert merged is None


def test_cuda_regression_raw_resume_from_nonzero_step_offset_matches_full_tail():
    network = _linear_network(stimulus=True)
    compiled, values = _compiled_linear(network)
    common = dict(
        sweep_values=values,
        initial_states=[_initial_state()],
        monitor_type="raw",
    )

    full = compiled.run(nstep=CUDA_RESUME_TOTAL, **common)
    first = compiled.run(nstep=CUDA_RESUME_SPLIT, **common)
    assert first["snapshot"]["step_offset"] == CUDA_RESUME_SPLIT
    resumed = compiled.run(
        nstep=CUDA_RESUME_TOTAL - CUDA_RESUME_SPLIT,
        sweep_values=values,
        snapshot=first["snapshot"],
        monitor_type="raw",
    )

    np.testing.assert_array_equal(
        resumed["raw"][0], full["raw"][0][:, CUDA_RESUME_SPLIT:]
    )
    assert resumed["snapshot"]["step_offset"] == CUDA_RESUME_TOTAL
    np.testing.assert_array_equal(
        resumed["snapshot"]["states"]["unit"],
        full["snapshot"]["states"]["unit"],
    )
    np.testing.assert_array_equal(
        resumed["snapshot"]["srcbufs"]["unit"],
        full["snapshot"]["srcbufs"]["unit"],
    )


def test_cuda_regression_stochastic_chunked_raw_matches_unchunked_after_rng_reset():
    network = _stochastic_linear_network()
    compiled, values = _compiled_linear(network)
    common = dict(
        nstep=CUDA_RESUME_TOTAL,
        sweep_values=values,
        initial_states=[_initial_state()],
        monitor_type="raw",
    )

    _reset_noise(network)
    unchunked = compiled.run(chunk_size=CUDA_RESUME_TOTAL, **common)
    _reset_noise(network)
    chunked = compiled.run(chunk_size=3, **common)

    np.testing.assert_array_equal(chunked["raw"][0], unchunked["raw"][0])
    np.testing.assert_array_equal(
        chunked["snapshot"]["states"]["unit"],
        unchunked["snapshot"]["states"]["unit"],
    )


def test_cuda_regression_stochastic_snapshot_owns_rng_state_for_resume():
    network = _stochastic_linear_network()
    compiled, values = _compiled_linear(network)
    common = dict(
        sweep_values=values,
        initial_states=[_initial_state()],
        monitor_type="raw",
    )

    _reset_noise(network)
    full = compiled.run(nstep=CUDA_RESUME_TOTAL, **common)
    _reset_noise(network)
    first = compiled.run(nstep=CUDA_RESUME_SPLIT, **common)
    assert "rng_states" in first["snapshot"], (
        "a stochastic CUDA snapshot must contain its RNG state"
    )

    network.subnets[0].scheme.noise.random_stream.randn(1000)
    resumed = compiled.run(
        nstep=CUDA_RESUME_TOTAL - CUDA_RESUME_SPLIT,
        sweep_values=values,
        snapshot=first["snapshot"],
        monitor_type="raw",
    )

    np.testing.assert_array_equal(
        resumed["raw"][0], full["raw"][0][:, CUDA_RESUME_SPLIT:]
    )
    np.testing.assert_array_equal(
        resumed["snapshot"]["states"]["unit"],
        full["snapshot"]["states"]["unit"],
    )


def test_cuda_regression_multimode_ctavg_preserves_modes_and_matches_cpu():
    weights, lengths = _make_8node_connectivity()
    network = _build_rsfhn_network(weights, lengths)
    network.projections[0].mode_map = np.eye(3, dtype=np.int64)
    network.configure()
    rng = np.random.RandomState(88)
    initial_states = [
        rng.uniform(0.01, 0.2, (4, 8, 3)).astype(np.float32),
        rng.uniform(0.01, 0.2, (4, 8, 3)).astype(np.float32),
    ]
    nstep = 3

    cuda = NbHybridCUDASweepBackend().run_sweep(
        network,
        sweep_values=np.array([[0.5]], dtype=np.float32),
        nstep=nstep,
        initial_states=initial_states,
    )
    cpu = NbHybridBackend().run_network(
        network,
        nstep=nstep,
        chunk_size=1,
        initial_states=initial_states,
    )
    cuda_ctavg = cuda["ctavg"][1][0]
    cpu_ctavg = cpu[1][2].mean(axis=0).astype(np.float32)

    assert np.all(np.any(cpu_ctavg != 0.0, axis=(0, 1)))
    assert not np.array_equal(cpu_ctavg[..., 0], cpu_ctavg[..., 1])
    assert not np.array_equal(cpu_ctavg[..., 1], cpu_ctavg[..., 2])
    assert np.all(np.any(cuda_ctavg != 0.0, axis=(0, 1)))
    assert not np.array_equal(cuda_ctavg[..., 0], cuda_ctavg[..., 1])
    assert not np.array_equal(cuda_ctavg[..., 1], cuda_ctavg[..., 2])
    np.testing.assert_allclose(cuda_ctavg, cpu_ctavg, rtol=2e-5, atol=2e-5)


def test_cuda_regression_compile_rejects_nonzerlaut_custom_template_model():
    model = CerebellarMF()
    model.configure()
    subnet = Subnetwork(
        name="cerebellar", model=model, scheme=EulerDeterministic(dt=DT), nnodes=2
    )
    subnet.projections = []
    subnet.configure()
    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()

    with pytest.raises(NotImplementedError, match="(?i)(custom|CerebellarMF)"):
        NbHybridCUDASweepBackend().compile_sweep(network)


def test_cuda_regression_compile_rejects_explicit_integrator_state_clamps():
    network = _linear_network()
    scheme = network.subnets[0].scheme
    scheme.clamped_state_variable_indices = np.array([0], dtype=np.int32)
    scheme.clamped_state_variable_values = np.array([0.125])

    with pytest.raises(NotImplementedError, match="(?i)clamp"):
        NbHybridCUDASweepBackend().compile_sweep(network)


@pytest.mark.parametrize(
    "descriptor",
    [
        {"type": "cfun", "projection": "missing", "param_idx": 0},
        {"type": "cfun", "projection": "intra", "param_idx": 99},
        {"type": "model", "subnet": "missing", "param": "gamma"},
        {"type": "model", "subnet": "unit", "param": "missing"},
    ],
    ids=[
        "unknown-projection",
        "unknown-cfun-parameter",
        "unknown-subnet",
        "unknown-model-parameter",
    ],
)
def test_cuda_regression_compile_validates_descriptor_references_before_render(
    monkeypatch, descriptor
):
    backend = NbHybridCUDASweepBackend()

    def fail_render(*args, **kwargs):
        pytest.fail("compile_sweep rendered an invalid sweep descriptor")

    monkeypatch.setattr(backend, "render_template", fail_render)
    with pytest.raises(ValueError):
        backend.compile_sweep(_linear_network(), sweep_descriptor=[descriptor])
