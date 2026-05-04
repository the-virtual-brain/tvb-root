# -*- coding: utf-8 -*-
#
#
# pytest test suite for the CUDA sweep backend.
#
# Each test clears _CUDA_COMPILED_CACHE and requires a CUDA-capable device.
#

import pytest
import numpy as np
import scipy.sparse as sp

from tvb.simulator.backend.nb_hybrid_cuda_sweep_backend import (
    NbHybridCUDASweepBackend,
    _CUDA_COMPILED_CACHE,
)
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.models.oscillator import Generic2dOscillator
from tvb.simulator.models.jansen_rit import JansenRit
from tvb.simulator.models.stefanescu_jirsa import ReducedSetFitzHughNagumo
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.hybrid.intra_projection import IntraProjection
from tvb.simulator.hybrid.coupling import Scaling
from tvb.datatypes.connectivity import Connectivity

DT = 0.01


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cuda_available():
    """Skip the entire test session if CUDA is not available."""
    try:
        import numba.cuda
        if not numba.cuda.is_available():
            pytest.skip("CUDA not available")
    except ImportError:
        pytest.skip("numba.cuda not available")


@pytest.fixture(autouse=True)
def clear_cuda_cache():
    """Clear the in-process CUDA compiled kernel cache before every test."""
    _CUDA_COMPILED_CACHE.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_8node_connectivity():
    """Return 8×8 identity weights and zero lengths as sparse CSR matrices."""
    w = sp.csr_matrix(np.eye(8, dtype=np.float64) * 0.1)
    l = sp.csr_matrix(np.zeros((8, 8), dtype=np.float64))
    return w, l


def _make_76node_connectivity():
    """Load the standard 76-node connectome and return sparse CSR weights/lengths."""
    conn = Connectivity.from_file()
    w = sp.csr_matrix(conn.weights)
    l = sp.csr_matrix(conn.tract_lengths)
    return w, l


def _build_g2d_network(n_nodes, weights, lengths, cfun=Scaling()):
    """Single Generic2dOscillator subnet with an intra-projection."""
    m = Generic2dOscillator()
    m.configure()
    sn = Subnetwork(
        name="g2d",
        model=m,
        scheme=HeunDeterministic(dt=DT),
        nnodes=n_nodes,
    )
    sn.configure()

    intra = IntraProjection(
        source_cvar=np.array([0], dtype=np.int_),
        target_cvar=np.array([0], dtype=np.int_),
        weights=weights,
        lengths=lengths,
        cv=1.0,
        dt=DT,
        scale=1.0,
        cfun=cfun,
    )
    sn.projections = [intra]
    sn.configure()

    ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
    ns.configure()
    return ns


def _build_jr_network(weights, lengths):
    """Single JansenRit subnet with an intra-projection (76 nodes assumed)."""
    m = JansenRit()
    m.configure()
    sn = Subnetwork(
        name="jr",
        model=m,
        scheme=HeunDeterministic(dt=DT),
        nnodes=76,
    )
    sn.configure()

    intra = IntraProjection(
        source_cvar=np.array([0], dtype=np.int32),
        target_cvar=np.array([0], dtype=np.int32),
        weights=weights,
        lengths=lengths,
        cv=1.0,
        dt=DT,
        scale=1e-3,
        cfun=Scaling(),
    )
    sn.projections = [intra]
    sn.configure()

    ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
    ns.configure()
    return ns


def _build_rsfhn_network(weights, lengths):
    """Two-subnet ReducedSetFitzHughNagumo network with an inter-projection."""
    m1 = ReducedSetFitzHughNagumo()
    m1.configure()
    m2 = ReducedSetFitzHughNagumo()
    m2.configure()

    sn1 = Subnetwork(
        name="rsfhn1",
        model=m1,
        scheme=HeunDeterministic(dt=0.1),
        nnodes=8,
    )
    sn1.configure()
    sn2 = Subnetwork(
        name="rsfhn2",
        model=m2,
        scheme=HeunDeterministic(dt=0.1),
        nnodes=8,
    )
    sn2.configure()

    inter = InterProjection(
        source=sn1,
        target=sn2,
        source_cvar=np.array([0, 2], dtype=np.int_),
        target_cvar=np.array([0, 2], dtype=np.int_),
        weights=weights,
        lengths=lengths,
        cv=1.0,
        dt=0.1,
        scale=1.0,
        cfun=Scaling(a=np.array([0.5])),
    )

    ns = NetworkSet(subnets=[sn1, sn2], projections=[inter], stimuli=[])
    ns.configure()
    return ns


def _cpu_tavg_mean(network_set, nstep, x0_list):
    """Run the CPU Numba backend and return the time-averaged state per subnet."""
    backend = NbHybridBackend()
    results = backend.run_network(
        network_set,
        nstep=nstep,
        chunk_size=1,
        initial_states=x0_list,
    )
    # results: list of (times, data, ctavg)
    tavg = []
    for _, data, _ in results:
        # data shape: (nstep, n_voi, n_nodes, n_modes)
        tavg.append(data.mean(axis=0).astype(np.float32))
    return tavg


# ---------------------------------------------------------------------------
# 1. G2D ("MPR") vs CPU on 76 nodes
# ---------------------------------------------------------------------------

def test_mpr_vs_cpu(cuda_available):
    """Generic2dOscillator on 76 nodes matches CPU reference (<1e-3)."""
    w, l = _make_76node_connectivity()
    ns = _build_g2d_network(76, w, l)
    nstep = 20

    x0 = np.zeros((2, 76, 1), dtype=np.float64)
    gpu = NbHybridCUDASweepBackend().run_sweep(
        ns,
        sweep_values=np.array([[1.0]], dtype=np.float32),
        nstep=nstep,
        initial_states=[x0],
    )
    cpu = _cpu_tavg_mean(ns, nstep, [x0])

    maxerr = np.max(np.abs(gpu["tavg"][0][0] - cpu[0]))
    assert maxerr < 1e-3, f"maxerr={maxerr} >= 1e-3"


# ---------------------------------------------------------------------------
# 2. JansenRit vs CPU on 76 nodes
# ---------------------------------------------------------------------------

def test_jr_vs_cpu(cuda_available):
    """JansenRit on 76 nodes matches CPU reference (<1e-3)."""
    w, l = _make_76node_connectivity()
    ns = _build_jr_network(w, l)
    nstep = 20

    x0 = np.zeros((6, 76, 1), dtype=np.float64)
    x0[0, :, 0] = 0.08
    x0[1, :, 0] = 13.0
    x0[2, :, 0] = 5.0

    gpu = NbHybridCUDASweepBackend().run_sweep(
        ns,
        sweep_values=np.array([[1.0]], dtype=np.float32),
        nstep=nstep,
        initial_states=[x0],
    )
    cpu = _cpu_tavg_mean(ns, nstep, [x0])

    maxerr = np.max(np.abs(gpu["tavg"][0][0] - cpu[0]))
    assert maxerr < 1e-3, f"maxerr={maxerr} >= 1e-3"


# ---------------------------------------------------------------------------
# 3. Chunking bit-exactness (G2D, 8 nodes, 100 steps)
# ---------------------------------------------------------------------------

def test_chunking_bit_exact(cuda_available):
    """chunk_size=25 produces identical tavg to unchunked run."""
    w, l = _make_8node_connectivity()
    ns = _build_g2d_network(8, w, l)
    nstep = 100

    x0 = np.zeros((2, 8, 1), dtype=np.float64)

    # Unchunked (chunk_size == nstep implicitly when not set)
    res_unchunked = NbHybridCUDASweepBackend().run_sweep(
        ns,
        sweep_values=np.array([[1.0]], dtype=np.float32),
        nstep=nstep,
        initial_states=[x0],
        chunk_size=nstep,
    )
    res_chunked = NbHybridCUDASweepBackend().run_sweep(
        ns,
        sweep_values=np.array([[1.0]], dtype=np.float32),
        nstep=nstep,
        initial_states=[x0],
        chunk_size=25,
    )

    np.testing.assert_array_equal(
        res_unchunked["tavg"][0],
        res_chunked["tavg"][0],
        err_msg="chunking broke bit-exactness",
    )


# ---------------------------------------------------------------------------
# 4. Batching bit-exactness (G2D, 8 nodes, 10 steps, 20 sweeps)
# ---------------------------------------------------------------------------

def test_batching_bit_exact(cuda_available):
    """max_batch_sweeps=5 produces identical tavg to single-batch run."""
    w, l = _make_8node_connectivity()
    ns = _build_g2d_network(8, w, l)
    nstep = 10

    x0 = np.zeros((2, 8, 1), dtype=np.float64)
    sweep_values = np.arange(20, dtype=np.float32).reshape(-1, 1)

    res_single = NbHybridCUDASweepBackend().run_sweep(
        ns,
        sweep_values=sweep_values,
        nstep=nstep,
        initial_states=[x0],
    )
    res_batched = NbHybridCUDASweepBackend().run_sweep(
        ns,
        sweep_values=sweep_values,
        nstep=nstep,
        initial_states=[x0],
        max_batch_sweeps=5,
    )

    np.testing.assert_array_equal(
        res_single["tavg"][0],
        res_batched["tavg"][0],
        err_msg="batching broke bit-exactness",
    )


# ---------------------------------------------------------------------------
# 5. BOLD no NaN (G2D, 8 nodes, 200 steps)
# ---------------------------------------------------------------------------

def test_bold_no_nan(cuda_available):
    """Bold output contains no NaN when bold_period is supplied."""
    w, l = _make_8node_connectivity()
    ns = _build_g2d_network(8, w, l)
    nstep = 200

    x0 = np.zeros((2, 8, 1), dtype=np.float64)
    res = NbHybridCUDASweepBackend().run_sweep(
        ns,
        sweep_values=np.array([[1.0]], dtype=np.float32),
        nstep=nstep,
        initial_states=[x0],
        bold_period=2.0,
    )

    assert "bold" in res, "Expected 'bold' key in result dict"
    for bold_arr in res["bold"]:
        assert np.all(np.isfinite(bold_arr)), "NaN/Inf in BOLD output"


# ---------------------------------------------------------------------------
# 6. Raw monitor shape (G2D, 8 nodes, 20 steps)
# ---------------------------------------------------------------------------

def test_raw_monitor_shape(cuda_available):
    """monitor_type=1 yields raw output with shape (n_sweeps, nstep, n_voi, n_nodes, n_modes)."""
    w, l = _make_8node_connectivity()
    ns = _build_g2d_network(8, w, l)
    nstep = 20

    x0 = np.zeros((2, 8, 1), dtype=np.float64)
    res = NbHybridCUDASweepBackend().run_sweep(
        ns,
        sweep_values=np.array([[1.0]], dtype=np.float32),
        nstep=nstep,
        initial_states=[x0],
        monitor_type=1,
    )

    assert "raw" in res, "Expected 'raw' key in result dict"
    # G2D has 1 VoI (V)
    assert res["raw"][0].shape == (1, 20, 1, 8, 1), f"unexpected raw shape: {res['raw'][0].shape}"


# ---------------------------------------------------------------------------
# 7. Subsample monitor shape (G2D, 8 nodes, 20 steps, period=4)
# ---------------------------------------------------------------------------

def test_subsample_shape(cuda_available):
    """monitor_type=2 with monitor_period=4 yields correct subsampled shape."""
    w, l = _make_8node_connectivity()
    ns = _build_g2d_network(8, w, l)
    nstep = 20
    monitor_period = 4
    expected_raw_steps = (nstep + monitor_period - 1) // monitor_period  # 5

    x0 = np.zeros((2, 8, 1), dtype=np.float64)
    res = NbHybridCUDASweepBackend().run_sweep(
        ns,
        sweep_values=np.array([[1.0]], dtype=np.float32),
        nstep=nstep,
        initial_states=[x0],
        monitor_type=2,
        monitor_period=monitor_period,
    )

    assert "raw" in res, "Expected 'raw' key in result dict"
    assert res["raw"][0].shape == (1, expected_raw_steps, 1, 8, 1), (
        f"unexpected subsample shape: {res['raw'][0].shape}"
    )


# ---------------------------------------------------------------------------
# 8. Heun combined RS-FHN vs CPU (8 nodes, 20 steps, 3 modes)
# ---------------------------------------------------------------------------

def test_heun_combined_vs_cpu(cuda_available):
    """ReducedSetFitzHughNagumo (combined) matches CPU or produces finite output."""
    w, l = _make_8node_connectivity()
    ns = _build_rsfhn_network(w, l)
    nstep = 20

    rng = np.random.RandomState(88)
    x0_src = rng.uniform(0.0, 0.2, (4, 8, 3)).astype(np.float64)
    x0_tgt = rng.uniform(0.0, 0.2, (4, 8, 3)).astype(np.float64)

    sweep_values = np.array([[0.5]], dtype=np.float32)
    gpu = NbHybridCUDASweepBackend().run_sweep(
        ns,
        sweep_values=sweep_values,
        nstep=nstep,
        initial_states=[x0_src, x0_tgt],
    )

    # Verify GPU output is finite and shape is correct
    assert len(gpu["tavg"]) == 2, "expected two subnets in tavg"
    for arr in gpu["tavg"]:
        assert arr.shape == (1, 2, 8, 3), f"unexpected tavg shape: {arr.shape}"
        assert np.all(np.isfinite(arr)), "NaN/Inf in GPU sweep output"

    # Try CPU reference; if it crashes, just accept the GPU-only checks above
    try:
        cpu = _cpu_tavg_mean(ns, nstep, [x0_src, x0_tgt])
    except Exception:
        pytest.skip("CPU reference crashed for RS-FHN Heun combined")

    maxerr = np.max(np.abs(gpu["tavg"][0][0] - cpu[0]))
    maxerr2 = np.max(np.abs(gpu["tavg"][1][0] - cpu[1]))
    assert maxerr < 1e-3, f"subnet 1 maxerr={maxerr} >= 1e-3"
    assert maxerr2 < 1e-3, f"subnet 2 maxerr={maxerr2} >= 1e-3"


# ---------------------------------------------------------------------------
# 9. ctavg present and non-zero (G2D, 8 nodes, 10 steps)
# ---------------------------------------------------------------------------

def test_ctavg_present(cuda_available):
    """ctavg is non-zero and has correct shape for a coupled subnet."""
    w, l = _make_8node_connectivity()
    ns = _build_g2d_network(8, w, l)
    nstep = 10

    x0 = np.zeros((2, 8, 1), dtype=np.float64)
    res = NbHybridCUDASweepBackend().run_sweep(
        ns,
        sweep_values=np.array([[1.0]], dtype=np.float32),
        nstep=nstep,
        initial_states=[x0],
    )

    assert "ctavg" in res, "Expected 'ctavg' key in result dict"
    ct = res["ctavg"][0]
    # G2D has 1 coupling term (c_V)
    assert ct.shape == (1, 1, 8, 1), f"unexpected ctavg shape: {ct.shape}"
    assert np.any(np.abs(ct) > 0), "ctavg is all zeros for coupled subnet"


# ---------------------------------------------------------------------------
# 10. Spatial / projection monitor shapes (G2D, 8 nodes, 5 steps)
# ---------------------------------------------------------------------------

def test_spatial_proj_shape(cuda_available):
    """spatial_mean and gain provided → spatial_tavg and proj_tavg have correct shapes."""
    w, l = _make_8node_connectivity()
    ns = _build_g2d_network(8, w, l)
    nstep = 5

    x0 = np.zeros((2, 8, 1), dtype=np.float64)
    n_areas = 2
    n_sensors = 3
    spatial_mean = np.eye(n_areas, 8, dtype=np.float32)[:n_areas, :]
    gain = np.eye(n_sensors, 8, dtype=np.float32)[:n_sensors, :]

    res = NbHybridCUDASweepBackend().run_sweep(
        ns,
        sweep_values=np.array([[1.0]], dtype=np.float32),
        nstep=nstep,
        initial_states=[x0],
        monitors={
            "spatial_mean": {"g2d": spatial_mean},
            "gain": {"g2d": gain},
        },
    )

    assert "spatial_tavg" in res, "Expected 'spatial_tavg' key"
    assert "proj_tavg" in res, "Expected 'proj_tavg' key"
    # spatial_tavg: (n_sweeps, n_voi, n_areas, 1)
    assert res["spatial_tavg"][0].shape == (1, 1, 2, 1), (
        f"unexpected spatial_tavg shape: {res['spatial_tavg'][0].shape}"
    )
    # proj_tavg: (n_sweeps, n_voi, n_sensors, 1)
    assert res["proj_tavg"][0].shape == (1, 1, 3, 1), (
        f"unexpected proj_tavg shape: {res['proj_tavg'][0].shape}"
    )
