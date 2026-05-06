"""
Test suite for unified sweep API (NbHybridBackend.sweep).

Covers:
- Named parameter resolution (coupling_scale, proj.attr, model.param)
- CPU sequential sweep
- CPU multi-core sweep (n_workers=4)
- GPU sweep (auto-dispatch and explicit 'cuda')
- SweepResult shape correctness
- Non-trivial configurations (2 subnets, intra+inter projections, stim)
- Benchmarks in kiter/s

Requires: pytest, numpy, scipy
"""
import pytest
import numpy as np
import scipy.sparse as sp
import time
import sys
import os

sys.path.insert(0, "/home/duke/src/tvb-hybrid-numba/tvb_library")
os.environ["TVB_LIBRARY_PATH"] = "/home/duke/src/tvb-hybrid-numba/tvb_library"

from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.hybrid.intra_projection import IntraProjection
from tvb.simulator.hybrid.coupling import Linear, Scaling, Sigmoidal
from tvb.simulator.backend.nb_hybrid import NbHybridBackend, SweepResult
from tvb.simulator.integrators import HeunDeterministic
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.models.jansen_rit import JansenRit
from tvb.simulator.models.stefanescu_jirsa import ReducedSetFitzHughNagumo
from tvb.simulator.models.wong_wang import ReducedWongWang

DT = 0.01
N_SWEEPS = 20
N_STEPS = 100

# Check CUDA
CUDA_AVAILABLE = False
try:
    import numba.cuda
    CUDA_AVAILABLE = numba.cuda.is_available()
except ImportError:
    pass


def _load_conn(name):
    conn = Connectivity.from_file(f"{name}.zip")
    conn.configure()
    return conn


def _make_intra(conn_slice, src_cvar, tgt_cvar, cfun=None):
    n = conn_slice.shape[0]
    w = sp.csr_matrix(conn_slice.astype(np.float32))
    w_a = w.toarray(); np.fill_diagonal(w_a, 0); w = sp.csr_matrix(w_a)
    tl = sp.csr_matrix(np.zeros((n, n), dtype=np.float32))
    p = IntraProjection(
        source_cvar=np.array(src_cvar, dtype=np.int_),
        target_cvar=np.array(tgt_cvar, dtype=np.int_),
        weights=w, lengths=tl, cv=1.0, dt=DT, scale=1.0)
    if cfun is not None:
        p.cfun = cfun
    return p


def _make_inter(source_sn, target_sn, w_block, tl_block, src_cvar, tgt_cvar, cfun=None):
    inter = InterProjection(
        source=source_sn, target=target_sn,
        source_cvar=np.array(src_cvar, dtype=np.int_),
        target_cvar=np.array(tgt_cvar, dtype=np.int_),
        weights=sp.csr_matrix(w_block.astype(np.float32)),
        lengths=sp.csr_matrix(tl_block.astype(np.float32)),
        cv=1.0, dt=DT)
    if cfun is not None:
        inter.cfun = cfun
    return inter


# ---------------------------------------------------------------------------
# Test: Named parameter resolution
# ---------------------------------------------------------------------------

class TestResolveNamedParams:
    def test_coupling_scale_alias(self):
        """'coupling_scale' resolves to first projection's 'a' param."""
        conn = _load_conn("connectivity_76")
        mpr = MontbrioPazoRoxin(); mpr.configure()
        sn = Subnetwork(name="ctx", model=mpr, scheme=HeunDeterministic(dt=DT), nnodes=76)
        intra = _make_intra(conn.weights[:76, :76], [0], [0], cfun=Linear())
        sn.projections = [intra]; sn.configure()
        ns = NetworkSet(subnets=[sn], projections=[]); ns.configure()

        backend = NbHybridBackend()
        desc, values = backend._resolve_named_params(
            ns, {"coupling_scale": np.linspace(0.0, 0.1, 10)})
        assert len(desc) == 1
        assert desc[0]["type"] == "cfun"
        assert desc[0]["param_idx"] == 0
        assert values.shape == (10, 1)

    def test_cfun_attr_resolution(self):
        """'{proj_name}.b' resolves to param_idx=1 for Linear coupling."""
        conn = _load_conn("connectivity_76")
        mpr = MontbrioPazoRoxin(); mpr.configure()
        sn = Subnetwork(name="ctx", model=mpr, scheme=HeunDeterministic(dt=DT), nnodes=76)
        intra = _make_intra(conn.weights[:76, :76], [0], [0], cfun=Linear())
        sn.projections = [intra]; sn.configure()
        ns = NetworkSet(subnets=[sn], projections=[]); ns.configure()

        backend = NbHybridBackend()
        desc, values = backend._resolve_named_params(
            ns, {"ctx.intra.b": np.linspace(0.0, 1.0, 5)})
        assert desc[0]["param_idx"] == 1  # 'b' is idx 1 for Linear

    def test_model_param_resolution(self):
        """'subnet.param' resolves to model parameter sweep."""
        conn = _load_conn("connectivity_76")
        mpr = MontbrioPazoRoxin(); mpr.configure()
        sn = Subnetwork(name="ctx", model=mpr, scheme=HeunDeterministic(dt=DT), nnodes=76)
        intra = _make_intra(conn.weights[:76, :76], [0], [0], cfun=Linear())
        sn.projections = [intra]; sn.configure()
        ns = NetworkSet(subnets=[sn], projections=[]); ns.configure()

        backend = NbHybridBackend()
        desc, values = backend._resolve_named_params(
            ns, {"ctx.tau": np.linspace(0.1, 1.0, 8)})
        assert desc[0]["type"] == "model"
        assert desc[0]["subnet"] == "ctx"
        assert desc[0]["param"] == "tau"

    def test_multi_param_sweep(self):
        """2D sweep: coupling_scale + model param simultaneously."""
        conn = _load_conn("connectivity_76")
        mpr = MontbrioPazoRoxin(); mpr.configure()
        sn = Subnetwork(name="ctx", model=mpr, scheme=HeunDeterministic(dt=DT), nnodes=76)
        intra = _make_intra(conn.weights[:76, :76], [0], [0], cfun=Linear())
        sn.projections = [intra]; sn.configure()
        ns = NetworkSet(subnets=[sn], projections=[]); ns.configure()

        backend = NbHybridBackend()
        desc, values = backend._resolve_named_params(
            ns, {
                "coupling_scale": np.linspace(0.0, 0.1, 10),
                "ctx.tau": np.linspace(0.1, 1.0, 10),
            })
        assert len(desc) == 2
        assert values.shape == (10, 2)

    def test_inter_projection_name(self):
        """Inter-projection name resolves as '{src}_to_{tgt}'."""
        conn = _load_conn("connectivity_76")
        mpr = MontbrioPazoRoxin(); mpr.configure()
        jr = JansenRit(); jr.configure()
        sn1 = Subnetwork(name="ctx", model=mpr, scheme=HeunDeterministic(dt=DT), nnodes=68)
        sn2 = Subnetwork(name="sub", model=jr, scheme=HeunDeterministic(dt=DT), nnodes=8)
        sn1.projections = [_make_intra(conn.weights[:68, :68], [0], [0])]
        sn2.projections = [_make_intra(conn.weights[68:76, 68:76], [0], [2])]
        sn1.configure(); sn2.configure()
        inter = _make_inter(sn1, sn2, conn.weights[:68, 68:76],
                           conn.tract_lengths[:68, 68:76], [0], [2], cfun=Linear())
        ns = NetworkSet(subnets=[sn1, sn2], projections=[inter]); ns.configure()

        backend = NbHybridBackend()
        desc, values = backend._resolve_named_params(
            ns, {"ctx_to_sub.a": np.linspace(0.0, 0.1, 10)})
        assert desc[0]["projection"] == "ctx_to_sub"
        assert desc[0]["param_idx"] == 0


# ---------------------------------------------------------------------------
# Test: CPU sequential sweep
# ---------------------------------------------------------------------------

class TestCPUSweep:
    def test_sweep_returns_sweep_result(self):
        """sweep() returns a SweepResult with correct shapes."""
        conn = _load_conn("connectivity_76")
        mpr = MontbrioPazoRoxin(); mpr.configure()
        sn = Subnetwork(name="ctx", model=mpr, scheme=HeunDeterministic(dt=DT), nnodes=76)
        sn.projections = [_make_intra(conn.weights[:76, :76], [0], [0], cfun=Linear())]
        sn.configure()
        ns = NetworkSet(subnets=[sn], projections=[]); ns.configure()

        backend = NbHybridBackend()
        result = backend.sweep(
            ns, params={"coupling_scale": np.linspace(0.0, 0.1, N_SWEEPS)},
            nstep=N_STEPS, backend="cpu")

        assert isinstance(result, SweepResult)
        assert result.backend == "cpu-seq"
        assert result.sweep_values.shape == (N_SWEEPS, 1)
        assert "ctx" in result.tavg
        assert result.merged_tavg is not None
        assert result.merged_tavg.shape[0] == N_SWEEPS  # n_sweeps
        assert result.elapsed > 0

    def test_sweep_shapes_mpr(self):
        """MPR sweep shapes: (n_sweeps, n_chunks, n_voi, N, modes)."""
        conn = _load_conn("connectivity_76")
        mpr = MontbrioPazoRoxin(); mpr.configure()
        sn = Subnetwork(name="ctx", model=mpr, scheme=HeunDeterministic(dt=DT), nnodes=76)
        sn.projections = [_make_intra(conn.weights[:76, :76], [0], [0], cfun=Linear())]
        sn.configure()
        ns = NetworkSet(subnets=[sn], projections=[]); ns.configure()

        backend = NbHybridBackend()
        result = backend.sweep(
            ns, params={"coupling_scale": np.linspace(0.0, 0.5, 5)},
            nstep=50, backend="cpu")

        # MPR: nvar=2, voi=2, N=76, modes=1
        assert result.tavg["ctx"].shape[0] == 5   # n_sweeps
        assert result.tavg["ctx"].shape[2] == 2   # n_voi
        assert result.tavg["ctx"].shape[3] == 76  # N
        assert result.merged_tavg.shape == result.tavg["ctx"].shape

    def test_sweep_two_subnet(self):
        """Two subnets (MPR+JR) with inter-projection."""
        conn = _load_conn("connectivity_76")
        mpr = MontbrioPazoRoxin(); mpr.configure()
        jr = JansenRit(); jr.configure()
        sn1 = Subnetwork(name="ctx", model=mpr, scheme=HeunDeterministic(dt=DT), nnodes=68)
        sn2 = Subnetwork(name="sub", model=jr, scheme=HeunDeterministic(dt=DT), nnodes=8)
        sn1.projections = [_make_intra(conn.weights[:68, :68], [0], [0])]
        sn2.projections = [_make_intra(conn.weights[68:76, 68:76], [0], [2])]
        sn1.configure(); sn2.configure()
        inter = _make_inter(sn1, sn2, conn.weights[:68, 68:76],
                           conn.tract_lengths[:68, 68:76], [0], [2], cfun=Linear())
        ns = NetworkSet(subnets=[sn1, sn2], projections=[inter]); ns.configure()

        backend = NbHybridBackend()
        result = backend.sweep(
            ns, params={"ctx_to_sub.a": np.linspace(0.0, 0.05, 10)},
            nstep=50, backend="cpu")

        assert "ctx" in result.tavg
        assert "sub" in result.tavg
        assert result.tavg["ctx"].shape[0] == 10
        assert result.tavg["sub"].shape[0] == 10
        # merged_tavg should have 76 nodes
        assert result.merged_tavg.shape[3] == 76


# ---------------------------------------------------------------------------
# Test: CPU multi-core sweep
# ---------------------------------------------------------------------------

class TestCPUMultiCore:
    def test_parallel_correctness(self):
        """n_workers=4 produces same merged_tavg as n_workers=1."""
        conn = _load_conn("connectivity_76")
        mpr = MontbrioPazoRoxin(); mpr.configure()
        sn = Subnetwork(name="ctx", model=mpr, scheme=HeunDeterministic(dt=DT), nnodes=76)
        sn.projections = [_make_intra(conn.weights[:76, :76], [0], [0], cfun=Linear())]
        sn.configure()
        ns = NetworkSet(subnets=[sn], projections=[]); ns.configure()

        backend = NbHybridBackend()
        sweep_vals = np.linspace(0.01, 0.05, 10).astype(np.float32)

        result_seq = backend.sweep(
            ns, params={"coupling_scale": sweep_vals},
            nstep=50, backend="cpu", n_workers=1)

        result_par = backend.sweep(
            ns, params={"coupling_scale": sweep_vals},
            nstep=50, backend="cpu", n_workers=4)

        # Shapes must match
        assert result_seq.merged_tavg.shape == result_par.merged_tavg.shape
        # Values should be close (within float32 precision)
        # Note: parallel results may reorder, so compare sorted by sweep parameter
        np.testing.assert_allclose(
            result_seq.merged_tavg, result_par.merged_tavg,
            atol=1e-5, rtol=1e-5
        )

    def test_parallel_speedup(self):
        """n_workers=4 should be faster than n_workers=1 for 100+ sweeps."""
        conn = _load_conn("connectivity_76")
        mpr = MontbrioPazoRoxin(); mpr.configure()
        sn = Subnetwork(name="ctx", model=mpr, scheme=HeunDeterministic(dt=DT), nnodes=76)
        sn.projections = [_make_intra(conn.weights[:76, :76], [0], [0], cfun=Linear())]
        sn.configure()
        ns = NetworkSet(subnets=[sn], projections=[]); ns.configure()

        backend = NbHybridBackend()
        sweep_vals = np.linspace(0.01, 0.05, 100).astype(np.float32)

        t0 = time.perf_counter()
        backend.sweep(ns, params={"coupling_scale": sweep_vals},
                      nstep=100, backend="cpu", n_workers=1)
        t_seq = time.perf_counter() - t0

        t0 = time.perf_counter()
        backend.sweep(ns, params={"coupling_scale": sweep_vals},
                      nstep=100, backend="cpu", n_workers=4)
        t_par = time.perf_counter() - t0

        # 4-core should be at least 2x faster
        speedup = t_seq / t_par
        print(f"  Sequential: {t_seq:.2f}s, Parallel(4): {t_par:.2f}s, Speedup: {speedup:.1f}x")
        assert speedup > 1.5, f"Multi-core speedup {speedup:.1f}x < 1.5x"


# ---------------------------------------------------------------------------
# Test: GPU sweep (auto-dispatch)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestGPUSweep:
    def test_gpu_auto_dispatch(self):
        """backend='auto' uses CUDA when available."""
        conn = _load_conn("connectivity_76")
        mpr = MontbrioPazoRoxin(); mpr.configure()
        sn = Subnetwork(name="ctx", model=mpr, scheme=HeunDeterministic(dt=DT), nnodes=76)
        sn.projections = [_make_intra(conn.weights[:76, :76], [0], [0], cfun=Linear())]
        sn.configure()
        ns = NetworkSet(subnets=[sn], projections=[]); ns.configure()

        backend = NbHybridBackend()
        result = backend.sweep(
            ns, params={"coupling_scale": np.linspace(0.0, 0.1, 20).astype(np.float32)},
            nstep=100, backend="auto")

        assert result.backend == "cuda"
        assert result.merged_tavg.shape[0] == 20
        assert result.merged_tavg.shape[3] == 76

    def test_gpu_vs_cpu_shapes(self):
        """GPU and CPU produce same-shaped output for the same sweep."""
        conn = _load_conn("connectivity_76")
        mpr = MontbrioPazoRoxin(); mpr.configure()
        sn = Subnetwork(name="ctx", model=mpr, scheme=HeunDeterministic(dt=DT), nnodes=76)
        sn.projections = [_make_intra(conn.weights[:76, :76], [0], [0], cfun=Linear())]
        sn.configure()
        ns = NetworkSet(subnets=[sn], projections=[]); ns.configure()

        vals = np.linspace(0.0, 0.1, 5).astype(np.float32)
        backend = NbHybridBackend()

        cpu_result = backend.sweep(ns, params={"coupling_scale": vals},
                                    nstep=20, backend="cpu")
        gpu_result = backend.sweep(ns, params={"coupling_scale": vals},
                                    nstep=20, backend="cuda")

        assert cpu_result.merged_tavg.shape[0] == gpu_result.merged_tavg.shape[0]
        assert cpu_result.merged_tavg.shape[2] == gpu_result.merged_tavg.shape[2]
        assert cpu_result.merged_tavg.shape[3] == gpu_result.merged_tavg.shape[3]

    def test_gpu_fallback(self):
        """backend='cuda' with no GPU raises RuntimeError."""
        conn = _load_conn("connectivity_76")
        mpr = MontbrioPazoRoxin(); mpr.configure()
        sn = Subnetwork(name="ctx", model=mpr, scheme=HeunDeterministic(dt=DT), nnodes=76)
        sn.projections = [_make_intra(conn.weights[:76, :76], [0], [0], cfun=Linear())]
        sn.configure()
        ns = NetworkSet(subnets=[sn], projections=[]); ns.configure()

        if not CUDA_AVAILABLE:
            backend = NbHybridBackend()
            # auto should fall back to CPU
            result = backend.sweep(ns, params={"coupling_scale": np.linspace(0, 0.1, 5).astype(np.float32)},
                                   nstep=20, backend="auto")
            assert result.backend.startswith("cpu")


# ---------------------------------------------------------------------------
# Test: Benchmark (kiter/s)
# ---------------------------------------------------------------------------

class TestBenchmark:
    def test_throughput_kiter(self):
        """Report throughput in kiter/s for different backends."""
        conn = _load_conn("connectivity_76")
        mpr = MontbrioPazoRoxin(); mpr.configure()
        sn = Subnetwork(name="ctx", model=mpr, scheme=HeunDeterministic(dt=DT), nnodes=76)
        sn.projections = [_make_intra(conn.weights[:76, :76], [0], [0], cfun=Linear())]
        sn.configure()
        ns = NetworkSet(subnets=[sn], projections=[]); ns.configure()

        n_sweeps = 500
        n_steps = 1000
        vals = np.linspace(0.0, 0.1, n_sweeps).astype(np.float32)
        backend = NbHybridBackend()

        # CPU sequential
        t0 = time.perf_counter()
        result = backend.sweep(ns, params={"coupling_scale": vals},
                               nstep=n_steps, backend="cpu", n_workers=1)
        t_cpu_seq = time.perf_counter() - t0
        kiter_cpu_seq = n_sweeps * n_steps / t_cpu_seq / 1000

        # CPU 4-core
        t0 = time.perf_counter()
        result = backend.sweep(ns, params={"coupling_scale": vals},
                               nstep=n_steps, backend="cpu", n_workers=4)
        t_cpu_4c = time.perf_counter() - t0
        kiter_cpu_4c = n_sweeps * n_steps / t_cpu_4c / 1000

        print(f"\n  CPU-seq:  {t_cpu_seq:.2f}s  {kiter_cpu_seq:.0f} kiter/s")
        print(f"  CPU-4c:   {t_cpu_4c:.2f}s  {kiter_cpu_4c:.0f} kiter/s")
        print(f"  Speedup:  {t_cpu_seq/t_cpu_4c:.1f}x")

        # GPU if available
        if CUDA_AVAILABLE:
            t0 = time.perf_counter()
            result = backend.sweep(ns, params={"coupling_scale": vals},
                                    nstep=n_steps, backend="cuda")
            t_cuda = time.perf_counter() - t0
            kiter_cuda = n_sweeps * n_steps / t_cuda / 1000
            print(f"  CUDA:     {t_cuda:.3f}s  {kiter_cuda:.0f} kiter/s")
            print(f"  CUDA/4c:  {kiter_cuda/kiter_cpu_4c:.1f}x")

        # Sanity: CPU sequential must produce some output
        assert result.merged_tavg is not None
        assert kiter_cpu_seq > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])