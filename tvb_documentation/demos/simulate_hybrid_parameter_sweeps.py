# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Hybrid Parameter Sweeps — Tuning a Multi-Subnetwork Model
#
# This notebook demonstrates how to use the **unified `sweep()` API** to tune
# coupling parameters in a multi-subnetwork brain model.  We build a
# cortex + thalamus network using **two JansenRit** nodes and sweep the
# inter-subnetwork coupling strength to characterize the response.
#
# ## What you'll learn
# - Build a two-subnetwork hybrid model (JansenRit cortex + JansenRit thalamus)
# - Sweep coupling parameters with named keys (`"coupling_scale"`)
# - Compare CPU sequential and GPU sweep performance
# - Visualize the coupling effect on both subnets
#
# **Prerequisites:** TVB Hybrid + Numba backends installed.  GPU results
# require a CUDA-capable NVIDIA GPU and `numba.cuda`.

# %% [markdown]
# ## 1. Imports and data loading

# %%
import numpy as np
import scipy.sparse as sp
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.hybrid.intra_projection import IntraProjection
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.jansen_rit import JansenRit
from tvb.simulator.backend.nb_hybrid import NbHybridBackend

DT = 0.1           # integration time step [ms]
NSTEP = 5000        # simulation length per sweep point [steps]
                    # = 500 ms — long enough for ~5 alpha cycles

# Load the standard 76-node connectome
conn = Connectivity.from_file("connectivity_76.zip")
conn.configure()
N_CORTEX = 68       # first 68 nodes → cortex
N_THALAMUS = 8      # remaining 8 nodes → thalamus
N_TOTAL = 76

# Helper: slice of the connectome weights matrix
def _slice_conn(row_slice, col_slice):
    return conn.weights[row_slice[0]:row_slice[1],
                        col_slice[0]:col_slice[1]].astype(np.float32)

def _slice_lengths(row_slice, col_slice):
    return conn.tract_lengths[row_slice[0]:row_slice[1],
                               col_slice[0]:col_slice[1]].astype(np.float32)

# %% [markdown]
# ## 2. Building the cortex + thalamus network
#
# Both subnetworks use the **JansenRit** model — a well-studied cortical
# neural mass that generates alpha-band (~10 Hz) oscillations.
#
# - **Cortex** (68 nodes): intra-connectivity via `Linear` coupling with
#   `a=0.03` — this sustains intrinsic oscillations.
# - **Thalamus** (8 nodes): NO intra-connectivity — the dynamics are
#   driven solely by the cortical input.  This lets us clearly see the
#   coupling effect.
# - **Inter-projection**: cortical excitatory output `y1` feeds into
#   the thalamic `y0` variable via a swept `Linear` coupling function.
#
# Both subnets share the same model type (JR), same VOI count (4), and
# are numerically stable at `dt=0.1`.

# %%
# -- Cortex (JansenRit, 68 nodes) --
jr_ctx = JansenRit()
jr_ctx.configure()

ctx = Subnetwork(name="cortex", model=jr_ctx,
                 scheme=HeunDeterministic(dt=DT), nnodes=N_CORTEX)
w_ctx = _slice_conn((0, N_CORTEX), (0, N_CORTEX))
tl_ctx = _slice_lengths((0, N_CORTEX), (0, N_CORTEX))
ctx_intra = IntraProjection(
    source_cvar=np.array([0], dtype=np.int_),
    target_cvar=np.array([0], dtype=np.int_),
    weights=sp.csr_matrix(w_ctx),
    lengths=sp.csr_matrix(tl_ctx),
    cv=1.0, dt=DT, scale=1.0,
    cfun=Linear(a=np.array([0.03]))
)
ctx.projections = [ctx_intra]
ctx.configure()

# -- Thalamus (JansenRit, 8 nodes) --
jr_thal = JansenRit()
jr_thal.configure()

thal = Subnetwork(name="thalamus", model=jr_thal,
                  scheme=HeunDeterministic(dt=DT), nnodes=N_THALAMUS)
# No intra-projections on thalamus — dynamics purely from cortical drive
thal.configure()

# -- Inter-projection: cortex → thalamus --
# Source: JR y1 (excitatory PSP, index 1). Target: JR y0 (index 0).
w_inter = _slice_conn((0, N_CORTEX), (N_CORTEX, N_TOTAL))
tl_inter = _slice_lengths((0, N_CORTEX), (N_CORTEX, N_TOTAL))
c2t = InterProjection(
    source=ctx, target=thal,
    source_cvar=1,        # JR y1
    target_cvar=0,        # JR y0
    weights=sp.csr_matrix(w_inter),
    lengths=sp.csr_matrix(tl_inter),
    cv=1.0, dt=DT, scale=1.0,
    cfun=Linear(a=np.array([0.01]))
)

# Assemble and configure
ns = NetworkSet(subnets=[ctx, thal], projections=[c2t])
ns.configure()

print(f"Cortex:   {ctx.nnodes} nodes, {len(jr_ctx.state_variables)} vars, "
      f"VOIs: {jr_ctx.variables_of_interest}")
print(f"Thalamus: {thal.nnodes} nodes, {len(jr_thal.state_variables)} vars, "
      f"VOIs: {jr_thal.variables_of_interest}")
print(f"Inter-projection: {c2t.source.name} → {c2t.target.name}, "
      f"{c2t.weights.shape[0]}×{c2t.weights.shape[1]}")

# %% [markdown]
# ## 3. Coupling strength parameter sweep (CPU)
#
# We sweep the inter-projection coupling parameter `a` over 20 values
# from 0.002 to 0.1.  Each value runs an independent simulation of 500 ms
# (5000 steps at dt=0.1 ms).
#
# The unified API uses named keys:
# ```python
# params={"coupling_scale": sweep_vals}
# ```
# `"coupling_scale"` automatically resolves to parameter `a` of the
# `Linear` coupling on the `cortex_to_thalamus` inter-projection.

# %%
backend = NbHybridBackend()
sweep_vals = np.linspace(0.002, 0.1, 20).astype(np.float32)

t0 = time.perf_counter()
result = backend.sweep(
    ns,
    params={"coupling_scale": sweep_vals},
    nstep=NSTEP,
    backend="cpu",
    n_workers=1,
)
elapsed = time.perf_counter() - t0

print(f"CPU sequential: {elapsed:.1f}s "
      f"({len(sweep_vals)} sweeps × {NSTEP} steps)")
print(f"Backend: {result.backend}")
print(f"Subnets: {list(result.tavg.keys())}")
for name, arr in result.tavg.items():
    print(f"  tavg['{name}'] shape: {arr.shape}")
    print(f"  mean: {np.mean(arr):.4f}, NaN: {np.any(np.isnan(arr))}")

# %% [markdown]
# ## 4. Visualizing coupling effect on each subnet
#
# We plot the **mean signal over all nodes** for one representative VOI
# per subnetwork as a function of coupling strength.

# %%
ctx_tavg = result.tavg["cortex"]    # (n_sweeps, n_voi, N_cortex, modes)
thal_tavg = result.tavg["thalamus"] # (n_sweeps, n_voi, N_thalamus, modes)

# Average over nodes (axis=2) and squeeze modes (axis=3)
ctx_mean = ctx_tavg.mean(axis=2).squeeze(-1)   # (n_sweeps, n_voi)
thal_mean = thal_tavg.mean(axis=2).squeeze(-1) # (n_sweeps, n_voi)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Cortex VOI y1 (excitatory PSP) — index 1
axes[0].plot(sweep_vals, ctx_mean[:, 1], 'o-', color='#1f77b4', linewidth=2)
axes[0].set_xlabel("Coupling strength a")
axes[0].set_ylabel("Mean y1 (excitatory PSP)")
axes[0].set_title("Cortex (JansenRit)")
axes[0].grid(True, alpha=0.3)

# Thalamus VOI y0 — index 0
axes[1].plot(sweep_vals, thal_mean[:, 0], 'o-', color='#d62728', linewidth=2)
axes[1].set_xlabel("Coupling strength a")
axes[1].set_ylabel("Mean y0")
axes[1].set_title("Thalamus (JansenRit, cortical-driven)")
axes[1].grid(True, alpha=0.3)

fig.suptitle("Coupling Strength vs Mean Activity (500 ms simulation)", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Coupling response amplitude
#
# We compute the **deviation from baseline** (lowest coupling) as a
# measure of how sensitive each subnet is to coupling changes.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, data, voi_idx, label, color in [
    (axes[0], ctx_mean, 1, "Cortex y1", '#1f77b4'),
    (axes[1], thal_mean, 0, "Thalamus y0", '#d62728'),
]:
    signal = data[:, voi_idx]
    amplitude = np.abs(signal - signal[0])  # deviation from baseline
    ax.plot(sweep_vals, amplitude, 'o-', linewidth=2, color=color)
    ax.set_xlabel("Coupling strength a")
    ax.set_ylabel(f"|{label} − baseline|")
    ax.set_title(f"{label} response amplitude")
    ax.grid(True, alpha=0.3)

fig.suptitle("Coupling Response Amplitude", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Performance: CPU vs GPU
#
# We benchmark the sweep on CPU sequential and GPU (CUDA).
# GPU compilation is a one-time cost (~2–3s for JIT); once cached,
# the GPU delivers 2–15× speedups depending on model complexity.
# Results are in **kiter/s**.

# %%
N_SWEEP = 500  # enough sweep points to saturate the GPU
# Single-subnet JR for clean benchmarking
jr_bench = JansenRit()
jr_bench.configure()
sn_bench = Subnetwork(name="cortex", model=jr_bench,
                      scheme=HeunDeterministic(dt=DT), nnodes=N_CORTEX)
w_bench = _slice_conn((0, N_CORTEX), (0, N_CORTEX))
tl_bench = _slice_lengths((0, N_CORTEX), (0, N_CORTEX))
sn_bench.projections = [IntraProjection(
    source_cvar=np.array([0], dtype=np.int_),
    target_cvar=np.array([0], dtype=np.int_),
    weights=sp.csr_matrix(w_bench),
    lengths=sp.csr_matrix(tl_bench),
    cv=1.0, dt=DT, scale=1.0,
    cfun=Linear()
)]
sn_bench.configure()
ns_bench = NetworkSet(subnets=[sn_bench], projections=[])
ns_bench.configure()

bench_vals = np.linspace(0.01, 0.05, N_SWEEP).astype(np.float32)

# --- CPU sequential ---
t0 = time.perf_counter()
backend.sweep(ns_bench, params={"coupling_scale": bench_vals},
              nstep=2000, backend="cpu", n_workers=1)
t_seq = time.perf_counter() - t0
kis_seq = N_SWEEP * 2000 / t_seq / 1000
print(f"CPU sequential: {t_seq:.1f}s → {kis_seq:.1f} kiter/s")

# --- GPU: warmup (compile + JIT), then timed run ---
try:
    # Warmup: compile CUDA kernel (one-time cost, cached)
    t_warmup = time.perf_counter()
    backend.sweep(ns_bench, params={"coupling_scale": bench_vals[:5]},
                  nstep=2000, backend="cuda")
    t_warmup = time.perf_counter() - t_warmup
    print(f"GPU warmup (compile): {t_warmup:.1f}s")
    
    # Timed run
    t0 = time.perf_counter()
    backend.sweep(ns_bench, params={"coupling_scale": bench_vals},
                  nstep=2000, backend="cuda")
    t_gpu = time.perf_counter() - t0
    kis_gpu = N_SWEEP * 2000 / t_gpu / 1000
    gpu_available = True
    print(f"GPU (CUDA): {t_gpu:.1f}s → {kis_gpu:.1f} kiter/s "
          f"({kis_gpu/kis_seq:.1f}× over CPU)")
except Exception as e:
    gpu_available = False
    print(f"GPU not available: {e}")

# %% [markdown]
# ## 7. Multi-core speedup (optional)
#
# On systems where Numba's JIT is fork-safe (most single-subnet models),
# multi-core CPU sweeps via `n_workers` provide near-linear speedup.
# Uncomment to test on your machine:

# %%
# N_CPU = 4
# t0 = time.perf_counter()
# backend.sweep(ns_bench, params={"coupling_scale": bench_vals},
#               nstep=NSTEP, backend="cpu", n_workers=N_CPU)
# t_par = time.perf_counter() - t0
# kis_par = N_SWEEP * NSTEP / t_par / 1000
# print(f"CPU {N_CPU}-core: {t_par:.1f}s, {kis_par:.1f} kiter/s "
#       f"({kis_par/kis_seq:.1f}× over sequential)")

# %% [markdown]
# ## 8. Summary
#
# - **Unified `sweep()` API**: same code for CPU and GPU — just change
#   `backend="cpu"` to `backend="cuda"`.
# - **Named parameters** (`"coupling_scale"`) automatically resolve to
#   the correct cfun/model parameter across subnets and projections.
# - **JansenRit + JansenRit** is a stable configuration suitable for
#   parameter sweeps — both subnets remain well-behaved at `dt=0.1` for
#   thousands of steps, even with strong inter-subnet coupling.
# - **GPU acceleration** provides 2–15× speedups depending on model
#   complexity — the advantage grows with more state variables and modes
#   per node.
# - **Multi-core CPU** sweeps are available via `n_workers` (fork-based);
#   a future prange-based kernel will provide thread-parallel sweeps
#   without fork-safety concerns.
#
# This workflow is the foundation for systematic model tuning and
# sensitivity analysis in TVB Hybrid simulations.
