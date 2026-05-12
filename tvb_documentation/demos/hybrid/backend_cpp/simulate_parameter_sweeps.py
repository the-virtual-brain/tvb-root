#!/usr/bin/env python
# coding: utf-8

# # C++ Hybrid Parameter Sweeps
# 
# This notebook is the C++ backend equivalent of `simulate_hybrid_parameter_sweeps.ipynb`.
# It builds the same cortex + thalamus hybrid `NetworkSet`, sweeps the inter-subnetwork `Linear.a` coupling parameter, and visualizes the response.
# 
# The C++ sweep API is intentionally close to the Numba API:
# 
# ```python
# CppHybridBackend().sweep(network, params={"coupling_scale": sweep_values}, nstep=NSTEP)
# ```
# 
# For C-function scale sweeps, the backend compiles the generated C++ extension once and applies each sweep value as a runtime projection scale. Model-parameter sweeps are also supported, but may trigger recompilation for each distinct model value.
# 

# ## 1. Imports and notebook path setup
# 
# The notebook can be run from this directory or from the repository root. This cell finds `tvb_library/`, adds it to `sys.path`, and uses a local C++ build cache under the backend_cpp demo folder unless `TVB_CPP_BUILD_DIR` is already set.
# 

from __future__ import annotations

import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

def find_repo_root(start: Path | None = None) -> Path:
    # __file__ is undefined in Jupyter kernels; fall back to cwd in that case.
    try:
        script_dir: Path | None = Path(__file__).resolve().parent
    except NameError:
        script_dir = None
    for search_root in (start, script_dir, Path.cwd()):
        if search_root is None:
            continue
        for candidate in (search_root.resolve(), *search_root.resolve().parents):
            if (candidate / "tvb_library" / "tvb" / "simulator").exists():
                return candidate
    raise RuntimeError("Could not locate the tvb-root repository root.")

REPO_ROOT = find_repo_root()
TVB_LIBRARY_ROOT = REPO_ROOT / "tvb_library"
DEMO_DIR = REPO_ROOT / "tvb_documentation" / "demos" / "hybrid" / "backend_cpp"
BUILD_ROOT = Path(os.environ.get("TVB_CPP_BUILD_DIR", str(DEMO_DIR / ".build"))).resolve()

if str(TVB_LIBRARY_ROOT) not in sys.path:
    sys.path.insert(0, str(TVB_LIBRARY_ROOT))

os.environ.setdefault("TVB_USER_HOME", str(Path(tempfile.gettempdir()) / "tvb-user"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
warnings.filterwarnings("ignore", message="Hybrid simulation is experimental.*")

print(f"Repository root: {REPO_ROOT}")
print(f"C++ build root:  {BUILD_ROOT}")

# ## 2. Build the cortex + thalamus network
# 
# Both subnetworks use `JansenRit`. The cortex has intra-connectivity; the thalamus has no intra-connectivity and is driven by the cortex through one inter-projection. The swept parameter is the `a` multiplier of that inter-projection's `Linear` coupling function.
# 

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.backend_cpp import CppHybridBackend
from tvb.simulator.hybrid import IntraProjection, InterProjection, NetworkSet, Subnetwork
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.jansen_rit import JansenRit
from tvb.simulator.monitors import TemporalAverage

DT = 0.1
NSTEP = 5000
TAVG_PERIOD = 1
N_CORTEX = 68
N_THALAMUS = 8
N_TOTAL = N_CORTEX + N_THALAMUS
N_SWEEP_POINTS = 50

conn = Connectivity.from_file("connectivity_76.zip")
conn.configure()

def slice_weights(row_slice, col_slice):
    return np.asarray(
        conn.weights[row_slice[0]:row_slice[1], col_slice[0]:col_slice[1]],
        dtype=np.float64,
    )

def slice_lengths(row_slice, col_slice):
    return np.asarray(
        conn.tract_lengths[row_slice[0]:row_slice[1], col_slice[0]:col_slice[1]],
        dtype=np.float64,
    )

ctx_model = JansenRit()
ctx_model.configure()
ctx = Subnetwork(
    name="cortex",
    model=ctx_model,
    scheme=HeunDeterministic(dt=DT),
    nnodes=N_CORTEX,
)
ctx.node_indices = np.arange(N_CORTEX)
ctx.projections = [
    IntraProjection(
        source_cvar=np.array([0], dtype=np.int_),
        target_cvar=np.array([0], dtype=np.int_),
        weights=sp.csr_matrix(slice_weights((0, N_CORTEX), (0, N_CORTEX))),
        lengths=sp.csr_matrix(slice_lengths((0, N_CORTEX), (0, N_CORTEX))),
        cv=1.0,
        dt=DT,
        scale=1.0,
        cfun=Linear(a=np.array([0.03])),
    )
]
ctx.configure()

thal_model = JansenRit()
thal_model.configure()
thal = Subnetwork(
    name="thalamus",
    model=thal_model,
    scheme=HeunDeterministic(dt=DT),
    nnodes=N_THALAMUS,
)
thal.node_indices = np.arange(N_CORTEX, N_TOTAL)
thal.configure()

cortex_to_thalamus = InterProjection(
    source=ctx,
    target=thal,
    source_cvar=1,
    target_cvar=0,
    weights=sp.csr_matrix(slice_weights((N_CORTEX, N_TOTAL), (0, N_CORTEX))),
    lengths=sp.csr_matrix(slice_lengths((N_CORTEX, N_TOTAL), (0, N_CORTEX))),
    cv=1.0,
    dt=DT,
    scale=1.0,
    cfun=Linear(a=np.array([0.01])),
)

network = NetworkSet(subnets=[ctx, thal], projections=[cortex_to_thalamus])
network.configure()

print(f"Cortex:   {ctx.nnodes} nodes, VOIs: {ctx.model.variables_of_interest}")
print(f"Thalamus: {thal.nnodes} nodes, VOIs: {thal.model.variables_of_interest}")
print(f"Projection: {cortex_to_thalamus.source.name} -> {cortex_to_thalamus.target.name}")

# ## 3. Run the C++ sweep
#
# `coupling_scale` resolves to the `a` parameter on the first compatible coupling function. In this network that is the linear coupling multiplier on the inter-projection from cortex to thalamus.
#
# The `TemporalAverage` period controls the output time resolution: one sample is emitted every `period` ms,
# so the result arrays have shape `(n_sweeps, n_time, n_voi, n_nodes, n_modes)` where `n_time = nstep * dt / period`.
#

sweep_values = np.linspace(0.002, 0.1, N_SWEEP_POINTS, dtype=np.float32)
node_indices = {
    "cortex": np.arange(N_CORTEX),
    "thalamus": np.arange(N_CORTEX, N_TOTAL),
}

backend = CppHybridBackend(build_root=BUILD_ROOT)
monitor = TemporalAverage(period=TAVG_PERIOD)

# --- sequential ---
start = time.perf_counter()
result = backend.sweep(
    network,
    params={"coupling_scale": sweep_values},
    nstep=NSTEP,
    monitors=[monitor],
    node_indices=node_indices,
    n_workers=1,
)
elapsed_seq = time.perf_counter() - start

# --- parallel (all available cores) ---
n_workers = os.cpu_count() or 1
start = time.perf_counter()
result_par = backend.sweep(
    network,
    params={"coupling_scale": sweep_values},
    nstep=NSTEP,
    monitors=[monitor],
    node_indices=node_indices,
    n_workers=n_workers,
)
elapsed_par = time.perf_counter() - start

print(f"C++ sequential sweep: {elapsed_seq:.2f}s ({len(sweep_values)} sweeps x {NSTEP} steps)")
print(f"C++ parallel sweep ({n_workers} workers): {elapsed_par:.2f}s  "
      f"(speedup {elapsed_seq / elapsed_par:.1f}x)")
print(f"Result backend: {result_par.backend}")
for name, arr in result_par.tavg.items():
    print(f"tavg[{name!r}] shape: {arr.shape}; mean={arr.mean():.4f}; NaN={np.isnan(arr).any()}")
print(f"merged_tavg shape: {None if result_par.merged_tavg is None else result_par.merged_tavg.shape}")

# Sanity-check: sequential and parallel results must agree.
for name in result.tavg:
    if not np.allclose(result.tavg[name], result_par.tavg[name], atol=1e-10):
        print(f"WARNING: seq vs par mismatch for '{name}'")
    else:
        print(f"OK: seq == par for '{name}'")

result = result_par  # use parallel result for the rest of the script

# ## 4. Visualize the coupling response
#
# The arrays have shape `(n_sweeps, n_time, n_voi, n_nodes, n_modes)`.
# Averaging over time (axis=1) and nodes (axis=3) gives one scalar per sweep point and VOI.
#

ctx_tavg = result.tavg["cortex"]    # (n_sweeps, n_time, n_voi, N_cortex, n_modes)
thal_tavg = result.tavg["thalamus"] # (n_sweeps, n_time, n_voi, N_thalamus, n_modes)
times = result.times                # (n_time,)

ctx_mean = ctx_tavg.mean(axis=(1, 3)).squeeze(-1)   # (n_sweeps, n_voi)
thal_mean = thal_tavg.mean(axis=(1, 3)).squeeze(-1) # (n_sweeps, n_voi)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(sweep_values, ctx_mean[:, 1], "o-", color="#1f77b4", linewidth=2)
axes[0].set_xlabel("Coupling strength a")
axes[0].set_ylabel("Mean y1")
axes[0].set_title("Cortex")
axes[0].grid(True, alpha=0.3)

axes[1].plot(sweep_values, thal_mean[:, 0], "o-", color="#d62728", linewidth=2)
axes[1].set_xlabel("Coupling strength a")
axes[1].set_ylabel("Mean y0")
axes[1].set_title("Thalamus")
axes[1].grid(True, alpha=0.3)

fig.suptitle("C++ backend sweep: coupling strength vs mean activity", fontsize=14)
plt.tight_layout()
plt.show()

# ## 4b. Time traces for selected coupling values
#
# Because TAVG_PERIOD < NSTEP * DT, each sweep point contains a full time series.
# Plot node-averaged traces for three coupling strengths.
#

selected_indices = [0, len(sweep_values) // 2, len(sweep_values) - 1]
trace_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for sw_idx, color in zip(selected_indices, trace_colors):
    label = f"a={sweep_values[sw_idx]:.3f}"
    # mean over nodes → (n_time,)
    axes[0].plot(times, ctx_tavg[sw_idx, :, 1, :, 0].mean(axis=-1), color=color, label=label)
    axes[1].plot(times, thal_tavg[sw_idx, :, 0, :, 0].mean(axis=-1), color=color, label=label)

for ax, title, ylabel in [
    (axes[0], "Cortex y1", "Mean y1"),
    (axes[1], "Thalamus y0", "Mean y0"),
]:
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("Time traces for selected coupling values", fontsize=14)
plt.tight_layout()
plt.show()

# ## 5. Response amplitude relative to baseline
# 
# This repeats the baseline-deviation view from the original notebook, using the lowest coupling value as the baseline.
# 

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, data, voi_idx, label, color in [
    (axes[0], ctx_mean, 1, "Cortex y1", "#1f77b4"),
    (axes[1], thal_mean, 0, "Thalamus y0", "#d62728"),
]:
    signal = data[:, voi_idx]
    amplitude = np.abs(signal - signal[0])
    ax.plot(sweep_values, amplitude, "o-", linewidth=2, color=color)
    ax.set_xlabel("Coupling strength a")
    ax.set_ylabel(f"|{label} - baseline|")
    ax.set_title(f"{label} response amplitude")
    ax.grid(True, alpha=0.3)

fig.suptitle("Coupling response amplitude", fontsize=14)
plt.tight_layout()
plt.show()

# ## 6. Compile-cache reuse benchmark
# 
# For C-function scale sweeps, the generated extension is compiled once and reused. 
# A second sweep with different values should skip compilation through the C++ build cache and mostly measure runtime.
 

bench_values = np.linspace(0.01, 0.05, N_SWEEP_POINTS, dtype=np.float32)
bench_nstep = 500
bench_monitor = TemporalAverage(period=bench_nstep * DT)

start = time.perf_counter()
bench_result = backend.sweep(
    network,
    params={"coupling_scale": bench_values},
    nstep=bench_nstep,
    monitors=[bench_monitor],
    node_indices=node_indices,
)
bench_elapsed = time.perf_counter() - start
kiter_s = len(bench_values) * bench_nstep / bench_elapsed / 1000.0

print(f"C++ cached sweep: {bench_elapsed:.2f}s -> {kiter_s:.1f} kiter/s")
print(f"Output shape, cortex: {bench_result.tavg['cortex'].shape}")

# ## 7. Summary
# 
# - `CppHybridBackend.sweep()` accepts the same named `params` style used by the hybrid Numba sweep API.
# - `coupling_scale`, `scale`, and explicit projection names such as `cortex_to_thalamus.a` map to C-function scale parameters.
# - C-function scale sweeps compile once and run sequentially on the CPU in the current C++ backend.
# - The returned `SweepResult.tavg` dictionary stores one array per subnetwork with shape `(n_sweeps, n_time, n_voi, n_nodes, n_modes)` where `n_time = nstep * dt / period`.
# 
