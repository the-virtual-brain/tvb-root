#!/usr/bin/env python3
"""
C++ Hybrid Backend — Getting-Started Demo
==========================================

C++ equivalent of ``tvb_documentation/demos/simulate_hybrid_getting_started.py``.

Setup
-----
* Cortex  : 38 nodes, JansenRit (1 mode)
* Thalamus: 38 nodes, ReducedSetFitzHughNagumo (3 modes)
* Projection: cortex (y1) ---> thalamus (xi)

Key difference vs Python/Numba backend
---------------------------------------
``compiled.run()`` returns a **list** of ``(times, data)`` tuples — one per subnet.
For multi-mode subnets (e.g. FHN with 3 modes) the data shape is
``(n_chunks, n_voi, n_nodes, n_modes)``; the Python/Numba API sums over modes
before returning.  Sum explicitly to match Numba's convention:
``data_summed = data.sum(axis=-1)``

Usage
-----
Run from the repo root::

    python tvb_library/tvb/simulator/backend_cpp/examples/cpp_hybrid_getting_started.py

Optional: set TVB_CPP_BUILD_DIR to control where the compiled extension lands.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

_EXAMPLES_DIR  = Path(__file__).resolve().parent
_LIBRARY_ROOT  = _EXAMPLES_DIR.parent.parent.parent.parent  # tvb_library/
for _p in (_LIBRARY_ROOT,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

os.environ.setdefault("TVB_USER_HOME", str(Path(tempfile.gettempdir()) / "tvb-user"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba-cache"))

warnings.filterwarnings("ignore", message="Hybrid simulation is experimental.*")

import numpy as np
import scipy.sparse as sp

from tvb.simulator.backend_cpp import CppHybridBackend
from tvb.simulator.hybrid import InterProjection, NetworkSet, Subnetwork
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models import JansenRit, ReducedSetFitzHughNagumo
from tvb.simulator.monitors import TemporalAverage


# ---------------------------------------------------------------------------
# Parameters (matching the getting-started demo)
# ---------------------------------------------------------------------------

DT               = 0.1    # ms
N_NODES          = 38
SIMULATION_MS    = 1000.0
TAVG_PERIOD      = 1.0    # ms — one sample per ms

N_STEP     = int(round(SIMULATION_MS / DT))
CHUNK_SIZE = int(round(TAVG_PERIOD / DT))   # steps per output sample

BUILD_ROOT = Path(
    os.environ.get(
        "TVB_CPP_BUILD_DIR",
        str(_EXAMPLES_DIR / ".build"),
    )
).resolve()


# ---------------------------------------------------------------------------
# Build subnetworks
# ---------------------------------------------------------------------------

cortex = Subnetwork(
    name="cortex",
    model=JansenRit(),
    scheme=HeunDeterministic(dt=DT),
    nnodes=N_NODES,
).configure()
cortex.node_indices = np.arange(N_NODES)

thalamus = Subnetwork(
    name="thalamus",
    model=ReducedSetFitzHughNagumo(),
    scheme=HeunDeterministic(dt=DT),
    nnodes=N_NODES,
).configure()
thalamus.node_indices = np.arange(N_NODES)

print("=== Cortex (JansenRit) ===")
print(f"  state_variables : {cortex.model.state_variables}")
print(f"  variables_of_interest: {cortex.model.variables_of_interest}")
print(f"  cvar: {cortex.model.cvar}  -> {[cortex.model.state_variables[i] for i in cortex.model.cvar]}")
print(f"  n_modes: {cortex.model.number_of_modes}")

print("\n=== Thalamus (ReducedSetFitzHughNagumo) ===")
print(f"  state_variables : {thalamus.model.state_variables}")
print(f"  variables_of_interest: {thalamus.model.variables_of_interest}")
print(f"  cvar: {thalamus.model.cvar}  -> {[thalamus.model.state_variables[i] for i in thalamus.model.cvar]}")
print(f"  n_modes: {thalamus.model.number_of_modes}")


# ---------------------------------------------------------------------------
# Inter-projection: cortex y1 → thalamus xi
# ---------------------------------------------------------------------------

rng = np.random.RandomState(42)
weights_dense  = np.abs(rng.randn(N_NODES, N_NODES)) * 0.01
weights_sparse = sp.csr_matrix(weights_dense)
lengths_sparse = sp.csr_matrix((N_NODES, N_NODES))     # zero delays

proj = InterProjection(
    source=cortex,
    target=thalamus,
    source_cvar="y1",    # JansenRit excitatory PSP
    target_cvar="xi",    # FHN slow variable
    weights=weights_sparse,
    lengths=lengths_sparse,
    cv=7.0,
    dt=DT,
    scale=0.1,
)

print(f"\nInter-projection: cortex(y1) -> thalamus(xi)")
print(f"  source_cvar index: {proj.source_cvar}")
print(f"  target_cvar slot : {proj.target_cvar}")


# ---------------------------------------------------------------------------
# Assemble NetworkSet and compile
# ---------------------------------------------------------------------------

network = NetworkSet(
    subnets=[cortex, thalamus],
    projections=[proj],
    stimuli=[],
)
network.configure()

monitor = TemporalAverage(period=TAVG_PERIOD)

print(f"\nCompiling C++ extension to {BUILD_ROOT} ...")
t0 = time.perf_counter()
backend  = CppHybridBackend(build_root=BUILD_ROOT)
compiled = backend.compile(
    network,
    monitors=[monitor],
    user_source_hint="cpp_hybrid_getting_started",
)
compile_time = time.perf_counter() - t0
print(f"Compilation done in {compile_time:.1f}s")
print(f"  subnetworks   : {compiled.debug_summary()['n_subnetworks']}")
print(f"  inter-projs   : {compiled.debug_summary()['n_inter_projections']}")
print(f"  intra-projs   : {compiled.debug_summary()['n_intra_projections']}")


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

ic_cortex   = np.zeros((cortex.model.nvar,   N_NODES, cortex.model.number_of_modes),
                        dtype=np.float64)    # JR: (6, 38, 1)
ic_thalamus = np.zeros((thalamus.model.nvar, N_NODES, thalamus.model.number_of_modes),
                        dtype=np.float64)    # FHN: (4, 38, 3)


# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------

print(f"\nRunning {SIMULATION_MS} ms ({N_STEP} steps, chunk_size={CHUNK_SIZE}) ...")
t0 = time.perf_counter()
results = compiled.run(
    initial_states=[ic_cortex, ic_thalamus],
    nstep=N_STEP,
    chunk_size=CHUNK_SIZE,
)
run_time = time.perf_counter() - t0
print(f"Simulation done in {run_time:.2f}s")

(times_ctx,  data_ctx)  = results[0]    # cortex
(times_thal, data_thal) = results[1]    # thalamus

print(f"\n  Cortex output   : times={times_ctx.shape}, data={data_ctx.shape}")
print(f"  Thalamus output : times={times_thal.shape}, data={data_thal.shape}")
print("  (FHN keeps 3 modes; sum over last axis to match Numba convention)")

# Sanity checks
assert not np.any(np.isnan(data_ctx)),  "NaN in cortex output!"
assert not np.any(np.isnan(data_thal)), "NaN in thalamus output!"
print("\n  No NaN in output — OK")


# ---------------------------------------------------------------------------
# Extract per-subnetwork VOI traces
# ---------------------------------------------------------------------------

# Cortex: 4 VOIs (y0, y1, y2, y3), 1 mode
y1_ctx  = data_ctx[:, 1, :, 0]             # JR y1, shape (1000, 38)
y1_mean = y1_ctx.mean(axis=1)              # mean over nodes

# Thalamus: 2 VOIs (xi, alpha), 3 modes  — sum over modes for scalar output
xi_thal  = data_thal[:, 0, :, :].sum(-1)  # xi, mode-summed, (1000, 38)
xi_mean  = xi_thal.mean(axis=1)

print(f"\n  JR y1 amplitude (last 500 ms): {y1_mean[-500:].max() - y1_mean[-500:].min():.4f}")
print(f"  FHN xi amplitude (last 500 ms): {xi_mean[-500:].max() - xi_mean[-500:].min():.4f}")


# ---------------------------------------------------------------------------
# Optional: compare with Numba backend
# ---------------------------------------------------------------------------

try:
    from tvb.simulator.backend.nb_hybrid import NbHybridBackend

    print("\nRunning Numba backend for comparison ...")
    nb_backend = NbHybridBackend()

    def _average_chunks(times, data, chunk_size):
        if chunk_size == 1:
            return times, data
        n_window = data.shape[0] // chunk_size
        n_keep = n_window * chunk_size
        if n_window == 0:
            return times, data
        avg_times = times[:n_keep].reshape(n_window, chunk_size).mean(axis=1)
        avg_data = data[:n_keep].reshape(n_window, chunk_size, *data.shape[1:]).mean(axis=1)
        return avg_times, avg_data

    try:
        nb_results = nb_backend.run_network(
            network,
            nstep=N_STEP,
            chunk_size=CHUNK_SIZE,
            initial_states=[ic_cortex.copy(), ic_thalamus.copy()],
        )
    except ValueError as exc:
        if "exceeds the minimum projection horizon" not in str(exc):
            raise
        print(
            "  Numba rejects chunk_size=10 for zero-delay projections; "
            "running chunk_size=1 and averaging to 1 ms."
        )
        nb_results = nb_backend.run_network(
            network,
            nstep=N_STEP,
            chunk_size=1,
            initial_states=[ic_cortex.copy(), ic_thalamus.copy()],
        )
        nb_results = [
            (*_average_chunks(t, d, CHUNK_SIZE), c)
            for t, d, c in nb_results
        ]

    nb_t0, nb_d0, _ = nb_results[0]
    nb_t1, nb_d1, _ = nb_results[1]

    # Compare cortex y1 (VOI index 1)
    nb_y1_mean  = np.asarray(nb_d0[:, 1, :, 0], dtype=np.float64).mean(axis=1)

    # Compare FHN xi (VOI index 0). Direct NbHybridBackend output may retain
    # modes; Simulator monitor output sums them.
    nb_xi = np.asarray(nb_d1[:, 0, :, :], dtype=np.float64)
    if nb_xi.shape[-1] != 1:
        nb_xi = nb_xi.sum(axis=-1)
    else:
        nb_xi = nb_xi[..., 0]
    nb_xi_mean = nb_xi.mean(axis=1)

    cortex_diff  = np.abs(y1_mean  - nb_y1_mean).max()
    thalamus_diff = np.abs(xi_mean - nb_xi_mean).max()
    print(f"  Cortex   max|C++ - Numba| = {cortex_diff:.3e}")
    print(f"  Thalamus max|C++ - Numba| = {thalamus_diff:.3e}")

except ImportError:
    print("(Numba backend not available — skipping comparison)")


# ---------------------------------------------------------------------------
# Optional: plot
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(times_ctx, y1_mean, linewidth=0.8, color="steelblue", label="C++ backend")
    axes[0].set_title("Cortex — JansenRit y₁ (mean over 38 nodes)")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("y₁ (mV)")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    axes[1].plot(times_thal, xi_mean, linewidth=0.8, color="firebrick", label="C++ backend")
    axes[1].set_title("Thalamus — FHN ξ, mode-summed (mean over 38 nodes)")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("ξ (a.u.)")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.3)

    fig.suptitle("C++ hybrid backend: cortex (JR) → thalamus (FHN)", fontsize=13)
    plt.tight_layout()

    out_path = Path("cpp_hybrid_getting_started.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")

except ImportError:
    print("(matplotlib not available — skipping plot)")

print("\nDone.")
