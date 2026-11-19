# %% [markdown]
# # Getting Started with Hybrid Brain Simulation
#
# Classic TVB simulates a single homogeneous network — every node runs the same
# model with the same equations.  The **Hybrid Simulation API** lifts that
# restriction: you can partition your network into *subnetworks*, assign a
# different neural-mass model to each, and wire them together with
# inter-subnetwork projections.
#
# **What you'll learn**
#
# - What a `Subnetwork` is and how to configure one
# - How to connect two subnetworks with an `InterProjection`
# - How to assemble a `NetworkSet` and run a `Simulator`
# - How to extract per-subnetwork time series from the concatenated output
# - How to verify numerical reproducibility and cross-check the C++ backend
#
# **Setup**: 38-node cortex (JansenRit) + 38-node thalamus
# (ReducedSetFitzHughNagumo), connected by a one-directional
# cortex → thalamus projection.

# %% Imports
import warnings
warnings.filterwarnings('ignore')   # suppress experimental-API warnings

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from tvb.simulator.models import JansenRit, ReducedSetFitzHughNagumo
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.monitors import TemporalAverage
from tvb.simulator.hybrid import (
    Subnetwork,
    InterProjection,
    NetworkSet,
    Simulator,
)

# Quick version check
import tvb.simulator
print("TVB simulator loaded from:", tvb.simulator.__file__)

# %% [markdown]
# ## Simulation parameters
#
# We use a shared integration time step of `dt = 0.1` ms throughout.
# We will run for **1000 ms** — long enough to see JansenRit alpha-band
# oscillations settle into a steady rhythm.

# %% Parameters
dt              = 0.1    # ms — integration time step
nnodes          = 38     # nodes per subnetwork
simulation_length = 1000.0  # ms

print(f"dt={dt} ms | nnodes={nnodes} per subnetwork | length={simulation_length} ms")

# %% [markdown]
# ## Building subnetworks
#
# A `Subnetwork` groups a neural-mass `model`, an integration `scheme`, and a
# node count.  `configure()` must be called before the subnetwork can be used
# in a `NetworkSet`.
#
# **JansenRit** models a cortical column's excitatory/inhibitory loop and
# spontaneously generates alpha-band (~10 Hz) oscillations.
#
# **ReducedSetFitzHughNagumo** is a population-level reduction of the classic
# FitzHugh–Nagumo neuron.  Under its default parameters it generates intrinsic
# limit-cycle oscillations whose amplitude and baseline are modulated by external
# inputs.  `NetworkSet.observe()` sums over modes before monitor output, so the
# default `number_of_modes = 3` works without modification.

# %% Build cortex subnetwork
cortex = Subnetwork(
    name='cortex',
    model=JansenRit(),
    scheme=HeunDeterministic(dt=dt),
    nnodes=nnodes,
).configure()

print("=== Cortex (JansenRit) ===")
print("  state_variables :", cortex.model.state_variables)
print("  variables_of_interest:", cortex.model.variables_of_interest)
print("  cvar (state-var indices used for coupling):", cortex.model.cvar)
print("  coupling-variable names:", [cortex.model.state_variables[i]
                                     for i in cortex.model.cvar])

# %% Build thalamus subnetwork
fhn_model = ReducedSetFitzHughNagumo()
# NetworkSet.observe() sums over modes before passing to monitors,
# so any number_of_modes (including the default of 3) works correctly.

thalamus = Subnetwork(
    name='thalamus',
    model=fhn_model,
    scheme=HeunDeterministic(dt=dt),
    nnodes=nnodes,
).configure()

print("=== Thalamus (ReducedSetFitzHughNagumo) ===")
print("  state_variables :", thalamus.model.state_variables)
print("  variables_of_interest:", thalamus.model.variables_of_interest)
print("  cvar (state-var indices used for coupling):", thalamus.model.cvar)
print("  coupling-variable names:", [thalamus.model.state_variables[i]
                                     for i in thalamus.model.cvar])

# %% [markdown]
# ## Creating an inter-subnetwork projection
#
# An `InterProjection` carries activity from one subnetwork's *source cvar*
# to another subnetwork's *target cvar*.  You name the variables using strings
# drawn from `model.state_variables`:
#
# | argument | value | meaning |
# |---|---|---|
# | `source_cvar` | `'y1'` | JansenRit excitatory PSP — a coupling variable |
# | `target_cvar` | `'xi'` | FHN slow variable — a coupling variable |
#
# The `weights` matrix has shape `(n_target_nodes, n_source_nodes)`.
# Zero-length tract lengths are passed here because we are not modelling
# conduction delays; `cv` and `dt` are required whenever `lengths` is given.

# %% Create InterProjection
rng = np.random.RandomState(42)
weights_dense  = np.abs(rng.randn(nnodes, nnodes)) * 0.01  # (target, source)
weights_sparse = sp.csr_matrix(weights_dense)               # shape: (38, 38)
lengths_sparse = sp.csr_matrix((nnodes, nnodes))            # zero delays

proj_ctx_to_thal = InterProjection(
    source=cortex,
    target=thalamus,
    source_cvar='y1',    # JansenRit coupling variable
    target_cvar='xi',    # FHN coupling variable
    weights=weights_sparse,
    lengths=lengths_sparse,
    cv=7.0,              # conduction velocity (mm/ms) — unused with zero lengths
    dt=dt,
    scale=0.1,           # global coupling strength
)

print("source_cvar index:", proj_ctx_to_thal.source_cvar)   # [1] (index of y1)
print("target_cvar slot :", proj_ctx_to_thal.target_cvar)   # [0] (slot of xi)

# %% [markdown]
# ## Assembling the NetworkSet and running the simulation
#
# `NetworkSet` collects all subnetworks and inter-projections into one model.
# `Simulator` owns the network and a list of TVB monitors.  After
# `sim.configure()`, `sim.run(...)` advances the system for `simulation_length`
# milliseconds from either explicit `initial_conditions` or randomly drawn
# initial conditions (controlled by `random_state`).
#
# The monitor returns one `(times, data)` tuple.  `data` has shape
# `(n_time, total_vois, total_nodes, 1)` where VOIs and nodes from each
# subnetwork are **concatenated in the order they appear in `subnets`**.

# %% Assemble NetworkSet
nets = NetworkSet(
    subnets=[cortex, thalamus],   # cortex first → occupies VOI rows 0:4, nodes 0:38
    projections=[proj_ctx_to_thal],
)

# %% Configure and run simulator
tavg = TemporalAverage(period=1.0)   # record every 1 ms

sim = Simulator(
    nets=nets,
    simulation_length=simulation_length,
    monitors=[tavg],
)
sim.configure()

# Use zero initial conditions for FHN to ensure numerical stability.
# ReducedSetFitzHughNagumo has 4 state vars and 3 modes.
# JansenRit has 6 state vars and 1 mode.
# Shape convention: (n_state_vars, n_nodes, n_modes)
ic_cortex   = np.zeros((6, nnodes, 1))   # JansenRit
ic_thalamus = np.zeros((4, nnodes, 3))   # ReducedSetFitzHughNagumo (3 modes)

((t, y),) = sim.run(initial_conditions=[ic_cortex, ic_thalamus])
# shape: (n_time, total_vois, total_nodes, 1)
print("t.shape :", t.shape)
print("y.shape :", y.shape)

# %% [markdown]
# ## Extracting per-subnetwork time series
#
# Because cortex is first in `subnets`, its VOIs occupy the leading rows and
# its nodes occupy the leading columns of the output array.  We use the
# subnetwork's own VOI count to find the boundary.

# %% Slice per-subnetwork arrays
n_cortex_vois   = len(cortex.model.variables_of_interest)    # 4 (y0,y1,y2,y3)
n_thalamus_vois = len(thalamus.model.variables_of_interest)  # 2 (xi, alpha)

y_cortex   = y[:, :n_cortex_vois,   :cortex.nnodes,    0]   # shape: (1000, 4, 38)
y_thalamus = y[:, n_cortex_vois:,   cortex.nnodes:,    0]   # shape: (1000, 2, 38)

print("y_cortex.shape  :", y_cortex.shape,
      " → (time, JR_vois, cortex_nodes)")
print("y_thalamus.shape:", y_thalamus.shape,
      " → (time, FHN_vois, thalamus_nodes)")
print("VOI labels — cortex  :", list(cortex.model.variables_of_interest))
print("VOI labels — thalamus:", list(thalamus.model.variables_of_interest))

# Check for NaN
assert not np.any(np.isnan(y_cortex)),   "NaN in cortex output!"
assert not np.any(np.isnan(y_thalamus)), "NaN in thalamus output!"
print("No NaN in output — OK.")

# %% [markdown]
# ## Visualization: coupled time series
#
# We plot the mean signal over all nodes for one representative VOI in each
# subnetwork.  **JansenRit `y1`** (excitatory PSP) exhibits alpha-band
# oscillations; **FHN `xi`** oscillates under the combined influence of its
# intrinsic limit-cycle dynamics and the excitatory cortical drive.

# %% Plot coupled time series
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Cortex: JansenRit y1 (index 1 in VOI list → y1)
ctx_mean = y_cortex[:, 1, :].mean(axis=1)   # mean over 38 nodes
axes[0].plot(t, ctx_mean, linewidth=0.8, color='steelblue')
axes[0].set_title("Cortex — JansenRit y₁\n(excitatory PSP, mean over 38 nodes)")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("y₁ (mV)")
axes[0].grid(alpha=0.3)

# Thalamus: FHN xi (index 0 in VOI list)
thal_mean = y_thalamus[:, 0, :].mean(axis=1)  # mean over 38 nodes
axes[1].plot(t, thal_mean, linewidth=0.8, color='firebrick')
axes[1].set_title("Thalamus — FHN ξ\n(mean over 38 nodes)")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("ξ (a.u.)")
axes[1].grid(alpha=0.3)

fig.suptitle("Hybrid simulation: coupled cortex → thalamus", fontsize=13)
plt.tight_layout()
plt.savefig("coupled_timeseries.png", dpi=100, bbox_inches="tight")
plt.show()
print("Coupled time series plotted.")

# %% [markdown]
# ## Coupled vs uncoupled comparison
#
# To demonstrate that the projection alters thalamic dynamics, we run a second
# simulation with `projections=[]`.  Under default parameters from zero initial
# conditions, the uncoupled `ReducedSetFitzHughNagumo` thalamus already exhibits
# intrinsic limit-cycle oscillations; adding cortical `y1` drive shifts its
# baseline and modulates the oscillation amplitude.

# %% Uncoupled run
fhn_uc = ReducedSetFitzHughNagumo()
cortex_uc  = Subnetwork(name='cortex',   model=JansenRit(),
                        scheme=HeunDeterministic(dt=dt), nnodes=nnodes).configure()
thalamus_uc = Subnetwork(name='thalamus', model=fhn_uc,
                         scheme=HeunDeterministic(dt=dt), nnodes=nnodes).configure()

nets_uc = NetworkSet(subnets=[cortex_uc, thalamus_uc], projections=[])  # no projection
sim_uc  = Simulator(nets=nets_uc, simulation_length=simulation_length,
                    monitors=[TemporalAverage(period=1.0)])
sim_uc.configure()
((t_uc, y_uc),) = sim_uc.run(initial_conditions=[ic_cortex, ic_thalamus])

n_cv = len(cortex_uc.model.variables_of_interest)  # 4
y_thal_uc = y_uc[:, n_cv:, cortex_uc.nnodes:, 0]  # shape: (1000, 2, 38)

# %% Plot coupled vs uncoupled
fig, ax = plt.subplots(figsize=(10, 4))

thal_coupled   = y_thalamus[:, 0, :].mean(axis=1)
thal_uncoupled = y_thal_uc[:, 0, :].mean(axis=1)

ax.plot(t,    thal_coupled,   label='Coupled (cortex → thalamus)',  color='firebrick',  linewidth=1.0)
ax.plot(t_uc, thal_uncoupled, label='Uncoupled (no projection)',     color='gray',       linewidth=1.0, linestyle='--')

ax.set_title("Thalamus FHN ξ — coupled vs uncoupled")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("ξ (a.u.)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("coupled_vs_uncoupled.png", dpi=100, bbox_inches="tight")
plt.show()

# Print summary statistics
print(f"Coupled   thalamus ξ amplitude (last 500 ms): {thal_coupled[-500:].max() - thal_coupled[-500:].min():.4f}")
print(f"Uncoupled thalamus ξ amplitude (last 500 ms): {thal_uncoupled[-500:].max() - thal_uncoupled[-500:].min():.4f}")

# %% [markdown]
# Both conditions oscillate continuously: under default parameters from zero ICs,
# the uncoupled thalamus settles onto its larger intrinsic limit cycle
# (peak-to-peak mean `xi` amplitude ~5.79), whereas cortical drive shifts the
# operating point and compresses the coupled thalamic oscillation to ~2.17.

# %% [markdown]
# ## Reproducibility
#
# Under deterministic integration (`HeunDeterministic`), two independent
# simulator instances started from identical initial conditions must produce
# bit-for-bit identical output (`np.array_equal`).  Here we verify exact
# reproducibility both for explicit `initial_conditions` on the coupled
# `JansenRit` + `ReducedSetFitzHughNagumo` network (avoiding `JansenRit`'s
# `[-500, 500]` default `y1` sampling range) and for `random_state=42` on a
# bounded `Generic2dOscillator` subnetwork.

# %% Reproducibility check
def _build_sim(backend="python"):
    """Create a fresh coupled simulator from scratch."""
    jr  = JansenRit()
    fhn_ = ReducedSetFitzHughNagumo()
    ctx_  = Subnetwork(name='cortex',   model=jr,   scheme=HeunDeterministic(dt=dt), nnodes=nnodes).configure()
    thal_ = Subnetwork(name='thalamus', model=fhn_, scheme=HeunDeterministic(dt=dt), nnodes=nnodes).configure()
    rng_  = np.random.RandomState(42)
    W_    = sp.csr_matrix(np.abs(rng_.randn(nnodes, nnodes)) * 0.01)
    L_    = sp.csr_matrix((nnodes, nnodes))
    proj_ = InterProjection(source=ctx_, target=thal_, source_cvar='y1', target_cvar='xi',
                             weights=W_, lengths=L_, cv=7.0, dt=dt, scale=0.1)
    nets_ = NetworkSet(subnets=[ctx_, thal_], projections=[proj_])
    s_    = Simulator(nets=nets_, simulation_length=simulation_length,
                      monitors=[TemporalAverage(period=1.0)], backend=backend)
    s_.configure()
    return s_

sim_r1 = _build_sim()
sim_r2 = _build_sim()

# 1. Explicit initial conditions on the hybrid cortex-thalamus simulator
ic_r = [np.zeros((6, nnodes, 1)), np.zeros((4, nnodes, 3))]
((_, y_r1),) = sim_r1.run(initial_conditions=ic_r)
((_, y_r2),) = sim_r2.run(initial_conditions=ic_r)

assert np.array_equal(y_r1, y_r2), "Expected bit-identical output for same initial conditions!"
print("\u2713 Reproducibility confirmed (explicit ICs): two independent runs are bit-identical.")
print(f"  max |y_r1 - y_r2| = {np.abs(y_r1 - y_r2).max():.2e}")

# 2. Seeded random_state=42 on a bounded Generic2dOscillator simulator
from tvb.simulator.models import Generic2dOscillator

def _build_g2d_sim():
    sn_ = Subnetwork(name='g2d', model=Generic2dOscillator(),
                     scheme=HeunDeterministic(dt=dt), nnodes=10).configure()
    return Simulator(nets=NetworkSet(subnets=[sn_], projections=[]),
                     simulation_length=100.0, monitors=[TemporalAverage(period=1.0)])

sim_g1 = _build_g2d_sim()
sim_g2 = _build_g2d_sim()
((_, y_g1),) = sim_g1.run(random_state=42)
((_, y_g2),) = sim_g2.run(random_state=42)
assert np.array_equal(y_g1, y_g2), "Expected bit-identical output for same random_state!"
print("\u2713 Reproducibility confirmed (random_state=42): two independent runs are bit-identical.")

# %% [markdown]
# ## Compiled C++ backend cross-check
#
# Passing `backend="cpp"` to `Simulator` runs the exact same `NetworkSet`
# through the compiled SIMD C++ backend (single-precision `float32` state
# updates vs double-precision `float64` in pure Python).

# %% C++ backend cross-check
import time

t0_cpp = time.perf_counter()
sim_cpp = _build_sim(backend="cpp")
((_, y_cpp),) = sim_cpp.run(initial_conditions=ic_r)
dt_cpp = time.perf_counter() - t0_cpp

early_diff_cpp = np.abs(y[:100] - y_cpp[:100]).max()
ctx_diff_cpp = np.abs(y[:, :n_cortex_vois, :cortex.nnodes] - y_cpp[:, :n_cortex_vois, :cortex.nnodes]).max()
thal_cpp_mean = y_cpp[:, n_cortex_vois, cortex.nnodes:, 0].mean(axis=1)
thal_cpp_amp = thal_cpp_mean[-500:].max() - thal_cpp_mean[-500:].min()
print(f"C++ backend finished in {dt_cpp:.3f}s")
print(f"  max diff (first 100 ms): {early_diff_cpp:.2e} | max cortex diff (1000 ms): {ctx_diff_cpp:.2e}")
print(f"  coupled thalamus ξ amplitude (last 500 ms): {thal_cpp_amp:.4f} (Python: {thal_coupled[-500:].max() - thal_coupled[-500:].min():.4f})")
assert not np.any(np.isnan(y_cpp)), "NaN in C++ backend output!"

# %% [markdown]
# ## Summary
#
# In this notebook we:
#
# 1. Built two subnetworks — **cortex** (JansenRit, 38 nodes) and
#    **thalamus** (ReducedSetFitzHughNagumo, default 3 modes, 38 nodes).
# 2. Wired them with an `InterProjection` using named coupling variables
#    (`source_cvar='y1'`, `target_cvar='xi'`).
# 3. Assembled a `NetworkSet`, configured a `Simulator`, and ran 1000 ms of
#    simulation.
# 4. Extracted per-subnetwork arrays from the concatenated output by indexing
#    on VOI counts and node counts.
# 5. Compared coupled vs uncoupled thalamus and quantified how cortical drive
#    modulates the thalamus's intrinsic limit-cycle oscillations (~5.79
#    uncoupled vs ~2.17 coupled peak-to-peak mean `xi` amplitude).
# 6. Verified bit-identical reproducibility for both explicit `initial_conditions`
#    and `random_state=42`, and cross-checked `Simulator(..., backend="cpp")`.
#
# **Companion notebooks**
#
# - `simulate_hybrid_stimuli_and_advanced.ipynb` — External stimulation,
#   per-subnetwork monitors, connectome-ordered output, and structural wiring.
# - `simulate_hybrid_stochastic_and_bold.ipynb` — Stochastic integration and
#   the BOLD monitor.
# - `simulate_hybrid_parameter_sweeps.ipynb` / `cpp_hybrid_sweep.ipynb` —
#   Multi-core and SIMD parameter sweeps.
# - `simulate_hybrid_multi_dt.ipynb` — Per-subnetwork integration time steps
#   (multi-dt).
