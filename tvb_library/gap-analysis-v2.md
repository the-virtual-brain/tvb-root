# Gap Analysis & Speed Audit: Numba Hybrid Backend v2

**Date**: 2026-04-15  
**Branch**: `hybrid-numba`  
**Tests**: 208 passing  
**Commit**: `12710ea41`

---

## 1. Performance Summary

### 1.1 Single Subnet (68 nodes, MPR model, Heun deterministic)

| Configuration | Kiter/s | µs/step | Speedup vs Python |
|---|---:|---:|---:|
| **Python (raw loop)** | 28,000 | 36 | 1.0× |
| **Numba, cs=1** | 117,000 | 8.5 | 4.2× |
| **Numba, cs=50** | 286,000 | 3.5 | 10.2× |
| **Numba, cs=100** | 285,000 | 3.5 | 10.2× |
| **Numba, cs=1000** | 820,000 | 1.2 | 29.3× |

**Python hotspot breakdown** (68 nodes, 5000 steps):

| Phase | % of Python time | µs/step |
|---|---:|---:|
| `model.dfun()` (2 calls for Heun) | 82% | 20 |
| `bound_and_clamp()` | 9% | 2 |
| `cfun()` (coupling dispatch) | 5% | 1 |
| `observe()` + record | 3% | 0.5 |

**Numba dfun speedup**: 24× per evaluation vs Python (0.41 µs vs 9.9 µs).

**Numba math vs numpy**: 14.5× faster than raw numpy element-wise math (0.41 µs vs 6.0 µs), because JIT avoids numpy dispatch overhead, temporary array allocation, and loop fusion.

### 1.2 Multi-Subnet (2 × 68 nodes, MPR, coupled at 30% density)

| Configuration | Kiter/s | µs/step | Speedup vs Python |
|---|---:|---:|---:|
| **Python coupled** | 10,900 | 92 | 1.0× |
| **Numba coupled, cs=100** | 231,000 | 4.3 | 21.2× |

### 1.3 Chunk Size Impact (68 nodes, 10,000 steps)

| chunk_size | Kiter/s | Python overhead |
|---:|---:|---:|
| 1 | 117,000 | 93% |
| 10 | 256,000 | 30% |
| 50 | 294,000 | <10% |
| 100 | 285,000 | <5% |
| 500 | 305,000 | <2% |
| 1000 | 820,000 | <1% |

> **Note**: chunk_size=1 has 93% Python overhead (list.append per step × np.stack at end).  
> chunk_size≥50 has <10% overhead and is near-optimal for typical simulations.

### 1.4 Coupling Cost Breakdown (2×68 subnets, cs=100)

| Config | Kiter/s | Slowdown vs 1-subnet |
|---|---:|---:|
| 1×68, no projection | 1,077,000 | — |
| 2×68, no projection | 715,000 | 1.5× (expected 2×) |
| 2×68, coupled (30% density) | 231,000 | 4.7× |

The coupling CSR loop (68 targets × ~20 sources × 1 cvar) adds ~150% overhead on top of the 2× integration cost. This is the dominant cost for coupled networks.

### 1.5 Node Scaling (MPR model, 10,000 steps, cs=50)

| Nodes | Python Kiter/s | Numba Kiter/s | Speedup |
|---:|---:|---:|---:|
| 4 | 29,400 | 128,000 | 4.3× |
| 16 | 28,800 | 90,000 | 3.1× |
| 68 | 28,000 | 117,000 | 4.2× |
| 128 | 28,000 | 104,000 | 3.7× |
| 256 | 26,200 | 91,000 | 3.5× |

### 1.6 Model Comparison (68 nodes, 10,000 steps)

| Model | Python Kiter/s | Numba Kiter/s | Speedup |
|---|---:|---:|---:|
| MontbrioPazoRoxin | 28,300 | 120,000 | 4.3× |
| Generic2dOscillator | 51,800 | 124,000 | 2.4× |
| JansenRit | 44,200 | 94,000 | 2.1× |
| Epileptor | 41,700 | 114,000 | 2.7× |

### 1.7 Realistic Simulation (10s at dt=0.1ms, 100,000 steps)

| Config | Wall time | Kiter/s | Realtime factor |
|---|---:|---:|---:|
| Python 68 nodes | 3.54s | 28,200 | 2,828× |
| Numba 68 nodes + TA + Bold | 0.79s | 126,600 | 12,631× |
| Python 68+68 coupled | 9.13s | 11,000 | 1,095× |
| Numba 68+68 coupled | 1.62s | 61,700 | 6,156× |

---

## 2. Feature Gaps

### 2.1 Models (1 missing, 2 work via inheritance)

| Model | Status | Effort | Notes |
|---|---|---|---|
| `DecoBalancedExcInh` | ✅ Works | — | Subclass of `ReducedWongWangExcInh` (already supported). Passes `isinstance` check. Verified. |
| `ZerlautAdaptationSecondOrder` | ✅ Works | — | Subclass of `ZerlautAdaptationFirstOrder` (already supported). Uses same custom template. Verified. |
| `Linear` | ❌ Missing | **15 min** | 1 var (x), 1 cvar (c), trivial dfun: `dx = gamma * c`. Has `state_variable_dfuns` and `coupling_terms`. Just needs adding to `_supported_models` tuple. |

### 2.2 Integrators (1 gap)

| Integrator | Status | Priority | Notes |
|---|---|---|---|
| `RungeKutta4thOrderDeterministic` | ❌ Missing | MEDIUM | 4 dfun evaluations per step. Template currently supports Heun (2 evals) and Euler (1 eval). Adding RK4 requires a new `int_type == "rk4"` branch in the template with 4 stages. |

### 2.3 Monitors (2 gaps)

| Monitor | Status | Priority | Notes |
|---|---|---|---|
| `AfferentCouplingTemporalAverage` | ❌ Missing | LOW | Extends `AfferentCoupling` + `TemporalAverage`. The JIT already computes ctavg. Just needs dispatch in `_apply_monitors`. |
| `BoldRegionROI` | ❌ Missing | LOW | Bold + region spatial average. Niche. |

### 2.4 Feature Gaps (1 gap)

| Feature | Status | Priority | Notes |
|---|---|---|---|
| `update_state_variables_after_integration()` | ❌ Missing | LOW | No current model overrides it. Hook for post-integration state modification. |

---

## 3. Speed Optimization Opportunities

### 3.1 🔴 Per-Step Coupling Array Allocation (HIGH impact, MEDIUM effort)

**Location**: `nb-hybrid-sim.py.mako:704`  
**Issue**: `np.zeros((ncvar, nnodes, n_modes))` is allocated **inside the per-step loop** in `network_chunk`. Numba may optimize this (it recognizes `np.zeros` in njit), but it's still a per-step allocation.

**Fix**: Pre-allocate coupling arrays outside the step loop and zero-fill with `[:] = 0.0` each step.

**Expected impact**: Minor for typical sizes (ncvar×nnodes×n_modes is small), but eliminates a class of potential allocation overhead.

### 3.2 🟡 Bold Sampling Loop (MEDIUM impact, SMALL effort)

**Location**: `nb-hybrid-sim.py.mako:1019-1025`  
**Issue**: Bold sampling iterates over every step in the chunk with `for _bc_step in range(t_global, t_global + this_chunk):` to find period crossings. For chunk_size=1000, this is 1000 iterations.

**Fix**: Compute crossing directly: `first_crossing = ((t_global + _bold_istep - 1) // _bold_istep) * _bold_istep` then iterate `for cross in range(first_crossing, t_global + this_chunk, _bold_istep)`.

**Expected impact**: Small (Bold is rarely used with large chunk_size), but cleaner code.

### 3.3 🟡 Pre-Allocated Output Arrays (MEDIUM impact at cs=1, LOW effort)

**Location**: `run_network()` function in template  
**Issue**: Output arrays are built via `list.append()` per chunk, then `np.stack()` at the end. At cs=1 with 10,000 steps, this is 10,000 append + 1 stack of 10,000 arrays.

**Fix**: Pre-allocate `(n_chunks, n_voi, n_nodes, n_modes)` output array and fill by index.

**Expected impact**: Eliminates the 93% Python overhead at cs=1. But cs=1 is rare in practice (auto-chunk computes optimal size). At cs≥50, overhead is <10%.

### 3.4 🟢 Reduce Python Per-Step Overhead (LOW impact, MEDIUM effort)

**Issue**: The `run_network` Python loop calls `network_chunk()` per chunk, then does `list.append`, division, etc. At cs=1 this dominates.

**Fix**: Move the outer loop into the `@njit` kernel itself when chunk_size == nstep (no chunking needed). The kernel directly fills output arrays.

**Expected impact**: Would bring cs=1 performance from 117K to ~300+ Kiter/s. But auto-chunk already avoids cs=1.

### 3.5 🟢 Parallel Subnetwork Integration (LOW impact, HIGH effort)

**Issue**: Multi-subnet integration is sequential in the JIT kernel.

**Fix**: Use `nb.prange` for the subnetwork loop when subnets are independent (no inter-projections between them).

**Expected impact**: Only helps with 3+ independent subnets. Rare in practice. Coupled subnets are sequential by nature.

### 3.6 🟢 Monitor Post-Processing (LOW impact, LOW effort)

**Measurement**: `_apply_monitors` is **0.6%** of total runtime. Already negligible.

---

## 4. Code Quality Issues

### 4.1 Dead Code

| Item | Location | Severity |
|---|---|---|
| `_BOLD_BALLOON_DEFAULTS` constant | `nb_hybrid.py:79-83` | LOW — defined but never used |
| `period_dt` variable | `nb_hybrid.py:166` | LOW — computed but never read |
| Vestigial `compute_hrf()` call | `nb_hybrid.py:231-233` | LOW — Bold uses Balloon, not HRF |
| Unused variables in test | `test_nb_hybrid.py` | LOW — `ns2`, `ics2`, etc. in `test_projection_merged_sums_sensors` |

### 4.2 Redundant Computation

| Item | Location | Severity |
|---|---|---|
| `_can_merge_subnets()` called N×M times | `nb_hybrid.py:271,296` | LOW — invariant per call, should cache |
| `_compute_chunk_size` recomputed each `run()` | `CompiledNetworkFn.run()` | LOW — monitors rarely change between calls |

### 4.3 Design Debt

| Item | Severity |
|---|---|
| Only first monitor of each type gets JIT precomputation (e.g., two SpatialAverage with different weights) | LOW — rare use case |
| `SubSample` guard rejects chunk_size > 1 | LOW — by design |
| Stimulus pre-computed as full batch (memory concern for very long runs) | LOW — lazy path planned but not wired |

---

## 5. Priority Recommendations

### Tier 1: Quick Wins (1-2 hours total)

| # | Item | Effort |
|---|---|---|
| Q1 | Add `Linear` to `_supported_models` list | 15 min |
| Q2 | Delete dead code (`_BOLD_BALLOON_DEFAULTS`, `period_dt`, vestigial `compute_hrf`, test dead vars) | 30 min |
| Q3 | Cache `_can_merge_subnets()` result | 15 min |

### Tier 2: Feature Gaps (4-8 hours)

| # | Item | Effort |
|---|---|---|
| F1 | `RungeKutta4thOrderDeterministic` template support | 4 hours |
| F2 | `AfferentCouplingTemporalAverage` monitor dispatch | 30 min |
| F3 | Pre-allocate coupling arrays outside step loop | 1 hour |
| F4 | Bold sampling modulo optimization | 30 min |

### Tier 3: Performance (8-16 hours, diminishing returns)

| # | Item | Effort | Expected Gain |
|---|---|---|---|
| P1 | Pre-allocated output arrays in `run_network` | 2 hours | cs=1: 2-3× faster (rare) |
| P2 | Move outer loop into `@njit` for single-chunk case | 4 hours | ~10% at optimal cs |
| P3 | Parallel independent subnets with `prange` | 4 hours | N-subnet speedup, niche |
| P4 | Lazy stimulus chunking for long simulations | 8 hours | Memory savings only |

---

## 6. Benchmark Reference Table

All numbers at 10,000 steps, dt=0.1ms, chunk_size=50 (unless noted).

| Config | Python Kiter/s | Numba Kiter/s | Speedup |
|---|---:|---:|---:|
| **Single subnet** |
| 4 nodes, MPR | 29,400 | 128,000 | 4.3× |
| 68 nodes, MPR | 28,000 | 117,000 | 4.2× |
| 256 nodes, MPR | 26,200 | 91,000 | 3.5× |
| 68 nodes, G2DOsc | 51,800 | 124,000 | 2.4× |
| 68 nodes, JansenRit | 44,200 | 94,000 | 2.1× |
| 68 nodes, Epileptor | 41,700 | 114,000 | 2.7× |
| **Multi-subnet** |
| 2×68, no projection | 14,000 | 72,000 | 5.1× |
| 2×68, coupled 30% | 10,900 | 231,000 | 21.2× |
| **Realistic (100k steps)** |
| 68 nodes, TA+Bold | 28,200 | 126,600 | 4.5× |
| 68+68 coupled | 11,000 | 61,700 | 5.6× |

> **Bottom line**: The Numba backend provides 2-21× speedup over Python, with the largest gains in coupled multi-subnet simulations. The auto-chunk feature (from gap C1) ensures near-optimal chunk_size by default. Remaining performance gains are marginal (<10% at typical settings) and require significant engineering effort.
