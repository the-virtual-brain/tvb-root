# Gap Analysis: Numba Hybrid Backend vs TVB Hybrid Simulator

**Date**: 2026-04-14
**Status**: Working document
**Branch**: `hybrid-numba`
**Tests**: 189 passing

---

## 1. What the Hybrid Simulator Does

The TVB hybrid simulator (`tvb/simulator/hybrid/`) is a Python framework for simulating brain networks composed of multiple subnetworks, each with its own dynamical model and integrator, coupled through delayed sparse projections.

### 1.1 Core Architecture

| Component | File | Role |
|-----------|------|------|
| `Simulator` | `hybrid/simulator.py` | Top-level: owns NetworkSet + monitors, drives time loop |
| `NetworkSet` | `hybrid/network.py` | Container: subnetworks + projections + stimuli; orchestrates stepping |
| `Subnetwork` | `hybrid/subnetwork.py` | Model + integrator + intra-projections + per-subnet monitors |
| `InterProjection` | `hybrid/inter_projection.py` | Delayed sparse coupling between different subnetworks |
| `IntraProjection` | `hybrid/intra_projection.py` | Delayed sparse coupling within a subnetwork |
| `BaseProjection` | `hybrid/base_projection.py` | Shared: CSR weights, idelays, circular history buffer, cfun pipeline |
| `Stim` | `hybrid/stimulus.py` | External stimulus injected through the coupling interface |
| `Recorder` | `hybrid/recorder.py` | Wraps TVB Monitor for in-memory recording |

### 1.2 Time Stepping (Simulator.run)

```
for step in 1..nstep:
    1. NetworkSet.cfun(step, xs) → compute inter-subnet coupling from delayed buffers
    2. For each subnet:
       a. Subnet.cfun(step, x) → compute intra-subnet coupling from delayed buffers
       b. total_c = inter_c + intra_c
       c. scheme.scheme(x, model.dfun, total_c, 0.0, 0.0) → integrate
       d. scheme.bound_and_clamp(nx)
       e. Recorder.record(step, model.observe(nx)) for each per-subnet monitor
    3. NetworkSet.observe(xs, flat=True) → flatten to (total_voi, total_nodes, modes)
    4. Monitor.record(step, ox) for each global monitor
    5. Update all projection buffers with new states
```

### 1.3 Integrators Supported

| Integrator | Type | Notes |
|------------|------|-------|
| `HeunDeterministic` | Predictor-corrector | Two dfun evaluations |
| `HeunStochastic` | Stochastic predictor-corrector | Noise via `noise.generate()` + `noise.gfun()` |
| `EulerDeterministic` | Forward Euler | Single dfun evaluation |
| `EulerStochastic` | Stochastic Euler-Maruyama | Noise via `noise.generate()` + `noise.gfun()` |
| `RungeKutta4thOrderDeterministic` | RK4 | 4 dfun evaluations |
| `Identity` / `IdentityStochastic` | No integration | Pass-through |
| SciPy ODE variants | `VODE`, `Dopri5`, `Dop853` | External solver |

**Key detail**: All integrators call `bound_and_clamp()` which applies `state_variable_boundaries` and `clamped_state_variable_values` from the model. They also call `model.update_state_variables_after_integration()` (currently a no-op for all models).

### 1.4 Coupling Functions

| Coupling | Pipeline Position | Formula |
|----------|-------------------|---------|
| `Linear` | post | `a * x + b` |
| `Scaling` | post | `a * x` |
| `Sigmoidal` | post | `cmax / (1 + exp(-a*(x-midpoint))) + cmin` (clamped) |
| `SigmoidalJansenRit` | pre | `e0 / (1 + exp(r*(v0 - x)))` applied to source in CSR loop |
| `Kuramoto` | post | `a * sin(x)` |
| `Difference` | post | `a * x` (same as Scaling) |
| `HyperbolicTangent` | post | `a * tanh(sigma * (x - midpoint))` |
| `PreSigmoidal` | post | Multi-parameter sigmoid with H, Q, G, P, theta |

All coupling functions have `pre()` and `post()` hooks bracketing the scalar `scale` factor.

### 1.5 Monitor Types

| Monitor | Mechanism | Output Shape |
|---------|-----------|--------------|
| `Raw` | Records every step | `(n, voi, nodes, modes)` |
| `SubSample` | Decimates by period | `(n, voi, nodes, modes)` |
| `TemporalAverage` | Running average over istep | `(n, voi, nodes, modes)` |
| `SpatialAverage` | `spatial_mean @ state` at period | `(n, voi, areas, modes)` |
| `GlobalAverage` | `mean(nodes)` at period | `(n, voi, 1, modes)` |
| `AfferentCoupling` | Raw coupling variables | `(n, cvar, nodes, modes)` |
| `AfferentCouplingTemporalAverage` | Averaged coupling variables | `(n, cvar, nodes, modes)` |
| `Projection` (EEG/MEG/iEEG) | `gain @ state` accumulated, sampled at period | `(n, voi, sensors, 1)` |
| `Bold` | HRF convolution with interim/stock buffers | `(n, voi, nodes, modes)` |
| `BoldRegionROI` | Bold + region spatial average | `(n, voi, regions, modes)` |
| `ProgressLogger` | Console logging | N/A |
| `RawVoi` | Raw with voi selection | `(n, voi, nodes, modes)` |

**Key detail**: The hybrid `Simulator.run()` calls `NetworkSet.observe(xs, flat=True)` which does `model.observe(x).sum(axis=-1)[..., None]` — modes are SUMMED and a singleton mode dim is added back.

### 1.6 Observe Function

`model.observe(state)` returns the state variables of interest. Some models have derived VOIs:
- `Epileptor`: `x2 - x1` (difference of two state variables)
- `EpileptorRestingState`: `x2 - x1`

The hybrid simulator sums modes after observe: `.sum(axis=-1)[..., None]`.

### 1.7 Stimulus

`Stim` wraps `SpatioTemporalPattern` and delivers coupling through the projection interface. Stimulus is computed step-by-step in Python via `stim.get_coupling(step)`, with optional spatial weights.

### 1.8 NetworkSet.observe — Merged Mode

When every subnetwork has `node_indices` set and all have the same voi count, `observe(flat=True)` places each subnet's output at original connectome positions rather than concatenating. This is for global monitors.

### 1.9 Projection Features

| Feature | Description |
|---------|-------------|
| CSR sparse weights | Per-connection weights |
| Per-connection delays | `idelays[k]` from `lengths / cv / dt` |
| Circular history buffer | `(nvar, nodes, modes, horizon)` with `t % horizon` indexing |
| Delay convention | `t - 1 - idelays[k]` (reads state from step t-1 minus delay) |
| `target_scales` | Per-target-cvar multiplicative scaling after mode mapping |
| `mode_map` | Source→target mode transformation matrix (inter-projections only) |
| `scale` | Global projection scaling factor |
| Epsilon trick | Adds `2*eps` to column 0 of every row for uniform CSR, then zeros it |

### 1.10 Boundary Conditions

Models declare `state_variable_boundaries` as `{sv_name: [lo, hi]}`. The integrator clamps each state variable to its bounds after each step. Examples:
- MPR: `r >= 0`
- ReducedWongWang: `S ∈ [0, 1]`
- Zerlaut: `E >= 0, I >= 0`
- KIonEx: `x >= 0, V >= -500`, etc.

---

## 2. What the Numba Backend Currently Implements

### 2.1 Supported Models (24)

| Model | File | nvar | Boundaries | Special |
|-------|------|------|------------|---------|
| MontbrioPazoRoxin | infinite_theta.py | 2 | r≥0 | |
| KIonEx | k_ion_exchange.py | 5 | x≥0, V≥-500, etc. | |
| JansenRit | jansen_rit.py | 6 | — | dfun_helpers |
| ZetterbergJansen | jansen_rit.py | 12 | — | dfun_helpers |
| Generic2dOscillator | oscillator.py | 2 | — | |
| SupHopf | oscillator.py | 2 | — | |
| Kuramoto | oscillator.py | 1 | — | sin-based |
| ReducedWongWang | wong_wang.py | 1 | S∈[0,1] | |
| ReducedWongWangExcInh | wong_wang_exc_inh.py | 2 | S∈[0,1] | |
| Epileptor | epileptor.py | 6 | — | Derived voi: x2-x1 |
| Epileptor2D | epileptor.py | 2 | — | |
| EpileptorCodim3 | epileptorcodim3.py | 3 | — | |
| EpileptorCodim3SlowMod | epileptorcodim3.py | 5 | — | |
| EpileptorRestingState | epileptor_rs.py | 8 | — | Derived voi: x2-x1 |
| WilsonCowan | wilson_cowan.py | 2 | — | shift_sigmoid required |
| Hopfield | hopfield.py | 2 | — | |
| LarterBreakspear | larter_breakspear.py | 3 | — | |
| CoombesByrne | infinite_theta.py | 4 | r≥0 | |
| CoombesByrne2D | infinite_theta.py | 2 | r≥0 | |
| GastSchmidtKnosche_SD | infinite_theta.py | 4 | r≥0 | |
| GastSchmidtKnosche_SF | infinite_theta.py | 4 | r≥0 | |
| DumontGutkin | infinite_theta.py | 8 | r_e≥0, r_i≥0 | |
| ZerlautAdaptationFirstOrder | zerlaut.py | 5 | E≥0, I≥0 | Custom template |
| ReducedSetFitzHughNagumo | stefanescu_jirsa.py | 4 | — | combined-mode dfun |
| ReducedSetHindmarshRose | stefanescu_jirsa.py | 6 | — | combined-mode dfun |

### 2.2 Supported Integrators (4)

- HeunDeterministic ✅
- EulerDeterministic ✅
- HeunStochastic ✅
- EulerStochastic ✅

### 2.3 Supported Coupling Functions (8)

- Linear, Scaling, Sigmoidal, SigmoidalJansenRit, Kuramoto, Difference, HyperbolicTangent, PreSigmoidal ✅

### 2.4 Supported Monitors (8 JIT + Python)

| Monitor | Where Processed | Notes |
|---------|----------------|-------|
| TemporalAverage | JIT (tavg accumulator) | Default output |
| Raw | JIT (chunk_size=1) | Passthrough |
| AfferentCoupling | JIT (ctavg accumulator) | Coupling average |
| SpatialAverage | JIT (spatial_tavg accumulator) | ✅ |
| Projection (EEG/MEG/iEEG) | JIT (proj_tavg accumulator) | ✅ |
| Bold | JIT (Balloon ODE) | Replaced HRF with Balloon model |
| SubSample | Python post-processing | Step mask |
| GlobalAverage | Python post-processing | Mean over nodes |

### 2.5 Features Implemented

- **Per-node heterogeneous parameters**: Spatial params passed as `(n_params, n_nodes)` arrays, indexed by node in dfun ✅
- **Stimulus**: Pre-computed batch arrays ✅
- **Inter-projections**: Delayed sparse CSR with mode_map ✅
- **Intra-projections**: Same template path as inter ✅
- **Boundaries**: Inline after integration ✅
- **Clamping**: Not yet (see gaps)
- **Disk-persistent JIT cache** ✅
- **Checkpointing / resume** ✅
- **Combined-mode dfun** (ReducedSet models) ✅
- **Zerlaut custom template** ✅
- **Bold Balloon ODE in JIT** ✅
- **Bold state persistence across run() calls** ✅
- **debug_nojit path** ✅

---

## 3. What is MISSING

### 3.1 Models Not Supported (3)

| Model | nvar | Issue |
|-------|------|-------|
| `DecoBalancedExcInh` | 2 | No `parameter_names` / `state_variable_dfuns` codegen attrs. Has boundaries: S_e∈[0,1], S_i∈[0,1]. |
| `Linear` | 1 | No codegen attrs. Trivial model: `dx/dt = coupling`. |
| `ZerlautAdaptationSecondOrder` | 8 | Not in supported list. Is a subclass of ZerlautFirstOrder, may or may not work with Zerlaut template. |

### 3.2 Integrators Not Supported (4+)

| Integrator | Priority | Notes |
|------------|----------|-------|
| `RungeKutta4thOrderDeterministic` | MEDIUM | 4 dfun evaluations per step. Used in some accuracy-sensitive simulations. |
| `Identity` / `IdentityStochastic` | LOW | No integration — pass-through. Uncommon. |
| SciPy ODE variants (`VODE`, `Dopri5`, `Dop853`) | SKIP | External solver, incompatible with JIT loop. |

### 3.3 Monitors Not Supported (5)

| Monitor | Priority | Notes |
|---------|----------|-------|
| `AfferentCouplingTemporalAverage` | MEDIUM | Hybrid uses this via `Recorder`. Requires ctavg temporal averaging (already computed in JIT but not dispatched as this type). |
| `BoldRegionROI` | LOW | Bold + region spatial average. Niche. |
| `ProgressLogger` | SKIP | Console only. |
| `RawVoi` | LOW | Subclass of Raw with voi selection — easy once Raw works. |
| `TemporalAverage` stock-based averaging | MEDIUM | **See §4.1 below — behavioral difference.** |

### 3.4 Observe / Mode Summation — Semantic Gap

**CRITICAL**: The hybrid simulator's `NetworkSet.observe()` does:

```python
sn.model.observe(x).sum(axis=-1)[..., None]
```

This **sums over modes** and adds a singleton mode dimension. The Numba backend's tavg accumulator does NOT sum modes — it stores each mode independently:

```python
tavg[vi, ni, mi] += state[voi_idx, ni, mi]
```

**Impact**: For multi-mode simulations, the hybrid simulator's monitor output has `modes=1` (summed), while the Numba backend preserves the original mode dimension. This means the Numba backend's output shape differs from the hybrid simulator when `n_modes > 1`.

**Severity**: MEDIUM — most hybrid simulations use `n_modes=1`. But for `ReducedSetFitzHughNagumo` (n_modes=3), the output shape is wrong compared to the hybrid simulator.

### 3.5 Clamped State Variables

The hybrid simulator's integrator calls `scheme.bound_and_clamp(nx)` which applies both boundaries AND clamping. The Numba template applies inline boundaries but does NOT apply clamping:

```python
# Clamping: X[clamped_indices] = clamped_values
```

No current model in the supported list uses clamping, so this is LOW priority but technically missing.

### 3.6 `node_indices` / Merged Mode Observe

The hybrid simulator supports `node_indices` on subnetworks for global monitor output in original connectome order. The Numba backend has no concept of `node_indices` — each subnet's output is independent. For multi-subnet simulations with global monitors, the output ordering differs.

**Severity**: MEDIUM — affects `GlobalAverage` and `SpatialAverage` with multi-subnet setups.

### 3.7 Per-Subnet Monitors (Recorder)

The hybrid simulator supports per-subnet monitors via `Subnetwork.monitors` (a list of `Recorder` objects). Each subnet can have different monitors attached. The Numba backend only accepts a single `monitors` list for all subnets.

**Severity**: LOW — uncommon usage pattern.

### 3.8 Stimulus Weights Matrix

The hybrid `Stim` supports an optional `weights` CSR matrix for spatial weighting of stimulus. The Numba backend pre-computes `stim.get_coupling(step)` which already applies the weights, so this is actually handled correctly.

### 3.9 `update_state_variables_before_integration` Hook

The hybrid simulator calls `model.update_state_variables_before_integration(state, coupling, local_coupling, stimulus)` before integration. Currently a no-op for all models. The Numba backend skips this entirely.

**Severity**: LOW — no model overrides it.

---

## 4. What is INCORRECT

### 4.1 TemporalAverage Averaging Semantics — STOCK vs CHUNK

**CRITICAL**: The hybrid simulator's `TemporalAverage` monitor accumulates state in a `_stock` buffer of shape `(istep, voi, nodes, modes)` and averages over `istep` steps. The midpoint time is `(step - istep/2) * dt`.

The Numba backend uses chunk-based averaging: accumulate for `chunk_size` steps, divide by count, assign midpoint time. When `chunk_size == istep`, the behavior matches. When they differ, the Numba backend produces different temporal averages.

**Current behavior**: The Numba backend's default output IS the temporal average (one chunk = one average). When `TemporalAverage(period=P)` is used, `_apply_monitors` returns `data` directly, assuming `chunk_size` matches the period. This is correct when the user sets the right chunk_size, but there's no automatic alignment.

### 4.2 Bold Model — Different Algorithm

The hybrid simulator uses HRF convolution (`Bold.sample()`) with interim stock buffers. The Numba backend replaced this with a Balloon ODE model. Both produce BOLD-like signals but with different numerical values and dynamics.

**Status**: By design — the Balloon model is a deliberate replacement. Not a bug. But users expecting HRF-convolution BOLD will get different results.

### 4.3 Projection Monitor — Accumulation Semantics

The hybrid simulator's `Projection.sample()` accumulates `gain @ state` over the period and divides by `_period_in_steps`. The Numba JIT accumulates `gain[sensor, node] * voi_value` at each step and divides by chunk step count. When `chunk_size != period_steps`, these differ.

### 4.4 Delay Convention — t-1-idelays vs t-idelays

The hybrid simulator reads: `(t - 1 - idelays[k] + horizon) % horizon`.
The Numba template reads: `(t - 1 - idelays[ptr] + horizon) % horizon`.

These match ✅.

### 4.5 Stimulus Timing — Off-by-One

The hybrid simulator applies stimulus at step `step` in `NetworkSet.cfun()` using `stim.get_coupling(step)`. The Numba backend pre-computes `stim.get_coupling(step_idx)` for `step_idx = 1..nstep` and the JIT reads `stim[..., t - 1]` where `t` is 1-based.

Need to verify: does `stim.get_coupling(step)` with step=1 match `stim_arr[..., 0]`? Let me check...

The backend code at line 865: `for step_idx in range(1, nstep + 1)` and `stim_arr[..., step_idx - 1]`. In the template, `stim[..., t - 1]` where `t` starts at `t_global` (0-based). So at t_global=0, it reads `stim[..., -1]` which wraps to the last element. This is only correct if t_global starts at 1 for the first step.

**Looking at the template**: `t_global` is passed as the starting step (0-based from `run_network`). Inside `network_chunk`, `t` goes from `t_start` to `t_start + nstep - 1`. The stimulus is read as `stim[cv, ni, :, t - 1]`. So at `t = 1` (first step), it reads `stim[..., 0]`. This should be correct if the pre-computed array starts at step 1.

**Potential issue**: The coupling delay uses `(t - 1 - idelays[ptr] + horizon) % horizon`. At `t = 1` with `idelays[0] = 0`, this gives `t - 1 - 0 = 0`, reading buffer slot 0 (initial state). Correct.

### 4.6 Observe Order — State Variables vs Variables of Interest

The hybrid simulator's `model.observe(state)` returns variables in the order of `model.variables_of_interest`. The Numba backend uses `voi_idx` to index `state[voi_idx, ni, mi]`. Both should produce the same ordering since `voi_idx` maps voi names to state variable indices.

**But**: Some models have derived VOIs (e.g., Epileptor's `x2 - x1`). The Numba template handles these with special `voi_exprs` that compute the expression inline. The hybrid simulator's `observe()` does the same via exec'd code. These should match.

---

## 5. What is EXTRA (in Numba, not in Hybrid Simulator)

### 5.1 Balloon ODE Bold Model

The Numba backend replaced HRF convolution with a Balloon ODE model (4 ODEs: s, f, v, q). This is NOT in the hybrid simulator — the hybrid simulator uses the standard TVB `Bold` monitor with HRF convolution.

### 5.2 Bold State Persistence

`CompiledNetworkFn._bold_states` persists Balloon state across `run()` calls. The hybrid simulator's `Bold` monitor already handles state internally.

### 5.3 Chunk-Based Processing

The Numba backend processes simulation in chunks (`chunk_size` parameter). The hybrid simulator processes one step at a time. Chunk-based processing enables JIT amortization but changes the averaging semantics for monitors.

### 5.4 Disk-Persistent JIT Cache

Numba-specific optimization — the hybrid simulator doesn't need this since it runs in Python.

### 5.5 Checkpointing / Resume API

`CompiledNetworkFn.run(return_snapshot=True)` + `resume(snapshot, nstep)`. The hybrid simulator doesn't have this API — users just pickle the state manually.

### 5.6 debug_nojit Path

Numba-specific debugging feature.

### 5.7 Pre-computed Stimulus Arrays

The Numba backend pre-computes all stimulus steps into a single `(cvar, nodes, modes, nstep)` array. The hybrid simulator computes stimulus step-by-step.

---

## 6. Priority Order for Closing the Gaps

### Tier 1: Correctness (should fix next)

| # | Gap | Effort | Impact |
|---|-----|--------|--------|
| **C1** | **TemporalAverage period alignment**: Auto-set chunk_size from TemporalAverage period | Medium | Currently silent wrong output if chunk_size ≠ period/dt |
| **C2** | ~~Observe mode summation~~ **FIXED** | | |
| **C3** | ~~TemporalAverage stock-based averaging~~ **Fixed by C1** (auto-chunk makes chunk==istep) | | |
| **C4** | ~~Projection monitor period alignment~~ **FIXED** | | |
| **C5** | ~~SpatialAverage period alignment~~ **FIXED** | | |

### Tier 2: Feature Parity (important for production use)

| # | Gap | Effort | Impact |
|---|-----|--------|--------|
| **P1** | **node_indices / merged mode**: Support connectome-ordered global monitor output | Medium | Required for multi-subnet + global monitor workflows |
| **P2** | **AfferentCouplingTemporalAverage monitor**: Dispatch ctavg as temporal average | Small | Used in some hybrid workflows |
| **P3** | **RungeKutta4thOrderDeterministic**: Add RK4 integrator to template | Medium | Used for accuracy-sensitive simulations |
| **P4** | **Missing models** (DecoBalancedExcInh, Linear, ZerlautSecondOrder): Add codegen attrs | Small each | Completeness |
| **P5** | **Clamped state variables**: Apply clamping after integration | Small | No current model needs it, but API completeness |

### Tier 3: Nice to Have (low priority)

| # | Gap | Effort | Impact |
|---|-----|--------|--------|
| **N1** | **Per-subnet monitors**: Accept different monitors per subnet | Medium | Uncommon pattern |
| **N2** | **BoldRegionROI monitor**: Bold + region spatial average | Small | Niche |
| **N3** | **RawVoi monitor**: voi selection on raw output | Small | Easy once Raw works |
| **N4** | **Stimulus lazy chunking**: Compute stimulus per-chunk instead of pre-computed batch | Medium | Memory savings for long simulations |
| **N5** | **update_state_variables_before/after hooks**: Add callback mechanism | Small | No model uses it currently |
| **N6** | **_BOLD_BALLOON_DEFAULTS cleanup**: Remove unused constant | Tiny | Code quality |

---

## Appendix A: Detailed Correctness Matrix

| Feature | Hybrid Simulator | Numba Backend | Match? |
|---------|-----------------|---------------|--------|
| Heun deterministic | `scheme(X, dfun, c, 0, 0)` with bound_and_clamp | Inline Heun + inline boundaries | ✅ (verified by tests) |
| Euler deterministic | `scheme(X, dfun, c, 0, 0)` with bound_and_clamp | Inline Euler + inline boundaries | ✅ |
| Heun stochastic | noise.generate() + gfun * noise | Pre-drawn noise + noise_nsig * noise | ✅ |
| Euler stochastic | noise.generate() + gfun * noise | Pre-drawn noise + noise_nsig * noise | ✅ |
| Delay convention | `t - 1 - idelays[k]` | `t - 1 - idelays[ptr]` | ✅ |
| Buffer write | Post-integration state | Post-integration state | ✅ |
| CSR coupling | `np.add.reduceat` | Manual loop over CSR | ✅ (different numerics at eps level) |
| Mode mapping | `aff @ mode_map` | Explicit m_src × m_tgt loop | ✅ |
| target_scales | Applied after mode_map | Applied after cfun dispatch | ✅ |
| Observe | `model.observe(x).sum(-1)[...,None]` | `state[voi_idx, ni, mi]` (no mode sum) | ❌ for n_modes>1 |
| Stimulus | Step-by-step get_coupling | Pre-computed batch array | ✅ |
| Boundaries | `_bound_state` per model | Inline boundary code | ✅ |
| Clamping | `X[idx] = values` | Not implemented | ❌ (no model uses it) |

## Appendix B: Monitor Correctness Matrix

| Monitor | Hybrid Semantics | Numba Semantics | Match? |
|---------|-----------------|-----------------|--------|
| TemporalAverage | Stock (istep steps), avg, midpoint time | Chunk avg (chunk_size steps), midpoint time | ⚠️ Only when chunk_size == istep |
| Raw | Every step, all voi | chunk_size=1, all voi | ✅ |
| SubSample | Decimate by istep | Step mask on chunk output | ⚠️ Only when chunk_size=1 |
| SpatialAverage | `spatial_mean @ state` at istep, modes summed | `spatial_mean * voi` every step, avg over chunk | ⚠️ Period alignment |
| Projection | `gain @ state` accumulated over istep, divided | `gain * voi` every step, avg over chunk | ⚠️ Period alignment |
| GlobalAverage | `mean(nodes)` at istep | `mean(nodes)` per chunk | ⚠️ Period alignment |
| Bold | HRF convolution with stock/interim | Balloon ODE | ❌ Different algorithm (by design) |
| AfferentCoupling | Raw coupling every step | ctavg per chunk | ⚠️ Only when chunk_size=1 |
| BoldRegionROI | Bold + region average | Not supported | ❌ |

## Appendix C: Key Insight — The Chunk Alignment Problem

The fundamental architectural mismatch is that the hybrid simulator operates **step-by-step** while the Numba backend operates **chunk-by-chunk**. This causes misalignment for any monitor that needs period-based sampling:

1. **TemporalAverage(period=P)** expects `istep = P/dt` steps of stock accumulation, then average. The Numba backend accumulates `chunk_size` steps. When `chunk_size ≠ istep`, the averaging window is wrong.

2. **SpatialAverage / Projection / GlobalAverage** all sample at `step % istep == 0`. The Numba backend samples every chunk. When `chunk_size ≠ istep`, the sampling times are wrong.

3. **SubSample** decimates by `step % istep == 0`. Only correct with `chunk_size=1` (currently enforced by a guard).

**Recommendation**: Auto-compute `chunk_size` from monitor periods to ensure alignment. If multiple monitors have different periods, use `GCD(periods) / dt` as chunk_size. This is the single most impactful correctness fix.

## Appendix D: Noise Handling Differences

The hybrid simulator generates noise step-by-step via `noise.generate(X.shape)` which uses numpy's random state. The Numba backend pre-draws all noise into a `(nvar, nodes, modes, nstep)` array. This means:

1. Noise sequences differ between hybrid and Numba (different random draws)
2. Numba noise is deterministic for a given nstep (entire sequence drawn at once)
3. The hybrid simulator's noise is stateful (RandomState advances per step)

For correctness verification, tests use the same pre-drawn noise array for both paths. For production use, the Numba backend generates noise using the same numpy RandomState seeded identically — the actual sequence values match since both draw `nvar * nodes * modes * nstep` floats from the same generator.
