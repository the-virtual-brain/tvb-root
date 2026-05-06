## Implementation Plan: CUDA Sweep Backend — Code Review Remediation

**Based on**: `CODE_REVIEW.md` (13 issues: 1 critical, 4 high, 5 medium, 1 sweep API, 1 docs, 1 Zerlaut)

---

### Phase 1: Critical Bug Fix (single file, ~30 new lines, ~20 removed)

#### 1.1 — F1: Fix Heun Combined dfun Cross-Mode Intermediate Recomputation

**File**: `tvb_library/tvb/simulator/backend/templates/nb-hybrid-sweep-cuda.py.mako`  
**Lines affected**: ~655–740 (the integration section inside `_is_combined` Heun path)

**What's wrong**: The CUDA Heun step for combined-dfun models uses a single-mode loop:
```
for m: k1(m) → i1(m, scalar) → recompute(WRONG: all modes use same scalar) → k2(m) → update(m)
```
The CPU uses a correct two-pass pattern:
```
Pass1: for m: k1(m) → store _k1[m] → compute _i1[m] (array)
       recompute intermediates from _i1 arrays (correct per-mk indexing)
Pass2: for m: load state[m], load _i1[m], k2(m) with correct intermediates → update(m)
```

**Fix**: Restructure the Mako template's Heun integration block to match the CPU two-pass pattern.

**Before** (pseudocode of generated output):
```python
for m in range(n_modes):
    # load state, coupling
    d0 = dfun(state, coupling, m, intermediates_from_state, _sp, i, sweep_params, tid)
    i1 = state + dt * d0
    # BUG: cross-mode i1 intermediates use i1 scalar for all modes
    for mi in range(n_modes):
        _Axi_i1[mi] = sum_k Aik[mi,k] * i1  # same i1 for all k!
    d1 = dfun(i1, coupling, m, *_intermediates_i1, _sp, i, sweep_params, tid)
    state = state + 0.5 * dt * (d0 + d1)
```

**After** (pseudocode of generated output):
```python
# Pass 1: compute k1 for all modes, store derivatives + i1 states
_k1_svar = cuda.local.array((n_modes,), dtype=numba.float32)
_i1_svar = cuda.local.array((n_modes,), dtype=numba.float32)
for m in range(n_modes):
    state_m = state[tid, :, i, m]
    coupling_m = coupling[i, m]
    d0 = dfun(state_m, coupling_m, m, intermediates_from_state, _sp, i, sweep_params, tid)
    for each svar:
        _k1_svar[m] = d0_svar
        _i1_svar[m] = state_svar + dt * d0_svar  # plus noise for stochastic

# Recompute cross-mode intermediates from _i1 arrays (correct per-mk)
for each (op_name, op_mat, op_svar) in derived_matrix_ops:
    _op_i1 = cuda.local.array((n_modes,), dtype=numba.float32)
    for mi in range(n_modes):
        for mk in range(n_modes):
            _op_i1[mi] += matrix[mi, mk] * _i1_svar[mk]  # CORRECT: uses _i1_svar[mk]

# Pass 2: compute k2 and update state
for m in range(n_modes):
    state_m = state[tid, :, i, m]
    coupling_m = coupling[i, m]
    d1 = dfun(_i1_svar, coupling_m, m, *_intermediates_i1, _sp, i, sweep_params, tid)
    for each svar:
        state[tid, svar, i, m] = state_svar + 0.5 * dt * (_k1_svar[m] + d1_svar)  # plus noise
```

**Detailed Mako changes**:

The current template at ~653 generates this structure inside the `for m in range(...)` loop:

```
## Mako block A (currently at ~653): cross-mode intermediates from state
% if _is_combined:
    % for _op_name, _op_mat, _op_svar in _dm_ops:
    _${_op_name} = cuda.local.array(...)
    ...
    % endfor
% endif

for m in range(${n_modes}):
    ## load state & coupling
    ...

    ## k1
% if _is_combined:   ← line ~676
    dfun(..., m, matrices, intermediates_from_state, ...)
% else:
    dfun(..., ...)
% endif

    ## Heun i1 intermediate
    i1_svar = svar + dt * d0_svar

    ## clamp i1

    ## i1 cross-mode recompute (BUG)
% if _is_combined:   ← line ~712
    % for _op_name, _op_mat, _op_svar in _dm_ops:
    _${_op_name}_i1 = cuda.local.array(...)
    ... * i1_${_op_svar}   ← BUG LINE
    % endfor
% endif

    ## k2
    dfun(i1, ..., m, intermediates_i1, ...)

    ## Heun update
    n_svar = svar + 0.5 * dt * (d0 + d1)
```

Must be restructured to:

```
## Phase 1 block (before the mode loop): compute FROM-STATE intermediates
% if _is_combined:
    % for _op_name, _op_mat, _op_svar in _dm_ops:
    _${_op_name} = cuda.local.array(...)
    ...
    % endfor
% endif

## Pass 1: compute k1 for all modes, store per-mode derivatives + i1 states
% if is_heun and _is_combined:
    ## Allocate per-mode storage
    % for sv in svars:
    _k1_${sv} = cuda.local.array((${n_modes},), dtype=numba.float32)
    _i1_${sv} = cuda.local.array((${n_modes},), dtype=numba.float32)
    % endfor
    for m in range(${n_modes}):
        ## load state & coupling for mode m
        ...
        ## k1
        (d0_sv, ...) = dfun(..., m, matrices, intermediates_from_state, ...)
        ## store per-mode
        % for sv in svars:
        _k1_${sv}[m] = d0_${sv}  ← store NOT d0_, use the raw derivative
        % endfor
        ## i1 from raw derivative
        % for sv in svars:
        _i1_${sv}[m] = ${sv} + dt_f * _k1_${sv}[m]
        % if is_stochastic:
        _i1_${sv}[m] += noise[tid, k2, i, m, t-1]
        % endif
        % endfor

    ## Recompute cross-mode intermediates from _i1 arrays (CORRECT)
    % for _op_name, _op_mat, _op_svar in _dm_ops:
    _${_op_name}_i1 = cuda.local.array((${n_modes},), dtype=numba.float32)
    for _mi in range(${n_modes}):
        for _mk in range(${n_modes}):
            _${_op_name}_i1[_mi] += matrix[mi, mk] * _i1_${_op_svar}[_mk]
    % endfor

    ## Pass 2: compute k2 and update state
    for m in range(${n_modes}):
        ## k2 using correct i1 intermediates
        (d1_sv, ...) = dfun(_i1 values, coupling, m, matrices, intermediates_i1, ...)
        ## Heun update
        % for sv in svars:
        n_${sv} = ${sv} + 0.5 * dt * (_k1_${sv}[m] + d1_${sv})
        % if is_stochastic:
        n_${sv} += noise[tid, k2, i, m, t-1]
        % endif
        % endfor
% else:
    ## ---- Original single-mode loop for Euler and non-combined Heun ----
    for m in range(${n_modes}):
        ...
        % if is_heun:
        ## non-combined Heun: single-mode loop is fine (no cross-mode intermed)
        ...
        % else:
        ## Euler
        ...
        % endif
% endif
```

**Note**: dfun in combined mode takes `(svars, cterms, _m, matrices, intermediates, _sp, ni, sweep_params, tid)`. The raw derivative for k1 should pass `d0_sv` directly (NOT via `_k1_sv` naming), then store it into `_k1_sv[m]`. The dfun call signature is unchanged — only the control flow changes.

**Euler path unaffected**: The EulerDeterministic/EulerStochastic path for combined dfun uses this single-mode loop:
```
for m: k1(m) → n = state + dt * k1 → clamp → store
```
This IS correct because there's no i1 state to recompute. The `_is_combined and is_heun` guard ensures the two-pass pattern only activates for Heun.

**Risk**: Must ensure that `cuda.local.array` allocations happen at function scope (not inside conditionals that may not execute). In Numba CUDA, local arrays in conditionals work at compile time since the Mako template resolves the branch.

**Test**: `test_cuda_reduced_set_heun_vs_cpu` — 8 nodes, 3 modes, 20 steps, assert maxerr < 1e-3.

---

### Phase 2: Missing Monitor Accumulators (single template file, ~50 new lines)

#### 2.1 — F2: Add CTAVG Accumulator

**File**: `tvb_library/tvb/simulator/backend/templates/nb-hybrid-sweep-cuda.py.mako`  
**Where**: After coupling computation and before integration, inside the time loop (after the `# ---- coupling ----` section completes for all projections, before `for i in range(N):` per subnet)

**What to add**:
```python
## Accumulate coupling temporal average (matching CPU lines 759–766)
% for sn in subnets:
% if sn_nmodes == 1:
for ci in range(${n_cvar}):
    for ni in range(N_${sn.name}):
        ${sn.name}_ctavg[tid, ci, ni, 0] += ${sn.name}_c[ci, ni, 0]
% else:
for ci in range(${n_cvar}):
    for ni in range(N_${sn.name}):
        for mi in range(${sn_nmodes}):
            ${sn.name}_ctavg[tid, ci, ni, mi] += ${sn.name}_c[ci, ni, mi]
% endif
% endfor
```

**Kernel signature changes**: Add `_ctavg` arrays per subnet to the `run_sweep_kernel` function signature and the top-level kernel wrapper.

**Backend changes** (`nb_hybrid_cuda_sweep_backend.py`): Allocate `ctavg_h[sn_info.name] = np.zeros((n_sweeps, n_cvar, nnodes, nmodes), dtype=np.float32)` per subnet around the same location as tavg_h allocation (~line 640). Normalize by `total_nstep` after all chunks (same as tavg). Return in `Results` dict as `_ctavg` key.

**Output shape**: `(n_sweeps, n_cvar, nnodes, nmodes)` per subnet — modes summed to mode 0 on host side (matching CPU behavior where `data_arr[..., :1]`).

**Test**: `test_cuda_ctavg_single_sweep` — 2 var model, 1 sweep point, 1 cvar, 10 steps, verify non-zero ctavg.

#### 2.2 — F3: Add SpatialAverage and Projection Monitors

**File**: `tvb_library/tvb/simulator/backend/templates/nb-hybrid-sweep-cuda.py.mako`  
**Where**: Inside the tavg accumulation block (~line 770), add nested loops matching CPU lines 806–821.

**What to add** (inside the `for _vi` loop that computes `_sv`):
```python
## Accumulate spatial average (region-of-interest)
if _spatial_mean.shape[0] > 0:
    for _ai in range(_spatial_mean.shape[0]):
        ${sn.name}_spatial_tavg[tid, _vi, _ai, 0] += _spatial_mean[_ai, ni] * _sv

## Accumulate projection output (sensor-level)
if _gain.shape[0] > 0:
    for _si in range(_gain.shape[0]):
        ${sn.name}_proj_tavg[tid, _vi, _si, 0] += _gain[_si, ni] * _sv
```

For multi-mode subnets, `_sv` accumulates over modes already (`n_modes > 1` path sums into mode 0). The spatial/proj accumulators only store in mode 0 (matching CPU).

**Kernel signature additions** (per subnet):
```python
${sn.name}_spatial_mean,  # (n_areas, n_nodes) float32 — spatial region mapping
${sn.name}_spatial_tavg,  # (n_sweeps, n_voi, n_areas, 1) float32 — accumulator
${sn.name}_gain,          # (n_sensors, n_nodes) float32 — sensor gain/projection matrix
${sn.name}_proj_tavg,     # (n_sweeps, n_voi, n_sensors, 1) float32 — accumulator
```

**Backend changes**: 
- Allocate `spatial_tavg_h[sn_info.name] = np.zeros((n_sweeps, n_voi, n_areas, 1), dtype=np.float32)` — where `n_areas` comes from the `spatial_mean` matrix shape[0] (passed in by user via monitors dict)
- Allocate `proj_tavg_h[sn_info.name] = np.zeros((n_sweeps, n_voi, n_sensors, 1), dtype=np.float32)` — where `n_sensors` comes from `gain` shape[0]
- Accept `spatial_mean` and `gain` matrices in the `monitors` dict passed to `run()`:
  ```python
  monitors = {
      'spatial_mean': {subnet_name: ndarray_or_None, ...},
      'gain': {subnet_name: ndarray_or_None, ...},
  }
  ```
- When `spatial_mean[name]` is None or empty, use shape `(0, nnodes)`; same for gain.
- Normalize by `total_nstep` after all chunks.

**Return in Results dict**: `_spatial_tavg` and `_proj_tavg` keys per subnet.

**Test**: `test_cuda_spatial_proj_monitors` — identity spatial_mean, identity gain, 1 sweep, 5 steps, verify shapes correct and non-zero.

---

### Phase 3: Latent Correctness + Quality (two files, ~10 new lines)

#### 3.1 — F4: Emit dfun_constants in Combined dfun Path

**File**: `tvb_library/tvb/simulator/backend/templates/nb-hybrid-sweep-cuda.py.mako`  
**Where**: Inside the `% if _is_combined:` branch of the dfun definition (~line 337), after `pi` and gparams, before sparams.

**Add**:
```mako
    % if hasattr(sn.model, 'dfun_constants') and sn.model.dfun_constants:
% for _cname, _cval in sn.model.dfun_constants.items():
    ${_cname} = np.float32(${_cval})
% endfor
% endif
```

**Location**: After line 382 (gparams/sweep_model_dims block), same position as the non-combined dfun.

**Test**: Latent — no model currently has both combined dfun and dfun_constants. Add `test_cuda_combined_dfun_constants_does_not_crash` with a mock model if one is created.

#### 3.2 — H1: Add normalize Flag to Kernel

**File**: `tvb_library/tvb/simulator/backend/templates/nb-hybrid-sweep-cuda.py.mako`  
**Where**: Add a `normalize` boolean argument to the kernel signature (default `True`).

When `normalize=True`, the kernel divides tavg/ctavg/spatial/proj by the actual step count at kernel exit. The host backend would skip its own division.

**Decision**: DEFER. The current contract (host divides) is documented in the kernel comment and works correctly. Adding a kernel-side normalization would require an `nstep` argument (which the kernel already has as `this_nstep`). Low value-add for the complexity. Instead, add an explicit comment in the kernel docstring.

**Action**: Add comment block at top of kernel function (~line 880):
```python
"""
CONTRACT: All temporal average accumulators (tavg, ctavg, spatial_tavg, proj_tavg) are
NOT normalized inside the kernel. The caller MUST divide by the total step count
after the kernel returns. This is handled automatically by NbHybridCUDASweepBackend.run().
"""
```

#### 3.3 — H2: Lift Coupling wsum Arrays Outside Target-Node Loop

**File**: `tvb_library/tvb/simulator/backend/templates/nb-hybrid-sweep-cuda.py.mako`  
**Where**: The coupling function (`_couple_*`) currently allocates `cuda.local.array((n_src_modes,), ...)` inside `for j in range(N_tgt)`. Move allocation before the `for j` loop, using 2D shape.

**Before** (~line 60):
```python
for j in range(N_tgt):
    wsum_0 = cuda.local.array((n_src_modes,), dtype=numba.float32)
    ...
```

**After**:
```python
wsum_all = cuda.local.array((n_cvar, n_src_modes), dtype=numba.float32)
for j in range(N_tgt):
    # zero wsum_all
    for ic in range(n_cvar):
        for ms in range(n_src_modes):
            wsum_all[ic, ms] = np.float32(0.0)
    ...
    # access as wsum_all[ic, ms] instead of wsum_ic[ms]
```

**Alternative (simpler)**: Just move the existing 1D allocations before the loop:
```python
% if cm in ('1_to_1', 'n_to_n'):
% for ic in range(n_tgt_cvar):
${p.name}_wsum_${ic} = cuda.local.array((${p.n_src_modes},), dtype=numba.float32)
% endfor
% elif cm == 'many_to_1':
${p.name}_wsum_0 = cuda.local.array((${p.n_src_modes},), dtype=numba.float32)
% elif cm == '1_to_many':
% for ic in range(n_tgt_cvar):
${p.name}_wsum_${ic} = cuda.local.array((${p.n_src_modes},), dtype=numba.float32)
% endfor
% endif
for j in range(N_${tgt}):
    # zero arrays
    ...
```

This avoids per-target-node re-allocation while keeping the existing Mako logic simple.

---

### Phase 4: Host-Side Monitors (backend file only, ~30 new lines)

#### 4.1 — M1: Add GlobalAverage Monitor

**File**: `tvb_library/tvb/simulator/backend/nb_hybrid_cuda_sweep_backend.py`  
**Where**: In `_run_single_batch`, after tavg normalization, compute global mean.

**Implementation**:
```python
# After h[tavg] normalization:
if global_average:
    for sn_info in analysis.subnetworks:
        h = d_tavgs[sn_info.name].copy_to_host()
        # h shape: (n_sweeps, n_voi, nnodes, nmodes) — mean over node dim
        global_avg = h.mean(axis=2, keepdims=True)  # (n_sweeps, n_voi, 1, nmodes)
        results['ga_avg'][sn_info.name] = global_avg
```

**No kernel changes needed** — pure host-side post-processing.

**Test**: `test_cuda_global_average` — 1 sweep, 3-var model, verify shape (n_sweeps, n_voi, 1, n_modes).

#### 4.2 — M2: Fix Connectome-Ordered Merge

**File**: `tvb_library/tvb/simulator/backend/nb_hybrid_cuda_sweep_backend.py`  
**Where**: `_merge_subnet_outputs` method.

**Current behavior**: `np.concatenate(tavg_list, axis=2)` — simple node concatenation. Works for single-subnet or contiguous partionings but wrong for arbitrary multi-subnet.

**Fix**: Accept `node_indices` dict `{subnet_name: indices_array}` that maps each subnet's local node indices to global connectome positions. Build merged array:
```python
def _merge_subnet_outputs(self, tavg_dict, node_indices):
    """Merge per-subnet tavg into global connectome-ordered array.
    
    Parameters
    ----------
    tavg_dict : dict[str, ndarray]
        Per-subnet tavg arrays of shape (n_sweeps, n_voi, n_local_nodes, n_modes).
    node_indices : dict[str, ndarray]
        Global node indices for each subnet.
    
    Returns
    -------
    ndarray of shape (n_sweeps, n_voi, n_global_nodes, n_modes)
    """
    n_global = max(max(idxs) for idxs in node_indices.values()) + 1
    n_sweeps = next(iter(tavg_dict.values())).shape[0]
    n_voi = next(iter(tavg_dict.values())).shape[1]
    n_modes = next(iter(tavg_dict.values())).shape[3]
    merged = np.zeros((n_sweeps, n_voi, n_global, n_modes), dtype=np.float32)
    for name, tavg in tavg_dict.items():
        idxs = node_indices[name]
        merged[:, :, idxs, :] = tavg
    return merged
```

**Accept `node_indices` in `run()` API**:
```python
compiled.run(nstep=100, sweep_values=values,
             node_indices={'A': np.arange(8), 'B': np.arange(8, 76)})
```

When `node_indices` is None, fall back to concatenation (current behavior).

#### 4.3 — H3: Bold State in Snapshot

**File**: `tvb_library/tvb/simulator/backend/nb_hybrid_cuda_sweep_backend.py`  
**Where**: Snapshot dict construction (currently `{'states': ..., 'srcbufs': ..., 'step_offset': ...}`)

**Add**:
```python
snapshot = {
    'states': dict of per-subnet state arrays,
    'srcbufs': dict of per-subnet circular buffer arrays,
    'step_offset': int,  # global step counter
    'bold_states': dict of per-subnet bold state arrays,  # (n_sweeps, n_voi, 4, n_nodes)
}
```

When resuming with snapshot, restore bold states alongside states and srcbufs.

---

### Phase 5: Medium Quality Improvements (two files, ~15 new lines)

#### 5.1 — M3: Document Stimulus Time Indexing Equivalence

**File**: `tvb_library/tvb/simulator/backend/nb_hybrid_cuda_sweep_backend.py`  
**Where**: In the stimulus loading comment block (~line 720)

**Add comment**:
```python
# Stimulus indexing: CPU backend uses stim.get_coupling(step_idx) with step_idx ∈ [1, nstep].
# CUDA kernel uses stim[tid, ic, j, m, t - 1] with t = t_local + t_offset ∈ [1+t_offset, nstep+t_offset].
# Both produce 0-based indexing into the (n_sweeps, nvar, N, n_modes, nstep) stimulus array.
```

#### 5.2 — M4: Document Zerlaut Template Args Contract

**File**: `tvb_library/tvb/simulator/backend/templates/nb-zerlaut-sweep-cuda.py.mako`  
**Where**: Top of template, after `<%page args="...">`

**Add comment**:
```python
## WARNING: This template is <%include%>ed from nb-hybrid-sweep-cuda.py.mako.
## The <%page args="..."> MUST include ALL args that the parent template passes,
## even if they are unused by Zerlaut models (is_heun, is_combined, dm_names, dm_data, dm_ops).
## If the parent adds new page args, this template must be updated.
```

#### 5.3 — M5: Fix Noise Float64 Hazard

**File**: `tvb_library/tvb/simulator/backend/nb_hybrid_cuda_sweep_backend.py`  
**Where**: Noise generation block (~line 700)

**Current**:
```python
dw = rng.randn(n_sweeps, nstep, nvar, nnodes, nmodes)
noise_std = np.sqrt(2.0 * sn_info.noise_nsig * dt)
dw *= noise_std[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
dw = np.ascontiguousarray(np.transpose(dw, (0, 2, 3, 4, 1))).astype(np.float32)
```

**Fix**: Cast to float32 before the transpose+contiguous path to avoid double memory:
```python
dw = rng.randn(n_sweeps, nstep, nvar, nnodes, nmodes).astype(np.float32, copy=False)
noise_std = np.sqrt(2.0 * sn_info.noise_nsig * dt).astype(np.float32)
dw *= noise_std[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
dw = np.ascontiguousarray(np.transpose(dw, (0, 2, 3, 4, 1)))
```

---

### Phase 6: Pytest Test Suite (new file, 8 tests)

#### 6.1 — F5: Create `test_nb_hybrid_cuda_sweep.py`

**File**: `tvb_library/tvb/tests/library/simulator/backend/test_nb_hybrid_cuda_sweep.py` (new)

| # | Test | Model | Nodes | Steps | Sweeps | Assertion | Skip if |
|---|------|-------|-------|-------|--------|-----------|---------|
| 1 | `test_mpr_vs_cpu` | Generic2dOscillator | 76 | 20 | 1 | maxerr < 1e-3 vs CPU | no CUDA |
| 2 | `test_jr_vs_cpu` | JansenRit | 76 | 20 | 1 | maxerr < 1e-3 vs CPU | no CUDA |
| 3 | `test_chunking_bit_exact` | Generic2dOscillator | 8 | 100 | 1 | chunked == unchunked | no CUDA |
| 4 | `test_batching_bit_exact` | Generic2dOscillator | 8 | 10 | 20 | batched == unbatched | no CUDA |
| 5 | `test_bold_no_nan` | Generic2dOscillator | 8 | 200 | 1 | no NaN, shape correct | no CUDA |
| 6 | `test_raw_monitor_shape` | Generic2dOscillator | 8 | 20 | 1 | shape (n_sweeps, nstep, nvoi, N, n_modes) | no CUDA |
| 7 | `test_subsample_shape` | Generic2dOscillator | 8 | 20 | 1 | shape with period=4 | no CUDA |
| 8 | `test_heun_combined_vs_cpu` | ReducedSetFitzHughNagumo | 8 | 20 | 1 | maxerr < 1e-3 | no CUDA |
| 9 | `test_ctavg_present` | MPR | 8 | 10 | 1 | ctavg non-zero, shape correct | no CUDA |
| 10 | `test_spatial_proj_shape` | Generic2dOscillator | 8 | 5 | 1 | spatial shape (nvoi, narea, 1), proj shape (nvoi, nsensor, 1) | no CUDA |

**CUDA detection fixture**:
```python
@pytest.fixture(scope="module")
def cuda_available():
    try:
        from numba import cuda
        if cuda.is_available():
            return True
    except Exception:
        pass
    return False

@pytest.fixture
def skip_if_no_cuda(cuda_available):
    if not cuda_available:
        pytest.skip("CUDA not available")
```

**Reuse CPU test helpers**: Import `NbHybridBackend`, `MPR`, `JansenRit`, `connectivity_76`, `test_tvb_library` fixture patterns from `test_nb_hybrid.py`.

---

### Phase 7: CPU Sweep API (single file, ~40 lines)

#### 7.1 — Add `NbHybridBackend.run_sweep()`

**File**: `tvb_library/tvb/simulator/backend/nb_hybrid.py`  
**Where**: New method on `NbHybridBackend` class (~line 1600, before end of class).

**Implementation** (~40 lines):
```python
def run_sweep(self, network_set, sweep_values, nstep=100,
              initial_states=None, sweep_descriptor=None,
              chunk_size=None, bold_period=None, **monitors):
    """Run parameter sweep sequentially on CPU.
    
    Each sweep point calls run_network() internally. Results are
    returned as a list of per-sweep-point tuples matching the
    run_network() return format.
    
    Parameters
    ----------
    sweep_values : ndarray (n_sweeps,) or (n_sweeps, n_sweep_dims)
    sweep_descriptor : list of dict, optional
        [{type: 'cfun', projection: 'proj_AB', param_idx: 0},
         {type: 'model', subnet: 'A', param: 'tau_E'}]
    """
    sweep_values = np.asarray(sweep_values, dtype=np.float32)
    if sweep_values.ndim == 1:
        sweep_values = sweep_values.reshape(-1, 1)
    
    if sweep_descriptor is None:
        if network_set.projections:
            first_proj = network_set.projections[0]
            sweep_descriptor = [{'type': 'cfun', 'projection': first_proj.name,
                                 'param_idx': 0}]
        else:
            sweep_descriptor = []
    
    n_sweeps = sweep_values.shape[0]
    results = []
    
    for tid in range(n_sweeps):
        sv = sweep_values[tid]
        
        # Apply sweep values to cfun/model params
        restore = {}
        for dim, desc in enumerate(sweep_descriptor):
            if desc['type'] == 'cfun':
                pname = desc['projection']
                pidx = desc.get('param_idx', 0)
                for proj in network_set.projections:
                    if proj.name == pname:
                        key = (proj.name, pidx)
                        restore[key] = proj.cfun.parameters[pidx]
                        proj.cfun.parameters[pidx] = float(sv[dim])
            elif desc['type'] == 'model':
                sname = desc['subnet']
                param = desc['param']
                for sn in network_set.subnets:
                    if sn.name == sname:
                        key = (sn.name, param)
                        val = float(getattr(sn.model, param))
                        restore[key] = val
                        setattr(sn.model, param, np.array([float(sv[dim])]))
        
        # Run single simulation
        result = self.run_network(network_set, nstep=nstep,
                                  initial_states=initial_states,
                                  chunk_size=chunk_size,
                                  bold_period=bold_period, **monitors)
        results.append(result)
        
        # Restore original values
        for (pname, pidx), orig in restore.items():
            # cfun params
            for proj in network_set.projections:
                if proj.name == pname:
                    proj.cfun.parameters[pidx] = orig
            # model params (if key matches model attr pattern)
            # Note: restore dict has mixed types; simpler to track separately
    
    return results
```

**Alternative — `CompiledNetwork.run_sweep()`**: Adds sweep to the compiled object:
```python
compiled = NbHybridBackend().compile(network_set)  # existing
results = compiled.run_sweep(sweep_values, nstep=100)  # new
```

This matches the CUDA pattern (`backend.compile_sweep(ns).run(...)`) and caches the JIT kernel: same kernel for all sweep points, only cfun/model params change via host-side mutation.

**Test**: `test_cpu_run_sweep_linear_cfun` in `test_nb_hybrid.py` — 3 sweep values, verify each result differs from neighbors.

---

### Phase 8: Documentation (new file)

#### 8.1 — Create `tvb_library/tvb/simulator/backend/CUDA_SWEEP_API.md`

Quick reference covering:
- Installation (CUDA toolkit, Numba CUDA)
- Basic usage: `NbHybridCUDASweepBackend().compile_sweep(ns).run()`
- Monitors: `tavg` (default), `raw`, `subsample`, `bold`
- Sweep descriptors: cfun, model, multi-projection
- Chunking, batching, snapshots
- Zerlaut models (automatic template selection)
- Memory estimation and `max_batch_sweeps`
- CPU sweep fallback via `NbHybridBackend.run_sweep()`

---

### Execution Order

```
┌─ Phase 1 (F1):     Fix Heun combined dfun bug         ~1 hr    Mako template only
├─ Phase 2.1 (F2):   Add CTAVG accumulator               ~1 hr    Mako + backend
├─ Phase 2.2 (F3):   Add SpatialAverage / Projection     ~1.5 hr  Mako + backend
├─ Phase 3 (F4/H2/M3/M5): Latent fixes + quality          ~30 min  Mako + backend
├─ Phase 4 (M1/M2/H3): Host-side monitors + merge        ~30 min  Backend only
├─ Phase 5 (M4 + doc comment): Zerlaut template doc       ~10 min  Zerlaut template
├─ Phase 6 (F5):     Pytest test suite                    ~2 hr    New test file
├─ Phase 7:          CPU run_sweep() API                  ~30 min  nb_hybrid.py
├─ Phase 8:          API documentation                    ~30 min  New .md file
└─ Validation:       Full test suite run                  ~15 min  pytest
───────────────────────────────────────────────────────────────────
Total estimate: ~8.5 hrs (~1 day)
```

---

### Files Changed (Summary)

| File | Phases | Lines Changed |
|------|--------|--------------|
| `nb-hybrid-sweep-cuda.py.mako` | 1, 2, 3 | +80 / −25 |
| `nb_hybrid_cuda_sweep_backend.py` | 2, 4, 5 | +55 / −5 |
| `nb-zerlaut-sweep-cuda.py.mako` | 5 | +5 / −0 |
| `nb_hybrid.py` | 7 | +40 / −0 |
| `test_nb_hybrid_cuda_sweep.py` (NEW) | 6 | +200 / −0 |
| `CUDA_SWEEP_API.md` (NEW) | 8 | +150 / −0 |
| **Total** | | **+530 / −30** (~500 net new) |
