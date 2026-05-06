## Code Review: CUDA Sweep Backend vs CPU Hybrid Numba Backend

**Scope**: New files `nb_hybrid_cuda_sweep_backend.py`, `nb-hybrid-sweep-cuda.py.mako`, `nb-zerlaut-sweep-cuda.py.mako`  
**Reference**: `nb_hybrid.py` + `nb-hybrid-sim.py.mako` (CPU backend)  
**Branch**: `hybrid-numba`  
**Verdict**: REQUEST CHANGES  

### Files Reviewed
| File | Lines | Type |
|------|-------|------|
| `nb_hybrid_cuda_sweep_backend.py` | 990 | Production (backend orchestrator) |
| `nb-hybrid-sweep-cuda.py.mako` | 910 | Production (CUDA kernel template) |
| `nb-zerlaut-sweep-cuda.py.mako` | 358 | Production (Zerlaut custom dfun) |
| `nb_hybrid_cuda_sweep.py` | 762 | Reference prototype |

---

## Critical Findings (Blocking)

### F1: Heun Combined dfun Cross-Mode Intermediate Bug — TEMPLATE:710-725

**Severity**: CRITICAL  
**Impact**: Wrong integration for HeunDeterministic/HeunStochastic on ReducedSetFitzHughNagumo, ReducedSetHindmarshRose

**The bug**: The CUDA template's Heun i1 recomputation of cross-mode intermediates uses scalar values from a single mode instead of per-mode values.

```python
# Inside `for m in range(n_modes):` loop:
# Step 1: compute k1 (d0) correctly using full cross-mode intermediates
_Axi[mi] = sum_k Aik[mi, mk] * state[tid, xi_idx, i, mk]  # CORRECT

# Step 2: Heun i1 state for THIS mode m only
i1_xi = xi + dt * d0_xi  # scalar, mode m only

# Step 3: BUG - recompute _Axi_i1 using only mode m's i1 value
_Axi_i1[mi] = 0
for mk in range(n_modes):
    _Axi_i1[mi] += Aik[mi, mk] * i1_xi  # BUG: i1_xi constant across all mk
```

The cross-mode intermediate `_Axi_i1` should use `i1_xi` at mode `mk`, not mode `m`. Since only mode `m`'s intermediate is available as a scalar, every column gets the same value. This collapses the cross-mode coupling inside the Heun predictor step.

**Validation gap**: ReducedSet tests passed with "no NaN" check, but this only verified the kernel didn't crash. No numerical comparison against CPU reference was done for combined-dfun Heun.

**Required test**: Run ReducedSetFitzHughNagumo HeunDeterministic at (4 vars, 3 modes, 8 nodes) CPU vs CUDA and assert max abs error < 0.01.

**Fix options**:
- Option A: Store all i1 state values in a `cuda.local.array(n_svars, n_modes)` temp, then recompute properly
- Option B: Restructure the mode loop: compute k1 for all modes first (storing d0 arrays), then compute k2 for all modes (using stored d0 + stored i1 per mode)
- Option C: Use EulerDeterministic only for combined-dfun models (trivial but limiting)

---

### F2: No CTAVG (Coupling Temporal Average) Accumulator — TEMPLATE/backend

**Severity**: HIGH  
**Impact**: AfferentCoupling and AfferentCouplingTemporalAverage monitors are impossible

The CPU backend accumulates `ctavg[cvar, node, mode]` inside the JIT kernel per timestep. This is the foundation for:
- `AfferentCoupling` monitor (returns raw coupling temporal average)
- `AfferentCouplingTemporalAverage` monitor (period-based CTAVG)

The CUDA kernel has coupling scratch `tgt_c` zeroed per step but never accumulates a running average.

**Required**: Add `ctavg` accumulator arrays to kernel signature and accumulate per-timestep coupling values. Without this, the sweep backend cannot reproduce AfferentCoupling analysis that is standard in TVB workflows.

---

### F3: No SpatialAverage / Projection Monitor — TEMPLATE/backend

**Severity**: HIGH  
**Impact**: EEG/MEG/iEEG source localization and region-of-interest analysis impossible

The CPU JIT kernel computes per-step:
- `spatial_tavg[voi, area, mode] += spatial_mean[area, node] * state_value`  (region-level activity)
- `proj_tavg[voi, sensor, mode] += gain[sensor, node] * state_value`  (sensor-level signals)

Neither exists in the CUDA kernel. The spatial mean matrix and gain matrix are not even passed as kernel arguments.

**Required**: Add `spatial_mean` and `gain` arrays per subnet; accumulate per-step like CPU. These are the standard monitors for EEG/MEG analysis.

---

### F4: Missing dfun_constants in Combined dfun Path — TEMPLATE:~350

**Severity**: MEDIUM (latent)  
**Impact**: Any future model with both `dfun_mode='combined'` AND `dfun_constants` would silently produce wrong results

The non-combined dfun path correctly emits:
```python
% if hasattr(sn.model, 'dfun_constants') and sn.model.dfun_constants:
    constant_name = np.float32(value)
```

But the combined dfun path (the `_is_combined` branch) does NOT emit `dfun_constants`. Currently no model has both attributes, but this is a latent correctness hazard.

**Fix**: Mirror the `dfun_constants` emission block into the combined dfun branch.

---

### F5: No pytest Test Suite — ALL FILEs

**Severity**: HIGH  
**Impact**: Zero CI regression protection; all validation is manual

The CPU backend has 189 pytest tests in `test_nb_hybrid.py`. The CUDA sweep backend has zero. Every feature claim is backed only by manual ad-hoc scripts that a developer must remember to run.

**Minimum required tests**:
1. `test_cuda_mpr_vs_cpu_heun` — MPR (2 vars, 76 nodes, 1 sweep)
2. `test_cuda_jr_vs_cpu_heun` — JansenRit (6 vars, 76 nodes, 1 sweep)
3. `test_cuda_chunking_bit_exact` — 100 steps, chunk_size=25 vs unchunked
4. `test_cuda_batching_bit_exact` — 20 sweeps, max_batch_sweeps=5
5. `test_cuda_bold_no_nan` — 200 steps, bold_period=2.0
6. `test_cuda_raw_monitor_shape` — monitor_type=1, nstep=20
7. `test_cuda_subsample_monitor_shape` — period=4
8. `test_cuda_reduced_set_fhn_no_crash` — 3 modes, 76 nodes, 10 steps

---

## High-Priority Issues

### H1: GPU tavg Normalization Relies on Host Division — BACKEND:~580

The kernel accumulates tavg WITHOUT normalizing (comment says "tavg normalization is performed by the host backend after all chunks"). The backend's `_run_single_batch()` does `h /= np.float32(total_nstep)`. This is correct BUT fragile: if anyone ever calls the kernel directly (or reuses it in another context), the un-normalized tavg would be silently wrong.

**Suggestion**: Add a `normalize` bool kernel arg (default True for safety), or keep a comment in the kernel docstring warning about this contract.

### H2: Per-Node cuda.local.array in Coupling Functions — TEMPLATE:~70

The coupling function allocates `cuda.local.array(n_src_modes)` PER TARGET NODE (`for j in range(N_tgt)`). For 76 nodes × 2 cvar × 3 modes = 456 local array allocations per coupling function. While Numba CUDA hoists these to function scope as static allocations, the pattern is unusual and compiler-dependent. For very large node counts or high mode counts, this could exceed local memory limits.

**Suggestion**: Pre-allocate coupling wsum arrays outside the `for j` loop (using a 2D local array `(n_cvar, n_src_modes)`) to make the allocation pattern explicit and reduce compiler ambiguity.

### H3: Snapshot Doesn't Persist Bold State Across Batches — BACKEND:~630

When `max_batch_sweeps` splits a sweep, the Bold state arrays are sliced per batch but not accumulated between batches. If the user wants to continue the simulation across batches, the Bold state (s,f,v,q) would need to be carried forward.

**Impact**: Currently low because `max_batch_sweeps` is per-kernel-launch, and all batches start from the same initial conditions. But for future multi-kernel resume, Bold state must be in the snapshot dict.

---

## Medium-Priority Issues

### M1: GlobalAverage Monitor Gap

CPU merges + averages across all connectome nodes. CUDA has no equivalent. This is a small implementation since it only requires a host-side mean across the node dimension of the tavg output. No kernel change needed.

### M2: Connectome-Ordered Merge is Naive

`_merge_subnet_outputs` does `np.concatenate(tavg_list, axis=2)` (simple node-dim concatenation). The CPU backend's merge is `node_indices`-aware: data for subnet A's nodes goes to their proper connectome positions. CUDA merge produces correct results only when subnet nodes have contiguous connectome indices, which happens to be true for the two-subnet 76-node tests but would break for arbitrary partitionings.

### M3: Stimulus step-index is 1-based in CPU but t-1 index in CUDA

CPU stimulus: `stim.get_coupling(step_idx)` where step_idx counts from 1. CUDA: `stim[..., t - 1]` where t counts from 1+t_offset. Both produce 0-based indexing into the stimulus array. These are equivalent but the documentation should note this.

### M4: Zerlaut Template Args Passthrough

The Zerlaut CUDA template's `<%page args="...">` now includes `is_heun`, `is_combined`, `dm_names`, `dm_data`, `dm_ops` which are unused by Zerlaut itself but required because the main template's `<%include>` passes them. This is fragile — if the main template adds more args, someone must remember to also update the Zerlaut template's `<%page args>` or the include will fail.

### M5: Noise Transpose Pattern Could Be a Float64 Hazard

Noise generation does `dw = rng.randn(n_sweeps, nstep, nvar, nnodes, nmodes)` then transposes. If `rng.randn` returns float64 (it does by default), the memory doubles. Should use `.astype(np.float32)` before `.transpose()`.

---

## Sweep API for CPU Hybrid Backend

### Current State

| Operation | CPU | CUDA |
|-----------|-----|------|
| Single simulation | `backend.run_network(ns, nstep=100)` | N/A (CUDA is sweep-only) |
| Parameter sweep | Manual for-loop by user | `compiled.run(nstep=100, sweep_values=values)` |
| Compile once, run many | `backend.compile(ns).run(nstep=100, initial_states=...).resume(...)` | `backend.compile_sweep(ns).run(nstep=100, sweep_values=values)` |

### Recommended API Addition

Add a `run_sweep()` convenience method to `NbHybridBackend`:

```python
class NbHybridBackend:
    def run_sweep(self, network_set, sweep_values, nstep=100,
                  initial_states=None, sweep_descriptor=None,
                  chunk_size=None, verbose=False):
        """Run parameter sweep sequentially on CPU.
        
        Parameters
        ----------
        sweep_values : ndarray (n_sweeps,) or (n_sweeps, n_sweep_dims)
        sweep_descriptor : list of dict
            Same shape as CUDA sweep_descriptor.
        
        Returns
        -------
        list of (times, data, ctavg, spatial, proj, bold_times, bold_data)
            Per-sweep-point results matching run_network() return format.
        """
        results = []
        for i in range(len(sweep_values)):
            sv = sweep_values[i]
            # Modify cfun/model params in-place based on sweep_descriptor
            ...
            result = self.run_network(network_set, nstep=nstep,
                                      initial_states=initial_states,
                                      chunk_size=chunk_size)
            results.append(result)
        return results
```

### Design Decisions

1. **Keep it simple as a sequential loop** — the CUDA backend is the high-throughput option. The CPU sweep is for debugging, small sweeps, and CUDA-unavailable environments.

2. **Mirror the CUDA API shape** — accept `sweep_descriptor` with same dict structure. This lets users write sweep code once and switch backends by changing the class.

3. **Don't parameterize the JIT kernel** — the CPU kernel is already compiled per-model. Adding runtime sweep parameters would require either kernel recompilation or many Python branches. A sequential loop is simpler and correct.

4. **Reuse cfun/model param modification** — when running each sweep point, temporarily mutate the `Projection.cfun` or model parameter, then restore. This avoids needing two separate analysis objects.

5. **Add `compile_sweep()`** — analogous to CUDA's `compile_sweep()`, returns a callable with `.run(sweep_values)`. This caches the compiled JIT kernel.

### Minimal Implementation

```python
def run_sweep(self, network_set, sweep_values, nstep=100,
              initial_states=None, sweep_descriptor=None,
              chunk_size=None, bold_monitor=None, **monitors):
    """Sequential CPU parameter sweep."""
    if sweep_descriptor is None:
        if network_set.projections:
            first_proj = network_set.projections[0]
            sweep_descriptor = [{'type': 'cfun', 'projection': first_proj.name, 'param_idx': 0}]
        else:
            sweep_descriptor = []
    
    sweep_values = np.asarray(sweep_values, dtype=np.float32)
    if sweep_values.ndim == 1:
        sweep_values = sweep_values.reshape(-1, 1)
    
    compiled = self.compile(network_set)
    results = []
    
    for sv in sweep_values:
        # Apply sweep values to cfun/model params
        for dim, desc in enumerate(sweep_descriptor):
            if desc['type'] == 'cfun':
                pname = desc['projection']
                pidx = desc.get('param_idx', 0)
                for proj in network_set.projections:
                    if proj.name == pname:
                        orig = proj.cfun.parameters[pidx]
                        proj.cfun.parameters[pidx] = float(sv[dim])
                        break
            elif desc['type'] == 'model':
                sname = desc['subnet']
                param = desc['param']
                for sn in network_set.subnets:
                    if sn.name == sname:
                        orig_val = float(getattr(sn.model, param))
                        setattr(sn.model, param, np.array([float(sv[dim])]))
                        break
        
        result = compiled.run(nstep=nstep, initial_states=initial_states,
                              chunk_size=chunk_size, _monitors=monitors)
        results.append(result)
        
        # Restore original values
        # (simplified — real impl would save/restore properly)
    
    return results
```

**File**: Add to `nb_hybrid.py` (new method on `NbHybridBackend`, ~20 lines)

**Test**: `test_cpu_run_sweep_scaling_cfun` — 3 sweep values, verify each result differs

---

## Summary

| Category | Count | Items |
|----------|-------|-------|
| **CRITICAL** | 1 | Heun combined-dfun cross-mode bug |
| **HIGH** | 4 | No CTAVG, no SpatialAvg/Projection monitors, missing dfun_constants in combined path, no pytest suite |
| **MEDIUM** | 5 | Normalization contract, coupling local arrays, Bold snapshot, GlobalAverage, naive merge |
| **Sweep API** | 1 | Add CPU `run_sweep()` method |

### Blocking Changes Required
1. Fix Heun combined-dfun cross-mode intermediate recomputation (F1)
2. Add pytest tests for all existing validated features (F5)
3. Add dfun_constants emission to combined dfun path (F4)

### Strongly Recommended (Non-blocking)
4. Add CTAVG accumulator in kernel (F2) — blocks AfferentCoupling
5. Add SpatialAverage/Projection monitor arrays (F3) — blocks EEG/MEG analysis
6. Add `NbHybridBackend.run_sweep()` — CPU sweep mirror of CUDA API
