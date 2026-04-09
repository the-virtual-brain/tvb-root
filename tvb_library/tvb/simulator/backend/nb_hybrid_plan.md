# Numba Backend for Hybrid Simulator

## Implementation Plan

> **Status (2026-04-09):** Phases 1–7 complete. Benchmarks validated.
> See §7 (Completed Work) for what has been built and §8 (Next Stages) for the
> performance and functionality roadmap. Sections below this header now reflect
> the *as-built* design rather than the original plan.

---

### 1. Architecture (as-built)

#### 1.1 Files

| File | Purpose |
|------|---------|
| `tvb/simulator/backend/nb_hybrid.py` | Backend class, dataclasses, cache |
| `tvb/simulator/backend/templates/nb-hybrid-sim.py.mako` | Full kernel template |
| `tvb/tests/library/simulator/backend/test_nb_hybrid.py` | 25 tests |

The original plan proposed four separate template files. The final design uses a
**single template** (`nb-hybrid-sim.py.mako`) that generates all functions
(coupling, dfun, integrator, inner `@njit` loop, outer Python loop) in one
render pass. This is simpler and avoids inter-template dependency management.

#### 1.2 Class hierarchy

```
NbHybridBackend(MakoUtilMix)
    .compile(network_set, print_source=False) -> CompiledNetworkFn
    .run_network(network_set, nstep, ...)     -> list[(times, data)]  [one-liner]
    ._run_compiled(fn, analysis, ...)         -> list[(times, data)]
    ._build(template, content, ...)           -> run_network callable (cached)
    ._analyse(network_set)                    -> NetworkAnalysis
    ._check_compatibility(network_set)        -> None | raises
    ._build_projection_info(p, is_inter)      -> ProjectionInfo
    ._make_projection_buffer(p, ...)          -> np.ndarray

CompiledNetworkFn                             # dataclass
    ._backend, ._analysis, ._run_network_fn, ._network_set
    .run(nstep, chunk_size, initial_states)   -> list[(times, data)]

NetworkAnalysis                               # dataclass
    .subnetworks: List[SubnetworkInfo]
    .inter_projections: List[ProjectionInfo]
    .intra_projections: List[ProjectionInfo]
    .stimuli_by_subnet: dict
    .all_projections (property)

SubnetworkInfo                                # dataclass
    name, model, integrator, n_nodes, n_modes
    is_stochastic, noise_nsig, has_stimulus

ProjectionInfo                                # dataclass
    name, source_subnet, target_subnet
    source_cvar, target_cvar                  # (n,) int32
    weights_data, weights_indices, weights_indptr  # CSR
    idelays, horizon, scale, target_scales, cfun
    is_inter, mode_map
    n_tgt_nodes, n_src_modes, n_tgt_modes     # properties
```

#### 1.3 In-process compilation cache

```python
_COMPILED_FN_CACHE: dict = {}   # module-level, process-lifetime
# key: SHA-256 of rendered Mako source
# value: exec()'d run_network callable
```

- Same topology → same SHA-256 → instant cache hit (no re-exec, no re-JIT)
- `cache=True` on `@nb.njit` was attempted and **reverted**: Numba's file-based
  locator fails for exec()-generated code (`co_filename='<string>'`). The
  process-lifetime dict is the practical alternative.
- `autopep8` is only called when `print_source=True` (removed from hot path).

---

### 2. Data Flow (as-built)

#### 2.1 Per-step kernel

```
for t_local in range(chunk_size):
    t = t_start + t_local

    1. Zero coupling arrays per subnetwork
    2. For each inter-projection  → compute_coupling_*(buf, CSR, idelays, …, t, tgt_c)
    3. For each intra-projection  → compute_coupling_*(buf, CSR, idelays, …, t, tgt_c)
    4. Add pre-computed stimulus  → tgt_c += stim[…, t-1]
    5. integrate_*(state, coupling [, noise, t-1])
    6. buf[:,:,:, t % horizon] = source_state   (per projection)
    7. tavg[:] += state[voi]
    tavg_count[0] += 1
```

#### 2.2 Outer Python loop

```python
while t_global <= nstep:
    this_chunk = min(chunk_size, remaining)
    reset_tavg()
    network_chunk(this_chunk, t_global, states, bufs, ..., tavg, tavg_count, noise, stim)
    outputs.append(tavg / tavg_count)
    t_global += this_chunk
return list_of(times_arr, data_arr) per subnetwork
```

Stimulus is pre-computed for the **full** `nstep` range before the loop (not
per-chunk), then passed as an `(n_cvar, n_nodes, n_modes, nstep)` array.

---

### 3. Generated Code (as-built)

#### 3.1 Coupling function — `compute_coupling_<proj>(...)`

Single unified function for both inter and intra projections, decorated
`@nb.njit(inline="always")`.  Mako specialises it at code-gen time:

| Condition | Generated code |
|-----------|---------------|
| `n_modes == 1` (mono_src) | scalar `wsum = nb.float32(0.0)`, no inner mode loop |
| `n_modes > 1` | `wsum = np.zeros(n_modes)`, explicit `for m` loop |
| `is_inter` | mode_map arg present; explicit `m_src × m_tgt` accumulation |
| `is_intra` | no mode_map arg; identity `m_src == m_tgt` |
| `cfun == "linear"` | `wsum = cfun_a * wsum + cfun_b` |
| `cfun == "scaling"` | `wsum = cfun_a * wsum` |
| `cfun == "none"` | no post-processing |
| cvar mapping | static branch: `1_to_1`, `n_to_n`, `many_to_1`, `1_to_many` |

#### 3.2 Integration — `integrate_<subnet>(state, coupling [, noise, t_abs])`

Decorated `@nb.njit(inline="always")`. Mako specialises at code-gen time:

| Condition | Generated code |
|-----------|---------------|
| `n_modes == 1` | mode loop elided, all indices hardcoded to `0` |
| `n_modes > 1` | explicit `for m in range(n_modes)` |
| HeunDeterministic | Heun predictor-corrector, two dfun evaluations |
| EulerDeterministic | forward Euler, one dfun evaluation |
| HeunStochastic | `i1svar = svar + dt*d0 + noise[k,i,0,t_abs]`, Heun average |
| EulerStochastic | `nsvar = svar + dt*d0 + noise[k,i,0,t_abs]` |

Boundary conditions are applied inline after each integration step.
MPR boundaries: `r ∈ [0, ∞)`, `V` unbounded.

#### 3.3 Inner kernel — `network_chunk(...)`

`@nb.njit` (not `inline="always"`).  Receives all arrays as arguments; all
subnetwork states and history buffers are updated **in-place**.

#### 3.4 Outer loop — `run_network(...)`

Plain Python function. Calls `network_chunk` in a `while` loop (chunked).
Returns `list[(times_arr, data_arr)]` matching `Simulator.run()` format:
- `times_arr`: `(n_chunks,)` float64, mid-chunk time in ms
- `data_arr`: `(n_chunks, n_voi, n_nodes, n_modes)` float32

---

### 4. History Buffers (as-built)

**Shape**: `(n_vars, n_nodes, n_modes, horizon)` where `horizon = max_delay + 1`.

**Write**: `buf[:, :, :, t % horizon] = source_state`

**Read** (inside coupling): `buf[cv, src_node, m, (t - 1 - idelays[ptr] + horizon) % horizon]`

**Pre-fill**: All `horizon` slots initialised to `initial_state` (matching
`NetworkSet.init_projection_buffers()`).

One buffer per projection (not per source subnet). Shared-per-source buffers
are a future optimisation (§9.1).

---

### 5. Stimulus Handling (as-built)

Pre-computation approach:
1. Python calls `stim.get_coupling(step)` for all steps `1..nstep`
2. Results accumulated into `(n_cvar, n_nodes, n_modes, nstep)` float32 array
3. Array passed to `network_chunk`; kernel reads `stim[..., t - 1]` each step

This is the simplest correct approach. Code-generated stimulus functions
(§9.3) remain a future optimisation for long simulations.

---

### 6. Supported Configurations (as-built)

| Feature | Status |
|---------|--------|
| MPR model | ✅ |
| HeunDeterministic | ✅ |
| EulerDeterministic | ✅ |
| HeunStochastic | ✅ |
| EulerStochastic | ✅ |
| Inter-projections (delayed, sparse) | ✅ |
| Intra-projections (delayed, sparse) | ✅ |
| Linear coupling function | ✅ |
| Scaling coupling function | ✅ |
| No coupling function (identity) | ✅ |
| target_scales | ✅ |
| mode_map (inter) | ✅ |
| Stimulus (pre-computed batch) | ✅ |
| Multiple subnetworks | ✅ |
| n_modes == 1 (elided loops) | ✅ |
| n_modes > 1 (general) | ✅ |
| Same-dt constraint enforced | ✅ |
| In-process JIT cache | ✅ |
| compile() / CompiledNetworkFn API | ✅ |
| Sigmoidal / SigmoidalJansenRit cfun | ❌ (future) |
| Sub-stepping (different dt) | ❌ (out of scope) |
| Disk-persistent JIT cache | ❌ (future) |
| nb.prange parallelism | ❌ (future) |
| Other models (JansenRit, FHN, …) | ❌ (future) |

---

### 7. Completed Work — Phase Summary

#### Phase 1: Infrastructure ✅
- `NbHybridBackend(MakoUtilMix)` with `_check_compatibility`, `_analyse`,
  `_build_projection_info`, `_make_projection_buffer`, `_build`
- `NetworkAnalysis`, `SubnetworkInfo`, `ProjectionInfo` dataclasses
- SHA-256 keyed `_COMPILED_FN_CACHE` (module-level, process-lifetime)
- `compile()` / `CompiledNetworkFn` public API
- `run_network()` as one-liner delegation

#### Phase 2: Inter-Projection ✅
- Sparse CSR per-connection delay access inside `@nb.njit(inline="always")`
- Per-connection `idelays[ptr]`, circular buffer modulo indexing
- mode_map transformation (`n_src_modes × n_tgt_modes`)
- cfun pipeline: weighted sum → scale → cfun.pre (identity) → scale → cfun.post
- All four cvar-mapping modes (static branch at code-gen time)
- target_scales support

#### Phase 3: Intra-Projection ✅
- Same template path as inter; `is_inter=False` → no mode_map arg, identity mapping

#### Phase 4: NetworkSet Orchestration ✅
- `network_chunk` `@nb.njit` kernel: coupling → integrate → buf-write → tavg
- Outer Python `run_network` loop: chunked, accumulate, format output

#### Phase 5: Model & Integrator ✅
- MPR dfun generated inline with all global params baked in as `nb.float32`
- HeunDeterministic and EulerDeterministic with in-line boundary conditions
- n_modes==1 mode loop elision (both dfun/integrate paths)

#### Phase 6: Stochastic Integrators ✅
- HeunStochastic and EulerStochastic
- Noise pre-drawn in Python (`(n_vars, n_nodes, n_modes, nstep)` float32)
- Passed to kernel; each step reads `noise[k, i, m, t_abs]`

#### Phase 6b: Stimulus ✅
- Pre-computed batch stimulus array
- Works for both deterministic and stochastic subnetworks

#### Phase 7: Tests & Benchmark ✅
- 25 tests passing: intra, inter (all cvar modes), delays, cfuns, stochastic,
  stimulus, end-to-end multi-subnetwork, compatibility checks
- Benchmark: N=100 nodes × 2 subnets, 1000 steps, cv=10 m/s, 20% density
  → **~6× speedup** over pure Python NetworkSet

#### Performance results (N=20, 1000 steps, cv=10, 20% density)
| N | Speedup | Numba steps/s |
|--:|--------:|--------------:|
| 10 | 21× | 73 000 |
| 20 | 16× | 54 568 |
| 50 | 8× | 20 800 |
| 100 | 6× | 15 392 |
| 200 | 3× | 1 672 |

The declining speedup at large N reflects the coupling kernel being O(nnz) per
step, where Numba's scalar loops compete with Python's NumPy CSR matmul which
benefits from BLAS vectorisation at large N.

---

### 8. Next Stages

#### 8.1 Performance — Coupling Kernel (HIGH PRIORITY)

**Problem**: At N=100+ with realistic density, the coupling inner loop is the
bottleneck.  The current scalar Numba loop is slower than NumPy's BLAS-backed
CSR matmul at large N.

**Option A — `nb.prange` on node loop (easiest)**

Replace `for j in range(n_tgt_nodes)` with `nb.prange` and add `parallel=True`
to the coupling `@nb.njit` decorator.  This parallelises across target nodes.

```python
# In template: compute_coupling_<proj>
@nb.njit(parallel=True, inline="never")   # no inline when parallel
def compute_coupling_<proj>(...):
    for j in nb.prange(n_tgt_nodes):      # parallelised
        ...
```

Requirements:
- `network_chunk` must be `@nb.njit(parallel=False)` — Numba does not support
  nested parallel regions; the caller must not be `parallel`.
- Remove `inline="always"` from coupling functions (incompatible with `parallel=True`).
- This requires `no-GIL` process threads; safe because all arrays are disjoint
  per target node (no write conflict across `j` iterations).

Expected gain: linear in number of CPU cores for the coupling step.

**Option B — Strip epsilon structural zeros before Numba**

The CSR matrices contain structural zeros from the epsilon trick in
`BaseProjection.__init__()`. These contribute `w * val = 0.0 * val = 0` to
the sum but still cost a multiply-add and a buffer read (cache miss).

Strip at `_build_projection_info` time:

```python
# Remove exact structural zeros
mask = p.weights.data != 0.0
pi.weights_data    = p.weights.data[mask]
pi.weights_indices = p.weights.indices[mask]
# Recompute indptr from the pruned COO
```

This reduces `nnz` by the density of structural zeros, directly reducing inner
loop iterations. Important note: this means the Numba and Python paths will have
different sparsity structure, so comparison tests need a tolerance that accounts
for the epsilon contributions being dropped.

**Option C — Float32 buffer reads (already done, confirm)**

History buffers and all state arrays are already `float32`.  Confirm that all
reads inside the coupling loop (`buf[cv, src_node, m, buf_idx]`) are typed
`float32` to avoid any implicit upcasting inside Numba.

#### 8.2 Performance — Disk-Persistent JIT Cache (MEDIUM PRIORITY)

**Problem**: First run after process restart pays full JIT cost (~5s for the
current MPR 2-subnet case). The in-process `_COMPILED_FN_CACHE` does not
survive across Python processes.

**Why `cache=True` failed**: Numba's `NumpyCacheLocator.from_function()` looks
up `inspect.getfile(py_func)`. For `exec()`-generated functions,
`co_filename == '<string>'` → no locator matches → `RuntimeError`.

**Solution: write source to a temp `.py` file, import as a module**

```python
import tempfile, importlib.util, sys

def _build_as_module(source: str, cache_key: str):
    # Write to a stable path keyed by SHA-256
    cache_dir = Path(tempfile.gettempdir()) / "tvb_nb_hybrid_cache"
    cache_dir.mkdir(exist_ok=True)
    mod_path = cache_dir / f"nbhybrid_{cache_key[:16]}.py"
    if not mod_path.exists():
        mod_path.write_text(source)
    # Import as a real module — Numba can now find the file
    spec = importlib.util.spec_from_file_location(f"nbhybrid_{cache_key[:16]}", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_network
```

With `cache=True` on `@nb.njit`, Numba writes a `.nbi`/`.nbc` pair next to the
`.py` file. Subsequent imports in new processes load the compiled native code
directly, skipping JIT (`<50 ms` instead of `~5s`).

Implementation checklist:
- [ ] Add `_build_as_module` path alongside `_build` (switchable via env var)
- [ ] Add `cache=True` to all four `@nb.njit` decorators in the template
- [ ] Add `cache_dir` cleanup utility (prune old SHA256 files)
- [ ] Test: run, kill process, re-run — verify no JIT delay on second run
- [ ] Confirm thread-safety of concurrent writes to cache dir

#### 8.3 Performance — Shared-Per-Source History Buffers (MEDIUM PRIORITY)

**Problem**: When multiple projections share the same source subnetwork, each
projection maintains a separate `(n_vars, n_nodes, n_modes, horizon)` buffer
and writes `src_state` into it every step. For `P` inter-projections from the
same source, that's `P` redundant copies and `P × n_vars × n_nodes × n_modes`
bytes of extra memory.

**Solution**: One buffer per source subnetwork, sized to the maximum horizon
across all outgoing projections from that source.

```python
# Shared buffer shape: (n_vars_src, n_nodes_src, n_modes_src, max_horizon)
# Each projection reads its own idelays against this shared buffer
```

Implementation changes:
- `_run_compiled`: allocate one buffer per source subnet, not one per projection
- `network_chunk` signature: pass source-subnet buffers, not per-projection buffers
- Template: `buf_write` step emits one write per source subnet
- Template: `compute_coupling_*` receives the source subnet's shared buffer

Tradeoff: slightly more complex naming in the template; significant memory savings
for heavily-connected networks (`P > 2` projections from one source).

#### 8.4 Performance — Stimulus Code Generation (LOW PRIORITY)

**Problem**: Pre-computing stimulus for all `nstep` steps in Python allocates an
`(n_cvar, n_nodes, n_modes, nstep)` array and iterates `nstep` times in Python.
For a 1 000 000-step simulation at N=100, this is 400 MB and ~10 s overhead.

**Solution**: For simple stimulus patterns, generate a `@nb.njit` function that
computes the coupling value for step `t` on demand.

```python
# Generated for a sinusoidal stimulus:
@nb.njit(inline="always")
def stim_src_sin(t, dt):
    # amplitude * sin(2*pi*t*dt / period)
    return nb.float32(0.1) * math.sin(2.0 * math.pi * nb.float32(t) * nb.float32(dt) / nb.float32(100.0))
```

The inner kernel calls this instead of indexing the pre-computed array. Only
applies to analytically-describable stimuli (pulse trains, sinusoids). Complex
stimulus arrays remain on the batch-precompute path.

#### 8.5 Functionality — Sigmoidal Coupling Functions (HIGH PRIORITY)

The existing `hybrid/coupling.py` includes `Sigmoidal` and `SigmoidalJansenRit`
which are used in standard TVB simulations. These must be added to the template
cfun dispatch.

Pipeline position is unchanged: `pre` (identity for all current cfuns) → scale
→ `post` (cfun-specific).

```python
# Sigmoidal.post: 1 / (1 + exp(-a*(x - midpoint)))
# SigmoidalJansenRit.post: e0 / (1 + exp(r*(v0 - x)))
```

Template changes:
- Add `"sigmoidal"` and `"sigmoidal_jr"` branches in cfun dispatch
- Extract parameters (`a`, `midpoint`, `e0`, `r`, `v0`) as `nb.float32` scalars
- Add corresponding `elif ct == "sigmoidal":` in `_cfun_type` and `_cfun_params`

#### 8.6 Functionality — Additional Models (MEDIUM PRIORITY)

The MPR-only constraint simplifies code generation (all params baked as scalars)
but limits the backend's usefulness for multi-model hybrid networks. Adding
JansenRit and FitzHugh-Nagumo would cover the most common use cases.

Approach: parameterised model template, same as the existing `nb-dfuns.py.mako`
pattern. Each model contributes a `dfun_<sn_name>(state_vars, coupling_terms)`
section — architecture is already correct, just need the model-specific dfun
expressions.

Implementation checklist per new model:
- [ ] Add model-specific dfun expression strings to the model class
  (following `MontbrioPazoRoxin.state_variable_dfuns` pattern)
- [ ] Add to `_check_compatibility` allowed model list
- [ ] Add to `_analyse` SubnetworkInfo construction (parameter extraction)
- [ ] Add template branch in `dfun_<sn.name>` section of the Mako template
- [ ] Write tests: dfun correctness, end-to-end match Python backend

#### 8.7 Functionality — AfferentCoupling Monitor (MEDIUM PRIORITY)

Currently only `TemporalAverage` (i.e., state variables) is recorded inside the
kernel. The `AfferentCoupling` monitor records the coupling input rather than
the state — useful for diagnosing inter-subnetwork dynamics.

Changes required:
- Add a second accumulator `<sn>_c_tavg` for coupling arrays
- Accumulate after zeroing but after all coupling contributions are added
- Return as a second `(times, data)` tuple in `run_network` output

#### 8.8 Functionality — Disk-Checkpointing and Resumable Runs (LOW PRIORITY)

Long hybrid simulations (millions of steps) need to be restartable. The public
`CompiledNetworkFn.run()` already returns `initial_states` as a parameter, so
the caller can pickle the final state and restart. What is missing:

- A public API to extract current buffer state from a completed run
- A `resume()` method on `CompiledNetworkFn` that accepts state + buffer snapshot
- Integration with TVB's existing `SimulationContinuation` mechanism

#### 8.9 Testing — Extend Coverage

| Gap | Test to add |
|-----|------------|
| Sigmoidal cfun | Compare Numba vs Python for `Sigmoidal` coupling |
| mode_map ≠ identity | Verify multi-mode inter-projection output |
| n_modes > 1 general path | Currently only n_modes=1 is exercised |
| Shared-per-source buffers | Numerical equivalence after buffer refactor |
| Disk cache persistence | Run → kill → rerun, assert JIT skipped |
| prange parallel coupling | Numerical equivalence with parallel=True |
| Large N scaling | Automated speedup regression at N=500 |

#### 8.10 Code Quality

- [ ] Add `__all__` to `nb_hybrid.py`
- [ ] Export `NbHybridBackend` and `CompiledNetworkFn` from `backend/__init__.py`
- [ ] Add type annotations to `_run_compiled` and `_analyse` signatures
- [ ] Add `debug_nojit=True` path to integration tests (faster CI, no JIT overhead)
- [ ] Add `nb_hybrid_plan.md` reference in docstring of `nb_hybrid.py`

---

### 9. Open Questions

1. **`nb.prange` + `inline="always"` incompatibility**: `parallel=True` requires
   `inline="never"` on coupling functions. Need to measure whether inlining by
   the non-parallel fallback matters for correctness or speed after adding prange.

2. **Epsilon structural zeros**: The CSR epsilon trick was deliberately preserved
   to allow direct Python ↔ Numba numerical comparison in tests. Stripping zeros
   (§8.1.B) breaks that property — tests will need tolerance adjustment.

3. **Pre-computed vs code-generated stimulus**: Currently all stimulus is
   pre-computed (good for correctness, bad for memory). The threshold for when
   code-generation pays off depends on the stimulus pattern. A good heuristic:
   if `nstep × n_cvar × n_nodes × n_modes × 4 bytes > 100 MB`, use code-gen.

4. **Shared-per-source buffers + different cvars**: If projection A reads
   `source_cvar=[0]` and projection B reads `source_cvar=[0,1]` from the same
   source, the shared buffer still stores all `n_vars` — no problem. But the
   naming logic in the template needs care to avoid accessing stale data from the
   wrong buffer slot.

5. **Multi-process safety for disk cache (§8.2)**: Two processes compiling the
   same topology concurrently could clobber each other's `.py` file. Use atomic
   write (`os.replace`) and file-level locking (`fcntl.flock` on Linux).

---

### 10. Files Summary

| File | Role |
|------|------|
| `tvb/simulator/backend/nb_hybrid.py` | Backend, dataclasses, cache |
| `tvb/simulator/backend/templates/nb-hybrid-sim.py.mako` | Single Mako kernel template |
| `tvb/tests/library/simulator/backend/test_nb_hybrid.py` | 25 tests + benchmark |
| `tvb/simulator/backend/nb_hybrid_plan.md` | This document |
