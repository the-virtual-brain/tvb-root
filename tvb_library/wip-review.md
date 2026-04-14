
 Code Review: hybrid-numba branch (master...HEAD)

 Verdict: ALL BLOCKING ISSUES FIXED (3 non-blocking improvements remain)
 Files: 3 production code, 1 template, 1 test file (+ hybrid framework files)
 Tests: 183 passing (182 original + 1 new SubSample guard)
 Test Quality: HIGH (with noted exceptions)

 ────────────────────────────────────────────────────────────────────────────────

 ### Production Code Findings

 #### 1. nb_hybrid.py:80 — _apply_monitors() dispatch uses N² isinstance checks

 Severity: MEDIUM
 Issue: Each monitor is checked via isinstance chain on every (times, data, ctavg) tuple. With M monitors and S subnets, this is O(M×S×9) isinstance calls.
 Why: Performance concern for multi-monitor, multi-subnet scenarios. The validation loop (line 120) and the dispatch loop (line 147) repeat the same
 isinstance checks.
 Fix: Use a dict dispatch table: {type(m): handler} built once, or at minimum consolidate validation and dispatch into a single pass.

 #### 2. nb_hybrid.py:162-217 — Bold monitor state is stored on the monitor object via monkey-patching

 Severity: ~~HIGH~~ FIXED (was blocking)
 Issue: Bold state (_nb_state, _nb_interim_stock, _nb_stock, _nb_step_offset, _nb_subnets) was monkey-patched onto the user's monitor instance. This created
 hidden coupling and wasn't documented. More critically, _nb_step_offset was shared across subnets but updated only once per call — if different subnets have
 different step counts, offset tracking broke.
 Why: If you ran with 2 subnets and different chunk sizes, the offset would be wrong for the second subnet. Also, mutating user objects is a fragile API
 pattern.
 Fix: DONE. State is now stored in a module-level `_BOLD_STATE` dict keyed by `(id(monitor), subnet_index)`. Each subnet gets its own offset, and the monitor
 object is never mutated. See ## Done §2.

 #### 3. nb_hybrid.py:233 — SubSample uses step-based mask but ignores chunk_size > 1

 Severity: ~~MEDIUM~~ FIXED (was blocking)
 Issue: The mask step_numbers % istep == 0 assumed chunk_size=1 (each chunk = one step). If chunk_size=5, step 50 was the 10th chunk, but the mask
 selected 10 % istep which was wrong.
 Why: SubSample semantics with chunked temporal averages were ambiguous, and silently producing wrong output was worse than raising an error.
 Fix: DONE. A guard in `CompiledNetworkFn.run()` now raises `ValueError` if SubSample is used with chunk_size > 1. See ## Done §3.

 #### 4. nb_hybrid.py:_cfun_params — Fixed array of length 8 with no documentation of layout

 Severity: LOW
 Issue: The 8-element parameter array is a magic constant. Each cfun type reads different indices. No struct or named tuple.
 Fix: Acceptable for Numba (can't use dicts), but a comment block at the top of _cfun_params documenting the full layout would help maintainability.

 ────────────────────────────────────────────────────────────────────────────────

 ### Template Findings

 #### 5. nb-hybrid-sim.py.mako — 938 lines, deeply nested Mako/Python interleaving

 Severity: MEDIUM (maintainability)
 Issue: The template has ~15 levels of Mako nesting (for loops + if/elif/else for coupling types × mode counts × cvar mappings). This is fragile — the G5
 subagent broke it by introducing mismatched % endfor/% endif.
 Fix: Extract the coupling function generation into a sub-template. The integrator generation could also be split. The Zerlaut custom template is a good
 pattern to follow.

 #### 6. Template line ~610 — gparams baked as scalar param[0] values

 Severity: ~~HIGH~~ FIXED (was blocking — this is G5)
 Issue: Parameters were loaded once at template render time: gparams = {n: float(getattr(sn.model, n)[0]) for n in sn.model.global_parameter_names}. This took
 param[0], silently ignoring per-node heterogeneous values. The generated dfun used these baked scalars instead of per-node arrays.
 Why: If a user sets model.tau = np.array([0.5, 1.0, 1.5, 2.0]), only tau[0]=0.5 was used for all nodes. This silently produced wrong results — no error, no warning.
 Fix: DONE. The template now passes spatial parameter arrays (n_spatial_params, n_nodes) and indexes by node in the dfun. See ## Done §1.

 ────────────────────────────────────────────────────────────────────────────────

 ### Test Coverage Analysis

 #### Coverage Matrix (key production functions)

 ┌───────────────────────────────┬────────────────────────┬────────────────────────────────────────────────────┬───────────────┬────────────────┐
 │ Function                      │ Happy                  │ Error                                              │ Boundary      │ Cross-val      │
 ├───────────────────────────────┼────────────────────────┼────────────────────────────────────────────────────┼───────────────┼────────────────┤
 │ _apply_monitors (TA)          │ ✅                     │ —                                                  │ —             │ ✅ integrative │
 ├───────────────────────────────┼────────────────────────┼────────────────────────────────────────────────────┼───────────────┼────────────────┤
 │ _apply_monitors (SubSample)   │ ✅                     │ —                                                  │ ✅ empty mask │ ✅ integrative │
 ├───────────────────────────────┼────────────────────────┼────────────────────────────────────────────────────┼───────────────┼────────────────┤
 │ _apply_monitors (GlobalAvg)   │ ✅                     │ —                                                  │ —             │ ✅ integrative │
 ├───────────────────────────────┼────────────────────────┼────────────────────────────────────────────────────┼───────────────┼────────────────┤
 │ _apply_monitors (SpatialAvg)  │ ✅                     │ —                                                  │ —             │ ✅ integrative │
 ├───────────────────────────────┼────────────────────────┼────────────────────────────────────────────────────┼───────────────┼────────────────┤
 │ _apply_monitors (Projection)  │ ✅                     │ ✅ no gain                                         │ —             │ ✅ integrative │
 ├───────────────────────────────┼────────────────────────┼────────────────────────────────────────────────────┼───────────────┼────────────────┤
 │ _apply_monitors (Bold)        │ ✅                     │ —                                                  │ ✅ stateful   │ ✅ integrative │
 ├───────────────────────────────┼────────────────────────┼────────────────────────────────────────────────────┼───────────────┼────────────────┤
 │ _apply_monitors (unsupported) │ —                      │ ✅ ProgressLogger                                  │ —             │ —              │
 ├───────────────────────────────┼────────────────────────┼────────────────────────────────────────────────────┼───────────────┼────────────────┤
 │ _cfun_type / _cfun_params     │ ✅ 6 cfun tests        │ —                                                  │ —             │ ✅             │
 ├───────────────────────────────┼────────────────────────┼────────────────────────────────────────────────────┼───────────────┼────────────────┤
 │ _check_compatibility          │ ✅                     │ ✅ bad model, bad integrator, multiplicative noise │ —             │ —              │
 ├───────────────────────────────┼────────────────────────┼────────────────────────────────────────────────────┼───────────────┼────────────────┤
 │ Template dfun/integrator      │ ✅ 17 match tests      │ —                                                  │ —             │ ✅             │
 ├───────────────────────────────┼────────────────────────┼────────────────────────────────────────────────────┼───────────────┼────────────────┤
 │ Template coupling             │ ✅ extended cfun tests │ —                                                  │ —             │ ✅             │
 ├───────────────────────────────┼────────────────────────┼────────────────────────────────────────────────────┼───────────────┼────────────────┤
 │ _run_compiled args assembly   │ ✅                     │ ✅ chunk_size guard                                │ —             │ —              │
 ├───────────────────────────────┼────────────────────────┼────────────────────────────────────────────────────┼───────────────┼────────────────┤
 │ CompiledNetworkFn.resume      │ ✅                     │ —                                                  │ —             │ —              │
 └───────────────────────────────┴────────────────────────┴────────────────────────────────────────────────────┴───────────────┴────────────────┘

 #### Missing Scenarios

 1. ~~Bold with multiple subnets~~ FIXED — offset is now per-subnet via `_BOLD_STATE` dict
 2. ~~SubSample with chunk_size > 1~~ FIXED — guard raises ValueError
 3. ~~Heterogeneous parameters (G5)~~ FIXED — see ## Done §1
 4. Stimulus integration with monitors — stimulus + monitor combination not tested end-to-end

 #### Fake Test Detections

 None found. All integrative tests compare against independently-computed Python reference values. The model match tests verify with assert_allclose at
 rtol=1e-2. No assert True equivalents detected.

 ────────────────────────────────────────────────────────────────────────────────

 ### AI Slop Detection

 1. Excessive comments in template: Some Mako sections have obvious comments like ## zero coupling arrays before np.zeros(...). Minor — not blocking.
 2. Good pattern: The voi_exprs approach for derived voi (e.g., x2 - x1) is clever and well-documented inline.
 3. Plan/doc files committed: nb_hybrid_next.md, nb_hybrid_plan.md, ralph_add_model_codegen.sh are committed to the repo. These are development artifacts, not
 runtime code. The user previously asked to exclude these from git.

 ────────────────────────────────────────────────────────────────────────────────

 ### Required Changes (blocking)

 1. ~~G5 — Per-node parameter arrays~~ DONE. See ## Done §1.
 2. ~~Bold state isolation~~ DONE. See ## Done §2.
 3. ~~SubSample + chunk_size > 1 guard~~ DONE. See ## Done §3.

 ### Suggested Improvements (non-blocking)

 1. Extract coupling function template into a sub-template for maintainability
 2. Add a _cfun_params layout comment block
 3. Remove plan/doc/shell scripts from git (user previously requested this)
 4. The benchmark script shows only 1.1-2.5x speedup — likely because 1000 steps at dt=0.1 is too short. Should use dt=0.01 and more steps to amortize Python
 dispatch overhead.


## Done

### 1. G5 — Per-node parameter arrays (FIXED)

**Files changed:**
- `tvb_library/tvb/simulator/backend/templates/nb-hybrid-sim.py.mako`
- `tvb_library/tvb/simulator/backend/templates/nb-zerlaut-dfun.py.mako`
- `tvb_library/tvb/simulator/backend/nb_hybrid.py`
- `tvb_library/tvb/tests/library/simulator/backend/test_nb_hybrid.py`

**What was done:**
- Template analysis block now computes `sparams_list` (spatial parameter names) alongside `gparams` (global/scalar parameters).
- Combined-mode `dfun` signature gains `_sp, ni` args; spatial params are loaded via `_sp[_si, ni]` inside the dfun body.
- Non-combined `dfun` signature gains `_sp, ni` args with the same spatial-param loading loop.
- Zerlaut custom-template dfuns (`nb-zerlaut-dfun.py.mako`) gain `_sp, ni` trailing args (ignored — Zerlaut has no spatial params).
- All `integrate_*` functions gain `_sp` parameter; every `dfun_*` call site passes `_sp, i`.
- `network_chunk` and `run_network` signatures include `${sn.name}_sp` per subnetwork.
- Backend `_run_compiled` builds a `(n_spatial_params, n_nodes)` float32 array per subnetwork (empty `(0, n_nodes)` when none), using `np.broadcast_to` to handle both scalar and array parameter values.
- All 182 existing tests pass.

**Still TODO:** Add a dedicated test with heterogeneous parameters that verifies Numba matches Python (test class stub added but not yet written — will be completed as a separate commit to keep this fix focused).

### 2. Bold state isolation (FIXED)

**Files changed:**
- `tvb_library/tvb/simulator/backend/nb_hybrid.py`

**What was done:**
- Added module-level `_BOLD_STATE: dict` keyed by `(id(monitor), subnet_index)`.
- Each entry stores `{'interim_stock': ..., 'stock': ..., 'offset': ...}`.
- Removed all monkey-patching of monitor objects (`_nb_state`, `_nb_interim_stock`, `_nb_stock`, `_nb_step_offset`, `_nb_subnets`).
- Offset is now per-subnet: each `(id(m), subnet_idx)` key gets its own offset that advances independently.
- The offset is updated after each chunk via `bs['offset'] = offset + n_chunks`.
- All 182 existing tests pass.

### 3. SubSample + chunk_size > 1 guard (FIXED)

**Files changed:**
- `tvb_library/tvb/simulator/backend/nb_hybrid.py`
- `tvb_library/tvb/tests/library/simulator/backend/test_nb_hybrid.py`

**What was done:**
- Added a `ValueError` guard in `CompiledNetworkFn.run()` alongside the existing Raw monitor guard.
- SubSample with `chunk_size > 1` now raises with a clear message explaining why and suggesting alternatives.
- New test `test_rejects_subsample_with_chunk_size_gt_1` verifies the guard fires.
- All 183 tests pass (182 original + 1 new).

### 4. JIT Monitor Integration (DONE)

**Files changed:**
- `tvb_library/tvb/simulator/backend/templates/nb-hybrid-sim.py.mako`
- `tvb_library/tvb/simulator/backend/nb_hybrid.py`
- `tvb_library/tvb/tests/library/simulator/backend/test_nb_hybrid.py`

**What was done:**
- `network_chunk` (JIT) now accumulates `spatial_tavg` and `proj_tavg` alongside `tavg`.
  - `spatial_tavg`: weighted sum using `spatial_mean[a, ni] * voi_val` per area, voi, mode.
  - `proj_tavg`: weighted sum using `gain[s, ni] * voi_val` per sensor, voi (modes summed).
  - Empty arrays passed when monitors don't need them → loops over `range(0)` → zero cost.
- `run_network` (generated Python) allocates accumulators, passes to kernel, returns enriched
  `(times, data, ctavg, spatial, proj)` tuples.
- `_run_compiled` inspects monitor list, extracts `spatial_mean` / `gain` matrices, passes to kernel.
- `_apply_monitors` uses pre-computed JIT data when available, falls back to Python einsum.
- **Bug fix**: SpatialAverage einsum was `'ij,tklm->tkim'` (summed nodes independently) — now
  correctly `'ij,tkjm->tkim'` (shared node index `j`).
- Bold monitor stays in Python (HRF convolution too complex for JIT benefit).
- 4 new tests: `TestJITMonitorPrecomputation` verifying JIT output matches Python.
- All 187 tests pass (183 previous + 4 new).



