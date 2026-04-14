
 Code Review: hybrid-numba branch (master...HEAD)

 Verdict: REQUEST CHANGES (3 blocking issues, several non-blocking improvements)
 Files: 3 production code, 1 template, 1 test file (+ hybrid framework files)
 Tests: 182 passing
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

 Severity: HIGH (blocking)
 Issue: Bold state (_nb_state, _nb_interim_stock, _nb_stock, _nb_step_offset, _nb_subnets) is monkey-patched onto the user's monitor instance. This creates
 hidden coupling and isn't documented. More critically, _nb_step_offset is shared across subnets but updated only once per call — if different subnets have
 different step counts, offset tracking breaks.
 Why: If you run with 2 subnets and different chunk sizes, the offset will be wrong for the second subnet. Also, mutating user objects is a fragile API
 pattern.
 Fix: Store Bold state in a separate dict keyed by (id(monitor), subnet_index) inside _apply_monitors, or return a state object the user passes back in.

 #### 3. nb_hybrid.py:233 — SubSample uses step-based mask but ignores chunk_size > 1

 Severity: MEDIUM
 Issue: The mask step_numbers % istep == 0 assumes chunk_size=1 (each chunk = one step). If chunk_size=5, step 50 is the 10th chunk, but the mask selects 10 %
 istep which is wrong.
 Why: SubSample semantics with chunked temporal averages are ambiguous, but silently producing wrong output is worse than raising an error.
 Fix: Either raise if SubSample is used with chunk_size > 1, or compute the step range from times instead.

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

 Severity: HIGH (blocking — this is G5)
 Issue: Parameters are loaded once at template render time: gparams = {n: float(getattr(sn.model, n)[0]) for n in sn.model.global_parameter_names}. This takes
 param[0], silently ignoring per-node heterogeneous values. The generated dfun uses these baked scalars instead of per-node arrays.
 Why: If a user sets model.tau = np.array([0.5, 1.0, 1.5, 2.0]), only tau[0]=0.5 is used for all nodes. This is silently wrong — no error, no warning.
 Fix: This is the G5 gap. The template should pass parameter arrays (n_params, n_nodes) and index by node. The subagent attempted this but broke the template
 syntax.

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

 1. Bold with multiple subnets — the _nb_step_offset is shared, which may break if subnets have different step counts
 2. SubSample with chunk_size > 1 — not tested, would produce wrong output
 3. Heterogeneous parameters (G5) — no test verifying per-node params are used
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

 1. G5 — Per-node parameter arrays: The template currently bakes param[0] for all nodes. This silently produces wrong results for heterogeneous parameters.
 Needs template change to pass (n_params, n_nodes) arrays and index by ni in the integrator loop. Add a test with heterogeneous params that verifies Numba
 matches Python.
 2. Bold state isolation: Monkey-patching state onto user monitor objects is fragile and buggy with multiple subnets. Store Bold accumulation state
 separately.
 3. SubSample + chunk_size > 1 guard: Add a check that raises if SubSample is used with chunk_size > 1, since the step-based mask assumes 1 step per chunk.

 ### Suggested Improvements (non-blocking)

 1. Extract coupling function template into a sub-template for maintainability
 2. Add a _cfun_params layout comment block
 3. Remove plan/doc/shell scripts from git (user previously requested this)
 4. The benchmark script shows only 1.1-2.5x speedup — likely because 1000 steps at dt=0.1 is too short. Should use dt=0.01 and more steps to amortize Python
 dispatch overhead.



