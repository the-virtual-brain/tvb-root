# backend_cpp — Project Overview

_Last updated: 2026-04-29_

---

## 1. Goal and Design Philosophy

`backend_cpp` is a JIT-compiled C++ execution backend for the TVB hybrid
simulator. The driving idea is identical to `NbHybridBackend` — lower a Python
`NetworkSet` description into a specialized execution unit — but the target is
native C++ compiled via `pybind11` rather than Numba.

**Key constraints (by design):**
- The Python hybrid API (`NetworkSet`, `Subnetwork`, monitors, …) is the public
  surface. Users never touch C++ directly.
- Python lowers, C++ executes. No per-step Python callbacks.
- Only monitor outputs cross the Python/C++ boundary.
- First optimize for correctness and architecture clarity; performance comes later.

---

## 2. Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│  User Python (TVB hybrid script / demo)                 │
│  NetworkSet / Subnetwork / monitors                      │
└────────────────────────┬────────────────────────────────┘
                         │ configure()
┌────────────────────────▼────────────────────────────────┐
│  CppHybridBackend  (backend.py)                         │
│  .lower()  .compile()  .run()                           │
└──────────┬──────────────────┬───────────────────────────┘
           │                  │
    ┌──────▼──────┐    ┌──────▼──────────────────────────┐
    │ lowering.py │    │  codegen.py                     │
    │             │    │  render_cpp_template()           │
    │ lower_net.. │    │  render_bindings_template()      │
    │ → Spec      │    │  render_cmake_template()         │
    └──────┬──────┘    │  generate_cpp_source()           │
           │           │  build_generated_extension()     │
    ┌──────▼──────┐    └──────────────┬──────────────────┘
    │  spec.py    │                   │
    │ SimulSpec   │       ┌───────────▼───────────────────┐
    │ SubnetSpec  │       │  .build/<hash>/               │
    │ ProjSpec    │       │  ├─ <module>.cpp  (generated) │
    │ MonitorSpec │       │  ├─ <module>_bindings.cpp     │
    └─────────────┘       │  ├─ CMakeLists.txt            │
                          │  └─ runtime/runtime.hpp       │
                          └───────────┬───────────────────┘
                                      │ cmake / c++
                          ┌───────────▼───────────────────┐
                          │  <module>.cpython-*.so        │
                          │  (pybind11 extension)         │
                          └───────────┬───────────────────┘
                                      │ importlib
                          ┌───────────▼───────────────────┐
                          │  CompiledCppNetwork.run()     │
                          │  → (times, data) numpy arrays │
                          └───────────────────────────────┘
```

---

## 3. Module Map

```
backend_cpp/
│
├── __init__.py              Public re-exports: CppHybridBackend, specs, codegen
│
├── spec.py                  Lowered spec dataclasses (Python-to-C++ IR)
│                            SimulationSpec / SubnetworkSpec / ProjectionSpec /
│                            MonitorSpec / StimulusSpec / IntegratorSpec
│                            Content-hash cache key via JSON → SHA-256
│
├── lowering.py              Python → spec conversion
│                            Reuses NbHybridBackend._analyse() for network analysis
│                            Normalizes dtypes / memory layout / shape conventions
│
├── codegen.py               C++ source generation and compilation
│                            Template rendering ({{PLACEHOLDER}} substitution)
│                            CMake-first build, fallback to direct c++ invocation
│                            pybind11 header discovery
│
├── backend.py               CppHybridBackend class (user-facing entry point)
│                            CompiledCppNetwork wrapper (load_module, run)
│
├── runtime/
│   └── runtime.hpp          Fixed, header-only C++ runtime (not generated)
│                            SimulationMetadata / SimulationResult structs
│                            StateBuffer   — (svar × node × mode) typed access
│                            MonitorBuffer — accumulate + write_chunk_average
│                            HistoryBuffer — ring buffer of StateBuffer snapshots
│                            delayed_state_value() — helper for delayed reads
│                            heun_step<Generated>()    — Heun integrator template
│                            run_simulation<Generated>() — full loop entry point
│
├── templates/
│   ├── sim_module.cpp.in    Generated translation unit template
│   │                        Holds GeneratedModel struct with compile-time constants,
│   │                        compute_dfun(), apply_state_constraints(), delegates to runtime
│   ├── module_bindings.cpp.in  pybind11 binding glue (describe, run_simulation,
│   │                            debug_probe_history)
│   └── CMakeLists.txt.in    CMake build description template
│
├── examples/
│   ├── compare_native_single_mpr.py           Python vs Numba vs Native comparison
│   ├── compare_native_delayed_self_feedback.py Delayed coupling correctness check
│   ├── show_runtime_usage.py                  Annotated call chain walkthrough
│   └── debug_lowering_demo.py                 Spec lowering inspection
│
└── notes/
    ├── implementation_plan.md   Step-by-step staged plan (primary planning doc)
    ├── reference_mapping.md     nb_hybrid / tvbk concept → C++ equivalent table
    ├── architecture_brainstorm.md
    ├── code_generation_ideas.md
    └── overview.md              ← this file
```

---

## 4. Data Flow (Single Simulation Run)

```
NetworkSet
    │
    ▼ lower_network_set()
SimulationSpec  ──── cache_key() ──── build dir name
    │
    ▼ render_cpp_template()
sim_module.cpp.in  →  <module>.cpp
    │                 <module>_bindings.cpp
    │                 CMakeLists.txt
    │                 runtime/runtime.hpp  (copied verbatim)
    │
    ▼ cmake --build
<module>.cpython-*.so
    │
    ▼ importlib.util.spec_from_file_location
Python module object
    │
    ▼ module.run_simulation(initial_state, nstep, chunk_size)
      [C++ executes: StateBuffer → HistoryBuffer → heun_step loop → MonitorBuffer]
    │
    ▼ pybind11 return
(times: np.ndarray, data: np.ndarray)
```

---

## 5. Progress Assessment

| Step | Title | Status | Reality |
|------|-------|--------|---------|
| 0 | Freeze scope | ✅ complete | Execution model and milestone path agreed in writing |
| 1 | Map reference implementations | ✅ complete | `notes/reference_mapping.md` produced |
| 2 | Define lowered spec | ✅ complete | All first-milestone spec dataclasses implemented and hashed |
| 3 | Python lowering pass | 🔶 ~85% | Lowering works end-to-end; stimuli spec exists but execution unimplemented; dedicated unit tests still missing |
| 4 | Fixed C++ runtime skeleton | 🔶 ~70% | `StateBuffer`, `HistoryBuffer`, `MonitorBuffer`, `heun_step`, `run_simulation` exist; CSR access and proper array-view types still absent |
| 5 | First end-to-end generated path | 🔶 ~75% | Single-subnet MPR + HeunDeterministic + chunked output works; projection path completely absent |
| 6 | Correctness baseline | 🔶 ~70% | Three automated tests exist (smoke, MPR vs Python/Numba, delayed self-feedback); reproducibility check and timestamp convention doc still missing |
| 7 | Delay buffers + sparse projection | ❌ not started | HistoryBuffer exists but not wired to projection code; CSR traversal not implemented |
| 8 | Monitor expansion | ❌ not started | Only `TemporalAverage` (via chunked average) works today |
| 9 | Generalize model + integrator | ❌ not started | MPR and HeunDeterministic hardcoded throughout |
| 10 | Build / cache / dev tooling | ❌ not started | Hash-based directories accumulate without cleanup |
| 11 | TVB backend selection | 🔶 ~40% | `CppHybridBackend` exists; not wired into TVB `backend_registry` / simulator backend selection flow |
| 12 | Performance validation | ❌ not started | No benchmarks yet |

**First milestone checklist** (from plan):
- [x] Python lowers one supported `NetworkSet` into a C++-ready spec
- [x] Python generates and compiles a simulation-specific C++ extension
- [x] `pybind11` exposes a callable entrypoint
- [x] Full simulation loop runs in C++
- [x] Results returned as NumPy arrays
- [🔶] Numerical output matches reference within tolerance — matches for no-projection case; timestamp convention difference is known but not formally documented

The first milestone is effectively feature-complete for the **no-projection,
single-subnetwork** path. The gap is documentation and test coverage.

---

## 6. Architectural Issues to Address

### 6.1 Template substitution approach will hit a wall

`codegen.py` uses `{{PLACEHOLDER}}` string replacement manually. This is fine
now, but projection code (CSR arrays, per-node delay loops, per-coupling-var
accumulation) will require conditional blocks and loops in the template that
string substitution cannot express cleanly. Consider switching to **Mako**
(already used by `nb_hybrid`) or Jinja2 before adding projections.

### 6.2 Model parameters are hardcoded by name

`render_cpp_template()` explicitly extracts `tau`, `Delta`, `eta`, `J`, `I`,
`cr`, `cv`. Step 9 (generalize models) requires this to be driven by
`SubnetworkSpec.parameter_values` generically. The spec already holds the
parameter dict — the template just needs to iterate it, not name-check it.

### 6.3 `heun_step` hardcodes two state variables

`runtime.hpp:231–232` and `runtime.hpp:241–242` write only `state(0, node, 0)`
and `state(1, node, 0)`. This must loop over `Generated::kNumStateVars` to be
correct for any model with more than two state variables.

### 6.4 `HistoryBuffer` stores full `StateBuffer` copies

Each ring-buffer slot is a complete `StateBuffer` object (`n_svars × n_nodes × n_modes`
doubles). For a 1 000-node network with a 100 ms delay horizon at dt=0.1 the
buffer holds 1 000 `StateBuffer` copies × (2 svars × 1 000 nodes × 1 mode) =
2 × 10⁶ doubles per slot → 2 GB just for the history. A flat ring buffer
indexed as `history[svar * stride + node * modes + mode + delay_slot * frame]`
is far more cache-friendly and avoids the vector-of-objects overhead.

### 6.5 Timestamp convention mismatch

The native runtime records the chunk **midpoint** (`chunk_start + (len-1)/2 × dt`).
The Python `TemporalAverage` monitor records the chunk **endpoint**. The test
verifies the `-0.5 * dt` offset but this is not formally specified anywhere.
Before wiring into the TVB backend selection flow (Step 11) this must be
resolved — either align the native convention or add a post-processing fixup
with clear documentation.

### 6.6 Build cache accumulates without cleanup

Five different hash directories exist under `.build/`. There is no eviction
policy, no `backend_version`-keyed invalidation, and no `--clean` flag. Step 10
should add at minimum an LRU count cap and a version-mismatch purge.

### 6.7 `DelayedSelfFeedbackConfig` is a temporary workaround

The current delayed coupling is injected as a hard-coded conditional in
`compute_dfun`. The correct architecture (Step 7) threads delayed reads through
the projection/coupling machinery. `DelayedSelfFeedbackConfig` should be removed
once projection support lands, not extended.

---

## 7. Suggested Next Steps (Ordered by Impact)

### Priority 1 — Close the first-milestone gap (small, now)

1. **Document the timestamp convention** formally in `notes/` and add a
   corrective fixup comment in `run_simulation` in `runtime.hpp`. Decide whether
   native should match Python (shift to endpoint) or if the caller normalizes.
2. **Add dedicated lowering unit tests** (`test_lowering.py`) that validate
   `SimulationSpec` field values for a small fixture network without requiring
   compilation. This closes the Step 3 deliverable.
3. **Add deterministic reproducibility test** — run the same spec twice and
   assert bit-identical output. Closes the Step 6 reproducibility item.

### Priority 2 — Runtime generalization (prerequisite for projections)

4. **Fix `heun_step` to loop over `kNumStateVars`** instead of hardcoding
   indices 0 and 1. This is a three-line change in `runtime.hpp` and is a
   correctness bug for any future model.
5. **Switch template rendering to Mako or Jinja2** before adding projection
   code. The existing Mako dependency from `nb_hybrid` is already available.
   This unblocks conditional blocks (e.g., `{% if has_projections %}`) and
   per-parameter iteration.
6. **Make model parameter codegen data-driven**: iterate
   `SubnetworkSpec.parameter_values` in the template instead of listing MPR
   parameter names in `codegen.py`.

### Priority 3 — Sparse projection traversal (Step 7)

7. **Implement CSR projection traversal in the runtime** — add a
   `ProjectionBuffer` type holding `weights_data`, `weights_indices`,
   `weights_indptr`, and `idelays` as flat arrays; add a
   `traverse_projection()` helper that accumulates coupling into a
   per-node coupling input buffer.
8. **Replace `HistoryBuffer` with a flat ring buffer** (see §6.4) before
   projection coupling is implemented, because projection access patterns
   require reading many different delay offsets per step.
9. **Wire projection arrays into `run_simulation`** — pass them from Python via
   pybind11 at runtime (not baked into generated constants) since they can be
   large.

### Priority 4 — Wiring and generalization

10. **Register `CppHybridBackend` in the TVB backend selection flow** (Step 11).
    Define which feature flags trigger fallback to the Python path and raise
    `NotImplementedError` with a helpful message for unsupported configs.
11. **Add a second model** (e.g., `Generic2dOscillator`) to prove the
    abstraction generalizes (Step 9). This will force the parameter codegen
    and `heun_step` generalization to be correct.

### Priority 5 — Tooling and performance (Steps 10, 12)

12. **Build cache cleanup**: add a `max_cached_builds` limit and a
    `backend_version`-based purge to `CppHybridBackend`.
13. **Benchmark** native vs Numba for the MPR path on a 100-node network as
    the performance baseline (Step 12).
14. **OpenMP** — only after single-threaded correctness and benchmarks are
    solid.
