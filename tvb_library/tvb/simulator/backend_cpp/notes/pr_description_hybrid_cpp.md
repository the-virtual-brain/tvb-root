# PR: Add Experimental C++ JIT Backend for TVB Hybrid Simulations

**Branch:** `hybrid-cpp` → **Target:** `hybrid-numba`

---

## Summary

This branch adds an experimental `backend_cpp` package that provides a native
JIT-compiled C++ execution path for the TVB hybrid simulator.

The existing Python hybrid API is the unchanged public surface: users still define
`NetworkSet`, `Subnetwork`, projections, integrators, and monitors in Python.
The new backend lowers that configuration into a C++-ready spec, generates
simulation-specific C++ source, compiles it as a `pybind11` extension at runtime,
and runs the full simulation loop in native code. Only monitor outputs cross back
to Python as NumPy arrays.

This is additive — the existing `NbHybridBackend` (Numba) path is unaffected and
remains the default. The C++ backend is a parallel opt-in path for users who want
native execution.

**Scope:** 47 commits, ~15,000 lines added across the new `backend_cpp` package,
test suites, examples, and documentation.

---

## Architecture

```
User Python (NetworkSet / Subnetwork / projections / monitors)
    │
    ▼  CppHybridBackend.lower()
SimulationSpec  (Python dataclasses, SHA-256 cache key)
    │
    ▼  CppHybridBackend.compile()
Mako templates → generated C++ source + CMake build files
    │
    ▼  cmake --build  (or fallback: direct c++ invocation)
.build/<hash>/<module>.cpython-*.so   (pybind11 extension)
    │
    ▼  importlib.util.spec_from_file_location()
module.run_simulation(initial_states, nstep, chunk_size, projections)
    │
    ▼
list[(times, data)]  ← one tuple per subnetwork, NumPy arrays
```

### New modules

| Module | Role |
|--------|------|
| `spec.py` | Lowered intermediate representation (`SimulationSpec`, `SubnetworkSpec`, `ProjectionSpec`, `MonitorSpec`, `IntegratorSpec`, `StimulusSpec`) |
| `lowering.py` | `NetworkSet` → spec conversion; scope/compatibility gates |
| `codegen.py` | Mako template rendering, `_CppExprGen` AST→C++ translator, CMake integration, pybind11 discovery |
| `backend.py` | `CppHybridBackend` (lower + compile), `CompiledCppNetwork` (run) |
| `runtime/runtime.hpp` | Fixed header-only C++ runtime: `StateBuffer`, `HistoryBuffer`, `MonitorBuffer`, Heun stepping, projection traversal, result packaging |

---

## Main Changes

### C++ backend package (`tvb/simulator/backend_cpp/`)

- `spec.py` — Python dataclasses forming a stable, serialisable intermediate
  representation between TVB objects and the C++ code generator.
- `lowering.py` — Converts a configured `NetworkSet` into a complete `SimulationSpec`
  with compatibility gates (rejects non-`HeunDeterministic` integrators, models
  without `state_variable_dfuns`, mismatched `dt` values).
- `codegen.py` — Renders Mako templates into per-simulation C++ source, generates
  `CMakeLists.txt`, invokes CMake (or a direct compiler fallback), caches generated
  modules by SHA-256 content hash, and dynamically imports the built extension.
- `backend.py` — User-facing `CppHybridBackend.lower()` / `.compile()` / `.run()`
  entry points and `CompiledCppNetwork` wrapper with `debug_summary()`.
- `runtime/runtime.hpp` — Fixed, reusable header-only C++ runtime:
  - `StateBuffer` — typed, index-checked state storage
  - `HistoryBuffer` — ring-buffer for delayed state access
  - `MonitorBuffer` — chunked temporal-average accumulation
  - `ProjectionArrays` + `accumulate_projection()` — sparse CSR traversal
  - Deterministic Heun stepping
  - Result packaging as `pybind11`-compatible structs

### Mako templates (`templates/`)

| Template | Generates |
|----------|-----------|
| `sim_module.cpp.mako` | `GeneratedModel` struct, `compute_dfun()`, Heun dispatch |
| `module_bindings.cpp.mako` | `pybind11` interface: `describe()`, `run_simulation()`, `debug_probe_history()` |
| `CMakeLists.txt.mako` | CMake config with pybind11 headers and compiler flags |

Expression translation (`_CppExprGen`): converts Python expression strings from
`state_variable_dfuns` to C++ — maps `exp()` → `std::exp()`, state variables to
`state(svar_idx, node, 0)`, model parameters to `param_at(kParam_name, node)`, and
handles arbitrary binary operations and function calls through an AST visitor.

### Supported models

Any model that exposes `state_variable_dfuns` (expression-based derivative
definitions) and uses `HeunDeterministic` can be lowered and compiled. Currently
exercised and tested:

- `MontbrioPazoRoxin` (2 state variables) — primary reference
- `ReducedSetFitzHughNagumo` (4 state variables)
- `JansenRit`
- `FitzHughNagumo`
- Linear test model

### Output format

```python
results = compiled.run(
    initial_states=[ic_cortex, ic_thalamus],
    nstep=1000,
    chunk_size=10,
)

(times_ctx, data_ctx) = results[0]    # (n_chunks,), (n_chunks, n_voi, n_nodes, n_modes)
(times_thal, data_thal) = results[1]
```

The explicit mode axis is preserved. Callers that want a collapsed output should
reduce over the last axis explicitly:

```python
xi_summed = data_thal[:, 0, :, :].sum(axis=-1)
```

### Changes to existing hybrid code

- `tvb.simulator.hybrid.simulator.py` — added `backend` parameter (`"python"` or
  `"numba"`), `_run_numba()` dispatch path, and shared `_resolve_ics()` helper.
- `tvb.simulator.backend.nb_hybrid.py` — added `_compute_chunk_size()` (GCD-based
  auto-computation from monitor periods) and `_aggregate_chunks_to_period()`
  (chunk-to-period downsampling with midpoint time alignment).

### Tests

| Suite | Location | Count |
|-------|----------|-------|
| C++ backend end-to-end | `tests/library/simulator/backend_cpp/test_cpp_hybrid_backend.py` | 5 classes / 43 tests |
| Lowering unit tests | `tests/library/simulator/backend_cpp/test_lowering.py` | 265 lines |
| Numba hybrid validation | `tests/library/simulator/backend/test_nb_hybrid_validate.py` | 37 tests |
| Numba hybrid extended | `tests/library/simulator/backend/test_nb_hybrid.py` | +995 lines |
| UX improvements | `tests/library/simulator/backend/test_ux_improvements.py` | 30 tests |
| Stimulus equation parity | `tests/library/equation_tests/test_stimulus_equation_parity.py` | 37 tests |

### Examples and documentation

- `examples/cpp_hybrid_getting_started.py` — minimal working demo
- `examples/cpp_hybrid_getting_started_explained.ipynb` — annotated notebook
- `examples/compare_native_single_mpr.py` — Python vs Numba vs C++ comparison
- `examples/compare_native_delayed_self_feedback.py` — delayed coupling validation
- `examples/benchmark_single_subnetwork_cpp.py` — compile-time vs run-time profiling
- `examples/benchmark_hybrid_simulator_vs_numba.py` — multi-subnet comparison
- `examples/visualize_cpp_models_timeseries.py` — C++ vs reference plots
- `notes/overview.md` — architecture, module map, data-flow diagram, progress
- `notes/implementation_plan.md` — 12-step incremental plan with per-step status
- `notes/reference_mapping.md` — conceptual mapping from `nb_hybrid` and `tvbk`

---

## Validation

Run the C++ backend tests:

```bash
pytest tvb_library/tvb/tests/library/simulator/backend_cpp/ -v
# 43 passed
```

Numerical correctness: C++ output matches Python reference within ~5 × 10⁻¹⁵
for noise-free deterministic runs. Deterministic reproducibility: bit-identical
results across repeated runs from the same initial conditions. A known timestamp
convention difference of 0.5 × dt between the native temporal-average midpoint
and the Python `TemporalAverage` endpoint is documented in the tests.

---

## Finished Tasks

### Package and API
- [x] Add `backend_cpp` package skeleton with `__init__.py`
- [x] Add user-facing `CppHybridBackend` (lower / compile entry points)
- [x] Add `CompiledCppNetwork` wrapper for dynamically imported extensions
- [x] Add `debug_summary()` for inspecting generated source paths

### Spec and lowering
- [x] Define `SimulationSpec`, `SubnetworkSpec`, `ProjectionSpec`, `MonitorSpec`, `IntegratorSpec`, `StimulusSpec`
- [x] Ensure spec contains only C++-friendly scalars, enums, and contiguous NumPy arrays
- [x] Implement `lower_network_set()` reading subnetworks, projections, parameters, delays, monitors
- [x] Add compatibility gates (model must have `state_variable_dfuns`; integrator must be `HeunDeterministic`; all subnets must share the same `dt`)
- [x] Add deterministic SHA-256 cache key from spec payload JSON
- [x] Add lowering unit tests (`test_lowering.py`)

### Code generation
- [x] Implement `_CppExprGen` AST visitor (Python expressions → C++ expressions)
- [x] Add Mako template for simulation module (`sim_module.cpp.mako`)
- [x] Add Mako template for pybind11 bindings (`module_bindings.cpp.mako`)
- [x] Add Mako template for CMake build (`CMakeLists.txt.mako`)
- [x] Write generated sources to `.build/<hash>/` directory
- [x] Add CMake-first native extension build path
- [x] Add direct C++ compiler fallback and improved error handling

### C++ runtime
- [x] Add `runtime/runtime.hpp` (fixed, header-only, not per-simulation)
- [x] Add `StateBuffer` for typed, index-checked state storage
- [x] Add `HistoryBuffer` ring buffer for delayed state access
- [x] Add `MonitorBuffer` for chunked temporal-average accumulation
- [x] Add `ProjectionArrays` and `accumulate_projection()` for sparse CSR traversal
- [x] Implement deterministic Heun stepping

### Simulation features
- [x] Single-subnetwork simulation loop generation
- [x] Multi-subnetwork simulation loop generation
- [x] Zero-delay projection reads (direct coupling accumulation)
- [x] Delayed projection reads through history buffer
- [x] Intra-subnetwork projection support (within a single subnet)
- [x] Inter-subnetwork projection support (across subnets)
- [x] Delayed self-feedback mechanism
- [x] Return one `(times, data)` tuple per subnetwork
- [x] Preserve explicit mode axis in returned C++ data `(n_chunks, n_voi, n_nodes, n_modes)`

### Model support
- [x] `MontbrioPazoRoxin` (2 state variables) — fully tested
- [x] `ReducedSetFitzHughNagumo` (4 state variables)
- [x] `JansenRit`
- [x] `FitzHughNagumo`
- [x] Generic support for any model exposing `state_variable_dfuns`

### Correctness and tests
- [x] Numerical comparison tests against Python and Numba references (`test_cpp_hybrid_backend.py`)
- [x] Deterministic reproducibility test (bit-identical results from same seed)
- [x] Intra-projection correctness test
- [x] Zero-weight projection equivalence test
- [x] Lowering unit tests (model parameter extraction, projection CSR flattening, cache key determinism)
- [x] Document timestamp convention difference (0.5 × dt offset) between native and Python `TemporalAverage`

### Existing backend enhancements
- [x] Add `backend` parameter to `hybrid.Simulator` (`"python"` / `"numba"`)
- [x] Add `_run_numba()` dispatch path in `hybrid.Simulator`
- [x] Add `_compute_chunk_size()` (GCD-based auto-computation from monitor periods)
- [x] Add `_aggregate_chunks_to_period()` (chunk-to-period downsampling with midpoint time alignment)
- [x] Expanded Numba backend tests (`test_nb_hybrid.py`, `test_nb_hybrid_validate.py`)
- [x] Stimulus equation parity tests (`test_stimulus_equation_parity.py`)
- [x] UX improvement tests (`test_ux_improvements.py`)

### Examples and documentation
- [x] Getting-started Python example (`cpp_hybrid_getting_started.py`)
- [x] Explained getting-started Jupyter notebook (`cpp_hybrid_getting_started_explained.ipynb`)
- [x] Python vs Numba vs C++ comparison example (`compare_native_single_mpr.py`)
- [x] Delayed self-feedback validation example (`compare_native_delayed_self_feedback.py`)
- [x] Runtime call chain annotation example (`show_runtime_usage.py`)
- [x] Lowering debug demo (`debug_lowering_demo.py`)
- [x] Single-subnet benchmark (`benchmark_single_subnetwork_cpp.py`)
- [x] Multi-subnet benchmark vs Numba (`benchmark_hybrid_simulator_vs_numba.py`)
- [x] C++ vs reference timeseries visualisation (`visualize_cpp_models_timeseries.py`)
- [x] Multi-model timeseries visualisation (`visualize_hybrid_models_timeseries.py`)
- [x] Architecture overview note (`notes/overview.md`)
- [x] Incremental implementation plan (`notes/implementation_plan.md`)
- [x] Conceptual reference mapping from `nb_hybrid` and `tvbk` (`notes/reference_mapping.md`)

---

## Remaining Tasks

### API decisions
- [ ] Decide whether `compiled.run()` should return a plain tuple (not a list) for
      single-subnetwork simulations, for ergonomic consistency with the Numba path.
- [ ] Decide whether mode reduction should be caller-managed or exposed as an
      optional backend/API parameter.

### Feature gaps
- [x] Complete inter-subnetwork delayed projection support in the C++ runtime:
      restructured the simulation loop into 4 explicit phases (zero coupling /
      accumulate intra / accumulate inter / integrate / push history) matching the
      Numba backend's ordering, so all inter-projection reads consistently see the
      t-1 state of every source subnet regardless of subnet traversal order.
- [x] Add `Raw` and `RawVoi` monitor support: both map to the existing
      `TemporalAverage` code path with `chunk_size=1` (matching the Numba
      backend — per-step VOI output, shape `(nstep, n_voi, n_nodes, n_modes)`).
      Monitor type validation added to `_validate_spec()` with a clear error for
      unsupported types.
- [x] Add `AfferentCoupling` and `AfferentCouplingTemporalAverage` monitor support:
      the generated module always computes a `ctavg` buffer (coupling temporally
      averaged over each chunk, matching Numba's pattern); `backend.py` returns
      `(times, ctavg)` for these monitors and `(times, data)` for all others.
      `AfferentCoupling` (base) forces `chunk_size=1`; `AfferentCouplingTemporalAverage`
      uses the caller-supplied `chunk_size` computed from its period.
- [x] Add `HeunStochastic` integrator support: noise pre-generated on the Python
      side as `sqrt(2*nsig*dt)*randn` shaped `(n_vars, n_nodes, n_modes, nstep)`
      (matching the Numba backend); `heun_step_stochastic` in `runtime.hpp` applies
      the same Wiener increment to both predictor and corrector (additive noise)
      for standard single-mode models; combined-mode (multi-mode) stochastic Heun
      adds noise at the same two points inside the inline per-node block, with the
      full `noise[sv, node, mode, step]` indexing.
- [ ] Implement native stimulus execution (spec dataclasses exist; C++ execution
      path not yet complete).
- [ ] Add Euler integrator support (Heun is currently the only option).
- [ ] Support models using custom Numba templates (e.g., Zerlaut) through a
      separate codegen path or explicit rejection message.

### Build and tooling
- [ ] Add true rebuild avoidance — reuse cached `.so` when spec hash matches,
      without re-invoking CMake.
- [ ] Add LRU eviction or version-based purge for `.build/` cache directories.
- [ ] Add verbose compile logging mode.
- [ ] Remove any accidentally committed generated build artifacts before merge.

### Integration
- [ ] Wire `CppHybridBackend` into the TVB backend selection/registry mechanism
      so users can select it without custom scripts.
- [ ] Add stronger validation messages for unsupported model/integrator combinations
      (currently raises `NotImplementedError` with minimal context).
- [ ] Document system requirements (C++17 compiler, CMake ≥ 3.12, pybind11) in
      a user-facing note.

### Testing and CI
- [ ] Run the full TVB test suite (not only `backend_cpp/` tests) to check for
      regressions in existing paths.
- [ ] Add CI coverage for the C++ backend build path (requires compiler and
      pybind11 availability in the CI environment).
- [ ] Add tests for chunking and monitor-period semantics edge cases (chunk_size
      not a divisor of nstep, monitor period larger than chunk_size).

### Performance
- [ ] Profile coupling traversal, monitor accumulation, and the Python/C++ boundary
      on large-network cases.
- [ ] Evaluate OpenMP parallelism for the node loop after single-thread correctness
      is solid.
- [ ] Measure and document memory layout impact (history buffer stores full
      `StateBuffer` copies; flat ring buffer may be more efficient for large delays).

---

## Build Requirements

- Python ≥ 3.8
- C++17 compiler (`g++`, `clang++`, or MSVC)
- CMake ≥ 3.12 (optional; direct `c++` fallback available)
- pybind11 headers (auto-discovered via `sysconfig` and standard locations)
- NumPy

---

## Notes on Known Limitations

- Only `HeunDeterministic` integrator is supported; stochastic integrators are out
  of scope for this PR.
- Only `TemporalAverage` monitor is fully implemented in the native path.
- Models using custom Numba expression templates (e.g., Zerlaut) are rejected by
  the compatibility gate.
- Timestamp convention: native `TemporalAverage` records chunk midpoints while the
  Python `TemporalAverage` records chunk endpoints (~0.5 × dt offset at the same
  chunk boundary). Documented and accepted for now.
- Build cache accumulates indefinitely; no automatic cleanup.
