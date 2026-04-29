# Hybrid C++ Backend Implementation Plan

## Goal

Build a hybrid simulator backend where:

- Python remains the configuration and user API layer.
- A user-provided Python hybrid script/configuration is the source of truth.
- The full simulation loop runs in C++.
- The Python side lowers TVB objects into a C++-ready specification.
- C++ code generation emits specialized code for the configured simulation.
- `pybind11` exposes the compiled runtime back to Python.
- Only monitor outputs and other final run results are returned to Python, so the
  Python/C++ boundary cost stays minimal.

This plan is intentionally incremental. We do not move to the next step until the
current step has explicit deliverables completed and checked off.

## Status Convention

- `[ ]` not started
- `[~]` in progress
- `[x]` completed
- `BLOCKED:` use this inline when a step cannot proceed

## Ground Rules

- Keep Python as the orchestration layer and C++ as the execution layer.
- The Python hybrid API remains the public frontend; users should not need to
  construct C++ objects directly for normal use.
- Do not try to translate arbitrary Python code to C++ in the first iteration.
- Reuse the existing hybrid analysis/codegen decomposition from
  `tvb/simulator/backend/nb_hybrid.py`.
- Use `tvbk` as a structural reference for C++ runtime layout, stepping kernels,
  and compiled extension organization, but use `pybind11` for bindings.
- First optimize for correctness and architecture clarity, then performance.

## Current Architectural Direction

### Stable C++ runtime

The non-generated C++ runtime should own:

- simulation loop
- state storage
- delay/ring buffers
- sparse projection traversal
- monitor accumulation
- integrator kernels
- result packaging for Python

### Generated C++ code

The generated code should specialize:

- model derivative functions
- coupling application
- network-specific constants and dimensions
- dispatch glue for the exact `NetworkSet`

This keeps the generated surface small and the runtime testable.

## Execution Plan

### Step 0. Freeze Scope and Terminology

Status: `[x]`

Purpose:
- Define the exact boundaries of the first usable version so implementation does
  not drift.

Tasks:
- [x] Confirm the first supported execution path:
  - one hybrid model family for the first milestone
  - one deterministic integrator
  - one coupling type
  - one monitor type
- [x] Confirm that the first version uses runtime compilation of generated C++.
- [x] Confirm that Python only configures and collects outputs, with no per-step
  Python callbacks.
- [x] Confirm terminology:
  - `spec` means lowered Python-side simulation description
  - `runtime` means fixed reusable C++ engine
  - `generated module` means simulation-specific compiled extension

Deliverables:
- This file updated if any scope decision changes.
- The agreed execution model is written down explicitly.

Exit criteria:
- The first supported path is written down explicitly and agreed.

Notes:
- Execution model agreed on 2026-04-20:
  - users configure simulations through the existing Python hybrid API
  - Python lowers the configured simulation into a C++-ready spec
  - a generated C++ module is compiled from that spec
  - the full simulation executes in C++
  - only monitor outputs and final results cross back to Python
- First milestone path agreed on 2026-04-20:
  - user API source: Python hybrid demos/scripts under `tvb_documentation/demos/hybrid`
  - model: `MontbrioPazoRoxin`
  - integrator: `HeunDeterministic`
  - coupling: `Linear`
  - monitor: `TemporalAverage`
  - network shape target: start with the simplest `NetworkSet`, then one delayed
    inter-projection path matching the benchmark demo
  - correctness reference: Python hybrid path first, then `NbHybridBackend`

---

### Step 1. Inspect and Map Existing Reference Implementations

Status: `[x]`

Purpose:
- Identify exactly what to reuse from TVB hybrid backend and from `tvbk`.

Tasks:
- [x] Extract the lowering/analysis responsibilities from
  `tvb/simulator/backend/nb_hybrid.py`.
- [x] Extract the generated-kernel boundaries from
  `tvb/simulator/backend/templates/nb-hybrid-sim.py.mako`.
- [x] Extract runtime organization ideas from `tvbk`:
  - model kernels
  - integrator kernels
  - stepping loops
  - extension layout
- [x] Write a concise mapping table:
  - `nb_hybrid` concept -> C++ backend equivalent
  - `tvbk` concept -> reusable runtime/reference idea

Deliverables:
- A short architecture mapping note in `notes/` or appended here.
- Include the user-facing demo paths that will serve as initial fixtures.

Completion note:
- Satisfied by `notes/reference_mapping.md`, which captures the conceptual reuse
  from `nb_hybrid.py`, `nb-hybrid-sim.py.mako`, `tvbk`, and the initial hybrid
  demo fixtures.

Exit criteria:
- We know which pieces are copied conceptually, which are adapted, and which are
  new.

---

### Step 2. Define the Python-to-C++ Lowered Spec

Status: `[x]`

Purpose:
- Create a stable intermediate representation between TVB Python objects and the
  C++ backend.

Tasks:
- [x] Define Python dataclasses or plain structures for:
  - `SimulationSpec`
  - `SubnetworkSpec`
  - `ProjectionSpec`
  - `IntegratorSpec`
  - `MonitorSpec`
  - optional `StimulusSpec`
- [x] Ensure the spec contains only C++-friendly data:
  - scalar values
  - strings/enums
  - contiguous NumPy arrays
  - integer maps and dimensions
- [x] Mirror the useful parts of `NetworkAnalysis` but remove Python object
  dependencies.
- [x] Decide what is baked into generated code versus passed as runtime arrays.
- [x] Define a stable hash key for spec-based code generation cache.

Deliverables:
- `spec.py` or equivalent module with the lowered-spec schema.
- A documented list of fields included in the first milestone.

Completion note:
- Implemented in `spec.py`; the first-milestone fields are captured directly by
  the spec dataclasses and their serialized payload/hash representation.

Exit criteria:
- A `NetworkSet` can be lowered into a complete spec without requiring C++ yet.

---

### Step 3. Implement a Python Lowering Pass

Status: `[x]`

Purpose:
- Convert TVB runtime objects into the spec from Step 2.

Tasks:
- [x] Build a lowering function that reads:
  - subnetworks
  - inter/intra projections
  - model parameters
  - initial states
  - delays
  - coupling mappings
  - monitor configuration
- [x] Reuse compatibility checks from the existing hybrid backend where possible.
- [x] Normalize all arrays:
  - dtype
  - memory layout
  - shape conventions
- [x] Decide initial treatment of stimuli:
  - either unsupported in milestone 1
  - or precomputed on Python side and passed as arrays

Deliverables:
- Lowering function with deterministic output.
- Tests that validate spec contents for a small example network.

Progress note:
- `lowering.py` implements the lowering pass and reuses
  `NbHybridBackend._analyse()` as the reference analysis path.
- Scope compatibility is enforced by a local C++-backend gate while still using
  `NbHybridBackend._analyse()` for reference lowering. The gate currently
  accepts configured subnetworks using `HeunDeterministic` and models exposing
  expression-based `state_variable_dfuns`.
- Stimuli are represented structurally in the spec but are explicitly outside
  the current native execution scope.
- Dedicated automated lowering tests exist in
  `tvb/tests/library/simulator/backend_cpp/test_lowering.py`.

Exit criteria:
- For the first supported path, the lowered spec is sufficient to run a
  simulation without touching original TVB objects.

---

### Step 4. Create the Fixed C++ Runtime Skeleton

Status: `[~]`

Purpose:
- Establish the reusable C++ execution core before introducing large-scale code
  generation.

Tasks:
- [x] Create C++ runtime directories and build structure.
- [~] Add core runtime types for:
  - lightweight projection array views
  - state buffers
  - delay/ring buffers
  - CSR connectivity access
  - output buffers
- [x] Implement a minimal simulation loop API:
  - initialize state
  - step for `nstep`
  - accumulate one monitor type
  - package outputs
- [x] Add `pybind11` module scaffolding.
- [x] Build the extension with CMake.

Deliverables:
- Compilable C++ runtime skeleton.
- A trivial `pybind11` extension importable from Python.

Progress note:
- The generated module path already includes `pybind11` bindings and native
  build support.
- A first reusable fixed runtime layer now exists in `runtime/runtime.hpp`,
  holding the shared simulation metadata/result types, Heun stepping loop,
  monitor accumulation, and result packaging for the current narrow path.
- The runtime now has explicit `StateBuffer` and `MonitorBuffer`
  abstractions instead of relying only on anonymous flat vectors in the step
  loop.
- A first `HistoryBuffer` ring-buffer abstraction now exists and the runtime
  writes state snapshots into it each step.
- Delayed reads are implemented at the runtime level and covered by a small
  backend_cpp test probe.
- `ProjectionArrays` and `accumulate_projection()` provide the first runtime
  CSR projection traversal path. The implemented scope is deliberately narrow:
  single-subnetwork, single-mode intra-projections with one source cvar feeding
  one target coupling slot. Inter-subnetwork projections, mode maps,
  multi-cvar mappings, target scales, and coupling-function transforms are not
  implemented in the C++ runtime yet.
- Generated modules now include that fixed runtime and delegate
  `describe()`/`run_simulation()` into it instead of owning the full loop.
- The runtime is still minimal and header-only; broader array/view abstractions,
  multi-subnetwork buffers, full projection semantics, and a broader runtime
  file layout are still missing.

Implementation note:
- `examples/show_runtime_usage.py` demonstrates the current call chain from
  Python -> generated module -> fixed runtime header and shows the generated
  file/runtime paths for inspection.

Exit criteria:
- Python can import the extension and call a no-op or trivial test run.

---

### Step 5. Add the First End-to-End Generated Module Path

Status: `[x]`

Purpose:
- Prove the central idea: Python lowers spec, emits C++, compiles it, imports it,
  runs C++, and gets results back.

Tasks:
- [x] Choose a single generated module layout:
  - one generated `.cpp`
  - optional generated `.hpp`
  - linked against fixed runtime sources
- [x] Implement template rendering for:
  - model `dfun`
  - one integrator path
  - one coupling path
  - one run entrypoint
- [x] Write codegen output to a cache/build directory based on a content hash.
- [x] Compile the generated module into a shared extension.
- [x] Import it dynamically from Python.

Deliverables:
- A single supported simulation path working end-to-end.

Progress note:
- The current generated path supports real native runs for single-subnetwork,
  single-mode, `HeunDeterministic` specs using expression-based model dfuns.
  It supports no-projection runs and the first constrained intra-projection
  path described in Step 7.
- The generated module delegates the simulation loop to `runtime/runtime.hpp`
  and exposes the compiled entrypoint through `pybind11`.

Exit criteria:
- The first generated module runs a real simulation from Python and returns arrays.

---

### Step 6. Implement the First Correctness Baseline

Status: `[x]`

Purpose:
- Make sure the generated C++ backend reproduces the existing backend behavior for
  the supported path.

Tasks:
- [x] Build comparison tests against the current hybrid backend.
- [x] Compare:
  - output shapes
  - time vectors
  - numerical values within tolerance
- [x] Test deterministic reproducibility for the supported path.
- [x] Document any accepted numerical differences and why they occur.

Deliverables:
- A small correctness test suite.

Progress note:
- `examples/compare_native_single_mpr.py` compares Python, Numba, and native
  C++ outputs for the single-network Montbrio path.
- Automated tests now exist in
  `tvb/tests/library/simulator/backend_cpp/test_cpp_hybrid_backend.py`:
  - a compile/runtime smoke test that checks the generated module includes the
    fixed runtime and copied runtime header
  - a compatibility test comparing native output against both Python hybrid and
    `NbHybridBackend` for the current single-network supported path
  - deterministic reproducibility for repeated native runs from identical
    initial conditions
  - a constrained intra-projection comparison against `NbHybridBackend`
  - a zero-weight projection regression proving the projection plumbing is
    neutral when it should be
- The known timestamp convention difference is documented in the tests:
  native and Numba count temporal-average chunk midpoints from step 1, while
  Python `TemporalAverage` is exactly `0.5 * dt` earlier for the same chunk.

Exit criteria:
- The first supported C++ path matches the reference backend within defined
  tolerance.

---

### Step 7. Add Delay Buffers and Sparse Projection Traversal Properly

Status: `[~]`

Purpose:
- Move from the simplest execution path to actual hybrid network mechanics.

Tasks:
- [x] Implement ring buffer semantics for delayed access.
- [~] Implement CSR traversal in the fixed runtime or generated kernel boundary.
- [ ] Support inter-projection delayed reads.
- [x] Support intra-projection delayed reads for the first single-subnetwork
  milestone path.
- [~] Validate indexing and horizon behavior against the Python backend.

Deliverables:
- Tested delayed sparse coupling path.

Progress note:
- `runtime/runtime.hpp` now contains `ProjectionArrays` and
  `accumulate_projection()`.
- `backend.py` forwards intra-projection CSR arrays from `ProjectionSpec` to
  the generated extension at run time.
- `templates/module_bindings.cpp.mako` converts Python lists of NumPy arrays
  into runtime `ProjectionArrays`.
- `templates/sim_module.cpp.mako` routes generated execution through
  `run_simulation(initial_state, projections, nstep, chunk_size)`.
- Current limitations are intentional and important:
  - still exactly one subnetwork
  - no inter-subnetwork projections
  - no mode maps
  - no multi-source/multi-target cvar mapping beyond the one-to-one path used
    by the current tests
  - no projection coupling-function transforms or target scales in native C++
    execution

Exit criteria:
- Delayed intra-projection access behaves correctly for the first supported
  single-subnetwork model path, and the next expansion target is inter-projection
  support.

---

### Step 8. Expand Monitor Support

Status: `[~]`

Purpose:
- Return useful outputs while keeping the C++ loop self-contained.

Tasks:
- [x] Support `Raw` or `TemporalAverage` first.
- [x] Decide monitor handling architecture:
  - fully in C++
  - partially postprocessed in Python
- [x] Add output shape conventions matching TVB expectations.
- [~] Add tests for chunking and monitor period semantics.

Deliverables:
- First production-usable monitor output path.

Progress note:
- Native execution currently returns chunked temporal-average-like state/VOI
  output with shape `(n_chunks, n_voi, n_nodes, n_modes)`.
- Monitor accumulation and result packaging are fully inside C++ for the
  supported path.
- Additional monitor types such as `Raw`, `AfferentCoupling`, `SpatialAverage`,
  `Projection`, and `Bold` remain out of scope.

Exit criteria:
- Monitor outputs can be returned from C++ without per-step Python work.

---

### Step 9. Generalize Model and Integrator Coverage

Status: `[~]`

Purpose:
- Extend from the first milestone model path to a broader hybrid backend.

Tasks:
- [x] Add a second model to prove the abstraction is not overfit.
- [ ] Add Euler support if Heun was first.
- [~] Separate model codegen from integrator codegen cleanly.
- [ ] Define a registration mechanism for supported models and integrators.

Deliverables:
- At least two models and two integrator paths, or a clear reason not to.

Progress note:
- The compatibility gate and expression translator now support more than
  `MontbrioPazoRoxin`: any model exposing `state_variable_dfuns` can be lowered
  and code-generated if its expressions are within the translator's supported
  AST subset.
- `examples/visualize_cpp_models_timeseries.py` and
  `examples/benchmark_single_subnetwork_cpp.py` exercise multiple
  single-subnetwork model cases. Models using custom Numba templates, such as
  Zerlaut, are still rejected.
- Only `HeunDeterministic` is supported.

Exit criteria:
- The code generator is organized around reusable model/integrator emitters, not
  one-off templates.

---

### Step 10. Add Build, Cache, and Developer Tooling

Status: `[~]`

Purpose:
- Make the backend usable repeatedly during development and tests.

Tasks:
- [x] Add generated-source cache keyed by spec and backend version.
- [ ] Add rebuild invalidation rules.
- [x] Preserve generated sources for debugging when requested.
- [ ] Add verbose compile/logging mode.
- [~] Document how to inspect generated C++.

Deliverables:
- Stable local development workflow for generated modules.

Progress note:
- Generated files are written under `.build/tvb_hybrid_cpp_<cache-key-prefix>/`
  and copied runtime headers are preserved beside the generated module.
- `CompiledCppNetwork.debug_summary()` and `examples/show_runtime_usage.py`
  expose generated source paths for inspection.
- The cache key controls build directory naming, but repeated execution still
  rebuilds the extension; true rebuild avoidance and invalidation policy remain
  pending.

Exit criteria:
- Re-running the same simulation spec avoids unnecessary regeneration and rebuilds.

---

### Step 11. Integrate with TVB Backend Selection

Status: `[~]`

Purpose:
- Make the C++ backend accessible through the TVB simulator backend layer.

- [x] Add a backend class such as `CppHybridBackend`.
- [~] Match the public entrypoints expected by current backend usage.
- [ ] Keep fallback behavior clear when a configuration is unsupported.
- [ ] Document backend selection and expected limitations.

Deliverables:
- Python backend class wired into the simulator backend ecosystem.

Progress note:
- `backend.py` already provides `CppHybridBackend` plus compile/run entrypoints,
  but it is not yet wired into the broader TVB backend-selection flow.

Exit criteria:
- A user can select the new backend from Python without custom scripts.

---

### Step 12. Performance Validation and Optimization

Status: `[~]`

Purpose:
- Confirm the C++ path is worth keeping and tune only after correctness.

Tasks:
- [x] Compare numerical results against:
  - pure Python hybrid execution
  - Numba hybrid backend
- [x] Benchmark against existing Python/Numba hybrid backend.
- [x] Profile compile time and execution time separately.
- [ ] Identify bottlenecks:
  - memory layout
  - coupling traversal
  - monitor accumulation
  - Python/C++ boundary
- [ ] Optimize only measured hotspots.
- [ ] Consider OpenMP only after correctness and single-thread baseline are solid.

Deliverables:
- Benchmark notes with clear before/after measurements.

Progress note:
- `examples/benchmark_single_subnetwork_cpp.py` compares pure Python,
  `NbHybridBackend`, and `CppHybridBackend` for single-subnetwork cases and
  reports compile/warmup time separately from run time.
- Current measurements show the native C++ run path can be faster than Numba
  for supported single-subnetwork cases, while C++ compile time is still high
  enough that short one-off runs may not benefit.

Exit criteria:
- We have evidence for the performance impact and know where optimization effort
  is justified.

---

## First Milestone Definition

The first milestone is complete when all of the following are true:

- [x] Python can lower one supported `NetworkSet` into a C++-ready spec.
- [x] Python can generate and compile a simulation-specific C++ extension.
- [x] `pybind11` exposes a callable entrypoint.
- [x] The full simulation loop runs in C++.
- [x] Results are returned to Python as NumPy arrays.
- [x] Numerical output matches the reference backend within tolerance.

Completion note:
- The first milestone is complete for the single-subnetwork, single-mode,
  `HeunDeterministic` path with expression-based models and temporal-average
  style output. A constrained intra-projection path is also present.
- The next milestone should focus on multi-subnetwork/inter-projection support.

## Suggested Immediate Next Action

Implement the next projection milestone:

- support inter-subnetwork projections in `CompiledCppNetwork.run()`
- add one source history buffer per subnetwork in the native runtime
- pass inter-projection arrays, source/target subnet ids, mode maps, cvar
  mappings, target scales, and coupling-function parameters through pybind11
- compare against `NbHybridBackend` on the coupled-MPR benchmark fixture

## Progress Log

- 2026-04-20: Initial staged plan created.
- 2026-04-29: Plan updated to reflect the current native backend: completed
  first single-subnetwork generated path, automated correctness/reproducibility
  tests, expression-based multi-model examples, benchmark script, and first
  constrained intra-projection runtime path. Inter-subnetwork projection support
  remains the main next milestone.
