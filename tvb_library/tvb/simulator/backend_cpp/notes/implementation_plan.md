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

Status: `[~]`

Purpose:
- Identify exactly what to reuse from TVB hybrid backend and from `tvbk`.

Tasks:
- [ ] Extract the lowering/analysis responsibilities from
  `tvb/simulator/backend/nb_hybrid.py`.
- [ ] Extract the generated-kernel boundaries from
  `tvb/simulator/backend/templates/nb-hybrid-sim.py.mako`.
- [ ] Extract runtime organization ideas from `tvbk`:
  - model kernels
  - integrator kernels
  - stepping loops
  - extension layout
- [ ] Write a concise mapping table:
  - `nb_hybrid` concept -> C++ backend equivalent
  - `tvbk` concept -> reusable runtime/reference idea

Deliverables:
- A short architecture mapping note in `notes/` or appended here.
- Include the user-facing demo paths that will serve as initial fixtures.

Exit criteria:
- We know which pieces are copied conceptually, which are adapted, and which are
  new.

---

### Step 2. Define the Python-to-C++ Lowered Spec

Status: `[ ]`

Purpose:
- Create a stable intermediate representation between TVB Python objects and the
  C++ backend.

Tasks:
- [ ] Define Python dataclasses or plain structures for:
  - `SimulationSpec`
  - `SubnetworkSpec`
  - `ProjectionSpec`
  - `IntegratorSpec`
  - `MonitorSpec`
  - optional `StimulusSpec`
- [ ] Ensure the spec contains only C++-friendly data:
  - scalar values
  - strings/enums
  - contiguous NumPy arrays
  - integer maps and dimensions
- [ ] Mirror the useful parts of `NetworkAnalysis` but remove Python object
  dependencies.
- [ ] Decide what is baked into generated code versus passed as runtime arrays.
- [ ] Define a stable hash key for spec-based code generation cache.

Deliverables:
- `spec.py` or equivalent module with the lowered-spec schema.
- A documented list of fields included in the first milestone.

Exit criteria:
- A `NetworkSet` can be lowered into a complete spec without requiring C++ yet.

---

### Step 3. Implement a Python Lowering Pass

Status: `[ ]`

Purpose:
- Convert TVB runtime objects into the spec from Step 2.

Tasks:
- [ ] Build a lowering function that reads:
  - subnetworks
  - inter/intra projections
  - model parameters
  - initial states
  - delays
  - coupling mappings
  - monitor configuration
- [ ] Reuse compatibility checks from the existing hybrid backend where possible.
- [ ] Normalize all arrays:
  - dtype
  - memory layout
  - shape conventions
- [ ] Decide initial treatment of stimuli:
  - either unsupported in milestone 1
  - or precomputed on Python side and passed as arrays

Deliverables:
- Lowering function with deterministic output.
- Tests that validate spec contents for a small example network.

Exit criteria:
- For the first supported path, the lowered spec is sufficient to run a
  simulation without touching original TVB objects.

---

### Step 4. Create the Fixed C++ Runtime Skeleton

Status: `[ ]`

Purpose:
- Establish the reusable C++ execution core before introducing large-scale code
  generation.

Tasks:
- [ ] Create C++ runtime directories and build structure.
- [ ] Add core runtime types for:
  - array views
  - state buffers
  - delay/ring buffers
  - CSR connectivity access
  - output buffers
- [ ] Implement a minimal simulation loop API:
  - initialize state
  - step for `nstep`
  - accumulate one monitor type
  - package outputs
- [ ] Add `pybind11` module scaffolding.
- [ ] Build the extension with CMake.

Deliverables:
- Compilable C++ runtime skeleton.
- A trivial `pybind11` extension importable from Python.

Exit criteria:
- Python can import the extension and call a no-op or trivial test run.

---

### Step 5. Add the First End-to-End Generated Module Path

Status: `[ ]`

Purpose:
- Prove the central idea: Python lowers spec, emits C++, compiles it, imports it,
  runs C++, and gets results back.

Tasks:
- [ ] Choose a single generated module layout:
  - one generated `.cpp`
  - optional generated `.hpp`
  - linked against fixed runtime sources
- [ ] Implement template rendering for:
  - model `dfun`
  - one integrator path
  - one coupling path
  - one run entrypoint
- [ ] Write codegen output to a cache/build directory based on a content hash.
- [ ] Compile the generated module into a shared extension.
- [ ] Import it dynamically from Python.

Deliverables:
- A single supported simulation path working end-to-end.

Exit criteria:
- The first generated module runs a real simulation from Python and returns arrays.

---

### Step 6. Implement the First Correctness Baseline

Status: `[ ]`

Purpose:
- Make sure the generated C++ backend reproduces the existing backend behavior for
  the supported path.

Tasks:
- [ ] Build comparison tests against the current hybrid backend.
- [ ] Compare:
  - output shapes
  - time vectors
  - numerical values within tolerance
- [ ] Test deterministic reproducibility for the supported path.
- [ ] Document any accepted numerical differences and why they occur.

Deliverables:
- A small correctness test suite.

Exit criteria:
- The first supported C++ path matches the reference backend within defined
  tolerance.

---

### Step 7. Add Delay Buffers and Sparse Projection Traversal Properly

Status: `[ ]`

Purpose:
- Move from the simplest execution path to actual hybrid network mechanics.

Tasks:
- [ ] Implement ring buffer semantics for delayed access.
- [ ] Implement CSR traversal in the fixed runtime or generated kernel boundary.
- [ ] Support inter-projection delayed reads.
- [ ] Support intra-projection delayed reads if needed for milestone expansion.
- [ ] Validate indexing and horizon behavior against the Python backend.

Deliverables:
- Tested delayed sparse coupling path.

Exit criteria:
- Delayed projection access behaves correctly for the first supported model path.

---

### Step 8. Expand Monitor Support

Status: `[ ]`

Purpose:
- Return useful outputs while keeping the C++ loop self-contained.

Tasks:
- [ ] Support `Raw` or `TemporalAverage` first.
- [ ] Decide monitor handling architecture:
  - fully in C++
  - partially postprocessed in Python
- [ ] Add output shape conventions matching TVB expectations.
- [ ] Add tests for chunking and monitor period semantics.

Deliverables:
- First production-usable monitor output path.

Exit criteria:
- Monitor outputs can be returned from C++ without per-step Python work.

---

### Step 9. Generalize Model and Integrator Coverage

Status: `[ ]`

Purpose:
- Extend from the first milestone model path to a broader hybrid backend.

Tasks:
- [ ] Add a second model to prove the abstraction is not overfit.
- [ ] Add Euler support if Heun was first.
- [ ] Separate model codegen from integrator codegen cleanly.
- [ ] Define a registration mechanism for supported models and integrators.

Deliverables:
- At least two models and two integrator paths, or a clear reason not to.

Exit criteria:
- The code generator is organized around reusable model/integrator emitters, not
  one-off templates.

---

### Step 10. Add Build, Cache, and Developer Tooling

Status: `[ ]`

Purpose:
- Make the backend usable repeatedly during development and tests.

Tasks:
- [ ] Add generated-source cache keyed by spec and backend version.
- [ ] Add rebuild invalidation rules.
- [ ] Preserve generated sources for debugging when requested.
- [ ] Add verbose compile/logging mode.
- [ ] Document how to inspect generated C++.

Deliverables:
- Stable local development workflow for generated modules.

Exit criteria:
- Re-running the same simulation spec avoids unnecessary regeneration and rebuilds.

---

### Step 11. Integrate with TVB Backend Selection

Status: `[ ]`

Purpose:
- Make the C++ backend accessible through the TVB simulator backend layer.

Tasks:
- [ ] Add a backend class such as `CppHybridBackend`.
- [ ] Match the public entrypoints expected by current backend usage.
- [ ] Keep fallback behavior clear when a configuration is unsupported.
- [ ] Document backend selection and expected limitations.

Deliverables:
- Python backend class wired into the simulator backend ecosystem.

Exit criteria:
- A user can select the new backend from Python without custom scripts.

---

### Step 12. Performance Validation and Optimization

Status: `[ ]`

Purpose:
- Confirm the C++ path is worth keeping and tune only after correctness.

Tasks:
- [ ] Compare numerical results against:
  - pure Python hybrid execution
  - Numba hybrid backend
- [ ] Benchmark against existing Python/Numba hybrid backend.
- [ ] Profile compile time and execution time separately.
- [ ] Identify bottlenecks:
  - memory layout
  - coupling traversal
  - monitor accumulation
  - Python/C++ boundary
- [ ] Optimize only measured hotspots.
- [ ] Consider OpenMP only after correctness and single-thread baseline are solid.

Deliverables:
- Benchmark notes with clear before/after measurements.

Exit criteria:
- We have evidence for the performance impact and know where optimization effort
  is justified.

---

## First Milestone Definition

The first milestone is complete when all of the following are true:

- [ ] Python can lower one supported `NetworkSet` into a C++-ready spec.
- [ ] Python can generate and compile a simulation-specific C++ extension.
- [ ] `pybind11` exposes a callable entrypoint.
- [ ] The full simulation loop runs in C++.
- [ ] Results are returned to Python as NumPy arrays.
- [ ] Numerical output matches the reference backend within tolerance.

## Suggested Immediate Next Action

Start with Step 0 and lock the first supported path in writing before coding.
That decision controls the spec shape, generated templates, and runtime API.

## Progress Log

- 2026-04-20: Initial staged plan created.
