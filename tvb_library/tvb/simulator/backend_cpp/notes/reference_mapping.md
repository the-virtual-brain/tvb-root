# Reference Mapping for Hybrid C++ Backend

## Purpose

This note captures what we will reuse conceptually from:

- `tvb/simulator/backend/nb_hybrid.py`
- `tvb/simulator/backend/templates/nb-hybrid-sim.py.mako`
- local reference project `tvbk`
- user-facing hybrid demos in `tvb_documentation/demos/hybrid`

It is the design anchor for Step 1 of the implementation plan.

## User-Facing Starting Point

The initial fixture should follow the existing hybrid demos:

- `tvb_documentation/demos/hybrid/benchmark_hybrid_simulator_vs_numba.py`
- `tvb_documentation/demos/hybrid/visualize_hybrid_models_timeseries.py`

These scripts establish the first practical path we should support:

- Python hybrid API is the frontend
- `NetworkSet` is configured in Python
- model is `MontbrioPazoRoxin`
- integrator is `HeunDeterministic`
- coupling is `Linear`
- monitor is `TemporalAverage`
- later validation compares:
  - pure Python hybrid execution
  - `NbHybridBackend`
  - C++ backend

## What to Reuse From `nb_hybrid.py`

### Responsibilities to preserve

`nb_hybrid.py` already has the right backend decomposition:

- compatibility checking
- lowering a `NetworkSet` into plain analysis metadata
- projection normalization
- source-history horizon computation
- build/cache boundary for generated execution code
- result shaping for monitor outputs

### Relevant internal boundaries

The key functions and responsibilities are:

- `_check_compatibility(network_set)`
  - validates supported models, integrators, and shared `dt`
- `_analyse(network_set)`
  - lowers TVB objects into backend-friendly analysis metadata
- `_build_projection_info(p, is_inter)`
  - normalizes sparse projection data and coupling metadata
- `_make_projection_buffer(...)`
  - defines the history buffer shape and initialization semantics
- `_build(template_source, content, print_source=False)`
  - render -> cache-key -> compile/load executable artifact

### C++ backend equivalent

These should map as follows:

- `_check_compatibility` -> Python frontend validation before C++ code generation
- `_analyse` -> Python lowering pass producing `SimulationSpec`
- `_build_projection_info` -> `ProjectionSpec` construction
- `_make_projection_buffer` -> C++ runtime buffer initialization policy
- `_build` -> C++ generated-source cache plus shared-module build/import path

## What to Reuse From `nb-hybrid-sim.py.mako`

The template shows the functional split that should remain true in C++:

- per-projection coupling functions
- per-subnetwork integration logic
- one inner stepping kernel
- one outer run entrypoint

### Conceptual mapping

- generated coupling functions -> generated C++ coupling kernels or inline helpers
- generated integrator/model path -> generated model-specific derivative code
- `network_chunk(...)` -> fixed C++ runtime stepping loop calling generated math
- `run_network(...)` -> `pybind11` entrypoint that returns NumPy arrays

### Important design decision

We should not generate the entire runtime from scratch for each simulation.
Instead:

- generate the model/coupling-specialized pieces
- keep stepping, buffering, and packaging in a fixed C++ runtime

This is the main architectural adaptation from the current Numba backend.

## What to Reuse From `tvbk`

`tvbk` is the structural reference for a compiled simulation backend.

### Useful patterns

- one C++ extension module exposing a narrow Python interface
- model-specific headers such as `mpr.hpp`, `jr.hpp`, `wilson_cowan.hpp`
- reusable integrator headers such as `heun.hpp`
- reusable stepping logic in `step.hpp`
- reusable connectivity/buffer types in files like `conn.hpp` and `cxb.hpp`
- a CMake-based native build pipeline

### What we should copy conceptually

- header-oriented organization for model kernels
- reusable runtime code separated from per-model math
- explicit extension boundary returning NumPy-compatible arrays
- native build pipeline that can be invoked from Python code generation

### What we should not copy directly

- `nanobind` as the binding layer
- `tvbk`'s exact data layout or vectorization strategy as a hard constraint

For this backend:

- bindings should use `pybind11`
- data layout should be chosen based on TVB hybrid semantics first
- vectorization and threading should remain later optimization steps

## Mapping Table

### `nb_hybrid` concept -> new backend equivalent

- `NetworkSet` backend input -> Python-side frontend object to lower into `SimulationSpec`
- `NetworkAnalysis` -> explicit spec classes with no live TVB object dependencies
- Mako-rendered Python+Numba kernel -> generated C++ source linked to fixed runtime
- in-process function cache -> generated-source and compiled-module cache
- Python return formatting -> `pybind11` result packaging

### `tvbk` concept -> reusable reference idea

- model header per system -> generated or hand-written model kernel unit
- `heun.hpp` -> reusable deterministic integrator implementation style
- `step.hpp` -> reusable stepping-loop organization
- `conn.hpp` and buffer structs -> starting point for CSR and delay-buffer runtime types
- `tvbk_ext.cpp` -> extension module structure and binding granularity
- `CMakeLists.txt` -> native build layout for generated modules

## Resulting Architecture

### Python side

Owns:

- user API
- compatibility validation
- lowering to `SimulationSpec`
- code generation
- build/cache orchestration
- benchmark and correctness comparisons

### Fixed C++ runtime

Owns:

- state arrays
- ring/delay buffers
- sparse projection traversal
- simulation loop
- monitor accumulation
- result packaging

### Generated C++ code

Owns:

- model derivative code for the chosen model
- coupling math specialization
- network-specific constants and dispatch glue

## Immediate Implications for Step 2

The lowered spec should be designed around this split:

- Python objects disappear after lowering
- the spec contains only POD-like metadata and contiguous arrays
- model and projection descriptors must be sufficient to emit generated C++
- monitor configuration must be explicit so monitor work stays in C++

## Scope Boundaries for the First Milestone

- Start with a single-subnetwork `MontbrioPazoRoxin` path.
- Expand next to one inter-projection case matching the benchmark demo.
- Use `HeunDeterministic`.
- Use `Linear` coupling.
- Use `TemporalAverage`.
- Defer broad model coverage, threading, and aggressive vectorization.

## Deferred Until Later

- generic translation of arbitrary Python model code to C++
- stochastic integrators
- broad monitor set
- persistent packaging/integration into all TVB backend selection paths
- performance tuning beyond a basic working path
