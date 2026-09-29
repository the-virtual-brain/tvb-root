# `cpp_hybrid` — C++ hybrid simulator backend (nanobind)

This is the compiled C++ backend for the TVB hybrid simulator. It lowers the
Python/NumPy hybrid implementation into tight, SIMD-enabled C++ with the
simulation control flow in C++ (no monolithic static codegen). Python keeps
control; data arrays stay NumPy `float32`. Parameter sweeps run multithreaded
and vectorized across the width-8 SIMD lane dimension.

The extension is built with **nanobind + scikit-build-core** (a `cp312-abi3`
stable-ABI module). The build lives at the **`tvb_library` project level**
(`tvb_library/pyproject.toml` + `tvb_library/CMakeLists.txt`), so a single
`pip install tvb-library` — editable or from a wheel — compiles and installs
the extension into this package directory. Nothing here needs to be built
separately.

## Layout

| File | Role |
|---|---|
| `_core.cpp` | Single C++ translation unit: nanobind bindings + SIMD kernels. |
| `backend.py` | `CppHybridBackend` — mirrors `nb_hybrid.NbHybridBackend`, imported via `from ._cpp_hybrid import Sim, enable_generic_dfuns`. |
| `notebook.py` | `sweep()` sugar for multithreaded SIMD parameter sweeps, tqdm, `SweepResult.to_dataframe()`. |
| `features.py` | Vendored Apache-2.0 feature module (VBI subset). |
| `dfungen.py` | Compiles `models_gen.so` via g++ at runtime (ctypes; independent of this build). |
| `README.md` | This file. |

Build config lives one level up: `tvb_library/pyproject.toml`
(`[tool.scikit-build]`) and `tvb_library/CMakeLists.txt`
(`nanobind_add_module(_cpp_hybrid STABLE_ABI NB_STATIC LTO NOMINSIZE _core.cpp)`).

## Building for development (editable install)

From the `tvb_library` root:

```bash
uv pip install -e .            # or: python -m pip install -e . --no-build-isolation -Ceditable.rebuild=true
```

That single command installs the whole `tvb` package **and** compiles the C++
extension in-place into `tvb/simulator/backend/cpp_hybrid/_cpp_hybrid.abi3.so`.
`rebuild = true` (the default here) means the extension is **recompiled
automatically on import** whenever `_core.cpp` changes — you see
`[1/2] Building ... _core.cpp.o → [2/2] Linking ... _cpp_hybrid.abi3.so` on the
next import.

> Do **not** delete the in-source CMake build tree (`CMakeCache.txt`,
> `build.ninja`, `editable_rebuild.lock`, …). It's what makes rebuild-on-import
> work, and it is git-ignored.

The same build config drives the CI wheels (`.github/workflows/wheels.yml`
builds `tvb_library` sdists + 3-OS cibuildwheel wheels, each containing the
compiled extension), so published wheels carry a prebuilt backend — no
compiler needed by end users installing from PyPI.

## Using the backend

```python
import tvb.simulator.backend.cpp_hybrid as cph        # builds/imports the extension
from tvb.simulator.backend.cpp_hybrid._cpp_hybrid import Sim, enable_generic_dfuns

s = Sim(8)                         # width-8 SIMD batch (W = 8)
s.add_subnet(n_node, n_svar, n_parm, n_cvar, model_id, horizon, n_modes)
# ... then drive it exactly like the nb_hybrid backend's CppHybridBackend
```

For a higher-level entry point, use `backend.CppHybridBackend` or the notebook
`notebook.sweep(...)` helpers. A worked example lives at
`tvb_documentation/demos/cpp_hybrid_sweep.ipynb`.

## Compiler flags & numerical parity

GNU-only flags (see `tvb_library/CMakeLists.txt`):

```
-O3 -march=native -funroll-loops -fopenmp-simd -ffp-contract=off
```

We deliberately keep **`-ffp-contract=off`** and do **not** use `-ffast-math`:
float32 bit-exactness parity with the numba reference hybrid backend depends on
no fast-math / fast-FP contraction. `-fopenmp-simd` (no runtime libgomp
dependency) enables the SIMD vectorization across the batch lane.

## Tests

From `tvb_library` root:

```bash
python -m pytest tvb/tests/library/simulator/backend/test_cpp_hybrid_core.py \
                 tvb/tests/library/simulator/backend/test_cpp_hybrid_parity.py \
                 tvb/tests/library/simulator/backend/test_cpp_hybrid_sweep.py \
                 tvb/tests/library/simulator/backend/test_cpp_hybrid_sugar.py \
                 -q -p no:cacheprovider
```

(The full 222-item battery mono-process run crashes at *interpreter shutdown*
due to numba/llvmlite teardown — run it fresh-process per class if you need a
single clean tally; see the test files for the collected classes.)
