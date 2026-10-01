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
| `dfungen.py` | Translates each model's `state_variable_dfuns` expressions into C++ dfun kernels (`emit_sources`, also a CLI), plus the runtime g++/ctypes fallback (`generate_lib`) for models the build did not compile. |
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

## Support matrix & wheel policy (decision: keep the cp312 stable-ABI pin)

Recorded 2026-09 (tvbkh handoff item 7). **Decision: do not widen the wheel
matrix — keep `CIBW_BUILD: 'cp312-*'`** in both
`.github/workflows/cpp-hybrid.yml` (tvbkh dev branch) and
`.github/workflows/wheels.yml` (release pipeline; publishes to PyPI on tagged
releases).

| Dimension | Support |
|---|---|
| CPython | **≥ 3.12** — wheels are `cp312-abi3` (stable ABI): one wheel serves 3.12/3.13/3.14+; the module cannot load on < 3.12 by construction, whether installed from a wheel or built from the sdist |
| Hybrid code syntax floor | ≥ 3.10 (PEP 604 unions) — the syntax is *not* the binding constraint; the compiled extension is |
| OS / arch | Linux x86_64 (`ubuntu-latest`), macOS arm64 (`macos-14`), Windows x86_64 (`windows-latest`) |
| Distributions | wheels + sdist (`pipx run build --sdist tvb_library`), from the CI workflows |

Why not cp310/cp311: with the static `py-api = "cp312"`, cibuildwheel builds
for 3.10/3.11 would **still** emit `cp312-abi3` wheels (the wheel tag follows
the ABI compile target, not the build interpreter), so widening really means
dropping the single-ABI pin, adding per-version `py-api` configuration and two
extra C++ compile passes × 3 OSes — while CPython 3.10 reaches EOL in Oct 2026
and 3.11 in Oct 2027. **Reopen if** a PyPI release needs 3.11 users, or 3.11
demand appears before its Oct 2027 EOL (3.10 is EOL as of Oct 2026).

**Portability caveat (predates this work):** every compiled artifact — the
extension in wheels and editable builds, and the runtime g++ fallback
(`dfungen.generate_lib`) — is compiled with `-march=native` (GNU flags in
`tvb_library/CMakeLists.txt`, and `dfungen.py` for the fallback). CI wheels
are therefore tuned to the build runner's CPU: x86_64 wheels assume the
`ubuntu-latest` (AVX2-class) baseline, and older x86_64 CPUs may not be able
to run them (SIGILL on unsupported instructions). This caveat predates the
2026 cp312 wheel work.

## Build-time generation of the generic dfuns

Models whose derivatives are declared as Python expression strings
(`state_variable_dfuns`, plus optional `dfun_intermediates`, `dfun_helpers`,
`dfun_constants`) do not need a hand-written kernel. `dfungen.py` translates
each expression into a small per-model C++ function — this is **not**
monolithic codegen: only the derivative functions are generated, all control
flow (integration, coupling, monitors, sweeps) stays in the hand-written core.

`tvb_library/CMakeLists.txt` runs, as part of every build:

```bash
python tvb/simulator/backend/cpp_hybrid/dfungen.py \
       --emit <build>/generated/models_gen.cpp \
       --meta <build>/generated/models_gen.json \
       --entry-name cph_builtin_dfun --strict --only-supported
```

and compiles the emitted translation unit **into `_cpp_hybrid`** with
`CPH_HAVE_BUILTIN_GEN=1`. What the emitted TU contains:

* one `dfun_gen_<id>` kernel per model, in the same SIMD-lane layout
  (`(n_svar, n_node, n_modes, W)`, `parr[k * W + i]`) as the hand-written ones;
* a dispatch array `_dfuns` indexed by `id - 100` behind an `extern "C"` entry
  `cph_builtin_dfun(...)` that returns **1** when it handled the id and **0**
  when it did not — which is what lets `dfun_dispatch` in `_core.cpp` fall back
  to the runtime library for ids the build does not own;
* a metadata table `cph_gen_entry` (name, id, `n_parm`, `n_cvar`, `n_svar`,
  ordered parameter names) read out through `cph_builtin_count` /
  `cph_builtin_entry` and exposed to Python as
  `_cpp_hybrid.generic_model_table()`. That table — not a Python-side
  re-derivation — is what `backend.py` uses to pick the model id, the buffer
  shapes and the parameter packing order, so Python and the kernels cannot
  disagree.

`--strict` makes any model module that cannot be imported, any
`state_variable_dfuns` model that cannot be configured, and any model that
cannot be emitted a **build failure** listing every offender, instead of a
silent shrink of the generated model set.

`--only-supported` scopes discovery *and* the strict expectation to the
model set the hybrid backends actually support (the 27 classes of
`nb_hybrid._get_supported_models_classes`, mirrored by
`dfungen._SUPPORTED_MODEL_CLASSES`/`_SUPPORTED_MODEL_MODULES`): the emitted
table is exactly the kernels the extension ships (25 expression-driven ones;
the two numba-only supported models have hand-written kernels 0..12).
Modules outside that set are neither imported nor emitted, so unsupported
model code in the tree cannot break or slow the build — while a supported
model that cannot be generated still fails the build loudly.  A dfun model
outside the supported set (e.g. `DecoBalancedExcInh`) is still runnable: the
runtime fallback keeps full discovery and compiles it on first use.  (Handoff
item 6, option 2; the runtime path intentionally defaults to full discovery
so user-defined models keep working.)

### Why compile it into the extension

The generated kernels get the *same* compile flags as the hand-written ones:
`-O3 -march=native -funroll-loops -fopenmp-simd -ffp-contract=off`. The runtime
fallback (`dfungen.generate_lib`, g++ + ctypes) compiles with
`-O3 -march=native -fopenmp-simd -std=c++17` and **omits `-ffp-contract=off`**
(and `-funroll-loops`), so a fallback-compiled model is only float32-tolerance
equal to the numba reference, not bit-exact. Compiling at build time also
removes a g++ call and a `models_gen.cpp`/`models_gen.so` pair from every
first run: a stock generic model needs no compiler at runtime at all
(`test_cpp_hybrid_buildgen.py` proves it by making `generate_lib` raise).

Turn the whole thing off with:

```bash
cmake -DTVB_CPP_GENERATE_MODELS=OFF
# through scikit-build-core:
pip install -Ccmake.args=-DTVB_CPP_GENERATE_MODELS=OFF
```

Then no TU is emitted, `generic_model_table()` is empty, and the backend falls
back to the runtime g++/ctypes path for every generic model — the behaviour
from before this feature, including the original 100-based id range.

### Why the build needs numpy, scipy, numba and six

`dfungen` emits the supported models' kernels by importing their modules
(`--only-supported` scopes discovery to exactly those; in this tree every
module under `tvb/simulator/models` holds at least one supported model, so
the imported set is the same — the scoping is what keeps the build
decoupled from unsupported model code) and instantiating each class, so those
imports must work before the package's own dependencies are installed. The
verified minimal set (omitting any one makes `dfungen --strict` fail with a
`ModuleNotFoundError` naming it, or silently drop models):
`numpy` (dfungen itself + every model module), `six` (`tvb.basic.neotraits._core`),
`scipy` (neotraits `NArray`, `stefanescu_jirsa`, `cerebellar_mf`), `numba` (the
numba-dfun model modules — without them those models are simply absent from the
table). See `tvb_library/pyproject.toml` `[build-system] requires`.

### Build cost of generation (handoff item 6)

Measured on this machine (clean `pip install . --no-deps -q`, warm pip
cache; 2026-09-30):

| | wall | generation delta |
|---|---|---|
| generation ON (default) | ~22.6 s | — |
| `-DTVB_CPP_GENERATE_MODELS=OFF` | ~17.2 s | ~5.4 s |

Of the ~5.4 s delta, ~4.5 s is `dfungen` emission (model imports +
instantiation/configure + expression translation) and ~1.4 s is the generated
TU compile (`g++ -O3 -march=native`, measured standalone); the OFF switch is
the documented escape hatch above.  The handoff's larger machine-specific
deltas (pip ~47 s vs ~10 s; wheel ~20 s vs ~10 s) were recorded on the
origin machine — same shape, different scale.

The 26th discovery (`DecoBalancedExcInh`, a dfun model in the tree but
outside the supported set) is no longer a built-in kernel: option 2 of the
handoff went in, so the built-in table is exactly the supported set (25
generic kernels; ids 100..124) and the build no longer imports/emits
unsupported model code.  Incremental per-model emission (option 3, keyed by
the Item-1 dfun fingerprint) was not taken: with `-flto` the link phase
re-processes every TU anyway, so per-model caching would only skip the ~1.4 s
front-end compile and the emission half of a *rebuild* — no clean-build gain
for a moderate CMake rework; the `CONFIGURE_DEPENDS` glob + deps-manifest
triggers already keep unchanged builds at zero regeneration cost.

### Model id ranges

| Range | Owner | Lookup |
|---|---|---|
| **0..12** | hand-written kernels in `_core.cpp` (MPR, Generic2dOscillator, Kuramoto, SupHopf, Linear, ReducedWongWang, WilsonCowan, JansenRit, Epileptor, Epileptor2D, Zerlaut 1st/2nd order, CerebellarMF) | `_MODEL_IDS`, checked first, so these take precedence over a generated kernel for the same model |
| **100..124** | the 25 built-in generic kernels compiled in at build time (the supported model set) | `generic_model_table()`, ids assigned in sorted class-name order |
| **125+** | runtime-compiled fallback library (`models_gen.so`) for models the build did not cover (user-defined models, and in-tree dfun models outside the supported set such as `DecoBalancedExcInh`) | emitted at `max(built-in id) + 1`, dispatch table gap-padded with `nullptr` over `100..124` |

The gap padding matters: ids are positional over sorted class names, so a user
model sorting before every stock name (`AaaProbe`) would otherwise take id 100
and shift all the stock ids — and since `_core.cpp` serves the built-in range
first, that model would silently run a stock kernel. With disjoint ranges that
is structurally impossible, and a gap id returns 0 from the fallback entry so
nothing calls a null slot.

### Residual caveat: models are looked up by class **name**

`_model_key` resolves a model by `type(model).__name__`. Consequences:

* A user class that **shadows a stock model name** (e.g. their own
  `class JansenRit(...)`) resolves to the stock kernel, not to theirs. It is
  not detected as "a model the build does not know about".
* What *is* caught: a parameter-count or parameter-name mismatch against the
  table the kernel was generated from. Packing `parr` positionally would
  otherwise read past the end of the array or land values in the wrong slots,
  so `_packing_parm_names` raises a `ValueError` naming the model and showing
  both lists.
* What is **not** caught: a shadowing class with the *same* state variables,
  coupling terms and parameters but **different equations**. Same buffer
  shapes, same packing — so it runs the stock equations silently. Avoid it by
  not reusing stock model names; a distinct class name lands in the 125+ range
  and gets its own generated kernel.

## Feature parity with nb_hybrid (all parity-tested, float32 tolerance)

Full audit: `parity_audit.md`. Summary of the supported feature set, which now
matches `NbHybridBackend`:

| Feature | Coverage | Parity test |
|---|---|---|
| **Models** (26 classes) | 13 hand-written dfuns (ids 0–12, incl. CerebellarMF) + generic expression-generated route (ids ≥ 100): 25 supported generic kernels compiled in at build time; the one in-tree dfun model outside the supported set (`DecoBalancedExcInh`) is runtime-compiled on first use | `test_cpp_hybrid_models.py` (31 items: 26 classes + CerebellarMF flag variants) |
| **Coupling functions** | Linear, Scaling, Sigmoidal, Difference, Kuramoto, HyperbolicTangent, SigmoidalJansenRit (classic **and** legacy), PreSigmoidal (static **and** dynamic, incl. `globalT`) | `test_cpp_hybrid_coupling.py` (11 items) |
| **Monitors** | Raw, RawVoi, TemporalAverage, SubSample, GlobalAverage, AfferentCoupling (+TemporalAverage), SpatialAverage, Projection, Bold. Bold's HRF convolution and Raw/SubSample collection run as **kernel monitor engines** inside the C++ step loop; the shared Python path remains the parity reference/fallback (`enable_kernel_monitors`) | `test_cpp_hybrid_monitors.py` (18 items) |
| **Stimuli** | constant / pulse / sinusoid patterns (single-node and all-node spatial), **multiple subnetworks with stimuli simultaneously** | `test_cpp_hybrid_stimuli.py` (6 items) |
| **Stochastic noise** | Additive noise, HeunStochastic/EulerStochastic, per-subnet RNG streams, **multiple stochastic subnetworks** | `test_cpp_hybrid_stimuli.py::test_stochastic_noise_parity`, `test_multi_subnet_stimuli_and_noise` |

Not supported by either backend (rejected identically): nonzero
`model_local_coupling`.

## Tests

From `tvb_library` root:

```bash
python -m pytest tvb/tests/library/simulator/backend/test_cpp_hybrid_core.py \
                 tvb/tests/library/simulator/backend/test_cpp_hybrid_models.py \
                 tvb/tests/library/simulator/backend/test_cpp_hybrid_coupling.py \
                 tvb/tests/library/simulator/backend/test_cpp_hybrid_monitors.py \
                 tvb/tests/library/simulator/backend/test_cpp_hybrid_stimuli.py \
                 tvb/tests/library/simulator/backend/test_cpp_hybrid_parity.py \
                 tvb/tests/library/simulator/backend/test_cpp_hybrid_sweep.py \
                 tvb/tests/library/simulator/backend/test_cpp_hybrid_sugar.py \
                 tvb/tests/library/simulator/backend/test_cpp_hybrid_buildgen.py \
                 -q -p no:cacheprovider
```

`test_cpp_hybrid_buildgen.py` is the permanent guard on the generated-kernel
feature: stock generic models running with `generate_lib` unable to run, the
extension table agreeing with `emit_sources` metadata, disjoint built-in /
user-model id ranges, `start_id` gap padding, the empty-table fallback range,
packing-mismatch errors, and fallback-vs-numba parity.

(The full 222-item battery mono-process run crashes at *interpreter shutdown*
due to numba/llvmlite teardown — run `test_cpp_hybrid_battery.py` class by
class in fresh processes if you need a single clean tally; see the test files
for the collected classes.)

