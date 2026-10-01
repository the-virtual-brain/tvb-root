cmake_minimum_required(VERSION 3.15...3.26)

project(tvb_library LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

# The C++ hybrid simulator backend (see tvb/simulator/backend/cpp_hybrid/).
# LIBRARY_OUTPUT_DIRECTORY places the built module directly in the package
# directory — this is what makes in-place editable installs work, because
# scikit-build-core's inplace editable mode runs `cmake --build` (which honours
# the target output dir) but not `cmake --install`.
nanobind_add_module(
  _cpp_hybrid
  STABLE_ABI NB_STATIC LTO NOMINSIZE
  tvb/simulator/backend/cpp_hybrid/_core.cpp
)

set_target_properties(_cpp_hybrid PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  LIBRARY_OUTPUT_DIRECTORY tvb/simulator/backend/cpp_hybrid
)

# Float-exactness parity with the numba reference backend: keep fast-math OFF
# and FP contraction OFF. This mirrors the original g++ build flags
# (-O3 -march=native -fopenmp-simd -ffp-contract=off) plus -funroll-loops;
# -fopenmp-simd needs no OpenMP runtime link.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(_cpp_hybrid PRIVATE -O3 -march=native -funroll-loops -fopenmp-simd -ffp-contract=off)
endif()

# ---------------------------------------------------------------------------
# Build-time generation of the expression-driven dfun kernels.
#
# tvb/simulator/backend/cpp_hybrid/dfungen.py translates each supported
# model's ``state_variable_dfuns`` expression strings into a C++ dfun kernel
# and emits one translation unit with an ``extern "C"`` dispatch entry (here
# named ``cph_builtin_dfun``).  Compiling it *into* the extension — rather
# than letting the runtime g++/ctypes fallback compile it on first use — is
# the point: the generated kernels then get the same flags as the hand-written
# ones (-O3 -march=native -funroll-loops -fopenmp-simd -ffp-contract=off),
# which the runtime path does not provide (-ffp-contract=off in particular).
#
# Discovery is scoped with --only-supported to the model set the hybrid
# backends actually support (the 27 classes of
# nb_hybrid._get_supported_models_classes; handoff item 6, option 2): the
# emitted table is exactly the kernels the extension ships, unsupported model
# modules in the tree are neither imported nor emitted (so they cannot fail
# the build), and the kernel count is 25 — the expression-driven subset, since
# the two numba-only supported models (ZerlautAdaptationFirstOrder,
# CerebellarMF) are hand-written kernels in ids 0..12.  A dfun model outside
# the supported set (DecoBalancedExcInh) is not a built-in kernel; the runtime
# fallback keeps full discovery and compiles it on first use.  Without
# --only-supported the emitted table would carry every in-tree dfun model
# instead (the pre-item-6 behaviour, used by the runtime fallback path).
#
# The generated TU is unreferenced by _core.cpp for now, so nanobind's hidden
# visibility plus -flto dead-strip it at link time; that is expected until the
# core dispatches to cph_builtin_dfun under CPH_HAVE_BUILTIN_GEN.
#
# Turn off with:  cmake -DTVB_CPP_GENERATE_MODELS=OFF  (or, through
# scikit-build-core:  pip install -Ccmake.args=-DTVB_CPP_GENERATE_MODELS=OFF)
# ---------------------------------------------------------------------------
option(TVB_CPP_GENERATE_MODELS
  "Generate expression-driven C++ dfuns with dfungen and compile them into _cpp_hybrid"
  ON)

if(TVB_CPP_GENERATE_MODELS)
  set(_tvb_dfungen ${CMAKE_CURRENT_SOURCE_DIR}/tvb/simulator/backend/cpp_hybrid/dfungen.py)
  set(_tvb_gen_dir ${CMAKE_CURRENT_BINARY_DIR}/generated)
  set(_tvb_gen_cpp ${_tvb_gen_dir}/models_gen.cpp)
  set(_tvb_gen_json ${_tvb_gen_dir}/models_gen.json)

  # Regeneration triggers.  CONFIGURE_DEPENDS re-globs at build time, so a
  # model file being added or removed re-runs configure ("GLOB mismatch!").
  # The glob results are then written into a manifest whose mtime the custom
  # command depends on: mtime comparison alone cannot notice a *deleted*
  # dependency, so without the manifest a removed model would leave a stale
  # generated table (and stale model ids) behind.
  #
  # Generated artifacts stay in the build directory.  Note that an inplace
  # editable install uses the source tree as its CMake binary dir, so
  # generated/ appears at the tvb_library root there (gitignored); wheel builds
  # write it under build/<wheel_tag>/generated/.
  file(GLOB _tvb_model_modules CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/tvb/simulator/models/*.py")
  file(MAKE_DIRECTORY "${_tvb_gen_dir}")
  set(_tvb_gen_manifest ${_tvb_gen_dir}/models_gen.deps)
  file(WRITE "${_tvb_gen_manifest}" "${_tvb_model_modules}")

  # dfungen is run as a script, not with -m: importing it as a submodule would
  # execute tvb/simulator/backend/cpp_hybrid/__init__.py, which imports the
  # very extension being built (not importable mid-build, and absent entirely
  # in a clean checkout / isolated PEP 517 build).  A script invocation puts
  # the *script's* directory on sys.path, not the working directory, so
  # PYTHONPATH is what makes the source tree's `tvb` package importable; model
  # discovery needs it before the package itself is installed.  (The
  # add_custom_command ENVIRONMENT keyword cannot be used here: the Ninja
  # generator appends it to its `cd` prefix.)
  #
  # The build environment's own PYTHONPATH must be preserved: pip's isolated
  # build env is an *overlay* on the outer interpreter, so Python_EXECUTABLE is
  # that outer interpreter and the overlay's numpy/scipy/numba are reachable
  # only through PYTHONPATH.  Replacing it would hide them (verified: the build
  # then fails with ModuleNotFoundError: No module named 'numpy').
  if(CMAKE_HOST_WIN32)
    set(_tvb_pathsep ";")
  else()
    set(_tvb_pathsep ":")
  endif()
  if(DEFINED ENV{PYTHONPATH})
    set(_tvb_pythonpath "${CMAKE_CURRENT_SOURCE_DIR}${_tvb_pathsep}$ENV{PYTHONPATH}")
  else()
    set(_tvb_pythonpath "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()

  add_custom_command(
    OUTPUT "${_tvb_gen_cpp}" "${_tvb_gen_json}"
    COMMAND ${CMAKE_COMMAND} -E env
      "PYTHONPATH=${_tvb_pythonpath}"
      ${Python_EXECUTABLE} "${_tvb_dfungen}"
      --emit "${_tvb_gen_cpp}"
      --meta "${_tvb_gen_json}"
      --entry-name cph_builtin_dfun
      --strict
      --only-supported
    DEPENDS "${_tvb_dfungen}" "${_tvb_gen_manifest}" ${_tvb_model_modules}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    VERBATIM
    COMMENT "dfungen: emitting expression-driven C++ dfuns (strict, supported set)"
  )

  # --strict makes any model that could not be imported/configured/emitted a
  # hard error listing every offender, so a coverage loss fails the build
  # instead of silently shrinking the generated model table.  With
  # --only-supported the expectation is scoped to the supported model set:
  # a supported model that cannot be generated fails the build loudly, while
  # unsupported model code (outside the build's dependency surface) is
  # irrelevant to it.  A non-zero exit from the custom command stops the build
  # (ninja prints the command and the CLI's stderr); nothing degrades into a
  # skip.
  add_custom_target(tvb_cpp_generated_dfuns
    DEPENDS "${_tvb_gen_cpp}" "${_tvb_gen_json}")
  add_dependencies(_cpp_hybrid tvb_cpp_generated_dfuns)

  # Adding the generated TU to the target is what gives it the target's
  # compile flags; CPH_HAVE_BUILTIN_GEN lets _core.cpp dispatch to
  # cph_builtin_dfun (see step 3) only when the kernels are really present.
  target_sources(_cpp_hybrid PRIVATE "${_tvb_gen_cpp}")
  target_compile_definitions(_cpp_hybrid PRIVATE CPH_HAVE_BUILTIN_GEN=1)
else()
  message(STATUS "TVB_CPP_GENERATE_MODELS=OFF: no build-time dfun generation")
endif()

# Wheel-build placement (editable mode uses LIBRARY_OUTPUT_DIRECTORY above):
# install the extension side-by-side with the cpp_hybrid package files.
install(TARGETS _cpp_hybrid
  LIBRARY DESTINATION tvb/simulator/backend/cpp_hybrid
  NAMELINK_SKIP
)
