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

# Wheel-build placement (editable mode uses LIBRARY_OUTPUT_DIRECTORY above):
# install the extension side-by-side with the cpp_hybrid package files.
install(TARGETS _cpp_hybrid
  LIBRARY DESTINATION tvb/simulator/backend/cpp_hybrid
  NAMELINK_SKIP
)
