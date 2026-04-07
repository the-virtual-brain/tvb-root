cmake_minimum_required(VERSION 3.16)

project(tvb_hybrid_cpp_574359f669ba1060 LANGUAGES CXX)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

set(Python3_EXECUTABLE "/home/ziaee/anaconda3/bin/python" CACHE FILEPATH "Python executable")
set(PYTHON_EXECUTABLE "/home/ziaee/anaconda3/bin/python" CACHE FILEPATH "Python executable")
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "/home/ziaee/anaconda3/bin/python" -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(tvb_hybrid_cpp_574359f669ba1060 MODULE tvb_hybrid_cpp_574359f669ba1060_bindings.cpp)
target_compile_features(tvb_hybrid_cpp_574359f669ba1060 PRIVATE cxx_std_17)

set_target_properties(tvb_hybrid_cpp_574359f669ba1060 PROPERTIES PREFIX "")
