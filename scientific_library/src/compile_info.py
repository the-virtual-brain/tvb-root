# this module just provides compilation flags and
# include directories for use in setup.py and
# .ycm_extra_conf.py

cpp_flags = '-Wall -Wextra -Werror -fexceptions -std=c++11 -x c++'.split()

extern_includes = [
  'cpuid.git/src/cpuid',
  'eigen-3.2.8/Eigen',
  'pybind11.git/include',
  'vdt-0.3.6/include'
]
