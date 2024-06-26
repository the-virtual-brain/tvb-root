# tvb_kernels

This is a library of computational kernels for TVB.

## scope

in order of priority

- [ ] sparse delay coupling functions
- [ ] fused heun neural mass model step functions
- [ ] neural ODE
- [ ] bold / tavg monitors

## building

For now, a `make` invocation is enough, which calls `mkispc.sh` to build
the ISPC kernels, then CMake to build the nanobind wrappers.  The next
steps will convert this to a pure CMake based process.
 
## variants

### implementation

ISPC compiler provides the best performance, followed by a C/C++ compiler.
To handle cases where all compilers are not available, the pattern will
be as follows

Python class owns kernel workspace (arrays, floats, int) and has a
short, obvious NumPy based implementation.

A nanobind binding layer optionally implements calls to
- a short, obvious C++ implementation
- an (possibly multithreaded) ISPC implementation
- a CUDA implementation, if arrays are on GPU

### batch variants

It may be desirable to compute batches of a kernel.

### parameters

Parameters may be global or "spatialized" meaning different
values for every node.  Kernels should provide separate calls
for both cases.

## matlab

Given some users in matlab, it could be useful to wrap some of the algorithms
as MEX files.
