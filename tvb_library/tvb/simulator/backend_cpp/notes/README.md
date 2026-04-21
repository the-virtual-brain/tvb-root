# C++ Backend for TVB Hybrid Simulator

## Overview
This directory contains the design and implementation of a C++ backend for the TVB hybrid simulator. The goal is to use code generation in C++ to minimize communication overhead between Python and the backend, moving computation-heavy parts to C++ while returning results to Python.

## Motivation
- Existing numba backend has overhead from Python-C communication per step.
- C++ code generation allows compiling the entire simulation loop in C++ for better performance.

## References
- [tvb-cpp](https://github.com/neich/tvb-cpp/tree/main): C++ implementation of TVB core functionality. Uses Eigen, Zlib, Intel MKL. Alpha code, tested on VS C++, gcc, Intel DPC++.
- tvbk: Local implementation at `/home/ziaee/git/inference/tvbk` (outside workspace, need to reference manually).
- **NEST Simulator**: Uses NESTML DSL to generate C++ models; templates for neuron/synapse code.
- **Brian2 Simulator**: Generates standalone C++ code from Python using Jinja2 templates; supports runtime/standalone modes with SymPy.

## Key Components
- Code generation from Python models to C++.
- C++ simulation engine.
- Python bindings for input/output.

## Next Steps
- Analyze hybrid simulator structure.
- Design code generation pipeline.
- Implement core C++ classes.
- Integrate with existing Python simulator.
