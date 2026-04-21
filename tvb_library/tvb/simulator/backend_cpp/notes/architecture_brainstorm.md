# Architecture Brainstorm

## Current Hybrid Simulator Structure
- Located in `tvb/simulator/hybrid/`
- Components: network.py, subnetwork.py, projections, coupling, etc.
- Numba backend in `_numba/`: coupling.py, integrators.py, models.py, etc.

## Proposed C++ Backend Architecture
1. **Code Generator**: Python module that analyzes the hybrid model and generates C++ code.
   - Input: Python model definitions, parameters.
   - Output: C++ source files for simulation.

2. **C++ Core Engine**:
   - Classes for networks, subnetworks, projections.
   - Integrators (e.g., Euler, Runge-Kutta).
   - Models (neural mass models).
   - Coupling functions.
   - Use Eigen for linear algebra, MKL for performance.

3. **Python Bindings**:
   - Use pybind11 or similar for interfacing.
   - Expose C++ classes/functions to Python.

4. **Build System**:
   - CMake for C++ compilation.
   - Integrate with TVB's build (pyproject.toml).

## Performance Considerations
- Compile-time code generation to avoid runtime overhead.
- Vectorization with Eigen/MKL.
- Memory management: avoid copies between Python/C++.

## Integration Points
- Replace numba backend in simulator.py.
- Maintain API compatibility.

## Challenges
- Complex hybrid model dynamics.
- Debugging generated C++ code.
- Cross-platform compilation.
