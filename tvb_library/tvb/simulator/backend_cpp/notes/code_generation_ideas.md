# Code Generation Ideas

## Approach
- Use Jinja2 or similar templating for generating C++ from Python models.
- Generate classes for each component (model, integrator, etc.).
- Inline parameters and equations for optimization.

## From tvb-cpp Reference
- Core simulation loop in C++.
- Use Eigen matrices for state variables.
- MKL for BLAS operations.

## From tvbk Reference
- (Need to inspect locally) Likely similar structure, perhaps with inference components.

## From Other Simulators
- **NEST**: Uses NESTML (DSL) to generate C++ neuron/synapse models. Templates for code generation in C++.
- **Brian2**: Generates standalone C++ code from Python models using Jinja2 templates. Supports runtime and standalone modes. Uses SymPy for symbolic math.

## TVB Numba Backend Templates
- Uses Mako templates (similar to Jinja2) to generate Python code with numba decorators.
- Templates: nb-sim.py.mako (main loop), nb-dfuns.py.mako (derivative functions), nb-integrate.py.mako (integrators), nb-coupling.py.mako (coupling).
- Generates numba-compiled functions for simulation.
- **Adaptation**: Create C++ equivalents using Mako/Jinja2 to output C++ code instead of Python.

## Steps
1. Parse Python model equations (use SymPy or AST parsing).
2. Generate C++ functions for derivatives.
3. Generate integrator loops.
4. Compile to shared library.
5. Load in Python via ctypes or pybind11.

## Example Pseudocode
```cpp
// Generated from Python model
class MyModel {
    Eigen::VectorXd state;
    void step(double dt) {
        // Generated derivative calculations
        state += derivative * dt;  // Euler
    }
};
```

## Tools
- SymPy for symbolic math if needed.
- Cython for hybrid Python/C++ if full C++ is too much initially.
- Jinja2 for templating, inspired by Brian2.
