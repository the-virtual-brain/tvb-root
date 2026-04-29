"""Hybrid C++ backend for the TVB simulator.

Pipeline: Python lowers a NetworkSet into a SimulationSpec → Mako templates
emit C++ → pybind11 extension compiled via CMake → full simulation loop runs
in C++ → monitor outputs returned as NumPy arrays.
"""

from .backend import CompiledCppNetwork, CppHybridBackend
from .codegen import (
    GeneratedSourceArtifact,
    build_generated_extension,
    generate_cpp_source,
    render_cpp_template,
)
from .lowering import SpecLoweringResult, lower_network_set
from .spec import (
    IntegratorSpec,
    MonitorSpec,
    ProjectionSpec,
    SimulationSpec,
    StimulusSpec,
    SubnetworkSpec,
)

__all__ = [
    "CompiledCppNetwork",
    "CppHybridBackend",
    "GeneratedSourceArtifact",
    "IntegratorSpec",
    "MonitorSpec",
    "ProjectionSpec",
    "SimulationSpec",
    "SpecLoweringResult",
    "StimulusSpec",
    "SubnetworkSpec",
    "build_generated_extension",
    "generate_cpp_source",
    "lower_network_set",
    "render_cpp_template",
]
