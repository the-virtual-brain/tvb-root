"""Hybrid C++ backend scaffolding.

This package holds the Python-side lowering and code-generation pipeline for a
simulation-specific C++ backend. The initial implementation focuses on the
lowered spec boundary so the Python hybrid API can be translated into a
backend-neutral representation before code generation is introduced.
"""

from .backend import CompiledCppNetwork, CppHybridBackend
from .codegen import (
    DelayedSelfFeedbackConfig,
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
    "DelayedSelfFeedbackConfig",
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
