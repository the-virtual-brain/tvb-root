from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
from typing import Any

from .codegen import (
    GeneratedSourceArtifact,
    build_generated_extension,
    generate_cpp_source,
)
from .lowering import SpecLoweringResult, lower_network_set
from .spec import SimulationSpec


@dataclasses.dataclass(frozen=True)
class CompiledCppNetwork:
    """Stub compiled artifact for the future C++ backend pipeline.

    Current status:
    - lowering is implemented
    - C++ code generation is not yet implemented
    - native compilation is not yet implemented
    - execution is not yet implemented
    """

    spec: SimulationSpec
    lowering: SpecLoweringResult
    build_dir: Path
    generated_cpp_path: Path
    module_name: str
    generated_source: GeneratedSourceArtifact
    pipeline_stage: str = "cpp_generated"

    def debug_summary(self) -> dict[str, Any]:
        return {
            "pipeline_stage": self.pipeline_stage,
            "module_name": self.module_name,
            "build_dir": str(self.build_dir),
            "generated_cpp_path": str(self.generated_cpp_path),
            "bindings_cpp_path": str(self.generated_source.bindings_cpp_path),
            "cmake_lists_path": str(self.generated_source.cmake_lists_path),
            "runtime_header_path": str(self.generated_source.runtime_header_path),
            "sim_template_path": str(self.generated_source.sim_template_path),
            "bindings_template_path": str(self.generated_source.bindings_template_path),
            "cmake_template_path": str(self.generated_source.cmake_template_path),
            "extension_path": None
            if self.generated_source.extension_path is None
            else str(self.generated_source.extension_path),
            "cache_key": self.spec.cache_key(),
            "n_subnetworks": len(self.spec.subnetworks),
            "n_inter_projections": len(self.spec.inter_projections),
            "n_intra_projections": len(self.spec.intra_projections),
            "n_monitors": len(self.spec.monitors),
        }

    def load_module(self):
        if self.generated_source.extension_path is None:
            raise RuntimeError(
                "No extension has been built yet for this CompiledCppNetwork."
            )
        spec = importlib.util.spec_from_file_location(
            self.module_name, self.generated_source.extension_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Unable to load extension module from {self.generated_source.extension_path}"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def run(self, *args, **kwargs):
        initial_states = kwargs.pop("initial_states", None)
        nstep = kwargs.pop("nstep", None)
        chunk_size = kwargs.pop("chunk_size", 1)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs.keys())}")
        if nstep is None:
            raise TypeError("run() requires nstep.")
        if len(self.spec.subnetworks) != 1:
            raise NotImplementedError(
                "CompiledCppNetwork.run() currently supports only single-subnetwork specs."
            )
        if initial_states is None:
            raise TypeError(
                "run() currently requires initial_states=[array] for the generated backend."
            )
        if isinstance(initial_states, (list, tuple)):
            if len(initial_states) != 1:
                raise ValueError("run() expects exactly one initial-state array.")
            initial_state = initial_states[0]
        else:
            initial_state = initial_states

        module = self.load_module()
        return module.run_simulation(initial_state, int(nstep), int(chunk_size))


class CppHybridBackend:
    """Python-side frontend stub for the future C++ hybrid backend.

    Intended pipeline:
    1. Validate and lower `NetworkSet` to `SimulationSpec`
    2. Generate simulation-specific C++ from the spec
    3. Compile a `pybind11` extension module
    4. Execute the full simulation loop in C++
    5. Return monitor outputs to Python

    Current implementation:
    - Step 1 only
    """

    def __init__(self, build_root: str | Path | None = None):
        self.build_root = Path(build_root) if build_root is not None else Path.cwd() / ".build"

    def lower(
        self,
        network_set,
        monitors: list[object] | None = None,
        user_source_hint: str | None = None,
    ) -> SpecLoweringResult:
        return lower_network_set(
            network_set=network_set,
            monitors=monitors,
            user_source_hint=user_source_hint,
        )

    def compile(
        self,
        network_set,
        monitors: list[object] | None = None,
        user_source_hint: str | None = None,
        build_native: bool = True,
    ) -> CompiledCppNetwork:
        lowering = self.lower(
            network_set=network_set,
            monitors=monitors,
            user_source_hint=user_source_hint,
        )
        spec = lowering.spec
        cache_key = spec.cache_key()
        module_name = f"tvb_hybrid_cpp_{cache_key[:16]}"
        build_dir = self.build_root / module_name
        generated_cpp_path = build_dir / f"{module_name}.cpp"
        generated_source = generate_cpp_source(
            spec=spec,
            build_dir=build_dir,
            module_name=module_name,
        )
        pipeline_stage = "cpp_generated"
        if build_native:
            generated_source = build_generated_extension(generated_source)
            pipeline_stage = "extension_built"
        return CompiledCppNetwork(
            spec=spec,
            lowering=lowering,
            build_dir=build_dir,
            generated_cpp_path=generated_source.cpp_path,
            module_name=module_name,
            generated_source=generated_source,
            pipeline_stage=pipeline_stage,
        )

    def run(self, *args, **kwargs):
        compiled = self.compile(*args, **kwargs)
        return compiled.run()
