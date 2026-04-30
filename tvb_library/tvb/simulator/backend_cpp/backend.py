from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from .codegen import (
    GeneratedSourceArtifact,
    build_generated_extension,
    generate_cpp_source,
)
from .lowering import SpecLoweringResult, lower_network_set
from .spec import ProjectionSpec, SimulationSpec


def _projection_arrays(proj: ProjectionSpec) -> dict[str, Any]:
    """Extract numpy arrays from a ProjectionSpec for the native binding."""
    return {
        "weights_data": np.ascontiguousarray(proj.weights_data, dtype=np.float64),
        "weights_indices": np.ascontiguousarray(proj.weights_indices, dtype=np.int32),
        "weights_indptr": np.ascontiguousarray(proj.weights_indptr, dtype=np.int32),
        "idelays": np.ascontiguousarray(proj.idelays, dtype=np.int32),
        "source_svar": int(proj.source_cvar[0]),
        "target_cvar_slot": int(proj.target_cvar[0]),
        "scale": float(proj.scale),
    }


def _inter_projection_arrays(proj: ProjectionSpec) -> dict[str, Any]:
    """Build arrays for an inter-projection, folding mode_map[0,0] into scale.

    For single-mode subnets the mode_map is always (1, 1).  We absorb the
    scalar into the projection scale so the C++ runtime needs no special mode
    logic — it uses the same accumulate_projection path as intra-projections.
    """
    effective_scale = float(proj.scale)
    if proj.mode_map is not None:
        effective_scale *= float(proj.mode_map.flat[0])
    return {
        "weights_data": np.ascontiguousarray(proj.weights_data, dtype=np.float64),
        "weights_indices": np.ascontiguousarray(proj.weights_indices, dtype=np.int32),
        "weights_indptr": np.ascontiguousarray(proj.weights_indptr, dtype=np.int32),
        "idelays": np.ascontiguousarray(proj.idelays, dtype=np.int32),
        "source_svar": int(proj.source_cvar[0]),
        "target_cvar_slot": int(proj.target_cvar[0]),
        "scale": effective_scale,
    }


@dataclasses.dataclass(frozen=True)
class CompiledCppNetwork:
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

    def run(self, **kwargs):
        initial_states = kwargs.pop("initial_states", None)
        nstep = kwargs.pop("nstep", None)
        chunk_size = kwargs.pop("chunk_size", 1)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs.keys())}")
        if nstep is None:
            raise TypeError("run() requires nstep.")
        if initial_states is None:
            raise TypeError("run() requires initial_states=[array, ...].")

        monitor_type = self.spec.monitors[0].type_name if self.spec.monitors else "TemporalAverage"
        # Per-step monitors: force chunk_size=1 (mirrors Numba _compute_chunk_size).
        if monitor_type in ("Raw", "RawVoi", "AfferentCoupling"):
            chunk_size = 1

        module = self.load_module()

        # Intra-projection arrays (within a single subnet).
        intra_data = [_projection_arrays(p) for p in self.spec.intra_projections]
        # Inter-projection arrays (between subnets); mode_map folded into scale.
        inter_data = [_inter_projection_arrays(p) for p in self.spec.inter_projections]

        # Ensure each initial state is a contiguous float64 array.
        flat_states = [
            np.ascontiguousarray(s, dtype=np.float64) for s in initial_states
        ]

        raw_results = module.run_simulation(
            flat_states,
            int(nstep),
            int(chunk_size),
            # --- intra projections ---
            [p["weights_data"]     for p in intra_data],
            [p["weights_indices"]  for p in intra_data],
            [p["weights_indptr"]   for p in intra_data],
            [p["idelays"]          for p in intra_data],
            [p["source_svar"]      for p in intra_data],
            [p["target_cvar_slot"] for p in intra_data],
            [p["scale"]            for p in intra_data],
            # --- inter projections ---
            [p["weights_data"]     for p in inter_data],
            [p["weights_indices"]  for p in inter_data],
            [p["weights_indptr"]   for p in inter_data],
            [p["idelays"]          for p in inter_data],
            [p["source_svar"]      for p in inter_data],
            [p["target_cvar_slot"] for p in inter_data],
            [p["scale"]            for p in inter_data],
        )

        # Select output based on monitor type.  AfferentCoupling variants return
        # the temporally-averaged coupling input (ctavg) instead of state VOIs
        # (data), matching the Numba backend's _apply_monitors behaviour.
        is_afferent = monitor_type in ("AfferentCoupling", "AfferentCouplingTemporalAverage")
        return [
            (times, ctavg) if is_afferent else (times, data)
            for times, data, ctavg in raw_results
        ]


class CppHybridBackend:
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
            generated_cpp_path=generated_source.cpp_path,  # kept for debug_summary
            module_name=module_name,
            generated_source=generated_source,
            pipeline_stage=pipeline_stage,
        )

    def run(self, *args, **kwargs):
        compiled = self.compile(*args, **kwargs)
        return compiled.run()
