from __future__ import annotations

import dataclasses
import importlib.util
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .codegen import (
    DEFAULT_BINDINGS_TEMPLATE,
    DEFAULT_CMAKE_TEMPLATE,
    DEFAULT_SIM_TEMPLATE,
    GeneratedSourceArtifact,
    _discover_extension_path,
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


def _build_stimulus_arrays(
    spec: SimulationSpec,
    lowering: "SpecLoweringResult",
    nstep: int,
) -> list[np.ndarray]:
    """Pre-compute stimulus coupling contributions for all steps.

    Mirrors the Numba backend: for each subnet with stimuli, call
    stim.get_coupling(step) for every step and accumulate into a
    (n_cvar, n_nodes, nstep) float64 array.  Values are placed at the
    target_cvar indices the stimulus specifies.  Deterministic (no-stimulus)
    subnets get an empty placeholder so the stim_ptrs vector stays aligned.
    """
    stim_arrays: list[np.ndarray] = []
    sn_infos = lowering.analysis.subnetworks
    for sn_spec, sn_info in zip(spec.subnetworks, sn_infos):
        if not sn_info.has_stimulus:
            stim_arrays.append(np.empty(0, dtype=np.float64))
            continue
        n_cvar = sn_spec.n_coupling_vars
        n_nodes = sn_spec.n_nodes
        arr = np.zeros((n_cvar, n_nodes, nstep), dtype=np.float64)
        for stim_obj in lowering.analysis.stimuli_by_subnet[sn_info.name]:
            target_cvar = np.asarray(stim_obj.target_cvar, dtype=int)
            for step_idx in range(1, nstep + 1):
                sc = np.asarray(stim_obj.get_coupling(step_idx), dtype=np.float64)
                # sc shape: (n_stim_cvar, n_nodes, n_modes) — use mode 0
                if sc.ndim == 3:
                    sc = sc[:, :, 0]  # (n_stim_cvar, n_nodes)
                arr[target_cvar, :, step_idx - 1] += sc
        stim_arrays.append(arr.ravel())
    return stim_arrays


def _build_noise_arrays(
    spec: SimulationSpec,
    lowering: "SpecLoweringResult",
    nstep: int,
) -> list[np.ndarray]:
    """Pre-generate Wiener increments for stochastic subnets.

    Mirrors the Numba backend: noise = sqrt(2*nsig*dt)*randn, shaped
    (n_vars, n_nodes, 1, nstep) C-contiguous float64. Deterministic subnets
    get an empty (0,) placeholder so the noise_ptrs vector is always n_subnets
    long, which lets the binding safely index it for every subnet.
    """
    noise_arrays: list[np.ndarray] = []
    sn_infos = lowering.analysis.subnetworks
    for sn_spec, sn_info in zip(spec.subnetworks, sn_infos):
        if not sn_info.is_stochastic:
            noise_arrays.append(np.empty(0, dtype=np.float64))
            continue
        dt = float(sn_spec.integrator.dt)
        nsig = np.asarray(sn_spec.integrator.noise_nsig, dtype=np.float64)  # (n_vars,)
        noise_std = np.sqrt(2.0 * nsig * dt)
        rng = sn_info.integrator.noise.random_stream
        # Draw (nstep, n_vars, n_nodes, n_modes), scale, transpose → (n_vars, n_nodes, n_modes, nstep)
        dw = rng.randn(nstep, sn_spec.n_state_vars, sn_spec.n_nodes, sn_spec.n_modes)
        dw *= noise_std[np.newaxis, :, np.newaxis, np.newaxis]
        dw = np.ascontiguousarray(np.transpose(dw, (1, 2, 3, 0)), dtype=np.float64)
        noise_arrays.append(dw.ravel())
    return noise_arrays


_LAST_USED_FILE = ".last_used"


def _touch_last_used(build_dir: Path) -> None:
    sentinel = build_dir / _LAST_USED_FILE
    sentinel.touch()


def _evict_old_cache_entries(build_root: Path, keep: int) -> None:
    if keep <= 0:
        return
    entries = [d for d in build_root.iterdir() if d.is_dir() and d.name.startswith("tvb_hybrid_cpp_")]
    if len(entries) <= keep:
        return
    def _mtime(d: Path) -> float:
        sentinel = d / _LAST_USED_FILE
        target = sentinel if sentinel.exists() else d
        return target.stat().st_mtime
    entries.sort(key=_mtime, reverse=True)
    for old_dir in entries[keep:]:
        shutil.rmtree(old_dir, ignore_errors=True)


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

        noise_arrays = _build_noise_arrays(self.spec, self.lowering, int(nstep))
        stim_arrays = _build_stimulus_arrays(self.spec, self.lowering, int(nstep))

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
            noise_arrays,
            stim_arrays,
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
    def __init__(
        self,
        build_root: str | Path | None = None,
        max_cached_builds: int = 16,
    ):
        self.build_root = Path(build_root) if build_root is not None else Path.cwd() / ".build"
        self.max_cached_builds = max_cached_builds

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
        verbose: bool = False,
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

        if build_native:
            try:
                extension_path = _discover_extension_path(build_dir, module_name)
                generated_source = GeneratedSourceArtifact(
                    module_name=module_name,
                    build_dir=build_dir,
                    cpp_path=build_dir / f"{module_name}.cpp",
                    bindings_cpp_path=build_dir / f"{module_name}_bindings.cpp",
                    cmake_lists_path=build_dir / "CMakeLists.txt",
                    runtime_header_path=build_dir / "runtime" / "runtime.hpp",
                    sim_template_path=DEFAULT_SIM_TEMPLATE,
                    bindings_template_path=DEFAULT_BINDINGS_TEMPLATE,
                    cmake_template_path=DEFAULT_CMAKE_TEMPLATE,
                    extension_path=extension_path,
                )
                if verbose:
                    print(f"[tvb-cpp] Cache hit:       {module_name}")
                _touch_last_used(build_dir)
                _evict_old_cache_entries(self.build_root, self.max_cached_builds)
                return CompiledCppNetwork(
                    spec=spec,
                    lowering=lowering,
                    build_dir=build_dir,
                    generated_cpp_path=generated_source.cpp_path,
                    module_name=module_name,
                    generated_source=generated_source,
                    pipeline_stage="extension_cached",
                )
            except FileNotFoundError:
                pass

        generated_source = generate_cpp_source(
            spec=spec,
            build_dir=build_dir,
            module_name=module_name,
            verbose=verbose,
        )
        pipeline_stage = "cpp_generated"
        if build_native:
            generated_source = build_generated_extension(generated_source, verbose=verbose)
            pipeline_stage = "extension_built"
        if build_native:
            _touch_last_used(build_dir)
            _evict_old_cache_entries(self.build_root, self.max_cached_builds)
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
