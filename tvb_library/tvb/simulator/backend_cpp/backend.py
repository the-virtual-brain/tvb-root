from __future__ import annotations

import dataclasses
import importlib.util
import shutil
from pathlib import Path
from typing import Any, Optional

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


# ---------------------------------------------------------------------------
# Sweep result container
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SweepResult:
    """Results from :meth:`CppHybridBackend.sweep`.

    Attributes
    ----------
    tavg : dict[str, np.ndarray]
        Per-subnet temporal averages, shape ``(n_sweeps, n_time, n_voi, N_subnet, n_modes)``.
        ``n_time = nstep // chunk_size`` where ``chunk_size = round(period / dt)``.
    merged_tavg : np.ndarray or None
        All subnets concatenated along the node axis,
        shape ``(n_sweeps, n_time, n_voi, N_total, n_modes)``.
    ctavg : dict[str, np.ndarray]
        Per-subnet coupling temporal averages (same shape as tavg).
    times : np.ndarray
        Time vector from the last subnet of the first sweep point.
    sweep_values : np.ndarray
        Parameter grid, shape ``(n_sweeps, n_dims)``.
    backend : str
        Always ``'cpp-seq'`` for this backend.
    elapsed : float
        Wall-clock seconds (excludes compilation).
    raw, bold : None
        Reserved for future use; always ``None`` in the current implementation.
    """
    tavg: dict = dataclasses.field(default_factory=dict)
    merged_tavg: Optional[np.ndarray] = None
    ctavg: dict = dataclasses.field(default_factory=dict)
    times: Optional[np.ndarray] = None
    sweep_values: Optional[np.ndarray] = None
    backend: str = ""
    elapsed: float = 0.0
    raw: Optional[dict] = None
    bold: Optional[dict] = None


# ---------------------------------------------------------------------------
# Cfun parameter tables — shared with the sweep resolver
# ---------------------------------------------------------------------------

_CFUN_PARAM_ATTRS: dict = {
    "Linear":             [("a", 0), ("b", 1)],
    "Scaling":            [("a", 0)],
    "Sigmoidal":          [("a", 0), ("sigma", 1), ("midpoint", 2), ("cmin", 3), ("cmax", 4)],
    "SigmoidalJansenRit": [("a", 0), ("e0", 1), ("r", 2), ("v0", 3)],
    "Kuramoto":           [("a", 0)],
    "HyperbolicTangent":  [("a", 0), ("midpoint", 1), ("sigma", 2)],
    "PreSigmoidal":       [("H", 0), ("Q", 1), ("G", 2), ("P", 3), ("theta", 4)],
}

_CFUN_ATTR_TO_IDX: dict = {
    (cls, attr): idx
    for cls, attrs in _CFUN_PARAM_ATTRS.items()
    for attr, idx in attrs
}

_NAMED_PARAM_ALIASES: dict = {
    "coupling_scale": {"attr": "a", "idx": 0},
    "scale":          {"attr": "a", "idx": 0},
    "coupling_a":     {"attr": "a", "idx": 0},
    "coupling_b":     {"attr": "b", "idx": 1},
    "sigma":          {"attr": "sigma", "idx": 1},
    "midpoint":       {"attr": "midpoint", "idx": 2},
}


# ---------------------------------------------------------------------------
# Initial-state helpers
# ---------------------------------------------------------------------------

def _default_initial_states(network_set) -> list[np.ndarray]:
    """Generate midpoint initial states for every subnet in *network_set*."""
    states = []
    for sn in network_set.subnets:
        model = sn.model
        x0 = np.zeros((model.nvar, sn.nnodes, model.number_of_modes), dtype=np.float64)
        for i, sv in enumerate(model.state_variables):
            rng = model.state_variable_range.get(sv)
            if rng is not None:
                lo, hi = float(rng[0]), float(rng[1])
                x0[i] = (lo + hi) / 2.0
        states.append(x0)
    return states


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


def _configure_bold_monitor(bold_monitor, spec_dt: float):
    """Set dt/istep on a Bold monitor and call compute_hrf() to prepare it."""
    bold_monitor.dt = spec_dt
    bold_monitor.istep = int(round(bold_monitor.period / spec_dt))
    bold_monitor.compute_hrf()


def _apply_bold_hrf(
    bold_monitor,
    interim_times: np.ndarray,
    interim_data: np.ndarray,
    spec_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply HRF convolution to C++ interim TA output to produce BOLD signal.

    Mirrors Bold.sample() in monitors.py exactly.  The C++ kernel runs with
    chunk_size = _interim_istep, so each returned chunk is already the temporal
    average over one interim period — equivalent to _interim_stock.mean(axis=0).

    Parameters
    ----------
    bold_monitor  : Bold, compute_hrf() already called.
    interim_times : (n_interim,)
    interim_data  : (n_interim, n_voi, n_nodes, n_modes)
    spec_dt       : simulation dt in ms

    Returns
    -------
    bold_times : (n_bold,)
    bold_data  : (n_bold, n_voi, n_nodes, n_modes)
    """
    from tvb.datatypes.equations import FirstOrderVolterra

    n_interim = interim_data.shape[0]
    k = int(bold_monitor._stock_steps)
    hrf = bold_monitor.hemodynamic_response_function          # (1, k)
    istep = int(bold_monitor.istep)
    interim_istep = int(bold_monitor._interim_istep)

    stock = np.zeros((k,) + interim_data.shape[1:])          # (k, n_voi, n_nodes, n_modes)
    bold_times: list[float] = []
    bold_data_list: list[np.ndarray] = []

    for i in range(n_interim):
        neural_step = (i + 1) * interim_istep
        stock_idx = (neural_step // interim_istep % k) - 1
        stock[stock_idx] = interim_data[i]

        if neural_step % istep == 0:
            hrf_rolled = np.roll(hrf, stock_idx, axis=1)
            # stock.transpose((1,2,0,3)): (n_voi, n_nodes, k, n_modes)
            # np.dot result: (1, n_voi, n_nodes, n_modes) → reshaped to (n_voi, n_nodes, n_modes)
            bold = np.dot(hrf_rolled, stock.transpose((1, 2, 0, 3)))
            bold = bold.reshape(stock.shape[1:])
            if isinstance(bold_monitor.hrf_kernel, FirstOrderVolterra):
                k1_V0 = (
                    bold_monitor.hrf_kernel.parameters["k_1"]
                    * bold_monitor.hrf_kernel.parameters["V_0"]
                )
                bold = (bold - 1.0) * k1_V0
            bold_times.append(float(neural_step * spec_dt))
            bold_data_list.append(bold)

    if not bold_times:
        n_voi, n_nodes, n_modes = interim_data.shape[1:]
        return np.empty(0), np.empty((0, n_voi, n_nodes, n_modes))

    return np.array(bold_times), np.stack(bold_data_list, axis=0)


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
        bold_monitor = None
        if monitor_type in ("Raw", "RawVoi", "AfferentCoupling"):
            chunk_size = 1
        elif monitor_type == "Bold":
            bold_monitor = next(
                (m for m in self.lowering.monitors if type(m).__name__ == "Bold"),
                None,
            )
            if bold_monitor is None:
                raise RuntimeError(
                    "Bold monitor type in spec but no Bold monitor object in lowering.monitors. "
                    "Ensure monitors=[Bold(...)] is passed to compile()."
                )
            _configure_bold_monitor(bold_monitor, self.spec.dt)
            chunk_size = bold_monitor._interim_istep

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
        if bold_monitor is not None:
            return [
                _apply_bold_hrf(bold_monitor, times, data, self.spec.dt)
                for times, data, _ in raw_results
            ]
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
        # On Windows MSVC exposes 'cl', not 'c++'; accept either.
        import sys as _sys
        cxx_candidates = ["cl"] if _sys.platform == "win32" else ["c++"]
        has_cxx = any(shutil.which(t) is not None for t in cxx_candidates)
        missing = ([] if shutil.which("cmake") else ["cmake"]) + ([] if has_cxx else cxx_candidates)
        if missing:
            raise RuntimeError(
                f"CppHybridBackend requires {missing} but they were not found on PATH.\n"
                "Install the missing tools and try again:\n"
                "  conda: conda install -c conda-forge cmake cxx-compiler\n"
                "  apt:   sudo apt install cmake g++\n"
                "  brew:  brew install cmake\n"
                "The rest of tvb-library works without a C++ toolchain."
            )
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

    def run(
        self,
        network_set,
        monitors=None,
        user_source_hint=None,
        build_native=True,
        verbose=False,
        initial_states=None,
        nstep=None,
        chunk_size=1,
    ):
        compiled = self.compile(
            network_set,
            monitors=monitors,
            user_source_hint=user_source_hint,
            build_native=build_native,
            verbose=verbose,
        )
        return compiled.run(
            initial_states=initial_states,
            nstep=nstep,
            chunk_size=chunk_size,
        )

    # -----------------------------------------------------------------------
    # Unified sweep API
    # -----------------------------------------------------------------------

    @staticmethod
    def _resolve_named_params(network_set, params):
        """Resolve a named-parameter dict to ``(sweep_descriptor, sweep_values)``.

        Mirrors ``NbHybridBackend._resolve_named_params`` so that the same
        ``params`` dict works for both backends.

        Parameters
        ----------
        network_set : NetworkSet
        params : dict
            Keys are parameter names; values are 1-D arrays of the same length.

        Returns
        -------
        sweep_descriptor : list[dict]
            One entry per dimension, e.g.
            ``[{'type': 'cfun', 'projection': 'ctx_to_thal', 'param_idx': 0}]``.
        sweep_values : np.ndarray, shape (n_sweeps, n_dims)
        """
        all_projs = []
        for proj in network_set.projections:
            actual_name = f"{proj.source.name}_to_{proj.target.name}"
            all_projs.append((actual_name, proj, actual_name))
        for sn in network_set.subnets:
            for p in sn.projections:
                raw_name = getattr(p, "name", None) or "intra"
                qualified = f"{sn.name}.{raw_name}" if raw_name == "intra" else raw_name
                all_projs.append((qualified, p, raw_name))

        dims = [np.asarray(v, dtype=np.float32) for v in params.values()]
        n_sweeps = len(dims[0])
        for i, d in enumerate(dims):
            if len(d) != n_sweeps:
                raise ValueError(
                    f"All param arrays must be the same length; "
                    f"param {i} has {len(d)} vs {n_sweeps}"
                )

        sweep_values = (
            np.column_stack(dims).astype(np.float32)
            if len(dims) > 1
            else dims[0].reshape(-1, 1)
        )
        sweep_descriptor: list[dict] = []

        for _dim_idx, (key, _values) in enumerate(params.items()):
            resolved = False

            # 1. Named alias ('coupling_scale', 'scale', …)
            if key in _NAMED_PARAM_ALIASES:
                alias = _NAMED_PARAM_ALIASES[key]
                for _lookup, proj, actual in all_projs:
                    cfun = proj.cfun
                    if cfun is None:
                        continue
                    cfun_cls = type(cfun).__name__
                    attr = alias["attr"]
                    if (cfun_cls, attr) in _CFUN_ATTR_TO_IDX:
                        sweep_descriptor.append({
                            "type": "cfun",
                            "projection": actual,
                            "param_idx": _CFUN_ATTR_TO_IDX[(cfun_cls, attr)],
                        })
                        resolved = True
                        break
                if resolved:
                    continue

            # 2. '{proj}.{attr}' — explicit projection cfun attribute
            if "." in key:
                proj_name, attr = key.rsplit(".", 1)
                for lookup, proj, actual in all_projs:
                    if lookup == proj_name or lookup.endswith("." + proj_name):
                        cfun = proj.cfun
                        if cfun is not None:
                            cfun_cls = type(cfun).__name__
                            if (cfun_cls, attr) in _CFUN_ATTR_TO_IDX:
                                sweep_descriptor.append({
                                    "type": "cfun",
                                    "projection": actual,
                                    "param_idx": _CFUN_ATTR_TO_IDX[(cfun_cls, attr)],
                                })
                                resolved = True
                                break
                if resolved:
                    continue

            # 3. '{subnet}.{param}' — model parameter (requires recompile)
            if "." in key:
                parts = key.split(".", 1)
                if len(parts) == 2 and any(
                    sn.name == parts[0] for sn in network_set.subnets
                ):
                    sname, param = parts
                    sweep_descriptor.append(
                        {"type": "model", "subnet": sname, "param": param}
                    )
                    continue

            raise ValueError(
                f"Cannot resolve sweep parameter '{key}'. "
                f"Use 'coupling_scale', '{{proj}}.{{attr}}', or '{{subnet}}.{{param}}'. "
                f"Projections: {[lk for lk, _, _ in all_projs]}. "
                f"Subnets: {[sn.name for sn in network_set.subnets]}."
            )

        return sweep_descriptor, sweep_values

    def sweep(
        self,
        network_set,
        params: dict,
        nstep: int = 100,
        *,
        monitors: Optional[list] = None,
        initial_states: Optional[list] = None,
        node_indices: Optional[dict] = None,
        verbose: bool = False,
        n_workers: int = 1,
    ) -> SweepResult:
        """Run a parameter sweep on the C++ backend.

        Parameters
        ----------
        network_set : NetworkSet
            Configured hybrid network.
        params : dict
            Named parameters to sweep.  Keys → 1-D arrays, all same length.

            * ``'coupling_scale'`` / ``'scale'``: first projection's ``cfun.a``
              (folded into the runtime scale — **no recompilation**).
            * ``'{proj_name}.{attr}'``: any named projection cfun attribute
              that resolves to a linear scale factor (``param_idx == 0``).
            * ``'{subnet}.{param}'``: model parameter; each unique value
              triggers a recompile (handled by the build cache).
        nstep : int
            Number of integration steps per sweep point.
        monitors : list, optional
            Monitor objects passed to :meth:`compile`.  Defaults to
            ``TemporalAverage``.
        initial_states : list of np.ndarray, optional
            Per-subnet initial states.  If *None*, midpoints of the model's
            ``state_variable_range`` are used.
        node_indices : dict, optional
            Mapping ``{subnet_name: array_of_global_node_indices}`` used to
            assemble ``merged_tavg``.
        verbose : bool
            Forward to :meth:`compile`.
        n_workers : int
            Number of parallel threads for cfun-only sweeps.  Each sweep point
            runs ``run_simulation`` concurrently; the GIL is released inside the
            C++ call so threads execute in true parallel.  Has no effect when the
            sweep includes model-parameter changes (those must recompile
            sequentially).  Default ``1`` (sequential).

        Returns
        -------
        SweepResult
        """
        sweep_descriptor, sweep_values = self._resolve_named_params(network_set, params)
        return self._sweep_cpu(
            network_set, sweep_descriptor, sweep_values,
            nstep, monitors, initial_states, node_indices, verbose,
            n_workers=n_workers,
        )

    def _sweep_cpu(
        self,
        network_set,
        sweep_descriptor: list[dict],
        sweep_values: np.ndarray,
        nstep: int,
        monitors,
        initial_states,
        node_indices,
        verbose: bool,
        n_workers: int = 1,
    ) -> SweepResult:
        import time as _t

        n_sweeps = sweep_values.shape[0]

        if initial_states is None:
            initial_states = _default_initial_states(network_set)
        flat_states = [
            np.ascontiguousarray(s, dtype=np.float64) for s in initial_states
        ]

        has_model_sweep = any(d["type"] == "model" for d in sweep_descriptor)

        # For cfun-only sweeps compile once; the build cache handles model sweeps.
        if not has_model_sweep:
            compiled = self.compile(network_set, monitors=monitors, verbose=verbose)

        raw_results: list = []
        t0 = _t.perf_counter()

        # Parallel path: cfun-only sweeps with n_workers > 1.
        # run_simulation releases the GIL internally so threads execute in true parallel.
        # Model-parameter sweeps must remain sequential (recompilation mutates network_set).
        if n_workers > 1 and not has_model_sweep:
            import concurrent.futures

            def _run_point(sv):
                intra_scales, inter_scales = self._compute_sweep_scales(
                    compiled, sweep_descriptor, sv
                )
                return self._run_for_sweep(compiled, nstep, flat_states, intra_scales, inter_scales)

            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_run_point, sweep_values[tid]) for tid in range(n_sweeps)]
                raw_results = [f.result() for f in futures]

        else:
            for tid in range(n_sweeps):
                sv = sweep_values[tid]

                # Apply model param changes and (re-)compile if needed.
                restore: dict = {}
                if has_model_sweep:
                    for dim_idx, desc in enumerate(sweep_descriptor):
                        if desc["type"] == "model":
                            sname, param = desc["subnet"], desc["param"]
                            for sn in network_set.subnets:
                                if sn.name == sname:
                                    old = float(
                                        np.atleast_1d(getattr(sn.model, param))[0]
                                    )
                                    restore[(sname, param)] = old
                                    setattr(
                                        sn.model, param,
                                        np.array([float(sv[dim_idx])]),
                                    )
                    compiled = self.compile(
                        network_set, monitors=monitors, verbose=verbose
                    )

                intra_scales, inter_scales = self._compute_sweep_scales(
                    compiled, sweep_descriptor, sv
                )
                result = self._run_for_sweep(
                    compiled, nstep, flat_states, intra_scales, inter_scales
                )
                raw_results.append(result)

                # Restore model params for the next iteration.
                for (sname, param), old_val in restore.items():
                    for sn in network_set.subnets:
                        if sn.name == sname:
                            setattr(sn.model, param, np.array([old_val]))

        elapsed = _t.perf_counter() - t0
        backend_label = f"cpp-par-{n_workers}" if n_workers > 1 and not has_model_sweep else "cpp-seq"
        return self._stack_cpp_results(
            raw_results, network_set, node_indices, sweep_values, backend_label, elapsed
        )

    def _compute_sweep_scales(
        self,
        compiled,
        sweep_descriptor: list[dict],
        sv: np.ndarray,
    ) -> tuple[list[float], list[float]]:
        """Return (intra_scales, inter_scales) with cfun.a folded in for this sweep point.

        For a ``Linear`` cfun with ``a=x`` and ``proj.scale=s``, the C++
        ``accumulate_projection`` only applies ``scale``, so we fold:
        ``effective_scale = x * s``.  Only ``param_idx=0`` (the linear
        multiplier ``a``) is supported; ``param_idx=1`` (additive ``b``) cannot
        be expressed as a pure scale and raises ``NotImplementedError``.
        """
        intra_scales = [float(p.scale) for p in compiled.spec.intra_projections]
        inter_scales = [
            float(p.scale) * (float(p.mode_map.flat[0]) if p.mode_map is not None else 1.0)
            for p in compiled.spec.inter_projections
        ]

        for dim_idx, desc in enumerate(sweep_descriptor):
            if desc["type"] != "cfun":
                continue
            proj_name = desc["projection"]
            param_idx = desc.get("param_idx", 0)
            sweep_val = float(sv[dim_idx])

            if param_idx != 0:
                raise NotImplementedError(
                    f"C++ sweep: only param_idx=0 (linear multiplier 'a') can be "
                    f"folded into scale; got param_idx={param_idx} for '{proj_name}'. "
                    f"Additive offsets (b) are not representable as a runtime scale."
                )

            found = False
            for i, p in enumerate(compiled.spec.intra_projections):
                if p.name == proj_name:
                    intra_scales[i] = sweep_val * float(p.scale)
                    found = True
                    break
            if not found:
                for i, p in enumerate(compiled.spec.inter_projections):
                    if p.name == proj_name:
                        mode_factor = (
                            float(p.mode_map.flat[0])
                            if p.mode_map is not None
                            else 1.0
                        )
                        inter_scales[i] = sweep_val * float(p.scale) * mode_factor
                        found = True
                        break
            if not found:
                raise ValueError(
                    f"Projection '{proj_name}' not found in compiled spec. "
                    f"Available intra: {[p.name for p in compiled.spec.intra_projections]}, "
                    f"inter: {[p.name for p in compiled.spec.inter_projections]}."
                )

        return intra_scales, inter_scales

    def _run_for_sweep(
        self,
        compiled,
        nstep: int,
        flat_states: list[np.ndarray],
        intra_scales: list[float],
        inter_scales: list[float],
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Run one sweep point; return ``[(times, data, ctavg), ...]`` per subnet."""
        monitor_type = (
            compiled.spec.monitors[0].type_name
            if compiled.spec.monitors
            else "TemporalAverage"
        )
        # chunk_size drives how many integration steps the C++ module processes per
        # output sample.  For TemporalAverage derive it from the monitor period so
        # callers get one output row per period (multiple time points per sweep).
        mon_spec = compiled.spec.monitors[0] if compiled.spec.monitors else None
        if (
            monitor_type == "TemporalAverage"
            and mon_spec is not None
            and mon_spec.period is not None
        ):
            chunk_size = max(1, round(mon_spec.period / compiled.spec.dt))
        else:
            chunk_size = nstep
        bold_monitor = None
        if monitor_type in ("Raw", "RawVoi", "AfferentCoupling",
                            "AfferentCouplingTemporalAverage"):
            chunk_size = 1
        elif monitor_type == "Bold":
            bold_monitor = next(
                (m for m in compiled.lowering.monitors if type(m).__name__ == "Bold"),
                None,
            )
            if bold_monitor is not None:
                _configure_bold_monitor(bold_monitor, compiled.spec.dt)
                chunk_size = bold_monitor._interim_istep

        module = compiled.load_module()

        intra_data = [_projection_arrays(p) for p in compiled.spec.intra_projections]
        inter_data = [_inter_projection_arrays(p) for p in compiled.spec.inter_projections]

        # Override projection scales for this sweep point.
        for i, s in enumerate(intra_scales):
            intra_data[i]["scale"] = s
        for i, s in enumerate(inter_scales):
            inter_data[i]["scale"] = s

        noise_arrays = _build_noise_arrays(compiled.spec, compiled.lowering, int(nstep))
        stim_arrays = _build_stimulus_arrays(compiled.spec, compiled.lowering, int(nstep))

        raw = module.run_simulation(
            flat_states,
            int(nstep),
            int(chunk_size),
            [p["weights_data"]     for p in intra_data],
            [p["weights_indices"]  for p in intra_data],
            [p["weights_indptr"]   for p in intra_data],
            [p["idelays"]          for p in intra_data],
            [p["source_svar"]      for p in intra_data],
            [p["target_cvar_slot"] for p in intra_data],
            [p["scale"]            for p in intra_data],
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

        if bold_monitor is not None:
            result = []
            for times, data, ctavg in raw:
                t_arr = np.asarray(times)
                d_arr = np.asarray(data)
                b_times, b_data = _apply_bold_hrf(
                    bold_monitor, t_arr, d_arr, compiled.spec.dt
                )
                result.append((b_times, b_data, np.asarray(ctavg)))
            return result

        return [
            (np.asarray(times), np.asarray(data), np.asarray(ctavg))
            for times, data, ctavg in raw
        ]

    def _stack_cpp_results(
        self,
        raw_results: list,
        network_set,
        node_indices: Optional[dict],
        sweep_values: np.ndarray,
        backend_label: str,
        elapsed: float,
    ) -> SweepResult:
        """Stack per-sweep-point tuples into a :class:`SweepResult`."""
        subnet_names = [sn.name for sn in network_set.subnets]
        tavg_dict: dict = {}
        ctavg_dict: dict = {}

        for si, sname in enumerate(subnet_names):
            # data shape per sweep point: (n_chunks, n_voi, n_nodes, n_modes)
            # stack over sweeps → (n_sweeps, n_chunks, n_voi, n_nodes, n_modes)
            tavg_dict[sname] = np.stack(
                [r[si][1] for r in raw_results], axis=0
            )
            ctavg_dict[sname] = np.stack(
                [r[si][2] for r in raw_results], axis=0
            )

        if node_indices:
            n_global = max(max(idxs) for idxs in node_indices.values()) + 1
            ref = next(iter(tavg_dict.values()))
            # ref shape: (n_sweeps, n_chunks, n_voi, n_nodes, n_modes)
            merged = np.zeros(
                (ref.shape[0], ref.shape[1], ref.shape[2], n_global, ref.shape[4]),
                dtype=ref.dtype,
            )
            for sname in subnet_names:
                if sname in node_indices:
                    merged[:, :, :, node_indices[sname], :] = tavg_dict[sname]
            merged_tavg: Optional[np.ndarray] = merged
        else:
            voi_counts = {a.shape[2] for a in tavg_dict.values()}
            merged_tavg = (
                np.concatenate(list(tavg_dict.values()), axis=3)
                if len(voi_counts) == 1
                else None
            )

        times = raw_results[0][0][0] if raw_results else np.array([])
        return SweepResult(
            tavg=tavg_dict,
            merged_tavg=merged_tavg,
            ctavg=ctavg_dict,
            times=times,
            sweep_values=sweep_values,
            backend=backend_label,
            elapsed=elapsed,
        )
