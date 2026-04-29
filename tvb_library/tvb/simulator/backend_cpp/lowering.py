from __future__ import annotations

import dataclasses

import numpy as np

from tvb.simulator.backend.nb_hybrid import (
    NbHybridBackend,
    _cfun_params,
    _cfun_type,
    _cvar_mapping_mode,
)
from tvb.simulator.integrators import HeunDeterministic

from .spec import (
    IntegratorSpec,
    MonitorSpec,
    ProjectionSpec,
    SimulationSpec,
    StimulusSpec,
    SubnetworkSpec,
)


BACKEND_VERSION = "0.1"


@dataclasses.dataclass(frozen=True)
class SpecLoweringResult:
    spec: SimulationSpec
    analysis: object


def _check_scope_compatibility(network_set) -> None:
    """Compatibility gate for the C++ backend.

    Accepts any TVB model that exposes `state_variable_dfuns` (the standard
    expression-based dfun interface). Models using a custom Numba template
    (e.g. Zerlaut) are rejected at this point with a clear message.
    """
    if not network_set.subnets:
        raise ValueError("NetworkSet must contain at least one subnetwork.")

    dt0 = float(network_set.subnets[0].scheme.dt)
    for subnet in network_set.subnets:
        dfuns = getattr(subnet.model, "state_variable_dfuns", None)
        if not dfuns:
            raise NotImplementedError(
                f"CppHybridBackend requires models that expose state_variable_dfuns. "
                f"'{type(subnet.model).__name__}' uses a custom dfun template and is "
                f"not yet supported."
            )
        if not isinstance(subnet.scheme, HeunDeterministic):
            raise NotImplementedError(
                "CppHybridBackend currently supports only HeunDeterministic integrators."
            )
        if float(subnet.scheme.dt) != dt0:
            raise ValueError(
                "All subnetworks must share the same dt."
            )


def _collect_model_parameters(model: object) -> dict[str, np.ndarray]:
    parameter_values: dict[str, np.ndarray] = {}
    for name in getattr(model, "parameter_names", ()):
        value = getattr(model, name)
        parameter_values[name] = np.ascontiguousarray(np.atleast_1d(value))
    return parameter_values


def _extract_state_bounds(
    model: object,
) -> tuple[dict[str, float], dict[str, float]]:
    lo_bounds: dict[str, float] = {}
    hi_bounds: dict[str, float] = {}
    svb = getattr(model, "state_variable_boundaries", None) or {}
    for svar, bounds in svb.items():
        bounds_arr = np.atleast_1d(np.asarray(bounds, dtype=float))
        if bounds_arr.shape[0] >= 1 and np.isfinite(bounds_arr[0]):
            lo_bounds[str(svar)] = float(bounds_arr[0])
        if bounds_arr.shape[0] >= 2 and np.isfinite(bounds_arr[1]):
            hi_bounds[str(svar)] = float(bounds_arr[1])
    return lo_bounds, hi_bounds


def _build_integrator_spec(subnet_info) -> IntegratorSpec:
    return IntegratorSpec(
        type_name=type(subnet_info.integrator).__name__,
        dt=float(subnet_info.integrator.dt),
        is_stochastic=bool(subnet_info.is_stochastic),
        noise_nsig=None
        if subnet_info.noise_nsig is None
        else np.ascontiguousarray(subnet_info.noise_nsig.astype(np.float64)),
    )


def _build_subnetwork_spec(subnet_info) -> SubnetworkSpec:
    model = subnet_info.model
    lo_bounds, hi_bounds = _extract_state_bounds(model)
    return SubnetworkSpec(
        name=subnet_info.name,
        model_type=type(model).__name__,
        integrator=_build_integrator_spec(subnet_info),
        n_nodes=int(subnet_info.n_nodes),
        n_modes=int(subnet_info.n_modes),
        n_state_vars=int(model.nvar),
        n_coupling_vars=int(len(model.cvar)),
        variables_of_interest=tuple(str(x) for x in model.variables_of_interest),
        state_variables=tuple(str(x) for x in model.state_variables),
        parameter_values=_collect_model_parameters(model),
        initial_state_shape=(int(model.nvar), int(subnet_info.n_nodes), int(subnet_info.n_modes)),
        has_stimulus=bool(subnet_info.has_stimulus),
        global_parameter_names=tuple(
            str(p) for p in (getattr(model, "global_parameter_names", None) or [])
        ),
        coupling_terms=tuple(
            str(ct) for ct in (getattr(model, "coupling_terms", None) or [])
        ),
        state_variable_dfuns={
            str(k): str(v)
            for k, v in (getattr(model, "state_variable_dfuns", None) or {}).items()
        },
        dfun_intermediates=tuple(
            (str(name), str(expr))
            for name, expr in (getattr(model, "dfun_intermediates", None) or [])
        ),
        dfun_helpers=tuple(
            (str(name), str(args), str(expr))
            for name, args, expr in (getattr(model, "dfun_helpers", None) or [])
        ),
        state_lower_bounds=lo_bounds,
        state_upper_bounds=hi_bounds,
    )


def _build_projection_spec(proj_info) -> ProjectionSpec:
    mode_map = None
    if proj_info.mode_map is not None:
        mode_map = np.ascontiguousarray(proj_info.mode_map.astype(np.float32))

    n_src_modes = None
    if not proj_info.is_inter:
        n_src_modes = int(proj_info.n_src_modes)

    return ProjectionSpec(
        name=proj_info.name,
        source_subnet=proj_info.source_subnet,
        target_subnet=proj_info.target_subnet,
        source_cvar=np.ascontiguousarray(proj_info.source_cvar.astype(np.int32)),
        target_cvar=np.ascontiguousarray(proj_info.target_cvar.astype(np.int32)),
        weights_data=np.ascontiguousarray(proj_info.weights_data.astype(np.float32)),
        weights_indices=np.ascontiguousarray(proj_info.weights_indices.astype(np.int32)),
        weights_indptr=np.ascontiguousarray(proj_info.weights_indptr.astype(np.int32)),
        idelays=np.ascontiguousarray(proj_info.idelays.astype(np.int32)),
        horizon=int(proj_info.horizon),
        scale=float(proj_info.scale),
        target_scales=np.ascontiguousarray(proj_info.target_scales.astype(np.float32)),
        cfun_type=_cfun_type(proj_info),
        cfun_params=np.ascontiguousarray(_cfun_params(proj_info).astype(np.float32)),
        cvar_mapping_mode=_cvar_mapping_mode(proj_info),
        is_inter=bool(proj_info.is_inter),
        mode_map=mode_map,
        n_src_modes=n_src_modes,
    )


def _build_monitor_specs(monitors: list[object] | None) -> tuple[MonitorSpec, ...]:
    if not monitors:
        return ()
    specs: list[MonitorSpec] = []
    for monitor in monitors:
        period = getattr(monitor, "period", None)
        specs.append(MonitorSpec(type_name=type(monitor).__name__, period=period))
    return tuple(specs)


def _build_stimulus_specs(analysis) -> tuple[StimulusSpec, ...]:
    items: list[StimulusSpec] = []
    for subnet_name, stimuli in sorted(analysis.stimuli_by_subnet.items()):
        if not stimuli:
            continue
        items.append(StimulusSpec(target_subnet=subnet_name, count=len(stimuli)))
    return tuple(items)


def lower_network_set(
    network_set,
    monitors: list[object] | None = None,
    user_source_hint: str | None = None,
) -> SpecLoweringResult:
    """Lower a configured hybrid `NetworkSet` into a backend-neutral spec.

    The current implementation reuses `NbHybridBackend._analyse()` as the
    reference lowering path so the new C++ backend can lock the spec boundary
    before re-implementing execution. Compatibility validation is currently a
    narrow local check for the first milestone scope.
    """

    backend = NbHybridBackend()
    _check_scope_compatibility(network_set)
    analysis = backend._analyse(network_set)

    subnetworks = tuple(_build_subnetwork_spec(sn) for sn in analysis.subnetworks)
    inter_projections = tuple(
        _build_projection_spec(proj) for proj in analysis.inter_projections
    )
    intra_projections = tuple(
        _build_projection_spec(proj) for proj in analysis.intra_projections
    )

    spec = SimulationSpec(
        backend_version=BACKEND_VERSION,
        dt=float(network_set.subnets[0].scheme.dt),
        subnetworks=subnetworks,
        inter_projections=inter_projections,
        intra_projections=intra_projections,
        monitors=_build_monitor_specs(monitors),
        stimuli=_build_stimulus_specs(analysis),
        source_horizons={str(k): int(v) for k, v in analysis.source_horizons.items()},
        user_source_hint=user_source_hint,
    )
    return SpecLoweringResult(spec=spec, analysis=analysis)
