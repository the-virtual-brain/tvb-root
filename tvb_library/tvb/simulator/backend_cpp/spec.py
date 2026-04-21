from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

import numpy as np


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_array(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value)


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _array_payload(arr: np.ndarray) -> dict[str, Any]:
    contiguous = _normalize_array(arr)
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "bytes_sha256": hashlib.sha256(contiguous.view(np.uint8).tobytes()).hexdigest(),
    }


@dataclasses.dataclass(frozen=True)
class IntegratorSpec:
    type_name: str
    dt: float
    is_stochastic: bool = False
    noise_nsig: np.ndarray | None = None

    def payload(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type_name": self.type_name,
            "dt": float(self.dt),
            "is_stochastic": self.is_stochastic,
        }
        if self.noise_nsig is not None:
            data["noise_nsig"] = _array_payload(self.noise_nsig)
        return data


@dataclasses.dataclass(frozen=True)
class SubnetworkSpec:
    name: str
    model_type: str
    integrator: IntegratorSpec
    n_nodes: int
    n_modes: int
    n_state_vars: int
    n_coupling_vars: int
    variables_of_interest: tuple[str, ...]
    state_variables: tuple[str, ...]
    parameter_values: dict[str, np.ndarray]
    initial_state_shape: tuple[int, int, int]
    has_stimulus: bool = False

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "integrator": self.integrator.payload(),
            "n_nodes": self.n_nodes,
            "n_modes": self.n_modes,
            "n_state_vars": self.n_state_vars,
            "n_coupling_vars": self.n_coupling_vars,
            "variables_of_interest": list(self.variables_of_interest),
            "state_variables": list(self.state_variables),
            "parameter_values": {
                key: _array_payload(value)
                for key, value in sorted(self.parameter_values.items())
            },
            "initial_state_shape": list(self.initial_state_shape),
            "has_stimulus": self.has_stimulus,
        }


@dataclasses.dataclass(frozen=True)
class ProjectionSpec:
    name: str
    source_subnet: str
    target_subnet: str
    source_cvar: np.ndarray
    target_cvar: np.ndarray
    weights_data: np.ndarray
    weights_indices: np.ndarray
    weights_indptr: np.ndarray
    idelays: np.ndarray
    horizon: int
    scale: float
    target_scales: np.ndarray
    cfun_type: str
    cfun_params: np.ndarray
    cvar_mapping_mode: str
    is_inter: bool
    mode_map: np.ndarray | None = None
    n_src_modes: int | None = None

    @property
    def n_tgt_nodes(self) -> int:
        return int(self.weights_indptr.shape[0] - 1)

    @property
    def n_tgt_modes(self) -> int:
        if self.is_inter:
            assert self.mode_map is not None
            return int(self.mode_map.shape[1])
        assert self.n_src_modes is not None
        return int(self.n_src_modes)

    def payload(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "source_subnet": self.source_subnet,
            "target_subnet": self.target_subnet,
            "source_cvar": _array_payload(self.source_cvar),
            "target_cvar": _array_payload(self.target_cvar),
            "weights_data": _array_payload(self.weights_data),
            "weights_indices": _array_payload(self.weights_indices),
            "weights_indptr": _array_payload(self.weights_indptr),
            "idelays": _array_payload(self.idelays),
            "horizon": self.horizon,
            "scale": float(self.scale),
            "target_scales": _array_payload(self.target_scales),
            "cfun_type": self.cfun_type,
            "cfun_params": _array_payload(self.cfun_params),
            "cvar_mapping_mode": self.cvar_mapping_mode,
            "is_inter": self.is_inter,
            "n_src_modes": self.n_src_modes,
        }
        if self.mode_map is not None:
            data["mode_map"] = _array_payload(self.mode_map)
        return data


@dataclasses.dataclass(frozen=True)
class MonitorSpec:
    type_name: str
    period: float | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "type_name": self.type_name,
            "period": _normalize_scalar(self.period),
        }


@dataclasses.dataclass(frozen=True)
class StimulusSpec:
    target_subnet: str
    count: int

    def payload(self) -> dict[str, Any]:
        return {
            "target_subnet": self.target_subnet,
            "count": self.count,
        }


@dataclasses.dataclass(frozen=True)
class SimulationSpec:
    backend_version: str
    dt: float
    subnetworks: tuple[SubnetworkSpec, ...]
    inter_projections: tuple[ProjectionSpec, ...]
    intra_projections: tuple[ProjectionSpec, ...]
    monitors: tuple[MonitorSpec, ...]
    stimuli: tuple[StimulusSpec, ...]
    source_horizons: dict[str, int]
    user_source_hint: str | None = None

    @property
    def all_projections(self) -> tuple[ProjectionSpec, ...]:
        return self.inter_projections + self.intra_projections

    def payload(self) -> dict[str, Any]:
        return {
            "backend_version": self.backend_version,
            "dt": float(self.dt),
            "subnetworks": [sn.payload() for sn in self.subnetworks],
            "inter_projections": [p.payload() for p in self.inter_projections],
            "intra_projections": [p.payload() for p in self.intra_projections],
            "monitors": [m.payload() for m in self.monitors],
            "stimuli": [s.payload() for s in self.stimuli],
            "source_horizons": dict(sorted(self.source_horizons.items())),
            "user_source_hint": self.user_source_hint,
        }

    def cache_key(self) -> str:
        return _hash_payload(self.payload())

