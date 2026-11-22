# -*- coding: utf-8 -*-
"""C++ backend for the Hybrid Simulator.

Lowers the Python/NumPy hybrid implementation into tight, SIMD-enabled C++
with the simulation control flow in C++ (no monolithic static codegen).
Python keeps control of the simulation; data arrays stay in NumPy float32.

Mirrors the :class:`~tvb.simulator.backend.nb_hybrid.NbHybridBackend` API.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import List, Optional

import numpy as np

from ._cpp_hybrid import Sim

# generic (expression-generated) model ids start at 100 (see _core.cpp)
_GENERIC_BASE = 100


@dataclasses.dataclass(frozen=True)
class _GeneratedLibrary:
    handle: object
    ids: dict
    metadata: dict
    function_pointer: int


_GEN_LIB = {}        # library path -> _GeneratedLibrary; handles never evicted
_GEN_IDS = {}        # model class name -> generic id
_GEN_META = {}       # model class name -> emit_sources metadata (runtime lib)
_BUILTIN_TABLE = None  # built-in (build-time) generic model table, or None
from tvb.simulator.backend.hybrid_common import (
    _CFUN_PARAM_ATTRS,
    _cfun_type,
    _apply_monitors,
    _aggregate_raw_outputs,
    _can_merge_subnets,
    _compute_chunk_size,
    _copy_monitor_states,
    _resolve_named_params,
)
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.monitors import BalloonWindkessel, _reject_balloon_windkessel_sweep
from tvb.simulator.integrators import (
    HeunDeterministic,
    EulerDeterministic,
    HeunStochastic,
    EulerStochastic,
)

# Scalars shared by all regions; lower region-specific parameters here once supported.
_BALLOON_WINDKESSEL_NOMINAL = dict(
    signal_decay_s=0.65, flow_feedback_s2=0.41, transit_time_s=0.98,
    grubb_exponent=0.32, resting_extraction=0.4, intravascular_ratio=0.5,
    frequency_offset_hz=40.3, relaxation_rate_per_s=25.0, echo_time_s=0.04,
    resting_blood_volume=0.04,
)

__all__ = [
    "CppHybridBackend",
    "CompiledCppSim",
    "NetworkAnalysis",
    "SubnetworkInfo",
    "ProjectionInfo",
    "SweepResult",
]

# cfun dispatch ids (must match _core.cpp); variants are resolved via
# nb_hybrid._cfun_type so classic/legacy and static/dynamic forms dispatch
# to distinct C++ implementations.
_CFUN_TYPE_IDS = {
    "none": -1,
    "linear": 0, "scaling": 1, "sigmoidal": 2, "difference": 3,
    "kuramoto": 4, "tanh": 5, "sigmoidal_jr": 6, "pre_sigmoidal": 7,
    "sigmoidal_jr_legacy": 8, "pre_sigmoidal_dynamic": 9,
}
_CFUN_IDS = {
    "Linear": 0, "Scaling": 1, "Sigmoidal": 2, "Difference": 3,
    "Kuramoto": 4, "HyperbolicTangent": 5, "SigmoidalJansenRit": 6,
    "PreSigmoidal": 7,
}
# PreSigmoidal needs the globalT flag at slot 5 for the dynamic variant.
_CPP_CFUN_PARAM_ATTRS = dict(_CFUN_PARAM_ATTRS)
_CPP_CFUN_PARAM_ATTRS["PreSigmoidal"] = (
    _CFUN_PARAM_ATTRS["PreSigmoidal"] + [("globalT", 5)])
_PASSTHROUGH_CFUN = -1  # no cfun: plain weighted sum
# model dispatch ids (must match _core.cpp)
_MODEL_IDS = {
    "MontbrioPazoRoxin": 0, "Generic2dOscillator": 1, "Kuramoto": 2,
    "SupHopf": 3, "Linear": 4, "ReducedWongWang": 5, "WilsonCowan": 6,
    "JansenRit": 7, "Epileptor": 8, "Epileptor2D": 9,
    "ZerlautAdaptationFirstOrder": 10, "ZerlautAdaptationSecondOrder": 11,
    "CerebellarMF": 12,
}
# per-model parameter attribute order (must match the C++ dfuns; this is the
# nb_hybrid gufunc argument order for each model)
_MODEL_PARM_NAMES = {
    "MontbrioPazoRoxin": ["tau", "I", "Delta", "J", "eta", "cr", "cv"],
    "Generic2dOscillator": ["tau", "I", "a", "b", "c", "d", "e", "f", "g",
                            "beta", "alpha", "gamma"],
    "Kuramoto": ["omega"],
    "SupHopf": ["a", "omega"],
    "Linear": ["gamma"],
    "ReducedWongWang": ["a", "b", "d", "gamma", "tau_s", "w", "J_N", "I_o"],
    "WilsonCowan": ["c_ee", "c_ei", "c_ie", "c_ii", "tau_e", "tau_i", "a_e",
                    "b_e", "c_e", "theta_e", "a_i", "b_i", "c_i", "theta_i",
                    "r_e", "r_i", "k_e", "k_i", "P", "Q", "alpha_e",
                    "alpha_i", "shift_sigmoid"],
    "JansenRit": ["nu_max", "r", "v0", "a", "a_1", "a_2", "a_3", "a_4",
                  "A", "b", "B", "J", "mu"],
    "Epileptor": ["x0", "Iext", "Iext2", "a", "b", "slope", "tt", "Kvf",
                  "c", "d", "r", "Ks", "Kf", "aa", "bb", "tau", "modification"],
    "Epileptor2D": ["x0", "Iext", "a", "b", "slope", "c", "d", "r", "Kvf",
                    "Ks", "tt", "modification"],
}
_MODEL_ZERLAUT_BASE = [
    "g_L", "E_L_e", "E_L_i", "C_m", "b_e", "a_e", "b_i", "a_i",
    "tau_w_e", "tau_w_i", "E_e", "E_i", "Q_e", "Q_i", "tau_e", "tau_i",
    "N_tot", "p_connect_e", "p_connect_i", "g", "K_ext_e", "K_ext_i",
    "T", "external_input_ex_ex", "external_input_ex_in",
    "external_input_in_ex", "external_input_in_in", "tau_OU", "weight_noise",
]
_MODEL_ZERLAUT_POLY = (["P_e:%d" % k for k in range(10)] +
                       ["P_i:%d" % k for k in range(10)])
_MODEL_PARM_NAMES["ZerlautAdaptationFirstOrder"] = (
    list(_MODEL_ZERLAUT_BASE) + list(_MODEL_ZERLAUT_POLY))
_MODEL_PARM_NAMES["ZerlautAdaptationSecondOrder"] = (
    list(_MODEL_ZERLAUT_BASE) + ["S_i"] + list(_MODEL_ZERLAUT_POLY))

# CerebellarMF: runtime (per-node) parameters in nb_hybrid template order
# (_nb_hybrid_runtime_parameter_names), then the polynomial coefficients
# expanded per index (P_<pop>:i read element i of the base array), then the
# two mode flags. Must match the CRBL param indexing in _core.cpp.
_MODEL_CRBL_RUNTIME = [
    "g_L_grc", "g_L_goc", "g_L_mli", "g_L_pc",
    "E_L_grc", "E_L_goc", "E_L_mli", "E_L_pc",
    "C_m_grc", "C_m_goc", "C_m_mli", "C_m_pc",
    "E_e", "E_i",
    "Q_mf_grc", "Q_mf_goc", "Q_grc_goc", "Q_grc_mli", "Q_grc_pc",
    "Q_goc_grc", "Q_goc_goc", "Q_mli_mli", "Q_mli_pc",
    "tau_mf_grc", "tau_mf_goc", "tau_grc_goc", "tau_grc_mli", "tau_grc_pc",
    "tau_goc_grc", "tau_goc_goc", "tau_mli_mli", "tau_mli_pc",
    "K_mossy_grc", "K_mossy_goc", "K_grc_goc", "K_grc_mli", "K_grc_pc",
    "K_goc_goc", "K_mli_mli", "K_mli_pc",
    "N_grc", "N_goc", "N_mli", "N_pc", "N_mossy",
    "alpha_grc", "alpha_goc", "alpha_mli", "alpha_pc",
    "T",
    "tau_OU", "weight_noise",
    "external_input_ex_ex", "external_input_ex_in", "external_input_in_ex",
    "frac_mossy", "frac_parallel",
    "mf_to_grc", "mf_to_goc",
    "pf_to_goc", "pf_to_mli", "pf_to_pc",
]
_MODEL_PARM_NAMES["CerebellarMF"] = (
    list(_MODEL_CRBL_RUNTIME)
    + [f"P_grc:{k}" for k in range(5)]
    + [f"P_goc:{k}" for k in range(5)]
    + [f"P_mli:{k}" for k in range(5)]
    + [f"P_pc:{k}" for k in range(5)]
    + ["use_legacy_goc_e_e", "add_noise_mli_pc"])



def _squeeze_lane(a):
    """Drop a trailing singleton SIMD-lane dim (nb_hybrid keeps 3-dim states)."""
    if a.ndim == 4 and a.shape[-1] == 1:
        return a[..., 0]
    return a


def _builtin_model_table():
    """The generic-model table the extension's build-time kernels describe.

    Read once from ``_cpp_hybrid.generic_model_table()`` and cached: model id,
    buffer shapes (n_parm/n_cvar/n_svar) and parameter order all come from the
    very table the generated kernels were emitted from, so a stock generic model
    needs no runtime g++ and no Python-side re-derivation of ids or packing
    order.  Empty when the extension was built with
    ``-DTVB_CPP_GENERATE_MODELS=OFF`` (or predates the table), which is the
    signal for the runtime g++/ctypes path to cover every generic model, as it
    did before the built-in table existed.
    """
    global _BUILTIN_TABLE
    if _BUILTIN_TABLE is None:
        from . import _cpp_hybrid
        getter = getattr(_cpp_hybrid, "generic_model_table", None)
        _BUILTIN_TABLE = ({} if getter is None else
                          {str(k): dict(v) for k, v in getter().items()})
    return _BUILTIN_TABLE


@dataclasses.dataclass
class ProjectionInfo:
    name: str = ""
    lookup: str = ""
    is_inter: bool = False
    src_sn: int = 0
    tgt_sn: int = 0
    w: np.ndarray = None
    idx: np.ndarray = None
    ptr: np.ndarray = None
    idelays: np.ndarray = None
    horizon: int = 1
    cfun_id: int = 0
    src_cvar: int = 0
    src_cvars_list: list = dataclasses.field(default_factory=lambda: [0])
    n_src_cvars: int = 1
    tgt_cvar: int = 0
    n_cfun_parm: int = 0
    scale: float = 1.0
    ts: float = 1.0
    tgt_state_cvar: int = 0
    cfun_params: np.ndarray = None
    cfun_use_classic: bool = False
    mode_map: np.ndarray = None
    target_scales: np.ndarray = None
    target_cvars_list: list = dataclasses.field(default_factory=lambda: None)

    @property
    def source_subnet(self):
        return self._source_subnet

    @property
    def n_tgt_nodes(self):
        return int(self.ptr.shape[0] - 1) if self.ptr is not None else 0


@dataclasses.dataclass
class SubnetworkInfo:
    name: str = ""
    n_nodes: int = 0
    n_svar: int = 0
    n_parm: int = 0
    n_cvar: int = 0
    model_id: int = 0
    model_name: str = ""
    model: object = None
    params: np.ndarray = None
    voi: list = dataclasses.field(default_factory=list)
    dt: float = 0.0
    k: int = 1  # steps per master tick: dt == k * dt0 (multi-dt)
    horizon: int = 1
    n_modes: int = 1
    is_stochastic: bool = False
    noise_nsig: np.ndarray = None
    noise_seed: int = 0
    noise_ntau: float = 0.0
    stimuli: list = dataclasses.field(default_factory=list)
    node_indices: np.ndarray = None

    @property
    def has_stimulus(self):
        return bool(self.stimuli)
    parm_index: dict = dataclasses.field(default_factory=dict)
    integ_id: int = 1
    sv_lo: Optional[np.ndarray] = None
    sv_hi: Optional[np.ndarray] = None
    clamp_indices: Optional[np.ndarray] = None
    clamp_values: Optional[np.ndarray] = None
    runtime_library: Optional[_GeneratedLibrary] = None  # owns the subnet dfun pointer


@dataclasses.dataclass
class NetworkAnalysis:
    subnets: list = dataclasses.field(default_factory=list)
    projections: list = dataclasses.field(default_factory=list)
    dt: float = 0.0

    @property
    def all_projections(self) -> List[ProjectionInfo]:
        return list(self.projections)

    @property
    def subnetworks(self):
        return self.subnets


@dataclasses.dataclass
class SweepResult:
    """Container for parameter-sweep results (mirrors nb_hybrid.SweepResult)."""

    tavg: dict = dataclasses.field(default_factory=dict)
    merged_tavg: np.ndarray = None
    ctavg: dict = dataclasses.field(default_factory=dict)
    times: np.ndarray = None
    sweep_values: np.ndarray = None
    snapshot: Optional[dict] = None
    backend: str = ""
    elapsed: float = 0.0
    raw: Optional[dict] = None
    bold: Optional[dict] = None
    param_keys: Optional[list] = None
    online_features: Optional[dict] = None

    # --- notebook sugar --------------------------------------------------
    def squeezed(self, subnet=None):
        """(sweep, time, svar, node) view for one subnet (or the merged).

        The trailing ``[..., 1]`` lane axis is dropped; ``subnet`` selects a
        named subnet, defaulting to the merged table for multi-subnet nets.
        """
        if subnet is None:
            return np.asarray(self.merged_tavg)[..., 0]
        return np.asarray(self.tavg[subnet])[..., 0]

    def ts(self, svar=0, subnet=None):
        """(sweep, node, time) time series for one state variable.

        Convenience for feeding straight to :func:`features` / feature
        computation: time is moved to the last axis.
        """
        a = self.squeezed(subnet)                      # (sweep, time, svar, node)
        return np.moveaxis(a[:, :, svar, :], 1, -1)    # (sweep, node, time)

    def to_dataframe(self, svar=0, subnet=None, average_time=True):
        """pandas DataFrame, one row per sweep member.

        Columns are the swept parameters then per-node values of ``svar``
        (time-averaged by default, as ``{node}``).  Requires pandas.
        """
        import pandas as pd
        sv = np.asarray(self.sweep_values)
        if sv.ndim == 1:
            sv = sv[:, None]
        n = sv.shape[0]
        keys = self.param_keys or [f"param{i}" for i in range(sv.shape[1])]
        data = {k: sv[:, j] for j, k in enumerate(keys)}
        a = self.squeezed(subnet)              # (sweep, time, svar, node)
        va = a[:, :, svar, :]
        for node in range(va.shape[2]):
            col = va[:, :, node]
            data[str(node)] = col.mean(axis=1) if average_time else col[:, -1]
        return pd.DataFrame(data, index=range(n))

    def features(self, feature_names, svar=0, subnet=None, fs=1.0, **params):
        """Compute data-features over the per-sweep time series.

        Returns a dict ``{feature: (n_sweep, n_node) ndarray}`` computed over
        the time axis of ``svar`` for every sweep member -- ideal for
        sweeping a parameter and plotting each feature against it.
        """
        from . import features as _feat
        out = {}
        remaining = []
        # Fast path: reuse Tier-1 online streaming features if already computed.
        onl_src = None
        if self.online_features is not None and not params.get("indices"):
            if subnet is not None and subnet in self.online_features:
                onl_src = self.online_features[subnet]
            elif subnet is None and len(self.online_features) == 1:
                onl_src = next(iter(self.online_features.values()))
        order_req = int(params.get("order", 4))
        for k in feature_names:
            if (onl_src is not None and isinstance(k, str) and k in onl_src
                    and (k != "moments" or order_req == 4)):
                arr = onl_src[k]
                if k == "moments":
                    out[k] = arr[:, :, svar, :].transpose(1, 0, 2)
                else:
                    out[k] = arr[:, svar, :]
            else:
                remaining.append(k)
        if not remaining:
            return out

        ts = self.ts(svar=svar, subnet=subnet)   # (sweep, node, time)
        str_feats = [k for k in remaining if isinstance(k, str)]
        call_feats = [k for k in remaining if not isinstance(k, str)]
        if str_feats:
            # Batch C++ path over (time, sweep, node)
            ts_t = np.moveaxis(ts, -1, 0)        # (time, sweep, node)
            for k, val in _feat.compute_features(
                    ts_t, str_feats, fs=fs, **params):
                out[k] = np.asarray(val, dtype=np.float64)
        for k in call_feats:
            res = np.empty(ts.shape[:2], dtype=np.float64)
            for si in range(ts.shape[0]):
                cur = np.moveaxis(ts[si], -1, 0)     # (time, node)
                res[si] = _feat.compute_feature(cur, k, fs=fs, **params)
            out[k] = res
        return out


class CompiledCppSim:
    """A compiled simulation for a NetworkSet; run without re-building."""

    def __init__(self, backend, analysis, network_set, sim, width):
        self._backend = backend
        self._analysis = analysis
        self._network_set = network_set
        self._sim = sim
        self._width = width
        self._bold_states = None
        self._kmon_armed_keys = set()
        self._pending_kmon_states = None
        self._pending_kmon_global_states = None
        self._pending_cpp_rng_states = None

    def warmup(self) -> float:
        return 0.0  # no JIT: nothing to warm up

    def run(
        self,
        nstep: int,
        chunk_size: int = None,
        initial_states: Optional[list] = None,
        monitors: Optional[list] = None,
        return_snapshot: bool = False,
        _step_offset: int = 0,
        _resuming: bool = False,
        rng_mode: str = "numpy",
        subnet_monitors: Optional[dict] = None,
        **kwargs,
    ):
        """Run the simulation for *nstep* steps.

        Mirrors :meth:`NbHybridBackend.run`.  With no monitors returns a list
        per subnetwork of ``(times, data, ctavg)``.  With monitors returns
        ``list[monitor][subnetwork]`` of ``(times, data)``.
        When *subnet_monitors* is provided (dict ``{sn.name: [Monitor, ...]}``
        or list aligned to subnets), returns ``(outputs, subnet_results)``
        (or ``(outputs, snapshot, subnet_results)`` if *return_snapshot* is
        True), collecting per-subnet monitors in the same single simulation
        pass with ``cadence_K = sn.k`` and ``dt = sn.dt``.
        """
        if rng_mode not in ("numpy", "philox"):
            raise ValueError(
                f"Unsupported rng_mode {rng_mode!r}; expected 'numpy' or 'philox'."
            )
        if rng_mode == "numpy":
            for sn_info in self._analysis.subnets:
                if sn_info.is_stochastic and sn_info.noise_ntau > 0.0:
                    raise NotImplementedError(
                        f"Colored OU noise (ntau={sn_info.noise_ntau} > 0) in "
                        f"subnetwork '{sn_info.name}' requires rng_mode='philox'."
                    )

        dt = float(min(sn.scheme.dt for sn in self._network_set.subnets))
        explicit_chunk_size = chunk_size
        if kwargs.get("_monitor_states") is not None:
            self._bold_states = _copy_monitor_states(kwargs["_monitor_states"])

        # Normalize optional per-subnet monitors (e.g. hybrid Recorder monitors)
        sn_mons_map = None
        if subnet_monitors is not None:
            local_lists = subnet_monitors.values() if isinstance(subnet_monitors, dict) else subnet_monitors
            if any(isinstance(getattr(m, "monitor", m), BalloonWindkessel)
                   for ms in local_lists for m in ms):
                raise NotImplementedError("BalloonWindkessel cannot be a local or Recorder monitor")
            if isinstance(subnet_monitors, dict):
                sn_mons_map = {
                    sn.name: [getattr(m, "monitor", m)
                              for m in subnet_monitors.get(sn.name, ())]
                    for sn in self._analysis.subnets
                }
            else:
                sn_mons_map = {
                    sn.name: [getattr(m, "monitor", m)
                              for m in (subnet_monitors[si]
                                        if si < len(subnet_monitors) else ())]
                    for si, sn in enumerate(self._analysis.subnets)
                }

        from tvb.simulator.monitors import (
            AfferentCoupling,
            AfferentCouplingTemporalAverage,
            Bold,
            GlobalAverage,
            Projection,
            Raw,
            SpatialAverage,
            SubSample,
            TemporalAverage,
        )

        n_sn = len(self._analysis.subnets)
        should_merge = (n_sn > 1) and _can_merge_subnets(self._analysis.subnets)
        kernel_global_specs = []   # (mi, m) attached via set_global_monitors
        complete_window_isteps: dict[int, int] = {}
        balloon_windkessel_specs = {}
        kernel_persn_specs = []    # (mi, m) attached via set_monitors(si, ...)
        py_global_indices = set()
        kernel_mon_indices = set()
        has_py_bold = False
        has_py_tavg = False

        if monitors is not None:
            for mi, m in enumerate(monitors):
                if isinstance(m, BalloonWindkessel):
                    balloon_windkessel_specs[mi] = self._balloon_windkessel_spec(mi, m, dt)
                    kernel_global_specs.append((mi, m))
                    kernel_mon_indices.add(mi)
                    continue
                # Multi-subnet globals remain on the unstaged route until their
                # source topology and clock policy are covered.
                if (n_sn == 1
                        and isinstance(m, (TemporalAverage, GlobalAverage, SpatialAverage))
                        and not isinstance(m, AfferentCouplingTemporalAverage)):
                    sn = self._analysis.subnets[0]
                    if not self._backend.enable_kernel_monitors:
                        raise NotImplementedError(
                            "C++ regional complete windows require native monitor collection"
                        )
                    if self._width != 1 or sn.k != 1 or sn.n_modes != 1 or sn.dt != dt:
                        raise NotImplementedError(
                            "C++ regional complete windows require width 1 and one source clock and mode"
                        )
                    complete_window_isteps[mi] = self._backend._complete_window_istep(m, dt)
                    kernel_global_specs.append((mi, m))
                    kernel_mon_indices.add(mi)
                    continue
                if isinstance(m, Projection):
                    if (getattr(m, "_gain", None) is None
                            and getattr(m, "projection", None) is None):
                        raise ValueError(
                            f"Projection monitor {type(m).__name__} has no gain matrix "
                            "configured. Set the gain matrix before running (e.g. via "
                            "config_for_sim or by setting m._gain directly)."
                        )
                can_kernel = (
                    self._backend.enable_kernel_monitors
                    and self._backend._kernel_collectable(m)
                )
                if can_kernel and self._bold_states is not None:
                    # Respect externally supplied Python monitor state dicts
                    if isinstance(m, Bold) and any(
                        k[0] == "bold" and k[1] == mi for k in self._bold_states
                    ):
                        can_kernel = False
                    elif isinstance(m, TemporalAverage) and not isinstance(
                        m, AfferentCouplingTemporalAverage
                    ) and any(
                        k[0] == "temporal_average" and k[1] == mi
                        for k in self._bold_states
                    ):
                        can_kernel = False

                if can_kernel and isinstance(m, (GlobalAverage, SpatialAverage, Projection)):
                    if should_merge or n_sn == 1:
                        total_nodes = (
                            max(int(sn.node_indices.max())
                                for sn in self._analysis.subnets) + 1
                            if should_merge
                            else self._analysis.subnets[0].n_nodes
                        )
                        if isinstance(m, SpatialAverage) and getattr(m, "spatial_mean", None) is not None:
                            if np.asarray(m.spatial_mean).shape[1] != total_nodes:
                                can_kernel = False
                        if isinstance(m, Projection):
                            if np.asarray(m.gain).shape[1] != total_nodes:
                                can_kernel = False
                        if can_kernel:
                            kernel_global_specs.append((mi, m))
                            kernel_mon_indices.add(mi)
                            continue
                    else:
                        # Unmerged multi-subnet: per-subnet if shapes match each subnet
                        if isinstance(m, SpatialAverage) and getattr(m, "spatial_mean", None) is not None:
                            sm_cols = np.asarray(m.spatial_mean).shape[1]
                            if not all(sn.n_nodes == sm_cols for sn in self._analysis.subnets):
                                can_kernel = False
                        if isinstance(m, Projection):
                            g_cols = np.asarray(m.gain).shape[1]
                            if not all(sn.n_nodes == g_cols for sn in self._analysis.subnets):
                                can_kernel = False
                        if can_kernel:
                            kernel_persn_specs.append((mi, m))
                            kernel_mon_indices.add(mi)
                            continue

                if can_kernel:
                    kernel_persn_specs.append((mi, m))
                    kernel_mon_indices.add(mi)
                else:
                    py_global_indices.add(mi)
                    if isinstance(m, Bold):
                        has_py_bold = True
                    if isinstance(m, TemporalAverage):
                        has_py_tavg = True

        if chunk_size is None:
            if balloon_windkessel_specs:
                # A native BW window does not depend on chunk boundaries.
                # Leave companion monitor chunk inference unchanged.
                companions = [m for m in monitors if not isinstance(m, BalloonWindkessel)]
                chunk_size = (_compute_chunk_size(companions, dt) if companions else
                              min(nstep or 1, *(s["istep"] for s in balloon_windkessel_specs.values())))
            else:
                chunk_size = _compute_chunk_size(monitors, dt) if monitors is not None else 1
        if monitors is not None:
            self._backend._validate_monitors(monitors, chunk_size)

        # Partition per-subnet monitors (Recorder monitors)
        sn_kernel_specs = {sn.name: [] for sn in self._analysis.subnets}
        sn_py_fallbacks = {sn.name: [] for sn in self._analysis.subnets}
        need_py_sn_tavg = False
        need_py_sn_ctavg = False
        if sn_mons_map is not None:
            for sn in self._analysis.subnets:
                for ri, rm in enumerate(sn_mons_map.get(sn.name, ())):
                    if (self._backend.enable_kernel_monitors
                            and self._backend._kernel_collectable(rm)):
                        sn_kernel_specs[sn.name].append((ri, rm))
                    else:
                        sn_py_fallbacks[sn.name].append((ri, rm))
                        need_py_sn_tavg = True
                        if isinstance(rm, AfferentCoupling):
                            need_py_sn_ctavg = True

        need_tavg = (
            (monitors is None and subnet_monitors is None)
            or bool(py_global_indices)
            or need_py_sn_tavg
        )
        need_ctavg = (
            (monitors is None and subnet_monitors is None)
            or bool(py_global_indices)
            or need_py_sn_ctavg
        )

        execution_chunk_size = chunk_size
        if has_py_bold or has_py_tavg or need_py_sn_tavg:
            # Python fallback stateful monitors / Recorders consume every step.
            execution_chunk_size = 1
            if self._bold_states is None:
                self._bold_states = {}

        self._kmon_armed_keys = {
            (mi, type(monitors[mi]).__name__) for mi in kernel_mon_indices
        }

        outputs, kernel_streams, subnet_streams = self._backend._run_compiled(
            self._sim, self._analysis, self._network_set,
            nstep=nstep, chunk_size=execution_chunk_size,
            initial_states=initial_states, monitors=monitors,
            width=self._width,
            kernel_persn=kernel_persn_specs,
            kernel_global=kernel_global_specs,
            complete_window_isteps=complete_window_isteps,
            balloon_windkessel_specs=balloon_windkessel_specs,
            sn_kernel_monitors=sn_kernel_specs if sn_mons_map is not None else None,
            explicit_chunk_size=explicit_chunk_size,
            need_tavg=need_tavg,
            need_ctavg=need_ctavg,
            _skip_init=_resuming,
            _step_offset=_step_offset,
            _return_snapshot=return_snapshot,
            rng_mode=rng_mode,
            _pending_kmon_states=self._pending_kmon_states,
            _pending_kmon_global_states=self._pending_kmon_global_states,
            _pending_cpp_rng_states=self._pending_cpp_rng_states,
        )

        # Assemble per-subnet Recorder results if requested
        subnet_results = None
        if sn_mons_map is not None:
            subnet_results = {}
            for si, sn in enumerate(self._analysis.subnets):
                total_r = len(sn_mons_map.get(sn.name, ()))
                res_list = [None] * total_r
                for ri, stream in subnet_streams.get(sn.name, ()):
                    res_list[ri] = stream
                if sn_py_fallbacks[sn.name]:
                    import copy as _copy
                    _, tavg_s, ctavg_s = outputs[si]
                    k_j = sn.k
                    for ri, rm in sn_py_fallbacks[sn.name]:
                        mon_inst = _copy.deepcopy(rm)
                        if hasattr(mon_inst, "_config_dt"):
                            mon_inst._config_dt(float(sn.dt))
                        if hasattr(mon_inst, "compute_hrf"):
                            mon_inst.compute_hrf()
                        src_arr = (
                            ctavg_s if isinstance(mon_inst, AfferentCoupling)
                            else tavg_s
                        )
                        if hasattr(mon_inst, "_config_stock"):
                            if (getattr(mon_inst, "variables_of_interest", None) is None
                                    or mon_inst.variables_of_interest.size == 0):
                                mon_inst.voi = np.arange(src_arr.shape[1], dtype=int)
                            else:
                                mon_inst.voi = np.asarray(
                                    mon_inst.variables_of_interest, dtype=int)
                            mon_inst._config_stock(*src_arr.shape[1:])
                        elif getattr(mon_inst, "voi", None) is None:
                            voi_a = getattr(mon_inst, "variables_of_interest", None)
                            mon_inst.voi = (
                                np.asarray(voi_a, dtype=int)
                                if voi_a is not None and np.asarray(voi_a).size > 0
                                else np.arange(src_arr.shape[1], dtype=int)
                            )
                        ts_l, ds_l = [], []
                        for step1 in range(1, nstep + 1):
                            if step1 % k_j != 0:
                                continue
                            sn_step = step1 // k_j
                            ty = mon_inst.sample(sn_step, src_arr[step1 - 1])
                            if ty is not None:
                                ts_l.append(ty[0])
                                ds_l.append(ty[1])
                        if ts_l:
                            res_list[ri] = (
                                np.asarray(ts_l, dtype=np.float64),
                                np.stack(ds_l).astype(np.float32),
                            )
                        else:
                            res_list[ri] = (
                                np.zeros(0, dtype=np.float64),
                                np.empty((0,) + src_arr.shape[1:], dtype=np.float32),
                            )
                subnet_results[sn.name] = res_list

        # Assemble global monitor outputs
        if monitors is not None:
            if len(monitors) == 0:
                outputs = []
            elif not py_global_indices:
                # All monitors served by C++ kmon engines!
                # Keep _apply_monitors call when all are legacy-shortcut types
                # (Raw/SubSample/Bold) so kernel-stream spy tests stay intact.
                all_shortcut = all(
                    isinstance(m, (Raw, SubSample, Bold))
                    and not isinstance(m, AfferentCoupling)
                    for m in monitors
                )
                if all_shortcut:
                    dummy_raw = [
                        (np.empty(0, dtype=np.float64),
                         np.empty((0,), dtype=np.float32),
                         np.empty((0,), dtype=np.float32))
                        for _ in self._analysis.subnets
                    ]
                    outputs = _apply_monitors(
                        dummy_raw, monitors, dt, chunk_size=chunk_size,
                        monitor_data=None,
                        subnet_infos=self._analysis.subnetworks,
                        bold_states=self._bold_states,
                        kernel_streams=kernel_streams or None,
                    )
                else:
                    outputs = [kernel_streams[mi] for mi in range(len(monitors))]
            else:
                bold_raw_outputs = outputs if has_py_bold else None
                temporal_raw_outputs = outputs if has_py_tavg else None
                if has_py_bold or has_py_tavg or need_py_sn_tavg:
                    outputs = _aggregate_raw_outputs(outputs, chunk_size)
                # Substitute Raw() placeholder for kernel-handled monitors not in
                # nb_hybrid's kernel_streams shortcut, then overwrite from kernel_streams.
                proxy_mons = [
                    (Raw() if (mi in kernel_streams and not (
                        isinstance(m, (Raw, SubSample, Bold))
                        and not isinstance(m, AfferentCoupling)
                    )) else m)
                    for mi, m in enumerate(monitors)
                ]
                outputs = _apply_monitors(
                    outputs, proxy_mons, dt, chunk_size=chunk_size,
                    monitor_data=None, subnet_infos=self._analysis.subnetworks,
                    bold_states=self._bold_states,
                    bold_raw_outputs=bold_raw_outputs,
                    temporal_raw_outputs=temporal_raw_outputs,
                    kernel_streams=kernel_streams or None,
                )
                for mi, stream in kernel_streams.items():
                    outputs[mi] = stream
        elif subnet_monitors is not None:
            outputs = []

        if not return_snapshot:
            if subnet_monitors is not None:
                return outputs, subnet_results
            return outputs

        snap = {
            "states": [
                _squeeze_lane(np.asarray(self._sim.get_subnet_state(i))).copy()
                for i in range(len(self._analysis.subnets))
            ],
            "buffers": {
                sn.name: np.asarray(
                    self._sim.get_subnet_buf(i), dtype=np.float32
                ).copy()
                for i, sn in enumerate(self._analysis.subnets)
            },
            "step": int(self._sim.t_abs()),
            "rng_states": self._backend._get_rng_states(
                self._analysis, self._network_set,
            ),
            "monitor_states": _copy_monitor_states(self._bold_states),
            "kmon_states": self._sim.get_monitor_states(),
            "kmon_global_states": self._sim.get_global_monitor_states(),
            "cpp_rng_states": self._sim.get_rng_states(),
        }
        if subnet_monitors is not None:
            return outputs, snap, subnet_results
        return outputs, snap

    def _balloon_windkessel_spec(self, slot, monitor, dt):
        if not self._backend.enable_kernel_monitors:
            raise NotImplementedError("BalloonWindkessel requires native monitor collection")
        if len(self._analysis.subnets) != 1:
            raise NotImplementedError("BalloonWindkessel requires one regional subnetwork")
        sn = self._analysis.subnets[0]
        if self._width != 1 or sn.k != 1 or sn.n_modes != 1 or sn.dt != dt:
            raise NotImplementedError("BalloonWindkessel requires width 1 and one source clock and mode")
        indices = self._network_set.subnets[0].node_indices
        if sn.n_nodes < 1 or (indices is not None and (
                indices.shape != (sn.n_nodes,)
                or not np.array_equal(np.sort(indices), np.arange(sn.n_nodes)))):
            raise ValueError("BalloonWindkessel requires a full regional node permutation when node_indices are supplied")
        voi = getattr(monitor, "variables_of_interest", None)
        if voi is None or voi.shape != (1,) or not 0 <= int(voi[0]) < len(sn.voi) // 3:
            raise ValueError("BalloonWindkessel requires exactly one valid observed VOI index")
        baseline = getattr(monitor, "drive_baseline", None)
        scale = getattr(monitor, "drive_scale", None)
        if baseline is None or scale is None or not np.isfinite(baseline) or not np.isfinite(scale):
            raise ValueError("BalloonWindkessel requires explicit finite drive_baseline and drive_scale")
        return dict(slot=slot, kind=10, dt=dt, istep=monitor._output_ticks(dt),
                    voi=[int(voi[0])], bw_drive_baseline=baseline, bw_drive_scale=scale,
                    bw_stream_params=dict(_BALLOON_WINDKESSEL_NOMINAL))

    def resume(
        self,
        snapshot: dict,
        nstep: int,
        chunk_size: int = None,
        return_snapshot: bool = False,
        monitors: Optional[list] = None,
        rng_mode: str = "numpy",
        subnet_monitors: Optional[dict] = None,
    ):
        """Resume from a snapshot returned by :meth:`run`."""
        if not monitors and any(
                "bw_stream_config" in state for state in snapshot.get("kmon_global_states") or ()):
            raise ValueError("BalloonWindkessel resume requires the compatible global monitors")
        states = snapshot["states"]
        self._backend._init_states(
            self._sim, self._analysis, self._network_set, states, self._width)
        buffers = snapshot.get("buffers") or {}
        for i, sn in enumerate(self._analysis.subnets):
            if sn.name in buffers:
                self._sim.set_subnet_buf(
                    i, np.ascontiguousarray(buffers[sn.name], dtype=np.float32))
        step_offset = int(snapshot.get("step", 0))
        self._sim.set_t_abs(step_offset)
        rng_states = snapshot.get("rng_states")
        if rng_states is not None:
            stoch_idx = 0
            for sn_info in self._analysis.subnets:
                if sn_info.is_stochastic:
                    sn_obj = next(
                        s for s in self._network_set.subnets
                        if s.name == sn_info.name
                    )
                    sn_obj.scheme.noise.random_stream.set_state(
                        rng_states[stoch_idx]
                    )
                    stoch_idx += 1
        if snapshot.get("monitor_states") is not None:
            self._bold_states = _copy_monitor_states(
                snapshot.get("monitor_states"))
        self._pending_kmon_states = snapshot.get("kmon_states")
        # Do not restore saved global streams unless global monitors were requested.
        self._pending_kmon_global_states = (
            snapshot.get("kmon_global_states") if monitors else None
        )
        self._pending_cpp_rng_states = snapshot.get("cpp_rng_states")
        try:
            return self.run(
                nstep,
                chunk_size,
                initial_states=None,
                monitors=monitors,
                return_snapshot=return_snapshot,
                _step_offset=step_offset,
                _resuming=True,
                rng_mode=rng_mode,
                subnet_monitors=subnet_monitors,
            )
        finally:
            self._pending_kmon_states = None
            self._pending_kmon_global_states = None
            self._pending_cpp_rng_states = None


class CppHybridBackend:
    """SIMD C++ backend for hybrid NetworkSet simulations."""

    _CACHE_DIR = None

    _RUN_FN_CACHE = {}  # topology-key -> cached _run_network_fn token

    def __init__(self):
        # Kernel-side monitor engines (Bold HRF convolution, Raw/SubSample
        # collection) are on by default; disable to force the shared Python
        # reference path (used by the parity tests).
        self.enable_kernel_monitors = True
        self._runtime_libraries = {}

    # ---- cache / misc API parity with NbHybridBackend ----

    @classmethod
    def get_cache_dir(cls):
        import os
        import tempfile
        from pathlib import Path
        if cls._CACHE_DIR is None:
            d = os.environ.get("TVB_NHYBRID_CACHE_DIR")
            cls._CACHE_DIR = (
                Path(d) if d
                else Path(tempfile.gettempdir()) / "tvb_cpp_hybrid_cache"
            )
            cls._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return cls._CACHE_DIR

    @classmethod
    def clear_cache(cls):
        import shutil
        if cls._CACHE_DIR is not None:
            try:
                shutil.rmtree(cls._CACHE_DIR)
            except OSError:
                pass
        cls._CACHE_DIR = None
        cls._RUN_FN_CACHE.clear()
        try:
            # Mirror legacy cache bookkeeping only when Numba is already loaded.
            nbh = sys.modules.get("tvb.simulator.backend.nb_hybrid")
            if nbh is not None:
                nbh._COMPILED_FN_CACHE.clear()
        except Exception:
            pass

    @classmethod
    def _stim_estimate_mb(cls, sn_info, nstep):
        n_cvar = len(sn_info.model.cvar)
        n_bytes = n_cvar * sn_info.n_nodes * sn_info.n_modes * nstep * 4
        return n_bytes / (1024 * 1024)

    def _resolve_named_params(self, network_set, params):
        return _resolve_named_params(network_set, params)

    # ------------------------------------------------------------------
    # compatibility / analysis
    # ------------------------------------------------------------------

    def _check_compatibility(self, network_set: NetworkSet) -> None:
        _allowed_integrators = (
            HeunDeterministic, EulerDeterministic,
            HeunStochastic, EulerStochastic,
        )
        from tvb.simulator.backend.hybrid_common import _validate_multi_dt
        dt0 = _validate_multi_dt(network_set)
        for sn in network_set.subnets:
            model_name = type(sn.model).__name__
            if not self._model_supported(model_name):
                self._model_key(sn.model)  # raises if unsupported
            mro_names = {c.__name__ for c in type(sn.model).__mro__}
            if "EpileptorRestingState" in mro_names and bool(
                np.any(np.asarray(getattr(sn.model, "modification", False)))
            ):
                raise NotImplementedError(
                    "CppHybridBackend: EpileptorRestingState with "
                    "modification=True is not supported. "
                    "Set model.modification = numpy.array([False])."
                )
            if "Hopfield" in mro_names and bool(
                np.any(np.asarray(getattr(sn.model, "dynamic", False)))
            ):
                raise NotImplementedError(
                    "CppHybridBackend: Hopfield with dynamic=True is not "
                    "supported. The hybrid codegen path only supports static "
                    "threshold (dynamic=False). "
                    "Set model.dynamic = numpy.array([False])."
                )
            if (sn.model.number_of_modes != 1
                    and type(sn.model).__name__ not in _MODEL_IDS
                    and getattr(sn.model, "dfun_mode", None) != "combined"):
                raise NotImplementedError(
                    "Multi-mode simulation is only supported for hand-written "
                    "or combined-mode (dfun_mode='combined') models."
                )
            if not isinstance(sn.scheme, _allowed_integrators):
                raise NotImplementedError(
                    f"CppHybridBackend only supports Heun/Euler "
                    f"(Deterministic or Stochastic); subnetwork "
                    f"'{sn.name}' uses {type(sn.scheme).__name__}"
                )
            if isinstance(sn.scheme, (HeunStochastic, EulerStochastic)):
                from tvb.simulator.noise import Additive
                if not isinstance(sn.scheme.noise, Additive):
                    raise NotImplementedError(
                        "CppHybridBackend only supports Additive noise."
                    )
            model_local_coupling = getattr(sn.scheme, "model_local_coupling", 0.0)
            if np.any(np.asarray(model_local_coupling) != 0.0):
                raise NotImplementedError(
                    "CppHybridBackend does not support nonzero local_coupling."
                )
        for p in self._all_projections(network_set):
            cfun_name = type(p.cfun).__name__ if p.cfun is not None else None
            if cfun_name is not None and cfun_name not in _CFUN_IDS:
                raise NotImplementedError(
                    f"CppHybridBackend does not support coupling {cfun_name}. "
                    f"Supported: {sorted(_CFUN_IDS)}"
                )
            n_src = np.size(p.source_cvar)
            n_tgt = np.size(p.target_cvar)
            if n_src < 1 or n_tgt < 1:
                raise ValueError(
                    "Projection must define non-empty source_cvar and target_cvar."
                )
            if n_src > 1 and n_tgt > 1 and n_src != n_tgt:
                cfun_type = _cfun_type(p)
                if cfun_type not in ("sigmoidal_jr", "pre_sigmoidal_dynamic"):
                    raise NotImplementedError(
                        f"CppHybridBackend does not support unmatched "
                        f"multi-to-multi cvar mapping ({n_src} -> {n_tgt})."
                    )

    @staticmethod
    def _all_projections(network_set: NetworkSet) -> list:
        out = []
        for sn in network_set.subnets:
            out.extend(sn.projections or [])
        out.extend(network_set.projections or [])
        return out

    def _analyse(self, network_set: NetworkSet) -> NetworkAnalysis:
        """Map a NetworkSet onto the C++ core's subnets/projections."""
        self._check_compatibility(network_set)
        from tvb.simulator.backend.hybrid_common import _validate_multi_dt
        subnet_index = {sn.name: i for i, sn in enumerate(network_set.subnets)}
        dt0 = _validate_multi_dt(network_set)

        subnets = []
        for sn in network_set.subnets:
            model_id, n_parm, n_cvar_m, n_svar_m, parm_names = \
                self._model_key(sn.model)
            params = self._params_for_model(
                sn.model, parm_names, n_nodes=int(sn.nnodes)
            )  # (n_nodes, n_parm)
            n_svar = int(sn.model.nvar) if n_svar_m is None else n_svar_m
            svar_names = tuple(str(n) for n in sn.model.state_variables)
            # voi entries: (kind, a, b) with kind 0 = state var a,
            # kind 1 = derived difference a - b (nb_hybrid parses simple
            # binary expressions like 'x2 - x1')
            voi = []
            for v in sn.model.variables_of_interest:
                if isinstance(v, (int, np.integer)):
                    voi.extend([0, int(v), 0])
                else:
                    expr = str(v).replace(" ", "")
                    if "-" in expr[1:]:
                        a, b = expr.split("-", 1)
                        voi.extend([1, svar_names.index(a), svar_names.index(b)])
                    else:
                        voi.extend([0, svar_names.index(expr), 0])
            is_stoch = isinstance(sn.scheme, (HeunStochastic, EulerStochastic))
            integ_id = 1 if isinstance(
                sn.scheme, (HeunDeterministic, HeunStochastic)) else 0
            sv_lo = sv_hi = None
            svb = getattr(sn.model, "state_variable_boundaries", None)
            if svb:
                sv_lo = np.full(n_svar, -np.inf, dtype=np.float32)
                sv_hi = np.full(n_svar, np.inf, dtype=np.float32)
                _f32_max = float(np.finfo(np.float32).max)
                for sv_name, bounds in svb.items():
                    idx = svar_names.index(sv_name)
                    if (bounds[0] is not None and np.isfinite(bounds[0])
                            and abs(float(bounds[0])) <= _f32_max):
                        sv_lo[idx] = np.float32(bounds[0])
                    if (bounds[1] is not None and np.isfinite(bounds[1])
                            and abs(float(bounds[1])) <= _f32_max):
                        sv_hi[idx] = np.float32(bounds[1])
            clamp_indices = clamp_values = None
            if getattr(sn.scheme, "clamped_state_variable_values", None) is not None:
                cidx = np.asarray(
                    sn.scheme.clamped_state_variable_indices, dtype=np.int32
                ).ravel()
                if cidx.size > 0:
                    clamp_indices = cidx
                    values = np.asarray(
                        sn.scheme.clamped_state_variable_values,
                        dtype=np.float32,
                    )
                    clamp_values = np.broadcast_to(
                        values,
                        (len(clamp_indices), sn.nnodes, sn.model.number_of_modes),
                    ).copy()
            model_name = type(sn.model).__name__
            nsig = None
            seed = 0
            ntau = 0.0
            if is_stoch:
                nsig = np.asarray(sn.scheme.noise.nsig, dtype=np.float64).ravel()
                if nsig.size == 1:
                    nsig = np.full(sn.model.nvar, float(nsig[0]))
                seed = int(getattr(sn.scheme.noise, "noise_seed", 0) or 0)
                ntau = float(getattr(sn.scheme.noise, "ntau", 0.0) or 0.0)
            subnets.append(SubnetworkInfo(
                stimuli=list(sn.stimuli or []),
                name=sn.name,
                n_nodes=int(sn.nnodes),
                n_svar=n_svar,
                n_parm=len(parm_names),
                n_modes=int(getattr(sn.model, "number_of_modes", 1)),
                n_cvar=(int(sn.model.cvar.size) if n_cvar_m is None
                        else n_cvar_m),
                model_id=model_id,
                runtime_library=self._runtime_libraries.get(model_name),
                model_name=model_name,
                model=sn.model,
                params=params,
                voi=voi,
                is_stochastic=is_stoch,
                noise_nsig=nsig,
                noise_seed=seed,
                noise_ntau=ntau,
                node_indices=(np.asarray(sn.node_indices, dtype=np.int64)
                             if getattr(sn, "node_indices", None) is not None
                             else None),
                dt=float(sn.scheme.dt),
                k=max(1, int(round(float(sn.scheme.dt) / dt0))),
                parm_index={k: i for i, k in enumerate(parm_names)},
                integ_id=integ_id,
                sv_lo=sv_lo,
                sv_hi=sv_hi,
                clamp_indices=clamp_indices,
                clamp_values=clamp_values,
            ))

        projections = []
        tgt_models = {sn.name: sn.model for sn in network_set.subnets}
        entries = []
        for sn in network_set.subnets:
            for p in (sn.projections or []):
                entries.append((p, sn.name, sn.name, False))
        for p in (network_set.projections or []):
            entries.append((p, p.source.name, p.target.name, True))
        for p, src_name, tgt_name, is_inter in entries:
            weights_csr = p.weights.copy().astype(np.float32)
            nz_mask = weights_csr.data != 0
            weights_csr.eliminate_zeros()
            idelays = np.atleast_1d(p.idelays).astype(np.int32)
            idelays = idelays[nz_mask] if idelays.size else idelays
            horizon = int(p._horizon)
            # source_cvar entries index state variables directly
            src_cvars_list = [int(v) for v in np.atleast_1d(p.source_cvar)]
            target_cvars_list = [int(v) for v in np.atleast_1d(p.target_cvar)]
            cfun_name = type(p.cfun).__name__ if p.cfun is not None else None
            cfun_use_classic = False
            if cfun_name is None:
                cfun_id, cfun_params, n_cfun_parm = (
                    _PASSTHROUGH_CFUN, np.zeros(0, np.float32), 0)
            else:
                cfun_id = _CFUN_TYPE_IDS[_cfun_type(p)]
                cfun_params = self._cfun_params(
                    p.cfun, n_src_cvar=len(src_cvars_list)
                )
                if cfun_id == 8 and getattr(p.cfun, "use_classic", 1):
                    cfun_params = cfun_params.copy()
                    cfun_params[0] *= cfun_params[0]
                    cfun_use_classic = True
                n_cfun_parm = len(_CPP_CFUN_PARAM_ATTRS[cfun_name])
            if is_inter:
                pr_name = (getattr(p, "name", None)
                           or f"{src_name}_to_{tgt_name}")
                lookup = pr_name
            else:
                pr_name = getattr(p, "name", None) or "intra"
                lookup = (f"{tgt_name}.{pr_name}"
                          if pr_name == "intra" else pr_name)
            if is_inter:
                _mm = getattr(p, "mode_map", None)
                mode_map = (
                    np.asarray(_mm, dtype=np.float32)
                    if _mm is not None else
                    np.ones((int(p.source.model.number_of_modes),
                             int(p.target.model.number_of_modes)),
                            dtype=np.float32)
                )
            else:
                mode_map = None

            tgt_model = p.target.model if is_inter else tgt_models[tgt_name]
            tgt_model_cvar = np.asarray(tgt_model.cvar, dtype=np.int32)
            tsc = self._target_scales_raw(p)
            is_collapsing = cfun_id in (6, 9)  # sigmoidal_jr, pre_sigmoidal_dynamic
            w_data = weights_csr.data.astype(np.float32)
            w_idx = weights_csr.indices.astype(np.uint32)
            w_ptr = weights_csr.indptr.astype(np.uint32)
            del_arr = idelays.astype(np.uint32)

            if len(target_cvars_list) == 1:
                tgt = target_cvars_list[0]
                tgt_state_cvar = int(tgt_model_cvar[tgt])
                ts = (float(tsc[0]) if tsc is not None and tsc.size > 0
                      else 1.0)
                if is_collapsing or len(src_cvars_list) == 1:
                    channel_pairs = [(list(src_cvars_list), tgt, ts, tgt_state_cvar)]
                else:
                    channel_pairs = [
                        ([sc], tgt, ts, tgt_state_cvar)
                        for sc in src_cvars_list
                    ]
            else:
                channel_pairs = []
                for ch, tgt in enumerate(target_cvars_list):
                    if is_collapsing or len(src_cvars_list) == 1:
                        srcs = list(src_cvars_list)
                    else:
                        srcs = [src_cvars_list[ch]]
                    ts = (float(tsc[ch]) if tsc is not None and ch < tsc.size
                          else (float(tsc[0]) if tsc is not None and tsc.size == 1
                                else 1.0))
                    tgt_state_cvar = int(tgt_model_cvar[tgt])
                    channel_pairs.append((srcs, tgt, ts, tgt_state_cvar))

            for srcs, tgt, ts, tgt_state_cvar in channel_pairs:
                projections.append(ProjectionInfo(
                    ts=ts,
                    target_scales=tsc,
                    target_cvars_list=[tgt],
                    tgt_state_cvar=tgt_state_cvar,
                    mode_map=mode_map,
                    name=pr_name,
                    lookup=lookup,
                    is_inter=is_inter,
                    src_sn=subnet_index[src_name],
                    tgt_sn=subnet_index[tgt_name],
                    w=w_data,
                    idx=w_idx,
                    ptr=w_ptr,
                    idelays=del_arr,
                    horizon=horizon,
                    cfun_id=cfun_id,
                    src_cvar=srcs[0],
                    src_cvars_list=srcs,
                    n_src_cvars=len(srcs),
                    tgt_cvar=tgt,
                    n_cfun_parm=n_cfun_parm,
                    scale=float(p.scale),
                    cfun_params=cfun_params,
                    cfun_use_classic=cfun_use_classic,
                ))

        # per-source-subnet horizon = max outgoing projection horizon
        for pr in projections:
            src = subnets[pr.src_sn]
            src.horizon = max(src.horizon, pr.horizon)

        return NetworkAnalysis(
            subnets=subnets, projections=projections, dt=dt0)

    @classmethod
    def _ensure_generic_dfuns(cls):
        """Generate + compile + load the runtime dfun library.

        Called only for a model that is neither hand-written (``_MODEL_IDS``)
        nor in the built-in table, so stock models never trigger a g++ compile.

        The library is emitted at ``start_id = max(built-in id) + 1``, i.e. in
        an id range strictly above every built-in id, and it *excludes* the
        built-in table's own names: stock models already ship as kernels
        inside the extension, so re-emitting and re-compiling them into the
        runtime library would only add dead kernels (and g++ time) to the
        cache.  Ids are positional over sorted class names, so without the
        disjoint range a user model sorting before the stock ones
        (``AaaProbe``) would take id 100, shift every stock id, and —
        because _core.cpp serves the built-in range first — silently run the
        wrong kernel.  Disjoint ranges make that structurally impossible:
        the built-in table owns 100..max_builtin, the runtime library owns
        max_builtin+1.., and dfungen gap-pads the runtime library's dispatch
        table with ``nullptr`` over the built-in range so those ids fall
        through instead of calling a null slot.  With no built-in table at
        all the original 100-based range is used unchanged (and nothing is
        excluded).

        Cache entries are content-addressed in a process-private directory.
        Each subnet retains its own entry; resolving another library never
        changes existing simulations or the extension's low-level default.
        """
        global _GEN_IDS, _GEN_META
        import ctypes
        from . import dfungen
        builtin = _builtin_model_table()
        builtin_ids = [int(m["mid"]) for m in builtin.values()]
        start_id = (_GENERIC_BASE if not builtin_ids
                    else max(_GENERIC_BASE, max(builtin_ids)) + 1)
        lib_path, ids, meta = dfungen.generate_lib(
            cls.get_cache_dir(), start_id=start_id, exclude=list(builtin))
        key = str(lib_path)
        if key not in _GEN_LIB:
            lib = ctypes.CDLL(key)
            fn = ctypes.cast(lib.cph_generic_dfun, ctypes.c_void_p).value
            # Never evict a loaded handle: native subnets retain raw pointers.
            _GEN_LIB[key] = _GeneratedLibrary(lib, dict(ids), dict(meta), fn)
        entry = _GEN_LIB[key]
        # Informational aliases for the most recently resolved library only.
        # Dispatch and parameter packing use the owning entry, not these aliases.
        _GEN_IDS, _GEN_META = entry.ids, entry.metadata
        return entry

    @staticmethod
    def _model_supported(name: str) -> bool:
        """Whether *name* has a kernel: hand-written, built-in generated, or in
        a runtime library compiled during this process."""
        return (name in _MODEL_IDS or name in _builtin_model_table()
                or any(name in entry.ids for entry in _GEN_LIB.values()))

    @staticmethod
    def _packing_parm_names(model, meta) -> list:
        """Parameter names to pack, taken from the table the kernel came from.

        ``parr`` is indexed positionally by the dfun, so the packing order must
        be the table's own ``parm_names`` — the order that kernel was generated
        with — not an independently derived one.  The instance's global+spatial
        names are checked against it: a different count means the kernel would
        read past the end of the packed array, a different set means values
        would land in the wrong slots.  Either is a wrong-dynamics bug, so it
        fails loudly here, naming the model.
        """
        table = [str(p) for p in meta["parm_names"]]
        own = (list(model.global_parameter_names)
               + list(model.spatial_parameter_names))
        if int(meta["n_parm"]) != len(table) or len(own) != len(table):
            raise ValueError(
                f"{type(model).__name__}: its dfun declares {meta['n_parm']} "
                f"parameter(s) {table} but this instance has {len(own)} {own}; "
                f"parameter packing would not match the kernel")
        if set(table) != set(own):
            raise ValueError(
                f"{type(model).__name__}: its dfun packs {table} but this "
                f"instance declares {own}; the model definition has changed "
                f"since the kernel was generated (rebuild the extension)")
        return table

    @staticmethod
    def _check_dfun_signature(model, meta):
        """Raise when a model's dfun fingerprint differs from its kernel's.

        The kernel a class name resolves to may have been generated from a
        *different* class that happens to share the name (a shadowing class
        with identical parameters but altered equations slips past the
        parameter-packing check, which only sees shapes).  The fingerprint is
        computed the same way at emission time and here
        (``dfungen.model_dfun_signature``), so any divergence between the
        equations a kernel was generated from and the instance being resolved
        fails loudly, naming the model.  A kernel emitted without a
        fingerprint (an extension that predates the feature) skips the check.
        """
        from . import dfungen
        expected = str(meta.get("signature") or "")
        if not expected:
            return
        name = type(model).__name__
        got = dfungen.model_dfun_signature(model)
        if got == expected:
            return
        # Generic kernels are emitted from a default instance where scalar
        # traits live in global_parameter_names; when a runtime instance sets
        # per-node arrays on any parameter, Model.spatial_parameter_names
        # shifts that name out of global_parameter_names without changing the
        # equations.  Verify the equations still match under the kernel's split.
        if list(model.spatial_parameter_names):
            table = [str(p) for p in meta.get("parm_names", ())]
            own = (list(model.global_parameter_names)
                   + list(model.spatial_parameter_names))
            if set(table) == set(own) and len(table) == len(own):
                class _SigProxy:
                    def __init__(self, m, g_names, s_names):
                        self._m = m
                        self.global_parameter_names = g_names
                        self.spatial_parameter_names = s_names

                    def __getattr__(self, attr):
                        return getattr(self._m, attr)

                for split in range(len(table), -1, -1):
                    proxy = _SigProxy(model, table[:split], table[split:])
                    if dfungen.model_dfun_signature(proxy) == expected:
                        return
        raise ValueError(
            f"{name}: a dfun kernel for this class name exists but was "
            f"generated from different equations (dfun fingerprint "
            f"{got} != {expected}); rename the class or rebuild the "
            f"extension")

    def _model_key(self, model):
        """Return (model_id, n_parm, n_cvar, n_svar, parm_names) for a model
        instance.

        Resolution order mirrors _core.cpp's dispatch: hand-written kernels
        first (parity was established against them), then the table compiled
        into the extension, then the runtime-compiled library for models
        neither covers.
        """
        name = type(model).__name__
        if name in _MODEL_IDS:
            parm = _MODEL_PARM_NAMES[name]
            return _MODEL_IDS[name], len(parm), None, None, parm
        builtin = _builtin_model_table()
        if name in builtin:
            meta = builtin[name]
            parm = self._packing_parm_names(model, meta)
            self._check_dfun_signature(model, meta)
            return (int(meta["mid"]), len(parm), int(meta["n_cvar"]),
                    int(meta["n_svar"]), parm)
        entry = self._ensure_generic_dfuns()
        ids, metadata = entry.ids, entry.metadata
        if name not in ids:
            raise NotImplementedError(
                f"CppHybridBackend does not support {name}"
            )
        meta = metadata[name]
        parm = self._packing_parm_names(model, meta)
        self._check_dfun_signature(model, meta)
        self._runtime_libraries[name] = entry
        return (ids[name], len(parm), int(meta["n_cvar"]),
                int(meta["n_svar"]), parm)

    @staticmethod
    def _cfun_params(cfun, n_src_cvar: int = 1) -> np.ndarray:
        """(n_parm,) float32 cfun parameter values, ordered like nb_hybrid."""
        attrs = _CPP_CFUN_PARAM_ATTRS[type(cfun).__name__]
        params = np.zeros(max(idx for _, idx in attrs) + 1, dtype=np.float32)
        cfun_name = type(cfun).__name__
        for name, idx in attrs:
            v = getattr(cfun, name, None)
            if v is None:
                if cfun_name == "Kuramoto" and name == "inv_N":
                    params[idx] = (1.0 / n_src_cvar) if n_src_cvar > 0 else 1.0
                else:
                    params[idx] = 1.0
            else:
                params[idx] = float(np.asarray(v).ravel()[0])
        return params

    @staticmethod
    def _params_for_model(model, names, n_nodes: int = 1) -> np.ndarray:
        """(n_nodes, n_parm) float32 — preserves per-node spatial parameters.

        Handles ``'<attr>:<idx>'`` array-element expansion (Zerlaut
        ``P_e:/P_i:``, CerebellarMF ``P_grc:/P_goc:/P_mli:/P_pc:``),
        broadcasts scalar/length-1 attributes across *n_nodes*, and distributes
        length-*n_nodes* arrays per node.
        """
        n_nodes = int(n_nodes)
        n_parm = len(names)
        out = np.empty((n_nodes, n_parm), dtype=np.float32)
        model_name = type(model).__name__
        for col_i, k in enumerate(names):
            if ":" in k:
                base, idx_s = k.split(":", 1)
                idx = int(idx_s)
                raw = np.asarray(getattr(model, base), dtype=np.float32)
                if raw.ndim == 1:
                    if idx >= raw.size:
                        raise ValueError(
                            f"{model_name}.{base}: index {idx} out of range "
                            f"for shape {raw.shape}"
                        )
                    out[:, col_i] = float(raw[idx])
                elif raw.ndim == 2 and raw.shape[0] == n_nodes and raw.shape[1] > idx:
                    out[:, col_i] = raw[:, idx]
                elif raw.ndim == 2 and raw.shape[0] == 1 and raw.shape[1] > idx:
                    out[:, col_i] = float(raw[0, idx])
                else:
                    raise ValueError(
                        f"{model_name}.{base}: expected 1D coefficient array "
                        f"or (n_nodes={n_nodes}, n_coeff) with n_coeff > {idx}, "
                        f"got shape {raw.shape}"
                    )
            else:
                raw = np.asarray(getattr(model, k), dtype=np.float32)
                flat = raw.ravel()
                if flat.size == 1:
                    out[:, col_i] = float(flat[0])
                elif flat.size == n_nodes and (
                    raw.ndim <= 1 or raw.squeeze().ndim <= 1
                ):
                    out[:, col_i] = flat
                else:
                    raise ValueError(
                        f"{model_name}.{k}: expected length 1 or {n_nodes} "
                        f"(n_nodes), got shape {raw.shape} (size {flat.size})"
                    )
        return out

    @staticmethod
    def _target_scales(p) -> float:
        ts = getattr(p, "target_scales", None)
        if ts is None or np.size(ts) == 0:
            return 1.0
        return float(np.asarray(ts).ravel()[0])

    @staticmethod
    def _target_scales_raw(p):
        ts = getattr(p, "target_scales", None)
        if ts is None:
            return None
        a = np.atleast_1d(np.asarray(ts, dtype=np.float32))
        return a if a.size else None

    # ------------------------------------------------------------------
    # build / run
    # ------------------------------------------------------------------

    def compile(
        self,
        network_set: NetworkSet,
        print_source: bool = False,
        debug_nojit: bool = False,
        eager: bool = True,
        width: int = 1,
    ) -> CompiledCppSim:
        """Build the C++ simulation for *network_set*.

        *width* is the SIMD lane count (batch members packed per lane):
        use 1 for single simulations, 8 for SIMD batch sweeps.
        """
        analysis = self._analyse(network_set)
        key = self._topology_key(analysis)
        token = self._RUN_FN_CACHE.get(key)
        if token is None:
            token = object()
            self._RUN_FN_CACHE[key] = token
            try:
                # Do not load a second compiler backend for cache bookkeeping.
                nbh = sys.modules.get("tvb.simulator.backend.nb_hybrid")
                if nbh is not None:
                    nbh._COMPILED_FN_CACHE[key] = token
            except Exception:
                pass
            try:
                d = self.get_cache_dir()
                (d / f"nbhybrid_{key[:12]}.py").write_text(
                    "# cpp_hybrid compiled unit cache artifact\n")
            except Exception:
                pass
        sim = Sim(width)
        self._build_sim(sim, analysis, network_set, width)
        cc = CompiledCppSim(self, analysis, network_set, sim, width)
        cc._run_network_fn = token
        return cc

    def _topology_key(self, analysis) -> str:
        import hashlib
        parts = []
        for sn in analysis.subnets:
            parts.append((
                sn.model_name, sn.n_nodes, sn.n_modes,
                sn.n_svar, sn.n_cvar, sn.horizon, sn.dt, sn.k,
                sn.integ_id,
                tuple(sn.sv_lo.tolist()) if sn.sv_lo is not None else None,
                tuple(sn.sv_hi.tolist()) if sn.sv_hi is not None else None,
                (tuple(sn.clamp_indices.tolist())
                 if sn.clamp_indices is not None else None),
            ))
        for pr in analysis.all_projections:
            parts.append((
                pr.src_sn, pr.tgt_sn, pr.cfun_id,
                pr.n_tgt_nodes, tuple(pr.src_cvars_list),
                pr.tgt_cvar, pr.tgt_state_cvar, pr.ts,
            ))
        # cache key only, not security-sensitive (Sonar)
        return hashlib.md5(repr(parts).encode(),
                           usedforsecurity=False).hexdigest()

    def _build_sim(self, sim, analysis, network_set, width):
        """Populate an empty Sim from an analysis (no per-lane params)."""
        for i, sn in enumerate(analysis.subnets):
            sim.add_subnet(
                sn.n_nodes, sn.n_svar, sn.n_parm, sn.n_cvar,
                sn.model_id, sn.horizon, sn.n_modes,
                k=sn.k, dt=float(sn.dt), integ_id=sn.integ_id,
                generic_dfun=(sn.runtime_library.function_pointer
                              if sn.runtime_library is not None else 0),
            )
            sim.set_voi_specs(i, sn.voi)
            if sn.sv_lo is not None:
                sim.set_subnet_bounds(
                    i,
                    np.ascontiguousarray(sn.sv_lo, dtype=np.float32),
                    np.ascontiguousarray(sn.sv_hi, dtype=np.float32),
                )
            if sn.clamp_indices is not None:
                sim.set_subnet_clamps(
                    i,
                    np.ascontiguousarray(sn.clamp_indices, dtype=np.int32),
                    np.ascontiguousarray(sn.clamp_values, dtype=np.float32),
                )
        integ_id = 1 if isinstance(
            network_set.subnets[0].scheme,
            (HeunDeterministic, HeunStochastic)) else 0
        sim.set_opts(integ_id, analysis.dt, 1)
        for i, pr in enumerate(analysis.projections):
            srcs = pr.src_cvars_list if pr.src_cvars_list else [pr.src_cvar]
            sim.add_projection(
                pr.src_sn, pr.tgt_sn, pr.w, pr.idx, pr.ptr,
                np.asarray(pr.idelays, dtype=np.uint32), pr.cfun_id,
                np.asarray(srcs, dtype=np.int32),
                pr.tgt_cvar, pr.n_cfun_parm, np.float32(pr.scale),
                np.float32(getattr(pr, "ts", 1.0)),
                int(getattr(pr, "tgt_state_cvar", 0)),
                pr.mode_map,
            )
            cp = np.broadcast_to(
                pr.cfun_params[:, None], (max(pr.n_cfun_parm, 0), width)
            ).copy() if pr.n_cfun_parm else np.zeros((0, width), np.float32)
            sim.set_cfun_params(i, cp)
        for i, sn in enumerate(analysis.subnets):
            pp = np.broadcast_to(
                sn.params[:, :, None],
                (sn.n_nodes, sn.n_parm, width),
            ).copy()
            sim.set_subnet_params(i, pp)

    def _init_states(self, sim, analysis, network_set, initial_states, width):
        for i, sn in enumerate(analysis.subnets):
            nm = sn.n_modes
            if initial_states is not None:
                st = np.asarray(initial_states[i], dtype=np.float32)
                st = np.reshape(st, (sn.n_svar, sn.n_nodes, nm))
            else:
                st = np.zeros((sn.n_svar, sn.n_nodes, nm), np.float32)
            # broadcast the (n_svar, n_nodes, n_modes) state across lanes
            x0 = np.broadcast_to(
                st[:, :, :, np.newaxis],
                (sn.n_svar, sn.n_nodes, nm, width),
            ).copy()
            # core expects (n_svar, n_node, n_modes, W)
            sim.set_subnet_state(i, np.ascontiguousarray(x0))

    def _build_stim_arrays(self, analysis, nstep, width, step_offset=0):
        """Per-subnetwork stimulus arrays, layout (nstep, n_cvar, n_nodes, W).

        Stimulus for step t (0-based within the run) is evaluated at global
        step step_offset + t + 1, matching the nb_hybrid convention
        (stim[t - offset - 1]).
        Multi-dt decision 7: the pattern is evaluated on the master dt0 grid
        (index = master step), via nb_hybrid._stim_master_grid.
        """
        from tvb.simulator.backend.hybrid_common import _stim_master_grid
        stims = []
        for sn in analysis.subnets:
            if not sn.stimuli:
                continue
            nm = sn.n_modes
            arr = np.zeros(
                (nstep, sn.n_cvar, sn.n_nodes, nm, width), np.float32)
            for stim in (_stim_master_grid(
                    s, analysis.dt, step_offset + int(nstep))
                    for s in sn.stimuli):
                target_slots = np.asarray(stim.target_cvar).astype(np.intp)
                for step_idx in range(step_offset + 1,
                                      step_offset + int(nstep) + 1):
                    sc = np.asarray(stim.get_coupling(step_idx), dtype=np.float32)
                    if sc.ndim == 2:
                        sc = sc[:, :, np.newaxis]
                    sc = np.broadcast_to(
                        sc, (target_slots.size, sn.n_nodes, nm))
                    for li, slot in enumerate(target_slots):
                        arr[step_idx - step_offset - 1, slot, :, :, 0] += sc[li]
            if width > 1:
                arr = np.repeat(arr, width, axis=4)
            stims.append(np.ascontiguousarray(arr))
        return stims

    def _validate_monitors(self, monitors, chunk_size):
        from tvb.simulator.monitors import Raw, SubSample, AfferentCoupling
        cs = chunk_size or 1
        for m in (monitors or []):
            if isinstance(m, SubSample) and cs != 1:
                raise ValueError(
                    "SubSample monitor requires chunk_size=1 "
                    f"(got chunk_size={chunk_size})")
            if (isinstance(m, Raw) and not isinstance(m, AfferentCoupling)
                    and cs != 1):
                raise ValueError(
                    "Raw monitor requires chunk_size=1; "
                    "pass chunk_size=1 to run_network()")

    @staticmethod
    def _get_rng_states(analysis, network_set):
        states = []
        for sn_info in analysis.subnets:
            if sn_info.is_stochastic:
                sn_obj = next(
                    s for s in network_set.subnets
                    if s.name == sn_info.name)
                states.append(sn_obj.scheme.noise.random_stream.get_state())
        return states

    def _validate_horizons(self, analysis):
        for pr in analysis.projections:
            if pr.idelays.size:
                need = int(np.max(pr.idelays)) + 1
                if pr.horizon < need:
                    raise ValueError(
                        f"Projection '{pr.name}' horizon {pr.horizon} < "
                        f"max(idelay)+1 = {need}; delays would alias."
                    )

    @staticmethod
    def _kernel_collectable(m):
        """True when *m* is handled by the C++ ``kmon<W>`` monitor engines."""
        from tvb.simulator.monitors import (
            Raw,
            SubSample,
            Bold,
            BoldRegionROI,
            TemporalAverage,
            AfferentCoupling,
            GlobalAverage,
            SpatialAverage,
            Projection,
        )
        if isinstance(m, BoldRegionROI):
            return False
        if isinstance(m, Projection):
            return (getattr(m, "_gain", None) is not None
                    or getattr(m, "projection", None) is not None)
        return isinstance(
            m,
            (
                Bold,
                SubSample,
                TemporalAverage,
                AfferentCoupling,
                GlobalAverage,
                SpatialAverage,
                Raw,
            ),
        )

    @staticmethod
    def _monitor_voi_list(m):
        voi = getattr(m, "variables_of_interest", None)
        if voi is None:
            return []
        arr = np.asarray(voi, dtype=np.int32).ravel()
        return arr.tolist() if arr.size > 0 else []

    @staticmethod
    def _complete_window_istep(m, dt):
        ticks = float(m.period) / dt
        if np.isfinite(ticks):
            nearest = round(ticks)
            # Allow a few binary64 rounding steps from period construction and
            # division, not a relative fraction of a source tick.
            if (1 <= nearest <= np.iinfo(np.int32).max
                    and abs(ticks - nearest) <= 8 * np.spacing(float(nearest))):
                return nearest
        raise ValueError(
            f"C++ regional {type(m).__name__} period {m.period!r} must be "
            f"commensurate with source dt {dt!r} (a positive integer number of ticks)"
        )

    def _monitor_attach_spec(
        self,
        slot,
        m,
        n_src_v,
        n_space,
        dt,
        cadence_K=1,
        explicit_chunk_size=None,
        nstep=0,
        is_resuming=False,
        return_snapshot=False,
        complete_window_istep=None,
    ):
        """Build a C++ ``kmon<W>`` spec dict for monitor *m*."""
        import copy
        from tvb.datatypes import equations
        from tvb.simulator.monitors import (
            AfferentCoupling,
            AfferentCouplingTemporalAverage,
            Bold,
            GlobalAverage,
            Projection,
            Raw,
            RawVoi,
            SpatialAverage,
            SubSample,
            TemporalAverage,
        )

        eff_dt = float(dt)
        ck = int(max(1, cadence_K))
        if complete_window_istep is not None:
            spec = {
                "slot": int(slot),
                "istep": complete_window_istep,
                "complete_window": True,
                "dt": eff_dt,
                "voi": self._monitor_voi_list(m),
            }
            if isinstance(m, SpatialAverage):
                weights = getattr(m, "spatial_mean", None)
                if weights is not None:
                    with np.errstate(over="ignore"):
                        weights = np.asarray(weights, dtype=np.float32, order="C")
                if (weights is None or weights.ndim != 2 or not weights.shape[0]
                        or weights.shape[1] != n_space or not np.isfinite(weights).all()):
                    raise ValueError(
                        "C++ regional SpatialAverage complete windows require finite "
                        "preconfigured spatial_mean weights with one column per source node"
                    )
                spec.update(kind=7, spatial_mean=weights)
            elif isinstance(m, GlobalAverage):
                spec["kind"] = 6
            else:  # TemporalAverage (not AfferentCouplingTemporalAverage)
                spec["kind"] = 4
            return spec
        if isinstance(m, AfferentCouplingTemporalAverage):
            istep = max(1, int(round(float(m.period) / eff_dt)))
            return {
                "slot": int(slot),
                "kind": 5,  # KMON_CTAVG
                "istep": istep,
                "cadence_K": ck,
                "dt": eff_dt,
                "midpoint_time": True,
                "voi": self._monitor_voi_list(m),
            }
        if isinstance(m, AfferentCoupling):
            return {
                "slot": int(slot),
                "kind": 5,  # KMON_CTAVG
                "istep": 1,
                "cadence_K": ck,
                "dt": eff_dt,
                "midpoint_time": False,
                "voi": self._monitor_voi_list(m),
            }
        if isinstance(m, TemporalAverage):
            istep = max(1, int(round(float(m.period) / eff_dt)))
            return {
                "slot": int(slot),
                "kind": 4,  # KMON_TAVG
                "istep": istep,
                "cadence_K": ck,
                "dt": eff_dt,
                "voi": self._monitor_voi_list(m),
            }
        if isinstance(m, Bold):
            voi = self._monitor_voi_list(m)
            tmp = copy.deepcopy(m)
            tmp._config_dt(eff_dt)
            tmp.compute_hrf()
            first_order = isinstance(
                m.hrf_kernel, equations.FirstOrderVolterra)
            k1v0 = (float(m.hrf_kernel.parameters["k_1"]
                          * m.hrf_kernel.parameters["V_0"])
                    if first_order else 1.0)
            hrf = np.asarray(tmp.hemodynamic_response_function,
                             dtype=np.float64).ravel()
            return {
                "slot": int(slot),
                "kind": 3,  # KMON_BOLD
                "istep": max(1, int(tmp.istep)),
                "cadence_K": ck,
                "dt": eff_dt,
                "interim_istep": max(1, int(tmp._interim_istep)),
                "stock_steps": max(1, int(tmp._stock_steps)),
                "k1v0": k1v0,
                "first_order": bool(first_order),
                "voi": voi,
                "hrf": hrf,
            }
        if isinstance(m, SubSample):
            istep = max(1, int(round(float(m.period) / eff_dt)))
            return {
                "slot": int(slot),
                "kind": 2,  # KMON_SUBSAMPLE
                "istep": istep,
                "cadence_K": ck,
                "dt": eff_dt,
                "voi": self._monitor_voi_list(m),
            }
        if isinstance(m, GlobalAverage):
            if explicit_chunk_size is not None and ck == 1:
                istep = max(1, int(explicit_chunk_size))
            else:
                istep = max(1, int(round(float(m.period) / eff_dt)))
            return {
                "slot": int(slot),
                "kind": 6,  # KMON_GLOBAL_AVG
                "istep": istep,
                "cadence_K": ck,
                "dt": eff_dt,
                "voi": self._monitor_voi_list(m),
            }
        if isinstance(m, SpatialAverage):
            istep = (max(1, int(round(float(m.period) / eff_dt)))
                     if getattr(m, "period", None) else 1)
            if explicit_chunk_size is not None and ck == 1:
                cs_i = max(1, int(explicit_chunk_size))
                if cs_i >= istep or (
                    nstep > 0 and nstep < istep
                    and not is_resuming and not return_snapshot
                ):
                    istep = cs_i
            sm = getattr(m, "spatial_mean", None)
            if sm is None:
                sm_arr = np.eye(max(1, int(n_space)), dtype=np.float32)
            else:
                sm_arr = np.ascontiguousarray(sm, dtype=np.float32)
            return {
                "slot": int(slot),
                "kind": 7,  # KMON_SPATIAL_AVG
                "istep": istep,
                "cadence_K": ck,
                "dt": eff_dt,
                "voi": self._monitor_voi_list(m),
                "spatial_mean": sm_arr,
            }
        if isinstance(m, Projection):
            istep = (max(1, int(round(float(m.period) / eff_dt)))
                     if getattr(m, "period", None) else 1)
            if explicit_chunk_size is not None and ck == 1:
                cs_i = max(1, int(explicit_chunk_size))
                if cs_i >= istep or (
                    nstep > 0 and nstep < istep
                    and not is_resuming and not return_snapshot
                ):
                    istep = cs_i
            gain_arr = np.ascontiguousarray(m.gain, dtype=np.float32)
            ref_vec = None
            rv = getattr(m, "_ref_vec", None)
            if rv is not None:
                rv_mask = getattr(m, "_ref_vec_mask", None)
                if rv_mask is not None:
                    ref_vec = np.zeros(gain_arr.shape[0], dtype=np.float32)
                    ref_vec[np.asarray(rv_mask, dtype=bool)] = np.asarray(
                        rv, dtype=np.float32)
                else:
                    ref_vec = np.ascontiguousarray(
                        rv, dtype=np.float32).ravel()
            return {
                "slot": int(slot),
                "kind": 8,  # KMON_PROJECTION
                "istep": istep,
                "cadence_K": ck,
                "dt": eff_dt,
                "voi": self._monitor_voi_list(m),
                "gain": gain_arr,
                "ref_vec": ref_vec,
            }
        if isinstance(m, RawVoi):
            return {
                "slot": int(slot),
                "kind": 1,  # KMON_RAW
                "istep": 1,
                "cadence_K": ck,
                "dt": eff_dt,
                "voi": self._monitor_voi_list(m),
            }
        # Raw: collect every observed VOI row
        return {
            "slot": int(slot),
            "kind": 1,  # KMON_RAW
            "istep": 1,
            "cadence_K": ck,
            "dt": eff_dt,
            "voi": [],
        }

    @staticmethod
    def _lower_subnet_stimuli(sn_info, dt0, total_steps):
        """Lower a subnet's Stim list to ``(cph_stim_specs, legacy_stims)``."""
        from tvb.datatypes import equations as _eqs
        from tvb.datatypes.patterns import SpatioTemporalPattern

        eq_map = (
            (_eqs.Linear, 1, ("a", "b")),
            (_eqs.Sinusoid, 2, ("amp", "frequency")),
            (_eqs.Cosine, 3, ("amp", "frequency")),
            (_eqs.PulseTrain, 4, ("T", "tau", "amp", "onset")),
            (_eqs.Alpha, 5, ("onset", "alpha", "beta")),
            (_eqs.GeneralizedSigmoid, 6, ("low", "high", "midpoint", "sigma")),
            (_eqs.Gaussian, 7, ("amp", "sigma", "midpoint", "offset")),
            (_eqs.DoubleGaussian, 8, (
                "amp_1", "sigma_1", "midpoint_1",
                "amp_2", "sigma_2", "midpoint_2")),
            (_eqs.Sigmoid, 9, ("amp", "radius", "sigma", "offset")),
        )

        specs = []
        legacy = []
        for s in (sn_info.stimuli or ()):
            target_slots = np.asarray(s.target_cvar)
            if target_slots.ndim != 1 or target_slots.size == 0:
                raise ValueError(
                    f"Stimulus for subnetwork '{sn_info.name}' must have a "
                    "non-empty one-dimensional target_cvar array; "
                    f"got shape {target_slots.shape}."
                )
            if target_slots.dtype.kind not in "iu":
                raise ValueError(
                    f"Stimulus for subnetwork '{sn_info.name}' has "
                    f"non-integer target_cvar {target_slots.tolist()}."
                )
            target_slots = target_slots.astype(np.int32, copy=False)
            if np.any(target_slots < 0) or np.any(target_slots >= sn_info.n_cvar):
                raise ValueError(
                    f"Stimulus for subnetwork '{sn_info.name}' has target "
                    f"slots {target_slots.tolist()} outside "
                    f"[0, {sn_info.n_cvar - 1}]."
                )

            pat = getattr(s, "stimulus", None)
            if not isinstance(pat, SpatioTemporalPattern):
                legacy.append(s)
                continue

            if getattr(pat, "_spatial_pattern", None) is None and hasattr(
                pat, "configure_space"
            ):
                pat.configure_space()
            sp_raw = getattr(pat, "_spatial_pattern", None)
            if sp_raw is None:
                legacy.append(s)
                continue
            sp_arr = np.asarray(sp_raw, dtype=np.float64)
            if sp_arr.ndim == 1:
                sp_arr = sp_arr[:, np.newaxis]
            if getattr(s, "weights", None) is not None:
                sp_arr = s.weights @ sp_arr
            if sp_arr.shape[0] > sn_info.n_nodes:
                sp_arr = sp_arr[:sn_info.n_nodes]
            sp_vec = np.ascontiguousarray(sp_arr, dtype=np.float32).ravel()
            if sp_vec.size != sn_info.n_nodes:
                legacy.append(s)
                continue

            temp = getattr(pat, "temporal", None)
            if temp is None:
                legacy.append(s)
                continue

            matched = None
            temp_cls = type(temp)
            # PulseTrain has discontinuous step boundaries (var > onset, phase < tau);
            # when dt0 is not exact in binary float32 (e.g. 0.1 ms), simw's float32
            # dt causes step*float32(dt) to drift across boundaries vs float64 arange.
            # Evaluate it on the float64 master grid as compact 1D STIM_DISCRETE_1D.
            if (temp_cls is _eqs.PulseTrain
                    and float(dt0) != float(np.float32(dt0))):
                matched = None
            else:
                for eq_cls, eq_kind, pkeys in eq_map:
                    if temp_cls is eq_cls:
                        matched = (eq_kind, pkeys)
                        break

            if matched is not None:
                eq_kind, pkeys = matched
                eq_params = [float(temp.parameters[k]) for k in pkeys]
                specs.append({
                    "n_node": int(sn_info.n_nodes),
                    "target_cvars": target_slots.tolist(),
                    "spatial": sp_vec,
                    "lane_varying": False,
                    "eq_kind": int(eq_kind),
                    "onset_t0": 0.0,
                    "dt0": float(dt0),
                    "eq_params": eq_params,
                    "scale": float(s.projection_scale),
                })
            else:
                # Arbitrary custom temporal equation -> 1D master-grid table
                t_grid = np.arange(
                    0, int(total_steps) + 1, dtype=np.float64) * float(dt0)
                t_1d = np.ascontiguousarray(
                    temp.evaluate(t_grid), dtype=np.float32).ravel()
                specs.append({
                    "n_node": int(sn_info.n_nodes),
                    "target_cvars": target_slots.tolist(),
                    "spatial": sp_vec,
                    "lane_varying": False,
                    "eq_kind": 0,  # STIM_DISCRETE_1D
                    "dt0": float(dt0),
                    "temporal_1d": t_1d,
                    "step_offset": 0,
                    "scale": float(s.projection_scale),
                })
        return specs, legacy

    def _build_legacy_stim_arrays(self, analysis, legacy_by_sn, nstep, width,
                                  step_offset=0):
        """Legacy 5D dense stimulus fallback for non-separable stimuli."""
        from tvb.simulator.backend.hybrid_common import _stim_master_grid
        if not any(legacy_by_sn):
            return None
        stim_by_sn = []
        for sn, leg_list in zip(analysis.subnets, legacy_by_sn):
            if not leg_list:
                stim_by_sn.append(None)
                continue
            nm = sn.n_modes
            arr = np.zeros(
                (nstep, sn.n_cvar, sn.n_nodes, nm, width), np.float32)
            for stim in (_stim_master_grid(
                    s, analysis.dt, step_offset + int(nstep))
                    for s in leg_list):
                target_slots = np.asarray(stim.target_cvar).astype(np.intp)
                for step_idx in range(step_offset + 1,
                                      step_offset + int(nstep) + 1):
                    sc = np.asarray(stim.get_coupling(step_idx), dtype=np.float32)
                    if sc.ndim == 2:
                        sc = sc[:, :, np.newaxis]
                    sc = np.broadcast_to(
                        sc, (target_slots.size, sn.n_nodes, nm))
                    for li, slot in enumerate(target_slots):
                        arr[step_idx - step_offset - 1, slot, :, :, 0] += sc[li]
            if width > 1:
                arr = np.repeat(arr, width, axis=4)
            stim_by_sn.append(np.ascontiguousarray(arr))
        return stim_by_sn

    def _run_compiled(
        self, sim, analysis, network_set, nstep=100, chunk_size=None,
        initial_states=None, monitors=None, width=1, _init_only=False,
        kernel_collect=None, _skip_init=False, _step_offset=0,
        kernel_persn=None, kernel_global=None, sn_kernel_monitors=None,
        complete_window_isteps=None, balloon_windkessel_specs=None,
        explicit_chunk_size=None, need_tavg=True, need_ctavg=True,
        _return_snapshot=False, rng_mode="numpy",
        _pending_kmon_states=None, _pending_kmon_global_states=None,
        _pending_cpp_rng_states=None,
    ):
        """Run the C++ core and return ``(results, kernel_streams, subnet_streams)``."""
        self._validate_monitors(monitors, chunk_size)
        self._validate_horizons(analysis)
        cs = int(chunk_size) if chunk_size else 1

        # Backward compat if an internal caller passes legacy `kernel_collect`
        if kernel_persn is None:
            kernel_persn = list(kernel_collect or ())
        if kernel_global is None:
            kernel_global = []

        n_sn = len(analysis.subnets)
        should_merge = (n_sn > 1) and _can_merge_subnets(analysis.subnets)

        if not _skip_init:
            # set_subnet_state resets the simulator tick. Arm/reset staged
            # monitors only after the fresh clock and state are initialized.
            self._init_states(sim, analysis, network_set, initial_states, width)
        elif balloon_windkessel_specs and _pending_kmon_global_states is None:
            raise ValueError("BalloonWindkessel resume requires global monitor snapshot state")
        elif complete_window_isteps and _pending_kmon_global_states is None:
            raise ValueError("C++ regional complete windows require global monitor snapshot state")

        # 1. Arm per-subnet monitors (global persn monitors + Recorder monitors)
        persn_counts = [0] * n_sn
        sn_mon_map_idx = [[] for _ in range(n_sn)]  # (kind_tag, key, slot_pos)
        for si, sn in enumerate(analysis.subnets):
            n_voi = len(sn.voi) // 3
            atts = []
            for mi, m in kernel_persn:
                atts.append(self._monitor_attach_spec(
                    mi, m, n_voi, sn.n_nodes, analysis.dt,
                    cadence_K=1,
                    explicit_chunk_size=explicit_chunk_size,
                    nstep=int(nstep),
                    is_resuming=_skip_init,
                    return_snapshot=_return_snapshot,
                ))
                sn_mon_map_idx[si].append(("global_persn", mi))
            if sn_kernel_monitors is not None:
                for ri, rm in sn_kernel_monitors.get(sn.name, ()):
                    atts.append(self._monitor_attach_spec(
                        10000 + ri, rm, n_voi, sn.n_nodes, float(sn.dt),
                        cadence_K=sn.k,
                        explicit_chunk_size=None,
                        nstep=int(nstep),
                        is_resuming=_skip_init,
                        return_snapshot=_return_snapshot,
                    ))
                    sn_mon_map_idx[si].append(("recorder", ri))
            sim.set_monitors(si, atts)
            persn_counts[si] = len(atts)

        # 2. Arm multi-subnet / single-subnet global monitors
        if kernel_global:
            gather_specs = []
            total_nodes = (
                max(int(sn.node_indices.max()) for sn in analysis.subnets) + 1
                if should_merge
                else analysis.subnets[0].n_nodes
            )
            for sn in analysis.subnets:
                gather_specs.append({
                    "n_voi": len(sn.voi) // 3,
                    "n_node": int(sn.n_nodes),
                    "voi_offset": 0,
                    "node_offset": 0,
                    "node_indices": (
                        sn.node_indices.astype(np.int32).tolist()
                        if should_merge else None
                    ),
                })
            n_voi_g = len(analysis.subnets[0].voi) // 3
            g_atts = [
                (balloon_windkessel_specs or {}).get(mi) or self._monitor_attach_spec(
                    mi, m, n_voi_g, total_nodes, analysis.dt,
                    cadence_K=1,
                    explicit_chunk_size=explicit_chunk_size,
                    nstep=int(nstep),
                    is_resuming=_skip_init,
                    return_snapshot=_return_snapshot,
                    complete_window_istep=(complete_window_isteps or {}).get(mi),
                )
                for mi, m in kernel_global
            ]
            sim.set_global_monitors(gather_specs, g_atts)
        else:
            sim.set_global_monitors(
                [{"n_voi": len(sn.voi) // 3, "n_node": int(sn.n_nodes),
                  "voi_offset": 0, "node_offset": 0}
                 for sn in analysis.subnets],
                [],
            )

        # Restore kmon states on resume if provided
        if _pending_kmon_states is not None:
            sim.set_monitor_states(_pending_kmon_states)
        if _pending_kmon_global_states is not None:
            sim.set_global_monitor_states(_pending_kmon_global_states)

        if not _skip_init:
            sim.monitor_begin_run()

        # 3. Configure RNG (NumPy external chunked OR C++ Philox4x32-10)
        stoch = [sn for sn in analysis.subnets if sn.is_stochastic]
        use_philox = (rng_mode == "philox") and bool(stoch)
        sim.set_rng_states([{"enabled": False} for _ in range(n_sn)])
        if use_philox:
            for si, sn in enumerate(analysis.subnets):
                if sn.is_stochastic:
                    gfun = np.sqrt(2.0 * sn.noise_nsig * sn.dt).astype(np.float32)
                    seed = int(sn.noise_seed) & 0xFFFFFFFFFFFFFFFF
                    sim.set_subnet_rng(
                        si, seed, gfun, ntau=float(sn.noise_ntau))
            if _pending_cpp_rng_states is not None:
                sim.set_rng_states(_pending_cpp_rng_states)

        if stoch and not use_philox:
            noise_std = [
                np.sqrt(2.0 * sn.noise_nsig * sn.dt) for sn in stoch
            ]
            rngs = []
            sn_objs = {sn.name: s for sn, s in
                       zip(analysis.subnets, network_set.subnets)}
            for sn in stoch:
                rngs.append(sn_objs[sn.name].scheme.noise.random_stream)

        # 4. Lower separable stimuli to C++ `cph::stim_spec<W>` (with legacy fallback)
        sim.clear_stim_specs()
        legacy_by_sn = []
        for si, sn in enumerate(analysis.subnets):
            if sn.stimuli:
                sp_list, leg_list = self._lower_subnet_stimuli(
                    sn, analysis.dt, _step_offset + int(nstep))
                if sp_list:
                    sim.set_stim_specs(si, sp_list)
                legacy_by_sn.append(leg_list)
            else:
                legacy_by_sn.append([])
        stim_by_sn = self._build_legacy_stim_arrays(
            analysis, legacy_by_sn, int(nstep), width,
            step_offset=_step_offset)

        global_calls = []
        if not stoch or use_philox:
            outs = sim.run(
                int(nstep), cs, None, stim_by_sn,
                need_tavg=bool(need_tavg), need_ctavg=bool(need_ctavg),
            )
            if kernel_global:
                global_calls.append(sim.take_global_monitor_outputs())
        else:
            # Chunk-wise loop to draw NumPy MT19937 noise per chunk
            remaining = int(nstep)
            done = 0
            chunk_results = []
            while remaining > 0:
                this_chunk = min(remaining, max(cs, 1))
                noise_by_sn = [None] * n_sn
                for sn, std, rng in zip(stoch, noise_std, rngs):
                    dw = rng.randn(this_chunk, sn.n_svar, sn.n_nodes,
                                   sn.n_modes)
                    dw = dw * std[np.newaxis, :, np.newaxis, np.newaxis]
                    nz = np.ascontiguousarray(
                        np.repeat(dw[..., np.newaxis], width,
                                  axis=4)).astype(np.float32)
                    noise_by_sn[analysis.subnets.index(sn)] = nz
                chunk_stim = (
                    [None if sa is None
                     else np.ascontiguousarray(sa[done:done + this_chunk])
                     for sa in stim_by_sn]
                    if stim_by_sn is not None else None
                )
                outs = sim.run(
                    this_chunk, cs, noise_by_sn, chunk_stim,
                    need_tavg=bool(need_tavg), need_ctavg=bool(need_ctavg),
                )
                if kernel_global:
                    # The next native run clears these per-call output buffers.
                    global_calls.append(sim.take_global_monitor_outputs())
                chunk_results.append(outs)
                done += this_chunk
                remaining -= this_chunk
            n_out = len(chunk_results[0])
            all_outs = []
            for idx_o in range(n_out):
                if chunk_results[0][idx_o] is None:
                    all_outs.append(None)
                else:
                    all_outs.append(
                        np.concatenate([cr[idx_o] for cr in chunk_results], axis=0)
                    )
            outs = all_outs

        # 5. Unpack per-subnet outputs and monitor streams
        results = []
        kernel_streams = {}
        subnet_streams = {}
        pos = 0
        dt0_q = float(np.float32(analysis.dt))
        for i, sn in enumerate(analysis.subnets):
            raw_t = outs[pos]
            raw_c = outs[pos + 1]
            tavg = (
                np.asarray(raw_t, dtype=np.float32)
                if raw_t is not None else None
            )
            ctavg = (
                np.asarray(raw_c, dtype=np.float32)
                if raw_c is not None else None
            )
            if tavg is not None:
                n_chunks = tavg.shape[0]
            elif ctavg is not None:
                n_chunks = ctavg.shape[0]
            else:
                n_chunks = (int(nstep) + cs - 1) // cs
            steps_lo = _step_offset + np.arange(n_chunks) * cs + 1
            steps_hi = _step_offset + np.minimum(
                np.arange(n_chunks) * cs + cs, int(nstep))
            times = (steps_lo + steps_hi) * 0.5 * dt0_q
            results.append((times, tavg, ctavg))

            pos += 2
            for tag, key in sn_mon_map_idx[i]:
                ts = np.asarray(outs[pos], dtype=np.float64)
                dd = np.asarray(outs[pos + 1], dtype=np.float32)
                pos += 2
                if tag == "global_persn":
                    kernel_streams.setdefault(key, [None] * n_sn)[i] = (ts, dd)
                else:
                    subnet_streams.setdefault(sn.name, []).append((key, (ts, dd)))

        for (mi, _m), calls in zip(kernel_global, zip(*global_calls)):
            if len(calls) == 1:
                ts, dd = calls[0]
            else:
                times, data = zip(*calls)
                ts, dd = np.concatenate(times), np.concatenate(data)
            kernel_streams[mi] = [(ts, dd)]

        return results, kernel_streams, subnet_streams

    def run_network(
        self,
        network_set: NetworkSet,
        nstep: int,
        chunk_size: int = None,
        print_source: bool = False,
        initial_states: Optional[list] = None,
        debug_nojit: bool = False,
        monitors: Optional[list] = None,
        rng_mode: str = "numpy",
        subnet_monitors: Optional[dict] = None,
        **kwargs,
    ):
        """Run a hybrid simulation with the C++ backend.

        Equivalent to ``self.compile(network_set).run(nstep, ...)``.
        """
        if monitors is not None and any(isinstance(m, BalloonWindkessel) for m in monitors):
            # Reject unsupported topology before compiling a model that might
            # itself be unsupported. Compiled runs also check native geometry.
            if len(network_set.subnets) != 1:
                raise NotImplementedError("BalloonWindkessel requires one regional subnetwork")
            if network_set.subnets[0].model.number_of_modes != 1:
                raise NotImplementedError("BalloonWindkessel requires width 1 and one source clock and mode")
        cn = self.compile(network_set, print_source=print_source)
        return cn.run(
            nstep, chunk_size, initial_states, monitors,
            rng_mode=rng_mode, subnet_monitors=subnet_monitors, **kwargs,
        )

    # ------------------------------------------------------------------
    # sweeps: multithreaded SIMD batches (lanes = sweep members)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_sweep(network_set, params):
        """Resolve a named-parameter dict like nb_hybrid._resolve_named_params.

        Returns (targets, sweep_values) where targets is a list of
        ('cfun', projection_name, param_idx) or ('model', subnet_name, parm).
        """
        descriptor, sweep_values = _resolve_named_params(
            network_set, params
        )
        targets = []
        for desc in descriptor:
            if desc["type"] == "cfun":
                targets.append(("cfun", desc["projection"],
                                int(desc["param_idx"])))
            else:
                targets.append(("model", desc["subnet"], desc["param"]))
        return targets, sweep_values

    def _apply_sweep_group(self, sim, analysis, targets, group_values, width):
        """Set per-lane swept parameters on one lane-group sim.

        group_values: (width, n_dims)
        """
        group_size = group_values.shape[0]
        cp_by_proj = {}
        pp_by_sn = {}
        for d, tgt in enumerate(targets):
            vals = group_values[:, d]  # (group_size,)
            if tgt[0] == "cfun":
                _, pname, pidx = tgt
                matched = False
                for pi, pr in enumerate(analysis.projections):
                    if (pr.name == pname or pr.lookup == pname
                            or pr.lookup.endswith("." + pname)):
                        matched = True
                        if pr.n_cfun_parm:
                            cp = cp_by_proj.get(pi)
                            if cp is None:
                                cp = np.broadcast_to(
                                    pr.cfun_params[:, None],
                                    (pr.n_cfun_parm, width),
                                ).copy()
                                cp_by_proj[pi] = cp
                            if pr.cfun_id == 8 and pidx == 0 and pr.cfun_use_classic:
                                cp[pidx, :group_size] = vals * vals
                            else:
                                cp[pidx, :group_size] = vals
                if not matched:
                    raise ValueError(f"Projection '{pname}' not found in sweep")
            else:
                _, sname, attr = tgt
                si = next(i for i, sn in enumerate(analysis.subnets)
                          if sn.name == sname)
                sn = analysis.subnets[si]
                k = sn.parm_index[attr]
                pp = pp_by_sn.get(si)
                if pp is None:
                    pp = np.broadcast_to(
                        sn.params[:, :, None],
                        (sn.n_nodes, sn.n_parm, width),
                    ).copy()
                    pp_by_sn[si] = pp
                pp[:, k, :group_size] = vals[None, :]
        for pi, cp in cp_by_proj.items():
            sim.set_cfun_params(pi, cp)
        for si, pp in pp_by_sn.items():
            sim.set_subnet_params(si, pp)

    _ONLINE_FEATURE_NAMES = frozenset({
        "mean", "var", "std", "min", "max", "rms", "abs_energy",
        "average_power", "auc", "auc_trapz", "zero_crossing",
        "burstiness", "moments", "skewness", "kurtosis", "centroid",
    })

    def sweep(
        self,
        network_set,
        params,
        nstep: int = 100,
        *,
        backend: str = "cpp",
        n_workers: int = None,
        monitor: str = "tavg",
        monitor_period: int = 1,
        bold_period: Optional[float] = None,
        chunk_size: Optional[int] = None,
        initial_states: Optional[list] = None,
        node_indices: Optional[dict] = None,
        width: Optional[int] = None,
        progress: bool = False,
        features: Optional[list] = None,
        noise_seeds: Optional[np.ndarray] = None,
        shared_noise: bool = False,
    ) -> "SweepResult":
        """Run a parameter sweep as multithreaded SIMD batches.

        Each SIMD lane (width in ``{1, 4, 8, 16}``) carries one sweep member;
        lane groups run concurrently in Python threads (the C++ kernel releases
        the GIL).  Supports deterministic and stochastic sweeps (via C++
        Philox4x32-10), separable stimuli, ``monitor`` in
        ``('tavg', 'raw', 'subsample', 'bold', 'none', 'features')``,
        ``bold_period``, ``chunk_size``, and optional Tier-1 ``features``.
        """
        import time as _time_mod
        from . import notebook as _nb_sugar

        _reject_balloon_windkessel_sweep(monitor, bold_period)
        if width is None or width == "auto":
            n_preview = len(next(iter(params.values()))) if isinstance(params, dict) and params else 1
            width = _nb_sugar.default_width(n_preview, network_set)
        if isinstance(width, bool) or width not in (1, 4, 8, 16):
            raise ValueError(
                f"Unsupported SIMD width {width!r}; expected one of (1, 4, 8, 16)."
            )
        width = int(width)

        if monitor is None:
            monitor = "none"
        if monitor not in ("tavg", "raw", "subsample", "bold", "none", "features"):
            raise ValueError(
                f"Unsupported sweep monitor {monitor!r}; expected "
                "'tavg', 'raw', 'subsample', 'bold', or 'none'."
            )
        if chunk_size is not None and (
            isinstance(chunk_size, (bool, np.bool_))
            or not isinstance(chunk_size, (int, np.integer))
            or chunk_size <= 0
        ):
            raise ValueError("chunk_size must be a positive integer")
        if (isinstance(monitor_period, (bool, np.bool_))
                or not isinstance(monitor_period, (int, np.integer))
                or monitor_period <= 0):
            raise ValueError("monitor_period must be a positive integer")
        if bold_period is not None:
            try:
                valid_bp = (
                    not isinstance(bold_period, (bool, np.bool_))
                    and np.isfinite(float(bold_period))
                    and float(bold_period) > 0
                )
            except (TypeError, ValueError):
                valid_bp = False
            if not valid_bp:
                raise ValueError("bold_period must be positive")
        if (isinstance(nstep, (bool, np.bool_))
                or not isinstance(nstep, (int, np.integer))
                or nstep <= 0):
            raise ValueError("nstep must be a positive integer")

        online_feat_names = []
        tier2_feat_names = []
        if features is not None:
            from . import features as _feat_mod
            for f in features:
                if isinstance(f, str):
                    if f in self._ONLINE_FEATURE_NAMES:
                        online_feat_names.append(f)
                    elif f in _feat_mod.FEATURES:
                        tier2_feat_names.append(f)
                    else:
                        raise KeyError(
                            f"unknown feature {f!r}; choices: "
                            f"{sorted(_feat_mod.FEATURES)}"
                        )
                else:
                    tier2_feat_names.append(f)

        targets, sweep_values = self._resolve_sweep(network_set, params)
        analysis = self._analyse(network_set)

        cs = int(chunk_size) if chunk_size is not None else 1
        want_raw_stream = (
            (monitor == "raw" and cs > 1)
            or (monitor == "subsample")
        )
        want_bold = (monitor == "bold") or (bold_period is not None)
        eff_bold_period = float(bold_period) if bold_period is not None else 2000.0
        need_tavg = (
            monitor in ("tavg", "raw", "subsample")
            or bool(tier2_feat_names)
        )
        need_ctavg = (monitor in ("tavg", "raw", "subsample"))

        results, workers, elapsed = self._sweep_impl(
            network_set, analysis, targets, sweep_values, int(nstep),
            initial_states, width, n_workers, _time_mod,
            label=f"cpp-simd(width={width})", progress=progress,
            chunk_size=cs,
            need_tavg=need_tavg,
            need_ctavg=need_ctavg,
            want_raw_kind=(
                1 if (monitor == "raw" and cs > 1)
                else (2 if monitor == "subsample" else 0)
            ),
            monitor_period=int(monitor_period),
            want_bold=want_bold,
            bold_period=eff_bold_period,
            online_feat_names=online_feat_names,
            noise_seeds=noise_seeds,
            shared_noise=shared_noise,
        )

        # Assemble outputs across lane groups
        tavg = {sn.name: [] for sn in analysis.subnets} if need_tavg else {}
        ctavg = {sn.name: [] for sn in analysis.subnets} if need_ctavg else {}
        raw_out = (
            {sn.name: [] for sn in analysis.subnets}
            if monitor in ("raw", "subsample") else None
        )
        bold_out = (
            {sn.name: [] for sn in analysis.subnets} if want_bold else None
        )
        online_out = (
            {sn.name: {fn: [] for fn in online_feat_names}
             for sn in analysis.subnets}
            if online_feat_names else None
        )
        bold_times = None

        n_extra_per_sn = (1 if want_raw_stream else 0) + (1 if want_bold else 0)
        for lo, hi, outs, feat_per_sn in results:
            k_lanes = hi - lo
            for i, sn in enumerate(analysis.subnets):
                base = i * (2 + 2 * n_extra_per_sn)
                if need_tavg:
                    arr = np.asarray(outs[base], dtype=np.float32)
                    tavg[sn.name].append(
                        arr.transpose(3, 0, 1, 2)[:k_lanes, :, :, :, None])
                if need_ctavg:
                    ct = np.asarray(outs[base + 1], dtype=np.float32)
                    ctavg[sn.name].append(
                        ct.transpose(3, 0, 1, 2)[:k_lanes, :, :, :, None])
                pos = base + 2
                if want_raw_stream:
                    r_arr = np.asarray(outs[pos + 1], dtype=np.float32)
                    raw_out[sn.name].append(
                        r_arr.transpose(3, 0, 1, 2)[:k_lanes, :, :, :, None])
                    pos += 2
                if want_bold:
                    b_t = np.asarray(outs[pos], dtype=np.float64)
                    b_arr = np.asarray(outs[pos + 1], dtype=np.float32)
                    bold_out[sn.name].append(
                        b_arr.transpose(3, 0, 1, 2)[:k_lanes, :, :, :, None])
                    if bold_times is None:
                        bold_times = b_t
                    pos += 2
                if online_out is not None:
                    fdict = feat_per_sn[i]
                    for fn in online_feat_names:
                        fa = np.asarray(fdict[fn], dtype=np.float64)
                        if fn == "moments":
                            # (n_comp, n_voi, n_node, W) -> (k_lanes, n_comp, n_voi, n_node)
                            online_out[sn.name][fn].append(
                                fa.transpose(3, 0, 1, 2)[:k_lanes])
                        else:
                            # (n_voi, n_node, W) -> (k_lanes, n_voi, n_node)
                            online_out[sn.name][fn].append(
                                fa.transpose(2, 0, 1)[:k_lanes])

        for name in tavg:
            tavg[name] = np.concatenate(tavg[name], axis=0)
        for name in ctavg:
            ctavg[name] = np.concatenate(ctavg[name], axis=0)
        if monitor == "raw" and cs == 1 and raw_out is not None:
            raw_out = {name: arr.copy() for name, arr in tavg.items()}
        elif raw_out is not None:
            for name in raw_out:
                raw_out[name] = np.concatenate(raw_out[name], axis=0)
        if bold_out is not None:
            for name in bold_out:
                bold_out[name] = np.concatenate(bold_out[name], axis=0)
        if online_out is not None:
            for name in online_out:
                for fn in online_feat_names:
                    online_out[name][fn] = np.concatenate(
                        online_out[name][fn], axis=0)

        merged = None
        if need_tavg and tavg:
            same_shape = len({(v.shape[2], v.shape[4]) for v in tavg.values()}) == 1
            same_dt = len({float(sn.dt) for sn in analysis.subnets}) == 1
            if len(analysis.subnets) == 1:
                merged = tavg[analysis.subnets[0].name]
            elif same_shape and same_dt:
                if node_indices:
                    n_global = max(
                        max(indices) for indices in node_indices.values()) + 1
                    ref = next(iter(tavg.values()))
                    merged = np.zeros(
                        (ref.shape[0], ref.shape[1], ref.shape[2],
                         n_global, ref.shape[4]),
                        dtype=ref.dtype,
                    )
                    for sn in analysis.subnets:
                        if sn.name in node_indices:
                            merged[:, :, :, node_indices[sn.name], :] = tavg[sn.name]
                elif _can_merge_subnets(analysis.subnets):
                    n_global = max(
                        int(sn.node_indices.max())
                        for sn in analysis.subnets
                    ) + 1
                    ref = next(iter(tavg.values()))
                    merged = np.zeros(
                        (ref.shape[0], ref.shape[1], ref.shape[2],
                         n_global, ref.shape[4]),
                        dtype=ref.dtype,
                    )
                    for sn in analysis.subnets:
                        merged[:, :, :, sn.node_indices, :] = tavg[sn.name]
                else:
                    merged = np.concatenate(
                        [tavg[sn.name] for sn in analysis.subnets], axis=3
                    )

        if need_tavg and tavg:
            n_samples = tavg[analysis.subnets[0].name].shape[1]
            steps_lo = np.arange(n_samples) * cs + 1
            steps_hi = np.minimum(np.arange(n_samples) * cs + cs, int(nstep))
            dt0_q = float(np.float32(analysis.dt))
            times = (steps_lo + steps_hi) * 0.5 * dt0_q
        elif want_bold and bold_times is not None:
            times = bold_times
        else:
            times = np.zeros(0, dtype=np.float64)

        return SweepResult(
            tavg=tavg,
            merged_tavg=merged,
            ctavg=ctavg,
            times=times,
            sweep_values=sweep_values,
            backend=f"cpp-simd(width={width},workers={workers})",
            elapsed=elapsed,
            raw=raw_out,
            bold=bold_out,
            param_keys=(list(params.keys()) if isinstance(params, dict)
                        else None),
            online_features=online_out,
        )

    def _sweep_impl(
        self, network_set, analysis, targets, sweep_values, nstep,
        initial_states, width, n_workers, _time_mod, label,
        progress=False, chunk_size=1, need_tavg=True, need_ctavg=True,
        want_raw_kind=0, monitor_period=1, want_bold=False, bold_period=2000.0,
        online_feat_names=(), noise_seeds=None, shared_noise=False,
    ):
        n_sweeps = sweep_values.shape[0]
        n_groups = (n_sweeps + width - 1) // width
        t0 = _time_mod.perf_counter()

        # Pre-lower separable stimuli and monitor specs once across groups
        stim_specs_by_sn = []
        legacy_by_sn = []
        for sn in analysis.subnets:
            if sn.stimuli:
                sp_list, leg_list = self._lower_subnet_stimuli(
                    sn, analysis.dt, int(nstep))
                stim_specs_by_sn.append(sp_list)
                legacy_by_sn.append(leg_list)
            else:
                stim_specs_by_sn.append([])
                legacy_by_sn.append([])
        legacy_stim_by_sn = self._build_legacy_stim_arrays(
            analysis, legacy_by_sn, int(nstep), width, step_offset=0)

        bold_specs_by_sn = []
        if want_bold:
            from tvb.simulator.monitors import Bold
            for sn in analysis.subnets:
                n_voi = len(sn.voi) // 3
                bm = Bold(period=float(bold_period))
                bold_specs_by_sn.append(
                    self._monitor_attach_spec(
                        1, bm, n_voi, sn.n_nodes, analysis.dt, cadence_K=1)
                )

        def run_group(gi):
            lo = gi * width
            hi = min(lo + width, n_sweeps)
            group_vals = np.zeros((width, sweep_values.shape[1]), np.float32)
            group_vals[: hi - lo] = sweep_values[lo:hi]
            sim = Sim(width)
            self._build_sim(sim, analysis, network_set, width)
            self._apply_sweep_group(sim, analysis, targets, group_vals, width)
            self._init_states(sim, analysis, network_set, initial_states, width)

            # Stimuli (separable C++ specs + optional legacy 5D fallback)
            for si, sp_list in enumerate(stim_specs_by_sn):
                if sp_list:
                    sim.set_stim_specs(si, sp_list)

            # Philox4x32-10 RNG for stochastic subnets
            for si, sn in enumerate(analysis.subnets):
                if sn.is_stochastic:
                    gfun = np.sqrt(
                        2.0 * sn.noise_nsig * sn.dt).astype(np.float32)
                    if noise_seeds is not None:
                        lk = [
                            (int(noise_seeds[lo + w]) & 0xFFFFFFFFFFFFFFFF)
                            if (lo + w) < hi else 0
                            for w in range(width)
                        ]
                    elif shared_noise:
                        lk = int(sn.noise_seed) & 0xFFFFFFFFFFFFFFFF
                    else:
                        base_seed = int(sn.noise_seed) & 0xFFFFFFFFFFFFFFFF
                        lk = [
                            ((base_seed + lo + w) & 0xFFFFFFFFFFFFFFFF)
                            if (lo + w) < hi else base_seed
                            for w in range(width)
                        ]
                    sim.set_subnet_rng(
                        si, lk, gfun, ntau=float(sn.noise_ntau))

            # Optional per-subnet monitor engines (raw/subsample/bold)
            if want_raw_kind or want_bold:
                for si, sn in enumerate(analysis.subnets):
                    atts = []
                    if want_raw_kind == 1:
                        atts.append({
                            "slot": 0, "kind": 1, "istep": 1,
                            "cadence_K": 1, "dt": float(analysis.dt),
                            "voi": [],
                        })
                    elif want_raw_kind == 2:
                        atts.append({
                            "slot": 0, "kind": 2,
                            "istep": int(monitor_period),
                            "cadence_K": 1, "dt": float(analysis.dt),
                            "voi": [],
                        })
                    if want_bold:
                        atts.append(bold_specs_by_sn[si])
                    sim.set_monitors(si, atts)
                sim.monitor_begin_run()

            # Optional Tier-1 online streaming features
            if online_feat_names:
                for si in range(len(analysis.subnets)):
                    sim.enable_online_features(si, float(analysis.dt))

            outs = sim.run(
                int(nstep), int(chunk_size), None, legacy_stim_by_sn,
                need_tavg=bool(need_tavg), need_ctavg=bool(need_ctavg),
            )
            feat_per_sn = None
            if online_feat_names:
                feat_per_sn = [
                    sim.take_online_features(si, list(online_feat_names), order=4)
                    for si in range(len(analysis.subnets))
                ]
            return lo, hi, outs, feat_per_sn

        if sys.platform == "emscripten":
            workers = 1
        else:
            workers = n_workers or min(n_groups, max(1, _default_workers()))
        results = [None] * n_groups
        if workers == 1:
            it = range(n_groups)
            if progress:
                try:
                    import tqdm.auto
                    it = tqdm.auto.tqdm(
                        it, total=n_groups, desc=label, unit="lane-group")
                except Exception:
                    it = range(n_groups)
            for gi in it:
                results[gi] = run_group(gi)
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(run_group, gi) for gi in range(n_groups)]
                it = futures
                if progress:
                    try:
                        import tqdm.auto
                        it = tqdm.auto.tqdm(
                            futures, total=n_groups, desc=label,
                            unit="lane-group")
                    except Exception:
                        it = futures
                for gi, fut in enumerate(it):
                    results[gi] = fut.result()
        elapsed = _time_mod.perf_counter() - t0
        return results, workers, elapsed

    def run_sweep(
        self,
        network_set,
        sweep_values: np.ndarray,
        nstep: int = 100,
        initial_states: Optional[list] = None,
        sweep_descriptor: Optional[list] = None,
        chunk_size: Optional[int] = None,
        bold_period: Optional[float] = None,
        print_source: bool = False,
        width: Optional[int] = None,
        **monitors,
    ):
        """Legacy nb_hybrid-style sweep entry point."""
        import time as _time_mod
        from . import notebook as _nb_sugar

        _reject_balloon_windkessel_sweep(bold_period, *monitors.values())
        sweep_values = np.asarray(sweep_values, dtype=np.float32)
        if sweep_values.ndim == 1:
            sweep_values = sweep_values.reshape(-1, 1)
        if sweep_descriptor is None:
            if network_set.projections:
                first_proj = network_set.projections[0]
                pname = (f"{first_proj.source.name}_to_"
                         f"{first_proj.target.name}")
                sweep_descriptor = [
                    {"type": "cfun", "projection": pname, "param_idx": 0}
                ]
            else:
                sweep_descriptor = []
        targets = []
        for desc in sweep_descriptor:
            if desc["type"] == "cfun":
                targets.append(
                    ("cfun", desc["projection"], int(desc.get("param_idx", 0)))
                )
            else:
                targets.append(("model", desc["subnet"], desc["param"]))
        analysis = self._analyse(network_set)
        if width is None:
            width = _nb_sugar.default_width(sweep_values.shape[0], network_set)
        width = int(width)
        cs = int(chunk_size) if chunk_size else 1
        results, workers, elapsed = self._sweep_impl(
            network_set, analysis, targets, sweep_values, int(nstep),
            initial_states, width, None, _time_mod, "cpp-simd",
            chunk_size=cs, need_tavg=True, need_ctavg=True,
        )
        n_sweeps = sweep_values.shape[0]
        out = []
        dt0_q = float(np.float32(analysis.dt))
        for tid in range(n_sweeps):
            gi = tid // width
            lane = tid % width
            lo, hi, outs, _ = results[gi]
            per_sn = []
            for i, sn in enumerate(analysis.subnets):
                arr = np.asarray(outs[2 * i], dtype=np.float32)
                ct = np.asarray(outs[2 * i + 1], dtype=np.float32)
                data = arr[:, :, :, lane:lane + 1]
                ctavg = ct[:, :, :, lane:lane + 1]
                n_samples = data.shape[0]
                steps_lo = np.arange(n_samples) * cs + 1
                steps_hi = np.minimum(np.arange(n_samples) * cs + cs, int(nstep))
                times = (steps_lo + steps_hi) * 0.5 * dt0_q
                per_sn.append((times, data, ctavg))
            out.append(per_sn)
        return out


def _default_workers():
    if sys.platform == "emscripten":
        return 1
    import os
    return max(1, (os.cpu_count() or 2) // 2)
