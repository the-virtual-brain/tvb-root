# -*- coding: utf-8 -*-
"""C++ backend for the Hybrid Simulator.

Lowers the Python/NumPy hybrid implementation into tight, SIMD-enabled C++
with the simulation control flow in C++ (no monolithic static codegen).
Python keeps control of the simulation; data arrays stay in NumPy float32.

Mirrors the :class:`~tvb.simulator.backend.nb_hybrid.NbHybridBackend` API.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np

from ._cpp_hybrid import Sim

# generic (expression-generated) model ids start at 100 (see _core.cpp)
_GENERIC_BASE = 100
_GEN_LIB = None      # ctypes handle
_GEN_IDS = {}        # model class name -> generic id
from tvb.simulator.backend.nb_hybrid import (
    _CFUN_PARAM_ATTRS,
    _apply_monitors,
    _aggregate_raw_outputs,
    _compute_chunk_size,
    _copy_monitor_states,
    NbHybridBackend,
)
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.integrators import (
    HeunDeterministic,
    EulerDeterministic,
    HeunStochastic,
    EulerStochastic,
)

__all__ = [
    "CppHybridBackend",
    "CompiledCppSim",
    "NetworkAnalysis",
    "SubnetworkInfo",
    "ProjectionInfo",
    "SweepResult",
]

# cfun dispatch ids (must match _core.cpp)
_CFUN_IDS = {
    "Linear": 0, "Scaling": 1, "Sigmoidal": 2, "Difference": 3,
    "Kuramoto": 4, "HyperbolicTangent": 5, "SigmoidalJansenRit": 6,
    "PreSigmoidal": 7,
}
_PASSTHROUGH_CFUN = -1  # no cfun: plain weighted sum
# model dispatch ids (must match _core.cpp)
_MODEL_IDS = {
    "MontbrioPazoRoxin": 0, "Generic2dOscillator": 1, "Kuramoto": 2,
    "SupHopf": 3, "Linear": 4, "ReducedWongWang": 5, "WilsonCowan": 6,
    "JansenRit": 7, "Epileptor": 8, "Epileptor2D": 9,
    "ZerlautAdaptationFirstOrder": 10, "ZerlautAdaptationSecondOrder": 11,
}
# per-model parameter attribute order (must match the C++ dfuns; this is the
# nb_hybrid gufunc argument order for each model)
_MODEL_PARM_NAMES = {
    "MontbrioPazoRoxin": ["tau", "I", "Delta", "J", "eta", "cr"],
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



def _squeeze_lane(a):
    """Drop a trailing singleton SIMD-lane dim (nb_hybrid keeps 3-dim states)."""
    if a.ndim == 4 and a.shape[-1] == 1:
        return a[..., 0]
    return a


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
    horizon: int = 1
    n_modes: int = 1
    is_stochastic: bool = False
    noise_nsig: np.ndarray = None
    noise_seed: int = 0
    stimuli: list = dataclasses.field(default_factory=list)
    node_indices: np.ndarray = None

    @property
    def has_stimulus(self):
        return bool(self.stimuli)
    parm_index: dict = dataclasses.field(default_factory=dict)


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
        ts = self.ts(svar=svar, subnet=subnet)   # (sweep, node, time)
        out = {}
        for k in feature_names:
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

    def warmup(self) -> float:
        return 0.0  # no JIT: nothing to warm up

    def run(
        self,
        nstep: int,
        chunk_size: int = None,
        initial_states: Optional[list] = None,
        monitors: Optional[list] = None,
        return_snapshot: bool = False,
        **kwargs,
    ):
        """Run the simulation for *nstep* steps.

        Mirrors :meth:`NbHybridBackend.run`.  With no monitors returns a list
        per subnetwork of ``(times, data, ctavg)``.  With monitors returns
        ``list[monitor][subnetwork]`` of ``(times, data)``.
        """
        dt = self._network_set.subnets[0].scheme.dt
        if chunk_size is None:
            chunk_size = (
                _compute_chunk_size(monitors, dt) if monitors is not None else 1
            )
        if monitors is not None:
            self._backend._validate_monitors(monitors, chunk_size)
        execution_chunk_size = chunk_size
        kernel_monitors = monitors
        has_bold = has_temporal_average = False
        if monitors is not None:
            from tvb.simulator.monitors import Bold, TemporalAverage
            has_bold = any(isinstance(m, Bold) for m in monitors)
            has_temporal_average = any(
                isinstance(m, TemporalAverage) for m in monitors)
            if has_bold or has_temporal_average:
                # Stateful monitors consume every observed state.
                execution_chunk_size = 1
                kernel_monitors = []
                if self._bold_states is None:
                    self._bold_states = {}
        outputs = self._backend._run_compiled(
            self._sim, self._analysis, self._network_set,
            nstep=nstep, chunk_size=execution_chunk_size,
            initial_states=initial_states, monitors=kernel_monitors,
            width=self._width,
        )
        if monitors is not None:
            bold_raw_outputs = outputs if has_bold else None
            temporal_raw_outputs = outputs if has_temporal_average else None
            if has_bold or has_temporal_average:
                outputs = _aggregate_raw_outputs(outputs, chunk_size)
            outputs = _apply_monitors(
                outputs, monitors, dt, chunk_size=chunk_size,
                monitor_data=None, subnet_infos=self._analysis.subnetworks,
                bold_states=self._bold_states,
                bold_raw_outputs=bold_raw_outputs,
                temporal_raw_outputs=temporal_raw_outputs,
            )
        if not return_snapshot:
            return outputs
        snap = {
            "states": [
                _squeeze_lane(np.asarray(self._sim.get_subnet_state(i)))
                for i in range(len(self._analysis.subnets))
            ],
            "buffers": {},
            "step": int(self._sim.t_abs()),
            "rng_states": self._backend._get_rng_states(
                self._analysis, self._network_set,
            ),
            "monitor_states": _copy_monitor_states(self._bold_states),
        }
        return outputs, snap

    def resume(
        self,
        snapshot: dict,
        nstep: int,
        chunk_size: int = None,
        return_snapshot: bool = False,
        monitors: Optional[list] = None,
    ) -> list:
        """Resume from a snapshot returned by :meth:`run`."""
        states = snapshot["states"]
        self._backend._init_states(
            self._sim, self._analysis, self._network_set, states, self._width)
        outs = self.run(nstep, chunk_size, states, monitors,
                        return_snapshot=return_snapshot)
        return outs


class CppHybridBackend:
    """SIMD C++ backend for hybrid NetworkSet simulations."""

    _CACHE_DIR = None

    _RUN_FN_CACHE = {}  # topology-key -> cached _run_network_fn token

    def __init__(self):
        pass

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
            from tvb.simulator.backend import nb_hybrid as nbh
            nbh._COMPILED_FN_CACHE.clear()
        except Exception:
            pass

    @classmethod
    def _stim_estimate_mb(cls, sn_info, nstep):
        n_cvar = len(sn_info.model.cvar)
        n_bytes = n_cvar * sn_info.n_nodes * sn_info.n_modes * nstep * 4
        return n_bytes / (1024 * 1024)

    def _resolve_named_params(self, network_set, params):
        return NbHybridBackend._resolve_named_params(network_set, params)

    # ------------------------------------------------------------------
    # compatibility / analysis
    # ------------------------------------------------------------------

    def _check_compatibility(self, network_set: NetworkSet) -> None:
        _allowed_integrators = (
            HeunDeterministic, EulerDeterministic,
            HeunStochastic, EulerStochastic,
        )
        dt0 = network_set.subnets[0].scheme.dt
        for sn in network_set.subnets:
            model_name = type(sn.model).__name__
            if (model_name not in _MODEL_IDS
                    and model_name not in _GEN_IDS):
                self._model_key(sn.model)  # raises if unsupported
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
            if sn.scheme.dt != dt0:
                raise ValueError("All subnetworks must share the same dt.")
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
            if cfun_name == "PreSigmoidal" and getattr(p.cfun, "dynamic", True):
                raise NotImplementedError(
                    "CppHybridBackend supports PreSigmoidal with "
                    "dynamic=False only."
                )
            if cfun_name is None:
                continue  # passthrough projections are supported
            if cfun_name not in _CFUN_IDS:
                raise NotImplementedError(
                    f"CppHybridBackend does not support coupling {cfun_name}. "
                    f"Supported: {sorted(_CFUN_IDS)}"
                )
            if np.size(p.target_cvar) != 1:
                raise NotImplementedError(
                    "CppHybridBackend currently supports a single target "
                    "cvar per projection."
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
        subnet_index = {sn.name: i for i, sn in enumerate(network_set.subnets)}
        dt = float(network_set.subnets[0].scheme.dt)

        subnets = []
        for sn in network_set.subnets:
            model_id, n_parm, n_cvar_m, n_svar_m, parm_names = \
                self._model_key(sn.model)
            params = self._params_for_model(sn.model, parm_names)  # (n_parm,)
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
            model_name = type(sn.model).__name__
            nsig = None
            seed = 0
            if is_stoch:
                nsig = np.asarray(sn.scheme.noise.nsig, dtype=np.float64).ravel()
                if nsig.size == 1:
                    nsig = np.full(sn.model.nvar, float(nsig[0]))
                seed = int(getattr(sn.scheme.noise, "noise_seed", 0) or 0)
            subnets.append(SubnetworkInfo(
                stimuli=list(sn.stimuli or []),
                name=sn.name,
                n_nodes=int(sn.nnodes),
                n_svar=int(sn.model.nvar) if n_svar_m is None else n_svar_m,
                n_parm=len(parm_names),
                n_modes=int(getattr(sn.model, "number_of_modes", 1)),
                n_cvar=(int(sn.model.cvar.size) if n_cvar_m is None
                        else n_cvar_m),
                model_id=model_id,
                model_name=model_name,
                model=sn.model,
                params=params,
                voi=voi,
                dt=dt,
                is_stochastic=is_stoch,
                noise_nsig=nsig,
                noise_seed=seed,
                node_indices=(np.asarray(sn.node_indices, dtype=np.int64)
                             if getattr(sn, "node_indices", None) is not None
                             else None),
                parm_index={k: i for i, k in enumerate(parm_names)},
            ))

        projections = []
        inter_set = {id(p) for p in (network_set.projections or [])}
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
            # source_cvar indexes the source model's state variables directly
            # source_cvar entries index state variables directly
            src_cvars_list = [int(v) for v in np.atleast_1d(p.source_cvar)]
            src_cvar_state = src_cvars_list[0]
            n_src_cvars = len(src_cvars_list)
            tgt_cvar = int(np.atleast_1d(p.target_cvar)[0])
            cfun_name = type(p.cfun).__name__ if p.cfun is not None else None
            if cfun_name is None:
                cfun_id, cfun_params, n_cfun_parm = (
                    _PASSTHROUGH_CFUN, np.zeros(0, np.float32), 0)
            else:
                cfun_id = _CFUN_IDS[cfun_name]
                cfun_params = self._cfun_params(p.cfun)
                n_cfun_parm = len(_CFUN_PARAM_ATTRS[cfun_name])
            if is_inter:
                pr_name = (getattr(p, "name", None)
                           or f"{src_name}_to_{tgt_name}")
                lookup = pr_name
            else:
                pr_name = getattr(p, "name", None) or "intra"
                lookup = (f"{tgt_name}.{pr_name}"
                          if pr_name == "intra" else pr_name)
            # target state var for per-edge pre(x_i): model.cvar[target_cvar]
            tgt_state_cvar = int(
                np.asarray(p.target.model.cvar, dtype=np.int32)[tgt_cvar]
            ) if is_inter else 0
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
            projections.append(ProjectionInfo(
                ts=self._target_scales(p),
                target_scales=self._target_scales_raw(p),
                target_cvars_list=[int(v) for v in np.atleast_1d(p.target_cvar)],
                tgt_state_cvar=tgt_state_cvar,
                mode_map=mode_map,
                name=pr_name,
                lookup=lookup,
                is_inter=is_inter,
                src_sn=subnet_index[src_name],
                tgt_sn=subnet_index[tgt_name],
                w=weights_csr.data.astype(np.float32),
                idx=weights_csr.indices.astype(np.uint32),
                ptr=weights_csr.indptr.astype(np.uint32),
                idelays=idelays.astype(np.uint32),
                horizon=horizon,
                cfun_id=cfun_id,
                src_cvar=src_cvar_state,
                src_cvars_list=src_cvars_list,
                n_src_cvars=n_src_cvars,
                tgt_cvar=tgt_cvar,
                n_cfun_parm=n_cfun_parm,
                scale=float(p.scale),
                cfun_params=cfun_params,
            ))

        # per-source-subnet horizon = max outgoing projection horizon
        for pr in projections:
            src = subnets[pr.src_sn]
            src.horizon = max(src.horizon, pr.horizon)

        return NetworkAnalysis(
            subnets=subnets, projections=projections, dt=dt)

    @classmethod
    def _ensure_generic_dfuns(cls):
        """Generate + compile + load the expression-driven model dfuns."""
        global _GEN_LIB, _GEN_IDS
        if _GEN_LIB is not None:
            return
        import ctypes
        from . import dfungen
        lib_path, ids, meta = dfungen.generate_lib(cls.get_cache_dir())
        _GEN_LIB = ctypes.CDLL(str(lib_path))
        _GEN_IDS = ids
        fn = ctypes.cast(_GEN_LIB.cph_generic_dfun, ctypes.c_void_p).value
        import tvb.simulator.backend.cpp_hybrid as _pkg
        _pkg.enable_generic_dfuns(fn)

    def _model_key(self, model):
        """Return (model_id, n_parm, n_cvar, n_svar, parm_names) for a model
        instance, using hand-written kernels first, then generated ones."""
        name = type(model).__name__
        if name in _MODEL_IDS:
            parm = _MODEL_PARM_NAMES[name]
            return _MODEL_IDS[name], len(parm), None, None, parm
        self._ensure_generic_dfuns()
        if name not in _GEN_IDS:
            raise NotImplementedError(
                f"CppHybridBackend does not support {name}"
            )
        import ctypes
        from . import dfungen
        meta = dfungen.generate_meta()
        m = meta[name]
        parm = list(model.global_parameter_names) + list(
            model.spatial_parameter_names)
        assert m["n_parm"] == len(parm), (name, m["n_parm"], parm)
        return _GEN_IDS[name], len(parm), m["n_cvar"], m["n_svar"], parm

    @staticmethod
    def _cfun_params(cfun) -> np.ndarray:
        """(n_parm,) float32 cfun parameter values, ordered like nb_hybrid."""
        attrs = _CFUN_PARAM_ATTRS[type(cfun).__name__]
        params = np.zeros(max(idx for _, idx in attrs) + 1, dtype=np.float32)
        for name, idx in attrs:
            # some cfuns (e.g. Kuramoto.inv_N) are post-scaling conventions
            # already covered by the C++ kernel; default missing attrs to 1.0
            v = getattr(cfun, name, None)
            params[idx] = 1.0 if v is None else float(np.asarray(v).ravel()[0])
        return params

    @staticmethod
    def _params_for_model(model, names) -> np.ndarray:
        """(n_parm,) float32 — handles Zerlaut P_e:/P_i: expansion."""
        vals = []
        for k in names:
            if k.startswith("P_e:") or k.startswith("P_i:"):
                base, _ = k.split(":")
                idx = int(k.split(":")[1])
                arr = np.asarray(getattr(model, base), dtype=np.float32).ravel()
                vals.append(float(arr[idx]))
            else:
                vals.append(float(np.asarray(
                    getattr(model, k), dtype=np.float32).ravel()[0]))
        return np.asarray(vals, dtype=np.float32)

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
                from tvb.simulator.backend import nb_hybrid as nbh
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
            parts.append((sn.model_name, sn.n_nodes, sn.n_modes,
                          sn.n_svar, sn.n_cvar, sn.horizon))
        for pr in analysis.all_projections:
            parts.append((pr.src_sn, pr.tgt_sn, pr.cfun_id,
                          pr.n_tgt_nodes, tuple(pr.src_cvars_list),
                          pr.tgt_cvar))
        return hashlib.md5(repr(parts).encode()).hexdigest()

    def _build_sim(self, sim, analysis, network_set, width):
        """Populate an empty Sim from an analysis (no per-lane params)."""
        for i, sn in enumerate(analysis.subnets):
            sim.add_subnet(
                sn.n_nodes, sn.n_svar, sn.n_parm, sn.n_cvar,
                sn.model_id, sn.horizon, sn.n_modes,
            )
            sim.set_voi_specs(i, sn.voi)
        integ_id = 1 if isinstance(
            network_set.subnets[0].scheme,
            (HeunDeterministic, HeunStochastic)) else 0
        sim.set_opts(integ_id, analysis.dt, 1)
        pix = 0
        for i, pr in enumerate(analysis.projections):
            tsc = pr.target_scales
            srcs_all = pr.src_cvars_list if pr.src_cvars_list else [pr.src_cvar]
            n_src = len(srcs_all)
            split = (tsc is not None and int(tsc.size) > 1
                     and int(tsc.size) == n_src
                     and pr.target_cvars_list and len(pr.target_cvars_list) == n_src)
            channels = list(range(n_src)) if split else [None]
            for ch in channels:
                if ch is not None:
                    srcs = [srcs_all[ch]]
                    tgt = pr.target_cvars_list[ch]
                    tsp = float(tsc[ch])
                else:
                    srcs = srcs_all
                    tgt = pr.tgt_cvar
                    tsp = float(getattr(pr, "ts", 1.0))
                sim.add_projection(
                    pr.src_sn, pr.tgt_sn, pr.w, pr.idx, pr.ptr,
                    np.asarray(pr.idelays, dtype=np.uint32), pr.cfun_id,
                    np.asarray(srcs, dtype=np.int32),
                    tgt, pr.n_cfun_parm, np.float32(pr.scale),
                    np.float32(tsp),
                    int(getattr(pr, "tgt_state_cvar", 0)),
                    pr.mode_map,
                )
                cp = np.broadcast_to(
                    pr.cfun_params[:, None], (max(pr.n_cfun_parm, 0), width)
                ).copy() if pr.n_cfun_parm else np.zeros((0, width), np.float32)
                sim.set_cfun_params(pix, cp)
                pix += 1
        for i, sn in enumerate(analysis.subnets):
            pp = np.broadcast_to(
                sn.params[None, :, None],
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

    def _build_stim_arrays(self, analysis, nstep, width):
        """Per-subnetwork stimulus arrays, layout (nstep, n_cvar, n_nodes, W).

        Stimulus for step t (0-based within the run) is evaluated at global
        step t+1, matching the nb_hybrid convention (stim[t - offset - 1]).
        """
        stims = []
        for sn in analysis.subnets:
            if not sn.stimuli:
                continue
            nm = sn.n_modes
            arr = np.zeros(
                (nstep, sn.n_cvar, sn.n_nodes, nm, width), np.float32)
            for stim in sn.stimuli:
                target_slots = np.asarray(stim.target_cvar).astype(np.intp)
                for step_idx in range(1, nstep + 1):
                    sc = np.asarray(stim.get_coupling(step_idx), dtype=np.float32)
                    if sc.ndim == 2:
                        sc = sc[:, :, np.newaxis]
                    sc = np.broadcast_to(
                        sc, (target_slots.size, sn.n_nodes, nm))
                    for li, slot in enumerate(target_slots):
                        arr[step_idx - 1, slot, :, :, 0] += sc[li]
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

    def _run_compiled(
        self, sim, analysis, network_set, nstep=100, chunk_size=None,
        initial_states=None, monitors=None, width=1, _init_only=False,
    ):
        self._validate_monitors(monitors, chunk_size)
        self._validate_horizons(analysis)
        cs = int(chunk_size) if chunk_size else 1

        # per-chunk stochastic noise generation (Python-side RNG, like nb_hybrid)
        noise_obj = None
        stoch = [sn for sn in analysis.subnets if sn.is_stochastic]
        if stoch:
            noise_std = [
                np.sqrt(2.0 * sn.noise_nsig * analysis.dt) for sn in stoch
            ]
            rngs = []
            sn_objs = {sn.name: s for sn, s in
                       zip(analysis.subnets, network_set.subnets)}
            for sn in stoch:
                rngs.append(sn_objs[sn.name].scheme.noise.random_stream)

        self._init_states(sim, analysis, network_set, initial_states, width)

        stims = self._build_stim_arrays(analysis, int(nstep), width)
        stim_obj = stims[0] if len(stims) == 1 else (stims if stims else None)
        if len(stims) > 1:
            raise NotImplementedError(
                "stimuli on multiple subnetworks not yet supported"
            )

        if not stoch:
            outs = sim.run(int(nstep), cs, None, stim_obj)
        else:
            # chunk-wise loop to draw noise per chunk (sequential RNG stream)
            all_outs = None
            remaining = int(nstep)
            chunk_results = []
            while remaining > 0:
                this_chunk = min(remaining, max(cs, 1))
                noises = []
                for sn, std, rng in zip(stoch, noise_std, rngs):
                    dw = rng.randn(this_chunk, sn.n_svar, sn.n_nodes,
                                   sn.n_modes)
                    dw = dw * std[np.newaxis, :, np.newaxis, np.newaxis]
                    # layout (step, svar, node, mode, lane) for the core
                    nz = np.ascontiguousarray(
                        np.repeat(dw[..., np.newaxis], width,
                                  axis=4)).astype(np.float32)
                    noises.append(nz)
                outs = sim.run(this_chunk, 0,
                               noises[0] if len(noises) == 1 else noises,
                               None) if len(stoch) == 1 else None
                if len(stoch) != 1:
                    raise NotImplementedError(
                        "multiple stochastic subnetworks not yet supported"
                    )
                chunk_results.append(outs)
                remaining -= this_chunk
            # concatenate chunk outputs along the chunk axis
            n_out = len(chunk_results[0])
            all_outs = [
                np.concatenate([cr[i] for cr in chunk_results], axis=0)
                for i in range(n_out)
            ]
            outs = all_outs

        results = []
        for i, sn in enumerate(analysis.subnets):
            tavg = np.asarray(outs[2 * i], dtype=np.float32)
            ctavg = np.asarray(outs[2 * i + 1], dtype=np.float32)
            n_chunks = tavg.shape[0]
            times = (np.arange(n_chunks) + 0.5) * cs * analysis.dt
            results.append((times, tavg, ctavg))
        return results

    def run_network(
        self,
        network_set: NetworkSet,
        nstep: int,
        chunk_size: int = None,
        print_source: bool = False,
        initial_states: Optional[list] = None,
        debug_nojit: bool = False,
        monitors: Optional[list] = None,
    ):
        """Run a hybrid simulation with the C++ backend.

        Equivalent to ``self.compile(network_set).run(nstep, ...)``.
        """
        cn = self.compile(network_set, print_source=print_source)
        return cn.run(nstep, chunk_size, initial_states, monitors)

    # ------------------------------------------------------------------
    # sweeps: multithreaded SIMD batches (lanes = sweep members)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_sweep(network_set, params):
        """Resolve a named-parameter dict like nb_hybrid._resolve_named_params.

        Returns (targets, sweep_values) where targets is a list of
        ('cfun', projection_name, param_idx) or ('model', subnet_name, parm).
        """
        descriptor, sweep_values = NbHybridBackend._resolve_named_params(
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
        for d, tgt in enumerate(targets):
            vals = group_values[:, d]  # (group_size,)
            if tgt[0] == "cfun":
                _, pname, pidx = tgt
                pi = next(
                    i for i, pr in enumerate(analysis.projections)
                    if pr.name == pname or pr.lookup == pname
                    or pr.lookup.endswith("." + pname)
                )
                pr = analysis.projections[pi]
                cp = np.broadcast_to(
                    pr.cfun_params[:, None], (pr.n_cfun_parm, width)).copy()
                cp[pidx, :group_size] = vals
                sim.set_cfun_params(pi, cp)
            else:
                _, sname, attr = tgt
                si = next(i for i, sn in enumerate(analysis.subnets)
                          if sn.name == sname)
                sn = analysis.subnets[si]
                k = sn.parm_index[attr]
                pp = np.broadcast_to(
                    sn.params[None, :, None],
                    (sn.n_nodes, sn.n_parm, width),
                ).copy()
                pp[:, k, :group_size] = vals[None, :]
                sim.set_subnet_params(si, pp)

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
        width: int = 8,
        progress: bool = False,
    ) -> "SweepResult":
        """Run a parameter sweep as multithreaded SIMD batches.

        Each SIMD lane (width, default 8) carries one sweep member; lane
        groups run concurrently in Python threads (the C++ kernel releases
        the GIL).

        Parameters mirror :meth:`NbHybridBackend.sweep`.
        """
        import time as _time_mod
        from concurrent.futures import ThreadPoolExecutor

        if monitor != "tavg":
            raise NotImplementedError(
                "CppHybridBackend.sweep currently supports monitor='tavg'"
            )
        if bold_period is not None:
            raise NotImplementedError(
                "CppHybridBackend.sweep does not support bold_period yet"
            )
        if not isinstance(nstep, (int, np.integer)) or nstep <= 0:
            raise ValueError("nstep must be a positive integer")

        targets, sweep_values = self._resolve_sweep(network_set, params)
        analysis = self._analyse(network_set)
        results, workers, elapsed = self._sweep_impl(
            network_set, analysis, targets, sweep_values, nstep,
            initial_states, width, n_workers, _time_mod,
            label=f"cpp-simd(width={width})", progress=progress,
        )

        # assemble per-subnet tavg over all sweeps
        n_sweeps = sweep_values.shape[0]
        tavg = {sn.name: [] for sn in analysis.subnets}
        ctavg = {sn.name: [] for sn in analysis.subnets}
        for lo, hi, outs in results:
            for i, sn in enumerate(analysis.subnets):
                arr = np.asarray(outs[2 * i], dtype=np.float32)
                ct = np.asarray(outs[2 * i + 1], dtype=np.float32)
                tavg[sn.name].append(
                    arr.transpose(3, 0, 1, 2)[: hi - lo, :, :, :, None])
                ctavg[sn.name].append(
                    ct.transpose(3, 0, 1, 2)[: hi - lo, :, :, :, None])
        for name in tavg:
            tavg[name] = np.concatenate(tavg[name], axis=0)
            ctavg[name] = np.concatenate(ctavg[name], axis=0)
        merged = np.concatenate(
            [tavg[sn.name] for sn in analysis.subnets], axis=3
        ) if len(analysis.subnets) > 1 else tavg[analysis.subnets[0].name]
        n_samples = tavg[analysis.subnets[0].name].shape[1]
        times = (np.arange(n_samples) + 0.5) * analysis.dt
        return SweepResult(
            tavg=tavg,
            merged_tavg=merged,
            ctavg=ctavg,
            times=times,
            sweep_values=sweep_values,
            backend=f"cpp-simd(width={width},workers={workers})",
            elapsed=elapsed,
            param_keys=(list(params.keys()) if isinstance(params, dict)
                        else None),
        )

    def _sweep_impl(self, network_set, analysis, targets, sweep_values, nstep,
                    initial_states, width, n_workers, _time_mod, label,
                    progress=False):
        from concurrent.futures import ThreadPoolExecutor

        n_sweeps = sweep_values.shape[0]
        n_groups = (n_sweeps + width - 1) // width
        t0 = _time_mod.perf_counter()

        def run_group(gi):
            lo = gi * width
            hi = min(lo + width, n_sweeps)
            group_vals = np.zeros((width, sweep_values.shape[1]), np.float32)
            group_vals[: hi - lo] = sweep_values[lo:hi]
            sim = Sim(width)
            self._build_sim(sim, analysis, network_set, width)
            self._apply_sweep_group(sim, analysis, targets, group_vals, width)
            self._init_states(sim, analysis, network_set, initial_states, width)
            outs = sim.run(int(nstep), 0, None, None)
            return lo, hi, outs

        workers = n_workers or min(n_groups, max(1, _default_workers()))
        results = [None] * n_groups
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(run_group, gi) for gi in range(n_groups)]
            it = futures
            if progress:
                try:
                    import tqdm.auto
                    it = tqdm.auto.tqdm(
                        futures, total=n_groups, desc=label, unit="lane-group")
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
        **monitors,
    ):
        """Legacy nb_hybrid-style sweep entry point."""
        import time as _time_mod

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
        results, workers, elapsed = self._sweep_impl(
            network_set, analysis, targets, sweep_values, nstep,
            initial_states, 8, None, _time_mod, "cpp-simd"
        )
        n_sweeps = sweep_values.shape[0]
        width = 8
        out = []
        for tid in range(n_sweeps):
            gi = tid // width
            lane = tid % width
            lo, hi, outs = results[gi]
            per_sn = []
            for i, sn in enumerate(analysis.subnets):
                arr = np.asarray(outs[2 * i], dtype=np.float32)
                ct = np.asarray(outs[2 * i + 1], dtype=np.float32)
                data = arr[:, :, :, lane:lane + 1]
                ctavg = ct[:, :, :, lane:lane + 1]
                times = (np.arange(data.shape[0]) + 0.5) * analysis.dt
                per_sn.append((times, data, ctavg))
            out.append(per_sn)
        return out


def _default_workers():
    import os
    return max(1, (os.cpu_count() or 2) // 2)
