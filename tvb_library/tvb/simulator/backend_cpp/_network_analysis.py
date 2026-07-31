"""Network analysis for the C++ backend.

Self-contained — no dependency on NbHybridBackend or nb_hybrid internals.
Reads only public attributes of NetworkSet, Subnetwork, and projection objects.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from tvb.simulator.integrators import EulerStochastic, HeunStochastic


# ---------------------------------------------------------------------------
# Local dataclasses — equivalent to SubnetworkInfo / ProjectionInfo /
# NetworkAnalysis in nb_hybrid, but owned by the C++ backend.
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _SubnetInfo:
    name: str
    model: object
    integrator: object
    n_nodes: int
    n_modes: int
    is_stochastic: bool = False
    noise_nsig: Optional[np.ndarray] = None
    has_stimulus: bool = False


@dataclasses.dataclass
class _ProjInfo:
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
    cfun: object
    is_inter: bool
    mode_map: Optional[np.ndarray] = None

    @property
    def n_src_modes(self) -> int:
        if self.is_inter:
            return self.mode_map.shape[0]
        return self._n_src_modes

    @n_src_modes.setter
    def n_src_modes(self, v: int) -> None:
        self._n_src_modes = v

    @property
    def n_tgt_modes(self) -> int:
        if self.is_inter:
            return self.mode_map.shape[1]
        return self._n_src_modes


@dataclasses.dataclass
class _NetworkAnalysis:
    subnetworks: list
    inter_projections: list
    intra_projections: list
    stimuli_by_subnet: dict = dataclasses.field(default_factory=dict)
    source_horizons: dict = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Coupling-function helpers
# ---------------------------------------------------------------------------

# Integer codes must match the kCfun* constants in runtime/runtime.hpp
_CFUN_TYPE_TO_INT: dict[str, int] = {
    "none":            0,
    "linear":          1,
    "scaling":         2,
    "sigmoidal":       3,
    "sigmoidal_jr":    4,   # legacy single-cvar
    "tanh":            5,
    "pre_sigmoidal":   6,   # static threshold single-cvar
    "kuramoto":        7,
    "sigmoidal_jr_2":  8,   # classic 2-cvar
    "pre_sigmoidal_2": 9,   # dynamic threshold 2-cvar
}


def _cfun_type(p: _ProjInfo) -> str:
    from tvb.simulator.hybrid.coupling import (
        Linear, Scaling, Sigmoidal, SigmoidalJansenRit,
        Kuramoto as KuramotoCfun, Difference, HyperbolicTangent, PreSigmoidal,
    )
    if p.cfun is None:
        return "none"
    if isinstance(p.cfun, Linear):
        return "linear"
    if isinstance(p.cfun, Scaling):
        return "scaling"
    if isinstance(p.cfun, Sigmoidal):
        return "sigmoidal"
    if isinstance(p.cfun, SigmoidalJansenRit):
        # 2 source cvars → classic mode (cmin/cmax/midpoint formula)
        if p.source_cvar.shape[0] == 2:
            return "sigmoidal_jr_2"
        return "sigmoidal_jr"  # legacy single-cvar (e0/v0 formula)
    if isinstance(p.cfun, KuramotoCfun):
        return "kuramoto"
    if isinstance(p.cfun, Difference):
        return "scaling"
    if isinstance(p.cfun, HyperbolicTangent):
        return "tanh"
    if isinstance(p.cfun, PreSigmoidal):
        # 2 source cvars → dynamic threshold mode
        if p.source_cvar.shape[0] == 2:
            return "pre_sigmoidal_2"
        return "pre_sigmoidal"  # static threshold single-cvar
    return "none"


def _cfun_type_int(p: _ProjInfo) -> int:
    """Return integer coupling function type code for C++ dispatch."""
    return _CFUN_TYPE_TO_INT.get(_cfun_type(p), 0)


def _cfun_params(p: _ProjInfo) -> np.ndarray:
    """Return float32[8] with cfun parameters.

    Layout:
      none:          [1.0, 0, ...]
      linear:        [a, b, 0, ...]
      scaling:       [a, 0, ...]
      sigmoidal:     [a, sigma, midpoint, cmin, cmax, 0, ...]
      sigmoidal_jr:  [a, e0, r, v0, 0, ...]
      kuramoto:      [a, 0, ...]
      tanh:          [a, midpoint, sigma, 0, ...]
      pre_sigmoidal: [H, Q, G, P, theta, 0, ...]
    """
    from tvb.simulator.hybrid.coupling import (
        Linear, Scaling, Sigmoidal, SigmoidalJansenRit,
        Kuramoto as KuramotoCfun, Difference, HyperbolicTangent, PreSigmoidal,
    )
    arr = np.zeros(8, dtype=np.float32)
    arr[0] = 1.0
    if p.cfun is None:
        return arr
    if isinstance(p.cfun, Linear):
        arr[0] = float(p.cfun.a[0]); arr[1] = float(p.cfun.b[0])
    elif isinstance(p.cfun, (Scaling, Difference, KuramotoCfun)):
        arr[0] = float(p.cfun.a[0])
    elif isinstance(p.cfun, Sigmoidal):
        arr[0] = float(p.cfun.a[0]); arr[1] = float(p.cfun.sigma[0])
        arr[2] = float(p.cfun.midpoint[0]); arr[3] = float(p.cfun.cmin[0])
        arr[4] = float(p.cfun.cmax[0])
    elif isinstance(p.cfun, SigmoidalJansenRit):
        if p.source_cvar.shape[0] == 2:
            # classic 2-cvar: [a, r, cmin, cmax, midpoint]
            arr[0] = float(p.cfun.a[0]); arr[1] = float(p.cfun.r[0])
            arr[2] = float(p.cfun.cmin[0]); arr[3] = float(p.cfun.cmax[0])
            arr[4] = float(p.cfun.midpoint[0])
        else:
            # legacy single-cvar: [a, e0, r, v0]
            arr[0] = float(p.cfun.a[0]); arr[1] = float(p.cfun.e0[0])
            arr[2] = float(p.cfun.r[0]); arr[3] = float(p.cfun.v0[0])
    elif isinstance(p.cfun, HyperbolicTangent):
        arr[0] = float(p.cfun.a[0]); arr[1] = float(p.cfun.midpoint[0])
        arr[2] = float(p.cfun.sigma[0]); arr[3] = float(p.cfun.b[0])
    elif isinstance(p.cfun, PreSigmoidal):
        arr[0] = float(p.cfun.H[0]); arr[1] = float(p.cfun.Q[0])
        arr[2] = float(p.cfun.G[0]); arr[3] = float(p.cfun.P[0])
        arr[4] = float(p.cfun.theta[0])
    return arr


def _cvar_mapping_mode(p: _ProjInfo) -> str:
    ns, nt = p.source_cvar.shape[0], p.target_cvar.shape[0]
    if ns == 1 and nt == 1:
        return "1_to_1"
    if nt == 1:
        return "many_to_1"
    if ns == 1:
        return "1_to_many"
    if ns == nt:
        return "n_to_n"
    raise ValueError(
        f"Projection '{p.name}': unsupported cvar mapping ({ns} source → {nt} target)"
    )


# ---------------------------------------------------------------------------
# Network analysis
# ---------------------------------------------------------------------------

def _build_projection_info(p, *, is_inter: bool) -> _ProjInfo:
    ts = p.target_scales if p.target_scales is not None else np.zeros(0, dtype=np.float64)

    if is_inter:
        src_name = p.source.name
        tgt_name = p.target.name
        n_src_modes = p.source.model.number_of_modes
        n_tgt_modes = p.target.model.number_of_modes
        mode_map = (
            p.mode_map.astype(np.float32)
            if p.mode_map is not None
            else np.ones((n_src_modes, n_tgt_modes), dtype=np.float32)
        )
        proj_name = f"{src_name}_to_{tgt_name}"
    else:
        src_name = ""
        tgt_name = ""
        mode_map = None
        proj_name = getattr(p, "name", None) or "intra"

    weights_csr = p.weights.copy()
    idelays_raw = np.atleast_1d(p.idelays)
    nz_mask = weights_csr.data != 0
    weights_csr.eliminate_zeros()
    idelays_stripped = idelays_raw[nz_mask].astype(np.int32)

    pi = _ProjInfo(
        name=proj_name,
        source_subnet=src_name,
        target_subnet=tgt_name,
        source_cvar=np.atleast_1d(p.source_cvar).astype(np.int32),
        target_cvar=np.atleast_1d(p.target_cvar).astype(np.int32),
        weights_data=weights_csr.data.astype(np.float32),
        weights_indices=weights_csr.indices.astype(np.int32),
        weights_indptr=weights_csr.indptr.astype(np.int32),
        idelays=idelays_stripped,
        horizon=int(p._horizon),
        scale=float(p.scale),
        target_scales=(
            np.atleast_1d(ts).astype(np.float32)
            if np.atleast_1d(ts).size > 0
            else np.zeros(0, dtype=np.float32)
        ),
        cfun=p.cfun,
        is_inter=is_inter,
        mode_map=mode_map,
    )
    if not is_inter:
        pi.n_src_modes = 1  # filled per-subnetwork by caller
    return pi


def analyse_network(network_set) -> _NetworkAnalysis:
    """Analyse a configured NetworkSet and return a backend-neutral analysis.

    Reads only public attributes of NetworkSet, Subnetwork, and projection
    objects. No dependency on NbHybridBackend.
    """
    from tvb.simulator.noise import Additive
    from tvb.simulator.hybrid import IntraProjection

    stims_by_subnet: dict = {sn.name: [] for sn in network_set.subnets}
    for sn in network_set.subnets:
        for stim in (sn.stimuli or []):
            stims_by_subnet[sn.name].append(stim)

    subnets = []
    for sn in network_set.subnets:
        is_stoch = isinstance(sn.scheme, (EulerStochastic, HeunStochastic))
        noise_nsig = None
        if is_stoch:
            noise_obj = sn.scheme.noise
            if isinstance(noise_obj, Additive):
                nsig = noise_obj.nsig
                noise_nsig = (
                    np.full(sn.model.nvar, float(nsig), dtype=np.float64)
                    if nsig.ndim == 0
                    else np.broadcast_to(nsig, (sn.model.nvar,)).copy().astype(np.float64)
                )
            else:
                raise NotImplementedError(
                    f"Subnetwork '{sn.name}': only tvb.simulator.noise.Additive is "
                    f"supported; got {type(noise_obj).__name__}."
                )
        subnets.append(_SubnetInfo(
            name=sn.name,
            model=sn.model,
            integrator=sn.scheme,
            n_nodes=sn.nnodes,
            n_modes=sn.model.number_of_modes,
            is_stochastic=is_stoch,
            noise_nsig=noise_nsig,
            has_stimulus=bool(stims_by_subnet[sn.name]),
        ))

    inter_projs = []
    for p in network_set.projections:
        if isinstance(p, IntraProjection):
            continue
        inter_projs.append(_build_projection_info(p, is_inter=True))

    intra_projs = []
    for sn_obj in network_set.subnets:
        for p in sn_obj.projections:
            pi = _build_projection_info(p, is_inter=False)
            pi.source_subnet = sn_obj.name
            pi.target_subnet = sn_obj.name
            pi.n_src_modes = sn_obj.model.number_of_modes
            intra_projs.append(pi)

    # Deduplicate projection names.
    seen: dict = {}
    for p in inter_projs + intra_projs:
        base = p.name
        if base in seen:
            seen[base] += 1
            p.name = f"{base}_{seen[base]}"
        else:
            seen[base] = 0

    # Per-source-subnet max horizon for history buffer sizing.
    source_horizons: dict = {}
    for p in inter_projs + intra_projs:
        src = p.source_subnet
        source_horizons[src] = max(source_horizons.get(src, 1), p.horizon)
    for sn in subnets:
        source_horizons.setdefault(sn.name, 1)

    return _NetworkAnalysis(
        subnetworks=subnets,
        inter_projections=inter_projs,
        intra_projections=intra_projs,
        stimuli_by_subnet=stims_by_subnet,
        source_horizons=source_horizons,
    )
