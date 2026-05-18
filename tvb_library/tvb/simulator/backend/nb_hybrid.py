# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
Numba backend for Hybrid Simulator.

Generates and JIT-compiles a simulation kernel for NetworkSet objects
whose subnetworks all use the MontbrioPazoRoxin model and deterministic
integrators (Heun or Euler).

Usage::

    from tvb.simulator.backend.nb_hybrid import NbHybridBackend

    backend = NbHybridBackend()
    results = backend.run_network(network_set, nstep=1000)
    # results: list of (times, data) tuples, one per subnetwork

.. moduleauthor:: TVB contributors

Design and implementation plan: ``nb_hybrid_plan.md`` (same directory).
"""

from __future__ import annotations

import dataclasses
import hashlib
import numpy as np
import autopep8
from typing import List, Optional

from .templates import MakoUtilMix
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.hybrid.intra_projection import IntraProjection
from tvb.simulator.integrators import (
    HeunDeterministic,
    EulerDeterministic,
    HeunStochastic,
    EulerStochastic,
)




__all__ = [
    "NbHybridBackend",
    "CompiledNetworkFn",
    "NetworkAnalysis",
    "SubnetworkInfo",
    "ProjectionInfo",
    "SweepResult",
]


# ---------------------------------------------------------------------------
# SweepResult — unified return type for both CPU and GPU sweep
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SweepResult:
    """Container for parameter-sweep results.

    Returned by :meth:`NbHybridBackend.sweep`.  Works identically
    regardless of whether the sweep ran on CPU or GPU.

    Attributes
    ----------
    tavg : dict[str, np.ndarray]
        Per-subnet temporal-average arrays with shape
        ``(n_sweeps, n_voi, N_subnet, n_modes)``.
    merged_tavg : np.ndarray
        All subnets concatenated / reordered along the node axis,
        shape ``(n_sweeps, n_voi, N_total, n_modes)``.
    ctavg : dict[str, np.ndarray]
        Per-subnet coupling temporal-average (same shape as tavg).
    times : np.ndarray
        Mid-point time vector, shape ``(n_chunks,)``.
    sweep_values : np.ndarray
        The sweep parameter grid, ``(n_sweeps, n_dims)``.
    snapshot : dict or None
        Final execution state for resume (GPU-only for now).
    backend : str
        ``'cpu-seq'``, ``'cpu-4c'``, ``'cpu-8c'``, etc., or ``'cuda'``.
    elapsed : float
        Wall-clock seconds for the sweep execution (excludes compile).
    raw : dict[str, np.ndarray] or None
        Full step-by-step output when ``monitor='raw'``.
    bold : dict[str, np.ndarray] or None
        BOLD signal when ``bold_period`` was given.
    """
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


# ---------------------------------------------------------------------------
# CFun parameter index mapping — named attributes to param_idx
# ---------------------------------------------------------------------------

_CFUN_PARAM_ATTRS: dict = {
    "Linear":            [("a", 0), ("b", 1)],
    "Scaling":           [("a", 0)],
    "Sigmoidal":         [("a", 0), ("sigma", 1), ("midpoint", 2),
                          ("cmin", 3), ("cmax", 4)],
    "SigmoidalJansenRit":[("a", 0), ("e0", 1), ("r", 2), ("v0", 3),
                          ("cmin", 4), ("cmax", 5), ("midpoint", 6)],
    "Kuramoto":          [("a", 0), ("inv_N", 1)],
    "Difference":        [("a", 0)],
    "HyperbolicTangent": [("a", 0), ("midpoint", 1), ("sigma", 2),
                          ("b", 3)],
    "PreSigmoidal":      [("H", 0), ("Q", 1), ("G", 2), ("P", 3), ("theta", 4)],
}

_CFUN_ATTR_TO_IDX: dict = {}
for _cls, _attrs in _CFUN_PARAM_ATTRS.items():
    for _attr, _idx in _attrs:
        _CFUN_ATTR_TO_IDX[(_cls, _attr)] = _idx

_NAMED_PARAM_ALIASES: dict = {
    "coupling_scale": {"attr": "a", "idx": 0},
    "scale":         {"attr": "a", "idx": 0},
    "coupling_a":    {"attr": "a", "idx": 0},
    "coupling_b":    {"attr": "b", "idx": 1},
    "sigma":         {"attr": "sigma", "idx": 1},
    "midpoint":      {"attr": "midpoint", "idx": 2},
}


# ---------------------------------------------------------------------------
# Lazy-supported-models cache
# ---------------------------------------------------------------------------
# Importing all 27 model modules costs ~2.5 s on first call (each module
# registers Numba dfun_helpers).  We cache the result so the cost is paid
# once per process rather than on every compile().

_SUPPORTED_MODELS_CACHE: tuple = ()


def _get_supported_models_classes() -> tuple:
    """Import all supported model classes and return them as a tuple.

    The imports are expensive (~2.5 s total) so this function caches the
    result in the module-level ``_SUPPORTED_MODELS_CACHE``.  Subsequent
    calls return the cached tuple instantly.
    """
    global _SUPPORTED_MODELS_CACHE
    if _SUPPORTED_MODELS_CACHE:
        return _SUPPORTED_MODELS_CACHE
    from tvb.simulator.models.infinite_theta import (
        MontbrioPazoRoxin,
        CoombesByrne2D,
        CoombesByrne,
        GastSchmidtKnosche_SD,
        GastSchmidtKnosche_SF,
        DumontGutkin,
    )
    from tvb.simulator.models.k_ion_exchange import KIonEx
    from tvb.simulator.models.jansen_rit import JansenRit, ZetterbergJansen
    from tvb.simulator.models.oscillator import (
        Generic2dOscillator,
        SupHopf,
        Kuramoto,
    )
    from tvb.simulator.models.wong_wang import ReducedWongWang
    from tvb.simulator.models.wong_wang_exc_inh import ReducedWongWangExcInh
    from tvb.simulator.models.epileptor import Epileptor, Epileptor2D
    from tvb.simulator.models.epileptorcodim3 import (
        EpileptorCodim3,
        EpileptorCodim3SlowMod,
    )
    from tvb.simulator.models.epileptor_rs import EpileptorRestingState
    from tvb.simulator.models.hopfield import Hopfield
    from tvb.simulator.models.larter_breakspear import LarterBreakspear
    from tvb.simulator.models.wilson_cowan import WilsonCowan
    from tvb.simulator.models.zerlaut import ZerlautAdaptationFirstOrder
    from tvb.simulator.models.stefanescu_jirsa import (
        ReducedSetFitzHughNagumo,
        ReducedSetHindmarshRose,
    )
    from tvb.simulator.models.linear import Linear
    _SUPPORTED_MODELS_CACHE = (
        MontbrioPazoRoxin,
        KIonEx,
        JansenRit,
        Generic2dOscillator,
        ReducedWongWang,
        ReducedWongWangExcInh,
        Epileptor,
        Epileptor2D,
        EpileptorCodim3,
        EpileptorCodim3SlowMod,
        EpileptorRestingState,
        WilsonCowan,
        ZerlautAdaptationFirstOrder,  # ZerlautSecondOrder is a subclass
        SupHopf,
        Kuramoto,
        Hopfield,
        LarterBreakspear,
        CoombesByrne2D,
        CoombesByrne,
        GastSchmidtKnosche_SD,
        GastSchmidtKnosche_SF,
        DumontGutkin,
        ZetterbergJansen,
        ReducedSetFitzHughNagumo,
        ReducedSetHindmarshRose,
        Linear,
    )
    return _SUPPORTED_MODELS_CACHE

# Alias for readability
_get_supported_models = _get_supported_models_classes


# Lazy cache for ReducedSetBase (needed by _check_compatibility)
_REDUCED_SET_BASE_CACHE = None


def _get_reduced_set_base():
    """Lazily import ReducedSetBase and cache it."""
    global _REDUCED_SET_BASE_CACHE
    if _REDUCED_SET_BASE_CACHE is not None:
        return _REDUCED_SET_BASE_CACHE
    from tvb.simulator.models.stefanescu_jirsa import ReducedSetBase
    _REDUCED_SET_BASE_CACHE = ReducedSetBase
    return _REDUCED_SET_BASE_CACHE


# ---------------------------------------------------------------------------
# Helpers used by both Python (NbHybridBackend) and Mako templates
# ---------------------------------------------------------------------------


def _compute_chunk_size(monitors, dt):
    """Compute the optimal chunk_size from monitor periods.

    For each monitor with a period, the number of integration steps per sample
    is ``istep = round(period / dt)``.  The chunk_size must divide evenly into
    every monitor's istep so that temporal averages align with sampling periods.

    Returns the GCD of all monitor isteps.  Returns 1 when no monitors have
    a meaningful period (e.g. only Raw).

    Parameters
    ----------
    monitors : list
        TVB monitor instances.
    dt : float
        Integration time step.

    Returns
    -------
    int
        The computed chunk_size.
    """
    import math
    from tvb.simulator.monitors import Raw, AfferentCoupling, RawVoi

    isteps = []
    for m in monitors:
        # Raw and base AfferentCoupling output every step — no period constraint.
        # AfferentCouplingTemporalAverage (subclass) HAS a period, so check it first.
        from tvb.simulator.monitors import AfferentCouplingTemporalAverage
        if isinstance(m, AfferentCouplingTemporalAverage):
            if hasattr(m, 'period') and m.period is not None:
                istep = max(1, int(round(float(m.period) / dt)))
                isteps.append(istep)
            continue
        if isinstance(m, (Raw, AfferentCoupling, RawVoi)):
            continue
        if hasattr(m, 'period') and m.period is not None:
            istep = max(1, int(round(float(m.period) / dt)))
            isteps.append(istep)
    if not isteps:
        return 1
    result = isteps[0]
    for s in isteps[1:]:
        result = math.gcd(result, s)
    return max(1, result)


def _aggregate_chunks_to_period(times, data, period, dt, chunk_size):
    """Aggregate per-chunk data into period-sized windows.

    When chunk_size < monitor period (in steps), the JIT produces more chunks
    than monitor samples. This function groups consecutive chunks into windows
    of `istep / chunk_size` chunks each, averages each window, and returns
    the downsampled result with midpoint times.

    Parameters
    ----------
    times : ndarray (n_chunks,)
        Per-chunk midpoint times.
    data : ndarray (n_chunks, ...)
        Per-chunk averaged data.
    period : float
        Monitor sampling period in ms.
    dt : float
        Integration time step in ms.
    chunk_size : int
        Number of integration steps per chunk.

    Returns
    -------
    new_times : ndarray (n_samples,)
    new_data : ndarray (n_samples, ...)
    """
    istep = max(1, int(round(period / dt)))
    chunks_per_period = max(1, istep // chunk_size)
    n_chunks = len(times)
    n_periods = n_chunks // chunks_per_period
    if n_periods == 0:
        # Not enough chunks for one full period — return per-chunk data as-is
        return times, data
    # Reshape and average over chunks_per_period
    truncated = n_periods * chunks_per_period
    reshaped = data[:truncated].reshape(n_periods, chunks_per_period, *data.shape[1:])
    avg_data = reshaped.mean(axis=1)
    # Compute period midpoint times: first period starts at step 0
    new_times = np.array([
        (i * chunks_per_period * chunk_size + istep / 2.0) * dt
        for i in range(n_periods)
    ], dtype=np.float64)
    return new_times, avg_data


def _apply_monitors(
    raw_outputs: list,
    monitors: list,
    dt: float,
    chunk_size: int = 1,
    monitor_data: Optional[dict] = None,
    subnet_infos: Optional[list] = None,
) -> list:
    """Transform per-subnet (times, data, ctavg) tuples into monitor-dispatched output.

    Parameters
    ----------
    raw_outputs : list of (times, data, ctavg)
        One tuple per subnetwork, as returned by the compiled kernel.
    monitors : list of Monitor instances
        Each monitor determines which view / transform of the raw data to return.
    dt : float
        Integration time step (ms).

    Returns
    -------
    list[list[tuple[ndarray, ndarray]]]
        Outer list indexed by monitor, inner list by subnetwork.
        Each element is (times, data).
    """
    from tvb.simulator.monitors import (
        TemporalAverage,
        Raw,
        SubSample,
        GlobalAverage,
        AfferentCoupling,
        SpatialAverage,
        Projection,
        Bold,
    )

    # Cache merge check — invariant for the entire call
    should_merge = bool(subnet_infos) and _can_merge_subnets(subnet_infos)

    for m in monitors:
        if isinstance(m, Raw):
            pass
        elif isinstance(
            m,
            (TemporalAverage, SubSample, GlobalAverage, AfferentCoupling,
             SpatialAverage),
        ):
            pass
        elif isinstance(m, Projection):
            if not hasattr(m, '_gain') or m._gain is None:
                raise ValueError(
                    f"Projection monitor {type(m).__name__} has no gain matrix "
                    "configured. Set the gain matrix before running (e.g. via "
                    "config_for_sim or by setting m._gain directly)."
                )
        elif isinstance(m, Bold):
            # Bold: ensure dt and istep are set (used by stock allocation)
            if not hasattr(m, 'istep') or m.istep is None:
                m.dt = dt
                m._config_dt(dt)
        else:
            raise NotImplementedError(
                f"Monitor {type(m).__name__} is not yet supported by the Numba backend. "
                "Supported: TemporalAverage, Raw, SubSample, GlobalAverage, "
                "AfferentCoupling, SpatialAverage, Projection (EEG/MEG/iEEG), Bold."
            )

    # Extract pre-computed JIT monitor data if available
    spatial_per_sn = monitor_data.get('spatial', []) if monitor_data else []
    proj_per_sn = monitor_data.get('proj', []) if monitor_data else []
    bold_per_sn = monitor_data.get('bold', []) if monitor_data else []

    results: list = []
    for m in monitors:
        per_subnet: list = []
        for si, (times, data, ctavg) in enumerate(raw_outputs):
            if isinstance(m, AfferentCoupling):
                per_subnet.append((times, ctavg))
            elif isinstance(m, Projection):
                if should_merge:
                    # In merged mode, each subnet has fewer nodes than the
                    # full gain matrix. Defer projection to the merge step,
                    # which operates on connectome-ordered data.
                    per_subnet.append((times, data))
                else:
                    # Non-merged: project this subnet's data
                    if si < len(proj_per_sn) and proj_per_sn[si].ndim == 4 and proj_per_sn[si].shape[1] > 0:
                        proj_data = proj_per_sn[si]
                    else:
                        gain = m.gain.astype(data.dtype)
                        data_2d = data.sum(axis=-1)
                        projected = np.einsum('ij,tkj->tki', gain, data_2d)
                        proj_data = projected[..., np.newaxis]
                    # Aggregate chunks to match monitor period
                    if chunk_size > 0 and hasattr(m, 'period'):
                        istep = max(1, int(round(float(m.period) / dt)))
                        if chunk_size < istep:
                            t, d = _aggregate_chunks_to_period(times, proj_data, float(m.period), dt, chunk_size)
                            per_subnet.append((t, d))
                            continue
                    per_subnet.append((times, proj_data))
            elif isinstance(m, GlobalAverage):
                # In merged mode, defer averaging until after merge
                if should_merge:
                    per_subnet.append((times, data))
                else:
                    per_subnet.append((times, data.mean(axis=-2, keepdims=True)))
            elif isinstance(m, Bold):
                # Use JIT-computed Balloon model BOLD signal
                if si < len(bold_per_sn):
                    bold_times, bold_data = bold_per_sn[si]
                    if bold_data.ndim >= 3 and bold_data.shape[0] > 0:
                        per_subnet.append((bold_times, bold_data))
                    else:
                        n_chunks, n_voi, n_nodes, n_modes = data.shape
                        per_subnet.append((
                            np.array([], dtype=np.float64),
                            np.empty((0, n_voi, n_nodes, n_modes), dtype=np.float32),
                        ))
                else:
                    n_chunks, n_voi, n_nodes, n_modes = data.shape
                    per_subnet.append((
                        np.array([], dtype=np.float64),
                        np.empty((0, n_voi, n_nodes, n_modes), dtype=np.float32),
                    ))
            elif isinstance(m, SpatialAverage):
                # In merged mode, spatial_mean covers all connectome nodes,
                # so per-subnet JIT spatial is wrong. Use merged data instead.
                if should_merge and hasattr(m, 'spatial_mean'):
                    # Defer: store raw data, compute spatial after merge
                    per_subnet.append((times, data))
                else:
                    # Use JIT-precomputed if available
                    if si < len(spatial_per_sn) and spatial_per_sn[si].ndim == 4 and spatial_per_sn[si].shape[2] > 0:
                        sa_data = spatial_per_sn[si]
                    elif hasattr(m, 'spatial_mean'):
                        sa_data = np.einsum('ij,tkjm->tkim', m.spatial_mean, data)
                    else:
                        sa_data = data
                    # Aggregate chunks to match monitor period
                    if chunk_size > 0 and hasattr(m, 'period'):
                        istep = max(1, int(round(float(m.period) / dt)))
                        if chunk_size < istep:
                            t, d = _aggregate_chunks_to_period(times, sa_data, float(m.period), dt, chunk_size)
                            per_subnet.append((t, d))
                            continue
                    per_subnet.append((times, sa_data))
            elif isinstance(m, SubSample):
                period = float(m.period)
                istep = max(1, int(round(period / dt)))
                # Step-based selection (1-indexed) to match Python monitor semantics
                # chunk i corresponds to step (offset + i + 1)
                n_chunks = len(times)
                step_numbers = np.arange(1, n_chunks + 1)
                mask = step_numbers % istep == 0
                if np.any(mask):
                    per_subnet.append((times[mask], data[mask]))
                else:
                    per_subnet.append((
                        np.array([], dtype=times.dtype),
                        np.empty((0,) + data.shape[1:], dtype=data.dtype),
                    ))
            elif isinstance(m, TemporalAverage):
                per_subnet.append((times, data))
            elif isinstance(m, Raw):
                per_subnet.append((times, data))
            else:
                raise NotImplementedError(
                    f"Monitor {type(m).__name__} is not yet supported by the Numba backend. "
                    "Supported: TemporalAverage, Raw, SubSample, GlobalAverage, "
                    "AfferentCoupling, SpatialAverage, Projection (EEG/MEG/iEEG)."
                )
        results.append(per_subnet)

    # Merge subnets when node_indices are available (connectome-ordered output)
    if should_merge:
        merged_results = []
        for mi, m in enumerate(monitors):
            if isinstance(m, (TemporalAverage, Raw, SubSample, Bold)):
                # Per-subnet monitors: no merging needed
                merged_results.append(results[mi])
                continue
            if isinstance(m, GlobalAverage):
                # Merge raw data first, then average over all nodes
                merged = _merge_and_global_average(results[mi], subnet_infos)
                merged_results.append(merged)
                continue
            if isinstance(m, SpatialAverage) and hasattr(m, 'spatial_mean'):
                # Merge raw data, then apply spatial_mean on the merged output
                merged = _merge_and_spatial_average(results[mi], subnet_infos, m.spatial_mean)
                merged_results.append(merged)
                continue
            if isinstance(m, Projection):
                # Merge raw data into connectome order, then apply gain matrix.
                # Each subnet stored raw data (not projected) in the per-subnet loop,
                # because the full gain spans all connectome nodes.
                merged = _merge_subnet_outputs(results[mi], subnet_infos)
                times_m, data_m = merged[0]
                gain = np.asarray(m.gain, dtype=data_m.dtype)
                data_2d = data_m.sum(axis=-1)  # (T, n_voi, total_nodes)
                projected = np.einsum('ij,tkj->tki', gain, data_2d)
                merged_results.append([(times_m, projected[..., np.newaxis])])
                continue
            # Fallback: merge by placing data at node_indices
            merged = _merge_subnet_outputs(results[mi], subnet_infos)
            merged_results.append(merged)
        results = merged_results

    return results


def _can_merge_subnets(subnet_infos: list) -> bool:
    """Check if all subnets have node_indices and same voi count."""
    if not subnet_infos:
        return False
    # All must have node_indices
    if not all(si.node_indices is not None for si in subnet_infos):
        return False
    # All must have same voi count
    voi_counts = [len(si.model.variables_of_interest) for si in subnet_infos]
    return len(set(voi_counts)) == 1


def _merge_subnet_outputs(
    per_subnet: list,
    subnet_infos: list,
) -> list:
    """Merge per-subnet (times, data) outputs into connectome-ordered output.

    Each subnet's data has shape (T, n_voi, n_subnet_nodes, 1). The merged
    output has shape (T, n_voi, total_nodes, 1) with each subnet placed at
    its node_indices positions.

    Returns a single-element list [(merged_times, merged_data)] to match
    the per-monitor list-of-subnet format.
    """
    total_nodes = max(int(ix.max()) for ix in [si.node_indices for si in subnet_infos]) + 1
    # Use the first subnet's times (all should be aligned after aggregation)
    merged_times = per_subnet[0][0]
    n_chunks = len(merged_times)
    n_voi = per_subnet[0][1].shape[1]

    merged_data = np.zeros((n_chunks, n_voi, total_nodes, 1), dtype=np.float32)
    for si, (t, d) in zip(subnet_infos, per_subnet):
        # d shape: (T, n_voi, n_subnet_nodes, 1)
        merged_data[:, :, si.node_indices, :] = d

    return [(merged_times, merged_data)]


def _merge_and_global_average(
    per_subnet: list,
    subnet_infos: list,
) -> list:
    """Merge per-subnet data then compute global average across all nodes.

    Returns a single-element list [(times, averaged_data)] where
    averaged_data has shape (T, n_voi, 1, 1).
    """
    merged = _merge_subnet_outputs(per_subnet, subnet_infos)
    times, data = merged[0]
    # Average over nodes axis (axis 2)
    averaged = data.mean(axis=2, keepdims=True)
    return [(times, averaged)]


def _merge_and_spatial_average(
    per_subnet: list,
    subnet_infos: list,
    spatial_mean: np.ndarray,
) -> list:
    """Merge per-subnet data then apply spatial_mean on merged connectome data.

    Returns a single-element list [(times, spatial_data)].
    """
    merged = _merge_subnet_outputs(per_subnet, subnet_infos)
    times, data = merged[0]
    # data: (T, n_voi, total_nodes, 1)
    # spatial_mean: (n_areas, total_nodes)
    sm = np.asarray(spatial_mean, dtype=data.dtype)
    # einsum: (n_areas, total_nodes) @ (T, n_voi, total_nodes, 1) -> (T, n_voi, n_areas, 1)
    spatial = np.einsum('ij,tkjm->tkim', sm, data)
    return [(times, spatial)]


def _cfun_type(p: "ProjectionInfo") -> str:
    """Return the string coupling-function type for a ProjectionInfo."""
    from tvb.simulator.hybrid.coupling import (
        Linear,
        Scaling,
        Sigmoidal,
        SigmoidalJansenRit,
        Kuramoto as KuramotoCfun,
        Difference,
        HyperbolicTangent,
        PreSigmoidal,
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
        if getattr(p.cfun, 'use_classic', 1):
            return "sigmoidal_jr"
        return "sigmoidal_jr_legacy"
    if isinstance(p.cfun, KuramotoCfun):
        return "kuramoto"
    if isinstance(p.cfun, Difference):
        return "difference"
    if isinstance(p.cfun, HyperbolicTangent):
        return "tanh"
    if isinstance(p.cfun, PreSigmoidal):
        if getattr(p.cfun, 'dynamic', 0):
            return "pre_sigmoidal_dynamic"
        return "pre_sigmoidal"
    return "none"


def _cfun_params(p: "ProjectionInfo") -> "np.ndarray":
    """Return a float32 array of length 16 with cfun parameters for a ProjectionInfo.

    Layout by cfun type:
      none:                  [1.0, 0, 0, 0, 0, 0, 0, 0, ...]
      linear:                [a, b, 0, 0, 0, 0, 0, 0, ...]
      scaling:               [a, 0, 0, 0, 0, 0, 0, 0, ...]
      sigmoidal:             [a, sigma, midpoint, cmin, cmax, 0, 0, 0, ...]
      sigmoidal_jr:          [a, cmin, cmax, r, midpoint, 0, 0, 0, ...]  (classic)
      sigmoidal_jr_legacy:   [a, e0, r, v0, 0, 0, 0, 0, ...]           (legacy)
      kuramoto:              [a, 1/N, 0, 0, 0, 0, 0, 0, ...]
      difference:            [a, 0, 0, 0, 0, 0, 0, 0, ...]
      tanh:                  [a, b, midpoint, sigma, 0, 0, 0, 0, ...]
      pre_sigmoidal:         [H, Q, G, P, theta, 0, 0, 0, ...]          (static)
      pre_sigmoidal_dynamic: [H, Q, G, P, 0, 0, 0, 0, ...]              (dynamic)
    """
    from tvb.simulator.hybrid.coupling import (
        Linear,
        Scaling,
        Sigmoidal,
        SigmoidalJansenRit,
        Kuramoto as KuramotoCfun,
        Difference,
        HyperbolicTangent,
        PreSigmoidal,
    )

    arr = np.zeros(16, dtype=np.float32)
    arr[0] = 1.0  # default: identity scale
    if p.cfun is None:
        return arr
    if isinstance(p.cfun, Linear):
        arr[0] = float(p.cfun.a[0])
        arr[1] = float(p.cfun.b[0])
        return arr
    if isinstance(p.cfun, Scaling):
        arr[0] = float(p.cfun.a[0])
        return arr
    if isinstance(p.cfun, Difference):
        arr[0] = float(p.cfun.a[0])
        return arr
    if isinstance(p.cfun, Sigmoidal):
        arr[0] = float(p.cfun.a[0])
        arr[1] = float(p.cfun.sigma[0])
        arr[2] = float(p.cfun.midpoint[0])
        arr[3] = float(p.cfun.cmin[0])
        arr[4] = float(p.cfun.cmax[0])
        return arr
    if isinstance(p.cfun, SigmoidalJansenRit):
        if getattr(p.cfun, 'use_classic', 1):
            # Classic mode: [a, cmin, cmax, r, midpoint]
            arr[0] = float(p.cfun.a[0])
            arr[1] = float(p.cfun.cmin[0])
            arr[2] = float(p.cfun.cmax[0])
            arr[3] = float(p.cfun.r[0])
            arr[4] = float(p.cfun.midpoint[0])
        else:
            # Legacy mode: [a, e0, r, v0]
            arr[0] = float(p.cfun.a[0])
            arr[1] = float(p.cfun.e0[0])
            arr[2] = float(p.cfun.r[0])
            arr[3] = float(p.cfun.v0[0])
        return arr
    if isinstance(p.cfun, KuramotoCfun):
        arr[0] = float(p.cfun.a[0])
        # N = number of coupling variables for normalization
        n_cvar = p.source_cvar.shape[0]
        arr[1] = 1.0 / n_cvar if n_cvar > 0 else 1.0
        return arr
    if isinstance(p.cfun, HyperbolicTangent):
        arr[0] = float(p.cfun.a[0])
        arr[1] = float(p.cfun.b[0])
        arr[2] = float(p.cfun.midpoint[0])
        arr[3] = float(p.cfun.sigma[0])
        return arr
    if isinstance(p.cfun, PreSigmoidal):
        arr[0] = float(p.cfun.H[0])
        arr[1] = float(p.cfun.Q[0])
        arr[2] = float(p.cfun.G[0])
        arr[3] = float(p.cfun.P[0])
        if getattr(p.cfun, 'dynamic', 0):
            # Dynamic mode: theta not used, threshold from x_j[1]
            arr[4] = 0.0
        else:
            # Static mode: theta
            arr[4] = float(p.cfun.theta[0])
        return arr
    return arr


def _needs_xi(p: "ProjectionInfo") -> bool:
    """Return True if the coupling function needs per-edge target state x_i.

    Difference and Kuramoto need x_i to compute x_j - x_i or sin(x_j - x_i)
    per edge before the weighted sum.
    """
    ct = _cfun_type(p)
    return ct in ("difference", "kuramoto")


def _n_src_cvar_pre(p: "ProjectionInfo") -> int:
    """Return the number of source coupling variables consumed by pre().

    Most coupling functions read one source cvar per projection edge.
    SigmoidalJansenRit (classic) and PreSigmoidal (dynamic) read two.
    """
    ct = _cfun_type(p)
    if ct == "sigmoidal_jr":
        return 2
    if ct == "pre_sigmoidal_dynamic":
        return 2
    return p.source_cvar.shape[0]


def _cvar_mapping_mode(p: "ProjectionInfo") -> str:
    """Determine which cvar-mapping branch to use at code-gen time."""
    ns = p.source_cvar.shape[0]
    nt = p.target_cvar.shape[0]
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
# Data classes for code-generation analysis
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SubnetworkInfo:
    name: str
    model: object  # Model instance
    integrator: object  # Integrator instance
    n_nodes: int
    n_modes: int
    is_stochastic: bool = False
    noise_nsig: Optional[np.ndarray] = None  # shape (n_vars,), only when is_stochastic
    has_stimulus: bool = False
    node_indices: Optional[np.ndarray] = None  # connectome positions, shape (n_nodes,)


@dataclasses.dataclass
class ProjectionInfo:
    name: str
    source_subnet: str
    target_subnet: str
    source_cvar: np.ndarray  # (n_src_cvar,)
    target_cvar: np.ndarray  # (n_tgt_cvar,) state-variable indices
    target_cvar_cpl: np.ndarray  # (n_tgt_cvar,) coupling-variable indices (for tgt array)
    weights_data: np.ndarray  # (nnz,) float32
    weights_indices: np.ndarray  # (nnz,) int
    weights_indptr: np.ndarray  # (n_tgt+1,) int
    idelays: np.ndarray  # (nnz,) int
    horizon: int
    scale: float
    target_scales: np.ndarray  # (n_tgt_cvar,) or empty
    cfun: object  # coupling function or None
    is_inter: bool
    n_tgt_nodes: int
    # mode_map only for inter projections
    mode_map: Optional[np.ndarray] = None  # (n_src_modes, n_tgt_modes)

    @property
    def n_src_modes(self) -> int:
        if self.is_inter:
            return self.mode_map.shape[0]
        # intra: stored in buf — derive from horizon dimension later, but
        # mode_map is None for intra; caller must pass n_modes separately.
        return self._n_src_modes

    @n_src_modes.setter
    def n_src_modes(self, v: int):
        self._n_src_modes = v

    @property
    def n_tgt_modes(self) -> int:
        if self.is_inter:
            return self.mode_map.shape[1]
        return self._n_src_modes  # same for intra


@dataclasses.dataclass
class NetworkAnalysis:
    subnetworks: List[SubnetworkInfo]
    inter_projections: List[ProjectionInfo]
    intra_projections: List[ProjectionInfo]
    # stimuli_by_subnet: dict mapping subnet name -> list of Stim objects
    stimuli_by_subnet: dict = dataclasses.field(default_factory=dict)
    # source_horizons: dict mapping source subnet name -> max horizon across outgoing projections
    source_horizons: dict = dataclasses.field(default_factory=dict)

    @property
    def all_projections(self) -> List[ProjectionInfo]:
        return self.inter_projections + self.intra_projections


# ---------------------------------------------------------------------------
# Stimulus lazy-evaluation threshold
# ---------------------------------------------------------------------------

# When the pre-computed stimulus array would exceed this many megabytes,
# a lazy chunk-by-chunk path should be used instead of pre-allocating the
# full (n_cvar, n_nodes, n_modes, nstep) array.
# Override at runtime via the TVB_HYBRID_LAZY_STIM_MB environment variable.
_STIM_LAZY_THRESHOLD_MB: int = 64

# ---------------------------------------------------------------------------
# Module-level compiled-function cache
# ---------------------------------------------------------------------------

# Keyed by SHA-256 of the rendered (pre-autopep8) source string so the same
# topology produces the same key regardless of which NbHybridBackend instance
# Module-level cache for compiled functions, keyed by SHA-256 of rendered source.
# Stores (run_network_fn, module_object) pairs so the sweep kernel can
# reference network_chunk from the same module.
_COMPILED_FN_CACHE: dict = {}
_COMPILED_MOD_CACHE: dict = {}


def _build_as_module(source: str, cache_key: str):
    """Write *source* to a real .py file so Numba's file-based cache works.

    Numba's ``cache=True`` requires a real ``co_filename`` on the compiled
    function.  By writing the generated source to a file and importing it
    as a proper Python module we give Numba that filename, allowing it to
    persist ``.nbi``/``.nbc`` files in ``__pycache__/`` next to the ``.py``.
    """
    import importlib.util
    import os
    import sys
    import tempfile
    from pathlib import Path

    cache_dir = Path(tempfile.gettempdir()) / "tvb_nb_hybrid_cache"
    cache_dir.mkdir(exist_ok=True)
    mod_name = f"nbhybrid_{cache_key[:16]}"
    mod_path = cache_dir / f"{mod_name}.py"
    if not mod_path.exists():
        # Atomic write: write to .tmp then os.replace to avoid partial reads.
        tmp_path = mod_path.with_suffix(".tmp")
        tmp_path.write_text(source, encoding="utf-8")
        os.replace(tmp_path, mod_path)
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so Numba can find it for cache lookup.
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    # Also cache the module object so the sweep kernel can reference network_chunk
    _COMPILED_MOD_CACHE[cache_key] = mod
    return mod.run_network


# ---------------------------------------------------------------------------
# CompiledNetworkFn — holds a compiled kernel + helper to run it
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CompiledNetworkFn:
    """A compiled Numba simulation kernel bound to a specific network topology.

    Obtain via :meth:`NbHybridBackend.compile`.
    Call :meth:`run` to execute the simulation without re-compiling.
    """

    _backend: "NbHybridBackend"
    _analysis: "NetworkAnalysis"
    _run_network_fn: object  # the exec'd Python callable
    _network_set: NetworkSet
    _bold_states: Optional[dict]  # per-subnetwork Bold Balloon state, persisted across calls
    _compiled: bool = False  # True after Numba JIT warmup has completed

    def warmup(self) -> float:
        """Eagerly trigger Numba JIT compilation with a single warmup step.

        Call this to front-load the one-time compilation cost (~1-3 s)
        so that subsequent :meth:`run` calls execute at full speed (~5 ms
        per 1 000 steps).  Called automatically by :meth:`NbHybridBackend.compile`
        when ``eager=True`` (the default).

        If compilation was already triggered (by a prior :meth:`run` call
        or a warmup), this is a no-op.

        Returns
        -------
        float
            Wall-clock seconds spent on compilation (0.0 if already compiled).
        """
        if self._compiled:
            return 0.0
        import time

        # For stochastic integrators, the warmup step draws from the noise
        # RNG.  Save and restore the RNG state so that the warmup does not
        # perturb subsequent simulation results.
        rng_states = []
        for sn in self._network_set.subnets:
            sn_info = next(
                si for si in self._analysis.subnetworks if si.name == sn.name
            )
            if sn_info.is_stochastic:
                rng = sn.scheme.noise.random_stream
                rng_states.append((sn, rng, rng.get_state()))

        t0 = time.perf_counter()
        self.run(nstep=1)
        elapsed = time.perf_counter() - t0

        # Restore RNG state so warmup noise draw is invisible to callers.
        for sn, rng, saved_state in rng_states:
            rng.set_state(saved_state)

        self._compiled = True
        # Reset Bold state — the warmup step was purely to trigger JIT;
        # real simulations should start from a clean Bold ODE state.
        self._bold_states = None
        return elapsed

    def run(
        self,
        nstep: int,
        chunk_size: int = None,
        initial_states: Optional[list] = None,
        return_snapshot: bool = False,
        _initial_buffers: Optional[dict] = None,
        monitors: Optional[list] = None,
    ) -> list:
        """Execute the pre-compiled kernel for *nstep* integration steps.

        Parameters
        ----------
        nstep : int
            Number of integration steps to run.
        chunk_size : int or None
            Number of steps per temporal-average chunk.  When *None* (default),
            the chunk_size is auto-computed from monitor periods to ensure
            sampling alignment: GCD of all monitor ``istep`` values.  If no
            monitors are provided, defaults to 1 (raw output per step).
        initial_states : list of ndarray, optional
            Initial states per subnetwork.  If *None* the subnetwork's
            ``zero_states()`` are used.
        return_snapshot : bool
            If True, also return a snapshot dict suitable for passing to
            :meth:`resume`.  Default False (backward compatible).
        _initial_buffers : dict, optional
            Pre-populated source history buffers keyed by subnet name, as
            returned in ``snapshot['buffers']`` by a prior :meth:`run` call.
            When supplied the existing buffer is reused instead of being
            re-initialised from the initial state.
        monitors : list of Monitor, optional
            If provided, raw outputs are post-processed per monitor type and
            the return format changes to ``list[list[tuple[times, data]]]``
            (outer list by monitor, inner list by subnetwork).  If *None*
            (default), the original ``(times, data, ctavg)`` format is used.

        Returns
        -------
        list or (list, dict)
            When *return_snapshot* is False (default):
              - If *monitors* is None: list of (times, data, ctavg) per subnetwork.
              - If *monitors* is provided: list per monitor of list per subnetwork
                of (times, data).
            When *return_snapshot* is True: ``(outputs, snapshot)`` where *snapshot*
            is a dict with keys ``'states'`` and ``'buffers'`` suitable for passing
            to :meth:`resume`.
        """
        # Resolve chunk_size: auto-compute from monitor periods when not specified
        dt = self._network_set.subnets[0].scheme.dt
        if chunk_size is None:
            if monitors is not None:
                chunk_size = _compute_chunk_size(monitors, dt)
            else:
                chunk_size = 1

        if monitors is not None:
            from tvb.simulator.monitors import Raw, SubSample

            from tvb.simulator.monitors import AfferentCoupling
            for m in monitors:
                if isinstance(m, Raw) and not isinstance(m, AfferentCoupling) and chunk_size != 1:
                    raise ValueError(
                        "Raw monitor requires chunk_size=1; "
                        "pass chunk_size=1 to run_network()"
                    )
                if isinstance(m, SubSample) and chunk_size != 1:
                    raise ValueError(
                        f"SubSample monitor requires chunk_size=1 "
                        f"(got chunk_size={chunk_size}). "
                        "The step-based selection mask assumes one step per chunk. "
                        "Pass chunk_size=1 or use TemporalAverage instead."
                    )
        outputs, final_states, final_bufs, monitor_data = self._backend._run_compiled(
            self._run_network_fn,
            self._analysis,
            self._network_set,
            nstep,
            chunk_size,
            initial_states,
            _initial_buffers=_initial_buffers,
            monitors=monitors,
            _bold_states=self._bold_states,
        )
        # Persist Bold state across calls (Balloon model needs continuity)
        if monitor_data.get('bold_states'):
            self._bold_states = monitor_data['bold_states']
        if monitors is not None:
            outputs = _apply_monitors(outputs, monitors, dt, chunk_size=chunk_size,
                                      monitor_data=monitor_data,
                                      subnet_infos=self._analysis.subnetworks)
        if not return_snapshot:
            return outputs
        snapshot = {
            "states": [
                final_states[sn.name].copy() for sn in self._analysis.subnetworks
            ],
            "buffers": {name: buf.copy() for name, buf in final_bufs.items()},
        }
        return outputs, snapshot

    def resume(
        self,
        snapshot: dict,
        nstep: int,
        chunk_size: int = None,
        return_snapshot: bool = False,
    ) -> list:
        """Resume a simulation from a snapshot returned by :meth:`run`.

        Parameters
        ----------
        snapshot : dict
            A snapshot dict as returned by ``run(..., return_snapshot=True)``.
            Must contain keys ``'states'`` (list of ndarray per subnetwork) and
            ``'buffers'`` (dict of ndarray per source subnet name).
        nstep : int
            Number of additional integration steps to run.
        chunk_size : int or None
            Steps per temporal-average chunk.  When *None*, defaults to 1.
        return_snapshot : bool
            If True, also return a new snapshot of the final state.

        Returns
        -------
        list or (list, dict)
            Same format as :meth:`run`.
        """
        if chunk_size is None:
            chunk_size = 1
        return self.run(
            nstep,
            chunk_size,
            initial_states=snapshot["states"],
            return_snapshot=return_snapshot,
            _initial_buffers=snapshot["buffers"],
        )


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class NbHybridBackend(MakoUtilMix):
    """Numba backend for hybrid simulator with multi-subnetwork support.

    Call ``run_network(network_set, nstep)`` to compile and run.  The
    network_set must already be configured (``configure()`` called).

    Supported models: ``MontbrioPazoRoxin``, ``KIonEx`` (and any model that
    provides ``state_variable_dfuns``, ``coupling_terms``, and
    ``global_parameter_names``).  Integrators: Heun/Euler deterministic or
    stochastic.
    """

    @staticmethod
    def get_cache_dir():
        """Return the directory where disk cache files are written."""
        import tempfile
        from pathlib import Path

        return Path(tempfile.gettempdir()) / "tvb_nb_hybrid_cache"

    @classmethod
    def clear_cache(cls):
        """Clear the in-process compiled function cache and the disk cache."""
        import shutil

        _COMPILED_FN_CACHE.clear()
        cache_dir = cls.get_cache_dir()
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    def compile(
        self,
        network_set: NetworkSet,
        print_source: bool = False,
        debug_nojit: bool = False,
        eager: bool = True,
    ) -> "CompiledNetworkFn":
        """Compile the simulation kernel for *network_set* and return it.

        The compiled kernel is cached in-process by a SHA-256 hash of the
        generated source.  Repeated calls with topologically identical networks
        return the cached kernel immediately (no re-compilation).

        On first call for a new topology, Numba must JIT-compile the generated
        ``network_chunk`` kernel (~1-3 s depending on model complexity).  When
        *eager* is True (default), this compilation happens here, avoiding a
        surprise latency spike on the first :meth:`CompiledNetworkFn.run` call.
        The compiled kernel is also saved to a Numba disk cache
        (``/tmp/tvb_nb_hybrid_cache/``) so that subsequent Python processes
        skip JIT entirely (~5 ms instead of ~1.7 s).

        Parameters
        ----------
        network_set : NetworkSet
            Fully configured network (``configure()`` must have been called).
        print_source : bool
            If True, print the generated (autopep8-formatted) source.
        debug_nojit : bool
            If True, disable Numba JIT for debugging (very slow).
        eager : bool
            If True (default), eagerly trigger Numba JIT compilation by running
            a single warmup step.  This front-loads the one-time ~1-3 s
            compilation cost so that subsequent :meth:`run` calls are fast.
            Set to False to defer compilation to the first :meth:`run` call.

        Returns
        -------
        CompiledNetworkFn
            Callable object whose :meth:`~CompiledNetworkFn.run` method
            executes the simulation without recompiling.
        """
        import os

        _use_nojit = debug_nojit or (
            os.environ.get("TVB_HYBRID_NO_JIT", "0")
            not in ("", "0", "false", "False", "no")
        )
        self._check_compatibility(network_set)
        analysis = self._analyse(network_set)
        run_network_fn = self._build(
            '<%include file="nb-hybrid-sim.py.mako"/>',
            dict(analysis=analysis, np=np, debug_nojit=_use_nojit),
            print_source=print_source,
        )
        cn = CompiledNetworkFn(
            _backend=self,
            _analysis=analysis,
            _run_network_fn=run_network_fn,
            _network_set=network_set,
            _bold_states=None,
        )
        if eager and not _use_nojit:
            cn.warmup()
        return cn

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
        """Run a hybrid simulation using the Numba code-generation path.

        Equivalent to ``self.compile(network_set).run(nstep, ...)``.  The
        compiled kernel is cached in-process so repeated calls with the same
        topology do not re-compile.

        Parameters
        ----------
        network_set : NetworkSet
            Fully configured network (``configure()`` must have been called).
        nstep : int
            Number of integration steps to run.
        chunk_size : int or None
            Number of steps per temporal-average chunk.  When *None* (default),
            auto-computed from monitor periods.  See :meth:`CompiledNetworkFn.run`.
        print_source : bool
            If True, print the generated source code with line numbers.
        monitors : list of Monitor, optional
            If provided, raw outputs are post-processed per monitor type and
            the return format changes to ``list[list[tuple[times, data]]]``
            (outer list by monitor, inner list by subnetwork).  If *None*
            (default), the original ``(times, data, ctavg)`` format is used.

        Returns
        -------
        list of (times, data, ctavg) or list[list[tuple[times, data]]]
            One tuple per subnetwork in ``network_set.subnets``, where
            ``times`` is a 1-D float64 array of mid-chunk time points,
            ``data`` is a float32 array of shape ``(n_chunks, n_voi, n_nodes, n_modes)``,
            and ``ctavg`` is a float32 array of shape
            ``(n_chunks, n_cvar, n_nodes, n_modes)`` holding the
            temporally-averaged afferent coupling input to each node.

            When *monitors* is provided the format is instead
            ``list[list[tuple[times, data]]]`` — outer list indexed by monitor,
            inner list by subnetwork.
        """
        return self.compile(network_set, print_source, debug_nojit=debug_nojit).run(
            nstep, chunk_size, initial_states, monitors=monitors
        )

    def compile_batch(
        self,
        network_sets: list,
        max_workers: Optional[int] = None,
        print_source: bool = False,
        debug_nojit: bool = False,
        eager: bool = True,
    ) -> list:
        """Compile multiple network topologies in parallel.

        Each topology is compiled in a separate process, allowing Numba's
        LLVM compilation to proceed concurrently.  After compilation, each
        process writes its Numba disk cache (``.nbi``/``.nbc`` files) so
        that subsequent calls in the main process hit the disk cache at
        ~5 ms instead of ~1.7 s.

        Parameters
        ----------
        network_sets : list of NetworkSet
            Fully configured networks (``configure()`` must have been called).
        max_workers : int or None
            Maximum number of parallel compilation processes.  Defaults to
            ``min(len(network_sets), os.cpu_count())``.
        print_source : bool
            If True, print the generated source for each topology.
        debug_nojit : bool
            If True, disable Numba JIT for debugging.
        eager : bool
            If True (default), eagerly warm up each compiled kernel in the
            worker process, populating the Numba disk cache.

        Returns
        -------
        list of CompiledNetworkFn
            One per input topology.
        """
        import os
        from concurrent.futures import ProcessPoolExecutor

        n = len(network_sets)
        if max_workers is None:
            max_workers = min(n, os.cpu_count() or 1)

        if n == 1:
            return [self.compile(network_sets[0], print_source=print_source,
                                debug_nojit=debug_nojit, eager=eager)]

        # Compile each topology in a separate process.
        # Each process will write its Numba disk cache independently.
        def _compile_one(ns):
            backend = NbHybridBackend()
            cn = backend.compile(ns, print_source=print_source,
                                debug_nojit=debug_nojit, eager=eager)
            # Return the cache key so the main process can reconstruct the
            # CompiledNetworkFn from its disk cache.
            return ns  # return the network_set so caller can match results

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(_compile_one, network_sets))

        # Now re-compile in the main process — this time the in-process and
        # disk caches are warm, so it should be ~5 ms per topology.
        results = []
        for ns in network_sets:
            results.append(self.compile(ns, print_source=print_source,
                                       debug_nojit=debug_nojit, eager=False))
        return results

    # ------------------------------------------------------------------
    # _run_compiled — arg assembly + kernel call (no compilation logic)
    # ------------------------------------------------------------------

    def _run_compiled(
        self,
        run_network_fn,
        analysis: NetworkAnalysis,
        network_set: NetworkSet,
        nstep: int,
        chunk_size: int,
        initial_states: Optional[list],
        _initial_buffers: Optional[dict] = None,
        monitors: Optional[list] = None,
        _bold_states: Optional[dict] = None,
    ) -> tuple:
        """Build the argument list and call the pre-compiled kernel."""
        # Guard: chunk_size must not exceed the minimum horizon
        if analysis.all_projections:
            min_horizon = min(p.horizon for p in analysis.all_projections)
            if chunk_size > min_horizon:
                raise ValueError(
                    f"chunk_size={chunk_size} exceeds the minimum projection horizon "
                    f"({min_horizon} steps = min_delay / dt). "
                    f"Reduce chunk_size to at most {min_horizon}, or increase tract "
                    "lengths / reduce dt."
                )
        # Build argument list matching the generated run_network() signature
        args = [nstep]

        # Per-subnetwork initial states (from provided list or zero)
        sn_states = {}
        for i, sn_info in enumerate(analysis.subnetworks):
            sn_obj = next(s for s in network_set.subnets if s.name == sn_info.name)
            if initial_states is not None:
                state = initial_states[i].astype(np.float32)
            else:
                state = sn_obj.zero_states().astype(np.float32)
            sn_states[sn_info.name] = state
            args.append(state)

        # Per-source-subnet shared history buffers (one buffer per source subnet)
        src_bufs = {}
        for sn_info in analysis.subnetworks:
            if _initial_buffers is not None and sn_info.name in _initial_buffers:
                buf = _initial_buffers[sn_info.name].astype(np.float32)
            else:
                horizon = analysis.source_horizons.get(sn_info.name, 1)
                state = sn_states[sn_info.name]
                n_vars, n_nodes, n_modes = state.shape
                buf = np.empty((n_vars, n_nodes, n_modes, horizon), dtype=np.float32)
                buf[:] = state[
                    :, :, :, np.newaxis
                ]  # broadcast ICs across all horizon slots
            src_bufs[sn_info.name] = buf
        for sn_info in analysis.subnetworks:
            args.append(src_bufs[sn_info.name])

        # Per-projection arrays
        for p in analysis.all_projections:
            args.append(p.weights_data.astype(np.float32))
            args.append(p.weights_indices.astype(np.int32))
            args.append(p.weights_indptr.astype(np.int32))
            args.append(p.idelays.astype(np.int32))
            if p.is_inter:
                args.append(p.mode_map.astype(np.float32))
            args.append(p.source_cvar.astype(np.int32))
            args.append(p.target_cvar.astype(np.int32))
            args.append(p.target_cvar_cpl.astype(np.int32))
            args.append(np.float32(p.scale))
            ts = (
                p.target_scales.astype(np.float32)
                if p.target_scales.size > 0
                else np.zeros(0, dtype=np.float32)
            )
            args.append(ts)
            cfun_params = _cfun_params(p)
            args.append(cfun_params)

        # Per-subnetwork noise arrays (stochastic integrators)
        for sn_info in analysis.subnetworks:
            if sn_info.is_stochastic:
                sn_obj = next(s for s in network_set.subnets if s.name == sn_info.name)
                dt = sn_obj.scheme.dt
                rng = sn_obj.scheme.noise.random_stream
                # Draw in (nstep, n_vars, n_nodes, n_modes) order so that
                # transposed [:, :, :, t] == t-th sequential randn(n_vars, n_nodes, n_modes) call
                dw = rng.randn(
                    nstep, sn_info.model.nvar, sn_info.n_nodes, sn_info.n_modes
                )
                noise_std = np.sqrt(2.0 * sn_info.noise_nsig * dt)  # (n_vars,)
                dw *= noise_std[np.newaxis, :, np.newaxis, np.newaxis]
                # Transpose to (n_vars, n_nodes, n_modes, nstep)
                dw = np.ascontiguousarray(np.transpose(dw, (1, 2, 3, 0))).astype(
                    np.float32
                )
                args.append(dw)

        # Per-subnetwork stimulus arrays (pre-computed batch)
        # TODO §8.4: use lazy chunk-by-chunk path when estimated stim_arr_mb
        #   exceeds _STIM_LAZY_THRESHOLD_MB (or TVB_HYBRID_LAZY_STIM_MB env var).
        #   See _compute_stimulus_lazy() for the planned implementation.
        for sn_info in analysis.subnetworks:
            if sn_info.has_stimulus:
                n_cvar = len(sn_info.model.cvar)
                stim_arr = np.zeros(
                    (n_cvar, sn_info.n_nodes, sn_info.n_modes, nstep),
                    dtype=np.float32,
                )
                for stim in analysis.stimuli_by_subnet[sn_info.name]:
                    for step_idx in range(1, nstep + 1):
                        sc = np.asarray(stim.get_coupling(step_idx), dtype=np.float32)
                        if sc.ndim == 2:
                            sc = sc[:, :, np.newaxis]
                        if sc.shape[2] == 1 and sn_info.n_modes > 1:
                            sc = np.broadcast_to(
                                sc, (sc.shape[0], sn_info.n_nodes, sn_info.n_modes)
                            ).copy()
                        # broadcast += matches Python path (tgt += stim.get_coupling(step))
                        stim_arr[:, :, :, step_idx - 1] += sc
                args.append(stim_arr)

        # Per-subnetwork spatial parameter arrays (heterogeneous per-node parameters)
        for sn_info in analysis.subnetworks:
            sp_names = list(getattr(sn_info.model, 'spatial_parameter_names', []))
            if sp_names:
                sp_arr = np.array(
                    [np.broadcast_to(getattr(sn_info.model, n), (sn_info.n_nodes,)).ravel()
                     for n in sp_names],
                    dtype=np.float32,
                )
            else:
                sp_arr = np.zeros((0, sn_info.n_nodes), dtype=np.float32)
            args.append(sp_arr)

        # Per-subnetwork monitor config arrays (SpatialAverage / Projection)
        from tvb.simulator.monitors import SpatialAverage as SpatialAverageMon, Projection as ProjectionMon, Bold as BoldMon
        _spatial_mean_mon = None
        _projection_mon = None
        _bold_mon = None
        if monitors:
            for m in monitors:
                if isinstance(m, SpatialAverageMon) and hasattr(m, 'spatial_mean'):
                    _spatial_mean_mon = m
                if isinstance(m, ProjectionMon):
                    _projection_mon = m
                if isinstance(m, BoldMon):
                    _bold_mon = m
        for sn_info in analysis.subnetworks:
            # spatial_mean: (n_areas, n_nodes) or empty (0, n_nodes)
            if _spatial_mean_mon is not None:
                sm = np.asarray(_spatial_mean_mon.spatial_mean, dtype=np.float32)
                if sm.shape[1] != sn_info.n_nodes:
                    sm = np.zeros((0, sn_info.n_nodes), dtype=np.float32)
            else:
                sm = np.zeros((0, sn_info.n_nodes), dtype=np.float32)
            args.append(sm)
            # gain: (n_sensors, n_nodes) or empty (0, n_nodes)
            if _projection_mon is not None and hasattr(_projection_mon, 'gain'):
                gn = np.asarray(_projection_mon.gain, dtype=np.float32)
                if gn.shape[1] != sn_info.n_nodes:
                    gn = np.zeros((0, sn_info.n_nodes), dtype=np.float32)
            else:
                gn = np.zeros((0, sn_info.n_nodes), dtype=np.float32)
            args.append(gn)

        # Per-subnetwork Bold Balloon model arrays
        # Bold parameters: [rtau_s, rtau_f, rtau_o, ra, e0, re0, k1, k2, k3]
        # Default values from vbjax/compute_bold_theta()
        _bold_params = None
        _bold_istep = 0
        _bold_v0 = np.float32(0.0)
        _bold_dt = np.float32(0.0)
        if _bold_mon is not None:
            dt = network_set.subnets[0].scheme.dt
            _bold_dt = np.float32(dt)
            # Extract Bold period in steps
            bold_period = float(_bold_mon.period)  # ms
            _bold_istep = max(1, int(round(bold_period / dt)))
            # Compute Balloon model parameters
            tau_s = np.float32(0.65)
            tau_f = np.float32(0.41)
            tau_o = np.float32(0.98)
            alpha = np.float32(0.32)
            te = np.float32(0.04)
            e0 = np.float32(0.4)
            epsilon = np.float32(0.5)
            nu_0 = np.float32(40.3)
            r_0 = np.float32(25.0)
            v0 = np.float32(4.0)
            _bold_v0 = v0
            k1 = np.float32(4.3) * nu_0 * e0 * te
            k2 = epsilon * r_0 * e0 * te
            k3 = np.float32(1.0) - epsilon
            _bold_params = np.array([
                np.float32(1.0) / tau_s,  # rtau_s
                np.float32(1.0) / tau_f,  # rtau_f
                np.float32(1.0) / tau_o,  # rtau_o
                np.float32(1.0) / alpha,  # ra
                e0,
                np.float32(1.0) / e0,    # re0
                k1, k2, k3,
            ], dtype=np.float32)

        _bold_state_arrays = {}  # name -> bold_state array (for persistence)

        for sn_info in analysis.subnetworks:
            n_voi = len(sn_info.model.variables_of_interest)
            n_nodes = sn_info.n_nodes
            svars = list(sn_info.model.state_variables)
            voi = list(sn_info.model.variables_of_interest)
            voi_idx = [svars.index(v) if v in svars else 0 for v in voi]
            if _bold_mon is not None:
                # Reuse existing Bold state if available (persistence across calls)
                if _bold_states is not None and sn_info.name in _bold_states:
                    bold_state = _bold_states[sn_info.name]
                else:
                    bold_state = np.zeros((n_voi, 4, n_nodes), dtype=np.float32)
                    # Initial conditions: s=0, f=1, v=1, q=1
                    bold_state[:, 1, :] = 1.0
                    bold_state[:, 2, :] = 1.0
                    bold_state[:, 3, :] = 1.0
                bold_params = _bold_params
                bold_voi_idx = np.array(voi_idx, dtype=np.int32)
            else:
                bold_state = np.zeros((0, 4, 0), dtype=np.float32)
                bold_params = np.zeros(9, dtype=np.float32)
                bold_voi_idx = np.zeros(0, dtype=np.int32)
            args.append(bold_state)
            args.append(bold_params)
            args.append(bold_voi_idx)
            # Track Bold state arrays for persistence
            if _bold_mon is not None:
                _bold_state_arrays[sn_info.name] = bold_state

        args.append(_bold_dt)
        args.append(np.int32(_bold_istep))
        args.append(_bold_v0)
        args.append(chunk_size)

        outputs = run_network_fn(*args)
        # outputs[i] = (times, data, ctavg, spatial, proj, bold_times, bold_data)
        raw_outputs = [(t, d, c) for t, d, c, s, p, bt, bd in outputs]
        # Bold state arrays are mutated in-place, so _bold_state_arrays now has updated values
        bold_states_dict = _bold_state_arrays if _bold_state_arrays else None
        monitor_data = {
            'spatial': [s for t, d, c, s, p, bt, bd in outputs],
            'proj': [p for t, d, c, s, p, bt, bd in outputs],
            'bold': [(bt, bd) for t, d, c, s, p, bt, bd in outputs],
            'bold_states': bold_states_dict,
        }
        return raw_outputs, sn_states, src_bufs, monitor_data

    @staticmethod
    def _compute_stimulus_lazy(
        analysis: "NetworkAnalysis",
        sn_info: "SubnetworkInfo",
        step_start: int,
        step_end: int,
    ) -> np.ndarray:
        """Compute stimulus for steps *step_start*..*step_end* (inclusive).

        Returns a ``(n_cvar, n_nodes, n_modes, window_size)`` float32 array
        whose last axis spans the requested step range.

        This is a stub for the planned lazy chunk-by-chunk stimulus path
        (§8.4).  The intent is that ``_run_compiled`` will call this per
        chunk instead of pre-allocating the full ``(…, nstep)`` array, so
        that peak RSS stays bounded by ``chunk_size`` rather than ``nstep``.

        TODO §8.4: Wire this into ``_run_compiled``:
          - Check ``_stim_estimate_mb(analysis, sn_info, nstep) > threshold``
          - If so, expose ``network_chunk`` from the generated module, replicate
            the ``run_network`` outer loop in Python, and call this method
            per chunk to build a ``(…, this_chunk)`` stim window.
          - Requires template change: stim indexing in ``network_chunk`` must
            use the local offset ``t_local`` rather than the global ``t - 1``
            so that chunk-sized stim views can be passed without out-of-bounds
            access on the second and subsequent chunks.
        """
        n_cvar = len(sn_info.model.cvar)
        window_size = step_end - step_start + 1
        stim_arr = np.zeros(
            (n_cvar, sn_info.n_nodes, sn_info.n_modes, window_size),
            dtype=np.float32,
        )
        for stim in analysis.stimuli_by_subnet[sn_info.name]:
            for local_idx, step_idx in enumerate(range(step_start, step_end + 1)):
                sc = np.asarray(stim.get_coupling(step_idx), dtype=np.float32)
                if sc.ndim == 2:
                    sc = sc[:, :, np.newaxis]
                if sc.shape[2] == 1 and sn_info.n_modes > 1:
                    sc = np.broadcast_to(
                        sc, (sc.shape[0], sn_info.n_nodes, sn_info.n_modes)
                    ).copy()
                stim_arr[:, :, :, local_idx] += sc
        return stim_arr

    @staticmethod
    def _stim_estimate_mb(
        sn_info: "SubnetworkInfo",
        nstep: int,
    ) -> float:
        """Estimate the memory (in MiB) that the pre-computed stim array would use."""
        import os

        n_cvar = len(sn_info.model.cvar)
        n_bytes = n_cvar * sn_info.n_nodes * sn_info.n_modes * nstep * 4  # float32
        return n_bytes / (1024 * 1024)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_compatibility(self, network_set: NetworkSet) -> None:
        # Supported models — lazy import because loading 27 model modules
        # costs ~2.5 s on first call.  Caching the tuple avoids repeated imports.
        _supported_models = _get_supported_models()
        _allowed_integrators = (
            HeunDeterministic,
            EulerDeterministic,
            HeunStochastic,
            EulerStochastic,
        )
        dt0 = network_set.subnets[0].scheme.dt
        for sn in network_set.subnets:
            if not isinstance(sn.model, _supported_models):
                raise NotImplementedError(
                    f"NbHybridBackend does not support {type(sn.model).__name__}. "
                    f"Supported: MontbrioPazoRoxin, KIonEx, JansenRit, ZetterbergJansen, "
                    f"Generic2dOscillator, SupHopf, Kuramoto, Hopfield, LarterBreakspear, "
                    f"ReducedWongWang, ReducedWongWangExcInh, "
                    f"Epileptor, Epileptor2D, EpileptorCodim3, EpileptorCodim3SlowMod, EpileptorRestingState, "
                    f"WilsonCowan, ZerlautAdaptation*, "
                    f"CoombesByrne2D, CoombesByrne, GastSchmidtKnosche_SD/SF, DumontGutkin, "
                    f"ReducedSetFitzHughNagumo, ReducedSetHindmarshRose, Linear."
                )
            if not isinstance(sn.scheme, _allowed_integrators):
                raise NotImplementedError(
                    f"NbHybridBackend only supports Heun/EulerDeterministic or Stochastic; "
                    f"subnetwork '{sn.name}' uses {type(sn.scheme).__name__}"
                )
            if sn.scheme.dt != dt0:
                raise ValueError(
                    "All subnetworks must share the same dt. "
                    f"Expected {dt0}, got {sn.scheme.dt} in '{sn.name}'"
                )
            # Model-specific validation (look up classes from lazy cache)
            _model_cls_by_name = {m.__name__: m for m in _supported_models}
            Epileptor_ = _model_cls_by_name.get('Epileptor')
            WilsonCowan_ = _model_cls_by_name.get('WilsonCowan')
            if Epileptor_ and isinstance(sn.model, Epileptor_):
                if sn.model.modification[0]:
                    raise NotImplementedError(
                        "NbHybridBackend: Epileptor with modification=True is not supported. "
                        "Set model.modification = numpy.array([False])."
                    )
            if WilsonCowan_ and isinstance(sn.model, WilsonCowan_):
                if not sn.model.shift_sigmoid[0]:
                    raise NotImplementedError(
                        "NbHybridBackend: WilsonCowan with shift_sigmoid=False is not supported. "
                        "Use the default shift_sigmoid=True."
                    )
        _ReducedSetBase = _get_reduced_set_base()
        for sn in network_set.subnets:
            if isinstance(sn.model, _ReducedSetBase):
                if (
                    not hasattr(sn.model, "dfun_mode")
                    or sn.model.dfun_mode != "combined"
                ):
                    raise NotImplementedError(
                        f"{type(sn.model).__name__} requires dfun_mode='combined' attribute. "
                        "Run update_derived_parameters() / configure() before using the backend."
                    )

    def _analyse(self, network_set: NetworkSet) -> "NetworkAnalysis":
        from tvb.simulator.noise import Additive

        # Build stimulus lookup: subnet name -> list of Stim objects
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
                    if nsig.ndim == 0:
                        noise_nsig = np.full(
                            sn.model.nvar, float(nsig), dtype=np.float64
                        )
                    else:
                        noise_nsig = (
                            np.broadcast_to(nsig, (sn.model.nvar,))
                            .copy()
                            .astype(np.float64)
                        )
                else:
                    raise NotImplementedError(
                        f"Subnetwork '{sn.name}': only tvb.simulator.noise.Additive is supported "
                        f"by the Numba backend; got {type(noise_obj).__name__}. "
                        "Use HeunDeterministic or EulerDeterministic for deterministic integration."
                    )
            subnets.append(
                SubnetworkInfo(
                    name=sn.name,
                    model=sn.model,
                    integrator=sn.scheme,
                    n_nodes=sn.nnodes,
                    n_modes=sn.model.number_of_modes,
                    is_stochastic=is_stoch,
                    noise_nsig=noise_nsig,
                    has_stimulus=bool(stims_by_subnet[sn.name]),
                    node_indices=getattr(sn, 'node_indices', None),
                )
            )

        inter_projs = []
        for p in network_set.projections:
            if isinstance(p, IntraProjection):
                continue
            inter_projs.append(self._build_projection_info(p, is_inter=True))

        intra_projs = []
        for sn_obj in network_set.subnets:
            for p in sn_obj.projections:
                pi = self._build_projection_info(p, is_inter=False)
                # For intra, source and target are the same subnetwork
                pi.source_subnet = sn_obj.name
                pi.target_subnet = sn_obj.name
                # Compute target_cvar_cpl: map state-var → coupling-var indices
                tgt_model_cvar = list(sn_obj.model.cvar.astype(int))
                for ic in range(len(pi.target_cvar)):
                    sv = int(pi.target_cvar[ic])
                    if sv in tgt_model_cvar:
                        pi.target_cvar_cpl[ic] = tgt_model_cvar.index(sv)
                    else:
                        pi.target_cvar_cpl[ic] = 0
                intra_projs.append(pi)

        # Assign unique names to avoid collisions
        all_names = {}
        for p in inter_projs + intra_projs:
            base = p.name
            if base in all_names:
                all_names[base] += 1
                p.name = f"{base}_{all_names[base]}"
            else:
                all_names[base] = 0

        # Compute per-source-subnet max horizon for shared history buffers
        _all_projs = inter_projs + intra_projs
        source_horizons: dict = {}
        for _p in _all_projs:
            src = _p.source_subnet
            source_horizons[src] = max(source_horizons.get(src, 1), _p.horizon)
        # Ensure every subnetwork has an entry (default 1 for subnets with no outgoing projections)
        for sn in subnets:
            if sn.name not in source_horizons:
                source_horizons[sn.name] = 1

        return NetworkAnalysis(
            subnetworks=subnets,
            inter_projections=inter_projs,
            intra_projections=intra_projs,
            stimuli_by_subnet=stims_by_subnet,
            source_horizons=source_horizons,
        )

    def _build_projection_info(self, p, is_inter: bool) -> "ProjectionInfo":
        ts = (
            p.target_scales
            if p.target_scales is not None
            else np.zeros(0, dtype=np.float64)
        )

        if is_inter:
            src_name = p.source.name
            tgt_name = p.target.name
            n_src_modes = p.source.model.number_of_modes
            n_tgt_modes = p.target.model.number_of_modes
            if p.mode_map is not None:
                mode_map = p.mode_map.astype(np.float32)
            else:
                mode_map = np.ones((n_src_modes, n_tgt_modes), dtype=np.float32)
            proj_name = f"{src_name}_to_{tgt_name}"
        else:
            src_name = ""  # filled by caller
            tgt_name = ""
            n_src_modes = 1  # will be filled
            mode_map = None
            proj_name = getattr(p, "name", None) or "intra"

        # Strip structural zeros from a copy so the original projection is not mutated.
        # p.idelays is positionally aligned with p.weights.data, so we apply the same mask.
        weights_csr = p.weights.copy()
        idelays_raw = np.atleast_1d(p.idelays)
        nz_mask = weights_csr.data != 0
        weights_csr.eliminate_zeros()
        idelays_stripped = idelays_raw[nz_mask].astype(np.int32)
        if is_inter:
            n_tgt_nodes = p.target.nnodes
        else:
            n_tgt_nodes = p.weights.shape[1]  # number of target nodes in intra-projection
        # Compute target_cvar_cpl: map target state-variable indices to coupling-variable indices.
        # The coupling scratch array (tgt) is indexed by coupling variable,
        # but target_cvar uses state variable indices.  We need the reverse
        # mapping: for each target_cvar[ic], find its position in the target
        # model's cvar array.
        target_cvar_arr = np.atleast_1d(p.target_cvar).astype(np.int32)
        if is_inter:
            tgt_model_cvar = list(p.target.model.cvar.astype(int))
        else:
            # intra: target model is the same as source; cvar available from p
            # Caller will fix src_name/tgt_name later; for now try to get model cvar
            tgt_model_cvar = None  # filled below
        pi = ProjectionInfo(
            name=proj_name,
            source_subnet=src_name,
            target_subnet=tgt_name,
            source_cvar=np.atleast_1d(p.source_cvar).astype(np.int32),
            target_cvar=target_cvar_arr,
            target_cvar_cpl=np.zeros(len(target_cvar_arr), dtype=np.int32),  # placeholder
            weights_data=weights_csr.data.astype(np.float32),
            weights_indices=weights_csr.indices.astype(np.int32),
            weights_indptr=weights_csr.indptr.astype(np.int32),
            idelays=idelays_stripped,
            horizon=int(p._horizon),
            scale=float(p.scale),
            target_scales=np.atleast_1d(ts).astype(np.float32)
            if ts.size > 0
            else np.zeros(0, dtype=np.float32),
            cfun=p.cfun,
            is_inter=is_inter,
            n_tgt_nodes=n_tgt_nodes,
            mode_map=mode_map,
        )
        if is_inter and tgt_model_cvar is not None:
            # Map state-var indices → coupling-var indices
            for ic in range(len(target_cvar_arr)):
                sv = int(target_cvar_arr[ic])
                if sv in tgt_model_cvar:
                    pi.target_cvar_cpl[ic] = tgt_model_cvar.index(sv)
                else:
                    # State var not in cvar: default to coupling index 0
                    pi.target_cvar_cpl[ic] = 0
        if not is_inter:
            pi.n_src_modes = n_src_modes  # placeholder; filled per-subnetwork
        return pi

    def _make_projection_buffer(
        self,
        p: ProjectionInfo,
        sn_states: dict,
        network_set: NetworkSet,
    ) -> np.ndarray:
        """Allocate and pre-fill the circular history buffer for a projection."""
        if p.is_inter:
            src_state = sn_states[p.source_subnet]
        else:
            src_state = sn_states[p.target_subnet]  # intra: same subnetwork

        n_vars, n_nodes, n_modes = src_state.shape
        buf = np.zeros((n_vars, n_nodes, n_modes, p.horizon), dtype=np.float32)
        # Pre-fill all slots with the initial state (matching init_projection_buffers)
        for slot in range(p.horizon):
            buf[:, :, :, slot] = src_state
        return buf

    def _build(self, template_source: str, content: dict, print_source: bool = False):
        """Render and exec the template; return the run_network callable.

        The compiled callable is cached in ``_COMPILED_FN_CACHE`` keyed by the
        SHA-256 of the rendered source so that repeated calls with the same
        network topology skip template rendering, ``exec()``, and Numba JIT.
        ``autopep8`` is applied only when *print_source* is True.
        """
        source = self.render_template(template_source, content)
        cache_key = hashlib.sha256(source.encode()).hexdigest()
        if cache_key in _COMPILED_FN_CACHE:
            if print_source:
                formatted = autopep8.fix_code(source)
                print(self.insert_line_numbers(formatted))
            return _COMPILED_FN_CACHE[cache_key]

        if print_source:
            formatted = autopep8.fix_code(source)
            print(self.insert_line_numbers(formatted))

        try:
            fn = _build_as_module(source, cache_key)
        except Exception as exc:
            print(self.insert_line_numbers(autopep8.fix_code(source)))
            raise exc
        _COMPILED_FN_CACHE[cache_key] = fn
        return fn

    # ----- Helper: get/set cfun parameter by index -----

    @staticmethod
    def _cfun_get_param(cfun, pidx):
        """Get the pidx-th parameter of a cfun object.

        Maps param_idx (as used in sweep_descriptor) to the named attribute
        of each coupling function type.  Mirrors the _cfun_params() layout:

          linear:                [a, b]
          scaling:               [a]
          sigmoidal:             [a, sigma, midpoint, cmin, cmax]
          sigmoidal_jr:          [a, cmin, cmax, r, midpoint]  (classic)
          sigmoidal_jr_legacy:   [a, e0, r, v0]                 (legacy)
          kuramoto:              [a, 1/N]
          difference:             [a]
          tanh:                  [a, b, midpoint, sigma]
          pre_sigmoidal:         [H, Q, G, P, theta]
          pre_sigmoidal_dynamic: [H, Q, G, P, 0]
        """
        from tvb.simulator.hybrid.coupling import (
            Linear, Scaling, Sigmoidal, SigmoidalJansenRit,
            Kuramoto as KuramotoCfun, Difference,
            HyperbolicTangent, PreSigmoidal,
        )
        if cfun is None:
            if pidx == 0:
                return 1.0
            raise IndexError(f"No parameter index {pidx} for None cfun")
        if isinstance(cfun, Linear):
            return [float(cfun.a[0]), float(cfun.b[0])][pidx]
        elif isinstance(cfun, Scaling):
            if pidx == 0:
                return float(cfun.a[0])
            raise IndexError(f"Scaling has only 1 parameter (index 0), got {pidx}")
        elif isinstance(cfun, Sigmoidal):
            return [float(cfun.a[0]), float(cfun.sigma[0]),
                    float(cfun.midpoint[0]), float(cfun.cmin[0]),
                    float(cfun.cmax[0])][pidx]
        elif isinstance(cfun, SigmoidalJansenRit):
            # Legacy indices 0-3 preserved for backward compat with saved sweeps.
            # Classic indices 4-6 are new.
            if pidx <= 3:
                return [float(cfun.a[0]), float(cfun.e0[0]),
                        float(cfun.r[0]), float(cfun.v0[0])][pidx]
            elif pidx <= 6:
                return [float(cfun.cmin[0]), float(cfun.cmax[0]),
                        float(cfun.midpoint[0])][pidx - 4]
            raise IndexError(
                f"SigmoidalJansenRit has 7 sweepable parameters (0-6), got {pidx}"
            )
        elif isinstance(cfun, KuramotoCfun):
            if pidx == 0:
                return float(cfun.a[0])
            raise IndexError(f"Kuramoto: only param_idx=0 is sweepable, got {pidx}")
        elif isinstance(cfun, Difference):
            if pidx == 0:
                return float(cfun.a[0])
            raise IndexError(f"Difference has only 1 sweepable parameter, got {pidx}")
        elif isinstance(cfun, HyperbolicTangent):
            # Indices 0-2 preserved for backward compat with saved sweeps;
            # b is new at index 3.
            if pidx <= 2:
                return [float(cfun.a[0]), float(cfun.midpoint[0]),
                        float(cfun.sigma[0])][pidx]
            elif pidx == 3:
                return float(cfun.b[0])
            raise IndexError(
                f"HyperbolicTangent has 4 sweepable parameters (0-3), got {pidx}"
            )
        elif isinstance(cfun, PreSigmoidal):
            return [float(cfun.H[0]), float(cfun.Q[0]), float(cfun.G[0]),
                    float(cfun.P[0]), float(cfun.theta[0])][pidx]
        else:
            raise TypeError(f"Unknown cfun type: {type(cfun).__name__}")

    @staticmethod
    def _cfun_set_param(cfun, pidx, value):
        """Set the pidx-th parameter of a cfun object.

        Sets the named attribute corresponding to param_idx.
        """
        from tvb.simulator.hybrid.coupling import (
            Linear, Scaling, Sigmoidal, SigmoidalJansenRit,
            Kuramoto as KuramotoCfun, Difference,
            HyperbolicTangent, PreSigmoidal,
        )
        if isinstance(cfun, Linear):
            for i, attr in enumerate(['a', 'b']):
                setattr(cfun, attr, np.array([float(value)])) if i == pidx else None
            setattr(cfun, ['a', 'b'][pidx], np.array([float(value)]))
        elif isinstance(cfun, Scaling):
            assert pidx == 0, f"Scaling: only param_idx=0, got {pidx}"
            cfun.a = np.array([float(value)])
        elif isinstance(cfun, Sigmoidal):
            setattr(cfun, ['a', 'sigma', 'midpoint', 'cmin', 'cmax'][pidx],
                    np.array([float(value)]))
            setattr(cfun, ['a', 'sigma', 'midpoint', 'cmin', 'cmax'][pidx],
                    np.array([float(value)]))
        elif isinstance(cfun, SigmoidalJansenRit):
            # Legacy indices 0-3 preserved for backward compat with saved sweeps.
            # Classic indices 4-6 are new.
            if pidx <= 3:
                setattr(cfun, ['a', 'e0', 'r', 'v0'][pidx],
                        np.array([float(value)]))
            elif pidx <= 6:
                setattr(cfun, ['cmin', 'cmax', 'midpoint'][pidx - 4],
                        np.array([float(value)]))
            else:
                raise IndexError(
                    f"SigmoidalJansenRit has 7 sweepable parameters (0-6), got {pidx}"
                )
        elif isinstance(cfun, KuramotoCfun):
            assert pidx == 0, f"Kuramoto: only param_idx=0, got {pidx}"
            cfun.a = np.array([float(value)])
        elif isinstance(cfun, Difference):
            assert pidx == 0, f"Difference: only param_idx=0, got {pidx}"
            cfun.a = np.array([float(value)])
        elif isinstance(cfun, HyperbolicTangent):
            # Indices 0-2 preserved for backward compat; b is new at index 3.
            setattr(cfun, ['a', 'midpoint', 'sigma', 'b'][pidx], np.array([float(value)]))
        elif isinstance(cfun, PreSigmoidal):
            setattr(cfun, ['H', 'Q', 'G', 'P', 'theta'][pidx], np.array([float(value)]))
        else:
            raise TypeError(f"Unknown cfun type: {type(cfun).__name__}")

    # ===================================================================
    # Unified sweep API
    # ===================================================================

    @staticmethod
    def _resolve_named_params(network_set, params):
        """Resolve a named-parameter dict to ``(sweep_descriptor, sweep_values)``."""
        # Collect all projections with names
        # Each entry: (lookup_name, proj, actual_name_for_descriptor)
        all_projs = []
        for proj in network_set.projections:
            actual_name = f"{proj.source.name}_to_{proj.target.name}"
            all_projs.append((actual_name, proj, actual_name))
        for sn in network_set.subnets:
            for p in sn.projections:
                raw_name = getattr(p, "name", None) or "intra"
                qualified = f"{sn.name}.{raw_name}" if raw_name == "intra" else raw_name
                all_projs.append((qualified, p, raw_name))

        dims = []
        for pname, values in params.items():
            dims.append(np.asarray(values, dtype=np.float32))

        n_sweeps = len(dims[0])
        for i, d in enumerate(dims):
            if len(d) != n_sweeps:
                raise ValueError(f"All param arrays must be same length; param {i} has {len(d)} vs {n_sweeps}")

        sweep_values = (np.column_stack(dims).astype(np.float32)
                        if len(dims) > 1 else dims[0].reshape(-1, 1))
        sweep_descriptor = []

        for dim_idx, (key, _values) in enumerate(params.items()):
            resolved = False

            # Try alias first ('coupling_scale', 'scale', etc.)
            if key in _NAMED_PARAM_ALIASES:
                alias = _NAMED_PARAM_ALIASES[key]
                for lookup, proj, actual in all_projs:
                    cfun = proj.cfun
                    if cfun is None:
                        continue
                    cfun_cls = type(cfun).__name__
                    attr = alias["attr"]
                    if (cfun_cls, attr) in _CFUN_ATTR_TO_IDX:
                        sweep_descriptor.append({
                            "type": "cfun", "projection": actual,
                            "param_idx": _CFUN_ATTR_TO_IDX[(cfun_cls, attr)],
                        })
                        resolved = True
                        break
                if resolved:
                    continue

            # Proj.attr: try "{proj}.{attr}" before model params so that
            # "ctx.intra.b" resolves to the intra-projection's cfun 'b' 
            # rather than treating "intra.b" as a model parameter.
            if "." in key:
                proj_name, attr = key.rsplit(".", 1)
                for lookup, proj, actual in all_projs:
                    if lookup == proj_name or lookup.endswith("." + proj_name):
                        cfun = proj.cfun
                        if cfun is not None:
                            cfun_cls = type(cfun).__name__
                            if (cfun_cls, attr) in _CFUN_ATTR_TO_IDX:
                                sweep_descriptor.append({
                                    "type": "cfun", "projection": actual,
                                    "param_idx": _CFUN_ATTR_TO_IDX[(cfun_cls, attr)],
                                })
                                resolved = True
                                break
                if resolved:
                    continue

            # Model param: "subnet.param" — only if param doesn't match
            # a known cfun attribute for any projection of that subnet.
            if "." in key:
                parts = key.split(".", 1)
                if len(parts) == 2 and any(sn.name == parts[0] for sn in network_set.subnets):
                    sname, param = parts
                    sweep_descriptor.append({"type": "model", "subnet": sname, "param": param})
                    continue

            raise ValueError(
                f"Cannot resolve sweep parameter '{key}'. Use 'coupling_scale', "
                f"'{{proj}}.{{attr}}', or '{{subnet}}.{{param}}'. "
                f"Projections: {[lookup for lookup, _, _ in all_projs]}. "
                f"Subnets: {[sn.name for sn in network_set.subnets]}."
            )

        return sweep_descriptor, sweep_values

    def sweep(
        self,
        network_set,
        params,
        nstep: int = 100,
        *,
        backend: str = "auto",
        n_workers: int = 1,
        monitor: str = "tavg",
        monitor_period: int = 1,
        bold_period: Optional[float] = None,
        chunk_size: Optional[int] = None,
        initial_states: Optional[list] = None,
        node_indices: Optional[dict] = None,
    ) -> "SweepResult":
        """Run a parameter sweep — auto dispatches to GPU or multi-core CPU.

        Parameters
        ----------
        params : dict
            Named parameters to sweep.  Keys → 1-D arrays.  All same length.
            ``'coupling_scale'`` or ``'scale'``: first projection's scaling.
            ``'{proj_name}.{attr}'``: named projection cfun attribute.
            ``'{subnet}.{param}'``: model parameter on a subnet.
        backend : str
            ``'auto'`` (try CUDA → fallback CPU), ``'cpu'``, or ``'cuda'``.
        n_workers : int
            CPU worker processes (fork-based, ignored for CUDA).
        monitor : str
            ``'tavg'``, ``'raw'``, or ``'subsample'``.
        """
        import time as _time_mod

        sweep_descriptor, sweep_values = self._resolve_named_params(network_set, params)

        use_cuda = False
        if backend == "cuda":
            use_cuda = True
        elif backend == "auto":
            try:
                from numba import cuda as _cuda
                use_cuda = _cuda.is_available()
            except ImportError:
                pass

        if use_cuda:
            return self._sweep_cuda(
                network_set, sweep_descriptor, sweep_values, nstep,
                monitor, monitor_period, bold_period, chunk_size,
                initial_states, node_indices)
        else:
            return self._sweep_cpu(
                network_set, sweep_descriptor, sweep_values, nstep,
                n_workers, initial_states, node_indices)

    def _sweep_cuda(self, network_set, sweep_descriptor, sweep_values,
                     nstep, monitor, monitor_period, bold_period,
                     chunk_size, initial_states, node_indices):
        import time as _time_mod
        from tvb.simulator.backend.nb_hybrid_cuda_sweep_backend import NbHybridCUDASweepBackend

        cuda_backend = NbHybridCUDASweepBackend()
        compiled = cuda_backend.compile_sweep(network_set, sweep_descriptor=sweep_descriptor)
        kwargs = dict(sweep_values=sweep_values, nstep=nstep, monitor_type=monitor,
                     monitor_period=monitor_period, node_indices=node_indices)
        if initial_states is not None: kwargs["initial_states"] = initial_states
        if bold_period is not None: kwargs["bold_period"] = bold_period
        if chunk_size is not None: kwargs["chunk_size"] = chunk_size

        t0 = _time_mod.perf_counter()
        raw = compiled.run(**kwargs)
        elapsed = _time_mod.perf_counter() - t0

        subnet_names = [sn.name for sn in network_set.subnets]
        result = SweepResult(sweep_values=sweep_values, backend="cuda", elapsed=elapsed,
                            snapshot=raw.get("snapshot"))
        if raw.get("tavg"):
            result.tavg = {}
            for si, sname in enumerate(subnet_names):
                arr = raw["tavg"][si]
                # GPU may return shape (n_sweeps, n_voi, N, n_modes) where n_modes > 1;
                # only mode-0 contains the summed signal.  Squeeze to match CPU format.
                if arr.ndim == 4 and arr.shape[3] > 1:
                    arr = arr[:, :, :, 0:1]  # keep dim but select mode 0
                result.tavg[sname] = arr
        if raw.get("merged_tavg") is not None:
            arr = raw["merged_tavg"]
            if arr.ndim == 4 and arr.shape[3] > 1:
                arr = arr[:, :, :, 0:1]
            result.merged_tavg = arr
        elif result.tavg:
            result.merged_tavg = np.concatenate(list(result.tavg.values()), axis=2)
        if raw.get("ctavg"):
            result.ctavg = dict(zip(subnet_names, raw["ctavg"]))
        if raw.get("raw"):
            result.raw = dict(zip(subnet_names, raw["raw"]))
        if raw.get("bold"):
            result.bold = dict(zip(subnet_names, raw["bold"]))
        dt = network_set.subnets[0].scheme.dt
        nc = raw["tavg"][0].shape[0] if raw.get("tavg") else nstep
        result.times = np.arange(0.5, nc + 0.5) * dt * nstep / nc if nc > 0 else np.array([])
        return result

    def _stack_cpu_results(self, raw_results, network_set, node_indices, sweep_values, backend_label, elapsed):
        """Stack CPU list-of-tuples into SweepResult.

        raw_results: list of n_sweeps tuples, each tuple has n_subnets entries
        of (times, data, ctavg).  data shape: (n_chunks, n_voi, N, modes).
        Stacks into (n_sweeps, n_chunks, n_voi, N, modes) preserving the
        time (chunk) dimension so callers can access per-timestep traces.
        """
        subnet_names = [sn.name for sn in network_set.subnets]
        tavg_dict = {}
        ctavg_dict = {}
        for si, sname in enumerate(subnet_names):
            # Preserve time (chunk) dimension for per-timestep traces
            tavg_arr = np.stack([r[si][1] for r in raw_results], axis=0)
            ctavg_arr = np.stack([r[si][2] for r in raw_results], axis=0)
            tavg_dict[sname] = tavg_arr
            ctavg_dict[sname] = ctavg_arr
        # Merge along node axis
        if node_indices and len(node_indices) > 0:
            n_global = max(max(idxs) for idxs in node_indices.values()) + 1
            ref = list(tavg_dict.values())[0]
            merged = np.zeros((ref.shape[0], ref.shape[1], ref.shape[2], n_global, ref.shape[4]), dtype=np.float32)
            for sname in subnet_names:
                if sname in node_indices:
                    merged[:, :, :, node_indices[sname], :] = tavg_dict[sname]
            merged_tavg = merged
        else:
            # Concatenate along node axis only if all subnets share the same n_voi
            vois = set(a.shape[2] for a in tavg_dict.values())
            if len(vois) == 1:
                merged_tavg = np.concatenate(list(tavg_dict.values()), axis=3)
            else:
                merged_tavg = None  # VOI counts differ — can't merge
        times = raw_results[0][0][0] if raw_results else np.array([])
        return SweepResult(tavg=tavg_dict, merged_tavg=merged_tavg, ctavg=ctavg_dict,
                          times=times, sweep_values=sweep_values, backend=backend_label, elapsed=elapsed)

    def _sweep_cpu(self, network_set, sweep_descriptor, sweep_values,
                    nstep, n_workers, initial_states, node_indices):
        import time as _time_mod
        if n_workers > 1:
            # Use prange-based parallel sweep instead of fork
            return self._sweep_cpu_prange(
                network_set, sweep_descriptor, sweep_values,
                nstep, initial_states, node_indices)
        t0 = _time_mod.perf_counter()
        raw = self.run_sweep(network_set, sweep_values=sweep_values, nstep=nstep,
                             sweep_descriptor=sweep_descriptor, initial_states=initial_states)
        elapsed = _time_mod.perf_counter() - t0
        return self._stack_cpu_results(raw, network_set, node_indices, sweep_values, "cpu-seq", elapsed)


    def _sweep_cpu_prange(self, network_set, sweep_descriptor, sweep_values,
                          nstep, initial_states, node_indices):
        """Multi-core CPU sweep using Numba prange (single-process threading).

        Compiles a ``@nb.njit(parallel=True)`` sweep kernel via a Mako
        template (``nb-hybrid-sweep-cpu.py.mako``), appended to the
        single-sim module so ``sweep_kernel`` can call ``network_chunk``
        directly from inside ``nb.prange``.  Each thread operates on its
        own slice of per-sweep arrays, giving true multi-core parallelism
        without the fork-safety issues of multiprocessing.
        """
        from tvb.simulator.backend.nb_hybrid_sweep_cpu import (
            compile_sweep_kernel, run_sweep_prange)

        # Ensure single-sim function is compiled (and module is cached)
        compiled = self.compile(network_set, eager=True)
        analysis = compiled._analysis

        # Compile the sweep kernel (Mako template + single-sim source)
        kernel_fn = compile_sweep_kernel(self, analysis)

        # Run the sweep
        return run_sweep_prange(
            kernel_fn, analysis, network_set, sweep_descriptor,
            sweep_values, nstep, self, initial_states=initial_states)



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
        """Run parameter sweep sequentially on CPU.

        Each sweep point calls run_network() internally. Results are
        returned as a list of per-sweep-point tuples matching the
        run_network() return format.

        Parameters
        ----------
        sweep_values : ndarray (n_sweeps,) or (n_sweeps, n_sweep_dims)
        sweep_descriptor : list of dict, optional
            [{type: 'cfun', projection: 'proj_AB', param_idx: 0},
             {type: 'model', subnet: 'A', param: 'tau_E'}]
        """
        sweep_values = np.asarray(sweep_values, dtype=np.float32)
        if sweep_values.ndim == 1:
            sweep_values = sweep_values.reshape(-1, 1)

        if sweep_descriptor is None:
            if network_set.projections:
                # Use naming convention for projections
                first_proj = network_set.projections[0]
                proj_name = f"{first_proj.source.name}_to_{first_proj.target.name}"
                sweep_descriptor = [{'type': 'cfun', 'projection': proj_name,
                                     'param_idx': 0}]
            else:
                sweep_descriptor = []

        n_sweeps = sweep_values.shape[0]
        results = []

        for tid in range(n_sweeps):
            sv = sweep_values[tid]

            # Apply sweep values to cfun/model params
            restore = {}
            for dim, desc in enumerate(sweep_descriptor):
                if desc['type'] == 'cfun':
                    pname = desc['projection']
                    pidx = desc.get('param_idx', 0)
                    # Find projection by naming convention
                    # Inter: "{src}_to_{tgt}", Intra: "intra"
                    matched_proj = None
                    for proj in network_set.projections:
                        expected_name = f"{proj.source.name}_to_{proj.target.name}"
                        if expected_name == pname:
                            matched_proj = proj
                            break
                    if matched_proj is None:
                        for sn in network_set.subnets:
                            for p in sn.projections:
                                expected_name = getattr(p, 'name', None) or 'intra'
                                if expected_name == pname:
                                    matched_proj = p
                                    break
                            if matched_proj is not None:
                                break
                    if matched_proj is None:
                        raise ValueError(f"Projection '{pname}' not found in sweep")
                    key = ('cfun', pname, pidx)
                    restore[key] = self._cfun_get_param(matched_proj.cfun, pidx)
                    self._cfun_set_param(matched_proj.cfun, pidx, float(sv[dim]))
                elif desc['type'] == 'model':
                    sname = desc['subnet']
                    param = desc['param']
                    for sn in network_set.subnets:
                        if sn.name == sname:
                            key = ('model', sname, param)
                            val = float(getattr(sn.model, param))
                            restore[key] = val
                            setattr(sn.model, param, np.array([float(sv[dim])]))

            # Run single simulation
            kwargs = dict(initial_states=initial_states, print_source=print_source)
            if chunk_size is not None:
                kwargs['chunk_size'] = chunk_size
            result = self.run_network(network_set, nstep=nstep, **kwargs)
            results.append(result)

            # Restore original values
            for key, orig in restore.items():
                if key[0] == 'cfun':
                    pname, pidx = key[1], key[2]
                    matched_proj = None
                    for proj in network_set.projections:
                        expected_name = f"{proj.source.name}_to_{proj.target.name}"
                        if expected_name == pname:
                            matched_proj = proj
                            break
                    if matched_proj is None:
                        for sn in network_set.subnets:
                            for p in sn.projections:
                                expected_name = getattr(p, 'name', None) or 'intra'
                                if expected_name == pname:
                                    matched_proj = p
                                    break
                            if matched_proj is not None:
                                break
                    if matched_proj is not None:
                        self._cfun_set_param(matched_proj.cfun, pidx, orig)
                elif key[0] == 'model':
                    for sn in network_set.subnets:
                        if sn.name == key[1]:
                            setattr(sn.model, key[2], np.array([orig]))

        return results
