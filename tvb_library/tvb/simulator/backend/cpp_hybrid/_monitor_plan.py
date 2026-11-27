# -*- coding: utf-8 -*-
"""Monitor routing, attachment spec lowering, and output reassembly for cpp_hybrid.

Extracted from ``backend.py`` so ``CompiledCppSim.run()`` stays concise while
preserving all existing seams and monkeypatch points on ``backend.py``.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
from typing import Optional

import numpy as np

from tvb.simulator.backend.hybrid_common import (
    _aggregate_raw_outputs,
    _apply_monitors,
    _can_merge_subnets,
    _compute_chunk_size,
)
from tvb.simulator.monitors import BalloonWindkessel, EEG, MEG, iEEG

REGIONAL_PROJECTION_MONITORS = (EEG, MEG, iEEG)
RECORDER_SLOT_BASE = 10000

_BALLOON_WINDKESSEL_NOMINAL_FALLBACK = dict(
    signal_decay_s=0.65,
    flow_feedback_s2=0.41,
    transit_time_s=0.98,
    grubb_exponent=0.32,
    resting_extraction=0.4,
    intravascular_ratio=0.5,
    frequency_offset_hz=40.3,
    relaxation_rate_per_s=25.0,
    echo_time_s=0.04,
    resting_blood_volume=0.04,
)


@dataclasses.dataclass
class MonitorPlan:
    sn_mons_map: Optional[dict]
    kernel_global_specs: list
    complete_window_isteps: dict
    gathered_window_sources: Optional[list]
    balloon_windkessel_specs: dict
    kernel_persn_specs: list
    py_global_indices: set
    kernel_mon_indices: set
    has_py_bold: bool
    has_py_tavg: bool
    chunk_size: int
    sn_kernel_specs: dict
    sn_py_fallbacks: dict
    local_window_specs: dict
    need_py_sn_tavg: bool
    need_py_sn_ctavg: bool
    need_tavg: bool
    need_ctavg: bool
    execution_chunk_size: int


def normalize_subnet_monitors(analysis, subnet_monitors):
    """Normalize optional per-subnet Recorder monitors into ``{sn.name: [Monitor, ...]}``."""
    if subnet_monitors is None:
        return None
    local_lists = (
        subnet_monitors.values()
        if isinstance(subnet_monitors, dict)
        else subnet_monitors
    )
    if any(
        isinstance(getattr(m, "monitor", m), REGIONAL_PROJECTION_MONITORS)
        for ms in local_lists
        for m in ms
    ):
        raise NotImplementedError(
            "C++ EEG/MEG/iEEG cannot be local or Recorder monitors"
        )
    if isinstance(subnet_monitors, dict):
        return {
            sn.name: [
                getattr(m, "monitor", m)
                for m in subnet_monitors.get(sn.name, ())
            ]
            for sn in analysis.subnets
        }
    return {
        sn.name: [
            getattr(m, "monitor", m)
            for m in (
                subnet_monitors[si] if si < len(subnet_monitors) else ()
            )
        ]
        for si, sn in enumerate(analysis.subnets)
    }


def plan_monitors(
    compiled,
    nstep: int,
    chunk_size: Optional[int],
    monitors: Optional[list],
    subnet_monitors,
    dt: float,
    kwargs: dict,
    *,
    can_merge_subnets_fn=_can_merge_subnets,
    compute_chunk_size_fn=_compute_chunk_size,
) -> MonitorPlan:
    """Route global and per-subnet monitors into C++ kernel specs or Python fallbacks."""
    from tvb.simulator.monitors import (
        AfferentCoupling,
        AfferentCouplingTemporalAverage,
        Bold,
        GlobalAverage,
        Projection,
        SpatialAverage,
        TemporalAverage,
    )

    backend = compiled._backend
    analysis = compiled._analysis
    network_set = compiled._network_set
    width = compiled._width
    bold_states = compiled._bold_states

    sn_mons_map = normalize_subnet_monitors(analysis, subnet_monitors)

    n_sn = len(analysis.subnets)
    should_merge = (n_sn > 1) and can_merge_subnets_fn(analysis.subnets)
    kernel_global_specs = []   # (mi, m) attached via set_global_monitors
    complete_window_isteps: dict[int, int] = {}
    gathered_window_sources_val = None
    balloon_windkessel_specs = {}
    kernel_persn_specs = []    # (mi, m) attached via set_monitors(si, ...)
    py_global_indices = set()
    kernel_mon_indices = set()
    has_py_bold = False
    has_py_tavg = False

    if monitors is not None:
        for mi, m in enumerate(monitors):
            if isinstance(m, BalloonWindkessel):
                if n_sn > 1:
                    if gathered_window_sources_val is None:
                        gathered_window_sources_val = compiled._gathered_window_sources(dt)
                    if not gathered_window_sources_val:
                        raise ValueError(
                            "BalloonWindkessel requires compatible regional sources and explicit node_indices"
                        )
                balloon_windkessel_specs[mi] = compiled._balloon_windkessel_spec(mi, m, dt)
                kernel_global_specs.append((mi, m))
                kernel_mon_indices.add(mi)
                continue
            if isinstance(m, REGIONAL_PROJECTION_MONITORS):
                if not backend.enable_kernel_monitors:
                    raise NotImplementedError(
                        "C++ EEG/MEG/iEEG require native monitor collection"
                    )
                if n_sn != 1:
                    raise NotImplementedError(
                        "C++ EEG/MEG/iEEG require one regional subnetwork"
                    )
                sn = analysis.subnets[0]
                if (
                    width != 1
                    or kwargs.get("width", 1) != 1
                    or sn.k != 1
                    or sn.n_modes != 1
                    or sn.dt != dt
                ):
                    raise NotImplementedError(
                        "C++ EEG/MEG/iEEG require width 1 and one source clock and mode"
                    )
                indices = network_set.subnets[0].node_indices
                if indices is not None and not np.array_equal(
                    indices, np.arange(sn.n_nodes)
                ):
                    raise ValueError(
                        "C++ EEG/MEG/iEEG require absent or identity node_indices"
                    )
                complete_window_isteps[mi] = backend._complete_window_istep(m, dt)
                kernel_global_specs.append((mi, m))
                kernel_mon_indices.add(mi)
                continue
            if (
                n_sn > 1
                and backend.enable_kernel_monitors
                and width == kwargs.get("width", 1) == 1
                and isinstance(m, (TemporalAverage, GlobalAverage, SpatialAverage))
                and not isinstance(m, AfferentCouplingTemporalAverage)
                and (
                    not isinstance(m, SpatialAverage)
                    or getattr(m, "spatial_mean", None) is not None
                )
                and not (
                    isinstance(m, TemporalAverage)
                    and bold_states
                    and any(
                        k[0] == "temporal_average" and k[1] == mi
                        for k in bold_states
                    )
                )
            ):
                if gathered_window_sources_val is None:
                    gathered_window_sources_val = compiled._gathered_window_sources(dt)
                if gathered_window_sources_val:
                    complete_window_isteps[mi] = backend._complete_window_istep(
                        m, dt, scope="gathered global", clock_name="master"
                    )
                    kernel_global_specs.append((mi, m))
                    kernel_mon_indices.add(mi)
                    continue
            if (
                n_sn == 1
                and isinstance(m, (TemporalAverage, GlobalAverage, SpatialAverage))
                and not isinstance(m, AfferentCouplingTemporalAverage)
            ):
                sn = analysis.subnets[0]
                if not backend.enable_kernel_monitors:
                    raise NotImplementedError(
                        "C++ regional complete windows require native monitor collection"
                    )
                if width != 1 or sn.k != 1 or sn.n_modes != 1 or sn.dt != dt:
                    raise NotImplementedError(
                        "C++ regional complete windows require width 1 and one source clock and mode"
                    )
                complete_window_isteps[mi] = backend._complete_window_istep(m, dt)
                kernel_global_specs.append((mi, m))
                kernel_mon_indices.add(mi)
                continue
            if isinstance(m, Projection):
                if (
                    getattr(m, "_gain", None) is None
                    and getattr(m, "projection", None) is None
                ):
                    raise ValueError(
                        f"Projection monitor {type(m).__name__} has no gain matrix "
                        "configured. Set the gain matrix before running (e.g. via "
                        "config_for_sim or by setting m._gain directly)."
                    )
            can_kernel = (
                backend.enable_kernel_monitors
                and backend._kernel_collectable(m)
            )
            if can_kernel and bold_states is not None:
                # Respect externally supplied Python monitor state dicts
                if isinstance(m, Bold) and any(
                    k[0] == "bold" and k[1] == mi for k in bold_states
                ):
                    can_kernel = False
                elif (
                    isinstance(m, TemporalAverage)
                    and not isinstance(m, AfferentCouplingTemporalAverage)
                    and any(
                        k[0] == "temporal_average" and k[1] == mi
                        for k in bold_states
                    )
                ):
                    can_kernel = False

            if can_kernel and isinstance(m, (GlobalAverage, SpatialAverage, Projection)):
                if should_merge or n_sn == 1:
                    total_nodes = (
                        max(int(sn.node_indices.max()) for sn in analysis.subnets) + 1
                        if should_merge
                        else analysis.subnets[0].n_nodes
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
                        if not all(sn.n_nodes == sm_cols for sn in analysis.subnets):
                            can_kernel = False
                    if isinstance(m, Projection):
                        g_cols = np.asarray(m.gain).shape[1]
                        if not all(sn.n_nodes == g_cols for sn in analysis.subnets):
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
            chunk_size = (
                compute_chunk_size_fn(companions, dt)
                if companions
                else min(
                    nstep or 1,
                    *(s["istep"] for s in balloon_windkessel_specs.values()),
                )
            )
        else:
            chunk_size = compute_chunk_size_fn(monitors, dt) if monitors is not None else 1
    if monitors is not None:
        backend._validate_monitors(monitors, chunk_size)

    # Partition per-subnet monitors (Recorder monitors)
    sn_kernel_specs = {sn.name: [] for sn in analysis.subnets}
    sn_py_fallbacks = {sn.name: [] for sn in analysis.subnets}
    local_window_specs = {}
    need_py_sn_tavg = False
    need_py_sn_ctavg = False
    if sn_mons_map is not None:
        for sn in analysis.subnets:
            for ri, rm in enumerate(sn_mons_map.get(sn.name, ())):
                if isinstance(rm, BalloonWindkessel):
                    local_window_specs[sn.name, ri] = compiled._recorder_window_spec(sn, ri, rm)
                    sn_kernel_specs[sn.name].append((ri, rm))
                elif (
                    isinstance(rm, (TemporalAverage, GlobalAverage, SpatialAverage))
                    and not isinstance(rm, AfferentCouplingTemporalAverage)
                ):
                    local_window_specs[sn.name, ri] = compiled._recorder_window_spec(sn, ri, rm)
                    sn_kernel_specs[sn.name].append((ri, rm))
                elif backend.enable_kernel_monitors and backend._kernel_collectable(rm):
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

    return MonitorPlan(
        sn_mons_map=sn_mons_map,
        kernel_global_specs=kernel_global_specs,
        complete_window_isteps=complete_window_isteps,
        gathered_window_sources=gathered_window_sources_val,
        balloon_windkessel_specs=balloon_windkessel_specs,
        kernel_persn_specs=kernel_persn_specs,
        py_global_indices=py_global_indices,
        kernel_mon_indices=kernel_mon_indices,
        has_py_bold=has_py_bold,
        has_py_tavg=has_py_tavg,
        chunk_size=chunk_size,
        sn_kernel_specs=sn_kernel_specs,
        sn_py_fallbacks=sn_py_fallbacks,
        local_window_specs=local_window_specs,
        need_py_sn_tavg=need_py_sn_tavg,
        need_py_sn_ctavg=need_py_sn_ctavg,
        need_tavg=need_tavg,
        need_ctavg=need_ctavg,
        execution_chunk_size=execution_chunk_size,
    )


def gathered_window_sources(analysis, network_set, dt: float):
    """Build gather declarations for homogeneous multi-subnet complete windows."""
    sources = analysis.subnets
    declared = network_set.subnets
    indices = [sn.node_indices for sn in declared]
    if (
        dt != analysis.dt
        or any(ix is None for ix in indices)
        or any(float(sn.scheme.dt) != source.dt for sn, source in zip(declared, sources))
        or any(sn.n_modes != 1 for sn in sources)
        or (all(sn.k == 1 for sn in sources) and any(sn.dt != dt for sn in sources))
    ):
        return []

    # Native observation triples must agree, and so must the names of the
    # state components they refer to (equal offsets alone do not suffice).
    def observation_terms(sn):
        names = tuple(map(str, sn.model.state_variables))
        return [
            names[a] if kind == 0 else f"{names[a]}-{names[b]}"
            for kind, a, b in zip(sn.voi[::3], sn.voi[1::3], sn.voi[2::3])
        ]

    first = sources[0]
    observed_terms = observation_terms(first)
    if not observed_terms or any(
        sn.voi != first.voi or observation_terms(sn) != observed_terms
        for sn in sources[1:]
    ):
        return []

    total_nodes = sum(sn.n_nodes for sn in sources)
    if total_nodes > np.iinfo(np.int32).max:
        raise ValueError(
            "C++ gathered global complete windows: node gather extent out of range"
        )
    gather = []
    seen = set()
    for sn, ix in zip(sources, indices):
        ix = np.asarray(ix)
        if (
            sn.n_nodes < 1
            or ix.shape != (sn.n_nodes,)
            or ix.dtype.kind not in "iu"
            or np.any(ix < 0)
            or np.any(ix >= total_nodes)
        ):
            raise ValueError(
                "C++ gathered global complete windows: invalid node_indices"
            )
        nodes = ix.tolist()
        seen.update(nodes)
        gather.append(
            dict(
                n_voi=len(observed_terms),
                n_node=sn.n_nodes,
                node_indices=nodes,
                source_name=sn.name,
                source_dt=sn.dt,
                observed_terms=observed_terms,
            )
        )
    if len(seen) != total_nodes:
        raise ValueError(
            "C++ gathered global complete windows require complete, disjoint node_indices"
        )
    return gather


def local_balloon_windkessel_spec(
    backend,
    analysis,
    width: int,
    sn,
    index: int,
    monitor,
    *,
    bw_nominal=_BALLOON_WINDKESSEL_NOMINAL_FALLBACK,
):
    """Local Recorder BW window on this subnetwork's own physical clock."""
    if not backend.enable_kernel_monitors:
        raise NotImplementedError(
            "BalloonWindkessel requires native monitor collection for local Recorders"
        )
    if width not in (1, 4, 8, 16):
        raise NotImplementedError(
            "BalloonWindkessel local Recorders require one of the SIMD widths 1, 4, 8 or 16"
        )
    if sn.n_modes != 1:
        raise NotImplementedError(
            "BalloonWindkessel local Recorders require one source mode"
        )
    if (
        sn.n_nodes < 1
        or sn.n_nodes > np.iinfo(np.int32).max
        or sn.n_nodes * width > np.iinfo(np.int32).max
    ):
        raise ValueError(
            "BalloonWindkessel local Recorders require a positive in-range region "
            "and region-by-lane extent"
        )
    # Reuses the monitor's native binary32 clock and 10 ms period validation.
    period = monitor._output_ticks(float(sn.dt))
    voi = getattr(monitor, "variables_of_interest", None)
    n_voi = len(sn.voi) // 3
    if (
        voi is None
        or np.shape(voi) != (1,)
        or np.asarray(voi).dtype.kind not in "iu"
        or not 0 <= int(voi[0]) < n_voi
    ):
        raise ValueError(
            "BalloonWindkessel local Recorders require exactly one valid observed "
            "VOI index"
        )
    baseline = getattr(monitor, "drive_baseline", None)
    scale = getattr(monitor, "drive_scale", None)
    if (
        baseline is None
        or scale is None
        or not np.isfinite(baseline)
        or not np.isfinite(scale)
    ):
        raise ValueError(
            "BalloonWindkessel local Recorders require explicit finite "
            "drive_baseline and drive_scale"
        )
    return dict(
        local_window=True,
        slot=RECORDER_SLOT_BASE + index,
        kind=11,
        istep=period,
        dt=float(sn.dt),
        master_dt=float(analysis.dt),
        cadence_K=sn.k,
        source_name=sn.name,
        node_indices=[] if sn.node_indices is None else sn.node_indices.tolist(),
        voi=[int(voi[0])],
        bw_drive_baseline=float(baseline),
        bw_drive_scale=float(scale),
        bw_stream_params=dict(bw_nominal),
    )


def recorder_window_spec(
    compiled,
    sn,
    index: int,
    monitor,
    *,
    bw_nominal=_BALLOON_WINDKESSEL_NOMINAL_FALLBACK,
):
    """Build a local Recorder complete-window attachment specification."""
    if isinstance(monitor, BalloonWindkessel):
        return compiled._local_balloon_windkessel_spec(sn, index, monitor)
    backend = compiled._backend
    if not backend.enable_kernel_monitors or compiled._width != 1 or sn.n_modes != 1:
        raise NotImplementedError(
            "C++ Recorder complete windows require native collection, width 1 and one source mode"
        )
    n_voi = len(sn.voi) // 3
    period = backend._complete_window_istep(monitor, float(sn.dt), scope="Recorder")
    spec = backend._monitor_attach_spec(
        RECORDER_SLOT_BASE + index,
        monitor,
        n_voi,
        sn.n_nodes,
        float(sn.dt),
        cadence_K=sn.k,
        complete_window_istep=period,
        local_window=True,
    )
    spec.update(
        master_dt=compiled._analysis.dt,
        source_name=sn.name,
        node_indices=[] if sn.node_indices is None else sn.node_indices.tolist(),
    )
    return spec


def balloon_windkessel_spec(
    backend,
    analysis,
    network_set,
    width: int,
    slot: int,
    monitor,
    dt: float,
    *,
    bw_nominal=_BALLOON_WINDKESSEL_NOMINAL_FALLBACK,
):
    """Build a global BalloonWindkessel stream attachment specification."""
    if not backend.enable_kernel_monitors:
        raise NotImplementedError("BalloonWindkessel requires native monitor collection")
    sn = analysis.subnets[0]
    if width not in (1, 4, 8, 16) or any(s.n_modes != 1 for s in analysis.subnets):
        raise NotImplementedError(
            "BalloonWindkessel requires a supported SIMD width and one source mode"
        )
    if len(analysis.subnets) == 1:
        if sn.k != 1 or sn.dt != dt:
            raise NotImplementedError("BalloonWindkessel requires one source clock")
        indices = network_set.subnets[0].node_indices
        if indices is not None:
            indices = np.asarray(indices)
        if sn.n_nodes < 1 or (
            indices is not None
            and (
                indices.shape != (sn.n_nodes,)
                or indices.dtype.kind not in "iu"
                or not np.array_equal(np.sort(indices), np.arange(sn.n_nodes))
            )
        ):
            raise ValueError(
                "BalloonWindkessel requires a full regional node permutation when node_indices are supplied"
            )
    voi = getattr(monitor, "variables_of_interest", None)
    voi_arr = None if voi is None else np.asarray(voi)
    if (
        voi_arr is None
        or voi_arr.shape != (1,)
        or voi_arr.dtype.kind not in "iu"
        or not 0 <= int(voi_arr[0]) < len(sn.voi) // 3
    ):
        raise ValueError(
            "BalloonWindkessel requires exactly one valid observed VOI index"
        )
    baseline = getattr(monitor, "drive_baseline", None)
    scale = getattr(monitor, "drive_scale", None)
    if (
        baseline is None
        or scale is None
        or not np.isfinite(baseline)
        or not np.isfinite(scale)
    ):
        raise ValueError(
            "BalloonWindkessel requires explicit finite drive_baseline and drive_scale"
        )
    return dict(
        slot=slot,
        kind=10,
        dt=dt,
        istep=monitor._output_ticks(dt),
        voi=[int(voi_arr[0])],
        bw_drive_baseline=baseline,
        bw_drive_scale=scale,
        bw_stream_params=dict(bw_nominal),
    )


def kernel_collectable(m) -> bool:
    """True when *m* is handled by the C++ ``kmon<W>`` monitor engines."""
    from tvb.simulator.monitors import (
        AfferentCoupling,
        Bold,
        BoldRegionROI,
        GlobalAverage,
        Projection,
        Raw,
        SpatialAverage,
        SubSample,
        TemporalAverage,
    )

    if isinstance(m, BoldRegionROI):
        return False
    if isinstance(m, Projection):
        return (
            getattr(m, "_gain", None) is not None
            or getattr(m, "projection", None) is not None
        )
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


def monitor_voi_list(m) -> list:
    voi = getattr(m, "variables_of_interest", None)
    if voi is None:
        return []
    arr = np.asarray(voi, dtype=np.int32).ravel()
    return arr.tolist() if arr.size > 0 else []


def checked_monitor_voi_list(m, n_src_v: int, scope: str) -> list:
    voi = m.variables_of_interest
    if voi is None:
        return []
    if (
        voi.ndim != 1
        or voi.dtype.kind not in "iu"
        or np.any(voi < 0)
        or np.any(voi >= n_src_v)
    ):
        raise ValueError(f"C++ {scope} require valid observed VOI indices")
    return voi.tolist()


def complete_window_istep(
    m, dt: float, scope: str = "regional", clock_name: str = "source"
) -> int:
    ticks = float(m.period) / dt
    if np.isfinite(ticks):
        nearest = round(ticks)
        # Allow a few binary64 rounding steps from period construction and
        # division, not a relative fraction of a source tick.
        if (
            1 <= nearest <= np.iinfo(np.int32).max
            and abs(ticks - nearest) <= 8 * np.spacing(float(nearest))
        ):
            return nearest
    raise ValueError(
        f"C++ {scope} {type(m).__name__} period {m.period!r} must be "
        f"commensurate with {clock_name} dt {dt!r} (a positive integer number of ticks)"
    )


def regional_projection_spec(m, n_src_v: int, n_space: int) -> dict:
    if m.obsnoise is not None:
        raise NotImplementedError("C++ EEG/MEG/iEEG require obsnoise=None")

    voi = checked_monitor_voi_list(m, n_src_v, "EEG/MEG/iEEG")

    # A lazy file cache can look regional by shape alone. Require the
    # caller to supply gain in local-region order through the setter.
    gain = np.asarray(m._gain if m._gain_is_explicit else None)
    if (
        gain.dtype.kind not in "fiu"
        or gain.ndim != 2
        or not gain.shape[0]
        or gain.shape[1] != n_space
    ):
        raise ValueError(
            "C++ EEG/MEG/iEEG require a finite preconfigured sensor-by-local-region gain"
        )
    with np.errstate(over="ignore"):
        gain = np.array(gain, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(gain).all():
        raise ValueError(
            "C++ EEG/MEG/iEEG require a finite preconfigured sensor-by-local-region gain"
        )

    n_sensors = gain.shape[0]
    sensors = getattr(m, "sensors", None)
    labels = usable = None
    if sensors is not None:
        labels = getattr(sensors, "labels", None)
        usable = sensors.usable
        if labels is None or labels.shape != (n_sensors,):
            raise ValueError(
                "C++ projection sensor labels must align with gain rows"
            )
        if usable is not None:
            if usable.shape != (n_sensors,):
                raise ValueError(
                    "C++ projection usable sensor mask must align with gain rows"
                )
            gain[~usable] = 0.0

    ref_vec = None
    if isinstance(m, EEG) and m.reference:
        if labels is None:
            raise ValueError("C++ EEG reference requires ordered sensor labels")
        if m.reference.lower() == "average":
            # Unusable channels contribute zero but still count in the
            # classic average denominator and remain on the output axis.
            ref_vec = np.full(n_sensors, 1.0 / n_sensors, dtype=np.float32)
        else:
            matches = np.flatnonzero(labels == m.reference)
            if matches.size != 1:
                raise ValueError(
                    f"C++ EEG reference {m.reference!r} must name one sensor"
                )
            ref_vec = np.zeros(n_sensors, dtype=np.float32)
            ref_vec[matches[0]] = 1.0
        if usable is not None:
            ref_vec[~usable] = 0.0

    return dict(kind=8, cadence_K=1, voi=voi, gain=gain, ref_vec=ref_vec)


def monitor_attach_spec(
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
    local_window=False,
    gathered_window=False,
):
    """Build a C++ ``kmon<W>`` spec dict for monitor *m*."""
    from tvb.datatypes import equations
    from tvb.simulator.monitors import (
        AfferentCoupling,
        AfferentCouplingTemporalAverage,
        Bold,
        GlobalAverage,
        Projection,
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
            "dt": eff_dt,
        }
        if local_window:
            spec.update(local_window=True, cadence_K=cadence_K)
        else:
            spec["complete_window"] = True
        if isinstance(m, REGIONAL_PROJECTION_MONITORS):
            spec.update(regional_projection_spec(m, n_src_v, n_space))
            return spec
        scope = (
            "Recorder"
            if local_window
            else "gathered global"
            if gathered_window
            else "regional"
        )
        spec["voi"] = checked_monitor_voi_list(m, n_src_v, f"{scope} complete windows")
        if isinstance(m, SpatialAverage):
            weights = getattr(m, "spatial_mean", None)
            if (
                (local_window or gathered_window)
                and weights is not None
                and np.asarray(weights).dtype.kind not in "fiu"
            ):
                raise ValueError(
                    f"C++ {scope} SpatialAverage requires real spatial_mean weights"
                )
            if weights is not None:
                with np.errstate(over="ignore"):
                    weights = np.asarray(weights, dtype=np.float32, order="C")
            if (
                weights is None
                or weights.ndim != 2
                or not weights.shape[0]
                or weights.shape[1] != n_space
                or not np.isfinite(weights).all()
            ):
                raise ValueError(
                    f"C++ {scope} SpatialAverage complete windows require finite "
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
            "voi": monitor_voi_list(m),
        }
    if isinstance(m, AfferentCoupling):
        return {
            "slot": int(slot),
            "kind": 5,  # KMON_CTAVG
            "istep": 1,
            "cadence_K": ck,
            "dt": eff_dt,
            "midpoint_time": False,
            "voi": monitor_voi_list(m),
        }
    if isinstance(m, TemporalAverage):
        istep = max(1, int(round(float(m.period) / eff_dt)))
        return {
            "slot": int(slot),
            "kind": 4,  # KMON_TAVG
            "istep": istep,
            "cadence_K": ck,
            "dt": eff_dt,
            "voi": monitor_voi_list(m),
        }
    if isinstance(m, Bold):
        voi = monitor_voi_list(m)
        tmp = copy.deepcopy(m)
        tmp._config_dt(eff_dt)
        tmp.compute_hrf()
        first_order = isinstance(m.hrf_kernel, equations.FirstOrderVolterra)
        k1v0 = (
            float(
                m.hrf_kernel.parameters["k_1"] * m.hrf_kernel.parameters["V_0"]
            )
            if first_order
            else 1.0
        )
        hrf = np.asarray(
            tmp.hemodynamic_response_function, dtype=np.float64
        ).ravel()
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
            "voi": monitor_voi_list(m),
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
            "voi": monitor_voi_list(m),
        }
    if isinstance(m, SpatialAverage):
        istep = (
            max(1, int(round(float(m.period) / eff_dt)))
            if getattr(m, "period", None)
            else 1
        )
        if explicit_chunk_size is not None and ck == 1:
            cs_i = max(1, int(explicit_chunk_size))
            if cs_i >= istep or (
                nstep > 0
                and nstep < istep
                and not is_resuming
                and not return_snapshot
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
            "voi": monitor_voi_list(m),
            "spatial_mean": sm_arr,
        }
    if isinstance(m, Projection):
        istep = (
            max(1, int(round(float(m.period) / eff_dt)))
            if getattr(m, "period", None)
            else 1
        )
        if explicit_chunk_size is not None and ck == 1:
            cs_i = max(1, int(explicit_chunk_size))
            if cs_i >= istep or (
                nstep > 0
                and nstep < istep
                and not is_resuming
                and not return_snapshot
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
                    rv, dtype=np.float32
                )
            else:
                ref_vec = np.ascontiguousarray(rv, dtype=np.float32).ravel()
        return {
            "slot": int(slot),
            "kind": 8,  # KMON_PROJECTION
            "istep": istep,
            "cadence_K": ck,
            "dt": eff_dt,
            "voi": monitor_voi_list(m),
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
            "voi": monitor_voi_list(m),
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


def projection_attachment_identity(spec, n_voi: int):
    """Fingerprint a Projection attachment spec for snapshot resume checks."""

    def array_identity(values):
        if values is None or not values.size:
            return None
        array = np.asarray(values, dtype="<f4", order="C")
        return array.shape, hashlib.sha256(array).hexdigest()

    return (
        spec["istep"],
        spec["cadence_K"],
        spec["dt"],
        tuple(spec["voi"] or range(n_voi)),
        array_identity(spec["gain"]),
        array_identity(spec["ref_vec"]),
    )


def assemble_subnet_results(
    analysis,
    sn_mons_map,
    sn_py_fallbacks,
    subnet_streams,
    outputs,
    nstep: int,
):
    """Assemble per-subnet Recorder results from C++ streams and Python fallbacks."""
    from tvb.simulator.monitors import AfferentCoupling

    if sn_mons_map is None:
        return None
    subnet_results = {}
    for si, sn in enumerate(analysis.subnets):
        total_r = len(sn_mons_map.get(sn.name, ()))
        res_list = [None] * total_r
        for ri, stream in subnet_streams.get(sn.name, ()):
            res_list[ri] = stream
        if sn_py_fallbacks[sn.name]:
            _, tavg_s, ctavg_s = outputs[si]
            k_j = sn.k
            for ri, rm in sn_py_fallbacks[sn.name]:
                mon_inst = copy.deepcopy(rm)
                if hasattr(mon_inst, "_config_dt"):
                    mon_inst._config_dt(float(sn.dt))
                if hasattr(mon_inst, "compute_hrf"):
                    mon_inst.compute_hrf()
                src_arr = (
                    ctavg_s if isinstance(mon_inst, AfferentCoupling) else tavg_s
                )
                if hasattr(mon_inst, "_config_stock"):
                    if (
                        getattr(mon_inst, "variables_of_interest", None) is None
                        or mon_inst.variables_of_interest.size == 0
                    ):
                        mon_inst.voi = np.arange(src_arr.shape[1], dtype=int)
                    else:
                        mon_inst.voi = np.asarray(
                            mon_inst.variables_of_interest, dtype=int
                        )
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
    return subnet_results


def assemble_global_monitor_outputs(
    analysis,
    monitors,
    plan: MonitorPlan,
    kernel_streams: dict,
    outputs,
    dt: float,
    bold_states,
    *,
    apply_monitors_fn=None,
    aggregate_raw_outputs_fn=None,
):
    """Assemble global monitor output streams from C++ kernels and Python fallbacks."""
    from tvb.simulator.monitors import (
        AfferentCoupling,
        Bold,
        Raw,
        SubSample,
    )

    if apply_monitors_fn is None:
        apply_monitors_fn = _apply_monitors
    if aggregate_raw_outputs_fn is None:
        aggregate_raw_outputs_fn = _aggregate_raw_outputs

    if monitors is None:
        return [] if plan.sn_mons_map is not None else outputs
    if len(monitors) == 0:
        return []
    if not plan.py_global_indices:
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
                (
                    np.empty(0, dtype=np.float64),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                )
                for _ in analysis.subnets
            ]
            return apply_monitors_fn(
                dummy_raw,
                monitors,
                dt,
                chunk_size=plan.chunk_size,
                monitor_data=None,
                subnet_infos=analysis.subnetworks,
                bold_states=bold_states,
                kernel_streams=kernel_streams or None,
            )
        return [kernel_streams[mi] for mi in range(len(monitors))]

    bold_raw_outputs = outputs if plan.has_py_bold else None
    temporal_raw_outputs = outputs if plan.has_py_tavg else None
    if plan.has_py_bold or plan.has_py_tavg or plan.need_py_sn_tavg:
        outputs = aggregate_raw_outputs_fn(outputs, plan.chunk_size)
    # Substitute Raw() placeholder for kernel-handled monitors not in
    # nb_hybrid's kernel_streams shortcut, then overwrite from kernel_streams.
    proxy_mons = [
        (
            Raw()
            if (
                mi in kernel_streams
                and not (
                    isinstance(m, (Raw, SubSample, Bold))
                    and not isinstance(m, AfferentCoupling)
                )
            )
            else m
        )
        for mi, m in enumerate(monitors)
    ]
    host_shortcuts = kernel_streams
    if plan.gathered_window_sources:
        # Already-gathered streams cannot be indexed per source by Raw.
        global_slots = {slot for slot, _ in plan.kernel_global_specs}
        host_shortcuts = {
            mi: stream
            for mi, stream in kernel_streams.items()
            if mi not in global_slots
        }
    outputs = apply_monitors_fn(
        outputs,
        proxy_mons,
        dt,
        chunk_size=plan.chunk_size,
        monitor_data=None,
        subnet_infos=analysis.subnetworks,
        bold_states=bold_states,
        bold_raw_outputs=bold_raw_outputs,
        temporal_raw_outputs=temporal_raw_outputs,
        kernel_streams=host_shortcuts or None,
    )
    for mi, stream in kernel_streams.items():
        outputs[mi] = stream
    return outputs
