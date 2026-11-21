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

"""Backend-neutral hybrid clock, coupling, monitor and parameter helpers.

Keep compiler imports out of this module: both compiled backends and the
public hybrid Simulator use these Python/NumPy operations.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from tvb.simulator.hybrid.network import NetworkSet


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


def _aggregate_raw_outputs(raw_outputs, chunk_size):
    """Restore the requested chunk view after a per-step BOLD run."""
    if chunk_size == 1:
        return raw_outputs
    aggregated = []
    for times, data, ctavg in raw_outputs:
        slices = [slice(start, min(start + chunk_size, len(times)))
                  for start in range(0, len(times), chunk_size)]
        aggregated.append((
            np.asarray([(times[part.start] + times[part.stop - 1]) * 0.5
                        for part in slices]),
            np.stack([data[part].mean(axis=0) for part in slices]),
            np.stack([ctavg[part].mean(axis=0) for part in slices]),
        ))
    return aggregated


def _copy_monitor_states(states):
    """Copy configured monitor runtimes without losing transient TVB fields."""
    if states is None:
        return None
    copied = {}
    for key, runtime in states.items():
        monitor = copy.copy(runtime["monitor"])
        for name, value in vars(runtime["monitor"]).items():
            setattr(monitor, name, value.copy() if isinstance(value, np.ndarray) else value)
        copied[key] = {**runtime, "monitor": monitor}
    return copied


def _apply_monitors(
    raw_outputs: list,
    monitors: list,
    dt: float,
    chunk_size: int = 1,
    monitor_data: Optional[dict] = None,
    subnet_infos: Optional[list] = None,
    bold_states: Optional[dict] = None,
    bold_raw_outputs: Optional[list] = None,
    temporal_raw_outputs: Optional[list] = None,
    kernel_streams: Optional[dict] = None,
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
    kernel_streams : dict, optional
        ``{monitor_index: [(times, data), ... per subnet]}`` produced by the
        C++ kernel monitor engines (Bold HRF convolution, Raw/SubSample
        collection).  When present for a monitor, the kernel stream is used
        directly; otherwise this pure-Python path remains the reference
        implementation every backend shares.

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

    results: list = []
    for monitor_index, m in enumerate(monitors):
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
                # Kernel engine stream (C++ side convolution) when available
                if kernel_streams and monitor_index in kernel_streams:
                    per_subnet.append(kernel_streams[monitor_index][si])
                    continue
                # Keep the mutable HRF stocks outside the topology-specific
                # compiled kernel and drive TVB's monitor with every state.
                if bold_raw_outputs is not None:
                    _, data, _ = bold_raw_outputs[si]
                state_key = ("bold", monitor_index, si)
                runtime = bold_states.get(state_key) if bold_states is not None else None
                if runtime is None:
                    bold = copy.deepcopy(m)
                    bold._config_dt(dt)
                    if bold.variables_of_interest is None or bold.variables_of_interest.size == 0:
                        bold.voi = np.arange(data.shape[1], dtype=int)
                    else:
                        bold.voi = np.asarray(bold.variables_of_interest, dtype=int)
                    bold.compute_hrf()
                    bold._config_stock(
                        num_vars=len(bold.voi),
                        num_nodes=data.shape[2],
                        num_modes=data.shape[3],
                    )
                    runtime = {"monitor": bold, "step": 0}
                    if bold_states is not None:
                        bold_states[state_key] = runtime
                else:
                    bold = runtime["monitor"]

                samples = [
                    sample
                    for step, state in enumerate(data, start=runtime["step"] + 1)
                    if (sample := bold.sample(step, state)) is not None
                ]
                runtime["step"] += len(data)
                if samples:
                    per_subnet.append((
                        np.asarray([sample[0] for sample in samples]),
                        np.stack([sample[1] for sample in samples]),
                    ))
                else:
                    per_subnet.append((
                        np.array([], dtype=np.float64),
                        np.empty(
                            (0, len(bold.voi), data.shape[2], data.shape[3]),
                            dtype=np.float64,
                        ),
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
                if kernel_streams and monitor_index in kernel_streams:
                    per_subnet.append(kernel_streams[monitor_index][si])
                    continue
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
                if temporal_raw_outputs is None:
                    per_subnet.append((times, data))
                    continue
                _, sample_data, _ = temporal_raw_outputs[si]
                state_key = ("temporal_average", monitor_index, si)
                runtime = bold_states.get(state_key) if bold_states is not None else None
                if runtime is None:
                    temporal_average = copy.deepcopy(m)
                    temporal_average._config_dt(dt)
                    if (temporal_average.variables_of_interest is None or
                            temporal_average.variables_of_interest.size == 0):
                        temporal_average.voi = np.arange(sample_data.shape[1], dtype=int)
                    else:
                        temporal_average.voi = np.asarray(
                            temporal_average.variables_of_interest, dtype=int
                        )
                    temporal_average._config_stock(
                        len(temporal_average.voi), sample_data.shape[2],
                        sample_data.shape[3]
                    )
                    runtime = {"monitor": temporal_average, "step": 0}
                    if bold_states is not None:
                        bold_states[state_key] = runtime
                else:
                    temporal_average = runtime["monitor"]
                samples = [
                    sample
                    for step, state in enumerate(
                        sample_data, start=runtime["step"] + 1
                    )
                    if (sample := temporal_average.sample(step, state)) is not None
                ]
                runtime["step"] += len(sample_data)
                if samples:
                    per_subnet.append((
                        np.asarray([sample[0] for sample in samples]),
                        np.stack([sample[1] for sample in samples]),
                    ))
                else:
                    per_subnet.append((
                        np.array([], dtype=np.float64),
                        np.empty((0,) + sample_data.shape[1:], dtype=np.float64),
                    ))
            elif isinstance(m, Raw):
                if kernel_streams and monitor_index in kernel_streams:
                    per_subnet.append(kernel_streams[monitor_index][si])
                    continue
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


def _validate_multi_dt(network_set: NetworkSet) -> float:
    """Validate per-subnetwork dts; return the master-clock base dt (dt0).

    Every subnetwork's dt must be an integer multiple of the smallest dt
    (dt0 = min over subnets); ``k_j = round(dt_j / dt0) >= 1`` is the number
    of master ticks per subnet step.  Raises ValueError otherwise.  The
    single-dt case (all equal) trivially satisfies the rule with all k = 1.
    An explicit projection dt must match its source clock (the owning
    subnetwork for intra-projections), since integer delays use source steps.
    """
    import numpy as _np

    dts = [float(sn.scheme.dt) for sn in network_set.subnets]
    dt0 = min(dts)
    for sn, dt_j in zip(network_set.subnets, dts):
        k = int(round(dt_j / dt0))
        if k < 1 or not _np.isclose(dt_j, k * dt0, rtol=1e-9, atol=1e-12):
            raise ValueError(
                "All subnetworks must use dts that are integer multiples of "
                f"the smallest dt {dt0}. Got dt={dt_j} in '{sn.name}' "
                f"({dt_j / dt0:.6g} master ticks, not an integer >= 1)."
            )
    projections_with_source = [(p, p.source) for p in network_set.projections]
    projections_with_source.extend(
        (p, sn) for sn in network_set.subnets for p in sn.projections)
    for projection, source in projections_with_source:
        if projection.dt is not None and not _np.isclose(
                float(projection.dt), float(source.scheme.dt),
                rtol=1e-9, atol=1e-12):
            raise ValueError(
                f"Projection dt={projection.dt} must match source subnetwork "
                f"'{source.name}' dt={source.scheme.dt}; integer delays are "
                "measured in source steps."
            )
    return dt0


def _stim_master_grid(stim, dt0: float, last_step: int):
    """Return a Stim whose temporal pattern is configured on the master grid.

    Multi-dt decision 7 (parity_audit.md §6): stimuli are evaluated on the
    master dt0 grid, one evaluation per master step, indexed by the global
    step.  ``Stim.configure`` builds the pattern's time axis on the target
    subnet's own dt (``target.scheme.dt``), which for a slow subnet
    (dt_j = k_j * dt0, k_j > 1) is the wrong resolution for master-step
    indexing: step indices would sample the wrong times and outrun the
    axis (IndexError).  This returns a deep copy whose pattern is
    configured on a dt0-resolution axis covering step indices
    0..last_step.  The original object is returned unchanged when its
    axis is already dt0-resolution and long enough — the single-dt case,
    so degenerate (all k_j = 1) runs keep bit-identical stimulus values.
    """
    import copy

    n_needed = int(last_step) + 1
    time = getattr(stim, "time", None)
    if (getattr(stim, "dt", None) is not None
            and float(stim.dt) == float(dt0)
            and time is not None
            and len(np.asarray(time)) >= n_needed):
        return stim
    # neotraits arrays do not survive a full Stim deepcopy (NArray traits
    # come back object-dtype), so copy shallowly and give the copy its own
    # (deep-copied) pattern; the pattern's lazily-configured caches are
    # rebuilt by the same configure sequence Stim.configure performs.
    stim = copy.copy(stim)
    stim.stimulus = copy.deepcopy(stim.stimulus)
    stim.dt = float(dt0)
    stim.time = np.arange(0.0, n_needed * dt0, dt0)
    if hasattr(stim.stimulus, "configure_time"):
        stim.stimulus.configure_time(
            np.asarray(stim.time).reshape((1, -1)))
    if hasattr(stim.stimulus, "configure_space"):
        stim.stimulus.configure_space()
    return stim


def _can_merge_subnets(subnet_infos: list) -> bool:
    """Check if all subnets have node_indices and same voi count."""
    if not subnet_infos:
        return False
    # All must have node_indices
    if not all(si.node_indices is not None for si in subnet_infos):
        return False
    # All must have same voi count
    voi_counts = [len(si.model.variables_of_interest) for si in subnet_infos]
    if len(set(voi_counts)) != 1:
        return False
    # All must share the same dt (multi-dt support, parity_audit.md §6
    # decision 9): merging per-subnet outputs onto one node axis would
    # silently erase the per-subnet dt distinction, so merged views require
    # provably equal dt (equal k_j).  A subnet info without a dt cannot be
    # proven equal, so it refuses the merge as well (no silent 0.0 default).
    dts = [getattr(si, "dt", None) for si in subnet_infos]
    if any(dt_val is None for dt_val in dts):
        return False
    return len({float(dt_val) for dt_val in dts}) == 1


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
    """Return the supported coupling type, rejecting unimplemented behavior."""
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

    cfun = p.cfun
    if cfun is None:
        return "none"
    cfun_class = type(cfun)
    if cfun_class is Linear:
        return "linear"
    if cfun_class is Scaling:
        return "scaling"
    if cfun_class is Sigmoidal:
        return "sigmoidal"
    if cfun_class is SigmoidalJansenRit:
        if (getattr(cfun, 'use_classic', 1)
                and p.source_cvar.shape[0] == 2):
            return "sigmoidal_jr"
        return "sigmoidal_jr_legacy"
    if cfun_class is KuramotoCfun:
        return "kuramoto"
    if cfun_class is Difference:
        return "difference"
    if cfun_class is HyperbolicTangent:
        return "tanh"
    if cfun_class is PreSigmoidal:
        if (getattr(cfun, 'dynamic', 0)
                and p.source_cvar.shape[0] == 2):
            return "pre_sigmoidal_dynamic"
        return "pre_sigmoidal"
    raise NotImplementedError(
        f"NbHybridBackend does not support coupling {cfun_class.__name__}. "
        "Use one of the explicitly supported hybrid coupling classes: Linear, "
        "Scaling, Sigmoidal, SigmoidalJansenRit, Kuramoto, Difference, "
        "HyperbolicTangent, PreSigmoidal, or None."
    )


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

