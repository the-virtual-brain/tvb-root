# -*- coding: utf-8 -*-
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package.
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
"""
CPU parameter sweep using Numba prange for thread-level parallelism.

Generates a ``@nb.njit(parallel=True)`` sweep kernel that wraps the
existing ``network_chunk`` function in a ``nb.prange`` loop.  Each
thread operates on its own slice of per-sweep arrays, giving true
multi-core parallelism without fork-safety issues.

Architecture
------------
The sweep kernel template (``nb-hybrid-sweep-cpu.py.mako``) is rendered
alongside the single-sim template and appended to the same module, so
``sweep_kernel`` can call ``network_chunk`` directly.  Per-sweep arrays
have a leading ``n_sweeps`` dimension; inside the prange loop, each
thread operates on its own slice.  cfun_params are pre-built with swept
values overridden per sweep point, stored as ``cfun_params_all[p.name]``
with shape ``(n_sweeps, 8)``.
"""

import copy
import hashlib
import importlib.util
import os
import sys
import time
from typing import Optional

import numpy as np

# Module-level cache for compiled sweep kernels, keyed by topology hash.
_SWEEP_KERNEL_CACHE = {}


def compile_sweep_kernel(backend, analysis):
    """Compile and cache the prange sweep kernel.

    Renders the Mako template, appends it to the single-sim module source,
    and compiles both together so ``sweep_kernel`` can call ``network_chunk``.

    Parameters
    ----------
    backend : NbHybridBackend
        Backend instance (used for render_template).
    analysis : NetworkAnalysis
        Network analysis (determines topology).

    Returns
    -------
    callable
        The compiled ``sweep_kernel`` function.
    """
    content = dict(analysis=analysis, np=np, debug_nojit=False)

    # Render the sweep kernel template
    sweep_source = backend.render_template(
        '<%include file="nb-hybrid-sweep-cpu.py.mako"/>', content)

    # Also render the single-sim template (contains network_chunk)
    sim_source = backend.render_template(
        '<%include file="nb-hybrid-sim.py.mako"/>', content)

    # Combine both sources into one module
    full_source = sim_source + "\n\n# ---- SWEEP KERNEL ----\n\n" + sweep_source

    # Cache key based on combined source
    cache_key = hashlib.sha256(full_source.encode()).hexdigest()

    if cache_key in _SWEEP_KERNEL_CACHE:
        return _SWEEP_KERNEL_CACHE[cache_key]

    # Write to disk for Numba caching
    cache_dir = backend.get_cache_dir()
    mod_name = f"nbhybrid_sweep_{cache_key}"
    mod_path = cache_dir / f"{mod_name}.py"

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not mod_path.exists():
            tmp_path = mod_path.with_suffix(".tmp")
            tmp_path.write_text(full_source, encoding="utf-8")
            os.replace(tmp_path, mod_path)
    except OSError as exc:
        raise OSError(
            "Unable to create or write the Numba hybrid cache directory "
            f"at '{cache_dir}' configured by TVB_NHYBRID_CACHE_DIR; "
            "ensure the cache directory is writable."
        ) from exc

    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)

    kernel_fn = mod.sweep_kernel
    _SWEEP_KERNEL_CACHE[cache_key] = kernel_fn
    return kernel_fn


def run_sweep_prange(kernel_fn, analysis, network_set, sweep_descriptor,
                     sweep_values, nstep, backend, initial_states=None):
    """Run a parameter sweep using the compiled prange kernel.

    Pre-allocates per-sweep and shared arrays, builds cfun_params with swept
    entries overridden per sweep point, and calls the kernel.

    Returns
    -------
    SweepResult
    """
    from tvb.simulator.backend.nb_hybrid import (
        NbHybridBackend,
        SweepResult,
        _cfun_params,
    )

    subnets = analysis.subnetworks
    all_projs = analysis.all_projections
    n_sweeps = sweep_values.shape[0]
    n_sweep_dims = sweep_values.shape[1] if sweep_values.ndim > 1 else 0

    # ---- Guard: model-parameter sweeps not yet supported on CPU prange path ----
    # _sweep_params is accepted by the kernel but never forwarded to network_chunk
    # or the dfun templates. cfun-parameter sweeps work correctly (applied below).
    # TODO: thread _sweep_params through network_chunk → state_update → dfun, then
    #       remove this guard.
    for dim, desc in enumerate(sweep_descriptor):
        if desc.get('type', '') not in ('cfun',):
            raise NotImplementedError(
                f"CPU sweep dimension {dim} ({desc.get('name', '?')}) has type "
                f"'{desc.get('type', '?')}' — model-parameter sweeps are not yet "
                f"supported on the CPU prange path. Use the Python sequential path "
                f"(backend='numba', parallel=False) or the CUDA sweep backend instead."
            )

    t_start = 1  # 1-based time step (matching single-sim)

    # ---- Build per-sweep mutable arrays ----
    state_all = {}
    srcbuf_all = {}
    tavg_all = {}
    tavg_count_all = {}
    ctavg_all = {}
    c_all = {}
    spatial_tavg_all = {}
    proj_tavg_all = {}
    bold_state_all = {}
    bold_params = {}
    bold_voi_idx = {}
    spatial_mean = {}
    gain = {}

    for sn in subnets:
        sn_obj = next(s for s in network_set.subnets if s.name == sn.name)
        nvar = sn.model.nvar
        n_nodes = sn.n_nodes
        n_modes = sn.n_modes
        nvoi = len(sn.model.variables_of_interest)
        ncvar = len(sn.model.cvar)
        horizon = analysis.source_horizons.get(sn.name, 1)

        # Initial state
        if initial_states is not None:
            init = initial_states[subnets.index(sn)].astype(np.float32)
        else:
            init = sn_obj.zero_states().astype(np.float32)

        # Per-sweep copies of state and srcbuf
        state_all[sn.name] = np.ascontiguousarray(
            np.broadcast_to(init[np.newaxis], (n_sweeps,) + init.shape).copy(),
            dtype=np.float32)

        # Source buffer
        srcbuf_all[sn.name] = np.ascontiguousarray(
            np.broadcast_to(
                init[:, :, :, np.newaxis] * np.ones((1, 1, 1, horizon), dtype=np.float32),
                (n_sweeps, nvar, n_nodes, n_modes, horizon)
            ).copy(), dtype=np.float32)

        # Output/scratch accumulators
        tavg_all[sn.name] = np.zeros(
            (n_sweeps, nstep, nvoi, n_nodes, n_modes), dtype=np.float32)
        ctavg_all[sn.name] = np.zeros(
            (n_sweeps, nstep, ncvar, n_nodes, n_modes), dtype=np.float32)
        c_all[sn.name] = np.zeros((n_sweeps, ncvar, n_nodes, n_modes), dtype=np.float32)

        # Monitor accumulators (zeros — no monitors in sweep for now)
        n_areas = 0
        n_sensors = 0
        spatial_mean[sn.name] = np.zeros((0, n_nodes), dtype=np.float32)
        gain[sn.name] = np.zeros((0, n_nodes), dtype=np.float32)
        spatial_tavg_all[sn.name] = np.zeros(
            (n_sweeps, nvoi, max(n_areas, 1), n_modes), dtype=np.float32)
        proj_tavg_all[sn.name] = np.zeros(
            (n_sweeps, nvoi, max(n_sensors, 1), 1), dtype=np.float32)

        # Bold state
        svars = list(sn.model.state_variables)
        voi = list(sn.model.variables_of_interest)
        voi_idx = [svars.index(v) if v in svars else 0 for v in voi]
        bold_state_all[sn.name] = np.zeros((n_sweeps, nvoi, 4, n_nodes), dtype=np.float32)
        # Initialize: f=v=q=1 (axis 2 = [s,f,v,q])
        bold_state_all[sn.name][:, :, 1, :] = 1.0
        bold_state_all[sn.name][:, :, 2, :] = 1.0
        bold_state_all[sn.name][:, :, 3, :] = 1.0
        bold_params[sn.name] = np.array([
            1/0.65, 1/0.41, 1/0.98, 1/0.32, 0.4, 1/0.4,
            4.3*40.3*0.04*0.5, 0.5*25*0.04*0.5, 1-0.5
        ], dtype=np.float32)
        bold_voi_idx[sn.name] = np.array(voi_idx, dtype=np.int32)

    # Single shared tavg counter for all subnets
    tavg_count_all = np.zeros((n_sweeps, nstep), dtype=np.int32)

    # ---- Build per-projection cfun_params with swept entries ----
    cfun_params_all = {}
    proj_arrays = {}

    for p in all_projs:
        base = _cfun_params(p)
        cfun_params_all[p.name] = np.broadcast_to(
            base[np.newaxis], (n_sweeps, len(base))
        ).copy().astype(np.float32)

        projection_dims = [
            (dim, desc)
            for dim, desc in enumerate(sweep_descriptor)
            if desc['type'] == 'cfun' and desc['projection'] == p.name
        ]
        if projection_dims:
            for row in range(n_sweeps):
                row_projection = copy.copy(p)
                row_projection.cfun = copy.copy(p.cfun)
                for dim, desc in projection_dims:
                    NbHybridBackend._cfun_set_param(
                        row_projection.cfun,
                        desc.get('param_idx', 0),
                        sweep_values[row, dim],
                    )
                cfun_params_all[p.name][row] = _cfun_params(row_projection)

        # Shared projection arrays
        proj_arrays[f'{p.name}_w_data'] = p.weights_data.astype(np.float32)
        proj_arrays[f'{p.name}_w_indices'] = p.weights_indices.astype(np.int32)
        proj_arrays[f'{p.name}_w_indptr'] = p.weights_indptr.astype(np.int32)
        proj_arrays[f'{p.name}_idelays'] = p.idelays.astype(np.int32)
        if p.is_inter:
            proj_arrays[f'{p.name}_mode_map'] = p.mode_map.astype(np.float32)
        proj_arrays[f'{p.name}_source_cvar'] = p.source_cvar.astype(np.int32)
        proj_arrays[f'{p.name}_target_cvar'] = p.target_cvar.astype(np.int32)
        proj_arrays[f'{p.name}_target_state_cvar'] = p.target_state_cvar.astype(np.int32)
        proj_arrays[f'{p.name}_scale'] = np.float32(p.scale)
        if p.target_scales.size > 0:
            proj_arrays[f'{p.name}_target_scales'] = p.target_scales.astype(np.float32)
        else:
            proj_arrays[f'{p.name}_target_scales'] = np.zeros(0, dtype=np.float32)

    # ---- Build noise and stimulus arrays ----
    noise_arrays = {}
    stim_arrays = {}
    for sn in subnets:
        if sn.is_stochastic:
            noise_arrays[sn.name] = np.empty(
                (n_sweeps, sn.model.nvar, sn.n_nodes, sn.n_modes, nstep),
                dtype=np.float32,
            )

    # Draw outside prange in the same sweep-major order as run_sweep(). Each
    # row then owns an immutable realization and resetting the RNG replays the
    # complete batch independently of thread scheduling.
    for row in range(n_sweeps):
        for sn in subnets:
            if not sn.is_stochastic:
                continue
            sn_obj = next(s for s in network_set.subnets if s.name == sn.name)
            dt = sn_obj.scheme.dt
            rng = sn_obj.scheme.noise.random_stream
            dw = rng.randn(nstep, sn.model.nvar, sn.n_nodes, sn.n_modes)
            noise_std = np.sqrt(2.0 * sn.noise_nsig * dt)
            dw *= noise_std[np.newaxis, :, np.newaxis, np.newaxis]
            noise_arrays[sn.name][row] = np.transpose(dw, (1, 2, 3, 0))

    for sn in subnets:
        sn_obj = next(s for s in network_set.subnets if s.name == sn.name)
        if sn.has_stimulus:
            n_cvar = len(sn.model.cvar)
            stim_arr = np.zeros(
                (n_cvar, sn.n_nodes, sn.n_modes, nstep), dtype=np.float32)
            for stim in analysis.stimuli_by_subnet.get(sn.name, []):
                target_slots = np.asarray(stim.target_cvar)
                if target_slots.ndim != 1 or target_slots.size == 0:
                    raise ValueError(
                        f"Stimulus for subnetwork '{sn.name}' must have a non-empty "
                        "one-dimensional target_cvar array"
                    )
                if target_slots.dtype.kind not in "iu":
                    raise ValueError(
                        f"Stimulus for subnetwork '{sn.name}' has non-integer target_cvar"
                    )
                target_slots = target_slots.astype(np.intp, copy=False)
                if np.any(target_slots < 0) or np.any(target_slots >= n_cvar):
                    raise ValueError(
                        f"Stimulus for subnetwork '{sn.name}' has target coupling slots "
                        f"{target_slots.tolist()} outside [0, {n_cvar - 1}]"
                    )
                target_shape = (target_slots.size, sn.n_nodes, sn.n_modes)
                for step_idx in range(1, nstep + 1):
                    sc = np.asarray(stim.get_coupling(step_idx), dtype=np.float32)
                    if sc.ndim == 2:
                        sc = sc[:, :, np.newaxis]
                    if sc.ndim != 3 or any(
                        actual not in (1, expected)
                        for actual, expected in zip(sc.shape, target_shape)
                    ):
                        raise ValueError(
                            f"Stimulus for subnetwork '{sn.name}' returned shape "
                            f"{sc.shape}; expected a shape broadcastable to {target_shape}"
                        )
                    stim_arr[target_slots, :, :, step_idx - 1] += np.broadcast_to(
                        sc, target_shape
                    )
            stim_arrays[sn.name] = stim_arr

    # ---- Build spatial params ----
    sp_arrays = {}
    for sn in subnets:
        if hasattr(sn.model, '_nb_hybrid_runtime_parameter_names'):
            sp_names = list(sn.model._nb_hybrid_runtime_parameter_names)
        elif hasattr(sn.model, '_nb_hybrid_custom_template'):
            sp_names = []
        else:
            sp_names = list(getattr(sn.model, 'spatial_parameter_names', []))
        if sp_names:
            sp_arrays[sn.name] = np.array(
                [np.broadcast_to(np.asarray(getattr(sn.model, n)).ravel(),
                                 (sn.n_nodes,))
                 for n in sp_names],
                dtype=np.float32)
        else:
            sp_arrays[sn.name] = np.zeros((0, sn.n_nodes), dtype=np.float32)

    # ---- Assemble argument list matching sweep_kernel signature ----
    dt = network_set.subnets[0].scheme.dt
    bold_dt = np.float32(dt)

    args = [
        np.int32(n_sweeps),
        np.int32(nstep),
        np.int32(t_start),
        sweep_values,  # (n_sweeps, n_sweep_dims) float32
    ]

    # Per-sweep state/srcbuf
    for sn in subnets:
        args.append(state_all[sn.name])
        args.append(srcbuf_all[sn.name])

    # Shared projection arrays + per-sweep cfun_params
    for p in all_projs:
        args.append(proj_arrays[f'{p.name}_w_data'])
        args.append(proj_arrays[f'{p.name}_w_indices'])
        args.append(proj_arrays[f'{p.name}_w_indptr'])
        args.append(proj_arrays[f'{p.name}_idelays'])
        if p.is_inter:
            args.append(proj_arrays[f'{p.name}_mode_map'])
        args.append(proj_arrays[f'{p.name}_source_cvar'])
        args.append(proj_arrays[f'{p.name}_target_cvar'])
        args.append(proj_arrays[f'{p.name}_target_state_cvar'])
        args.append(proj_arrays[f'{p.name}_scale'])
        args.append(proj_arrays[f'{p.name}_target_scales'])
        args.append(cfun_params_all[p.name])  # (n_sweeps, len(cfun_params)) per-sweep

    # Per-sweep accumulators
    for sn in subnets:
        args.append(tavg_all[sn.name])
        args.append(ctavg_all[sn.name])
        args.append(c_all[sn.name])
    args.append(tavg_count_all)

    # Noise
    for sn in subnets:
        if sn.is_stochastic:
            args.append(noise_arrays[sn.name])

    # Stimulus
    for sn in subnets:
        if sn.has_stimulus:
            args.append(stim_arrays[sn.name])

    # Spatial params
    for sn in subnets:
        args.append(sp_arrays[sn.name])

    # Monitor arrays
    for sn in subnets:
        args.append(spatial_mean[sn.name])
        args.append(spatial_tavg_all[sn.name])
        args.append(gain[sn.name])
        args.append(proj_tavg_all[sn.name])

    # Bold arrays
    for sn in subnets:
        args.append(bold_state_all[sn.name])
        args.append(bold_params[sn.name])
        args.append(bold_voi_idx[sn.name])

    args.append(bold_dt)

    # ---- Call the kernel ----
    t0 = time.perf_counter()
    kernel_fn(*args)
    elapsed = time.perf_counter() - t0

    # ---- Post-process: divide by tavg_count ----
    result_tavg = {}
    result_ctavg = {}
    for sn in subnets:
        count = tavg_count_all.astype(np.float32)
        count = np.where(count > 0, count, np.float32(1.0))
        count_5d = count[:, :, np.newaxis, np.newaxis, np.newaxis]
        result_tavg[sn.name] = tavg_all[sn.name] / count_5d
        result_ctavg[sn.name] = ctavg_all[sn.name] / count_5d

    # Build SweepResult
    subnet_names = [sn.name for sn in network_set.subnets]
    n_vois = set(v.shape[2] for v in result_tavg.values())
    if len(n_vois) == 1 and len(subnet_names) > 1:
        n_global = sum(result_tavg[sn.name].shape[3] for sn in subnets)
        ref = list(result_tavg.values())[0]
        merged = np.zeros(
            (ref.shape[0], ref.shape[1], ref.shape[2], n_global, ref.shape[4]),
            dtype=np.float32)
        offset = 0
        for sn_info in subnets:
            n = result_tavg[sn_info.name].shape[3]
            merged[:, :, :, offset:offset + n, :] = result_tavg[sn_info.name]
            offset += n
        merged_tavg = merged
    elif len(subnet_names) == 1:
        merged_tavg = list(result_tavg.values())[0]
    else:
        merged_tavg = None

    dt = float(network_set.subnets[0].scheme.dt)
    times = np.arange(1, nstep + 1, dtype=np.float64) * dt

    return SweepResult(
        tavg=result_tavg,
        merged_tavg=merged_tavg,
        ctavg=result_ctavg,
        times=times,
        sweep_values=sweep_values,
        backend="cpu-prange",
        elapsed=elapsed,
    )
