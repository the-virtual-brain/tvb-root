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

import hashlib
import importlib.util
import os
import sys
import tempfile
import time
from pathlib import Path
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
    cache_dir = Path(tempfile.gettempdir()) / "tvb_nb_hybrid_cache"
    cache_dir.mkdir(exist_ok=True)
    mod_name = f"nbhybrid_sweep_{cache_key[:16]}"
    mod_path = cache_dir / f"{mod_name}.py"

    if not mod_path.exists():
        tmp_path = mod_path.with_suffix(".tmp")
        tmp_path.write_text(full_source, encoding="utf-8")
        os.replace(tmp_path, mod_path)

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
    from tvb.simulator.backend.nb_hybrid import SweepResult, _cfun_params

    subnets = analysis.subnetworks
    all_projs = analysis.all_projections
    n_sweeps = sweep_values.shape[0]

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
        ncvar = len(sn.model.coupling_terms)
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
        tavg_all[sn.name] = np.zeros((n_sweeps, nvoi, n_nodes, n_modes), dtype=np.float32)
        tavg_count_all[sn.name] = np.zeros(n_sweeps, dtype=np.int32)
        ctavg_all[sn.name] = np.zeros((n_sweeps, ncvar, n_nodes, n_modes), dtype=np.float32)
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

    # ---- Build per-projection cfun_params with swept entries ----
    cfun_params_all = {}
    proj_arrays = {}

    for p in all_projs:
        base = _cfun_params(p)
        cfun_params_all[p.name] = np.broadcast_to(
            base[np.newaxis], (n_sweeps, 8)
        ).copy().astype(np.float32)

        # Override swept entries
        for dim, desc in enumerate(sweep_descriptor):
            if desc['type'] == 'cfun':
                pname = desc['projection']
                pidx = desc.get('param_idx', 0)
                if p.name == pname:
                    cfun_params_all[p.name][:, pidx] = sweep_values[:, dim]

        # Shared projection arrays
        proj_arrays[f'{p.name}_w_data'] = p.weights_data.astype(np.float32)
        proj_arrays[f'{p.name}_w_indices'] = p.weights_indices.astype(np.int32)
        proj_arrays[f'{p.name}_w_indptr'] = p.weights_indptr.astype(np.int32)
        proj_arrays[f'{p.name}_idelays'] = p.idelays.astype(np.int32)
        if p.is_inter:
            proj_arrays[f'{p.name}_mode_map'] = p.mode_map.astype(np.float32)
        proj_arrays[f'{p.name}_source_cvar'] = p.source_cvar.astype(np.int32)
        proj_arrays[f'{p.name}_target_cvar'] = p.target_cvar.astype(np.int32)
        proj_arrays[f'{p.name}_scale'] = np.float32(p.scale)
        if p.target_scales.size > 0:
            proj_arrays[f'{p.name}_target_scales'] = p.target_scales.astype(np.float32)
        else:
            proj_arrays[f'{p.name}_target_scales'] = np.zeros(0, dtype=np.float32)

    # ---- Build noise and stimulus arrays ----
    noise_arrays = {}
    stim_arrays = {}
    for sn in subnets:
        sn_obj = next(s for s in network_set.subnets if s.name == sn.name)
        if sn.is_stochastic:
            dt = sn_obj.scheme.dt
            rng = sn_obj.scheme.noise.random_stream
            dw = rng.randn(nstep, sn.model.nvar, sn.n_nodes, sn.n_modes)
            noise_std = np.sqrt(2.0 * sn.noise_nsig * dt)
            dw *= noise_std[np.newaxis, :, np.newaxis, np.newaxis]
            noise_arrays[sn.name] = np.ascontiguousarray(
                np.transpose(dw, (1, 2, 3, 0))).astype(np.float32)
        if sn.has_stimulus:
            n_cvar = len(sn.model.coupling_terms)
            stim_arr = np.zeros(
                (n_cvar, sn.n_nodes, sn.n_modes, nstep), dtype=np.float32)
            for stim in analysis.stimuli_by_subnet.get(sn.name, []):
                for step_idx in range(1, nstep + 1):
                    sc = np.asarray(stim.get_coupling(step_idx), dtype=np.float32)
                    if sc.ndim == 2:
                        sc = sc[:, :, np.newaxis]
                    if sc.shape[2] == 1 and sn.n_modes > 1:
                        sc = np.broadcast_to(
                            sc, (sc.shape[0], sn.n_nodes, sn.n_modes)).copy()
                    stim_arr[:, :, :, step_idx - 1] += sc
            stim_arrays[sn.name] = stim_arr

    # ---- Build spatial params ----
    sp_arrays = {}
    for sn in subnets:
        sp_names = list(getattr(sn.model, 'spatial_parameter_names', []))
        if sp_names:
            sp_arrays[sn.name] = np.array(
                [np.broadcast_to(getattr(sn.model, n), (sn.n_nodes,)).ravel()
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
        args.append(proj_arrays[f'{p.name}_scale'])
        args.append(proj_arrays[f'{p.name}_target_scales'])
        args.append(cfun_params_all[p.name])  # (n_sweeps, 8) per-sweep

    # Per-sweep accumulators
    for sn in subnets:
        args.append(tavg_all[sn.name])
        args.append(tavg_count_all[sn.name])
        args.append(ctavg_all[sn.name])
        args.append(c_all[sn.name])

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
        count = tavg_count_all[sn.name].astype(np.float32)
        count = np.where(count > 0, count, np.float32(1.0))
        count_4d = count[:, np.newaxis, np.newaxis, np.newaxis]
        result_tavg[sn.name] = tavg_all[sn.name] / count_4d
        result_ctavg[sn.name] = ctavg_all[sn.name] / count_4d

        # Mode-0 selection: sum modes into mode-0 (matching CPU backend)
        if result_tavg[sn.name].shape[-1] > 1:
            result_tavg[sn.name] = result_tavg[sn.name].sum(axis=-1, keepdims=True)
            result_ctavg[sn.name] = result_ctavg[sn.name].sum(axis=-1, keepdims=True)

    # Build SweepResult
    subnet_names = [sn.name for sn in network_set.subnets]
    n_vois = set(v.shape[1] for v in result_tavg.values())
    if len(n_vois) == 1 and len(subnet_names) > 1:
        n_global = sum(result_tavg[sn.name].shape[2] for sn in subnets)
        ref = list(result_tavg.values())[0]
        merged = np.zeros(
            (ref.shape[0], ref.shape[1], n_global, ref.shape[3]),
            dtype=np.float32)
        offset = 0
        for sn_info in subnets:
            n = result_tavg[sn_info.name].shape[2]
            merged[:, :, offset:offset + n, :] = result_tavg[sn_info.name]
            offset += n
        merged_tavg = merged
    elif len(subnet_names) == 1:
        merged_tavg = list(result_tavg.values())[0]
    else:
        merged_tavg = None

    times = np.array([0.0], dtype=np.float64)

    return SweepResult(
        tavg=result_tavg,
        merged_tavg=merged_tavg,
        ctavg=result_ctavg,
        times=times,
        sweep_values=sweep_values,
        backend="cpu-prange",
        elapsed=elapsed,
    )