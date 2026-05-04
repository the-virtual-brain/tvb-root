#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numba-CUDA parameter sweep backend for the Hybrid Numba framework.

Generates a @cuda.jit kernel from a Mako template (nb-hybrid-sweep-cuda.py.mako),
compiles it, and runs parameter sweeps where each GPU thread simulates one
parameter value independently.

Usage::

    from tvb.simulator.backend.nb_hybrid_cuda_sweep_backend import NbHybridCUDASweepBackend
    backend = NbHybridCUDASweepBackend()
    result = backend.run_sweep(network_set, sweep_param='a', sweep_values=values, nstep=1000)

.. moduleauthor:: TVB contributors
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from numba import cuda

from tvb.simulator.backend.nb_hybrid import (
    NbHybridBackend,
    NetworkAnalysis,
    SubnetworkInfo,
    ProjectionInfo,
    _cfun_type,
    _cfun_params,
    _cvar_mapping_mode,
)
from tvb.simulator.backend.templates import MakoUtilMix


# ---------------------------------------------------------------------------
# Module-level cache for compiled CUDA kernels (keyed by SHA-256 of source)
# ---------------------------------------------------------------------------
_CUDA_COMPILED_CACHE: dict = {}


class NbHybridCUDASweepBackend(MakoUtilMix):
    """Backend that generates and runs CUDA parameter-sweep kernels.

    Usage::

        from tvb.simulator.backend.nb_hybrid_cuda_sweep_backend import NbHybridCUDASweepBackend
        backend = NbHybridCUDASweepBackend()
        compiled = backend.compile_sweep(network_set)
        result = compiled.run(nstep=1000, sweep_values=values)
    """

    def _build_sweep_analysis(self, network_set) -> NetworkAnalysis:
        """Build NetworkAnalysis from a NetworkSet (reuses NbHybridBackend._analyse)."""
        backend = NbHybridBackend()
        return backend._analyse(network_set)

    def compile_sweep(
        self,
        network_set,
        sweep_descriptor=None,
        print_source: bool = False,
    ) -> "CompiledCUDASweepKernel":
        """Generate and compile the CUDA sweep kernel for *network_set*.

        Parameters
        ----------
        network_set : NetworkSet
            Fully configured network (``configure()`` must have been called).
        print_source : bool
            If True, print the generated source.

        Returns
        -------
        CompiledCUDASweepKernel
            Object whose :meth:`run` method executes the sweep on GPU.
        """
        # Check model/feature compatibility before code generation
        NbHybridBackend()._check_compatibility(network_set)

        analysis = self._build_sweep_analysis(network_set)

        # Process sweep_descriptor
        if sweep_descriptor is None:
            if analysis.all_projections:
                first_proj = analysis.all_projections[0]
                sweep_descriptor = [{'type': 'cfun', 'projection': first_proj.name, 'param_idx': 0}]
            else:
                sweep_descriptor = []

        n_sweep_dims = len(sweep_descriptor)

        sweep_cfun_dims = {}
        sweep_model_dims = {}
        for dim_idx, desc in enumerate(sweep_descriptor):
            if desc['type'] == 'cfun':
                pname = desc['projection']
                pidx = desc.get('param_idx', 0)
                if pname not in sweep_cfun_dims:
                    sweep_cfun_dims[pname] = {}
                sweep_cfun_dims[pname][pidx] = dim_idx
            elif desc['type'] == 'model':
                sname = desc['subnet']
                param = desc['param']
                if sname not in sweep_model_dims:
                    sweep_model_dims[sname] = {}
                sweep_model_dims[sname][param] = dim_idx
            else:
                raise ValueError(f"Unknown sweep descriptor type: {desc['type']}")

        # Render template
        # Import cfun helpers for the template
        from tvb.simulator.hybrid.coupling import (
            Linear, Scaling, Sigmoidal, SigmoidalJansenRit,
            Kuramoto as KuramotoCfun, Difference, HyperbolicTangent, PreSigmoidal,
        )

        source = self.render_template(
            '<%include file="nb-hybrid-sweep-cuda.py.mako"/>',
            dict(
                analysis=analysis,
                np=np,
                _cfun_type=_cfun_type,
                _cfun_params=_cfun_params,
                _cvar_mapping_mode=_cvar_mapping_mode,
                all_projs=analysis.all_projections,
                subnets=analysis.subnetworks,
                sweep_cfun_dims=sweep_cfun_dims,
                sweep_model_dims=sweep_model_dims,
                n_sweep_dims=n_sweep_dims,
                has_custom_template=lambda m: hasattr(m, '_nb_hybrid_custom_template'),
                is_zerlaut=lambda m: hasattr(m, '_nb_hybrid_custom_template') and 'zerlaut' in getattr(m, '_nb_hybrid_custom_template', ''),
            ),
        )

        cache_key = hashlib.sha256(source.encode()).hexdigest()
        if cache_key in _CUDA_COMPILED_CACHE:
            cached = _CUDA_COMPILED_CACHE[cache_key]
            if print_source:
                print(self.insert_line_numbers(source))
            return CompiledCUDASweepKernel(
                _analysis=analysis,
                _network_set=network_set,
                _compiled_fn=cached,
                _source=source,
                sweep_descriptor=sweep_descriptor,
                n_sweep_dims=n_sweep_dims,
                sweep_cfun_dims=sweep_cfun_dims,
                sweep_model_dims=sweep_model_dims,
            )

        if print_source:
            print(self.insert_line_numbers(source))

        # Write source to file so Numba can cache PTX
        cache_dir = Path(os.environ.get("TVB_CUDA_CACHE_DIR", "/tmp/tvb_cuda_sweep_cache"))
        cache_dir.mkdir(exist_ok=True)
        mod_name = f"cuda_sweep_{cache_key[:16]}"
        mod_path = cache_dir / f"{mod_name}.py"
        if not mod_path.exists():
            tmp_path = mod_path.with_suffix(".tmp")
            tmp_path.write_text(source, encoding="utf-8")
            os.replace(tmp_path, mod_path)

        # Import module to trigger CUDA compilation
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)

        fn = mod.run_sweep
        _CUDA_COMPILED_CACHE[cache_key] = fn

        return CompiledCUDASweepKernel(
            _analysis=analysis,
            _network_set=network_set,
            _compiled_fn=fn,
            _source=source,
            sweep_descriptor=sweep_descriptor,
            n_sweep_dims=n_sweep_dims,
            sweep_cfun_dims=sweep_cfun_dims,
            sweep_model_dims=sweep_model_dims,
        )

    def run_sweep(
        self,
        network_set,
        sweep_values: np.ndarray,
        nstep: int = 100,
        initial_states: Optional[list] = None,
        sweep_descriptor=None,
        print_source: bool = False,
        bold_period: Optional[float] = None,
        verbose: bool = False,
    ):
        """Convenience: compile + run in one call."""
        compiled = self.compile_sweep(network_set, sweep_descriptor=sweep_descriptor, print_source=print_source)
        return compiled.run(
            nstep=nstep,
            sweep_values=sweep_values,
            initial_states=initial_states,
            bold_period=bold_period,
            verbose=verbose,
        )


@dataclasses.dataclass
class CompiledCUDASweepKernel:
    """Holds a compiled CUDA sweep kernel and manages argument assembly / launch."""

    _analysis: NetworkAnalysis
    _network_set: object  # NetworkSet
    _compiled_fn: object  # The run_sweep @cuda.jit kernel function
    _source: str
    sweep_descriptor: Optional[list] = None
    n_sweep_dims: int = 1
    sweep_cfun_dims: dict = dataclasses.field(default_factory=dict)
    sweep_model_dims: dict = dataclasses.field(default_factory=dict)

    @staticmethod
    def _merge_subnet_outputs(tavg_list, subnetworks, node_indices=None):
        """Merge per-subnet tavg arrays along the node dimension.

        Each tavg array has shape (n_sweeps, n_voi, N_subnet, n_modes).
        If node_indices is provided (dict mapping subnet name -> global index array),
        output is connectome-ordered. Otherwise falls back to simple concatenation.

        Returns an array of shape (n_sweeps, n_voi, N_total, n_modes).
        """
        if not tavg_list:
            return np.array([])
        if node_indices is not None and len(node_indices) > 0:
            n_global = max(max(idxs) for idxs in node_indices.values()) + 1
            n_sweeps = tavg_list[0].shape[0]
            n_voi = tavg_list[0].shape[1]
            n_modes = tavg_list[0].shape[3]
            merged = np.zeros((n_sweeps, n_voi, n_global, n_modes), dtype=np.float32)
            for i, (name, tavg) in enumerate(zip(
                [sn.name for sn in subnetworks], tavg_list)):
                if name in node_indices:
                    idxs = node_indices[name]
                    merged[:, :, idxs, :] = tavg
                else:
                    # Fallback: concatenate in order
                    start = sum(t.shape[2] for t in tavg_list[:i])
                    merged[:, :, start:start + tavg.shape[2], :] = tavg
            return merged
        return np.concatenate(tavg_list, axis=2)

    def _run_single_batch(
        self,
        nstep: int,
        sweep_values_batch: np.ndarray,
        sn_states_h: dict,
        src_bufs_h: dict,
        tavg_h: dict,
        raw_h: dict,
        noise_h: dict,
        stim_h: dict,
        voi_idx_h: dict,
        sp_h: dict,
        proj_cfun_params: list,
        d_proj_data: dict,
        d_voi_idxs: dict,
        d_sps: dict,
        analysis: NetworkAnalysis,
        network_set: object,
        dt: float,
        horizon: int,
        chunk_size: Optional[int],
        monitor_type: int,
        monitor_period: int,
        step_offset: int,
        bold_istep: int,
        n_bold_samples: int,
        bold_states_h: dict,
        bold_params_h: dict,
        bold_voi_idx_h: dict,
        bold_out_h: dict,
        ctavg_h: dict,
        spatial_mean_h: dict,
        spatial_tavg_h: dict,
        gain_h: dict,
        proj_tavg_h: dict,
        verbose: bool,
    ):
        """Execute one batch of sweep points on GPU."""
        n_sweeps = sweep_values_batch.shape[0]

        # ---- Transfer to device ----
        t_xfer = time.perf_counter()

        d_states = {}
        for sn_info in analysis.subnetworks:
            d_states[sn_info.name] = cuda.to_device(sn_states_h[sn_info.name])

        d_srcbufs = {}
        for sn_info in analysis.subnetworks:
            d_srcbufs[sn_info.name] = cuda.to_device(src_bufs_h[sn_info.name])

        d_tavgs = {}
        for sn_info in analysis.subnetworks:
            d_tavgs[sn_info.name] = cuda.to_device(tavg_h[sn_info.name])

        d_ctavgs = {}
        for sn_info in analysis.subnetworks:
            d_ctavgs[sn_info.name] = cuda.to_device(ctavg_h[sn_info.name])

        d_spatial_means = {}
        d_spatial_tavgs = {}
        d_gains = {}
        d_proj_tavgs = {}
        for sn_info in analysis.subnetworks:
            d_spatial_means[sn_info.name] = cuda.to_device(spatial_mean_h[sn_info.name])
            d_spatial_tavgs[sn_info.name] = cuda.to_device(spatial_tavg_h[sn_info.name])
            d_gains[sn_info.name] = cuda.to_device(gain_h[sn_info.name])
            d_proj_tavgs[sn_info.name] = cuda.to_device(proj_tavg_h[sn_info.name])

        d_raws = {}
        for sn_info in analysis.subnetworks:
            if sn_info.name in raw_h:
                d_raws[sn_info.name] = cuda.to_device(raw_h[sn_info.name])

        d_sweep_params = cuda.to_device(sweep_values_batch)

        d_noises = {}
        for sn_info in analysis.subnetworks:
            if sn_info.is_stochastic:
                d_noises[sn_info.name] = cuda.to_device(noise_h[sn_info.name])

        d_stims = {}
        for sn_info in analysis.subnetworks:
            if sn_info.has_stimulus:
                d_stims[sn_info.name] = cuda.to_device(stim_h[sn_info.name])

        d_bold_states = {}
        d_bold_params = {}
        d_bold_voi_idxs = {}
        d_bold_outs = {}
        for sn_info in analysis.subnetworks:
            if sn_info.name in bold_states_h:
                d_bold_states[sn_info.name] = cuda.to_device(bold_states_h[sn_info.name])
                d_bold_params[sn_info.name] = cuda.to_device(bold_params_h[sn_info.name])
                d_bold_voi_idxs[sn_info.name] = cuda.to_device(bold_voi_idx_h[sn_info.name])
                d_bold_outs[sn_info.name] = cuda.to_device(bold_out_h[sn_info.name])

        t_xfer = time.perf_counter() - t_xfer

        # ---- Chunking setup ----
        if chunk_size is None:
            chunk_size = nstep
        total_nstep = nstep
        n_chunks = (total_nstep + chunk_size - 1) // chunk_size

        threads_per_block = 256
        blocks = (n_sweeps + threads_per_block - 1) // threads_per_block

        if verbose:
            print(f"[cuda_sweep batch] grid=({blocks},), block=({threads_per_block},)")
            print(f"[cuda_sweep batch] transfer in: {t_xfer*1000:.1f} ms")
            print(f"[cuda_sweep batch] chunks={n_chunks}, chunk_size={chunk_size}, n_sweeps={n_sweeps}")

        cuda.synchronize()
        t_kernel = time.perf_counter()

        # ---- Launch kernel chunks ----
        for chunk_idx in range(n_chunks):
            this_offset = step_offset + chunk_idx * chunk_size
            this_nstep = min(chunk_size, total_nstep - chunk_idx * chunk_size)
            if this_nstep <= 0:
                break

            args = []

            # States
            for sn_info in analysis.subnetworks:
                args.append(d_states[sn_info.name])

            # Source buffers
            for sn_info in analysis.subnetworks:
                args.append(d_srcbufs[sn_info.name])

            # tavg outputs
            for sn_info in analysis.subnetworks:
                args.append(d_tavgs[sn_info.name])

            # raw outputs
            for sn_info in analysis.subnetworks:
                if sn_info.name in d_raws:
                    args.append(d_raws[sn_info.name])
                else:
                    nvoi = len(sn_info.model.variables_of_interest)
                    args.append(
                        cuda.to_device(
                            np.empty((n_sweeps, 0, nvoi, sn_info.n_nodes, sn_info.n_modes), dtype=np.float32)
                        )
                    )

            # Bold arrays
            for sn_info in analysis.subnetworks:
                if sn_info.name in d_bold_states:
                    args.append(d_bold_states[sn_info.name])
                    args.append(d_bold_params[sn_info.name])
                    args.append(d_bold_voi_idxs[sn_info.name])
                    args.append(d_bold_outs[sn_info.name])
                else:
                    nvoi = len(sn_info.model.variables_of_interest)
                    args.append(
                        cuda.to_device(np.empty((n_sweeps, nvoi, 4, sn_info.n_nodes), dtype=np.float32))
                    )  # bold_state
                    args.append(cuda.to_device(np.empty((10,), dtype=np.float32)))  # bold_params
                    args.append(cuda.to_device(np.empty((nvoi,), dtype=np.int32)))  # bold_voi_idx
                    args.append(
                        cuda.to_device(np.empty((n_sweeps, 0, nvoi, sn_info.n_nodes), dtype=np.float32))
                    )  # bold_out

            # Sweep params
            args.append(d_sweep_params)

            # per-projection data
            for p in analysis.all_projections:
                pd = d_proj_data[p.name]
                args.append(pd['w_data'])
                args.append(pd['w_indices'])
                args.append(pd['w_indptr'])
                args.append(pd['idelays'])
                if p.is_inter:
                    args.append(pd['mode_map'])
                args.append(pd['source_cvar'])
                args.append(pd['target_cvar'])
                args.append(pd['scale'])
                args.append(pd['target_scales'])
                args.append(pd['cfun_params'])

            # per-subnet ctavg
            for sn_info in analysis.subnetworks:
                args.append(d_ctavgs[sn_info.name])

            # per-subnet spatial / projection monitor arrays
            for sn_info in analysis.subnetworks:
                args.append(d_spatial_means[sn_info.name])
                args.append(d_spatial_tavgs[sn_info.name])
                args.append(d_gains[sn_info.name])
                args.append(d_proj_tavgs[sn_info.name])

            # Per-subnet spatial params
            for sn_info in analysis.subnetworks:
                args.append(d_sps[sn_info.name])

            # Per-subnet voi_idx
            for sn_info in analysis.subnetworks:
                args.append(d_voi_idxs[sn_info.name])

            # Per-subnet noise (stochastic only)
            for sn_info in analysis.subnetworks:
                if sn_info.is_stochastic:
                    args.append(d_noises[sn_info.name])

            # Per-subnet stimulus (stimulus subnets only)
            for sn_info in analysis.subnetworks:
                if sn_info.has_stimulus:
                    args.append(d_stims[sn_info.name])

            # Scalar parameters
            args.append(np.int32(this_offset))
            args.append(np.int32(horizon))
            args.append(np.float32(dt))
            args.append(np.int32(this_nstep))
            args.append(np.int32(monitor_type))
            args.append(np.int32(monitor_period))
            for sn_info in analysis.subnetworks:
                args.append(np.int32(sn_info.n_nodes))
            args.append(np.int32(n_sweeps))
            args.append(np.int32(bold_istep))
            args.append(np.int32(n_bold_samples))

            self._compiled_fn[blocks, threads_per_block](*args)
            cuda.synchronize()

        t_kernel = time.perf_counter() - t_kernel

        # ---- Copy results back ----
        results_tavg = []
        results_ctavg = []
        results_spatial = []
        results_proj = []
        for sn_info in analysis.subnetworks:
            h = d_tavgs[sn_info.name].copy_to_host()
            # Normalize by total_nstep (kernel accumulates without normalizing)
            h /= np.float32(total_nstep)
            results_tavg.append(h)
            h_ct = d_ctavgs[sn_info.name].copy_to_host()
            h_ct /= np.float32(total_nstep)
            results_ctavg.append(h_ct)
            h_sp = d_spatial_tavgs[sn_info.name].copy_to_host()
            h_sp /= np.float32(total_nstep)
            results_spatial.append(h_sp)
            h_pr = d_proj_tavgs[sn_info.name].copy_to_host()
            h_pr /= np.float32(total_nstep)
            results_proj.append(h_pr)

        results_raw = []
        for sn_info in analysis.subnetworks:
            if sn_info.name in d_raws:
                results_raw.append(d_raws[sn_info.name].copy_to_host())

        results_bold = []
        for sn_info in analysis.subnetworks:
            if sn_info.name in d_bold_outs:
                results_bold.append(d_bold_outs[sn_info.name].copy_to_host())

        # ---- Copy final states / srcbufs for snapshot ----
        final_states = {}
        final_srcbufs = {}
        for sn_info in analysis.subnetworks:
            final_states[sn_info.name] = d_states[sn_info.name].copy_to_host()
            final_srcbufs[sn_info.name] = d_srcbufs[sn_info.name].copy_to_host()

        final_step_offset = step_offset + total_nstep

        if verbose:
            print(f"[cuda_sweep batch] kernel: {t_kernel*1000:.1f} ms")

        result = {
            'tavg': results_tavg,
            'ctavg': results_ctavg,
            'spatial_tavg': results_spatial,
            'proj_tavg': results_proj,
            'elapsed': t_kernel,
            'snapshot': {
                'states': final_states,
                'srcbufs': final_srcbufs,
                'step_offset': final_step_offset,
            },
        }
        if results_raw:
            result['raw'] = results_raw
        if results_bold:
            result['bold'] = results_bold
        return result

    def run(
        self,
        nstep: int,
        sweep_values: np.ndarray,
        initial_states: Optional[list] = None,
        chunk_size: Optional[int] = None,
        snapshot: Optional[dict] = None,
        monitor_type = 0,
        monitor_period: int = 1,
        max_batch_sweeps: Optional[int] = None,
        bold_period: Optional[float] = None,
        monitors: Optional[dict] = None,
        node_indices: Optional[dict] = None,
        global_average: bool = False,
        verbose: bool = False,
    ):
        """Execute the parameter sweep on GPU with automatic batching.

        Parameters
        ----------
        nstep : int
            Number of integration steps per sweep point.
        sweep_values : ndarray (n_sweeps, n_sweep_dims) or (n_sweeps,) float32
            Values of the sweep parameters.
        initial_states : list of ndarray, optional
            Initial state for each subnetwork.
        chunk_size : int, optional
            Steps per chunk. None means no chunking.
        snapshot : dict, optional
            Resume dict with keys 'states', 'srcbufs', 'step_offset', 'bold_states'.
        monitor_type : int or str
            0 or 'tavg' = temporal average only (default),
            1 or 'raw' = full step-by-step output,
            2 or 'subsample' = periodic subsampled output.
        monitor_period : int
            Steps between subsample writes (monitor_type == 2).
        max_batch_sweeps : int, optional
            Maximum sweep points per GPU batch. None = auto-detect from VRAM.
        bold_period : float, optional
            Bold monitor sampling period in ms. If provided, Balloon-Windkessel
            BOLD signal is computed and returned under the 'bold' key.
        monitors : dict, optional
            Monitor config dicts for spatial/projection monitors.
            {'spatial_mean': {subnet_name: ndarray(n_areas, N)},
             'gain': {subnet_name: ndarray(n_sensors, N)}}
        node_indices : dict, optional
            Global connectome node indices per subnet. If provided, merged output
            is connectome-ordered instead of naively concatenated.
        global_average : bool
            If True, compute mean across nodes in the 'ga_avg' output key.
        verbose : bool
            Print timing information.

        Returns
        -------
        dict with keys:
            'tavg': list of ndarray — temporal-average per subnetwork
            'merged_tavg': ndarray — concatenated tavg
            'ctavg': list of ndarray — coupling temporal average per subnetwork
            'spatial_tavg': list of ndarray — spatial average per subnetwork
            'proj_tavg': list of ndarray — projection output per subnetwork
            'ga_avg': dict — global average (if global_average=True)
            'raw': list of ndarray — optional raw output per subnetwork
            'bold': list of ndarray — optional BOLD output per subnetwork
            'elapsed': float — GPU kernel wall-clock seconds
            'snapshot': dict — final states, srcbufs, step_offset, bold_states
        """
        sweep_values = np.asarray(sweep_values, dtype=np.float32)
        if sweep_values.ndim == 1:
            sweep_values = sweep_values.reshape(-1, 1)
        if sweep_values.ndim != 2:
            raise ValueError(f"sweep_values must be 1D or 2D, got shape {sweep_values.shape}")
        n_sweeps, n_dims = sweep_values.shape
        if n_dims != self.n_sweep_dims:
            raise ValueError(f"Expected sweep_values with {self.n_sweep_dims} dimensions, got {n_dims}")
        # Convert string monitor_type to int
        _monitor_map = {'tavg': 0, 'raw': 1, 'subsample': 2}
        if isinstance(monitor_type, str):
            monitor_type = _monitor_map.get(monitor_type.lower(), 0)

        # Normalize monitors dict
        if monitors is None:
            monitors = {}
        if not isinstance(monitors, dict):
            monitors = {}

        analysis = self._analysis
        network_set = self._network_set

        # ---- Determine dt (must be same across subnets) ----
        dt = float(network_set.subnets[0].scheme.dt)

        # ---- Bold monitor setup ----
        bold_istep = 0
        n_bold_samples = 0
        bold_states_h = {}
        bold_params_h = {}
        bold_voi_idx_h = {}
        bold_out_h = {}
        if bold_period is not None:
            bp = float(bold_period)
            bold_istep = max(1, int(round(bp / dt)))
            n_bold_samples = (nstep + bold_istep - 1) // bold_istep
            tau_s = np.float32(0.65)
            tau_f = np.float32(0.41)
            tau_o = np.float32(0.98)
            alpha = np.float32(0.32)
            e0 = np.float32(0.4)
            V0 = np.float32(4.0)
            k1 = np.float32(4.3) * np.float32(40.3) * e0 * np.float32(0.04)
            k2 = np.float32(0.5) * np.float32(25.0) * e0 * np.float32(0.04)
            k3 = np.float32(1.0) - np.float32(0.5)
            bold_params_arr = np.array([
                np.float32(1.0) / tau_s,
                np.float32(1.0) / tau_f,
                np.float32(1.0) / tau_o,
                np.float32(1.0) / alpha,
                e0,
                np.float32(1.0) / e0,
                k1, k2, k3,
                V0,
            ], dtype=np.float32)
            for sn_info in analysis.subnetworks:
                n_voi = len(sn_info.model.variables_of_interest)
                n_nodes = sn_info.n_nodes
                svars = list(sn_info.model.state_variables)
                voi = list(sn_info.model.variables_of_interest)
                voi_idx = [svars.index(v) if v in svars else -1 for v in voi]
                bold_states_h[sn_info.name] = np.zeros((n_sweeps, n_voi, 4, n_nodes), dtype=np.float32)
                bold_states_h[sn_info.name][:, :, 1, :] = 1.0
                bold_states_h[sn_info.name][:, :, 2, :] = 1.0
                bold_states_h[sn_info.name][:, :, 3, :] = 1.0
                bold_params_h[sn_info.name] = bold_params_arr
                bold_voi_idx_h[sn_info.name] = np.array(voi_idx, dtype=np.int32)
                bold_out_h[sn_info.name] = np.zeros((n_sweeps, n_bold_samples, n_voi, n_nodes), dtype=np.float32)

        # ---- Determine max horizon ----
        if analysis.all_projections:
            horizon = max(p.horizon for p in analysis.all_projections)
        else:
            horizon = 1

        if verbose:
            for sn in analysis.subnetworks:
                print(f"  subnet '{sn.name}': model={type(sn.model).__name__}, "
                      f"nvar={sn.model.nvar}, N={sn.n_nodes}, modes={sn.n_modes}")
            for p in analysis.all_projections:
                print(f"  proj '{p.name}': {p.source_subnet}→{p.target_subnet}, "
                      f"nnz={len(p.weights_data)}, horizon={p.horizon}")

        # ---- Handle snapshot / resume ----
        step_offset = 0
        if snapshot is not None:
            step_offset = snapshot.get('step_offset', 0)
            sn_states_h = {}
            src_bufs_h = {}
            for sn_info in analysis.subnetworks:
                sn_states_h[sn_info.name] = snapshot['states'][sn_info.name].astype(np.float32)
                src_bufs_h[sn_info.name] = snapshot['srcbufs'][sn_info.name].astype(np.float32)
            # Restore Bold state from snapshot if available
            if bold_period is not None and 'bold_states' in snapshot:
                for sn_info in analysis.subnetworks:
                    if sn_info.name in snapshot['bold_states']:
                        bold_states_h[sn_info.name] = snapshot['bold_states'][sn_info.name].astype(np.float32)
        else:
            # Build per-subnet initial states
            sn_states_h = {}
            for i, sn_info in enumerate(analysis.subnetworks):
                if initial_states is not None:
                    x0 = initial_states[i].astype(np.float32)
                    if x0.ndim == 3:
                        # (nvar, nnodes, nmodes) — add sweep dimension
                        sn_states_h[sn_info.name] = np.broadcast_to(
                            x0[np.newaxis], (n_sweeps,) + x0.shape
                        ).copy()
                    elif x0.ndim == 4 and x0.shape[0] == n_sweeps:
                        # (n_sweeps, nvar, nnodes, nmodes) — already has sweep dimension
                        sn_states_h[sn_info.name] = x0.copy()
                    else:
                        raise ValueError(
                            f"initial_states[{i}] must have shape (nvar, nnodes, nmodes) "
                            f"or (n_sweeps, nvar, nnodes, nmodes), got {x0.shape}"
                        )
                else:
                    rng = np.random.RandomState(42 + i)
                    x0 = np.abs(rng.uniform(0.01, 0.05,
                                 (sn_info.model.nvar, sn_info.n_nodes, sn_info.n_modes))
                                 ).astype(np.float32)
                    sn_states_h[sn_info.name] = np.broadcast_to(
                        x0[np.newaxis], (n_sweeps,) + x0.shape
                    ).copy()

            # Build per-source-subnet history buffers
            src_bufs_h = {}
            for sn_info in analysis.subnetworks:
                hz = analysis.source_horizons.get(sn_info.name, 1)
                state = sn_states_h[sn_info.name]
                nvar, nnodes, nmodes = state.shape[1:]
                buf = np.zeros((n_sweeps, nvar, nnodes, nmodes, hz), dtype=np.float32)
                buf[:, :, :, :, :] = state[:, :, :, :, np.newaxis]
                src_bufs_h[sn_info.name] = buf

        # ---- Build per-subnet tavg accumulators (zeroed) ----
        tavg_h = {}
        for sn_info in analysis.subnetworks:
            nvoi = len(sn_info.model.variables_of_interest)
            nnodes = sn_info.n_nodes
            nmodes = sn_info.n_modes
            tavg_h[sn_info.name] = np.zeros((n_sweeps, nvoi, nnodes, nmodes), dtype=np.float32)

        # ---- Build per-subnet ctavg accumulators (zeroed) ----
        ctavg_h = {}
        for sn_info in analysis.subnetworks:
            ncvar = len(sn_info.model.coupling_terms)
            nnodes = sn_info.n_nodes
            nmodes = sn_info.n_modes
            ctavg_h[sn_info.name] = np.zeros((n_sweeps, ncvar, nnodes, nmodes), dtype=np.float32)

        # ---- Build per-subnet spatial / projection monitor arrays ----
        spatial_mean_h = {}
        spatial_tavg_h = {}
        gain_h = {}
        proj_tavg_h = {}
        monitors_dict = monitors if isinstance(monitors, dict) else {}
        for sn_info in analysis.subnetworks:
            nvoi = len(sn_info.model.variables_of_interest)
            nnodes = sn_info.n_nodes
            # SpatialAverage monitor
            sm = monitors_dict.get('spatial_mean', {}).get(sn_info.name)
            if sm is not None:
                spatial_mean_h[sn_info.name] = np.ascontiguousarray(sm.astype(np.float32))
                n_areas = sm.shape[0]
            else:
                spatial_mean_h[sn_info.name] = np.zeros((0, nnodes), dtype=np.float32)
                n_areas = 0
            n_areas = spatial_mean_h[sn_info.name].shape[0]
            spatial_tavg_h[sn_info.name] = np.zeros((n_sweeps, nvoi, n_areas, 1), dtype=np.float32)
            # Projection monitor
            gm = monitors_dict.get('gain', {}).get(sn_info.name)
            if gm is not None:
                gain_h[sn_info.name] = np.ascontiguousarray(gm.astype(np.float32))
                n_sensors = gm.shape[0]
            else:
                gain_h[sn_info.name] = np.zeros((0, nnodes), dtype=np.float32)
                n_sensors = 0
            n_sensors = gain_h[sn_info.name].shape[0]
            proj_tavg_h[sn_info.name] = np.zeros((n_sweeps, nvoi, n_sensors, 1), dtype=np.float32)

        # ---- Build per-subnet raw output accumulators (zeroed) ----
        raw_h = {}
        if monitor_type == 1:
            for sn_info in analysis.subnetworks:
                nvoi = len(sn_info.model.variables_of_interest)
                nnodes = sn_info.n_nodes
                nmodes = sn_info.n_modes
                raw_h[sn_info.name] = np.zeros((n_sweeps, nstep, nvoi, nnodes, nmodes), dtype=np.float32)
        elif monitor_type == 2:
            n_raw_steps = (nstep + monitor_period - 1) // monitor_period
            for sn_info in analysis.subnetworks:
                nvoi = len(sn_info.model.variables_of_interest)
                nnodes = sn_info.n_nodes
                nmodes = sn_info.n_modes
                raw_h[sn_info.name] = np.zeros((n_sweeps, n_raw_steps, nvoi, nnodes, nmodes), dtype=np.float32)

        # ---- Build per-subnet voi index arrays ----
        voi_idx_h = {}
        for sn_info in analysis.subnetworks:
            svars = list(sn_info.model.state_variables)
            voi = list(sn_info.model.variables_of_interest)
            voi_idx = np.array([svars.index(v) if v in svars else -1 for v in voi], dtype=np.int32)
            voi_idx_h[sn_info.name] = voi_idx

        # ---- Build per-subnet noise arrays (stochastic subnets only) ----
        noise_h = {}
        for sn_info in analysis.subnetworks:
            if sn_info.is_stochastic:
                rng = sn_info.integrator.noise.random_stream
                nvar = sn_info.model.nvar
                nnodes = sn_info.n_nodes
                nmodes = sn_info.n_modes
                # Cast to float32 early to avoid doubling memory during transpose
                dw = rng.randn(n_sweeps, nstep, nvar, nnodes, nmodes).astype(np.float32, copy=False)
                noise_std = np.float32(np.sqrt(2.0 * sn_info.noise_nsig * dt))
                dw *= noise_std[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
                dw = np.ascontiguousarray(np.transpose(dw, (0, 2, 3, 4, 1)))
                noise_h[sn_info.name] = dw

        # ---- Build per-subnet stimulus arrays (stimulus subnets only) ----
        stim_h = {}
        for sn_info in analysis.subnetworks:
            if sn_info.has_stimulus:
                # Stimulus indexing: CPU backend uses stim.get_coupling(step_idx) with
                # step_idx in [1, nstep]; CUDA kernel uses stim[tid, ic, j, m, t - 1]
                # with t = t_local + t_offset. Both produce 0-based indexing into the
                # stimulus array. The last dimension is step index in [0, nstep).
                n_cvar = len(sn_info.model.coupling_terms)
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
                        stim_arr[:, :, :, step_idx - 1] += sc
                stim_broadcast = np.broadcast_to(
                    stim_arr[np.newaxis], (n_sweeps, n_cvar, sn_info.n_nodes, sn_info.n_modes, nstep)
                ).copy()
                stim_h[sn_info.name] = stim_broadcast

        # ---- Build per-subnet spatial parameter arrays ----
        sp_h = {}
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
            sp_h[sn_info.name] = sp_arr

        # ---- Build per-projection cfun params ----
        proj_cfun_params = []
        for p in analysis.all_projections:
            cp = _cfun_params(p)
            proj_cfun_params.append(cp)

        # ---- Memory estimation ----
        bytes_per_sweep = 0
        for sn_info in analysis.subnetworks:
            nvar = sn_info.model.nvar
            nnodes = sn_info.n_nodes
            nmodes = sn_info.n_modes
            hz = analysis.source_horizons.get(sn_info.name, 1)
            bytes_per_sweep += nvar * nnodes * nmodes * 4
            bytes_per_sweep += nvar * nnodes * nmodes * hz * 4
            nvoi = len(sn_info.model.variables_of_interest)
            bytes_per_sweep += nvoi * nnodes * nmodes * 4

        for sn_info in analysis.subnetworks:
            if sn_info.is_stochastic:
                nvar = sn_info.model.nvar
                nnodes = sn_info.n_nodes
                nmodes = sn_info.n_modes
                bytes_per_sweep += nvar * nnodes * nmodes * nstep * 4

        for sn_info in analysis.subnetworks:
            if sn_info.has_stimulus:
                n_cvar = len(sn_info.model.coupling_terms)
                nnodes = sn_info.n_nodes
                nmodes = sn_info.n_modes
                bytes_per_sweep += n_cvar * nnodes * nmodes * nstep * 4

        if monitor_type > 0:
            n_raw_steps = nstep if monitor_type == 1 else (nstep + monitor_period - 1) // monitor_period
            for sn_info in analysis.subnetworks:
                nvoi = len(sn_info.model.variables_of_interest)
                nnodes = sn_info.n_nodes
                nmodes = sn_info.n_modes
                bytes_per_sweep += nvoi * nnodes * nmodes * n_raw_steps * 4

        if bold_period is not None:
            for sn_info in analysis.subnetworks:
                nvoi = len(sn_info.model.variables_of_interest)
                nnodes = sn_info.n_nodes
                bytes_per_sweep += nvoi * nnodes * 4 * 4  # bold_state (4 floats per voi per node)
                bytes_per_sweep += 10 * 4  # bold_params
                bytes_per_sweep += nvoi * 4  # bold_voi_idx
                bytes_per_sweep += n_bold_samples * nvoi * nnodes * 4  # bold_out

        # Available GPU memory
        try:
            gpu_available = cuda.current_context().get_memory_info()[0]
            budget = int(gpu_available * 0.8)
        except Exception:
            budget = 20 * 1024**3  # 20 GB fallback

        if max_batch_sweeps is None:
            max_batch_sweeps = max(1, budget // max(1, bytes_per_sweep))
        max_batch_sweeps = max(1, int(max_batch_sweeps))
        max_batch_sweeps = min(max_batch_sweeps, n_sweeps)

        n_batches = (n_sweeps + max_batch_sweeps - 1) // max_batch_sweeps

        if verbose and n_batches > 1:
            print(f"[cuda_sweep] memory budget={budget/1e9:.2f} GB, "
                  f"bytes_per_sweep={bytes_per_sweep}, "
                  f"max_batch_sweeps={max_batch_sweeps}, n_batches={n_batches}")

        # ---- Pre-build constant device arrays (shared across batches) ----
        d_proj_data = {}
        for p in analysis.all_projections:
            d_proj_data[p.name] = {
                'w_data': cuda.to_device(p.weights_data.astype(np.float32)),
                'w_indices': cuda.to_device(p.weights_indices.astype(np.int32)),
                'w_indptr': cuda.to_device(p.weights_indptr.astype(np.int32)),
                'idelays': cuda.to_device(p.idelays.astype(np.int32)),
                'source_cvar': cuda.to_device(p.source_cvar.astype(np.int32)),
                'target_cvar': cuda.to_device(p.target_cvar.astype(np.int32)),
                'scale': np.float32(p.scale),
                'target_scales': cuda.to_device(p.target_scales.astype(np.float32)),
                'cfun_params': cuda.to_device(proj_cfun_params[analysis.all_projections.index(p)]),
            }
            if p.is_inter:
                if p.mode_map is not None:
                    d_proj_data[p.name]['mode_map'] = cuda.to_device(p.mode_map.astype(np.float32))
                else:
                    n_src_m = p.n_src_modes
                    n_tgt_m = p.n_tgt_modes
                    mm = np.ones((n_src_m, n_tgt_m), dtype=np.float32)
                    d_proj_data[p.name]['mode_map'] = cuda.to_device(mm)

        d_voi_idxs = {}
        for sn_info in analysis.subnetworks:
            d_voi_idxs[sn_info.name] = cuda.to_device(voi_idx_h[sn_info.name])

        d_sps = {}
        for sn_info in analysis.subnetworks:
            d_sps[sn_info.name] = cuda.to_device(sp_h[sn_info.name])

        # ---- Batch loop ----
        result_tavg = []
        result_ctavg = []
        result_spatial = []
        result_proj = []
        result_raw = None
        result_bold = None
        total_elapsed = 0.0
        snapshot_states = {}
        snapshot_srcbufs = {}

        for batch_idx in range(n_batches):
            batch_start = batch_idx * max_batch_sweeps
            batch_end = min(batch_start + max_batch_sweeps, n_sweeps)
            batch_n = batch_end - batch_start

            # ---- Slice sweep-dependent host arrays ----
            batch_sv = sweep_values[batch_start:batch_end]

            batch_sn_states = {}
            batch_src_bufs = {}
            for sn_info in analysis.subnetworks:
                batch_sn_states[sn_info.name] = sn_states_h[sn_info.name][batch_start:batch_end]
                batch_src_bufs[sn_info.name] = src_bufs_h[sn_info.name][batch_start:batch_end]

            batch_tavg = {}
            for sn_info in analysis.subnetworks:
                batch_tavg[sn_info.name] = tavg_h[sn_info.name][batch_start:batch_end]

            batch_raw = {}
            for sn_info in analysis.subnetworks:
                if sn_info.name in raw_h:
                    batch_raw[sn_info.name] = raw_h[sn_info.name][batch_start:batch_end]

            batch_noise = {}
            for sn_info in analysis.subnetworks:
                if sn_info.is_stochastic:
                    batch_noise[sn_info.name] = noise_h[sn_info.name][batch_start:batch_end]

            batch_stim = {}
            for sn_info in analysis.subnetworks:
                if sn_info.has_stimulus:
                    batch_stim[sn_info.name] = stim_h[sn_info.name][batch_start:batch_end]

            batch_bold_states = {}
            batch_bold_params = {}
            batch_bold_voi_idx = {}
            batch_bold_out = {}
            for sn_info in analysis.subnetworks:
                if sn_info.name in bold_states_h:
                    batch_bold_states[sn_info.name] = bold_states_h[sn_info.name][batch_start:batch_end]
                    batch_bold_params[sn_info.name] = bold_params_h[sn_info.name]
                    batch_bold_voi_idx[sn_info.name] = bold_voi_idx_h[sn_info.name]
                    batch_bold_out[sn_info.name] = bold_out_h[sn_info.name][batch_start:batch_end]

            # Slice ctavg, spatial, proj for this batch
            batch_ctavg = {}
            batch_spatial_tavg = {}
            batch_proj_tavg = {}
            for sn_info in analysis.subnetworks:
                batch_ctavg[sn_info.name] = ctavg_h[sn_info.name][batch_start:batch_end]
                batch_spatial_tavg[sn_info.name] = spatial_tavg_h[sn_info.name][batch_start:batch_end]
                batch_proj_tavg[sn_info.name] = proj_tavg_h[sn_info.name][batch_start:batch_end]

            # ---- Run batch ----
            batch_result = self._run_single_batch(
                nstep=nstep,
                sweep_values_batch=batch_sv,
                sn_states_h=batch_sn_states,
                src_bufs_h=batch_src_bufs,
                tavg_h=batch_tavg,
                raw_h=batch_raw,
                noise_h=batch_noise,
                stim_h=batch_stim,
                voi_idx_h=voi_idx_h,
                sp_h=sp_h,
                proj_cfun_params=proj_cfun_params,
                d_proj_data=d_proj_data,
                d_voi_idxs=d_voi_idxs,
                d_sps=d_sps,
                analysis=analysis,
                network_set=network_set,
                dt=dt,
                horizon=horizon,
                chunk_size=chunk_size,
                monitor_type=monitor_type,
                monitor_period=monitor_period,
                step_offset=step_offset,
                bold_istep=bold_istep,
                n_bold_samples=n_bold_samples,
                bold_states_h=batch_bold_states,
                bold_params_h=batch_bold_params,
                bold_voi_idx_h=batch_bold_voi_idx,
                bold_out_h=batch_bold_out,
                ctavg_h=batch_ctavg,
                spatial_mean_h=spatial_mean_h,
                spatial_tavg_h=batch_spatial_tavg,
                gain_h=gain_h,
                proj_tavg_h=batch_proj_tavg,
                verbose=verbose and batch_idx == 0,
            )

            # ---- Accumulate results ----
            for i, tavg_arr in enumerate(batch_result['tavg']):
                if batch_idx == 0:
                    result_tavg.append(np.zeros((n_sweeps,) + tavg_arr.shape[1:], dtype=np.float32))
                result_tavg[i][batch_start:batch_end] = tavg_arr

            for i, ct_arr in enumerate(batch_result['ctavg']):
                if batch_idx == 0:
                    result_ctavg.append(np.zeros((n_sweeps,) + ct_arr.shape[1:], dtype=np.float32))
                result_ctavg[i][batch_start:batch_end] = ct_arr

            for i, sp_arr in enumerate(batch_result['spatial_tavg']):
                if batch_idx == 0:
                    result_spatial.append(np.zeros((n_sweeps,) + sp_arr.shape[1:], dtype=np.float32))
                result_spatial[i][batch_start:batch_end] = sp_arr

            for i, pr_arr in enumerate(batch_result['proj_tavg']):
                if batch_idx == 0:
                    result_proj.append(np.zeros((n_sweeps,) + pr_arr.shape[1:], dtype=np.float32))
                result_proj[i][batch_start:batch_end] = pr_arr

            if batch_result.get('raw') is not None:
                if result_raw is None:
                    result_raw = []
                for i, raw_arr in enumerate(batch_result['raw']):
                    if batch_idx == 0:
                        result_raw.append(np.zeros((n_sweeps,) + raw_arr.shape[1:], dtype=np.float32))
                    result_raw[i][batch_start:batch_end] = raw_arr

            if batch_result.get('bold') is not None:
                if result_bold is None:
                    result_bold = []
                for i, bold_arr in enumerate(batch_result['bold']):
                    if batch_idx == 0:
                        result_bold.append(np.zeros((n_sweeps,) + bold_arr.shape[1:], dtype=np.float32))
                    result_bold[i][batch_start:batch_end] = bold_arr

            total_elapsed += batch_result['elapsed']

            for sn_info in analysis.subnetworks:
                if batch_idx == 0:
                    snapshot_states[sn_info.name] = np.zeros(
                        (n_sweeps,) + batch_result['snapshot']['states'][sn_info.name].shape[1:],
                        dtype=np.float32
                    )
                    snapshot_srcbufs[sn_info.name] = np.zeros(
                        (n_sweeps,) + batch_result['snapshot']['srcbufs'][sn_info.name].shape[1:],
                        dtype=np.float32
                    )
                snapshot_states[sn_info.name][batch_start:batch_end] = batch_result['snapshot']['states'][sn_info.name]
                snapshot_srcbufs[sn_info.name][batch_start:batch_end] = batch_result['snapshot']['srcbufs'][sn_info.name]

        merged_tavg = self._merge_subnet_outputs(result_tavg, analysis.subnetworks, node_indices)

        # GlobalAverage: mean across node dimension
        ga_avg = None
        if global_average:
            ga_avg = {}
            for i, sn_info in enumerate(analysis.subnetworks):
                ga_avg[sn_info.name] = result_tavg[i].mean(axis=2, keepdims=True)  # (n_sweeps, n_voi, 1, n_modes)

        final_step_offset = step_offset + nstep

        if verbose:
            print(f"[cuda_sweep] total kernel time: {total_elapsed*1000:.1f} ms")
            if n_batches > 1:
                print(f"[cuda_sweep] throughput: {n_sweeps * nstep / total_elapsed / 1000:.1f} kstep/s")

        result = {
            'tavg': result_tavg,
            'merged_tavg': merged_tavg,
            'ctavg': result_ctavg,
            'spatial_tavg': result_spatial,
            'proj_tavg': result_proj,
            'elapsed': total_elapsed,
            'snapshot': {
                'states': snapshot_states,
                'srcbufs': snapshot_srcbufs,
                'step_offset': final_step_offset,
            },
        }
        # Add Bold state to snapshot if Bold was enabled
        if bold_period is not None:
            result['snapshot']['bold_states'] = bold_states_h
        if ga_avg is not None:
            result['ga_avg'] = ga_avg
        if result_raw is not None:
            result['raw'] = result_raw
        if result_bold is not None:
            result['bold'] = result_bold
        return result
