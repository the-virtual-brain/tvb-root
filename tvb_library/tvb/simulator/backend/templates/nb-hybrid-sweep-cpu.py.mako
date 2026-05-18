## -*- coding: utf-8 -*-
##
## nb-hybrid-sweep-cpu.py.mako
##
## Generates a @nb.njit(parallel=True) sweep kernel that wraps the
## existing network_chunk in a nb.prange loop for multi-core CPU sweeps.
##
## Required content dict keys:
##   analysis  — NetworkAnalysis instance (see nb_hybrid.py)
##   np        — numpy module reference
##   debug_nojit: bool — if True, skip @nb.njit decorators (for debugging)
##
## This template is appended to the single-sim template output so that
## sweep_kernel can reference network_chunk and compute_coupling_* directly.

import math
import numpy as np
import numba as nb
sin, cos, exp, log = math.sin, math.cos, math.exp, math.log

<%
from tvb.simulator.backend.nb_hybrid import _cfun_type, _cvar_mapping_mode
subnets = analysis.subnetworks
inter_projs = analysis.inter_projections
intra_projs = analysis.intra_projections
all_projs = analysis.all_projections
source_horizons_map = analysis.source_horizons
sn_ncvar_dict = {sn.name: len(sn.model.cvar) for sn in subnets}
sn_nnodes_dict = {sn.name: sn.n_nodes for sn in subnets}
sn_nmodes_dict = {sn.name: sn.n_modes for sn in subnets}
sn_nvoi_dict = {sn.name: len(sn.model.variables_of_interest) for sn in subnets}
%>

## ============================================================
## CPU parallel sweep kernel
## ============================================================

${'' if debug_nojit else '@nb.njit(parallel=True, cache=True)'}
def sweep_kernel(
    _n_sweeps, _nstep, _t_start,
    _sweep_params,  # (n_sweeps, n_sweep_dims) float32
    ## per-subnet state arrays (leading n_sweeps dimension)
    % for sn in subnets:
    ${sn.name}_state_all,   # (n_sweeps, n_svars, n_nodes, n_modes) float32
    ${sn.name}_srcbuf_all,  # (n_sweeps, n_vars, n_nodes, n_modes, max_horizon) float32
    % endfor
    ## per-projection arrays (shared, read-only)
    % for p in all_projs:
    ${p.name}_w_data, ${p.name}_w_indices, ${p.name}_w_indptr,
    ${p.name}_idelays,
    % if p.is_inter:
    ${p.name}_mode_map,
    % endif
    ${p.name}_source_cvar, ${p.name}_target_cvar, ${p.name}_target_cvar_cpl,
    ${p.name}_scale, ${p.name}_target_scales,
    ${p.name}_cfun_params_all,  # (n_sweeps, n_params) float32 — per-sweep copies
    % endfor
    ## per-subnet tavg output arrays (leading n_sweeps dimension)
    % for sn in subnets:
    ${sn.name}_tavg_all,     # (n_sweeps, n_voi, n_nodes, n_modes)
    ${sn.name}_ctavg_all,    # (n_sweeps, n_cvar, n_nodes, n_modes)
    ${sn.name}_c_all,        # (n_sweeps, n_cvar, n_nodes, n_modes) — coupling scratch
    % endfor
    _tavg_count_all,  # (n_sweeps,) int32 — shared tavg counter
    ## per-subnet noise arrays (stochastic only — shared across sweeps)
    % for sn in subnets:
    % if sn.is_stochastic:
    ${sn.name}_noise,  # (n_vars, n_nodes, n_modes, nstep_total) float32
    % endif
    % endfor
    ## per-subnet stimulus arrays (shared)
    % for sn in subnets:
    % if sn.has_stimulus:
    ${sn.name}_stim,  # (n_cvar, n_nodes, n_modes, nstep_total) float32
    % endif
    % endfor
    ## per-subnet spatial parameter arrays (shared, read-only)
    % for sn in subnets:
    ${sn.name}_sp,
    % endfor
    ## per-subnet monitor arrays
    % for sn in subnets:
    ${sn.name}_spatial_mean,  # (n_areas, n_nodes) float32 — shared
    ${sn.name}_spatial_tavg_all,  # (n_sweeps, n_voi, n_areas, n_modes) float32
    ${sn.name}_gain,  # (n_sensors, n_nodes) float32 — shared
    ${sn.name}_proj_tavg_all,  # (n_sweeps, n_voi, n_sensors, 1) float32
    % endfor
    ## per-subnet Bold Balloon model arrays
    % for sn in subnets:
    ${sn.name}_bold_state_all,  # (n_sweeps, n_voi, 4, n_nodes) float32
    ${sn.name}_bold_params,     # (9,) float32 — shared
    ${sn.name}_bold_voi_idx,   # (n_voi,) int32 — shared
    % endfor
    _bold_dt,  # float32
):
    for _tid in nb.prange(_n_sweeps):
        ## Per-sweep state views
        % for sn in subnets:
        ${sn.name}_state = ${sn.name}_state_all[_tid]
        ${sn.name}_srcbuf = ${sn.name}_srcbuf_all[_tid]
        ${sn.name}_tavg = ${sn.name}_tavg_all[_tid]
        ${sn.name}_ctavg = ${sn.name}_ctavg_all[_tid]
        ${sn.name}_c = ${sn.name}_c_all[_tid]
        % endfor
        ## Per-sweep cfun_params views
        % for p in all_projs:
        ${p.name}_cfun_params = ${p.name}_cfun_params_all[_tid]
        % endfor
        ## Per-sweep monitor views
        % for sn in subnets:
        ${sn.name}_spatial_tavg = ${sn.name}_spatial_tavg_all[_tid]
        ${sn.name}_proj_tavg = ${sn.name}_proj_tavg_all[_tid]
        ${sn.name}_bold_state = ${sn.name}_bold_state_all[_tid]
        % endfor

        ## Zero accumulators for this sweep point
        % for sn in subnets:
        ${sn.name}_tavg[:, :, :] = np.float32(0.0)
        ${sn.name}_ctavg[:, :, :] = np.float32(0.0)
        ${sn.name}_c[:, :, :] = np.float32(0.0)
        ${sn.name}_spatial_tavg[:, :, :] = np.float32(0.0)
        ${sn.name}_proj_tavg[:, :, :] = np.float32(0.0)
        % endfor
        _tavg_count_all[_tid] = 0
        tavg_count = _tavg_count_all[_tid:_tid+1]  ## 1-element view, no heap allocation

        ## Call network_chunk for this sweep point
        network_chunk(
            _nstep,
            _t_start,
            ## per-subnet states
            % for sn in subnets:
            ${sn.name}_state,
            % endfor
            ## per-subnet source buffers
            % for sn in subnets:
            ${sn.name}_srcbuf,
            % endfor
            ## per-projection arrays
            % for p in all_projs:
            ${p.name}_w_data, ${p.name}_w_indices, ${p.name}_w_indptr,
            ${p.name}_idelays,
            % if p.is_inter:
            ${p.name}_mode_map,
            % endif
            ${p.name}_source_cvar, ${p.name}_target_cvar, ${p.name}_target_cvar_cpl,
            ${p.name}_scale, ${p.name}_target_scales,
            ${p.name}_cfun_params,
            % endfor
            ## per-subnet tavg accumulators
            % for sn in subnets:
            ${sn.name}_tavg,
            % endfor
            tavg_count,
            ## per-subnet ctavg accumulators
            % for sn in subnets:
            ${sn.name}_ctavg,
            % endfor
            ## per-subnet coupling scratch arrays
            % for sn in subnets:
            ${sn.name}_c,
            % endfor
            ## per-subnet noise (stochastic only)
            % for sn in subnets:
            % if sn.is_stochastic:
            ${sn.name}_noise,
            % endif
            % endfor
            ## per-subnet stimulus
            % for sn in subnets:
            % if sn.has_stimulus:
            ${sn.name}_stim,
            % endif
            % endfor
            ## per-subnet spatial parameters
            % for sn in subnets:
            ${sn.name}_sp,
            % endfor
            ## per-subnet monitor accumulators
            % for sn in subnets:
            ${sn.name}_spatial_mean,
            ${sn.name}_spatial_tavg,
            ${sn.name}_gain,
            ${sn.name}_proj_tavg,
            % endfor
            ## per-subnet Bold Balloon arrays
            % for sn in subnets:
            ${sn.name}_bold_state,
            ${sn.name}_bold_params,
            ${sn.name}_bold_voi_idx,
            % endfor
            _bold_dt,
        )

        ## tavg_count is already a view into _tavg_count_all — no writeback needed