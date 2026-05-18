## -*- coding: utf-8 -*-
##
## nb-hybrid-sweep-cuda.py.mako
##
## Generates a @cuda.jit parameter-sweep kernel: one GPU thread per sweep point,
## each running an independent simulation.  Supports multi-subnet, multi-projection
## networks with CSR sparse coupling and real tract-length delays.
##
## v2 scope: multi-mode, combined dfun, chunking resume, Raw/SubSample monitors.

import math
import numpy as np
from numba import cuda
import numba

<%
    def _cparam(pname, pidx):
        d = sweep_cfun_dims.get(pname, {})
        if pidx in d:
            return f"sweep_params[tid, {d[pidx]}]"
        return f"{pname}_cfun_params[{pidx}]"
%>

##
## ---- Coupling device functions (one per projection) ----
##

% for p in all_projs:
<%
    ct = _cfun_type(p)
    cm = _cvar_mapping_mode(p)
    n_src_cvar = p.source_cvar.shape[0]
    n_tgt_cvar = p.target_cvar.shape[0]
    src = p.source_subnet
    tgt = p.target_subnet
    # pre_ct: coupling functions that apply PER-EDGE before weighting
    if ct in ('sigmoidal_jr', 'sigmoidal_jr_legacy', 'tanh', 'pre_sigmoidal', 'pre_sigmoidal_dynamic', 'difference', 'kuramoto'):
        pre_ct = ct
    else:
        pre_ct = 'none'
    # post_ct: coupling functions that apply AFTER weighted sum
    if ct == 'linear':
        post_ct = 'linear'
    elif ct in ('scaling', 'difference'):
        post_ct = 'scaling'
    elif ct == 'sigmoidal':
        post_ct = 'sigmoidal'
    elif ct == 'kuramoto':
        post_ct = 'kuramoto_norm'  # a * (1/N) * wsum — sin is now in pre
    else:
        post_ct = 'none'
    needs_xi = _needs_xi(p)
    is_multi_cvar = (ct in ('sigmoidal_jr', 'pre_sigmoidal_dynamic'))
    has_ts = p.target_scales.shape[0] > 0
%>

@cuda.jit(device=True)
def compute_coupling_${p.name}(
    ${src}_srcbuf,
    ${p.name}_w_data,
    ${p.name}_w_indices,
    ${p.name}_w_indptr,
    ${p.name}_idelays,
    horizon,
% if src != tgt:
    N_${src},
% endif
    N_${tgt},
    t,
    tidy,
    ${p.name}_source_cvar,
    ${p.name}_target_cvar, ${p.name}_target_cvar_cpl,
    ${p.name}_scale,
    ${p.name}_target_scales,
    ${p.name}_cfun_params,
% if p.is_inter:
    ${p.name}_mode_map,
% endif
    ${tgt}_state,
    ${tgt}_c,
    sweep_params,
    tid,
):
    """Compute coupling from ${src} → ${tgt} via CSR sparse weights + delays."""
    ## Per-target-cvar, per-source-mode wsum accumulators (allocated once, zeroed per target node)
    % if cm in ('1_to_1', 'n_to_n'):
    % for ic in range(n_tgt_cvar):
    ${p.name}_wsum_${ic} = cuda.local.array((${p.n_src_modes},), dtype=numba.float32)
    % endfor
    % elif cm == 'many_to_1':
    ${p.name}_wsum_0 = cuda.local.array((${p.n_src_modes},), dtype=numba.float32)
    % elif cm == '1_to_many':
    % for ic in range(n_tgt_cvar):
    ${p.name}_wsum_${ic} = cuda.local.array((${p.n_src_modes},), dtype=numba.float32)
    % endfor
    % endif

    for j in range(N_${tgt}):
        row_start = ${p.name}_w_indptr[j]
        row_end = ${p.name}_w_indptr[j + 1]

        ## Per-target-cvar, per-source-mode wsum accumulators (zeroed per target node)
        % for ic in range(n_tgt_cvar):
        for _ms in range(${p.n_src_modes}):
            ${p.name}_wsum_${ic}[_ms] = np.float32(0.0)
        % endfor

        for ptr in range(row_start, row_end):
            src_node = ${p.name}_w_indices[ptr]
            w = ${p.name}_w_data[ptr]
            d = ${p.name}_idelays[ptr]
            slot = (t - 1 - d + horizon) % horizon

            ## cvar accumulation with per-edge pre-cfun and source-mode loop
            % if cm in ('1_to_1', 'n_to_n'):
            % for ic in range(n_tgt_cvar):
            for _ms in range(${p.n_src_modes}):
                _edge_val = ${src}_srcbuf[tidy, ${p.source_cvar[ic]}, src_node, _ms, slot]
                % if pre_ct == 'sigmoidal_jr':
                # Classic: cmin + (cmax-cmin)/(1+exp(r*(midpoint-diff)))
                # cfun_params: [0]=a, [1]=cmin, [2]=cmax, [3]=r, [4]=midpoint
                cv0 = ${p.name}_source_cvar[0]
                cv1 = ${p.name}_source_cvar[1]
                x0 = ${src}_srcbuf[tidy, cv0, src_node, _ms, slot]
                x1 = ${src}_srcbuf[tidy, cv1, src_node, _ms, slot]
                _edge_val = ${p.name}_cfun_params[1] + (${p.name}_cfun_params[2] - ${p.name}_cfun_params[1]) / (np.float32(1.0) + math.exp(${p.name}_cfun_params[3] * (${p.name}_cfun_params[4] - (x0 - x1))))
                % elif pre_ct == 'sigmoidal_jr_legacy':
                # Legacy: a * 2*e0 / (1+exp(r*(v0-x)))
                # cfun_params: [0]=a, [1]=e0, [2]=r, [3]=v0
                _edge_val = ${_cparam(p.name, 0)} * np.float32(2.0) * ${p.name}_cfun_params[1] / (np.float32(1.0) + math.exp(${p.name}_cfun_params[2] * (${p.name}_cfun_params[3] - _edge_val)))
                % elif pre_ct == 'tanh':
                _edge_val = ${_cparam(p.name, 0)} * (np.float32(1.0) + math.tanh((${p.name}_cfun_params[1] * _edge_val - ${p.name}_cfun_params[2]) / ${p.name}_cfun_params[3]))
                % elif pre_ct == 'pre_sigmoidal':
                _edge_val = ${_cparam(p.name, 0)} * (${p.name}_cfun_params[1] + math.tanh(${p.name}_cfun_params[2] * (${p.name}_cfun_params[3] * _edge_val - ${p.name}_cfun_params[4])))
                % elif pre_ct == 'pre_sigmoidal_dynamic':
                # Dynamic: H*(Q + tanh(G*(P*x_j[0] - x_j[1])))
                # cfun_params: [0]=H, [1]=Q, [2]=G, [3]=P
                cv0 = ${p.name}_source_cvar[0]
                cv1 = ${p.name}_source_cvar[1]
                x0 = ${src}_srcbuf[tidy, cv0, src_node, _ms, slot]
                x1 = ${src}_srcbuf[tidy, cv1, src_node, _ms, slot]
                _edge_val = ${p.name}_cfun_params[0] * (${p.name}_cfun_params[1] + math.tanh(${p.name}_cfun_params[2] * (${p.name}_cfun_params[3] * x0 - x1)))
                % elif pre_ct == 'difference':
                xi_val = ${tgt}_state[tidy, ${p.name}_target_cvar[ic], j, _ms]
                _edge_val = _edge_val - xi_val
                % elif pre_ct == 'kuramoto':
                xi_val = ${tgt}_state[tidy, ${p.name}_target_cvar[ic], j, _ms]
                _edge_val = math.sin(_edge_val - xi_val)
                % endif
                ${p.name}_wsum_${ic}[_ms] += w * _edge_val
            % endfor
            % elif cm == 'many_to_1':
            for _ms in range(${p.n_src_modes}):
                % for ic in range(n_src_cvar):
                _edge_val = ${src}_srcbuf[tidy, ${p.source_cvar[ic]}, src_node, _ms, slot]
                % if pre_ct == 'sigmoidal_jr':
                # Classic: cmin + (cmax-cmin)/(1+exp(r*(midpoint-diff)))
                # cfun_params: [0]=a, [1]=cmin, [2]=cmax, [3]=r, [4]=midpoint
                cv0 = ${p.name}_source_cvar[0]
                cv1 = ${p.name}_source_cvar[1]
                x0 = ${src}_srcbuf[tidy, cv0, src_node, _ms, slot]
                x1 = ${src}_srcbuf[tidy, cv1, src_node, _ms, slot]
                _edge_val = ${p.name}_cfun_params[1] + (${p.name}_cfun_params[2] - ${p.name}_cfun_params[1]) / (np.float32(1.0) + math.exp(${p.name}_cfun_params[3] * (${p.name}_cfun_params[4] - (x0 - x1))))
                % elif pre_ct == 'sigmoidal_jr_legacy':
                # Legacy: a * 2*e0 / (1+exp(r*(v0-x)))
                # cfun_params: [0]=a, [1]=e0, [2]=r, [3]=v0
                _edge_val = ${_cparam(p.name, 0)} * np.float32(2.0) * ${p.name}_cfun_params[1] / (np.float32(1.0) + math.exp(${p.name}_cfun_params[2] * (${p.name}_cfun_params[3] - _edge_val)))
                % elif pre_ct == 'tanh':
                _edge_val = ${_cparam(p.name, 0)} * (np.float32(1.0) + math.tanh((${p.name}_cfun_params[1] * _edge_val - ${p.name}_cfun_params[2]) / ${p.name}_cfun_params[3]))
                % elif pre_ct == 'pre_sigmoidal':
                _edge_val = ${_cparam(p.name, 0)} * (${p.name}_cfun_params[1] + math.tanh(${p.name}_cfun_params[2] * (${p.name}_cfun_params[3] * _edge_val - ${p.name}_cfun_params[4])))
                % elif pre_ct == 'pre_sigmoidal_dynamic':
                # Dynamic: H*(Q + tanh(G*(P*x_j[0] - x_j[1])))
                # cfun_params: [0]=H, [1]=Q, [2]=G, [3]=P
                cv0 = ${p.name}_source_cvar[0]
                cv1 = ${p.name}_source_cvar[1]
                x0 = ${src}_srcbuf[tidy, cv0, src_node, _ms, slot]
                x1 = ${src}_srcbuf[tidy, cv1, src_node, _ms, slot]
                _edge_val = ${p.name}_cfun_params[0] * (${p.name}_cfun_params[1] + math.tanh(${p.name}_cfun_params[2] * (${p.name}_cfun_params[3] * x0 - x1)))
                % elif pre_ct == 'difference':
                xi_val = ${tgt}_state[tidy, ${p.name}_target_cvar[0], j, _ms]
                _edge_val = _edge_val - xi_val
                % elif pre_ct == 'kuramoto':
                xi_val = ${tgt}_state[tidy, ${p.name}_target_cvar[0], j, _ms]
                _edge_val = math.sin(_edge_val - xi_val)
                % endif
                ${p.name}_wsum_0[_ms] += w * _edge_val
                % endfor
            % elif cm == '1_to_many':
            for _ms in range(${p.n_src_modes}):
                _edge_val = ${src}_srcbuf[tidy, ${p.source_cvar[0]}, src_node, _ms, slot]
                % if pre_ct == 'sigmoidal_jr':
                # Classic: cmin + (cmax-cmin)/(1+exp(r*(midpoint-diff)))
                # cfun_params: [0]=a, [1]=cmin, [2]=cmax, [3]=r, [4]=midpoint
                cv0 = ${p.name}_source_cvar[0]
                cv1 = ${p.name}_source_cvar[1]
                x0 = ${src}_srcbuf[tidy, cv0, src_node, _ms, slot]
                x1 = ${src}_srcbuf[tidy, cv1, src_node, _ms, slot]
                _edge_val = ${p.name}_cfun_params[1] + (${p.name}_cfun_params[2] - ${p.name}_cfun_params[1]) / (np.float32(1.0) + math.exp(${p.name}_cfun_params[3] * (${p.name}_cfun_params[4] - (x0 - x1))))
                % elif pre_ct == 'sigmoidal_jr_legacy':
                # Legacy: a * 2*e0 / (1+exp(r*(v0-x)))
                # cfun_params: [0]=a, [1]=e0, [2]=r, [3]=v0
                _edge_val = ${_cparam(p.name, 0)} * np.float32(2.0) * ${p.name}_cfun_params[1] / (np.float32(1.0) + math.exp(${p.name}_cfun_params[2] * (${p.name}_cfun_params[3] - _edge_val)))
                % elif pre_ct == 'tanh':
                _edge_val = ${_cparam(p.name, 0)} * (np.float32(1.0) + math.tanh((${p.name}_cfun_params[1] * _edge_val - ${p.name}_cfun_params[2]) / ${p.name}_cfun_params[3]))
                % elif pre_ct == 'pre_sigmoidal':
                _edge_val = ${_cparam(p.name, 0)} * (${p.name}_cfun_params[1] + math.tanh(${p.name}_cfun_params[2] * (${p.name}_cfun_params[3] * _edge_val - ${p.name}_cfun_params[4])))
                % elif pre_ct == 'pre_sigmoidal_dynamic':
                # Dynamic: H*(Q + tanh(G*(P*x_j[0] - x_j[1])))
                # cfun_params: [0]=H, [1]=Q, [2]=G, [3]=P
                cv0 = ${p.name}_source_cvar[0]
                cv1 = ${p.name}_source_cvar[1]
                x0 = ${src}_srcbuf[tidy, cv0, src_node, _ms, slot]
                x1 = ${src}_srcbuf[tidy, cv1, src_node, _ms, slot]
                _edge_val = ${p.name}_cfun_params[0] * (${p.name}_cfun_params[1] + math.tanh(${p.name}_cfun_params[2] * (${p.name}_cfun_params[3] * x0 - x1)))
                % elif pre_ct == 'difference':
                xi_val = ${tgt}_state[tidy, ${p.name}_target_cvar[0], j, _ms]
                _edge_val = _edge_val - xi_val
                % elif pre_ct == 'kuramoto':
                xi_val = ${tgt}_state[tidy, ${p.name}_target_cvar[0], j, _ms]
                _edge_val = math.sin(_edge_val - xi_val)
                % endif
                % for ic in range(n_tgt_cvar):
                ${p.name}_wsum_${ic}[_ms] += w * _edge_val
                % endfor
            % endif

        ## pre-cfun already applied per-edge; now post-cfun then scale
        % if cm in ('1_to_1', 'n_to_n'):
        % for ic in range(n_tgt_cvar):
        for _ms in range(${p.n_src_modes}):
            % if post_ct == 'scaling':
            ${p.name}_wsum_${ic}[_ms] = ${_cparam(p.name, 0)} * ${p.name}_wsum_${ic}[_ms]
            % elif post_ct == 'linear':
            ${p.name}_wsum_${ic}[_ms] = ${_cparam(p.name, 0)} * ${p.name}_wsum_${ic}[_ms] + ${p.name}_cfun_params[1]
            % elif post_ct == 'sigmoidal':
            ${p.name}_wsum_${ic}[_ms] = ${p.name}_cfun_params[3] + (${p.name}_cfun_params[4] - ${p.name}_cfun_params[3]) / (np.float32(1.0) + math.exp(-${_cparam(p.name, 0)} * ((${p.name}_wsum_${ic}[_ms] - ${p.name}_cfun_params[2]) / ${p.name}_cfun_params[1])))
            % elif post_ct == 'kuramoto_norm':
            # a * (1/N) * wsum — sin is now applied per-edge in pre
            ${p.name}_wsum_${ic}[_ms] = ${_cparam(p.name, 0)} * ${p.name}_cfun_params[1] * ${p.name}_wsum_${ic}[_ms]
            % endif
            ${p.name}_wsum_${ic}[_ms] *= ${p.name}_scale
        ## target_scales and mode_map → accumulate into tgt_c
        % if p.is_inter:
        for _mt in range(${p.n_tgt_modes}):
            _contrib = np.float32(0.0)
            for _ms in range(${p.n_src_modes}):
                _contrib += ${p.name}_wsum_${ic}[_ms] * ${p.name}_mode_map[_ms, _mt]
            % if has_ts:
            ${tgt}_c[${ic}, j, _mt] += ${p.name}_target_scales[ic] * _contrib
            % else:
            ${tgt}_c[${ic}, j, _mt] += _contrib
            % endif
        % else:
        for _ms in range(${p.n_src_modes}):
            % if has_ts:
            ${tgt}_c[${ic}, j, _ms] += ${p.name}_target_scales[ic] * ${p.name}_wsum_${ic}[_ms]
            % else:
            ${tgt}_c[${ic}, j, _ms] += ${p.name}_wsum_${ic}[_ms]
            % endif
        % endif
        % endfor

        % elif cm == 'many_to_1':
        for _ms in range(${p.n_src_modes}):
            % if post_ct == 'scaling':
            ${p.name}_wsum_0[_ms] = ${_cparam(p.name, 0)} * ${p.name}_wsum_0[_ms]
            % elif post_ct == 'linear':
            ${p.name}_wsum_0[_ms] = ${_cparam(p.name, 0)} * ${p.name}_wsum_0[_ms] + ${p.name}_cfun_params[1]
            % elif post_ct == 'sigmoidal':
            ${p.name}_wsum_0[_ms] = ${p.name}_cfun_params[3] + (${p.name}_cfun_params[4] - ${p.name}_cfun_params[3]) / (np.float32(1.0) + math.exp(-${_cparam(p.name, 0)} * ((${p.name}_wsum_0[_ms] - ${p.name}_cfun_params[2]) / ${p.name}_cfun_params[1])))
            % elif post_ct == 'kuramoto_norm':
            # a * (1/N) * wsum — sin is now applied per-edge in pre
            ${p.name}_wsum_0[_ms] = ${_cparam(p.name, 0)} * ${p.name}_cfun_params[1] * ${p.name}_wsum_0[_ms]
            % endif
            ${p.name}_wsum_0[_ms] *= ${p.name}_scale
        % if p.is_inter:
        for _mt in range(${p.n_tgt_modes}):
            _contrib = np.float32(0.0)
            for _ms in range(${p.n_src_modes}):
                _contrib += ${p.name}_wsum_0[_ms] * ${p.name}_mode_map[_ms, _mt]
            % if has_ts:
            ${tgt}_c[0, j, _mt] += ${p.name}_target_scales[0] * _contrib
            % else:
            ${tgt}_c[0, j, _mt] += _contrib
            % endif
        % else:
        for _ms in range(${p.n_src_modes}):
            % if has_ts:
            ${tgt}_c[0, j, _ms] += ${p.name}_target_scales[0] * ${p.name}_wsum_0[_ms]
            % else:
            ${tgt}_c[0, j, _ms] += ${p.name}_wsum_0[_ms]
            % endif
        % endif

        % elif cm == '1_to_many':
        % for ic in range(n_tgt_cvar):
        for _ms in range(${p.n_src_modes}):
            % if post_ct == 'scaling':
            ${p.name}_wsum_${ic}[_ms] = ${_cparam(p.name, 0)} * ${p.name}_wsum_${ic}[_ms]
            % elif post_ct == 'linear':
            ${p.name}_wsum_${ic}[_ms] = ${_cparam(p.name, 0)} * ${p.name}_wsum_${ic}[_ms] + ${p.name}_cfun_params[1]
            % elif post_ct == 'sigmoidal':
            ${p.name}_wsum_${ic}[_ms] = ${p.name}_cfun_params[3] + (${p.name}_cfun_params[4] - ${p.name}_cfun_params[3]) / (np.float32(1.0) + math.exp(-${_cparam(p.name, 0)} * ((${p.name}_wsum_${ic}[_ms] - ${p.name}_cfun_params[2]) / ${p.name}_cfun_params[1])))
            % elif post_ct == 'kuramoto_norm':
            # a * (1/N) * wsum — sin is now applied per-edge in pre
            ${p.name}_wsum_${ic}[_ms] = ${_cparam(p.name, 0)} * ${p.name}_cfun_params[1] * ${p.name}_wsum_${ic}[_ms]
            % endif
            ${p.name}_wsum_${ic}[_ms] *= ${p.name}_scale
        % if p.is_inter:
        for _mt in range(${p.n_tgt_modes}):
            _contrib = np.float32(0.0)
            for _ms in range(${p.n_src_modes}):
                _contrib += ${p.name}_wsum_${ic}[_ms] * ${p.name}_mode_map[_ms, _mt]
            % if has_ts:
            ${tgt}_c[${ic}, j, _mt] += ${p.name}_target_scales[ic] * _contrib
            % else:
            ${tgt}_c[${ic}, j, _mt] += _contrib
            % endif
        % else:
        for _ms in range(${p.n_src_modes}):
            % if has_ts:
            ${tgt}_c[${ic}, j, _ms] += ${p.name}_target_scales[ic] * ${p.name}_wsum_${ic}[_ms]
            % else:
            ${tgt}_c[${ic}, j, _ms] += ${p.name}_wsum_${ic}[_ms]
            % endif
        % endif
        % endfor
        % endif

% endfor


##
## ---- dfun device functions (one per subnetwork) ----
##

% for sn in subnets:
<%
    svars = list(sn.model.state_variables)
    n_svars = len(svars)
    cterms = list(sn.model.coupling_terms)

    _has_custom_template = hasattr(sn.model, '_nb_hybrid_custom_template')
    _is_zerlaut = _has_custom_template and 'zerlaut' in sn.model._nb_hybrid_custom_template

    if _has_custom_template:
        cterms = ['Coupling_Term']
        gparams = {}
        sparams_list = []
        dfuns = {}
        _helpers = []
        _intermediates = []
    else:
        gparams = {name: float(getattr(sn.model, name)[0])
                   for name in sn.model.global_parameter_names}
        sparams_list = list(sn.model.spatial_parameter_names)
        dfuns = {}
        for sv in svars:
            expr = sn.model.state_variable_dfuns[sv]
            dfuns[sv] = expr
        _helpers = getattr(sn.model, 'dfun_helpers', None) or []
        _intermediates = getattr(sn.model, 'dfun_intermediates', None) or []

    n_nodes = sn.n_nodes
    n_modes = sn.n_modes
    boundaries = sn.model.state_variable_boundaries or {}
    lo_map = {}
    hi_map = {}
    for sv in svars:
        if sv in boundaries:
            b = boundaries[sv]
            try:
                lo_map[sv] = float(b[0]) if b[0] is not None else float('-inf')
            except (TypeError, ValueError):
                lo_map[sv] = float('-inf')
            try:
                hi_map[sv] = float(b[1]) if b[1] is not None else float('inf')
            except (TypeError, ValueError):
                hi_map[sv] = float('inf')

    from tvb.simulator.integrators import HeunDeterministic, EulerDeterministic, HeunStochastic, EulerStochastic
    is_heun = isinstance(sn.integrator, HeunDeterministic) or isinstance(sn.integrator, HeunStochastic)

    svars_str = ', '.join(svars)
    cterms_str = ', '.join(cterms)
    i1svars_str = ', '.join(['i1_' + s for s in svars])
    dt_val = float(sn.integrator.dt)

    # Combined dfun support
    _is_combined = getattr(sn.model, 'dfun_mode', None) == 'combined'
    if _is_combined:
        import re as _re, numpy as _np
        _sv_ct = set(svars) | set(cterms)
        _dm_names = list(getattr(sn.model, 'derived_matrix_names', []))
        _dm_name_set = set(_dm_names)
        _dm_data = {}
        for _dn in _dm_names:
            _da = getattr(sn.model, _dn, None)
            if _da is not None:
                _arr = _np.asarray(_da, dtype=_np.float32)
                if _arr.ndim > 1 and 1 in _arr.shape:
                    _arr = _arr.ravel()
                _dm_data[_dn] = _arr
        _dm_ops = list(getattr(sn.model, 'derived_matrix_ops', []))
        def _repl_m(mo):
            nm = mo.group(1)
            if nm in _sv_ct:
                return nm
            if nm in _dm_name_set:
                return f'_{sn.name}_{nm}[_m]'
            return f'_{nm}[_m]'
        _cdfuns = {sv: _re.sub(r'(\w+)_\{m\}', _repl_m, dfuns[sv]) for sv in svars}
    else:
        _cdfuns = None
        _dm_names = []
        _dm_data = {}
        _dm_ops = []
%>

% if _is_zerlaut:
## ---- Custom Zerlaut dfun for ${sn.name} ----
<%include file="nb-zerlaut-sweep-cuda.py.mako" args="sn=sn, debug_nojit=False, svars=svars, cterms=cterms, svars_str=svars_str, cterms_str=cterms_str, n_svars=n_svars, dt_val=dt_val, n_nodes=n_nodes, n_modes=n_modes, int_type='int32', lo_map=lo_map, hi_map=hi_map, i1svars_str=i1svars_str, is_heun=is_heun, is_combined=_is_combined, dm_names=_dm_names, dm_data=_dm_data, dm_ops=_dm_ops"/>
% else:
## ---- dfun helpers for ${sn.name} ----
% for _hname, _hargs, _hexpr in _helpers:

@cuda.jit(device=True)
def _dfun_helper_${sn.name}_${_hname}(${_hargs}):
    return ${_hexpr.replace('nb.float32(', 'np.float32(').replace('exp(', 'math.exp(').replace('sin(', 'math.sin(').replace('cos(', 'math.cos(').replace('log(', 'math.log(').replace('tanh(', 'math.tanh(')}

% endfor

% if _is_combined:
## Derived parameter arrays for combined-mode ${type(sn.model).__name__}
% for _dn, _da in _dm_data.items():
_${sn.name}_${_dn} = np.array(${repr(_da.tolist())}, dtype=np.float32)
% endfor

## ---- Combined dfun_${sn.name} ----
@cuda.jit(device=True)
def dfun_${sn.name}(${svars_str}, ${cterms_str}, _m, ${', '.join(['_' + sn.name + '_' + dn for dn in _dm_names])}, ${', '.join(['_' + op[0] for op in _dm_ops])}, _sp, ni, sweep_params, tid):
    pi = np.float32(3.141592653589793)
    % for name, val in gparams.items():
% if sn.name in sweep_model_dims and name in sweep_model_dims[sn.name]:
    ${name} = sweep_params[tid, ${sweep_model_dims[sn.name][name]}]
% else:
    ${name} = np.float32(${val})
% endif
    % endfor
    % if hasattr(sn.model, 'dfun_constants') and sn.model.dfun_constants:
% for _cname, _cval in sn.model.dfun_constants.items():
    ${_cname} = np.float32(${_cval})
% endfor
% endif
    % for _si, _sn_name in enumerate(sparams_list):
    ${_sn_name} = np.float32(_sp[${_si}, ni])
    % endfor
    % for _iname, _iexpr in _intermediates:
<%
    _rewritten = _iexpr
    for _hname, _, _ in _helpers:
        _rewritten = _rewritten.replace(_hname + '(', '_dfun_helper_' + sn.name + '_' + _hname + '(')
    _rewritten = _rewritten.replace('nb.float32(', 'np.float32(').replace('exp(', 'math.exp(').replace('sin(', 'math.sin(').replace('cos(', 'math.cos(').replace('log(', 'math.log(').replace('tanh(', 'math.tanh(')
%>
    ${_iname} = ${_rewritten}
    % endfor
    % for svar in svars:
<%
    _dexpr = _cdfuns[svar]
    for _hname, _, _ in _helpers:
        _dexpr = _dexpr.replace(_hname + '(', '_dfun_helper_' + sn.name + '_' + _hname + '(')
    _dexpr = _dexpr.replace('nb.float32(', 'np.float32(').replace('exp(', 'math.exp(').replace('sin(', 'math.sin(').replace('cos(', 'math.cos(').replace('log(', 'math.log(').replace('tanh(', 'math.tanh(')
%>
    d_${svar} = np.float32(${_dexpr})
    % endfor
    return (${', '.join(['d_' + s for s in svars])},)
% else:
## ---- dfun_${sn.name} ----
@cuda.jit(device=True)
def dfun_${sn.name}(${svars_str}, ${cterms_str}, _sp, ni, sweep_params, tid):
    pi = np.float32(3.141592653589793)
    % for name, val in gparams.items():
% if sn.name in sweep_model_dims and name in sweep_model_dims[sn.name]:
    ${name} = sweep_params[tid, ${sweep_model_dims[sn.name][name]}]
% else:
    ${name} = np.float32(${val})
% endif
    % endfor
    % if hasattr(sn.model, 'dfun_constants') and sn.model.dfun_constants:
% for _cname, _cval in sn.model.dfun_constants.items():
    ${_cname} = np.float32(${_cval})
% endfor
% endif
    % for _si, _sn_name in enumerate(sparams_list):
    ${_sn_name} = np.float32(_sp[${_si}, ni])
    % endfor
    % for _iname, _iexpr in _intermediates:
<%
    _rewritten = _iexpr
    for _hname, _, _ in _helpers:
        _rewritten = _rewritten.replace(_hname + '(', '_dfun_helper_' + sn.name + '_' + _hname + '(')
    _rewritten = _rewritten.replace('nb.float32(', 'np.float32(').replace('exp(', 'math.exp(').replace('sin(', 'math.sin(').replace('cos(', 'math.cos(').replace('log(', 'math.log(').replace('tanh(', 'math.tanh(')
%>
    ${_iname} = ${_rewritten}
    % endfor
    % for svar in svars:
<%
    _dexpr = dfuns[svar]
    for _hname, _, _ in _helpers:
        _dexpr = _dexpr.replace(_hname + '(', '_dfun_helper_' + sn.name + '_' + _hname + '(')
    _dexpr = _dexpr.replace('nb.float32(', 'np.float32(').replace('exp(', 'math.exp(').replace('sin(', 'math.sin(').replace('cos(', 'math.cos(').replace('log(', 'math.log(').replace('tanh(', 'math.tanh(')
%>
    d_${svar} = np.float32(${_dexpr})
    % endfor
    return (${', '.join(['d_' + s for s in svars])},)
% endif
% endif

% endfor


##
## ---- Main sweep kernel ----
##

@cuda.jit
def run_sweep(
    ## per-subnetwork state arrays  [tid, nvar, nnodes, n_modes]
    % for sn in subnets:
    ${sn.name}_state,      # (n_sweeps, ${sn.model.nvar}, ${sn.n_nodes}, ${sn.n_modes}) float32
    % endfor

    ## per-subnetwork source history buffers  [tid, nvar, nnodes, n_modes, horizon]
    % for sn in subnets:
    ${sn.name}_srcbuf,     # (n_sweeps, ${sn.model.nvar}, ${sn.n_nodes}, ${sn.n_modes}, horizon) float32
    % endfor

    ## per-subnetwork tavg output  [tid, nvoi, nnodes, n_modes]
    % for sn in subnets:
    ${sn.name}_tavg,       # (n_sweeps, ${len(sn.model.variables_of_interest)}, ${sn.n_nodes}, ${sn.n_modes}) float32
    % endfor

    ## per-subnetwork raw output  [tid, n_step_out, nvoi, nnodes, n_modes]
    % for sn in subnets:
    ${sn.name}_raw,        # (n_sweeps, nstep, ${len(sn.model.variables_of_interest)}, ${sn.n_nodes}, ${sn.n_modes}) float32
    % endfor

    ## per-subnetwork Bold Balloon model arrays
    % for sn in subnets:
    ${sn.name}_bold_state,  # (n_sweeps, ${len(sn.model.variables_of_interest)}, 4, ${sn.n_nodes}) float32
    ${sn.name}_bold_params, # (10,) float32
    ${sn.name}_bold_voi_idx, # (${len(sn.model.variables_of_interest)},) int32
    ${sn.name}_bold_out,    # (n_sweeps, n_bold_samples, ${len(sn.model.variables_of_interest)}, ${sn.n_nodes}) float32
    % endfor

    ## sweep parameter array
    sweep_params,          # (n_sweeps, n_sweep_dims) float32

    ## per-projection CSR arrays (read-only, shared)
    % for p in all_projs:
    ${p.name}_w_data,
    ${p.name}_w_indices,
    ${p.name}_w_indptr,
    ${p.name}_idelays,
    % if p.is_inter:
    ${p.name}_mode_map,   # (${p.n_src_modes}, ${p.n_tgt_modes}) float32
    % endif
    ${p.name}_source_cvar,
    ${p.name}_target_cvar, ${p.name}_target_cvar_cpl,
    ${p.name}_scale,
    ${p.name}_target_scales,
    ${p.name}_cfun_params,
    % endfor

    ## per-subnet coupling temporal average accumulators  [tid, ncvar, nnodes, n_modes]
    % for sn in subnets:
<%
    n_cvar_sn = len(sn.model.cvar)
%>
    ${sn.name}_ctavg,     # (n_sweeps, ${n_cvar_sn}, ${sn.n_nodes}, ${sn.n_modes}) float32
    % endfor

    ## per-subnet spatial average / projection monitor arrays
    % for sn in subnets:
    ${sn.name}_spatial_mean,  # (n_areas, ${sn.n_nodes}) float32 - spatial region mapping
    ${sn.name}_spatial_tavg,  # (n_sweeps, ${len(sn.model.variables_of_interest)}, n_areas, 1) float32
    ${sn.name}_gain,          # (n_sensors, ${sn.n_nodes}) float32 - sensor gain matrix
    ${sn.name}_proj_tavg,     # (n_sweeps, ${len(sn.model.variables_of_interest)}, n_sensors, 1) float32
    % endfor

    ## per-subnetwork spatial parameter arrays (may be empty)
    % for sn in subnets:
    ${sn.name}_sp,         # (n_spatial_params, ${sn.n_nodes}) float32
    % endfor

    ## per-subnetwork voi index arrays
    % for sn in subnets:
    ${sn.name}_voi_idx,    # (n_voi,) int32
    % endfor

    ## per-subnetwork noise arrays (stochastic subnets only)
    % for sn in subnets:
    % if isinstance(sn.integrator, (HeunStochastic, EulerStochastic)):
    ${sn.name}_noise,      # (n_sweeps, ${sn.model.nvar}, ${sn.n_nodes}, ${sn.n_modes}, nstep) float32
    % endif
    % endfor

    ## per-subnetwork stimulus arrays (stimulus subnets only)
    % for sn in subnets:
    % if sn.has_stimulus:
<%
    n_cvar_sn = len(sn.model.cvar)
%>
    ${sn.name}_stim,       # (n_sweeps, ${n_cvar_sn}, ${sn.n_nodes}, ${sn.n_modes}, nstep) float32
    % endif
    % endfor

    ## scalar parameters
    t_offset,
    horizon,
    dt,
    nstep,
    monitor_type,          # 0=tavg, 1=raw, 2=subsample
    monitor_period,        # steps between subsample writes
    % for sn in subnets:
    N_${sn.name},
    % endfor
    n_sweeps,
    bold_istep,            # int32, Bold sampling period (0 = disabled)
    n_bold_samples,        # int32, total Bold samples per sweep
):
    tid = cuda.grid(1)
    if tid >= n_sweeps:
        return

    dt_f = np.float32(dt)

    ## ---- Per-subnet coupling scratch (cuda.local.array) ----
    % for sn in subnets:
<%
    n_cvar_sn = len(sn.model.cvar)
%>
    % if n_cvar_sn > 0:
    ${sn.name}_c = cuda.local.array((${n_cvar_sn}, ${sn.n_nodes}, ${sn.n_modes}), dtype=numba.float32)
    % endif
    % endfor

    for t_local in range(1, nstep + 1):
        t = t_local + t_offset

        ## ---- 1. Zero coupling scratch ----
        % for sn in subnets:
<%
    n_cvar_sn = len(sn.model.cvar)
%>
        % if n_cvar_sn > 0:
        for _ic in range(${n_cvar_sn}):
            for _j in range(${sn.n_nodes}):
                for _m in range(${sn.n_modes}):
                    ${sn.name}_c[_ic, _j, _m] = np.float32(0.0)
        % endif
        % endfor

        ## ---- 2. Compute coupling for each projection ----
        % for p in all_projs:
<%
    src = p.source_subnet
    tgt = p.target_subnet
%>
        compute_coupling_${p.name}(
            ${src}_srcbuf,
            ${p.name}_w_data, ${p.name}_w_indices, ${p.name}_w_indptr,
            ${p.name}_idelays,
            horizon,
% if src != tgt:
            ${'N_' + src},
% endif
            ${'N_' + tgt},
            t, tid,
            ${p.name}_source_cvar, ${p.name}_target_cvar, ${p.name}_target_cvar_cpl,
            ${p.name}_scale, ${p.name}_target_scales,
            ${p.name}_cfun_params,
% if p.is_inter:
            ${p.name}_mode_map,
% endif
            ${tgt}_state,
            ${tgt}_c,
            sweep_params,
            tid,
        )
        % endfor

        ## ---- 2b. Inject stimulus into coupling scratch ----
        % for sn in subnets:
<%
    n_cvar_sn = len(sn.model.cvar)
%>
        % if sn.has_stimulus:
        for _ic in range(${n_cvar_sn}):
            for _j in range(${sn.n_nodes}):
                for _m in range(${sn.n_modes}):
                    ${sn.name}_c[_ic, _j, _m] += ${sn.name}_stim[tid, _ic, _j, _m, t - 1]
        % endif
        % endfor

        ## ---- 2c. Accumulate coupling temporal average ----
        % for sn in subnets:
<%
    n_cvar_sn = len(sn.model.cvar)
%>
        % if n_cvar_sn > 0:
        for _ci in range(${n_cvar_sn}):
            for _j in range(${sn.n_nodes}):
                % if sn.n_modes == 1:
                ${sn.name}_ctavg[tid, _ci, _j, 0] += ${sn.name}_c[_ci, _j, 0]
                % else:
                for _m in range(${sn.n_modes}):
                    ${sn.name}_ctavg[tid, _ci, _j, 0] += ${sn.name}_c[_ci, _j, _m]
                % endif
        % endif
        % endfor

        ## ---- 3. Integrate each subnetwork ----
        % for sn in subnets:
<%
    svars = list(sn.model.state_variables)
    n_svars = len(svars)
    cterms = list(sn.model.coupling_terms)
    n_cvar = len(cterms)
    n_nodes = sn.n_nodes
    n_modes = sn.n_modes
    boundaries = sn.model.state_variable_boundaries or {}
    lo_map = {}
    hi_map = {}
    for sv in svars:
        if sv in boundaries:
            b = boundaries[sv]
            try:
                lo_map[sv] = float(b[0]) if b[0] is not None else float('-inf')
            except (TypeError, ValueError):
                lo_map[sv] = float('-inf')
            try:
                hi_map[sv] = float(b[1]) if b[1] is not None else float('inf')
            except (TypeError, ValueError):
                hi_map[sv] = float('inf')
    sparams_list = []
    try:
        sparams_list = list(sn.model.spatial_parameter_names)
    except AttributeError:
        sparams_list = []

    from tvb.simulator.integrators import HeunDeterministic, EulerDeterministic, HeunStochastic, EulerStochastic
    is_heun = isinstance(sn.integrator, HeunDeterministic) or isinstance(sn.integrator, HeunStochastic)
    is_stochastic = isinstance(sn.integrator, (HeunStochastic, EulerStochastic))
    voi_names = list(sn.model.variables_of_interest)
    voi_idx_list = [svars.index(v) if v in svars else -1 for v in voi_names]

    _is_combined = getattr(sn.model, 'dfun_mode', None) == 'combined'
    if _is_combined:
        import re as _re, numpy as _np
        _sv_ct = set(svars) | set(cterms)
        _dm_names = list(getattr(sn.model, 'derived_matrix_names', []))
        _dm_name_set = set(_dm_names)
        _dm_data = {}
        for _dn in _dm_names:
            _da = getattr(sn.model, _dn, None)
            if _da is not None:
                _arr = _np.asarray(_da, dtype=_np.float32)
                if _arr.ndim > 1 and 1 in _arr.shape:
                    _arr = _arr.ravel()
                _dm_data[_dn] = _arr
        _dm_ops = list(getattr(sn.model, 'derived_matrix_ops', []))
    else:
        _dm_names = []
        _dm_data = {}
        _dm_ops = []
%>

        for i in range(N_${sn.name}):
            ## load spatial parameters (if any)
            % for _si, _sn_name in enumerate(sparams_list):
            ${_sn_name}_val = np.float32(${sn.name}_sp[${_si}, i]) if ${sn.name}_sp.shape[0] > 0 else np.float32(0.0)
            % endfor
            % for _si, _sn_name in enumerate(sparams_list):
            ${_sn_name} = ${_sn_name}_val
            % endfor

% if _is_combined:
            ## Combined dfun: compute cross-mode intermediates from current state
            % for _op_name, _op_mat, _op_svar in _dm_ops:
            _${_op_name} = cuda.local.array((${n_modes},), dtype=numba.float32)
            for _mi in range(${n_modes}):
                _${_op_name}[_mi] = np.float32(0.0)
                for _mk in range(${n_modes}):
                    _${_op_name}[_mi] += _${sn.name}_${_op_mat}[_mi, _mk] * ${sn.name}_state[tid, ${svars.index(_op_svar)}, i, _mk]
            % endfor

% if is_heun:
            ## ---- Heun two-pass for combined-dfun cross-mode coupling ----
            ## Pass 1: compute k1 for all modes, store derivatives + i1 states
            % for sv in svars:
            _k1_${sv} = cuda.local.array((${n_modes},), dtype=numba.float32)
            _i1_${sv} = cuda.local.array((${n_modes},), dtype=numba.float32)
            % endfor
            for m in range(${n_modes}):
                ## load state variables
                % for k, sv in enumerate(svars):
                ${sv} = ${sn.name}_state[tid, ${k}, i, m]
                % endfor

                ## load coupling terms
                % for k, ct in enumerate(cterms):
                ${ct} = ${sn.name}_c[${k}, i, m] if ${n_cvar} > 0 else np.float32(0.0)
                % endfor

                ## dfun at current state (k1)
                (d0_${', d0_'.join(svars)},) = dfun_${sn.name}(
                    ${', '.join(svars)}, ${', '.join(cterms)}, m,
                    ${', '.join(['_' + sn.name + '_' + dn for dn in _dm_names])},
                    ${', '.join(['_' + op[0] for op in _dm_ops])},
                    ${sn.name}_sp, i, sweep_params, tid
                )
                ## store per-mode derivatives
                % for sv in svars:
                _k1_${sv}[m] = d0_${sv}
                % endfor
                ## compute i1 intermediate for this mode
                % if is_stochastic:
                % for k2, sv in enumerate(svars):
                _i1_${sv}[m] = ${sv} + dt_f * d0_${sv} + ${sn.name}_noise[tid, ${k2}, i, m, t - 1]
                % endfor
                % else:
                % for sv in svars:
                _i1_${sv}[m] = ${sv} + dt_f * d0_${sv}
                % endfor
                % endif

            ## Recompute cross-mode intermediates from i1 arrays (correct per-mk indexing)
            % for _op_name, _op_mat, _op_svar in _dm_ops:
            _${_op_name}_i1 = cuda.local.array((${n_modes},), dtype=numba.float32)
            for _mi in range(${n_modes}):
                _${_op_name}_i1[_mi] = np.float32(0.0)
                for _mk in range(${n_modes}):
                    _${_op_name}_i1[_mi] += _${sn.name}_${_op_mat}[_mi, _mk] * _i1_${_op_svar}[_mk]
            % endfor

            ## Pass 2: compute k2 and update state for all modes
            for m in range(${n_modes}):
                ## load state and coupling for mode m
                % for k, sv in enumerate(svars):
                ${sv} = ${sn.name}_state[tid, ${k}, i, m]
                % endfor
                % for k, ct in enumerate(cterms):
                ${ct} = ${sn.name}_c[${k}, i, m] if ${n_cvar} > 0 else np.float32(0.0)
                % endfor

                ## load i1 state for mode m
                % for sv in svars:
                i1_${sv} = _i1_${sv}[m]
                % endfor

                ## dfun at intermediate (k2) using correct cross-mode i1 intermediates
                (d1_${', d1_'.join(svars)},) = dfun_${sn.name}(
                    ${', '.join(['i1_' + s for s in svars])}, ${', '.join(cterms)}, m,
                    ${', '.join(['_' + sn.name + '_' + dn for dn in _dm_names])},
                    ${', '.join(['_' + op[0] + '_i1' for op in _dm_ops])},
                    ${sn.name}_sp, i, sweep_params, tid
                )

                ## Heun update
                % if is_stochastic:
                % for k2, sv in enumerate(svars):
                n_${sv} = ${sv} + dt_f * np.float32(0.5) * (_k1_${sv}[m] + d1_${sv}) + ${sn.name}_noise[tid, ${k2}, i, m, t - 1]
                % endfor
                % else:
                % for sv in svars:
                n_${sv} = ${sv} + dt_f * np.float32(0.5) * (_k1_${sv}[m] + d1_${sv})
                % endfor
                % endif

                ## clamp final state
                % for sv in svars:
                % if sv in lo_map and lo_map[sv] != float('-inf'):
                if n_${sv} < np.float32(${lo_map[sv]}):
                    n_${sv} = np.float32(${lo_map[sv]})
                % endif
                % if sv in hi_map and hi_map[sv] != float('inf'):
                if n_${sv} > np.float32(${hi_map[sv]}):
                    n_${sv} = np.float32(${hi_map[sv]})
                % endif
                % endfor

                ## write back state
                % for k, sv in enumerate(svars):
                ${sn.name}_state[tid, ${k}, i, m] = n_${sv}
                % endfor

% else:
            ## ---- Euler for combined-dfun (single-mode loop, no cross-mode i1 needed) ----
            for m in range(${n_modes}):
                ## load state variables
                % for k, sv in enumerate(svars):
                ${sv} = ${sn.name}_state[tid, ${k}, i, m]
                % endfor

                ## load coupling terms
                % for k, ct in enumerate(cterms):
                ${ct} = ${sn.name}_c[${k}, i, m] if ${n_cvar} > 0 else np.float32(0.0)
                % endfor

                (d0_${', d0_'.join(svars)},) = dfun_${sn.name}(
                    ${', '.join(svars)}, ${', '.join(cterms)}, m,
                    ${', '.join(['_' + sn.name + '_' + dn for dn in _dm_names])},
                    ${', '.join(['_' + op[0] for op in _dm_ops])},
                    ${sn.name}_sp, i, sweep_params, tid
                )

                ## Euler update
                % if is_stochastic:
                % for k2, sv in enumerate(svars):
                n_${sv} = ${sv} + dt_f * d0_${sv} + ${sn.name}_noise[tid, ${k2}, i, m, t - 1]
                % endfor
                % else:
                % for sv in svars:
                n_${sv} = ${sv} + dt_f * d0_${sv}
                % endfor
                % endif

                ## clamp final state
                % for sv in svars:
                % if sv in lo_map and lo_map[sv] != float('-inf'):
                if n_${sv} < np.float32(${lo_map[sv]}):
                    n_${sv} = np.float32(${lo_map[sv]})
                % endif
                % if sv in hi_map and hi_map[sv] != float('inf'):
                if n_${sv} > np.float32(${hi_map[sv]}):
                    n_${sv} = np.float32(${hi_map[sv]})
                % endif
                % endfor

                ## write back state
                % for k, sv in enumerate(svars):
                ${sn.name}_state[tid, ${k}, i, m] = n_${sv}
                % endfor
% endif  ## is_heun vs euler for combined

% else:
            ## ---- Non-combined dfun: single-mode loop (no cross-mode intermediates) ----
            for m in range(${n_modes}):
                ## load state variables
                % for k, sv in enumerate(svars):
                ${sv} = ${sn.name}_state[tid, ${k}, i, m]
                % endfor

                ## load coupling terms
                % for k, ct in enumerate(cterms):
                ${ct} = ${sn.name}_c[${k}, i, m] if ${n_cvar} > 0 else np.float32(0.0)
                % endfor

                ## ---- dfun at current state (k1) ----
                (d0_${', d0_'.join(svars)},) = dfun_${sn.name}(${', '.join(svars)}, ${', '.join(cterms)}, ${sn.name}_sp, i, sweep_params, tid)

                ## ---- intermediate state (Heun) ----
                % if is_heun:
                % if is_stochastic:
                % for k2, sv in enumerate(svars):
                i1_${sv} = ${sv} + dt_f * d0_${sv} + ${sn.name}_noise[tid, ${k2}, i, m, t - 1]
                % endfor
                % else:
                % for sv in svars:
                i1_${sv} = ${sv} + dt_f * d0_${sv}
                % endfor
                % endif

                ## clamp intermediate
                % for sv in svars:
                % if sv in lo_map and lo_map[sv] != float('-inf'):
                if i1_${sv} < np.float32(${lo_map[sv]}):
                    i1_${sv} = np.float32(${lo_map[sv]})
                % endif
                % if sv in hi_map and hi_map[sv] != float('inf'):
                if i1_${sv} > np.float32(${hi_map[sv]}):
                    i1_${sv} = np.float32(${hi_map[sv]})
                % endif
                % endfor

                ## dfun at intermediate (k2)
                (d1_${', d1_'.join(svars)},) = dfun_${sn.name}(${', '.join(['i1_' + s for s in svars])}, ${', '.join(cterms)}, ${sn.name}_sp, i, sweep_params, tid)

                ## Heun update
                % if is_stochastic:
                % for k2, sv in enumerate(svars):
                n_${sv} = ${sv} + dt_f * np.float32(0.5) * (d0_${sv} + d1_${sv}) + ${sn.name}_noise[tid, ${k2}, i, m, t - 1]
                % endfor
                % else:
                % for sv in svars:
                n_${sv} = ${sv} + dt_f * np.float32(0.5) * (d0_${sv} + d1_${sv})
                % endfor
                % endif

                % else:
                ## ---- Euler update ----
                % if is_stochastic:
                % for k2, sv in enumerate(svars):
                n_${sv} = ${sv} + dt_f * d0_${sv} + ${sn.name}_noise[tid, ${k2}, i, m, t - 1]
                % endfor
                % else:
                % for sv in svars:
                n_${sv} = ${sv} + dt_f * d0_${sv}
                % endfor
                % endif
                % endif

                ## clamp final state
                % for sv in svars:
                % if sv in lo_map and lo_map[sv] != float('-inf'):
                if n_${sv} < np.float32(${lo_map[sv]}):
                    n_${sv} = np.float32(${lo_map[sv]})
                % endif
                % if sv in hi_map and hi_map[sv] != float('inf'):
                if n_${sv} > np.float32(${hi_map[sv]}):
                    n_${sv} = np.float32(${hi_map[sv]})
                % endif
                % endfor

                ## write back state
                % for k, sv in enumerate(svars):
                ${sn.name}_state[tid, ${k}, i, m] = n_${sv}
                % endfor
            ## end for m (non-combined)
% endif  ## _is_combined

            ## accumulate temporal average (sum all modes into mode-0)
            % for vi, voi_name in enumerate(voi_names):
<%
    voi_idx_val = voi_idx_list[vi]
    is_derived = voi_idx_val < 0
%>
% if not is_derived:
            _sv = np.float32(0.0)
            for _m in range(${n_modes}):
                _sv += ${sn.name}_state[tid, ${voi_idx_val}, i, _m]
            ${sn.name}_tavg[tid, ${vi}, i, 0] += _sv
% else:
<%
    import re
    expr = voi_name
    for si, sv in enumerate(svars):
        expr = re.sub(r'\b' + re.escape(sv) + r'\b', f'{sn.name}_state[tid, {si}, i, _m]', expr)
    expr = expr.replace('nb.float32(', 'np.float32(')
    expr = expr.replace('exp(', 'math.exp(')
    expr = expr.replace('sin(', 'math.sin(')
    expr = expr.replace('cos(', 'math.cos(')
    expr = expr.replace('log(', 'math.log(')
    expr = expr.replace('tanh(', 'math.tanh(')
%>
            _sv = np.float32(0.0)
            for _m in range(${n_modes}):
                _sv += ${expr}
            ${sn.name}_tavg[tid, ${vi}, i, 0] += _sv
% endif
            % endfor

            ## accumulate spatial average and projection monitors (sum all modes)
            % for vi, voi_name in enumerate(voi_names):
<%
    voi_idx_val = voi_idx_list[vi]
    is_derived = voi_idx_val < 0
%>
% if not is_derived:
            _sv = np.float32(0.0)
            for _m in range(${n_modes}):
                _sv += ${sn.name}_state[tid, ${voi_idx_val}, i, _m]
% else:
<%
    import re
    expr = voi_name
    for si, sv in enumerate(svars):
        expr = re.sub(r'\\b' + re.escape(sv) + r'\\b', f'{sn.name}_state[tid, {si}, i, _m]', expr)
    expr = expr.replace('nb.float32(', 'np.float32(')
    expr = expr.replace('exp(', 'math.exp(')
    expr = expr.replace('sin(', 'math.sin(')
    expr = expr.replace('cos(', 'math.cos(')
    expr = expr.replace('log(', 'math.log(')
    expr = expr.replace('tanh(', 'math.tanh(')
%>
            _sv = np.float32(0.0)
            for _m in range(${n_modes}):
                _sv += ${expr}
% endif
                for _ai in range(${sn.name}_spatial_mean.shape[0]):
                    ${sn.name}_spatial_tavg[tid, ${vi}, _ai, 0] += ${sn.name}_spatial_mean[_ai, i] * _sv
                for _si in range(${sn.name}_gain.shape[0]):
                    ${sn.name}_proj_tavg[tid, ${vi}, _si, 0] += ${sn.name}_gain[_si, i] * _sv
            % endfor

            ## monitor raw / subsample — store mode-0 (sum of all modes)
            if monitor_type == 1:
                % for vi, voi_name in enumerate(voi_names):
<%
    voi_idx_val = voi_idx_list[vi]
    is_derived = voi_idx_val < 0
%>
% if not is_derived:
                _sv = np.float32(0.0)
                for _m in range(${n_modes}):
                    _sv += ${sn.name}_state[tid, ${voi_idx_val}, i, _m]
                ${sn.name}_raw[tid, t - 1, ${vi}, i, 0] = _sv
% else:
<%
    import re
    expr = voi_name
    for si, sv in enumerate(svars):
        expr = re.sub(r'\b' + re.escape(sv) + r'\b', f'{sn.name}_state[tid, {si}, i, _m]', expr)
    expr = expr.replace('nb.float32(', 'np.float32(')
    expr = expr.replace('exp(', 'math.exp(')
    expr = expr.replace('sin(', 'math.sin(')
    expr = expr.replace('cos(', 'math.cos(')
    expr = expr.replace('log(', 'math.log(')
    expr = expr.replace('tanh(', 'math.tanh(')
%>
                _sv = np.float32(0.0)
                for _m in range(${n_modes}):
                    _sv += ${expr}
                ${sn.name}_raw[tid, t - 1, ${vi}, i, 0] = _sv
% endif
                % endfor
            elif monitor_type == 2:
                if (t - 1) % monitor_period == 0:
                    _raw_idx = (t - 1) // monitor_period
                    % for vi, voi_name in enumerate(voi_names):
<%
    voi_idx_val = voi_idx_list[vi]
    is_derived = voi_idx_val < 0
%>
% if not is_derived:
                    _sv = np.float32(0.0)
                    for _m in range(${n_modes}):
                        _sv += ${sn.name}_state[tid, ${voi_idx_val}, i, _m]
                    ${sn.name}_raw[tid, _raw_idx, ${vi}, i, 0] = _sv
% else:
<%
    import re
    expr = voi_name
    for si, sv in enumerate(svars):
        expr = re.sub(r'\b' + re.escape(sv) + r'\b', f'{sn.name}_state[tid, {si}, i, _m]', expr)
    expr = expr.replace('nb.float32(', 'np.float32(')
    expr = expr.replace('exp(', 'math.exp(')
    expr = expr.replace('sin(', 'math.sin(')
    expr = expr.replace('cos(', 'math.cos(')
    expr = expr.replace('log(', 'math.log(')
    expr = expr.replace('tanh(', 'math.tanh(')
%>
                    _sv = np.float32(0.0)
                    for _m in range(${n_modes}):
                        _sv += ${expr}
                    ${sn.name}_raw[tid, _raw_idx, ${vi}, i, 0] = _sv
% endif
                    % endfor

        % endfor  ## per-subnet integration

        ## ---- 3b. Bold Balloon model integration ----
        if bold_istep > 0:
            % for sn in subnets:
            for _bvi in range(${sn.name}_bold_voi_idx.shape[0]):
                _bsvi = ${sn.name}_bold_voi_idx[_bvi]
                for _bni in range(N_${sn.name}):
                    _bx = np.float32(0.0)
                    for _bmi in range(${sn.n_modes}):
                        _bx += ${sn.name}_state[tid, _bsvi, _bni, _bmi]
                    _bs = ${sn.name}_bold_state[tid, _bvi, 0, _bni]
                    _bf = ${sn.name}_bold_state[tid, _bvi, 1, _bni]
                    _bv = ${sn.name}_bold_state[tid, _bvi, 2, _bni]
                    _bq = ${sn.name}_bold_state[tid, _bvi, 3, _bni]
                    _bp = ${sn.name}_bold_params
                    _bds = _bx - _bp[0] * _bs - _bp[1] * (_bf - np.float32(1.0))
                    _bdf = _bs
                    _bdv = _bp[2] * (_bf - _bv ** _bp[3])
                    _bdq = _bp[2] * (_bf * (np.float32(1.0) - (np.float32(1.0) - _bp[4]) ** (np.float32(1.0) / _bf)) * _bp[5] - _bv ** _bp[3] * (_bq / _bv))
                    ${sn.name}_bold_state[tid, _bvi, 0, _bni] += dt_f * _bds
                    ${sn.name}_bold_state[tid, _bvi, 1, _bni] += dt_f * _bdf
                    ${sn.name}_bold_state[tid, _bvi, 2, _bni] += dt_f * _bdv
                    ${sn.name}_bold_state[tid, _bvi, 3, _bni] += dt_f * _bdq

            ## Bold sampling
            if (t - 1) % bold_istep == 0:
                _bold_idx = (t - 1) // bold_istep
                if _bold_idx < n_bold_samples:
                    for _bvi in range(${sn.name}_bold_voi_idx.shape[0]):
                        for _bni in range(N_${sn.name}):
                            _bv = ${sn.name}_bold_state[tid, _bvi, 2, _bni]
                            _bq = ${sn.name}_bold_state[tid, _bvi, 3, _bni]
                            _bp = ${sn.name}_bold_params
                            ${sn.name}_bold_out[tid, _bold_idx, _bvi, _bni] = _bp[9] * (_bp[6] * (np.float32(1.0) - _bq) + _bp[7] * (np.float32(1.0) - _bq / _bv) + _bp[8] * (np.float32(1.0) - _bv))
            % endfor

        ## ---- 4. Update source history buffers ----
        % for sn in subnets:
<%
    svar_list = list(sn.model.state_variables)
%>
        % if analysis.source_horizons.get(sn.name, 1) > 1:
        _slot_${sn.name} = t % horizon
        for i in range(N_${sn.name}):
            % for k, sv in enumerate(svar_list):
            for _m in range(${sn.n_modes}):
                ${sn.name}_srcbuf[tid, ${k}, i, _m, _slot_${sn.name}] = ${sn.name}_state[tid, ${k}, i, _m]
            % endfor
        % endif
        % endfor

    ## end for t in range(1, nstep + 1)

    ## tavg normalization is performed by the host backend after all chunks
    ## CONTRACT: All temporal average accumulators (tavg, ctavg, spatial_tavg, proj_tavg)
    ## are NOT normalized inside the kernel. The caller MUST divide by the total
    ## step count after the kernel returns. Handled by NbHybridCUDASweepBackend.run().
