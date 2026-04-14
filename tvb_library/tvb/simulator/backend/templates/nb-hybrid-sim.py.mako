## -*- coding: utf-8 -*-
##
## nb-hybrid-sim.py.mako
##
## Generates the full hybrid simulation kernel: per-projection coupling
## functions, per-subnetwork state-update functions, and the inner @nb.njit
## time-stepping loop plus a Python-level run_network() entry point.
##
## Required content dict keys:
##   analysis  — NetworkAnalysis instance (see nb_hybrid.py)
##   np        — numpy module reference
##   debug_nojit: bool — if True, skip @nb.njit decorators (for debugging)

import math
import numpy as np
import numba as nb
sin, cos, exp, log = math.sin, math.cos, math.exp, math.log

<%
from tvb.simulator.backend.nb_hybrid import NetworkAnalysis, _cfun_type, _cvar_mapping_mode
from tvb.simulator.integrators import HeunDeterministic, EulerDeterministic
subnets = analysis.subnetworks
inter_projs = analysis.inter_projections
intra_projs = analysis.intra_projections
all_projs = analysis.all_projections
source_horizons_map = analysis.source_horizons
%>

## ============================================================
## Coupling functions (one per projection)
## ============================================================

% for p in all_projs:
<%
    cm = _cvar_mapping_mode(p)
    ct = _cfun_type(p)
    nsrc_m = p.n_src_modes
    ntgt_m = p.n_tgt_modes
    is_inter = p.is_inter
    mono_src = (nsrc_m == 1)
    mono_tgt = (ntgt_m == 1)
    pre_ct = ct if ct in ('sigmoidal_jr', 'tanh', 'pre_sigmoidal') else 'none'
    post_ct = ct if ct not in ('sigmoidal_jr', 'tanh', 'pre_sigmoidal', 'none') else 'none'
%>

${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def compute_coupling_${p.name}(
    srcbuf,
    w_data, w_indices, w_indptr,
    idelays,
    % if is_inter:
    mode_map,
    % endif
    source_cvar,
    target_cvar,
    scale,
    target_scales,
    cfun_params,
    t,
    tgt,
):
    n_src_cvar = source_cvar.shape[0]
    n_tgt_cvar = target_cvar.shape[0]
    has_ts = target_scales.shape[0] > 0

    for j in range(${p.n_tgt_nodes}):
        row_start = w_indptr[j]
        row_end = w_indptr[j + 1]

        for ic in range(n_src_cvar):
            cv = source_cvar[ic]
            % if mono_src:
            ## n_modes == 1: scalar wsum, no inner mode loop
            wsum = nb.float32(0.0)
            for ptr in range(row_start, row_end):
                w = w_data[ptr]
                src_node = w_indices[ptr]
                buf_idx = (t - 1 - idelays[ptr] + ${source_horizons_map[p.source_subnet]}) % ${source_horizons_map[p.source_subnet]}
                wsum += w * srcbuf[cv, src_node, 0, buf_idx]
            ## pre-cfun: applied to summed weighted afferent (before scale)
            % if pre_ct == 'sigmoidal_jr':
            wsum = cfun_params[0] * nb.float32(2.0) * cfun_params[1] / (nb.float32(1.0) + exp(cfun_params[2] * (cfun_params[3] - wsum)))
            % elif pre_ct == 'tanh':
            wsum = cfun_params[0] * (nb.float32(1.0) + nb.float32(math.tanh((wsum - cfun_params[1]) / cfun_params[2])))
            % elif pre_ct == 'pre_sigmoidal':
            wsum = cfun_params[0] * (cfun_params[1] + nb.float32(math.tanh(cfun_params[2] * (cfun_params[3] * wsum - cfun_params[4]))))
            % endif
            wsum *= scale
            % if post_ct == "linear":
            wsum = cfun_params[0] * wsum + cfun_params[1]
            % elif post_ct == "scaling":
            wsum = cfun_params[0] * wsum
            % elif post_ct == "sigmoidal":
            wsum = cfun_params[3] + (cfun_params[4] - cfun_params[3]) / (nb.float32(1.0) + exp(-cfun_params[0] * ((wsum - cfun_params[2]) / cfun_params[1])))
            % elif post_ct == "kuramoto":
            wsum = cfun_params[0] * sin(wsum)
            % endif

            ## accumulate into target (mono_src)
            % if is_inter:
            % if mono_tgt:
            contrib = wsum * mode_map[0, 0]
            % if cm in ("1_to_1", "n_to_n"):
            ts = target_scales[ic] if has_ts else nb.float32(1.0)
            tgt[target_cvar[ic], j, 0] += ts * contrib
            % elif cm == "many_to_1":
            ts = target_scales[0] if has_ts else nb.float32(1.0)
            tgt[target_cvar[0], j, 0] += ts * contrib
            % elif cm == "1_to_many":
            for itc in range(n_tgt_cvar):
                ts = target_scales[itc] if has_ts else nb.float32(1.0)
                tgt[target_cvar[itc], j, 0] += ts * contrib
            % endif
            % else:
            for m_tgt in range(${ntgt_m}):
                contrib = np.float32(0.0)
                for m_src in range(${nsrc_m}):
                    contrib += wsum * mode_map[0, m_tgt]
                % if cm in ("1_to_1", "n_to_n"):
                ts = target_scales[ic] if has_ts else np.float32(1.0)
                tgt[target_cvar[ic], j, m_tgt] += ts * contrib
                % elif cm == "many_to_1":
                ts = target_scales[0] if has_ts else np.float32(1.0)
                tgt[target_cvar[0], j, m_tgt] += ts * contrib
                % elif cm == "1_to_many":
                for itc in range(n_tgt_cvar):
                    ts = target_scales[itc] if has_ts else np.float32(1.0)
                    tgt[target_cvar[itc], j, m_tgt] += ts * contrib
                % endif
            % endif
            % else:
            contrib = wsum
            % if cm in ("1_to_1", "n_to_n"):
            ts = target_scales[ic] if has_ts else nb.float32(1.0)
            tgt[target_cvar[ic], j, 0] += ts * contrib
            % elif cm == "many_to_1":
            ts = target_scales[0] if has_ts else nb.float32(1.0)
            tgt[target_cvar[0], j, 0] += ts * contrib
            % elif cm == "1_to_many":
            for itc in range(n_tgt_cvar):
                ts = target_scales[itc] if has_ts else nb.float32(1.0)
                tgt[target_cvar[itc], j, 0] += ts * contrib
            % endif
            % endif
            % else:
            ## general: array wsum with mode loop
            wsum = np.zeros(${nsrc_m}, dtype=np.float32)
            for ptr in range(row_start, row_end):
                w = w_data[ptr]
                src_node = w_indices[ptr]
                buf_idx = (t - 1 - idelays[ptr] + ${source_horizons_map[p.source_subnet]}) % ${source_horizons_map[p.source_subnet]}
                for m in range(${nsrc_m}):
                    wsum[m] += w * srcbuf[cv, src_node, m, buf_idx]
            ## pre-cfun: applied to summed weighted afferent (before scale)
            % if pre_ct == 'sigmoidal_jr':
            for m in range(${nsrc_m}):
                wsum[m] = cfun_params[0] * nb.float32(2.0) * cfun_params[1] / (nb.float32(1.0) + exp(cfun_params[2] * (cfun_params[3] - wsum[m])))
            % elif pre_ct == 'tanh':
            for m in range(${nsrc_m}):
                wsum[m] = cfun_params[0] * (nb.float32(1.0) + nb.float32(math.tanh((wsum[m] - cfun_params[1]) / cfun_params[2])))
            % elif pre_ct == 'pre_sigmoidal':
            for m in range(${nsrc_m}):
                wsum[m] = cfun_params[0] * (cfun_params[1] + nb.float32(math.tanh(cfun_params[2] * (cfun_params[3] * wsum[m] - cfun_params[4]))))
            % endif
            for m in range(${nsrc_m}):
                wsum[m] *= scale
            % if post_ct == "linear":
            for m in range(${nsrc_m}):
                wsum[m] = cfun_params[0] * wsum[m] + cfun_params[1]
            % elif post_ct == "scaling":
            for m in range(${nsrc_m}):
                wsum[m] = cfun_params[0] * wsum[m]
            % elif post_ct == "sigmoidal":
            for m in range(${nsrc_m}):
                wsum[m] = cfun_params[3] + (cfun_params[4] - cfun_params[3]) / (nb.float32(1.0) + exp(-cfun_params[0] * ((wsum[m] - cfun_params[2]) / cfun_params[1])))
            % elif post_ct == "kuramoto":
            for m in range(${nsrc_m}):
                wsum[m] = cfun_params[0] * sin(wsum[m])
            % endif

            ## accumulate into target (general)
            % if is_inter:
            for m_tgt in range(${ntgt_m}):
                contrib = np.float32(0.0)
                for m_src in range(${nsrc_m}):
                    contrib += wsum[m_src] * mode_map[m_src, m_tgt]
                % if cm in ("1_to_1", "n_to_n"):
                ts = target_scales[ic] if has_ts else np.float32(1.0)
                tgt[target_cvar[ic], j, m_tgt] += ts * contrib
                % elif cm == "many_to_1":
                ts = target_scales[0] if has_ts else np.float32(1.0)
                tgt[target_cvar[0], j, m_tgt] += ts * contrib
                % elif cm == "1_to_many":
                for itc in range(n_tgt_cvar):
                    ts = target_scales[itc] if has_ts else np.float32(1.0)
                    tgt[target_cvar[itc], j, m_tgt] += ts * contrib
                % endif
            % else:
            for m in range(${nsrc_m}):
                contrib = wsum[m]
                % if cm in ("1_to_1", "n_to_n"):
                ts = target_scales[ic] if has_ts else np.float32(1.0)
                tgt[target_cvar[ic], j, m] += ts * contrib
                % elif cm == "many_to_1":
                ts = target_scales[0] if has_ts else np.float32(1.0)
                tgt[target_cvar[0], j, m] += ts * contrib
                % elif cm == "1_to_many":
                for itc in range(n_tgt_cvar):
                    ts = target_scales[itc] if has_ts else np.float32(1.0)
                    tgt[target_cvar[itc], j, m] += ts * contrib
                % endif
            % endif
            % endif

% endfor

## ============================================================
## Per-subnetwork dfun + integrator
## ============================================================

% for sn in subnets:
<%
    from tvb.simulator.integrators import HeunDeterministic, EulerDeterministic, HeunStochastic, EulerStochastic
    int_type = "heun_stochastic" if isinstance(sn.integrator, HeunStochastic) else \
               "euler_stochastic" if isinstance(sn.integrator, EulerStochastic) else \
               "heun" if isinstance(sn.integrator, HeunDeterministic) else "euler"
    svars = list(sn.model.state_variables)

    # Zerlaut models use a custom template for dfun generation;
    # they don't have coupling_terms / state_variable_dfuns / global_parameter_names.
    _has_custom_template = hasattr(sn.model, '_nb_hybrid_custom_template')
    if _has_custom_template:
        cterms = ['Coupling_Term']  # Zerlaut: single cvar=[0]
        dfuns = None
        gparams = {}
    else:
        cterms = list(sn.model.coupling_terms)
        dfuns = sn.model.state_variable_dfuns
        gparams = {n: float(getattr(sn.model, n)[0]) for n in sn.model.global_parameter_names}

    dt_val = sn.integrator.dt
    n_nodes = sn.n_nodes
    n_modes = sn.n_modes
    svb = sn.model.state_variable_boundaries
    lo_map = {k: float(v[0]) if v[0] != float('-inf') and not (v[0] != v[0]) else None for k, v in svb.items()} if svb else {}
    hi_map = {k: float(v[1]) if v[1] != float('inf') and not (v[1] != v[1]) else None for k, v in svb.items()} if svb else {}
    import math as _math, numpy as _np
    lo_map = {}
    hi_map = {}
    if svb:
        import numpy as _np
        for k, v in svb.items():
            lo_map[k] = float(v[0]) if _np.isfinite(v[0]) else None
            hi_map[k] = float(v[1]) if _np.isfinite(v[1]) else None
    svars_str = ', '.join(svars)
    cterms_str = ', '.join(cterms)
    i1svars_str = ', '.join(['i1' + s for s in svars])
    n_svars = len(svars)
    _intermediates = list(getattr(sn.model, 'dfun_intermediates', None) or [])
    _is_combined = getattr(sn.model, 'dfun_mode', None) == 'combined'
    if _is_combined:
        import re as _re
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

% if _has_custom_template:
## ---- Custom dfun generation for ${type(sn.model).__name__} ----
<%include file="${sn.model._nb_hybrid_custom_template}" args="sn=sn, debug_nojit=debug_nojit, svars=svars, cterms=cterms, svars_str=svars_str, cterms_str=cterms_str, n_svars=n_svars, dt_val=dt_val, n_nodes=n_nodes, n_modes=n_modes, int_type=int_type, lo_map=lo_map, hi_map=hi_map, i1svars_str=i1svars_str"/>
% else:
% if _is_combined:
## Derived parameter arrays for combined-mode ${type(sn.model).__name__}
% for _dn, _da in _dm_data.items():
_${sn.name}_${_dn} = np.array(${repr(_da.tolist())}, dtype=np.float32)
% endfor

${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def dfun_${sn.name}(${svars_str}, ${cterms_str}, _m, ${', '.join(['_' + sn.name + '_' + dn for dn in _dm_names])}, ${', '.join(['_' + op[0] for op in _dm_ops])}):
    pi = math.pi
    % for name, val in gparams.items():
    ${name} = nb.float32(${val})
    % endfor
    % for svar in svars:
    d_${svar} = nb.float32(${_cdfuns[svar]})
    % endfor
    return (${', '.join(['d_' + s for s in svars])},)

% else:
## emit physical constants if model provides them (e.g. KIonEx-style)
% if hasattr(sn.model, 'dfun_constants') and sn.model.dfun_constants:
% for _cname, _cval in sn.model.dfun_constants.items():
${_cname} = ${_cval}
% endfor

% endif
## emit helper functions if model provides them (e.g. KIonEx-style)
% if hasattr(sn.model, 'dfun_helpers') and sn.model.dfun_helpers:
% for _fname, _fargs, _fexpr in sn.model.dfun_helpers:
${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def ${_fname}(${_fargs}):
    return ${_fexpr}

% endfor
% endif
${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def dfun_${sn.name}(${svars_str}, ${cterms_str}):
    pi = math.pi
    % for name, val in gparams.items():
    ${name} = nb.float32(${val})
    % endfor
    ## emit shared intermediate computations if model provides them (e.g. KIonEx-style)
    % for _iname, _iexpr in _intermediates:
    ${_iname} = ${_iexpr}
    % endfor
    % for svar in svars:
    d_${svar} = nb.float32(${dfuns[svar]})
    % endfor
    return (${', '.join(['d_' + s for s in svars])},)

% endif  ## _is_combined
% endif  ## end _has_custom_template else

% if _is_combined:
${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def integrate_${sn.name}(state, coupling${',' if sn.is_stochastic else ''} ${'noise, t_abs' if sn.is_stochastic else ''}):
    """Integrate subnetwork ${sn.name} one step in-place (combined-mode)."""
    dt = nb.float32(${dt_val})

    for i in range(${n_nodes}):
        ## Compute cross-mode intermediates from current state
        % for _op_name, _op_mat, _op_svar in _dm_ops:
        _${_op_name} = np.zeros(${n_modes}, dtype=np.float32)
        for _mi in range(${n_modes}):
            for _mk in range(${n_modes}):
                _${_op_name}[_mi] += _${sn.name}_${_op_mat}[_mi, _mk] * state[${svars.index(_op_svar)}, i, _mk]
        % endfor

        % if int_type in ("euler", "euler_stochastic"):
        ## Euler single-pass
        for m in range(${n_modes}):
            % for k, svar in enumerate(svars):
            ${svar} = state[${k}, i, m]
            % endfor
            % for k, ct in enumerate(cterms):
            ${ct} = coupling[${k}, i, m]
            % endfor
            (${', '.join(['d0_' + s for s in svars])},) = dfun_${sn.name}(
                ${svars_str}, ${cterms_str}, m,
                ${', '.join(['_' + sn.name + '_' + dn for dn in _dm_names])},
                ${', '.join(['_' + op[0] for op in _dm_ops])}
            )
            % for svar in svars:
            n${svar} = ${svar} + dt * d0_${svar}
            % endfor
            % if int_type == "euler_stochastic":
            % for k2, svar in enumerate(svars):
            n${svar} = n${svar} + noise[${k2}, i, m, t_abs]
            % endfor
            % endif
            % for k, svar in enumerate(svars):
            <%
                lo = lo_map.get(svar)
                hi = hi_map.get(svar)
            %>
            % if lo is not None:
            if n${svar} < nb.float32(${lo}):
                n${svar} = nb.float32(${lo})
            % endif
            % if hi is not None:
            if n${svar} > nb.float32(${hi}):
                n${svar} = nb.float32(${hi})
            % endif
            % endfor
            % for k, svar in enumerate(svars):
            state[${k}, i, m] = n${svar}
            % endfor

        % elif int_type in ("heun", "heun_stochastic"):
        ## Heun two-pass for cross-mode coupling
        ## Pass 1: compute k1 for all modes
        % for svar in svars:
        _k1_${svar} = np.zeros(${n_modes}, dtype=np.float32)
        % endfor
        for m in range(${n_modes}):
            % for k, svar in enumerate(svars):
            ${svar} = state[${k}, i, m]
            % endfor
            % for k, ct in enumerate(cterms):
            ${ct} = coupling[${k}, i, m]
            % endfor
            (${', '.join(['d0_' + s for s in svars])},) = dfun_${sn.name}(
                ${svars_str}, ${cterms_str}, m,
                ${', '.join(['_' + sn.name + '_' + dn for dn in _dm_names])},
                ${', '.join(['_' + op[0] for op in _dm_ops])}
            )
            % for svar in svars:
            _k1_${svar}[m] = d0_${svar}
            % endfor

        ## Compute i1 intermediate states for all modes
        % for svar in svars:
        _i1_${svar} = np.zeros(${n_modes}, dtype=np.float32)
        % endfor
        for m in range(${n_modes}):
            % for k, svar in enumerate(svars):
            _i1_${svar}[m] = state[${k}, i, m] + dt * _k1_${svar}[m]
            % endfor
            % if int_type == "heun_stochastic":
            % for k2, svar in enumerate(svars):
            _i1_${svar}[m] = _i1_${svar}[m] + noise[${k2}, i, m, t_abs]
            % endfor
            % endif

        ## Recompute cross-mode intermediates from i1 states
        % for _op_name, _op_mat, _op_svar in _dm_ops:
        _${_op_name}_i1 = np.zeros(${n_modes}, dtype=np.float32)
        for _mi in range(${n_modes}):
            for _mk in range(${n_modes}):
                _${_op_name}_i1[_mi] += _${sn.name}_${_op_mat}[_mi, _mk] * _i1_${_op_svar}[_mk]
        % endfor

        ## Pass 2: compute k2 and update state
        for m in range(${n_modes}):
            % for k, ct in enumerate(cterms):
            ${ct} = coupling[${k}, i, m]
            % endfor
            % for k, svar in enumerate(svars):
            ${svar} = state[${k}, i, m]
            i1${svar} = _i1_${svar}[m]
            % endfor
            (${', '.join(['d1_' + s for s in svars])},) = dfun_${sn.name}(
                ${i1svars_str}, ${cterms_str}, m,
                ${', '.join(['_' + sn.name + '_' + dn for dn in _dm_names])},
                ${', '.join(['_' + op[0] + '_i1' for op in _dm_ops])}
            )
            % for svar in svars:
            n${svar} = ${svar} + dt * nb.float32(0.5) * (_k1_${svar}[m] + d1_${svar})
            % endfor
            % if int_type == "heun_stochastic":
            % for k2, svar in enumerate(svars):
            n${svar} = n${svar} + noise[${k2}, i, m, t_abs]
            % endfor
            % endif
            % for k, svar in enumerate(svars):
            <%
                lo = lo_map.get(svar)
                hi = hi_map.get(svar)
            %>
            % if lo is not None:
            if n${svar} < nb.float32(${lo}):
                n${svar} = nb.float32(${lo})
            % endif
            % if hi is not None:
            if n${svar} > nb.float32(${hi}):
                n${svar} = nb.float32(${hi})
            % endif
            % endfor
            % for k, svar in enumerate(svars):
            state[${k}, i, m] = n${svar}
            % endfor
        % endif  ## euler vs heun

% else:
${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def integrate_${sn.name}(state, coupling${',' if sn.is_stochastic else ''} ${'noise, t_abs' if sn.is_stochastic else ''}):
    """Integrate subnetwork ${sn.name} one step in-place."""
    dt = nb.float32(${dt_val})

    for i in range(${n_nodes}):
    % if n_modes == 1:
        ## n_modes == 1: mode loop elided, index hardcoded to 0
        % for k, svar in enumerate(svars):
        ${svar} = state[${k}, i, 0]
        % endfor
        % for k, ct in enumerate(cterms):
        ${ct} = coupling[${k}, i, 0]
        % endfor

        (${', '.join(['d0_' + s for s in svars])},) = dfun_${sn.name}(${svars_str}, ${cterms_str})

        % if int_type == "euler":
        % for svar in svars:
        n${svar} = ${svar} + dt * d0_${svar}
        % endfor
        % elif int_type == "euler_stochastic":
        % for k2, svar in enumerate(svars):
        n${svar} = ${svar} + dt * d0_${svar} + noise[${k2}, i, 0, t_abs]
        % endfor
        % elif int_type == "heun":
        % for svar in svars:
        i1${svar} = ${svar} + dt * d0_${svar}
        % endfor
        (${', '.join(['d1_' + s for s in svars])},) = dfun_${sn.name}(${i1svars_str}, ${cterms_str})
        % for svar in svars:
        n${svar} = ${svar} + dt * nb.float32(0.5) * (d0_${svar} + d1_${svar})
        % endfor
        % elif int_type == "heun_stochastic":
        % for k2, svar in enumerate(svars):
        i1${svar} = ${svar} + dt * d0_${svar} + noise[${k2}, i, 0, t_abs]
        % endfor
        (${', '.join(['d1_' + s for s in svars])},) = dfun_${sn.name}(${i1svars_str}, ${cterms_str})
        % for k2, svar in enumerate(svars):
        n${svar} = ${svar} + dt * nb.float32(0.5) * (d0_${svar} + d1_${svar}) + noise[${k2}, i, 0, t_abs]
        % endfor
        % endif

        % for k, svar in enumerate(svars):
        <%
            lo = lo_map.get(svar)
            hi = hi_map.get(svar)
        %>
        % if lo is not None:
        if n${svar} < nb.float32(${lo}):
            n${svar} = nb.float32(${lo})
        % endif
        % if hi is not None:
        if n${svar} > nb.float32(${hi}):
            n${svar} = nb.float32(${hi})
        % endif
        % endfor

        % for k, svar in enumerate(svars):
        state[${k}, i, 0] = n${svar}
        % endfor
    % else:
        for m in range(${n_modes}):
            % for k, svar in enumerate(svars):
            ${svar} = state[${k}, i, m]
            % endfor
            % for k, ct in enumerate(cterms):
            ${ct} = coupling[${k}, i, m]
            % endfor

            (${', '.join(['d0_' + s for s in svars])},) = dfun_${sn.name}(${svars_str}, ${cterms_str})

            % if int_type == "euler":
            % for svar in svars:
            n${svar} = ${svar} + dt * d0_${svar}
            % endfor
            % elif int_type == "euler_stochastic":
            % for k2, svar in enumerate(svars):
            n${svar} = ${svar} + dt * d0_${svar} + noise[${k2}, i, m, t_abs]
            % endfor
            % elif int_type == "heun":
            % for svar in svars:
            i1${svar} = ${svar} + dt * d0_${svar}
            % endfor
            (${', '.join(['d1_' + s for s in svars])},) = dfun_${sn.name}(${i1svars_str}, ${cterms_str})
            % for svar in svars:
            n${svar} = ${svar} + dt * nb.float32(0.5) * (d0_${svar} + d1_${svar})
            % endfor
            % elif int_type == "heun_stochastic":
            % for k2, svar in enumerate(svars):
            i1${svar} = ${svar} + dt * d0_${svar} + noise[${k2}, i, m, t_abs]
            % endfor
            (${', '.join(['d1_' + s for s in svars])},) = dfun_${sn.name}(${i1svars_str}, ${cterms_str})
            % for k2, svar in enumerate(svars):
            n${svar} = ${svar} + dt * nb.float32(0.5) * (d0_${svar} + d1_${svar}) + noise[${k2}, i, m, t_abs]
            % endfor
            % endif

            ## boundary conditions
            % for k, svar in enumerate(svars):
            <%
                lo = lo_map.get(svar)
                hi = hi_map.get(svar)
            %>
            % if lo is not None:
            if n${svar} < nb.float32(${lo}):
                n${svar} = nb.float32(${lo})
            % endif
            % if hi is not None:
            if n${svar} > nb.float32(${hi}):
                n${svar} = nb.float32(${hi})
            % endif
            % endfor

            % for k, svar in enumerate(svars):
            state[${k}, i, m] = n${svar}
            % endfor
    % endif

% endif  ## _is_combined integration

% endfor

## ============================================================
## Inner @njit time-stepping loop
## ============================================================

${'' if debug_nojit else '@nb.njit(cache=True)'}
def network_chunk(
    nstep,
    t_start,
    ## per-subnetwork state arrays
    % for sn in subnets:
    ${sn.name}_state,   # (n_svars, n_nodes, n_modes) float32 — updated in-place
    % endfor
    ## per-subnet shared source history buffers
    % for sn in subnets:
    ${sn.name}_srcbuf,  # (n_vars, n_nodes, n_modes, max_horizon) float32
    % endfor
    ## per-projection arrays
    % for p in all_projs:
    ${p.name}_w_data, ${p.name}_w_indices, ${p.name}_w_indptr,
    ${p.name}_idelays,
    % if p.is_inter:
    ${p.name}_mode_map,
    % endif
    ${p.name}_source_cvar, ${p.name}_target_cvar,
    ${p.name}_scale, ${p.name}_target_scales,
    ${p.name}_cfun_params,
    % endfor
    ## temporal-average accumulators
    % for sn in subnets:
    ${sn.name}_tavg,  # (n_voi, n_nodes, n_modes) — updated in-place
    % endfor
    tavg_count,  # (1,) int32 — updated in-place
    ## coupling temporal-average accumulators (AfferentCoupling monitor)
    % for sn in subnets:
    ${sn.name}_ctavg,  # (n_cvar, n_nodes, n_modes) — updated in-place
    % endfor
    ## per-subnetwork noise arrays (stochastic only)
    % for sn in subnets:
    % if sn.is_stochastic:
    ${sn.name}_noise,  # (n_vars, n_nodes, n_modes, nstep_total) float32
    % endif
    % endfor
    ## per-subnetwork stimulus arrays (stimulus subnetworks only)
    % for sn in subnets:
    % if sn.has_stimulus:
    ${sn.name}_stim,  # (n_cvar, n_nodes, n_modes, nstep_total) float32
    % endif
    % endfor
):
<%
    all_sn_names = [sn.name for sn in subnets]
    all_sn_ncvars = [len(sn.model.coupling_terms) for sn in subnets]
    all_sn_nnodes = [sn.n_nodes for sn in subnets]
    all_sn_nmodes = [sn.n_modes for sn in subnets]
    all_sn_nvoi = [len(sn.model.variables_of_interest) for sn in subnets]
    sn_ncvar_dict = {sn.name: len(sn.model.coupling_terms) for sn in subnets}
    sn_nnodes_dict = {sn.name: sn.n_nodes for sn in subnets}
    sn_nmodes_dict = {sn.name: sn.n_modes for sn in subnets}
    sn_nvoi_dict = {sn.name: len(sn.model.variables_of_interest) for sn in subnets}
    sn_voi_dict = {sn.name: list(sn.model.variables_of_interest)
                   for sn in subnets}
    sn_svars_dict = {sn.name: list(sn.model.state_variables) for sn in subnets}
%>
    for t_local in range(nstep):
        t = t_start + t_local

        ## zero coupling arrays
        % for sn in subnets:
        ${sn.name}_c = np.zeros((${sn_ncvar_dict[sn.name]}, ${sn_nnodes_dict[sn.name]}, ${sn_nmodes_dict[sn.name]}), dtype=np.float32)
        % endfor

        ## inter-projection coupling
        % for p in inter_projs:
        compute_coupling_${p.name}(
            ${p.source_subnet}_srcbuf,
            ${p.name}_w_data, ${p.name}_w_indices, ${p.name}_w_indptr,
            ${p.name}_idelays,
            ${p.name}_mode_map,
            ${p.name}_source_cvar, ${p.name}_target_cvar,
            ${p.name}_scale, ${p.name}_target_scales,
            ${p.name}_cfun_params,
            t,
            ${p.target_subnet}_c,
        )
        % endfor

        ## intra-projection coupling
        % for p in intra_projs:
        compute_coupling_${p.name}(
            ${p.source_subnet}_srcbuf,
            ${p.name}_w_data, ${p.name}_w_indices, ${p.name}_w_indptr,
            ${p.name}_idelays,
            ${p.name}_source_cvar, ${p.name}_target_cvar,
            ${p.name}_scale, ${p.name}_target_scales,
            ${p.name}_cfun_params,
            t,
            ${p.target_subnet}_c,
        )
        % endfor

        ## add pre-computed stimulus contributions
        % for sn in subnets:
        % if sn.has_stimulus:
        % if sn_nmodes_dict[sn.name] == 1:
        for _sc in range(${sn_ncvar_dict[sn.name]}):
            for _sn in range(${sn_nnodes_dict[sn.name]}):
                ${sn.name}_c[_sc, _sn, 0] += ${sn.name}_stim[_sc, _sn, 0, t - 1]
        % else:
        for _sc in range(${sn_ncvar_dict[sn.name]}):
            for _sn in range(${sn_nnodes_dict[sn.name]}):
                for _sm in range(${sn_nmodes_dict[sn.name]}):
                    ${sn.name}_c[_sc, _sn, _sm] += ${sn.name}_stim[_sc, _sn, _sm, t - 1]
        % endif
        % endif
        % endfor

        ## accumulate coupling temporal average (AfferentCoupling monitor)
        % for sn in subnets:
        % if sn_nmodes_dict[sn.name] == 1:
        for ci in range(${sn_ncvar_dict[sn.name]}):
            for ni in range(${sn_nnodes_dict[sn.name]}):
                ${sn.name}_ctavg[ci, ni, 0] += ${sn.name}_c[ci, ni, 0]
        % else:
        for ci in range(${sn_ncvar_dict[sn.name]}):
            for ni in range(${sn_nnodes_dict[sn.name]}):
                for mi in range(${sn_nmodes_dict[sn.name]}):
                    ${sn.name}_ctavg[ci, ni, mi] += ${sn.name}_c[ci, ni, mi]
        % endif
        % endfor

        ## integrate each subnetwork in-place
        % for sn in subnets:
        integrate_${sn.name}(${sn.name}_state, ${sn.name}_c${',' if sn.is_stochastic else ''} ${'%s_noise, t - 1' % sn.name if sn.is_stochastic else ''})
        % endfor

        ## update shared source buffers (one write per source subnet)
        % for sn in subnets:
        ${sn.name}_srcbuf[:, :, :, t % ${source_horizons_map[sn.name]}] = ${sn.name}_state
        % endfor

        ## accumulate temporal average
        % for sn in subnets:
        <%
            voi_names = sn_voi_dict[sn.name]
            svars_list = sn_svars_dict[sn.name]
        %>
        % if sn_nmodes_dict[sn.name] == 1:
        for vi in range(${sn_nvoi_dict[sn.name]}):
            for ni in range(${sn_nnodes_dict[sn.name]}):
                ${sn.name}_tavg[vi, ni, 0] += ${sn.name}_state[vi, ni, 0]
        % else:
        for vi in range(${sn_nvoi_dict[sn.name]}):
            for ni in range(${sn_nnodes_dict[sn.name]}):
                for mi in range(${sn_nmodes_dict[sn.name]}):
                    ${sn.name}_tavg[vi, ni, mi] += ${sn.name}_state[vi, ni, mi]
        % endif
        % endfor
        tavg_count[0] += 1


def run_network(
    nstep,
    % for sn in subnets:
    ${sn.name}_state,
    % endfor
    % for sn in subnets:
    ${sn.name}_srcbuf,
    % endfor
    % for p in all_projs:
    ${p.name}_w_data, ${p.name}_w_indices, ${p.name}_w_indptr,
    ${p.name}_idelays,
    % if p.is_inter:
    ${p.name}_mode_map,
    % endif
    ${p.name}_source_cvar, ${p.name}_target_cvar,
    ${p.name}_scale, ${p.name}_target_scales,
    ${p.name}_cfun_params,
    % endfor
    ## noise arrays (stochastic subnetworks only)
    % for sn in subnets:
    % if sn.is_stochastic:
    ${sn.name}_noise,
    % endif
    % endfor
    ## stimulus arrays (stimulus subnetworks only)
    % for sn in subnets:
    % if sn.has_stimulus:
    ${sn.name}_stim,
    % endif
    % endfor
    chunk_size,
):
    """Outer Python loop that calls the @njit kernel in chunks and collects output."""
    <%
        sn_nvoi_list = [(sn.name, len(sn.model.variables_of_interest), sn.n_nodes, sn.n_modes) for sn in subnets]
    %>
    ## allocate temporal average accumulators
    % for sn in subnets:
    ${sn.name}_tavg = np.zeros((${sn_nvoi_dict[sn.name]}, ${sn_nnodes_dict[sn.name]}, ${sn_nmodes_dict[sn.name]}), dtype=np.float32)
    % endfor
    tavg_count = np.zeros(1, dtype=np.int32)
    ## allocate coupling temporal average accumulators (AfferentCoupling monitor)
    % for sn in subnets:
    ${sn.name}_ctavg = np.zeros((${sn_ncvar_dict[sn.name]}, ${sn_nnodes_dict[sn.name]}, ${sn_nmodes_dict[sn.name]}), dtype=np.float32)
    % endfor

    ## storage for raw outputs per subnetwork
    % for sn in subnets:
    ${sn.name}_outputs = []
    ${sn.name}_ctavg_outputs = []
    ${sn.name}_times   = []
    % endfor
    time_step = np.float32(${subnets[0].integrator.dt})

    t_global = 1
    while t_global <= nstep:
        this_chunk = min(chunk_size, nstep - t_global + 1)

        ## reset accumulators
        % for sn in subnets:
        ${sn.name}_tavg[:] = np.float32(0.0)
        ${sn.name}_ctavg[:] = np.float32(0.0)
        % endfor
        tavg_count[0] = 0

        network_chunk(
            this_chunk,
            t_global,
            % for sn in subnets:
            ${sn.name}_state,
            % endfor
            % for sn in subnets:
            ${sn.name}_srcbuf,
            % endfor
            % for p in all_projs:
            ${p.name}_w_data, ${p.name}_w_indices, ${p.name}_w_indptr,
            ${p.name}_idelays,
            % if p.is_inter:
            ${p.name}_mode_map,
            % endif
            ${p.name}_source_cvar, ${p.name}_target_cvar,
            ${p.name}_scale, ${p.name}_target_scales,
            ${p.name}_cfun_params,
            % endfor
            % for sn in subnets:
            ${sn.name}_tavg,
            % endfor
            tavg_count,
            % for sn in subnets:
            ${sn.name}_ctavg,
            % endfor
            % for sn in subnets:
            % if sn.is_stochastic:
            ${sn.name}_noise,
            % endif
            % endfor
            % for sn in subnets:
            % if sn.has_stimulus:
            ${sn.name}_stim,
            % endif
            % endfor
        )

        n = tavg_count[0]
        mid_t = (t_global + t_global + this_chunk - 1) * 0.5 * float(time_step)
        % for sn in subnets:
        ${sn.name}_times.append(mid_t)
        ${sn.name}_outputs.append(${sn.name}_tavg / np.float32(n))
        ${sn.name}_ctavg_outputs.append(${sn.name}_ctavg / np.float32(n))
        % endfor

        t_global += this_chunk

    ## package outputs in Simulator.run() format: list of (times, data) per subnetwork
    results = []
    % for sn in subnets:
    times_arr = np.array(${sn.name}_times, dtype=np.float64)
    ## stack outputs: each entry is (n_voi, n_nodes, n_modes) → (T, n_voi, n_nodes, n_modes)
    data_arr = np.stack(${sn.name}_outputs, axis=0)
    ## stack coupling outputs: each entry is (n_cvar, n_nodes, n_modes) → (T, n_cvar, n_nodes, n_modes)
    ctavg_arr = np.stack(${sn.name}_ctavg_outputs, axis=0)
    results.append((times_arr, data_arr, ctavg_arr))
    % endfor
    return results
