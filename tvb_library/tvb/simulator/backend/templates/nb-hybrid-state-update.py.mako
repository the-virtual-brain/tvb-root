## -*- coding: utf-8 -*-
##
## nb-hybrid-state-update.py.mako
##
## Generates per-subnetwork dfun + Heun/Euler integration step as an
## @nb.njit(inline="always") function.
##
## Required template variables (via <%def> invocation):
##   sn_name       : str  — unique subnetwork name suffix
##   svars         : list[str]  — state variable names, e.g. ["r", "V"]
##   cvars_terms   : list[str]  — coupling term names, e.g. ["Coupling_Term_r", "Coupling_Term_V"]
##   dfun_exprs    : dict[str->str] — svar -> dfun expression string
##   global_params : dict[str -> float] — name -> scalar value
##   integrator_type: str — "heun" or "euler"
##   dt            : float
##   n_nodes       : int
##   n_modes       : int
##   has_boundaries: bool — True if the scheme has boundary conditions
##   lim_lo        : float | None  — lower bound for r (state var index 0)
##   lim_hi        : float | None  — upper bound
##

import numba as nb
import numpy as np
import math

pi = math.pi


<%def name="render_state_update(sn_name, svars, cvar_terms, dfun_exprs,
                                 global_params, integrator_type, dt,
                                 n_nodes, n_modes,
                                 boundary_lo, boundary_hi)">
<%
    svars_str = ', '.join(svars)
    cterms_str = ', '.join(cvar_terms)
    i1svars_str = ', '.join(['i1' + s for s in svars])
    n_svars = len(svars)
    n_terms = len(cvar_terms)
%>

@nb.njit(inline="always")
def dfun_${sn_name}(${svars_str}, ${cterms_str}, parmat):
    ## parmat: (n_spatial_params, n_nodes) — empty if all params are global
    pi = math.pi
    ## unpack global parameters
    % for name, val in global_params.items():
    ${name} = nb.float32(${val})
    % endfor
    ## compute dfuns
    % for svar in svars:
    d_${svar} = nb.float32(${dfun_exprs[svar]})
    % endfor
    return (${', '.join(['d_' + s for s in svars])},)


@nb.njit(inline="always")
def integrate_${sn_name}(t, state, coupling, parmat):
    """Integrate subnetwork ${sn_name} one step.

    state   : (n_svars, n_nodes, n_modes) — updated in-place
    coupling: (n_svars, n_nodes, n_modes) — external coupling input
    parmat  : (n_spatial_params, n_nodes) — spatial parameters (may be empty)
    """
    dt = nb.float32(${dt})

    for i in range(${n_nodes}):
        for m in range(${n_modes}):
            ## unpack current state
            % for k, svar in enumerate(svars):
            ${svar} = state[${k}, i, m]
            % endfor
            ## unpack coupling terms
            % for k, ct in enumerate(cvar_terms):
            ${ct} = coupling[${k}, i, m]
            % endfor
            ## first derivative
            (${', '.join(['d0_' + s for s in svars])},) = dfun_${sn_name}(${svars_str}, ${cterms_str}, parmat[:, i] if parmat.shape[0] > 0 else parmat[:, 0])

            % if integrator_type == "euler":
            ## Euler step
            % for svar in svars:
            n${svar} = ${svar} + dt * d0_${svar}
            % endfor

            % elif integrator_type == "heun":
            ## Heun step — first Euler estimate
            % for svar in svars:
            i1${svar} = ${svar} + dt * d0_${svar}
            % endfor
            ## second derivative at estimate
            (${', '.join(['d1_' + s for s in svars])},) = dfun_${sn_name}(${i1svars_str}, ${cterms_str}, parmat[:, i] if parmat.shape[0] > 0 else parmat[:, 0])
            ## corrector
            % for svar in svars:
            n${svar} = ${svar} + dt * nb.float32(0.5) * (d0_${svar} + d1_${svar})
            % endfor
            % endif

            ## apply boundaries
            % for k, svar in enumerate(svars):
            <%
                lo, hi = boundary_lo.get(svar), boundary_hi.get(svar)
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

            ## write back
            % for k, svar in enumerate(svars):
            state[${k}, i, m] = n${svar}
            % endfor

</%def>
