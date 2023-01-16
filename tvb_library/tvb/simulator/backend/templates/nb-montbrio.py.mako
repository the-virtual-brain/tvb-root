# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

import math
import numpy as np
import numba as nb

<%include file="nb-dfuns.py.mako" />


<%
    import numpy as np
    from tvb.simulator.integrators import IntegratorStochastic
    cvars = [sim.model.state_variables[i] for i in sim.model.cvar]
    stochastic = isinstance(sim.integrator, IntegratorStochastic)

    cvar_symbols =  ','.join([f'{cvar}_c' for cvar in cvars])
%>

# Coupling
${'' if debug_nojit else '@nb.njit(inline="always")'}
def cx(t, i, N, weights, ${','.join(cvars)}, idelays):
% for par in sim.coupling.parameter_names:
    ${par} = ${getattr(sim.coupling, par)[0]}
% endfor

% for cterm in sim.model.coupling_terms:
    ${cterm} = 0.0
% endfor
    for j in range(N):
% for cterm, cvar in zip(sim.model.coupling_terms, cvars):
        x_j = ${cvar}[j, t - idelays[i,j]]
        ${cterm} +=  weights[i, j]* ${sim.coupling.pre_expr}
% endfor
% for cterm, cvar in zip(sim.model.coupling_terms, cvars):
    gx = ${cterm}
    x_i = ${cvar}[i, t]
    ${cterm} = ${sim.coupling.post_expr}
% endfor
    return ${','.join(sim.model.coupling_terms)}

# svar bound functions
% if sim.model.state_variable_boundaries is not None:
% for svar, (lo, hi) in sim.model.state_variable_boundaries.items():
${'' if debug_nojit else '@nb.njit(inline="always")'}
def bound_${svar}(x):
% if lo > -np.inf: # this doesn't work, fix later
    x = x if x >= ${lo} else ${lo} 
% endif
% if hi < np.inf:
    x = x if x <= ${hi} else ${hi}
% endif
    return x
% endfor
% endif

## this is fragile due to fixed intentation
<%def name='call_bound_svars(svars)'>
% if sim.model.state_variable_boundaries is not None:
% for svar, lsvar in zip(sim.model.state_variables, svars):
% if svar in sim.model.state_variable_boundaries.keys():
            ${lsvar} = bound_${svar}(${lsvar})
% endif
% endfor    
% endif
</%def>


@nb.njit
def integrate(
        N,       # number of regions
        dt,
        nstep,   # integration length
        i0,      # index to t0
% for svar in sim.model.state_variables:
        ${svar},       # ${svar} buffer with initial history and pre-filled with noise
% endfor
        weights, 
        idelays,
        parmat,  # spatial parameters [nparams, nnodes]
        stimulus # stimulus [nnodes, ntimes] or None
):

    for i in range(i0, i0 + nstep):
        for n in range(N):
            ${cvar_symbols} = cx(i-1, n, N, weights, ${','.join(cvars)}, idelays)

% for svar in sim.model.state_variables:
% if stochastic:
            # precomputed additive noise 
            ${svar}_noise = ${svar}[n,i]
% else:
            ${svar}_noise = 0.0
% endif
% endfor

% if sim.stimulus is not None:
% for i, svar in enumerate(sim.model.state_variables):
% if i in sim.model.stvar:
            ${svar}_stim = dt*stimulus[n,i-i0]
% else:
            ${svar}_stim = 0.0
% endif
% endfor
% else:
% for svar in sim.model.state_variables:
            ${svar}_stim = 0.0
% endfor
% endif
            # Heun integration step
% for svar in sim.model.state_variables:
            d${svar}_0 = dx_${svar}(
% for ssvar in sim.model.state_variables:
                                ${ssvar}[n,i-1], 
% endfor
                                ${cvar_symbols},
                                parmat[n] 
            ) 
% endfor

% for svar in sim.model.state_variables:
            ${svar}_int = ${svar}[n,i-1] + dt*d${svar}_0 + ${svar}_noise + ${svar}_stim
% endfor

            ${call_bound_svars([f'{ssvar}_int' for ssvar in sim.model.state_variables])}

% if not compatibility_mode:
            # coupling
            ${cvar_symbols} = cx(i, n, N, weights, ${','.join(cvars)}, idelays)
% endif
% for svar in sim.model.state_variables:
            ${svar}_n = ${svar}[n,i-1] + dt*(d${svar}_0 + dx_${svar}(
                ${','.join([f'{ssvar}_int' for ssvar in sim.model.state_variables])},
                ${cvar_symbols}, parmat[n]  ))/2.0 + ${svar}_noise + ${svar}_stim
% endfor
            ${call_bound_svars([f'{ssvar}_n' for ssvar in sim.model.state_variables])}

% for svar in sim.model.state_variables:
            ${svar}[n,i] = ${svar}_n
% endfor

    return ${','.join(svar for svar in sim.model.state_variables)}
