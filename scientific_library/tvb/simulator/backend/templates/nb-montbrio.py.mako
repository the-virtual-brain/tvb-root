# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

import math
import numpy as np
import numba as nb

<%include file="nb-dfuns.py.mako" />


<%
    cvars = [sim.model.state_variables[i] for i in sim.model.cvar]
    stochastic = isinstance(sim.integrator, IntegratorStochastic)
%>

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


@nb.njit
def _mpr_integrate(
        N,       # number of regions
        dt,
        nstep,   # integration length
        i0,      # index to t0
        r,       # r buffer with initial history and pre-filled with noise
        V,       # V buffer with initial history and pre-filled with noise
        weights, 
        idelays,
        G,       # coupling scaling
        parmat,  # spatial parameters [nparams, nnodes]
        stimulus # stimulus [nnodes, ntimes] or None
):

    def r_bound(r):
        return r if r >= 0. else 0. # max(0., r) is faster?

    for i in range(i0, i0 + nstep):
        for n in range(N):
            r_c, V_c = cx(i-1, n, N, weights, ${','.join(cvars)}, idelays)

% if stochastic:
% for svar in sim.model.state_variables:
            ${svar}_noise = ${svar}[n,i]
% endfor
% endif
            # precomputed additive noise 
            #r_noise = r[n,i]
            #V_noise = V[n,i]
% if sim.stimulus is not None:
            # stimulus TODO reflect stvar
            r_stim = 0.0
            V_stim = stimulus[n,i-i0]
% endif
            # Heun integration step
            dr_0 = dx_r(r[n,i-1], V[n,i-1], r_c, V_c, parmat[n] ) 
            dV_0 = dx_V(r[n,i-1], V[n,i-1], r_c, V_c, parmat[n] ) 

            r_int = r[n,i-1] + dt*dr_0 + r_noise ${'' if sim.stimulus is None else '+dt*r_stim'}
            V_int = V[n,i-1] + dt*dV_0 + V_noise ${'' if sim.stimulus is None else '+dt*V_stim'}
            r_int = r_bound(r_int)

% if not compatibility_mode:
            # coupling
            r_c, V_c = cx(i, n, N, weights, ${','.join(cvars)}, idelays)
% endif
            r[n,i] = r[n,i-1] + dt*(dr_0 + dx_r(r_int, V_int, r_c, V_c, parmat[n]  ))/2.0 + r_noise ${'' if sim.stimulus is None else '+dt*r_stim'}

            V[n,i] = V[n,i-1] + dt*(dV_0 + dx_V(r_int, V_int, r_c, V_c, parmat[n]))/2.0 + V_noise ${'' if sim.stimulus is None else '+dt*V_stim'}
            r[n,i] = r_bound(r[n,i])

    return r, V
