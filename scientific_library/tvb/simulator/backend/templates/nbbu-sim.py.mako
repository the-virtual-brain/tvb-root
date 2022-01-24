## -*- coding: utf-8 -*-
##
##
##  TheVirtualBrain-Scientific Package. This package holds all simulators, and
## analysers necessary to run brain-simulations. You can use it stand alone or
## in conjunction with TheVirtualBrain-Framework Package. See content of the
## documentation-folder for more details. See also http://www.thevirtualbrain.org
##
## (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
##
## This program is free software: you can redistribute it and/or modify it under the
## terms of the GNU General Public License as published by the Free Software Foundation,
## either version 3 of the License, or (at your option) any later version.
## This program is distributed in the hope that it will be useful, but WITHOUT ANY
## WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
## PARTICULAR PURPOSE.  See the GNU General Public License for more details.
## You should have received a copy of the GNU General Public License along with this
## program.  If not, see <http://www.gnu.org/licenses/>.
##
##
##   CITATION:
## When using The Virtual Brain for scientific publications, please cite it as follows:
##
##   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
##   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
##       The Virtual Brain: a simulator of primate brain network dynamics.
##   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
##
##

import numpy as np
import numba as nb
from math import sqrt, sin, cos, exp, pi, tanh

<%
    svars = ', '.join(sim.model.state_variables)
    cvars = ', '.join(sim.model.coupling_terms)
%>

% if not debug_nojit:
@nb.njit(fastmath=${fastmath}, boundscheck=${boundscheck}, parallel=${parallel})
% endif
def kernel(dt, weights, idelays, states, parmat):
    ## unpack global parameters
% for par in sim.model.global_parameter_names:
    ${par} = ${getattr(sim.model, par)[0]}
% endfor
    ## unpack spatialized parameters
% for par in sim.model.spatial_parameter_names:
    ${par} = parmat[${loop.index}]
% endfor

    # loop over cores
    for k in nb.prange(r.shape[0]):
        # loop over time
        for t in range(${nt}-1):
            # loop over nodes
            for i in range(r.shape[1]):
                ## TODO break out into separate template
                ## TODO handle coupling vars
% for i in range(nl):
                acc${i} = nb.float32(0.0)
% endfor
                # loop over afferent connections
                for j in range(r.shape[1]):
                    if weights[i,j] == nb.float32(0):
                        continue
% for i in range(nl):
  % if nh > 1:
                    rj${i} = r[k, j, ${nh} + t - idelays[i, j], ${i}]
  % else:
                    rj${i} = r[k, j, ${nh} + t, ${i}]
  % endif
                    acc${i} += weights[i,j] * rj${i}
% endfor

                # TODO unpack state vars
                # TODO unpack noise vars

% for i in range(nl):
  % for svar in sim.model.state_variables:
                # TODO need a sympy replacement trick for _${i} vars
                ${svar}_${i} = ${sim.model.state_variable_dfuns[svar]};
  % endfor
% endfor



          # TODO TODO and see, this is an Euler scheme.  For Heun, 
          # we need to do this whole thing twice, but rk4 doesn't apply
          # to noise, and euler is not good.  Could just default to Heun.


% for i in range(nl):
                r_c${i} = g[k,${i}] * acc${i}
                r${i} = r[k, i, ${nh} + t, ${i}]
                V${i} = V[k, i, ${nh} + t, ${i}]
                r_noise${i} = r[k, i, ${nh} + t + 1, ${i}]
                V_noise${i} = V[k, i, ${nh} + t + 1, ${i}]
                dr${i} = rtau * (Delta_rpitau + 2 * V${i} * r${i})
                dV${i} = 1/tau * ( V${i}**2 - np.pi**2 * tau**2 * r${i}**2 + eta + J * tau * r${i} + I + r_c${i} ) 
                nr${i} = r${i} + dt*dr${i} + nscl*r_noise${i}
                r[k, i, ${nh} + t + 1, ${i}] = nr${i} * (nr${i} > 0.0)
                V[k, i, ${nh} + t + 1, ${i}] = V${i} + dt*dV${i} + nscl*V_noise${i}
% endfor

