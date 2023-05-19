## -*- coding: utf-8 -*-
##
##
## TheVirtualBrain-Scientific Package. This package holds all simulators, and
## analysers necessary to run brain-simulations. You can use it stand alone or
## in conjunction with TheVirtualBrain-Framework Package. See content of the
## documentation-folder for more details. See also http://www.thevirtualbrain.org
##
## (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
## When using The Virtual Brain for scientific publications, please cite it as explained here:
## https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
##
##

import math
import numpy as np
import numba as nb
sin, cos, exp = math.sin, math.cos, math.exp

<%
    svars = ', '.join(sim.model.state_variables)
    any_delays = sim.connectivity.idelays.any()
%>

% for cvar, cterm in zip(sim.model.cvar, sim.model.coupling_terms):
${'' if debug_nojit else '@nb.njit(inline="always")'}
def cx_${cterm}(t, i, weights, cvar
                ${', di' if any_delays else ''}):
% for par in sim.coupling.parameter_names:
    ${par} = nb.float32(${getattr(sim.coupling, par)[0]})
% endfor
    gx = nb.float32(0.0)
    for j in range(weights.shape[0]):
        wij = nb.float32(weights[i, j])
        if (wij == nb.float32(0.0)):
            continue
        x_i = cvar[i, t]
        dij = ${'di[i, j]' if any_delays else 'nb.uint32(0)'}
        x_j = cvar[j, t - dij]
        gx += wij * ${sim.coupling.pre_expr}
    return ${sim.coupling.post_expr}
% endfor
