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

sin, cos, exp = math.sin, math.cos, math.exp

@nb.jit
def coupling(cX, weights, state, di):
    
    n_svar = state.shape[0]
    n_cvar = cX.shape[0]
    n_node = cX.shape[1]
    assert cX.shape[1] == weights.shape[0] == weights.shape[1] == state.shape[2]

% for par in sim.coupling.parameter_names:
    ${par} = nb.float32(${getattr(sim.coupling, par)[0]})
% endfor

    # special names in cfun definitions
    x_i = nb.float32(0.0)
    x_j = nb.float32(0.0)
    gx = nb.float32(0.0)

    for i in range(n_node):
        for j in range(n_node):
            wij = nb.float32(weights[i, j])
            if (wij == nb.float32(0.0)):
                continue

% for cvar, cterm in zip(sim.model.cvar, sim.model.coupling_terms):
            x_i = state[${cvar}, 0, i]
	    x_j = state[${cvar}, di[i, j], j]
            cX[${loop.index}, i] += wij * ${sim.coupling.pre_expr}
% endfor

% for cterm in sim.model.coupling_terms:
        gx = cX[${loop.index}, i]
        cX[${loop.index}, i] = ${sim.coupling.post_expr}
% endfor
