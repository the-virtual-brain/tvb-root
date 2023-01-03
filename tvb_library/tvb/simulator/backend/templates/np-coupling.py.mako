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

import numpy as np

sin, cos, exp = np.sin, np.cos, np.exp

def coupling(cX, weights, state
% if sim.connectivity.idelays.any():
    , delay_indices
% endif
):
    
    n_svar = state.shape[0]
    n_cvar = cX.shape[0]
    n_node = cX.shape[1]
    assert cX.shape[1] == weights.shape[0] == weights.shape[1] == state.shape[2]

% for par in sim.coupling.parameter_names:
    ${par} = ${getattr(sim.coupling, par)[0]}
% endfor
## generate code per cvar
% for cvar, cterm in zip(sim.model.cvar, sim.model.coupling_terms):

## don't generate x_i if not required
% if 'x_i' in sim.coupling.pre_expr:
    x_i = np.tile(state[${cvar}, 0], (1, n_node))
% endif
## if no non-zero idelays, use current state
% if sim.connectivity.idelays.any():
    x_j = state[${cvar}].flat[delay_indices]
% else:
    x_j = np.tile(state[${cvar},0], (n_node, 1))
% endif
## apply weights, do summation and store
    gx = np.sum(weights * (${sim.coupling.pre_expr}), axis=-1)
    cX[${loop.index}, :] = ${sim.coupling.post_expr}
% endfor


## A buffer of (time,nodes), ix is idelays, nn in np.tile(...)
## delay_indices = idelays * nn + np.r_[:nn]
## use A.flat[delay_] and np.roll(state,1,axis=1)

## but better to have state (svar, time, node)
## to use single 
