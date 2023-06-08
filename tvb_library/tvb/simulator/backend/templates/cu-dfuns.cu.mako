/*
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
*/

__device__ void dfuns(
	unsigned int id,
	unsigned int n_node,
	float * __restrict__ dX,
	float * __restrict__ state,
	float * __restrict__ cX,
	float * __restrict__ parmat
) {

% for par in sim.model.global_parameter_names:
    const float ${par} = ${getattr(sim.model, par)[0]}f;
% endfor

% for par in sim.model.spatial_parameter_names:
	const float ${par} = parmat[${loop.index}*n_node + id];
% endfor

    const float pi = M_PI_F;

% for cterm in sim.model.coupling_terms:
    const float ${cterm} = cX[${loop.index}*n_node + id];
% endfor

% for svar in sim.model.state_variables:
    const float ${svar} = state[n_node*${loop.index} + id];
% endfor

% for svar in sim.model.state_variables:
    dX[${loop.index}*n_node + id] = ${sim.model.state_variable_dfuns[svar]};
% endfor    
}