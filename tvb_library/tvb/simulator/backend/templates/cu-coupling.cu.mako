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

__device__ void coupling(
	unsigned int id,
	unsigned int n_node,
	float * __restrict__ cX,
	float * __restrict__ weights,
	float * __restrict__ state
)
{
% for par in sim.coupling.parameter_names:
	const float ${par} = ${getattr(sim.coupling, par)[0]}f;
% endfor

	float x_i, x_j, gx; // special names in cfun definitions

% for cterm in sim.model.coupling_terms:
	cX[${loop.index}*n_node + id] = 0.0f;
% endfor

	for (unsigned int j=0; j < n_node; j++)
	{
		const float wij = weights[j*n_node + id];
		if (wij == 0.0f)
			continue;

% for cvar, cterm in zip(sim.model.cvar, sim.model.coupling_terms):
		x_i = state[${cvar}*n_node + id];
		x_j = state[${cvar}*n_node +  j];
		cX[${loop.index}*n_node + id] += wij * ${sim.coupling.pre_expr};
% endfor
	}

% for cterm in sim.model.coupling_terms:
	gx = cX[${loop.index}*n_node + id];
	cX[${loop.index}*n_node + id] = ${sim.coupling.post_expr};
% endfor
}