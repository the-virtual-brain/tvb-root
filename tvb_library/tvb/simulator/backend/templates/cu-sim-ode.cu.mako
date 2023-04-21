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

<%namespace name="cu" file="cu-defs.mako" />

#define M_PI_F 3.14159265358979f

<%include file="cu-coupling.cu.mako" />
<%include file="cu-dfuns.cu.mako" />


__global__ void kernel(
    float * __restrict__ state,
    float * __restrict__ weights,
    float * __restrict__ trace,
    float * __restrict__ parmat
)
{
    const unsigned int id = threadIdx.x;
    const unsigned int n_node = ${sim.connectivity.weights.shape[0]};

    /* shared memory */
    __shared__ float shared[${sim.connectivity.weights.shape[0] * (len(sim.model.state_variables) + len(sim.model.cvar))}];
    float *dX = &(shared[0]);
    float *cX = &(shared[n_node*${len(sim.model.state_variables)}]);

    /* simulator constants */
    float dt = ${sim.integrator.dt}f;
    unsigned int nt = ${int(sim.simulation_length/sim.integrator.dt)};


    if (threadIdx.x < n_node)
    {
        for (unsigned int t = 0; t < nt; t++)
        {
            __syncthreads();
            coupling(id, n_node, cX, weights, state);
            dfuns(id, n_node, dX, state, cX, parmat);

            /* integrate */
% for svar in sim.model.state_variables:
            state[${loop.index}*n_node + id] += dt * dX[${loop.index}*n_node + id];
% endfor

            /* monitor */
% for svar in sim.model.state_variables:
            trace[t*2*n_node + ${loop.index}*n_node + id] = state[${loop.index}*n_node + id];
% endfor 
        } 
    }
}
