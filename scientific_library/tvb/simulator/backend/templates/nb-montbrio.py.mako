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

import numpy as np
import numba




@numba.njit
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
        parmat   # spatial parameters [nparams, nnodes]
):
        ## unpack global parameters
% for par in sim.model.global_parameter_names:
    ${par} = ${getattr(sim.model, par).item()}
% endfor

    def dr(r, V, ${', '.join(sim.model.spatial_parameter_names)} ):
        dr = 1/tau * ( Delta / (np.pi * tau) + 2 * V * r)
        return dr
        
    def dV(r, V, r_c, ${', '.join(sim.model.spatial_parameter_names)}):
        dV = 1/tau * ( V**2 - np.pi**2 * tau**2 * r**2 + eta + J * tau * r + I + r_c ) 
        return dV

    def r_bound(r):
        return r if r >= 0. else 0. # max(0., r) is faster?

    for i in range(i0, i0 + nstep):
        for n in range(N):
            ## unpack spatialized parameters
% for par in sim.model.spatial_parameter_names:
            ${par} = parmat[${loop.index}][n]
% endfor
            # coupling
            r_c = 0
            for m in range(N):
                r_c += weights[n,m] * r[m, i - idelays[n, m] - 1]
            r_c = r_c * G # post

            # precomputed additive noise 
            r_noise = r[n,i]
            V_noise = V[n,i]

            # Heun integration step
            dr_0 = dr(r[n,i-1], V[n,i-1], ${', '.join(sim.model.spatial_parameter_names)} ) 
            dV_0 = dV(r[n,i-1], V[n,i-1], r_c, ${', '.join(sim.model.spatial_parameter_names)} ) 

            r_int = r[n,i-1] + dt*dr_0 + r_noise
            V_int = V[n,i-1] + dt*dV_0 + V_noise
            r_int = r_bound(r_int)

% if not compatibility_mode:
            # coupling
            r_c = 0
            for m in range(N):
                r_c += weights[n,m] * r[m, i - idelays[n, m]]
            r_c = r_c * G # post
% endif
            r[n,i] = r[n,i-1] + dt*(dr_0 + dr(r_int, V_int, ${', '.join(sim.model.spatial_parameter_names)} ))/2.0 + r_noise
            V[n,i] = V[n,i-1] + dt*(dV_0 + dV(r_int, V_int, r_c, ${', '.join(sim.model.spatial_parameter_names)}))/2.0 + V_noise
            r[n,i] = r_bound(r[n,i])

    return r, V
