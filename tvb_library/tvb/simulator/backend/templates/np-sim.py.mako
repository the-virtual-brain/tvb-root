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

import numpy as np

<%include file="np-coupling.py.mako" />
<%include file="np-dfuns.py.mako" />
<%include file="np-integrate.py.mako" />

<%
    from tvb.simulator.integrators import IntegratorStochastic
    stochastic = isinstance(sim.integrator, IntegratorStochastic)
    any_delays = sim.connectivity.idelays.any()
%>

def kernel(state, weights, trace, parmat
           ${', nsig' if stochastic else ''}
           ${', idelays' if any_delays else ''}
           ):

    # problem dimensions
    n_node = ${sim.connectivity.weights.shape[0]}
    n_svar = ${len(sim.model.state_variables)}
    n_cvar = ${len(sim.model.cvar)}
    nt = ${nstep} 

    # work space arrays
    dX = np.zeros((${sim.integrator.n_dx}, n_svar, n_node))
    cX = np.zeros((n_cvar, n_node))

    for t in range(nt):
        integrate(state, weights, parmat, dX, cX
           ${', nsig' if stochastic else ''}
           ${', idelays' if any_delays else ''}
           )
        trace[t] = state[:,0].copy()
