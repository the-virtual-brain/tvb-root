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
import numba as nb

<%include file="nb-coupling2.py.mako" />
<%include file="nb-dfuns.py.mako" />
<%include file="nb-integrate.py.mako" />

<%
    from tvb.simulator.integrators import IntegratorStochastic
    stochastic = isinstance(sim.integrator, IntegratorStochastic)
    any_delays = sim.connectivity.idelays.any()
%>

${'' if debug_nojit else '@nb.njit(inline="always")'}
def loop(horizon, nstep, state, weights, parmat
           ${', nsig' if stochastic else ''}
           ${', idelays' if any_delays else ''}
           ):
    for t in range(horizon - 1, horizon + nstep - 1):
        integrate(t, state, weights, parmat
           ${', nsig' if stochastic else ''}
           ${', idelays' if any_delays else ''}
           )


def run_sim(sim, nstep=None, sim_time=None):
    # shapes
    nstep = nstep or int((sim_time or sim.simulation_length)/sim.integrator.dt)
    horizon = sim.connectivity.horizon
    nnode = sim.connectivity.weights.shape[0]
    nsvar = len(sim.model.state_variables)
    # arrays
    parmat = sim.model.spatial_parameter_matrix.T.astype(np.float32)
    weights = sim.connectivity.weights.astype(np.float32)
    idelays = sim.connectivity.idelays.astype(np.uint32)
    # allocate buffers
    state = np.zeros((nsvar, nnode, horizon + nstep), np.float32)
    assert sim.current_step == 0, 'requires an un-run simulator'
    state[:,:,:horizon-1] = np.transpose(sim.history.buffer[1:,:,:,0], (1,2,0))
    state[:,:, horizon-1] = sim.history.buffer[0,:,:,0]
% if stochastic:
    nsig = sim.integrator.noise.nsig
    # TODO use newer RNG infra in NumPy
    # TODO pre-apply sqrt(2*nsig*dt)
    # draw samples in same order as tvb sim for tests:
    state[...,horizon:] = np.transpose(
        np.random.randn(nstep, state.shape[0], state.shape[1]),
        (1, 2, 0))
% endif
    # run
    loop(horizon, nstep, state, weights, parmat
           ${', nsig' if stochastic else ''}
           ${', idelays' if any_delays else ''}
    )
    return state
