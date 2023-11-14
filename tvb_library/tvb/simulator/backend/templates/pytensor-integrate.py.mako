## -*- coding: utf-8 -*-
##
##
## TheVirtualBrain-Scientific Package. This package holds all simulators, and
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
import pytensor
from pytensor import tensor as pyt

<%
    from tvb.simulator.integrators import (IntegratorStochastic,
        EulerDeterministic, EulerStochastic,
        HeunDeterministic, HeunStochastic,
        Identity, IdentityStochastic, RungeKutta4thOrderDeterministic,
        SciPyODEBase)
    from tvb.simulator.noise import Additive, Multiplicative

    if isinstance(sim.integrator, SciPyODEBase):
        raise NotImplementedError

    if isinstance(sim.integrator, IntegratorStochastic):
        if isinstance(sim.integrator.noise, Multiplicative):
            raise NotImplementedError
%>

def integrate(state, weights, parmat, dX, cX
% if isinstance(sim.integrator, IntegratorStochastic):
    , z_t
% endif
% if sim.connectivity.idelays.any():
    , delay_indices
% endif
, **iparams
):
    dt = ${sim.integrator.dt}
    cX = coupling(cX, weights, state
% if sim.connectivity.idelays.any():
    , delay_indices
% endif
)
    dX = pyt.set_subtensor(dX[0], dfuns(dX[0], state[:, 0], cX, parmat))
% if isinstance(sim.integrator, EulerDeterministic):
    next_state = state[:, 0] + dt * dX[0]
% endif
% if isinstance(sim.integrator, EulerStochastic):
    next_state = state[:, 0] + dt * dX[0] + z_t
% endif
% if isinstance(sim.integrator, HeunDeterministic):
    dX = pyt.set_subtensor(dX[1], dfuns(dX[1], state[:, 0] + dt * dX[0], cX, parmat))
    next_state = state[:, 0] + dt / 2 * (dX[0] + dX[1])
% endif
% if isinstance(sim.integrator, HeunStochastic):
    dX = pyt.set_subtensor(dX[1], dfuns(dX[1], state[:, 0] + dt * dX[0] + z_t, cX, parmat))
    next_state = state[:,0] + dt / 2 * (dX[0] + dX[1]) + z_t
% endif
% if isinstance(sim.integrator, Identity):
    next_state = dX[0]
% endif
% if isinstance(sim.integrator, IdentityStochastic):
    next_state = dX[0] + z_t
% endif
% if isinstance(sim.integrator, RungeKutta4thOrderDeterministic):
    dX = pyt.set_subtensor(dX[1], dfuns(dX[1], state[:, 0] + dt / 2 * dX[0], cX, parmat))
    dX = pyt.set_subtensor(dX[2], dfuns(dX[2], state[:, 0] + dt / 2 * dX[1], cX, parmat))
    dX = pyt.set_subtensor(dX[3], dfuns(dX[3], state[:, 0] + dt * dX[2], cX, parmat))
    next_state = state[:, 0] + dt / 6 * (dX[0] + 2*(dX[1] + dX[2]) + dX[3])
% endif
    state = pyt.set_subtensor(state[:], pyt.roll(state, 1, axis=1))
    state = pyt.set_subtensor(state[:, 0], next_state)

    return state
