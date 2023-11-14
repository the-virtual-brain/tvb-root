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

<%include file="pytensor-coupling.py.mako" />
<%include file="pytensor-dfuns.py.mako" />
<%include file="pytensor-integrate.py.mako" />

<%
    from tvb.simulator.integrators import IntegratorStochastic
    from tvb.simulator.noise import Additive
    stochastic = isinstance(sim.integrator, IntegratorStochastic)
    any_delays = sim.connectivity.idelays.any()
%>

## TODO handle multiplicative noise
% if isinstance(sim.integrator, IntegratorStochastic):
def default_noise(nsig):
    n_node = ${sim.connectivity.weights.shape[0]}
    n_svar = ${len(sim.model.state_variables)}
    nt = ${int(sim.simulation_length/sim.integrator.dt)}
    sqrt_dt = ${np.sqrt(sim.integrator.dt)}

    white_noise = sqrt_dt * pyt.as_tensor_variable(np.random.randn(nt, n_svar, n_node))
    noise_gfun = pyt.sqrt(2 * nsig)
    return pyt.transpose(pyt.transpose(white_noise, (1, 0, 2)) * noise_gfun, (1, 0, 2))
% else:
def default_noise():
    # no noise function rendered for integrator ${type(sim.integrator)}
    return None
% endif

def kernel(state, weights, trace, parmat
           ${', noise' if stochastic else ''}
           ${', idelays' if any_delays else ''}
           ):

    # problem dimensions
    n_node = ${sim.connectivity.weights.shape[0]}
    n_svar = ${len(sim.model.state_variables)}
    n_cvar = ${len(sim.model.cvar)}
    nt = ${int(sim.simulation_length/sim.integrator.dt)}

    # work space arrays
    dX = pyt.zeros((${sim.integrator.n_dx}, n_svar, n_node))
    cX = pyt.zeros((n_cvar, n_node))

    def scan_fn(${'noise,' if stochastic else ''} state, weights, parmat, dX, cX
           ${', idelays' if any_delays else ''}
           ):
           return integrate(state, weights, parmat, dX, cX
           ${', noise' if stochastic else ''}
           ${', idelays' if any_delays else ''}
           )
    args = [weights, parmat, dX, cX ${', idelays' if any_delays else ''}]
    trace, updates = pytensor.scan(fn=scan_fn, outputs_info=state, non_sequences=args, n_steps=nt ${', sequences=[noise]' if stochastic else ''})

    return trace
