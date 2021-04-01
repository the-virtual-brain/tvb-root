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
    nt = ${int(sim.simulation_length/sim.integrator.dt)}

    # work space arrays
    dX = np.zeros((${sim.integrator.n_dx}, n_svar, n_node))
    cX = np.zeros((n_cvar, n_node))

    for t in range(nt):
        integrate(state, weights, parmat, dX, cX
           ${', nsig' if stochastic else ''}
           ${', idelays' if any_delays else ''}
           )
        trace[t] = state[:,0].copy()
