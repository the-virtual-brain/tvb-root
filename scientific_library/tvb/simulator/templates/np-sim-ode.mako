import numpy as np

<%include file="np-coupling.mako" />
<%include file="np-dfuns.mako" />

def kernel(state, weights, trace, parmat):

    # problem dimensions
    n_node = ${sim.connectivity.weights.shape[0]}
    n_svar = ${len(sim.model.state_variables)}
    n_cvar = ${len(sim.model.cvar)}
    dt = ${sim.integrator.dt}
    nt = ${int(sim.simulation_length/sim.integrator.dt)}

    # work space arrays
    dX = np.zeros((n_svar, n_node))
    cX = np.zeros((n_cvar, n_node))

    # time loop
    for t in range(nt):
        coupling(cX, weights, state)
        dfuns(dX, state, cX, parmat)

        # integrate w/ Euler
% for svar in sim.model.state_variables:
        state[${loop.index}] += dt * dX[${loop.index}];
% endfor

        # update monitor
        trace[t] = state.copy()
