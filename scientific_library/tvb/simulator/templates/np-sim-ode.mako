import numpy as np

<%include file="np-coupling.mako" />

<%namespace name="util" file="util.mako" />

def kernel(state, weights, trace

    # varying parameters
% for par in sim.model.parameter_names:
% if getattr(sim.model, par).size > 1:
    , ${par}
% endif
% endfor
):

    # problem dimensions
    n_node = ${sim.connectivity.weights.shape[0]}
    n_svar = ${len(sim.model.state_variables)}
    n_cvar = ${len(sim.model.cvar)}
    dt = ${sim.integrator.dt}
    nt = ${int(sim.simulation_length/sim.integrator.dt)}

    # work space arrays
    dX = np.zeros((n_svar, n_node))
    cX = np.zeros((n_cvar, n_node))

    # constant parameters
% for par in sim.model.parameter_names:
% if getattr(sim.model, par).size == 1:
    ${par} = ${getattr(sim.model, par)[0]}
% endif
% endfor
    pi = np.pi

    # time loop
    for t in range(nt):
        coupling(cX, weights, state)

        # unpack coupling terms and states as in dfuns
        ${','.join(sim.model.coupling_terms)} = cX
        ${','.join(sim.model.state_variables)} = state

        # compute dfuns
% for svar in sim.model.state_variables:
        dX[${loop.index}] = ${sim.model.state_variable_dfuns[svar]};
% endfor

        # integrate w/ Euler
% for svar in sim.model.state_variables:
        state[${loop.index}] += dt * dX[${loop.index}];
% endfor

        # update monitor
% for svar in sim.model.state_variables:
        trace[t] = state
% endfor 
