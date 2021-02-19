import numpy as np

<%include file="np-coupling.mako" />
<%include file="np-dfuns.mako" />
<%include file="np-integrate.mako" />
<%
    from tvb.simulator.integrators import IntegratorStochastic
%>

def kernel(state, weights, trace, parmat
% if isinstance(sim.integrator, IntegratorStochastic):
    , nsig
% endif
% if sim.connectivity.idelays.any():
    , delay_indices
% endif
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
% if isinstance(sim.integrator, IntegratorStochastic):
                  , nsig
% endif
% if sim.connectivity.idelays.any():
    , delay_indices
% endif
)
        trace[t] = state.copy()
