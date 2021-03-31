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
    state[:,:,:horizon] = np.transpose(sim.history.buffer[...,0], (1,2,0))
% if stochastic:
    nsig = sim.integrator.noise.nsig
    # TODO use newer RNG infra in NumPy
    # TODO pre-apply sqrt(2*nsig*dt)
    state[...,horizon:] = np.random.randn(*state[...,horizon:].shape)
% endif
    # run
    loop(horizon, nstep, state, weights, parmat
           ${', nsig' if stochastic else ''}
           ${', idelays' if any_delays else ''}
    )
    return state
