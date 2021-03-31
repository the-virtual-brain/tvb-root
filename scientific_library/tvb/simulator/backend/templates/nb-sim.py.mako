import numpy as np
import numba as nb

<%include file="nb-coupling2.py.mako" />
<%include file="nb-dfuns.py.mako" />
<%include file="nb-integrate.py.mako" />

@nb.jit
def run_sim(sim, nstep=None, sim_time=None):
    # shapes
    nstep = nstep or int((sim_time or sim.simulation_length)/sim.integrator.dt)
    horizon = sim.connectivity.horizon
    nnode = sim.connectivity.weights.shape[0]
    nsvar = len(sim.model.state_variables)
    # arrays
    parmat = sim.model.spatial_parameter_matrix.astype('f')
    nsig = sim.integrator.noise.nsig
    weights = sim.connectivity.weights.astype('f')
    idelays = sim.connectivity.idelays.astype(np.uint32)
    # allocate buffers
    state = np.zeros((nsvar, nnode, horizon + nstep), 'f')
    # run
    for t in range(horizon, horizon + nstep):
        integrate(t, state, weights, parmat
           ${', nsig' if stochastic else ''}
           ${', idelays' if any_delays else ''}
           )
    return state
