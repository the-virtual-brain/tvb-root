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

% if isinstance(sim.integrator, IntegratorStochastic):
def noise(nsig):
    n_node = ${sim.connectivity.weights.shape[0]}
    n_svar = ${len(sim.model.state_variables)}
    sqrt_dt = ${np.sqrt(sim.integrator.dt)}
    dWt = np.random.randn(n_svar, n_node)
    D = np.sqrt(2 * nsig)
    return sqrt_dt * D * dWt
% else:
# no noise function rendered for integrator ${type(sim.integrator)}
% endif

def integrate(state, weights, parmat, dX, cX
% if isinstance(sim.integrator, IntegratorStochastic):
    , nsig
% endif
% if sim.connectivity.idelays.any():
    , delay_indices
% endif
):
    dt = ${sim.integrator.dt}
    coupling(cX, weights, state
% if sim.connectivity.idelays.any():
    , delay_indices
% endif
)
    dfuns(dX[0], state[:,0], cX, parmat)
% if isinstance(sim.integrator, EulerDeterministic):
    next_state = state[:,0] + dt * dX[0]
% endif
% if isinstance(sim.integrator, EulerStochastic):
    next_state = state[:,0] + dt * dX[0] + noise(nsig)
% endif
% if isinstance(sim.integrator, HeunDeterministic):
    dfuns(dX[1], state[:,0] + dt * dX[0], cX, parmat)
    next_state = state[:,0] + dt / 2 * (dX[0] + dX[1])
% endif
% if isinstance(sim.integrator, HeunStochastic):
    z = noise(nsig)
    dfuns(dX[1], state[:,0] + dt * dX[0] + z, cX, parmat)
    next_state = state[:,0] + dt / 2 * (dX[0] + dX[1]) + z
% endif
% if isinstance(sim.integrator, Identity):
    state[:,0] = dX[0]
% endif
% if isinstance(sim.integrator, IdentityStochastic):
    state[:,0] = dX[0] + noise(nsig)
% endif
% if isinstance(sim.integrator, RungeKutta4thOrderDeterministic):
    dfuns(dX[1], state[:,0] + dt / 2 * dX[0], cX, parmat)
    dfuns(dX[2], state[:,0] + dt / 2 * dX[1], cX, parmat)
    dfuns(dX[3], state[:,0] + dt * dX[2], cX, parmat)
    next_state = state[:,0] + dt / 6 * (dX[0] + 2*(dX[1] + dX[2]) + dX[3])
% endif
    state[:] = np.roll(state, 1, axis=1)
    state[:,0] = next_state

