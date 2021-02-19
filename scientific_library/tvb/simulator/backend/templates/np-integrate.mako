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
% endif

def integrate(state, weights, parmat, dX, cX
% if isinstance(sim.integrator, IntegratorStochastic):
    , nsig
% endif
):
    dt = ${sim.integrator.dt}
    coupling(cX, weights, state)
    dfuns(dX[0], state, cX, parmat)
% if isinstance(sim.integrator, EulerDeterministic):
    state += dt * dX[0]
% endif
% if isinstance(sim.integrator, EulerStochastic):
    state += dt * dX[0] + noise(nsig)
% endif
% if isinstance(sim.integrator, HeunDeterministic):
    dfuns(dX[1], state + dt * dX[0], cX, parmat)
    state += dt / 2 * (dX[0] + dX[1])
% endif
% if isinstance(sim.integrator, HeunStochastic):
    z = noise(nsig)
    dfuns(dX[1], state + dt * dX[0] + z, cX, parmat)
    state += dt / 2 * (dX[0] + dX[1]) + z
% endif
% if isinstance(sim.integrator, Identity):
    state[:] = dX[0]
% endif
% if isinstance(sim.integrator, IdentityStochastic):
    state[:] = dX[0] + noise(nsig)
% endif
% if isinstance(sim.integrator, RungeKutta4thOrderDeterministic):
    dfuns(dX[1], state + dt / 2 * dX[0], cX, parmat)
    dfuns(dX[2], state + dt / 2 * dX[1], cX, parmat)
    dfuns(dX[3], state + dt * dX[2], cX, parmat)
    state += dt / 6 * (dX[0] + 2*(dX[1] + dX[2]) + dX[3])
% endif
