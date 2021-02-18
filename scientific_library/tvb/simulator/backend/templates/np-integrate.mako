<%
    from tvb.simulator.integrators import IntegratorStochastic
    from tvb.simulator.integrators import EulerDeterministic, EulerStochastic
    from tvb.simulator.integrators import HeunDeterministic, HeunStochastic
    from tvb.simulator.noise import Additive, Multiplicative
%>

% if isinstance(sim.integrator, IntegratorStochastic):
<%
    assert isinstance(sim.integrator.noise, noise.Additive)
%>
def noise():
    n_node = ${sim.connectivity.weights.shape[0]}
    n_svar = ${len(sim.model.state_variables)}
    sqrt_dt = ${np.sqrt(sim.integrator.dt)}
    dWt = np.random.randn(n_svar, n_node)
    D = np.array(${np.sqrt(2.0 * sim.integrator.noise.nsig)})
    return sqrt_dt * D * dWt
% endif

def integrate(state, weights, parmat, dX, cX):

    dt = ${sim.integrator.dt}
    coupling(cX, weights, state)
    dfuns(dX[0], state, cX, parmat)

% if isinstance(sim.integrator, EulerDeterministic):
    state += dt * dX[0]
% endif

% if isinstance(sim.integrator, EulerStochastic):
    state += dt * dX[0] + noise()
% endif

% if isinstance(sim.integrator, HeunDeterministic):
    dfuns(dX[1], state + dt * dX[0], cX, parmat)
    state += dt / 2 * (dX[0] + dX[1])
% endif

## TODO optinos for higher order coupling and stochastic intermediate
% if isinstance(sim.integrator, HeunStochastic):
    z = noise()
    dfuns(dX[1], state + dt * dX[0] + z, cX, parmat)
    state += dt / 2 * (dX[0] + dX[1]) + z
% endif
