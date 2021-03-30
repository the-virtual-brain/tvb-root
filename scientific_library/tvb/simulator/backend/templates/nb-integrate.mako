import numba as nb

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

    stochastic = isinstance(sim.integrator, IntegratorStochastic)
    any_delays = sim.connectivity.idelays.any()
    svars = ', '.join(sim.model.state_variables)
    cvars = ', '.join(sim.model.coupling_terms)
%>

## TODO multiplicative noise
% if stochastic:
@nb.njit(inline='always', boundscheck=False)
def noise(isvar, nsig):
    n_node = ${sim.connectivity.weights.shape[0]}
    n_svar = ${len(sim.model.state_variables)}
    sqrt_dt = ${np.sqrt(sim.integrator.dt)}
    dWt = np.random.randn(n_svar, n_node)
    D = np.sqrt(2 * nsig)
    return sqrt_dt * D * dWt
% else:
# no noise function rendered for integrator ${type(sim.integrator)}
% endif

@nb.njit
def integrate(t, state, weights, parmat
    ${', nsig' if stochastic else ''}
    ${', idelays' if any_delays else ''}
):
    dt = ${sim.integrator.dt}

    for i in range(weights.shape[0]):

        # unpack current states
% for svar in sim.model.state_variables:
        ${svar} = state[${loop.index}, i, t]
% endfor

        # compute coupling terms
% for cvar, cterm in zip(sim.model.cvar, sim.model.coupling_terms):
        ${cterm} = cx_${cterm}(t, i, weights, state[${cvar}], ${'idelays' if any_delays else ''})
% endfor

        # compute derivatives and next states
% for svar in sim.model.state_variables:
        d0${svar} = dx_${svar}(${svars}, ${cvars}, parmat)
% endfor

## Euler
% if isinstance(sim.integrator, EulerDeterministic):
% for svar in sim.model.state_variables:
        n${svar} = ${svar} + dt * d0${svar}
% endfor
% endif

% if isinstance(sim.integrator, EulerStochastic):
% for svar in sim.model.state_variables:
        n${svar} = ${svar} + dt * d0${svar} + noise(${loop.index}, nsig)
% endfor
% endif

## Heun
<%
    i1svars = ', '.join([f'i1{svar}' for svar in sim.model.state_variables])
%>
% if isinstance(sim.integrator, HeunDeterministic):
% for svar in sim.model.state_variables:
        i1${svar} = ${svar} + dt * d0${svar}
% endfor
% for svar in sim.model.state_variables:
        d1${svar} = dx_${svar}(${i1svars}, ${cvars}, parmat)
% endfor
% for svar in sim.model.state_variables:
        n${svar} = ${svar} + dt / 2 * (d0${svar} + d1${svar})
% endfor
% endif

% if isinstance(sim.integrator, HeunStochastic):
% for svar in sim.model.state_variables:
        z${svar} = noise(${loop.index}, nsig)
% endfor
% for svar in sim.model.state_variables:
        i1${svar} = ${svar} + dt * d0${svar} + z${svar}
% endfor
% for svar in sim.model.state_variables:
        d1${svar} = dx_${svar}(${i1svars}, ${cvars}, parmat)
% endfor
% for svar in sim.model.state_variables:
        n${svar} = ${svar} + dt / 2 * (d0${svar} + d1${svar}) + z${svar}
% endfor
% endif

## Others
% if isinstance(sim.integrator, Identity):
% for svar in sim.model.state_variables:
        n${svar} = d0${svar}
% endfor
% endif

% if isinstance(sim.integrator, IdentityStochastic):
% for svar in sim.model.state_variables:
        n${svar} = d0${svar} + noise(${loop.index}, nsig)
% endfor
% endif

<%
    i1svars = ', '.join([f'i1{svar}' for svar in sim.model.state_variables])
    i2svars = ', '.join([f'i2{svar}' for svar in sim.model.state_variables])
    i3svars = ', '.join([f'i3{svar}' for svar in sim.model.state_variables])
%>
% if isinstance(sim.integrator, RungeKutta4thOrderDeterministic):
% for svar in sim.model.state_variables:
        i1${svar} = ${svar} + dt / 2 * d0${svar}
% endfor
% for svar in sim.model.state_variables:
        d1${svar} = dx_${svar}(${i1svars}, ${cvars}, parmat)
% endfor
% for svar in sim.model.state_variables:
        i2${svar} = ${svar} + dt / 2 * d1${svar}
% endfor
% for svar in sim.model.state_variables:
        d2${svar} = dx_${svar}(${i2svars}, ${cvars}, parmat)
% endfor
% for svar in sim.model.state_variables:
        i3${svar} = ${svar} + dt * d2${svar}
% endfor
% for svar in sim.model.state_variables:
        d3${svar} = dx_${svar}(${i3svars}, ${cvars}, parmat)
% endfor
% for svar in sim.model.state_variables:
        n${svar} = ${svar} + dt / 6 * (d0${svar} + 2*(d1${svar} + d2${svar}) + d3${svar})
% endfor
% endif

## Update buffer
% for svar in sim.model.state_variables:
        state[${loop.index},i,t+1] = next_state
% endfor
