import numpy as np

sin, cos, exp = np.sin, np.cos, np.exp

def coupling(cX, weights, state
% if sim.connectivity.idelays.any():
    , delay_indices
% endif
):
    
    n_svar = state.shape[0]
    n_cvar = cX.shape[0]
    n_node = cX.shape[1]
    assert cX.shape[1] == weights.shape[0] == weights.shape[1] == state.shape[2]

% for par in sim.coupling.parameter_names:
    ${par} = ${getattr(sim.coupling, par)[0]}
% endfor
## generate code per cvar
% for cvar, cterm in zip(sim.model.cvar, sim.model.coupling_terms):

## don't generate x_i if not required
% if 'x_i' in sim.coupling.pre_expr:
    x_i = np.tile(state[${cvar}, 0], (1, n_node))
% endif
## if no non-zero idelays, use current state
% if sim.connectivity.idelays.any():
    x_j = state[${cvar}].flat[delay_indices]
% else:
    x_j = np.tile(state[${cvar},0], (n_node, 1))
% endif
## apply weights, do summation and store
    gx = np.sum(weights * (${sim.coupling.pre_expr}), axis=-1)
    cX[${loop.index}, :] = ${sim.coupling.post_expr}
% endfor


## A buffer of (time,nodes), ix is idelays, nn in np.tile(...)
## delay_indices = idelays * nn + np.r_[:nn]
## use A.flat[delay_] and np.roll(state,1,axis=1)

## but better to have state (svar, time, node)
## to use single 