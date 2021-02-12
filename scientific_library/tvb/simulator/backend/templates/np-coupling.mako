import numpy as np

sin, cos, exp = np.sin, np.cos, np.exp

def coupling(cX, weights, state):
    
    n_svar = state.shape[0]
    n_cvar = cX.shape[0]
    n_node = cX.shape[1]
    assert cX.shape[1] == weights.shape[0] == weights.shape[1] == state.shape[1]

% for par in sim.coupling.parameter_names:
    ${par} = ${getattr(sim.coupling, par)[0]}
% endfor

% for cvar, cterm in zip(sim.model.cvar, sim.model.coupling_terms):
    x_i = state[${cvar}].reshape((-1, 1))
    x_j = state[${cvar}].reshape((1, -1))
    x_i, x_j = x_i + 0*x_j, x_j + 0*x_i
    gx = np.sum(weights * (${sim.coupling.pre_expr}), axis=-1)
    cX[${loop.index}, :] = ${sim.coupling.post_expr}
% endfor
