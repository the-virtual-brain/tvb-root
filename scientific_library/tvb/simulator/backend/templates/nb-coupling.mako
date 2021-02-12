import math
import numpy as np
import numba as nb

sin, cos, exp = math.sin, math.cos, math.exp

@nb.njit
def coupling(cX, weights, state):
    
    n_svar = state.shape[0]
    n_cvar = cX.shape[0]
    n_node = cX.shape[1]
    assert cX.shape[1] == weights.shape[0] == weights.shape[1] == state.shape[1]

% for par in sim.coupling.parameter_names:
    ${par} = nb.float32(${getattr(sim.coupling, par)[0]})
% endfor

    # special names in cfun definitions
    x_i = nb.float32(0.0)
    x_j = nb.float32(0.0)
    gx = nb.float32(0.0)

    for i in range(n_node):
        for j in range(n_node):
            wij = nb.float32(weights[i, j])
            if (wij == nb.float32(0.0)):
                continue

% for cvar, cterm in zip(sim.model.cvar, sim.model.coupling_terms):
            x_i = state[${cvar}, i]
            x_j = state[${cvar}, j]
            cX[${loop.index}, i] += wij * ${sim.coupling.pre_expr}
% endfor

% for cterm in sim.model.coupling_terms:
        gx = cX[${loop.index}, i]
        cX[${loop.index}, i] = ${sim.coupling.post_expr}
% endfor
