import math
import numpy as np
import numba as nb

sin, cos, exp = math.sin, math.cos, math.exp

# TODO inline='always'
@nb.jit(boundscheck=True)
def coupling(i, cX, weights, state, di):
    
    n_svar = state.shape[0]
    n_cvar = cX.shape[0]
    n_node = weights.shape[0]
    assert weights.shape[0] == weights.shape[1] == state.shape[2]

% for par in sim.coupling.parameter_names:
    ${par} = nb.float32(${getattr(sim.coupling, par)[0]})
% endfor

    for j in range(n_node):
        wij = nb.float32(weights[i, j])
        if (wij == nb.float32(0.0)):
            continue

% for cvar, cterm in zip(sim.model.cvar, sim.model.coupling_terms):
        x_i = state[${cvar}, 0, i]
        x_j = state[${cvar}, di[i, j], j]
        cX[${loop.index}] += wij * ${sim.coupling.pre_expr}
% endfor

% for cterm in sim.model.coupling_terms:
    gx = cX[${loop.index}]
    cX[${loop.index}] = ${sim.coupling.post_expr}
% endfor
