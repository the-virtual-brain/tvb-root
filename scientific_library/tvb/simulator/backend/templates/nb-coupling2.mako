import math
import numpy as np
import numba as nb
sin, cos, exp = math.sin, math.cos, math.exp

% for cvar, cterm in zip(sim.model.cvar, sim.model.coupling_terms):
@nb.jit(boundscheck=True, inline='always')
def cx_${cterm}(t, i, weights, state, di):
% for par in sim.coupling.parameter_names:
    ${par} = nb.float32(${getattr(sim.coupling, par)[0]})
% endfor
    gx = nb.float32(0.0)
    for j in range(weights.shape[0]):
        wij = nb.float32(weights[i, j])
        if (wij == nb.float32(0.0)):
            continue
        x_i = state[${cvar}, i, t]
        x_j = state[${cvar}, j, t - di[i, j]]
	gx += wij * ${sim.coupling.pre_expr}
    return ${sim.coupling.post_expr}
% endfor
