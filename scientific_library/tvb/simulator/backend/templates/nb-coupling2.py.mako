import math
import numpy as np
import numba as nb
sin, cos, exp = math.sin, math.cos, math.exp

<%
    svars = ', '.join(sim.model.state_variables)
    any_delays = sim.connectivity.idelays.any()
%>

% for cvar, cterm in zip(sim.model.cvar, sim.model.coupling_terms):
${'' if debug_nojit else '@nb.njit(inline="always")'}
def cx_${cterm}(t, i, weights, cvar
                ${', di' if any_delays else ''}):
% for par in sim.coupling.parameter_names:
    ${par} = nb.float32(${getattr(sim.coupling, par)[0]})
% endfor
    gx = nb.float32(0.0)
    for j in range(weights.shape[0]):
        wij = nb.float32(weights[i, j])
        if (wij == nb.float32(0.0)):
            continue
        x_i = cvar[i, t]
        dij = ${'di[i, j]' if any_delays else 'nb.uint32(0)'}
        x_j = cvar[j, t - dij]
        gx += wij * ${sim.coupling.pre_expr}
    return ${sim.coupling.post_expr}
% endfor
