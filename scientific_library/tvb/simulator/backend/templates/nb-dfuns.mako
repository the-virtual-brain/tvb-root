import math
import numpy as np
import numba as nb
sin, cos, exp = math.sin, math.cos, math.exp

% for svar in sim.model.state_variables:
#@nb.njit
def dx_${svar}(state, cX, parmat):
    ## unpack global parameters
% for par in sim.model.global_parameter_names:
    ${par} = ${getattr(sim.model, par)[0]}
% endfor
    ## unpack spatialized parameters
% for par in sim.model.spatial_parameter_names:
    ${par} = parmat[${loop.index}]
% endfor
    pi = np.pi
    ## unpack coupling terms and states as in dfuns
    ${','.join(sim.model.coupling_terms)} = cX
    ${','.join(sim.model.state_variables)} = state
    ## compute dx
    return ${sim.model.state_variable_dfuns[svar]};
% endfor
