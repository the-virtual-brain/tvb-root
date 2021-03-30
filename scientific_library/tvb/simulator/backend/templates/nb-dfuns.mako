import math
import numpy as np
import numba as nb
sin, cos, exp = math.sin, math.cos, math.exp

<%
    svars = ', '.join(sim.model.state_variables)
    cvars = ', '.join(sim.model.coupling_terms)
%>

% for svar in sim.model.state_variables:
@nb.njit(boundscheck=False, inline='always')
def dx_${svar}(${svars}, ${cvars}, parmat):
    ## unpack global parameters
% for par in sim.model.global_parameter_names:
    ${par} = ${getattr(sim.model, par)[0]}
% endfor
    ## unpack spatialized parameters
% for par in sim.model.spatial_parameter_names:
    ${par} = parmat[${loop.index}]
% endfor
    pi = np.pi
    ## compute dx
    return ${sim.model.state_variable_dfuns[svar]};
% endfor
