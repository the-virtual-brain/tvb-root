import numpy as np

def dfuns(dX, state, cX, parmat):

% for par in sim.model.global_parameter_names:
    ${par} = ${getattr(sim.model, par)[0]}
% endfor

% for par in sim.model.spatial_parameter_names:
    ${par} = parmat[${loop.index}]
% endfor

    pi = np.pi

    # unpack coupling terms and states as in dfuns
    ${','.join(sim.model.coupling_terms)} = cX
    ${','.join(sim.model.state_variables)} = state

    # compute dfuns
% for svar in sim.model.state_variables:
    dX[${loop.index}] = ${sim.model.state_variable_dfuns[svar]};
% endfor