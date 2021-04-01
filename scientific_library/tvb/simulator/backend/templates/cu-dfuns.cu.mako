__device__ void dfuns(
	unsigned int id,
	unsigned int n_node,
	float * __restrict__ dX,
	float * __restrict__ state,
	float * __restrict__ cX,
	float * __restrict__ parmat
) {

% for par in sim.model.global_parameter_names:
    const float ${par} = ${getattr(sim.model, par)[0]}f;
% endfor

% for par in sim.model.spatial_parameter_names:
	const float ${par} = parmat[${loop.index}*n_node + id];
% endfor

    const float pi = M_PI_F;

% for cterm in sim.model.coupling_terms:
    const float ${cterm} = cX[${loop.index}*n_node + id];
% endfor

% for svar in sim.model.state_variables:
    const float ${svar} = state[n_node*${loop.index} + id];
% endfor

% for svar in sim.model.state_variables:
    dX[${loop.index}*n_node + id] = ${sim.model.state_variable_dfuns[svar]};
% endfor    
}