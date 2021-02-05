__device__ void coupling(
	unsigned int id,
	unsigned int n_node,
	float * __restrict__ cX,
	float * __restrict__ weights,
	float * __restrict__ state
)
{
% for par in sim.coupling.parameter_names:
	const float ${par} = ${getattr(sim.coupling, par)[0]}f;
% endfor

	float x_i, x_j, gx; // special names in cfun definitions

	for (unsigned int j=0; j < n_node; j++)
	{
		const float wij = weights[j*n_node + id];
		if (wij == 0.0f)
			continue;

% for cvar, cterm in zip(sim.model.cvar, sim.model.coupling_terms):
		x_i = state[${cvar}*n_node + id];
		x_j = state[${cvar}*n_node +  j];
		cX[${loop.index}*n_node + id] += wij * ${sim.coupling.pre_expr};
% endfor
	}

% for cterm in sim.model.coupling_terms:
	gx = cX[${loop.index}*n_node + id];
	cX[${loop.index}*n_node + id] = ${sim.coupling.post_expr};
% endfor
}