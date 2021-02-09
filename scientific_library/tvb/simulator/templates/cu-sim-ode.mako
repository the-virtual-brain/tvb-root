<%namespace name="cu" file="cu-defs.mako" />

<%include file="cu-coupling.mako"/>

#define M_PI_F 3.14159265358979f

__global__ void kernel(
    float * __restrict__ state,
    float * __restrict__ weights,
    float * __restrict__ trace

## data args for inhomogeneous mass model parameters
% for par in sim.model.parameter_names:
% if getattr(sim.model, par).size > 1:
    , float * __restrict__ ${par}_array
% endif
% endfor
)
{
    const unsigned int id = threadIdx.x;
    const unsigned int n_node = ${sim.connectivity.weights.shape[0]};

    /* shared memory */
    __shared__ float shared[${sim.connectivity.weights.shape[0] * (len(sim.model.state_variables) + len(sim.model.cvar))}];
    float *dX = &(shared[0]);
    float *cX = &(shared[n_node*${len(sim.model.state_variables)}]);

    /* simulator constants */
    float dt = ${sim.integrator.dt}f;
    unsigned int nt = ${int(sim.simulation_length/sim.integrator.dt)};
% for par in sim.model.parameter_names:
% if getattr(sim.model, par).size == 1:
    const float ${par} = ${getattr(sim.model, par)[0]}f;
% endif
% endfor
    const float pi = M_PI_F;

    if (threadIdx.x < n_node)
    {
        for (unsigned int t = 0; t < nt; t++)
        {
            /* compute coupling */
            coupling(id, n_node, cX, weights, state);

            /* inhomogeneous parameters */
% for par in sim.model.parameter_names:
% if getattr(sim.model, par).size > 1:
            float ${par} = ${par}_array[id];
% endif
% endfor

            /* compute dfuns */
% for cterm in sim.model.coupling_terms:
            float ${cterm} = cX[${loop.index}*n_node + id];
% endfor

% for svar in sim.model.state_variables:
            float ${svar} = state[n_node*${loop.index} + id];
% endfor

% for svar in sim.model.state_variables:
            dX[${loop.index}*n_node + id] = ${sim.model.state_variable_dfuns[svar]};
% endfor

            /* integrate */
% for svar in sim.model.state_variables:
            state[${loop.index}*n_node + id] += dt * dX[${loop.index}*n_node + id];
% endfor

            /* monitor */
% for svar in sim.model.state_variables:
            trace[t*2*n_node + ${loop.index}*n_node + id] = state[${loop.index}*n_node + id];
% endfor 
        } 
    }
    __syncthreads();
    if (id==0) printf("kernel is done!\n");
}
