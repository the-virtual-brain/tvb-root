<%namespace name="cu" file="cu-defs.mako" />

<%include file="cu-coupling.mako" />
<%include file="cu-dfuns.mako" />

#define M_PI_F 3.14159265358979f

__global__ void kernel(
    float * __restrict__ state,
    float * __restrict__ weights,
    float * __restrict__ trace,
    float * __restrict__ parmat
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


    if (threadIdx.x < n_node)
    {
        for (unsigned int t = 0; t < nt; t++)
        {
            __syncthreads();
            coupling(id, n_node, cX, weights, state);
            dfuns(id, n_node, dX, state, cX, parmat);

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
}
