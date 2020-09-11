#include <stdio.h> // for printf
#define PI_2 (2 * M_PI_F)

// buffer length defaults to the argument to the integrate kernel
// but if it's known at compile time, it can be provided which allows
// compiler to change i%n to i&(n-1) if n is a power of two.
#ifndef NH
#define NH nh
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif


#include <curand_kernel.h>
#include <curand.h>

__device__ float wrap_2_pi_(float x)/*{{{*/
{
    bool neg_mask = x < 0.0f;
    bool pos_mask = !neg_mask;
    // fmodf diverges 51% of time
    float pos_val = fmodf(x, PI_2);
    float neg_val = PI_2 - fmodf(-x, PI_2);
    return neg_mask * neg_val + pos_mask * pos_val;
}/*}}}*/

__device__ float wrap_2_pi(float x) // not divergent/*{{{*/
{
    bool lt_0 = x < 0.0f;
    bool gt_2pi = x > PI_2;
    return (x + PI_2)*lt_0 + x*(!lt_0)*(!gt_2pi) + (x - PI_2)*gt_2pi;
}/*}}}*/

__global__ void integrate(/*{{{*/
        // config
        unsigned int i_step, unsigned int n_node, unsigned int nh, unsigned int n_step, unsigned int n_params,
        float dt, float speed,
        float * __restrict__ rand_omega,
        float * __restrict__ weights,
        float * __restrict__ lengths,
        float * __restrict__ params_pwi, // pwi: per work item
        // state
        float * __restrict__ state_pwi,
        // outputs
        float * __restrict__ tavg_pwi
        )
{/*}}}*/

    // work id & size/*{{{*/
    const unsigned int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned int size = blockDim.x * gridDim.x * gridDim.y;/*}}}*/

    // ND array accessors (TODO autogen from py shape info)/*{{{*/
#define params(i_par) (params_pwi[(size * (i_par)) + id])
#define state(time, i_node) (state_pwi[((time) * n_node + (i_node))*size + id])
#define tavg(i_node) (tavg_pwi[((i_node) * size) + id])/*}}}*/

    // unpack params/*{{{*/
    const float coupling_value = params(1);
    const float speed_value = params(0);/*}}}*/

    // derived/*{{{*/
    const float rec_n = 1.0f / n_node;
    const float rec_speed_dt = 1.0f / speed_value / (dt);
    //const float omega = 60.0 * 2.0 * M_PI_F / 1e3;
    //const float omega = 1.0 * 2.0 * M_PI_F / 1e3;
    float omega = 0.0;
    const float sig = sqrt(dt) * sqrt(2.0 * 1e-5);/*}}}*/ //-->noise sigma value

    curandState s;
    curand_init(id * (blockDim.x * gridDim.x * gridDim.y), 0, 0, &s);


    for (unsigned int i_node = 0; i_node < n_node; i_node++)
        tavg(i_node) = 0.0f;

    for (unsigned int t = i_step; t < (i_step + n_step); t++)
    {

        for (unsigned int i_node = threadIdx.y; i_node < n_node; i_node+=blockDim.y)
        {
            if (i_node >= n_node) continue;

            float theta_i = state(t % NH, i_node);
            unsigned int i_n = i_node * n_node;
            float sum = 0.0f;

            for (unsigned int j_node = 0; j_node < n_node; j_node++)
            {
                float wij = weights[i_n + j_node]; // nb. not coalesced
                if (wij == 0.0)
                    continue;
                unsigned int dij = lengths[i_n + j_node] * rec_speed_dt;
                // int dij = 5;
                float theta_j = state((t - dij + NH) % NH, j_node);
                sum += wij * sin(theta_j - theta_i);
            } // j_node

            omega = rand_omega[i_node];

            theta_i += dt * (omega + coupling_value * rec_n * sum);
            theta_i += sig * curand_normal2(&s).x;
            theta_i = wrap_2_pi_(theta_i);
            state((t + 1) % NH, i_node) = theta_i;
            tavg(i_node) = sin(theta_i); //removed +=

            // sync across warps executing nodes for single sim, before going on to next time step
            __syncthreads();

        } // for i_node
    } // for t

// cleanup macros/*{{{*/
#undef params
#undef state
#undef tavg/*}}}*/

} // kernel integrate

