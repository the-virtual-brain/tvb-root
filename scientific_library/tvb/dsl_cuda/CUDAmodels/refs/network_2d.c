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


const float tau = 1.0;
const float I = 0.0;
const float a = -2.0;
const float b = -10.0;
const float c = 0.0;
const float d = 0.02;
const float e = 3.0;
const float f = 1.0;
const float g = 0.0;
const float beta = 1.0;
const float alpha = 1.0;
const float gam = 1.0;


__global__ void integrate_2d(
        // config
        unsigned int i_step, unsigned int n_node, unsigned int nh, unsigned int n_step, unsigned int n_params,
        float dt, float speed,
        float * weights,
        float * lengths,
        float * params_pwi, // pwi: per work item
        // state
        float * state_pwi,
        // outputs
        float * tavg_pwi
        )
{
    // work id & size
    const unsigned int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned int size = blockDim.x * gridDim.x * gridDim.y;

    // ND array accessors (TODO autogen from py shape info)
#define params(i_par) (params_pwi[(size * (i_par)) + id])
#define state(time, i_node) (state_pwi[((time) *2 * n_node + (i_node))*(size) + id])
#define tavg(i_node) (tavg_pwi[((i_node) * size) + id])

    // unpack params
    

    // derived
    const float sig = 0.0001; //params(0);//0.001;//sqrt(dt) * sqrt(2.0 * 1e-3);
    const float rec_speed_dt = params(0);
    const float G = params(1);
    const float lc = 0.0;
    
    curandState s;
    curand_init(id + (unsigned int) clock64(), 0, 0, &s);
 
    double derivV = 0.0;
    double derivW = 0.0;
    double V = 0.0;
    double W = 0.0;
   
    double sum = 0.0;
    float wij = 0.0f;
    float V_j = 0.0;
    unsigned int dij = 0;
    

    for (unsigned int i_node = 0; i_node < n_node; i_node++){
        tavg(i_node) = 0.0f;
        if (i_step == 0){
            state(i_step, i_node) = 0.001;
        }
    }

    for (unsigned int t = i_step; t < (i_step + n_step); t++)
    {
        for (unsigned int i_node = 0; i_node < n_node; i_node++)
        {
            sum = 0.0f;
            V = state((t) % nh, i_node);
            W = state((t) % nh, i_node + n_node);
            for (unsigned int j_node = 0; j_node < n_node; j_node++)
            {
                float wij = weights[(i_node*n_node) + j_node]; // nb. not coalesced
                if (wij == 0.0)
                    continue;
                dij = lengths[(i_node*n_node) + j_node] * rec_speed_dt;
                V_j = state((t - dij + NH) % NH, j_node);
                sum += wij * sin(V_j - V);
            }
            sum = G*sum; // Global coupling

            derivV = d * tau * (alpha * W - f * powf(V,3) + e * powf(V,2) + g * V + gam * I + gam * sum + lc * V);
            derivW = d * (a + b * V + c * powf(V,2) - beta * W) / tau;
            V = (V)+(dt*(sig * curand_normal(&s)))+((derivV));
            W = (W)+(dt*(sig * curand_normal(&s)))+((derivW));
            if(V>4) V = 4;
            if(W>6) W = 6;
            if(V<-2) V = -2;
            if(W<-6) W = -6;
            state((t+1) % nh, i_node) = V;
            state((t+1) % nh, i_node+(n_node)) = W;
            tavg(i_node) = V;

            // sync across warps executing nodes for single sim, before going on to next time step
            __syncthreads();
        } // for i_node
    } // for t
    // cleanup macros/*{{{*/
    #undef params
    #undef state
    #undef tavg/*}}}*/
} // kernel integrate
// vim: sw=4 sts=4 ts=8 et ai
