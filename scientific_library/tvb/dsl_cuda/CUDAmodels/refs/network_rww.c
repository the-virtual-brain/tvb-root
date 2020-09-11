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


const float w_plus=1.4f;
const float a_E=310.0f;
const float b_E=125.0f;
const float d_E=0.154f;
const float a_I=615.0f;
const float b_I=177.0f;
const float d_I=0.087f;
const float gamma_E=0.641f / 1000.0f;
const float tau_E=100.0f;
const float tau_I=10.0f;
const float I_0=0.382f;
const float w_E=1.0f;
const float w_I=0.7f;
const float gamma_I= 1.0f / 1000.0f;
const float min_d_E = (-1.0f * d_E);
const float min_d_I = (-1.0f * d_I);
const float imintau_E = (-1.0f / tau_E);
const float imintau_I = (-1.0f / tau_I);
const float w_E__I_0 = (w_E * I_0);
const float w_I__I_0 = (w_I * I_0);

__global__ void integrate_wongwang(
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
    // const int i_step_dev = i_step;
    // const int n_node_dev = n_node;
    // work id & size
    const unsigned int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned int size = blockDim.x * gridDim.x * gridDim.y;

    // ND array accessors (TODO autogen from py shape info)
#define params(i_par) (params_pwi[(size * (i_par)) + id])
#define state(time, i_node) (state_pwi[((time) *2 * n_node + (i_node))*(size) + id])
#define tavg(i_node) (tavg_pwi[((i_node) * size) + id])

    // unpack params
    const float G = params(0);
    const float J_NMDA = 0.15;//params(0);
    const float JI= 1.0;
    const float G_J_NMDA = G*J_NMDA;
    // derived
    
    const float w_plus__J_NMDA = (w_plus * J_NMDA);
    const float sig = params(1);//0.001;//sqrt(dt) * sqrt(2.0 * 1e-3);
    // We have three variables which could be changed here. Actually 4
    // G (the global coupling), sigma (the noise), J_NMDA(the excitatory synaptic coupling) and J_i(the inner inhibition for each region)
    // For now we are making things simple and only change two parameters, G and J_NMDA.

    curandState s;
    curand_init(id + (unsigned int) clock64(), 0, 0, &s);
 
    double tmp_I_E;
    double tmp_H_E;
    double tmp_I_I;
    double tmp_H_I;
    double sum;
    double S_E = 0.0;
    double S_I = 0.0;
    

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
            S_E = state((t) % nh, i_node);
            S_I = state((t) % nh, i_node + n_node);
            for (unsigned int j_node = 0; j_node < n_node; j_node++)
            {
                //we are not considering delays in this model
                float wij = G_J_NMDA*weights[(i_node*n_node) + j_node]; // nb. not coalesced
                if (wij == 0.0)
                    continue;
                sum += wij * state((t) % nh, j_node); //of J
            }
            // external Input set to 0, no task evoked activity
            tmp_I_E = JI*S_I; // Inner inhibition set to 1
            tmp_I_E = sum - tmp_I_E ;    
            tmp_I_E = ((w_E__I_0)+(w_plus__J_NMDA * S_E)) + tmp_I_E ;
            tmp_I_E = a_E * tmp_I_E - b_E;
            tmp_H_E = tmp_I_E/(1.0-exp(min_d_E * tmp_I_E));
            tmp_I_I = (a_I*(((w_I__I_0)+(J_NMDA * S_E))-(S_I)))-b_I;
            tmp_H_I = tmp_I_I/(1.0-exp(min_d_I*tmp_I_I));

            S_E = (S_E)+(dt*(sig * curand_normal(&s)))+(dt*((imintau_E* S_E)+(tmp_H_E*((1-S_E)*gamma_E))));
            S_I = (S_I)+(dt*(sig * curand_normal(&s)))+(dt*((imintau_I* S_I)+(tmp_H_I*gamma_I)));
            if(S_E>1) S_E = 1;
            if(S_I>1) S_I = 1;
            if(S_E<0) S_E = 0;
            if(S_I<0) S_I = 0;
            state((t+1) % nh, i_node) = S_E;
            state((t+1) % nh, i_node+(n_node)) = S_I;
            tavg(i_node) = S_E;

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
