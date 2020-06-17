#include <stdio.h> // for printf
#include <curand_kernel.h>
#include <curand.h>

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

#ifdef RAND123/*{{{*/
#include "Random123/threefry.h"
#include "Random123/boxmuller.hpp"

struct rng_state
{
    threefry4x32_ctr_t ctr;
    threefry4x32_key_t key;
    long long int seed;
    float out[4];
};

__device__ void rng_gen_normal(struct rng_state *r)
{
    threefry4x32_ctr_t result;
    r123::float2 normal;

    ++r->ctr.v[0];

    result = threefry4x32(r->ctr, r->key);

    normal = r123::boxmuller(result.v[0], result.v[1]);

    r->out[0] = normal.x;
    r->out[1] = normal.y;

    normal = r123::boxmuller(result.v[2], result.v[3]);
    r->out[2] = normal.x;
    r->out[3] = normal.y;
}

__device__ void rng_init(struct rng_state *r, int seed1, int seed2)
{
    r->ctr[0] = 0;
    r->ctr[1] = 0;
    r->ctr[2] = seed1;
    r->ctr[3] = seed2;
}

__device__ float rng_next_normal(struct rng_state *r)
{
    const int count = r->ctr.v[0] % 4;
    if (count == 0)
        rng_gen_normal(r);
    return r->out[count];
}
#endif //RANDOM123/*}}}*/

#ifdef CURAND/*{{{*/
#include <curand_kernel.h>
#include <curand.h>
#endif //CURAND/*}}}*/

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
    const float rec_speed_dt = 1.0f / speed_value / dt;
    const float omega = 10.0 * 2.0 * M_PI_F / 1e3;
    const float sig = sqrt(dt) * sqrt(2.0 * 1e-5);/*}}}*/

    curandState s;
    curand_init(id * (blockDim.x * gridDim.x * gridDim.y), 0, 0, &s);

    for (unsigned int i_node = 0; i_node < n_node; i_node++)
        tavg(i_node) = 0.0f;

    for (unsigned int t = i_step; t < (i_step + n_step); t++) {
        for (unsigned int i_node = threadIdx.y; i_node < n_node; i_node+=blockDim.y) {
            if(i_node >= n_node) continue;

            float theta_i = state(t % NH, i_node);
            unsigned int i_n = i_node * n_node;
            float sum = 0.0f;

            for (unsigned int j_node = 0; j_node < n_node; j_node++) {
                float wij = weights[i_n + j_node]; // nb. not coalesced
                if(lengths[i_n + j_node]>0 && i_node>0 && threadIdx.x == 0)
                    // printf("%d %d %d %f\n", t, i_n + j_node, i_node, lengths[i_n + j_node]);
                
                if (wij == 0.0) continue;
                unsigned int dij = lengths[i_n + j_node] * rec_speed_dt;
                unsigned time = (t - dij + NH) % NH;
                float theta_j = state_pwi[(time * n_node + j_node)*size + id];
                sum += wij * sin(theta_j - theta_i);
            } // j_node
            theta_i += dt * (omega + coupling_value * rec_n * sum);

            theta_i += sig * curand_normal2(&s).x;

            theta_i = wrap_2_pi(theta_i);
            state((t + 1) % NH, i_node) = theta_i;
            tavg(i_node) += sin(theta_i);

            // sync across warps executing nodes for single sim, before going on to next time step
            __syncthreads();

        } // for i_node
    } // for t

// cleanup macros/*{{{*/
#undef params
#undef state
#undef tavg/*}}}*/

} // kernel integrate

// const float w_plus=1.4f;
// const float a_E=310.0f;
// const float b_E=125.0f;
// const float d_E=0.16f;
// const float a_I=615.0f;
// const float b_I=177.0f;
// const float d_I=0.087f;
// // const float gamma_E=0.641f / 1000.0f;
// // const float tau_E=100.0f;
// // const float tau_I=10.0f;
// const float I_0=0.382f;
// const float w_E=1.0f;
// const float w_I=0.7f;
// // const float gamma_I= 1.0f / 1000.0f;
// const float min_d_E = (-1.0f * d_E);
// const float min_d_I = (-1.0f * d_I);
// // const float imintau_E = (-1.0f / tau_E);
// // const float imintau_I = (-1.0f / tau_I);
// const float w_E__I_0 = (w_E * I_0);
// const float w_I__I_0 = (w_I * I_0);

// __global__ void integrate_wongwang(
//         // config
//         unsigned int i_step, unsigned int n_node, unsigned int nh, unsigned int n_step, unsigned int n_params,
//         float dt, float speed,
//         float * weights,
//         float * lengths,
//         float * params_pwi, // pwi: per work item
//         // state
//         float * state_pwi,
//         // outputs
//         float * tavg_pwi
//         )
// {
//     // const int i_step_dev = i_step;
//     // const int n_node_dev = n_node;
//     // work id & size
//     const unsigned int id = (blockIdx.x * blockDim.x) + threadIdx.x;
//     const unsigned int size = blockDim.x * gridDim.x;

//     // ND array accessors (TODO autogen from py shape info)
// #define params(i_par) (params_pwi[(size * (i_par)) + id])
// #define state(time, i_node) (state_pwi[((time) *2 * n_node + (i_node))*(size) + id])
// #define tavg(i_node) (tavg_pwi[((i_node) * size) + id])

//     // unpack params
//     const float G = params(1);
//     const float J_NMDA = params(0);
//     const float G_J_NMDA = G*J_NMDA;
//     // derived
    
//     const float w_plus__J_NMDA = (w_plus * J_NMDA);
//     const float sig = sqrt(dt) * sqrt(2.0 * 1e-5);
//     // We have three variables which could be changed here. Actually 4
//     // G (the global coupling), sigma (the noise), J_NMDA(the excitatory synaptic coupling) and J_i(the inner inhibition for each region)
//     // For now we are making things simple and only change two parameters, G and J_NMDA.
// #ifdef RAND123
//     // rng
//     struct rng_state rng;
//     rng_init(&rng, id, i_step);
// #endif
// #ifdef CURAND
//     curandState s;
//     curand_init(id * (blockDim.x * gridDim.x), 0, 0, &s);
// #endif
//     float tmp_I_E;
//     float tmp_H_E;
//     float tmp_I_I;
//     float tmp_H_I;
//     float sum;
    

//     for (unsigned int i_node = 0; i_node < n_node; i_node++)
//         tavg(i_node) = 0.0f;

//     for (unsigned int t = i_step; t < (i_step + n_step); t++)
//     {
//         for (unsigned int i_node = 0; i_node < n_node; i_node++)
//         {
//             sum = 0.0f;
//             float S_E = state((t) % nh, i_node);
//             float S_I = state((t) % nh, i_node + n_node);
//             for (unsigned int j_node = 0; j_node < n_node; j_node++)
//             {
//                 //we are not considering delays in this model
//                 float wij = G_J_NMDA*weights[(i_node*n_node) + j_node]; // nb. not coalesced
//                 if (wij == 0.0)
//                     continue;
//                 sum += wij * state((t) % nh, j_node); //of J
//             }
//             // external Input set to 0, no task evoked activity
//             tmp_I_E = S_I; // Inner inhibition set to 1
//             tmp_I_E = sum - tmp_I_E ;    
//             tmp_I_E = ((w_E__I_0)+(w_plus__J_NMDA * S_E)) + tmp_I_E ;
//             tmp_I_E = a_E * tmp_I_E ;
//             tmp_I_E = tmp_I_E - b_E;
//             tmp_I_E = (a_E * (((w_E__I_0)+(w_plus__J_NMDA * S_E))+( sum-(S_I))))-b_E;
//             tmp_H_E = tmp_I_E/(1-expf(min_d_E * tmp_I_E));
//             //meanFR[i] += tmp_H_E; Not storing mean firing rate
//             // r_Edd_i = tmp_H_E; not observing the firing rate for now
//             tmp_I_I = (a_I*(((w_I__I_0)+(J_NMDA * S_E))-( S_I)))-b_I;
//             tmp_H_I = tmp_I_I/(1-expf(min_d_I*tmp_I_I));
//             // r_I[i] = tmp_H_I; not observing the firing rate for now

// #ifdef RAND123
//             S_E = ((sig * rng_next_normal(&rng))+S_E)+(dt*((imintau_E* S_E)+(tmp_H_E*((1-S_E)*gamma_E))));
//             S_I = ((sig * rng_next_normal(&rng))+S_I)+(dt*((imintau_I* S_I)+(tmp_H_I*gamma_I)));
// #endif
// #ifdef CURAND
//             S_E = ((sig * curand_normal2(&s).x)+S_E)+(dt*((imintau_E* S_E)+(tmp_H_E*((1-S_E)*gamma_E))));
//             S_I = ((sig * curand_normal2(&s).x)+S_I)+(dt*((imintau_I* S_I)+(tmp_H_I*gamma_I)));
// #endif
//             state((t+1) % nh, i_node) = S_E;
//             state((t+1) % nh, i_node+(n_node)) = S_I;
//             tavg(i_node) += S_E + S_I;
//         } // for i_node
//     } // for t
// } // kernel integrate
// vim: sw=4 sts=4 ts=8 et ai
