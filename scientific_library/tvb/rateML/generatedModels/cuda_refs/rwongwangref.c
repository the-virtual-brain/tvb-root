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
#include <stdbool.h>

__device__ float wrap_it_V(float V)
{
    float Vdim[] = {0.0, 1.0};
    if (V < Vdim[0]) V = Vdim[0];
    else if (V > Vdim[1]) V = Vdim[1];

    return V;
}
__device__ float wrap_it_W(float W)
{
    float Wdim[] = {0.0, 1.0};
    if (W < Wdim[0]) W = Wdim[0];
    else if (W > Wdim[1]) W = Wdim[1];

    return W;
}

__global__ void rwongwangref(

        // config
        unsigned int i_step, unsigned int n_node, unsigned int nh, unsigned int n_step, unsigned int n_work_items,
        float dt, float * __restrict__ weights, float * __restrict__ lengths,
        float * __restrict__ params_pwi, // pwi: per work item
        // state
        float * __restrict__ state_pwi,
        // outputs
        float * __restrict__ tavg_pwi
        )
{
    // work id & size
    const unsigned int id = (gridDim.x * blockDim.x * threadIdx.y) + threadIdx.x;
    const unsigned int size = n_work_items;

#define params(i_par) (params_pwi[(size * (i_par)) + id])
#define state(time, i_node) (state_pwi[((time) * 2 * n_node + (i_node))*size + id])
#define tavg(i_node) (tavg_pwi[((i_node) * size) + id])

    // only threat those ID that have a corresponding parameters combination
    if (id >= size) return;

    // unpack params
    // These are the two parameters which are usually explore in fitting in this model
    const float global_speed = params(0);
    const float global_coupling = params(1);

    // regular constants
    const float w_plus = 1.4;
    const float a_E = 310.0;
    const float b_E = 125.0;
    const float d_E = 0.154;
    const float a_I = 615.0;
    const float b_I = 177.0;
    const float d_I = 0.087;
    const float gamma_E = 0.641 / 1000.0;
    const float tau_E = 100.0;
    const float tau_I = 10.0;
    const float I_0 = 0.382;
    const float w_E = 1.0;
    const float w_I = 0.7;
    const float gamma_I = 1.0 / 1000.0;
    const float J_N = 0.15;
    const float J_I = 1.0;
    const float G = 2.0;
    const float lamda = 0.0;
    const float J_NMDA = 0.15;
    const float JI = 1.0;

    // coupling constants, coupling itself is hardcoded in kernel
    const float a = 1;

    // coupling parameters
    float c_pop1 = 0.0;

    // derived parameters
    const float rec_n = 1 / n_node;
    const float rec_speed_dt = 0;
    const float nsig = sqrt(dt) * sqrt(2.0 * 1e-5);
    // the dynamic derived variables declarations
    float min_d_E = 0.0;
    float min_d_I = 0.0;
    float imintau_E = 0.0;
    float imintau_I = 0.0;
    float w_E__I_0 = 0.0;
    float w_I__I_0 = 0.0;
    float G_J_NMDA = 0.0;
    float w_plus__J_NMDA = 0.0;
    float tmp_I_E = 0.0;
    float tmp_H_E = 0.0;
    float tmp_I_I = 0.0;
    float tmp_H_I = 0.0;



    float V = 0.0;
    float W = 0.0;

    float dV = 0.0;
    float dW = 0.0;

    //***// This is only initialization of the observable
    for (unsigned int i_node = 0; i_node < n_node; i_node++)
    {
        tavg(i_node) = 0.0f;
        if (i_step == 0){
            state(i_step, i_node) = 0.001;
        }
    }

    //***// This is the loop over time, should stay always the same
    for (unsigned int t = i_step; t < (i_step + n_step); t++)
    {
    //***// This is the loop over nodes, which also should stay the same
        for (int i_node = 0; i_node < n_node; i_node++)
        {
            c_pop1 = 0.0f;

            V = state((t) % nh, i_node + 0 * n_node);
            W = state((t) % nh, i_node + 1 * n_node);

            // This variable is used to traverse the weights and lengths matrix, which is really just a vector. It is just a displacement. /
            unsigned int i_n = i_node * n_node;

            for (unsigned int j_node = 0; j_node < n_node; j_node++)
            {
                //***// Get the weight of the coupling between node i and node j
                float wij = weights[i_n + j_node]; // nb. not coalesced
                if (wij == 0.0)
                    continue;

                // no delay specified
                unsigned int dij = 0;

                //***// Get the state of node j which is delayed by dij
                float V_j = state(((t - dij + nh) % nh), j_node + 0 * n_node);

                // Sum it all together using the coupling function. Kuramoto coupling: (postsyn * presyn) == ((a) * (sin(xj - xi))) 
                c_pop1 += wij * a * V_j * G_J_NMDA;

            } // j_node */

            // rec_n is used for the scaling over nodes
            c_pop1 *= powf(V, 2);
            // the dynamic derived variables
            min_d_E = -1.0 * d_E;
            min_d_I = -1.0 * d_I;
            imintau_E = -1.0 / tau_E;
            imintau_I = -1.0 / tau_I;
            w_E__I_0 = w_E * I_0;
            w_I__I_0 = w_I * I_0;
            G_J_NMDA = G*J_NMDA;
            w_plus__J_NMDA = w_plus * J_NMDA;
            tmp_I_E = a_E * (w_E__I_0 + w_plus__J_NMDA * V + c_pop1 - JI*W) - b_E;
            tmp_H_E = tmp_I_E/(1.0-exp(min_d_E * tmp_I_E));
            tmp_I_I = (a_I*((w_I__I_0+(J_NMDA * V))-W))-b_I;
            tmp_H_I = tmp_I_I/(1.0-exp(min_d_I*tmp_I_I));


            // Integrate with stochastic forward euler
            dV = dt * ((imintau_E* V)+(tmp_H_E*(1-V)*gamma_E));
            dW = dt * ((imintau_I* W)+(tmp_H_I*gamma_I));

            // No noise is added because it is not present in model
            V += dV;
            W += dW;

            // Wrap it within the limits of the model
            V = wrap_it_V(V);
            W = wrap_it_W(W);

            // Update the state
            state((t + 1) % nh, i_node + 0 * n_node) = V;
            state((t + 1) % nh, i_node + 1 * n_node) = W;

            // Update the observable only for the last timestep
            if (t == (i_step + n_step - 1)){
                tavg(i_node + 0 * n_node) = V;
                tavg(i_node + 1 * n_node) = W;
            }

            // sync across warps executing nodes for single sim, before going on to next time step
            __syncthreads();

        } // for i_node
    } // for t

// cleanup macros/*{{{*/
#undef params
#undef state
#undef tavg/*}}}*/

} // kernel integrate

// defaults from Stefan 2007, cf tvb/analyzers/fmri_balloon.py
#define TAU_S 0.65f
#define TAU_F 0.41f
#define TAU_O 0.98f
#define ALPHA 0.32f
#define TE 0.04f
#define V0 4.0f
#define E0 0.4f
#define EPSILON 0.5f
#define NU_0 40.3f
#define R_0 25.0f

#define RECIP_TAU_S (1.0f / TAU_S)
#define RECIP_TAU_F (1.0f / TAU_F)
#define RECIP_TAU_O (1.0f / TAU_O)
#define RECIP_ALPHA (1.0f / ALPHA)
#define RECIP_E0 (1.0f / E0)

// "derived parameters"
#define k1 (4.3f * NU_0 * E0 * TE)
#define k2 (EPSILON * R_0 * E0 * TE)
#define k3 (1.0f - EPSILON)

__global__ void bold_update(int n_node, float dt,
                      // bold.shape = (4, n_nodes, n_threads)
            float * __restrict__ bold_state,
                      // nrl.shape = (n_nodes, n_threads)
            float * __restrict__ neural_state,
                      // out.shape = (n_nodes, n_threads)
            float * __restrict__ out)
{
    const unsigned int it = (gridDim.x * blockDim.x * threadIdx.y) + threadIdx.x;
    const unsigned int nt = blockDim.x * blockDim.y * gridDim.x * gridDim.y;

    int var_stride = n_node * nt;
    for (int i_node=0; i_node < n_node; i_node++)
    {
        float *node_bold = bold_state + i_node * nt + it;

        float s = node_bold[0 * var_stride];
        float f = node_bold[1 * var_stride];
        float v = node_bold[2 * var_stride];
        float q = node_bold[3 * var_stride];

        float x = neural_state[i_node * nt + it];

        float ds = x - RECIP_TAU_S * s - RECIP_TAU_F * (f - 1.0f);
        float df = s;
        float dv = RECIP_TAU_O * (f - pow(v, RECIP_ALPHA));
        float dq = RECIP_TAU_O * (f * (1.0f - pow(1.0f - E0, 1.0f / f))
                * RECIP_E0 - pow(v, RECIP_ALPHA) * (q / v));

        s += dt * ds;
        f += dt * df;
        v += dt * dv;
        q += dt * dq;

        node_bold[0 * var_stride] = s;
        node_bold[1 * var_stride] = f;
        node_bold[2 * var_stride] = v;
        node_bold[3 * var_stride] = q;

        out[i_node * nt + it] = V0 * (    k1 * (1.0f - q    )
                                        + k2 * (1.0f - q / v)
                                        + k3 * (1.0f -     v) );
    } // i_node
} // kernel
