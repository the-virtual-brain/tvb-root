#include <stdio.h> // for printf
#define PI_2 (2 * M_PI_F)
#define PI M_PI_F
#define INF INFINITY

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

__device__ float wrap_it_r(float r)
{
    float rdim[] = {0.0, inf};
    if (r < rdim[0]) r = rdim[0];
    else if (r > rdim[1]) r = rdim[1];

    return r;
}

__global__ void montbrio_heun(

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
    const float nsig = params(0);
    const float global_coupling = params(1);

    // regular constants
    const float tau = 1.0;
    const float I = 0.0;
    const float Delta = 0.7;
    const float J = 14.5;
    const float eta = -4.6;
    const float Gamma = 5.0;
    const float cr = 1.0;
    const float cv = 1.0;

    // coupling constants, coupling itself is hardcoded in kernel

    // coupling parameters
    float c_pop0 = 0.0;
    float c_pop1 = 0.0;

    // derived parameters
    const float rec_n = 1 / n_node;
//    const float rec_speed_dt = 1.0f / global_speed / dt;
    const float rec_speed_dt = 1.0f / 1.0f / dt;
    float nsig_r = 0.01;
    float nsig_V = 0.02;

    curandState crndst;
//    curand_init(42, 0, 0, &crndst);
    curand_init(id + (unsigned int) clock64(), 0, 0, &crndst);
    float noise_r = 0.0;
    float noise_V = 0.0;
    float sqrtdt = 0.0;
    float sqrtnsig = 0.0;

    float r = 0.0;
    float V = 0.0;
    float dr_i = 0.0;
    float dV_i = 0.0;
    float dr = 0.0;
    float dV = 0.0;

    unsigned int dij_i = 0;
    float dij = 0.0;
    float wij = 0.0;

    float r_j = 0.0;
    float V_j = 0.0;

    //***// This is only initialization of the observable
    for (unsigned int i_node = 0; i_node < n_node; i_node++)
    {
        tavg(i_node) = 0.0f;
        if (i_step == 0){
            state(i_step, i_node) = 0.0f;
        }
    }

    //***// This is the loop over time, should stay always the same
    for (unsigned int t = i_step; t < (i_step + n_step); t++)
    {
    //***// This is the loop over nodes, which also should stay the same
        for (int i_node = 0; i_node < n_node; i_node++)
        {
            c_pop0 = 0.0f;
            c_pop1 = 0.0f;

            if (t == (i_step)){
                tavg(i_node + 0 * n_node) = 0;
                tavg(i_node + 1 * n_node) = 0;
            }

            r = state((t) % nh, i_node + 0 * n_node);
            V = state((t) % nh, i_node + 1 * n_node);

            // This variable is used to traverse the weights and lengths matrix, which is really just a vector. It is just a displacement. /
            unsigned int i_n = i_node * n_node;

            for (unsigned int j_node = 0; j_node < n_node; j_node++)
            {
                //***// Get the weight of the coupling between node i and node j
                float wij = weights[i_n + j_node]; // nb. not coalesced
                if (wij == 0.0)
                    continue;

                // Get the delay between node i and node j
                dij = lengths[i_n + j_node] * rec_speed_dt;
                dij = dij + 0.5;
                dij_i = (int)dij;

                //***// Get the state of node j which is delayed by dij
                r_j = state(((t - dij_i + nh) % nh), j_node + 0 * n_node);
                V_j = state(((t - dij_i + nh) % nh), j_node + 1 * n_node);

                // Sum it all together using the coupling function. Kuramoto coupling: (postsyn * presyn) == ((a) * (sin(xj - xi))) 
                c_pop0 += wij * 1 * r_j;

                c_pop1 += wij * 1 * V_j;
            } // j_node */

            // global coupling handling, rec_n used to scale nodes
            c_pop0 *= global_coupling;
            c_pop1 *= global_coupling;

            // Integrate with Heun (2nd order)
            dr = 1/tau * (Delta / (pi * tau) + 2 * V * r);
            dV = 1/tau * (powf(V, 2) - powf(pi, 2) * powf(tau, 2) * powf(r, 2) + eta + J * tau * r + I + cr * c_pop0 + cv * c_pop1);

            // additive white noise generation in tvb (random_stream.normal(size=shape));
            sqrtdt = sqrt(dt);
            noise_r = sqrtdt * curand_normal(&crndst);
            noise_V = sqrtdt * curand_normal(&crndst);
            // gfun in tvb
            noise_r *= sqrt(2.0 * nsig);
            noise_V *= sqrt(2.0 * nsig * 2.0);

            dr_i = r + dr * dt + noise_r;
            dV_i = V + dV * dt + noise_V;
//            dr_i = r + dr * dt;
//            dV_i = V + dV * dt;

            // Wrap it within the limits of the model
            dr_i = wrap_it_r(dr_i);

            dr = dt/2.0 * (dr + (1/tau * (Delta / (pi * tau) + 2 * dV_i * dr_i)));
            dV = dt/2.0 * (dV + (1/tau * (powf(dV_i, 2) - powf(pi, 2) * powf(tau, 2) * powf(dr_i, 2) + eta + J * tau * dr_i + I + cr * c_pop0 + cv * c_pop1)));

            // No noise is added because it is not present in model
            r += dr + noise_r;
            V += dV + noise_V;
//            r += dr;
//            V += dV;

            // Wrap it within the limits of the model
            r = wrap_it_r(r);

            // Update the state
            state((t + 1) % nh, i_node + 0 * n_node) = r;
            state((t + 1) % nh, i_node + 1 * n_node) = V;

            // Update the observable
            tavg(i_node + 0 * n_node) += r/n_step;
            tavg(i_node + 1 * n_node) += V/n_step;

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
