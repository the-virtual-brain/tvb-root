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

__device__ float wrap_it_x1(float x1)
{
    float x1dim[] = {-2.0, 1.0};
    if (x1 < x1dim[0]) x1 = x1dim[0];
    else if (x1 > x1dim[1]) x1 = x1dim[1];

    return x1;
}
__device__ float wrap_it_y1(float y1)
{
    float y1dim[] = {-20.0, 2.0};
    if (y1 < y1dim[0]) y1 = y1dim[0];
    else if (y1 > y1dim[1]) y1 = y1dim[1];

    return y1;
}
__device__ float wrap_it_z(float z)
{
    float zdim[] = {-2.0, 5.0};
    if (z < zdim[0]) z = zdim[0];
    else if (z > zdim[1]) z = zdim[1];

    return z;
}
__device__ float wrap_it_x2(float x2)
{
    float x2dim[] = {-2.0, 0.0};
    if (x2 < x2dim[0]) x2 = x2dim[0];
    else if (x2 > x2dim[1]) x2 = x2dim[1];

    return x2;
}
__device__ float wrap_it_y2(float y2)
{
    float y2dim[] = {0.0, 2.0};
    if (y2 < y2dim[0]) y2 = y2dim[0];
    else if (y2 > y2dim[1]) y2 = y2dim[1];

    return y2;
}
__device__ float wrap_it_g(float g)
{
    float gdim[] = {-1.0, 1.0};
    if (g < gdim[0]) g = gdim[0];
    else if (g > gdim[1]) g = gdim[1];

    return g;
}

__global__ void epileptorref(

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
#define state(time, i_node) (state_pwi[((time) * 6 * n_node + (i_node))*size + id])
#define tavg(i_node) (tavg_pwi[((i_node) * size) + id])

    // only threat those ID that have a corresponding parameters combination
    if (id >= size) return;

    // unpack params
    // These are the two parameters which are usually explore in fitting in this model
    const float global_speed = params(0);
    const float global_coupling = params(1);

    // regular constants
    const float a = 1.0;
    const float b = 3.0;
    const float c = 1.0;
    const float d = 5.0;
    const float r = 0.00035;
    const float s = 4.0;
    const float x0 = -1.6;
    const float Iext = 3.1;
    const float slope = 0.;
    const float Iext2 = 0.45;
    const float tau = 10.0;
    const float aa = 6.0;
    const float bb = 2.0;
    const float Kvf = 0.0;
    const float Kf = 0.0;
    const float Ks = 0.0;
    const float tt = 1.0;
    const float modification = 0;

    // coupling constants, coupling itself is hardcoded in kernel

    // coupling parameters
    float c_pop1 = 0.0;
    float c_pop2 = 0.0;

    // derived parameters
    const float rec_n = 1 / n_node;
    const float rec_speed_dt = powf(2.0f, global_speed) / dt;
    const float nsig = sqrt(dt) * sqrt(2.0 * 1e-5);

    // conditional_derived variable declaration
    float ydot0 = 0.0;
    float ydot2 = 0.0;
    float h = 0.0;
    float ydot4 = 0.0;


    float x1 = 0.0;
    float y1 = 0.0;
    float z = 0.0;
    float x2 = 0.0;
    float y2 = 0.0;
    float g = 0.0;

    float dx1 = 0.0;
    float dy1 = 0.0;
    float dz = 0.0;
    float dx2 = 0.0;
    float dy2 = 0.0;
    float dg = 0.0;

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
            c_pop2 = 0.0f;

            x1 = state((t) % nh, i_node + 0 * n_node);
            y1 = state((t) % nh, i_node + 1 * n_node);
            z = state((t) % nh, i_node + 2 * n_node);
            x2 = state((t) % nh, i_node + 3 * n_node);
            y2 = state((t) % nh, i_node + 4 * n_node);
            g = state((t) % nh, i_node + 5 * n_node);

            // This variable is used to traverse the weights and lengths matrix, which is really just a vector. It is just a displacement. /
            unsigned int i_n = i_node * n_node;

            for (unsigned int j_node = 0; j_node < n_node; j_node++)
            {
                //***// Get the weight of the coupling between node i and node j
                float wij = weights[i_n + j_node]; // nb. not coalesced
                if (wij == 0.0)
                    continue;

                // Get the delay between node i and node j
                unsigned int dij = lengths[i_n + j_node] * rec_speed_dt;

                //***// Get the state of node j which is delayed by dij
                float x1_j = state(((t - dij + nh) % nh), j_node + 0 * n_node);

                // Sum it all together using the coupling function. Kuramoto coupling: (postsyn * presyn) == ((a) * (sin(xj - xi))) 
                c_pop1 += wij * 1.0 * sin(x1_j - x1);

            } // j_node */

            // rec_n is used for the scaling over nodes
            c_pop1 *= global_coupling;
            c_pop2 *= g;

            // The conditional variables
            if (x1 < 0.0) {
                ydot0 = -a * powf(x1, 2) + b * x1;
            } else {
                ydot0 = slope - x2 + 0.6 * powf(z-4, 2);
            }
            if (z < 0.0) {
                ydot2 = - 0.1 * powf(z, 7);
            } else {
                ydot2 = 0;
            }
            if (modification) {
                h = x0 + 3. / (1. + exp(-(x1 + 0.5) / 0.1));
            } else {
                h = 4 * (x1 - x0) + ydot2;
            }
            if (x2 < -0.25) {
                ydot4 = 0.0;
            } else {
                ydot4 = aa * (x2 + 0.25);
            }

            // Integrate with stochastic forward euler
            dx1 = dt * (tt * (y1 - z + Iext + Kvf * c_pop1 + ydot0 ));
            dy1 = dt * (tt * (c - d * powf(x1, 2) - y1));
            dz = dt * (tt * (r * (h - z + Ks * c_pop1)));
            dx2 = dt * (tt * (-y2 + x2 - powf(x2, 3) + Iext2 + bb * g - 0.3 * (z - 3.5) + Kf * c_pop2));
            dy2 = dt * (tt * (-y2 + ydot4) / tau);
            dg = dt * (tt * (-0.01 * (g - 0.1 * x1) ));

            // No noise is added because it is not present in model
            x1 += dx1;
            y1 += dy1;
            z += dz;
            x2 += dx2;
            y2 += dy2;
            g += dg;

            // Wrap it within the limits of the model
            x1 = wrap_it_x1(x1);
            y1 = wrap_it_y1(y1);
            z = wrap_it_z(z);
            x2 = wrap_it_x2(x2);
            y2 = wrap_it_y2(y2);
            g = wrap_it_g(g);

            // Update the state
            state((t + 1) % nh, i_node + 0 * n_node) = x1;
            state((t + 1) % nh, i_node + 1 * n_node) = y1;
            state((t + 1) % nh, i_node + 2 * n_node) = z;
            state((t + 1) % nh, i_node + 3 * n_node) = x2;
            state((t + 1) % nh, i_node + 4 * n_node) = y2;
            state((t + 1) % nh, i_node + 5 * n_node) = g;

            // Update the observable only for the last timestep
            if (t == (i_step + n_step - 1)){
                tavg(i_node + 0 * n_node) = powf(x1, x2);
                tavg(i_node + 1 * n_node) = x2;
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
