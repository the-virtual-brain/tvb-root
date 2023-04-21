/*

This is the partly rateML generated and partly manual implementation of the Zerlaut [1] HPC GPU  model for parameters
exploration of: global_coupling, b_e, E_L_e, E_L_i and T as in [2]. A Jyputer notebook can be found here:
https://lab.ch.ebrains.eu/hub/user-redirect/lab/tree/shared/Public_3Species_TVBAdEx_EITN_FallSchool/Human/Different_brain_states_simulated_in_the_human_brain.ipynb

[1] Zerlaut, Yann, Sandrine Chemla, Frederic Chavane, and Alain Destexhe. “Modeling Mesoscopic Cortical Dynamics Using
a Mean-Field Model of Conductance-Based Networks of Adaptive Exponential Integrate-and-Fire Neurons.”
Journal of Computational Neuroscience 44, no. 1 (February 1, 2018): 45–61. https://doi.org/10.1007/s10827-017-0668-2.

[2]  A comprehensive neural simulation of slow-wave sleep and highly responsive wakefulness dynamics
Jennifer S. Goldman, Lionel Kusch, Bahar Hazal Yalçinkaya, Damien Depannemaecker, Trang-Anh E. Nghiem, Viktor Jirsa, Alain Destexhe
bioRxiv 2021.08.31.458365; doi: https://doi.org/10.1101/2021.08.31.458365

.. moduleauthor:: Michiel. A. van der Vlag <m.van.der.vlag@fz-juelich.de>

*/

#include <stdio.h> // for printf
#define PI_2 (2 * M_PI_F)
#define PI M_PI_F
#define INF INFINITY
#define SQRT2 1.414213562

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


//float b_e = 60.0;
//float E_L_e = -63.0;
//float E_L_i = -65.0;
//float T = 40.0;

// regular global constants
const float g_L = 10.0;
const float C_m = 200.0;
const float a_e = 0.0;
const float b_i = 0.0;
const float a_i = 0.0;
const float tau_w_e = 500.0;
const float tau_w_i = 1.0;
const float E_e = 0.0;
const float E_i = -80.0;
const float Q_e = 1.5;
const float Q_i = 5.0;
const float tau_e = 5.0;
const float tau_i = 5.0;
const float N_tot = 10000;
const float p_connect_e = 0.05;
const float p_connect_i = 0.05;
const float g = 0.2;
const float P_e0 = -0.0498;
const float P_e1 = 0.00506;
const float P_e2 = -0.025;
const float P_e3 = 0.0014;
const float P_e4 = -0.00041;
const float P_e5 = 0.0105;
const float P_e6 = -0.036;
const float P_e7 = 0.0074;
const float P_e8 = 0.0012;
const float P_e9 = -0.0407;

const float P_i0 = -0.0514;
const float P_i1 = 0.004;
const float P_i2 =  -0.0083;
const float P_i3 = 0.0002;
const float P_i4 = -0.0005;
const float P_i5 = 0.0014;
const float P_i6 = -0.0146;
const float P_i7 = 0.0045;
const float P_i8 = 0.0028;
const float P_i9 = -0.0153;

const float external_input_ex_ex = 0.315*1e-3;
const float external_input_ex_in = 0.000;
const float external_input_in_ex = 0.315*1e-3;
const float external_input_in_in = 0.000;

const float tau_OU = 5.0;
const float weight_noise = 1e-4;

const float K_ext_e = 400;
const float K_ext_i = 0;

#include "zerlaut_func.h"

__global__ void zerlaut(

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
#define state(time, i_node) (state_pwi[((time) * 7 * n_node + (i_node))*size + id])
#define tavg(i_node) (tavg_pwi[((i_node) * size) + id])

    // only threat those ID that have a corresponding parameters combination
    if (id >= size) return;

    // unpack params
    // The parameters explored
    const float global_coupling = params(0);
    const float b_e = params(1);
    const float E_L_e = params(2);
    const float E_L_i = params(3);
    const float T = params(4);


    // for function pointers
    float mu_V;
	float sigma_V;
	float T_V;

    // coupling constants, coupling itself is hardcoded in kernel
    const float c_a = 1;

    // coupling parameters
    float c_pop0 = 0.0;
    float global_speed = 4.0;

    // derived parameters
    const float rec_n = 1 / n_node;
    const float rec_speed_dt = 1.0f / global_speed / (dt);
//    const float nsig = sqrt(dt) * sqrt(2.0 * 1e-3);
//    const float local_coupling = 0.0;

    // the dynamic derived variables declarations
    float E_input_excitatory = 0.0;
    float E_input_inhibitory = 0.0;
    float I_input_excitatory = 0.0;
    float I_input_inhibitory = 0.0;

    float N_e = N_tot * (1-global_coupling);
    float N_i = N_tot * global_coupling;

    float E = 0.0;
    float I = 0.0;
    float C_ee = 0.0;
    float C_ei = 0.0;
    float C_ii = 0.0;
    float W_e = 100.0;
    float W_i = 0.0;
    float noise = 0.0;

    float dE = 0.0;
    float dI = 0.0;
    float dC_ee = 0.0;
    float dC_ei = 0.0;
    float dC_ii = 0.0;
    float dW_e = 0.0;
    float dW_i = 0.0;
    float dnoise = 0.0;

    unsigned int dij_i = 0;
    float dij = 0.0;
    float wij = 0.0;

    float V_j = 0.0;

    float df = 1e-7;
    float powdf = powf((df*1e3), 2);
    float lc_E = 0;
    float lc_I = 0;

    curandState crndst;
    curand_init(id + (unsigned int) clock64(), 0, 0, &crndst);

    //***// This is only initialization of the observable
    for (unsigned int i_node = 0; i_node < n_node; i_node++)
    {

//        tavg(i_node) = 0.0f;
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

            if (t == (i_step)){
                tavg(i_node + 0 * n_node) = 0;
                tavg(i_node + 1 * n_node) = 0;
                tavg(i_node + 2 * n_node) = 0;
                tavg(i_node + 3 * n_node) = 0;
                tavg(i_node + 4 * n_node) = 0;
                tavg(i_node + 5 * n_node) = 0;
                tavg(i_node + 6 * n_node) = 0;
            }

            E = state((t) % nh, i_node + 0 * n_node);
            I = state((t) % nh, i_node + 1 * n_node);
            C_ee = state((t) % nh, i_node + 2 * n_node);
            C_ei = state((t) % nh, i_node + 3 * n_node);
            C_ii = state((t) % nh, i_node + 4 * n_node);
            if (t!=0) W_e = state((t) % nh, i_node + 5 * n_node);
            W_i = state((t) % nh, i_node + 6 * n_node);
            noise = state((t) % nh, i_node + 7 * n_node);

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
                V_j = state(((t - dij_i + nh) % nh), j_node + 0 * n_node);

                // Sum it all together using the coupling function.
            } // j_node */

            // global coupling handling
            c_pop0 *= global_coupling;
            // the dynamic derived variables declarations
            E_input_excitatory = c_pop0+lc_E+external_input_ex_ex + weight_noise * noise;
            E_input_inhibitory = c_pop0+lc_E+external_input_in_ex + weight_noise * noise;

            // The conditional variables
            if (E_input_excitatory < 0.0) E_input_excitatory = 0.0;
            if (E_input_inhibitory < 0.0) E_input_inhibitory = 0.0;
            I_input_excitatory = lc_I+external_input_ex_in;
            I_input_inhibitory = lc_I+external_input_in_in;

            // Transfer function of excitatory and inhibitory neurons
            float _TF_e = TF_excitatory(E, I, E_input_excitatory, I_input_excitatory, W_e, E_L_e);
            float _TF_i = TF_inhibitory(E, I, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i);

            float _diff_fe_e = (TF_excitatory(E+df, I, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                               -TF_excitatory(E-df, I, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                               )/(2*df*1e3);

            float _diff_fe_i = (TF_inhibitory(E+df, I, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
                               -TF_inhibitory(E-df, I, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
                               )/(2*df*1e3);

            float _diff_fi_e = (TF_excitatory(E, I+df, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                               -TF_excitatory(E, I-df, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                               )/(2*df*1e3);

            float _diff_fi_i = (TF_inhibitory(E, I+df, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
                               -TF_inhibitory(E, I-df, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
                               )/(2*df*1e3);

            float _diff2_fe_fe_e = (TF_excitatory(E+df, I, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                                   -2*_TF_e
                                   +TF_excitatory(E-df, I, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                                   )/powdf;

            float _diff2_fe_fe_i = (TF_inhibitory(E+df, I, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
                                   -2*_TF_i
                                   +TF_inhibitory(E-df, I, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
                                   )/powdf;

            // the _diff2_fe_fi implementation is equal to _diff2_fi_fe
            float _diff2_fe_fi = (TF_excitatory(E+df, I+df, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                                 -TF_excitatory(E+df, I-df, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                                 -TF_excitatory(E-df, I+df, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                                 +TF_excitatory(E-df, I-df, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                                 )/(4*powdf);

            float _diff2_fi_fi_e = (TF_excitatory(E, I+df, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                                   -2*_TF_e
                                   +TF_excitatory(E, I-df, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
                                   )/powdf;

            float _diff2_fi_fi_i = (TF_inhibitory(E, I+df, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
                                   -2*_TF_i
                                   +TF_inhibitory(E, I-df, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
                                   )/powdf;

            // Integrate with forward euler

            // equation is inspired from github of Zerlaut :
            // https://github.com/yzerlaut/notebook_papers/blob/master/modeling_mesoscopic_dynamics/mean_field/master_equation.py

            // Excitatory firing rate derivation
            dE = dt * ((_TF_e - E
                + .5*C_ee*_diff2_fe_fe_e
                + C_ei*_diff2_fe_fi
                + .5*C_ii*_diff2_fi_fi_e
                    )/T);
//            printf("dE %f \n", dE);

            // Inhibitory firing rate derivation
            dI = dt * ((_TF_i - I
                + .5*C_ee*_diff2_fe_fe_i
                + C_ei*_diff2_fe_fi
                + .5*C_ii*_diff2_fi_fi_i
                    )/T);

            // Covariance excitatory-excitatory derivation
            dC_ee = dt * ((_TF_e*(1./T-_TF_e)/N_e
                + powf((_TF_e-E), 2)
                + 2.*C_ee*_diff_fe_e
                + 2.*C_ei*_diff_fi_e
                - 2.*C_ee
                    )/T);

            // Covariance excitatory-inhibitory or inhibitory-excitatory derivation
            dC_ei = dt * (((_TF_e-E)*(_TF_i-I)
                + C_ee*_diff_fe_i
                + C_ei*_diff_fi_i
                + C_ei*_diff_fe_e
                + C_ii*_diff_fi_e
                - 2.*C_ei
                    )/T);

            // Covariance inhibitory-inhibitory derivation
            dC_ii = dt * ((_TF_i*(1./T-_TF_i)/N_i
                + powf((_TF_i-I), 2)
                + 2.*C_ii*_diff_fi_i
                + 2.*C_ei*_diff_fe_i
                - 2.*C_ii
                    )/T);

            // Adaptation excitatory
            get_fluct_regime_vars(E, I, E_input_excitatory, I_input_excitatory, W_e, E_L_e,
                                    &mu_V, &sigma_V, &T_V);
            dW_e = dt * (-W_e/tau_w_e+b_e*E+a_e*(mu_V-E_L_e)/tau_w_e);

            get_fluct_regime_vars(E, I, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i,
                                    &mu_V, &sigma_V, &T_V);
            dW_i = dt * (-W_i/tau_w_i+b_i*I+a_i*(mu_V-E_L_i)/tau_w_i);

            dnoise = dt * (-noise/tau_OU);

            E += dE;
            I += dI;
            C_ee += dC_ee;
            C_ei += dC_ei;
            C_ii += dC_ii;
            W_e += dW_e;
            W_i += dW_i;
            // noise according to Ornstein–Uhlenbeck process
            noise += dnoise + (curand_normal(&crndst));

            // Update the state
            state((t + 1) % nh, i_node + 0 * n_node) = E;
            state((t + 1) % nh, i_node + 1 * n_node) = I;
            state((t + 1) % nh, i_node + 2 * n_node) = C_ee;
            state((t + 1) % nh, i_node + 3 * n_node) = C_ei;
            state((t + 1) % nh, i_node + 4 * n_node) = C_ii;
            state((t + 1) % nh, i_node + 5 * n_node) = W_e;
            state((t + 1) % nh, i_node + 6 * n_node) = W_i;
            state((t + 1) % nh, i_node + 7 * n_node) = noise;

            // Update the observable
            tavg(i_node + 0 * n_node) += E/n_step;
            tavg(i_node + 1 * n_node) += I/n_step;
            tavg(i_node + 2 * n_node) += C_ee/n_step;
            tavg(i_node + 3 * n_node) += C_ei/n_step;
            tavg(i_node + 4 * n_node) += C_ii/n_step;
            tavg(i_node + 5 * n_node) += W_e/n_step;
            tavg(i_node + 6 * n_node) += W_i/n_step;

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
