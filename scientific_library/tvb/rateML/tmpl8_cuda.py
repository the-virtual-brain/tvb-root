// -*- coding: utf-8 -*-
//
//
// TheVirtualBrain-Scientific Package. This package holds all simulators, and
// analysers necessary to run brain-simulations. You can use it stand alone or
// in conjunction with TheVirtualBrain-Framework Package. See content of the
// documentation-folder for more details. See also http://www.thevirtualbrain.org
//
// (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
//
// This program is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with this
// program.  If not, see <http://www.gnu.org/licenses/>.
//
//
//   CITATION:
// When using The Virtual Brain for scientific publications, please cite it as follows:
//
//  Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
//  Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
//      The Virtual Brain: a simulator of primate brain network dynamics.
//   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
//
//

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

% for state_var in (dynamics.state_variables):
    % if (not state_var.exposure == "None" and not state_var.exposure == "none"):
__device__ float wrap_it_${state_var.name}(float ${state_var.name})
{
    float ${state_var.name}dim[] = {${state_var.exposure}};
    if (${state_var.name} < ${state_var.name}dim[0]) ${state_var.name} = ${state_var.name}dim[0];
    else if (${state_var.name} > ${state_var.name}dim[1]) ${state_var.name} = ${state_var.name}dim[1];

    return ${state_var.name};
}
    % endif
% endfor

__global__ void ${modelname}(

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
#define state(time, i_node) (state_pwi[((time) * ${dynamics.state_variables.__len__()} * n_node + (i_node))*size + id])
#define tavg(i_node) (tavg_pwi[((i_node) * size) + id])

    // only threat those ID that have a corresponding parameters combination
    if (id >= size) return;

    // unpack params
    // These are the two parameters which are usually explore in fitting in this model
    ## printing the to be sweeped parameters
    % for paramcounter, par_var in enumerate(params):
    const float ${par_var.name} = params(${paramcounter});
    % endfor

    // regular constants
% for item in const:
    const float ${item.name} = ${item.value};
% endfor /

    // coupling constants, coupling itself is hardcoded in kernel
% for m in range(len(coupling)):
    % for cc in (coupling[m].constants):
    const float ${cc.name} = ${cc.value};
    %endfor /
% endfor

    // coupling parameters
% for m in range(len(coupling)):
    % for cc in (coupling[m].derived_parameters):
    float ${cc.name} = 0.0;
    %endfor /
% endfor

    % if derparams:
    // derived parameters
    % if nsigpresent==False and noisepresent==True:
    const float nsig = 1;
    % endif /
    % for par_var in derparams:
    const float ${par_var.name} = ${par_var.value};
    % endfor /
    %endif /

    % if dynamics.derived_variables:
    // the dynamic derived variables declarations
    % for i, dv in enumerate(dynamics.derived_variables):
    float ${dv.name} = 0.0;
    % endfor /
    % endif /

    % if dynamics.conditional_derived_variables:
    // conditional_derived variable declaration
    % for cd in dynamics.conditional_derived_variables:
    float ${cd.name} = 0.0;
    % endfor
    % endif \

    % if noisepresent==True:
    curandState crndst;
    curand_init(id + (unsigned int) clock64(), 0, 0, &crndst);
    % endif

    % for state_var in (dynamics.state_variables):
    float ${state_var.name} = ${state_var.dimension};
    % endfor /

    % for td in (dynamics.time_derivatives):
    float ${td.variable} = 0.0;
    % endfor /

    unsigned int dij_i = 0;
    float dij = 0.0;
    float wij = 0.0;

    % for m in range(len(coupling)):
        % for cp in (coupling[m].parameters):
    float ${cp.name} = 0.0;
        % endfor /
    %endfor

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
            % for m in range(len(coupling)):
                % for cdp in (coupling[m].derived_parameters):
            ${cdp.name} = 0.0f;
                %endfor
            % endfor /

            if (t == (i_step)){
            % for i, item in enumerate(dynamics.state_variables):
                tavg(i_node + ${i} * n_node) = 0;
            % endfor /
            }

            % for i, item in enumerate(dynamics.state_variables):
            ${item.name} = state((t) % nh, i_node + ${i} * n_node);
            % endfor /

            // This variable is used to traverse the weights and lengths matrix, which is really just a vector. It is just a displacement. /
            unsigned int i_n = i_node * n_node;

            for (unsigned int j_node = 0; j_node < n_node; j_node++)
            {
                //***// Get the weight of the coupling between node i and node j
                float wij = weights[i_n + j_node]; // nb. not coalesced
                if (wij == 0.0)
                    continue;

                % if 'rec_speed_dt' in derparams and derparams['rec_speed_dt'].value != '0':
                // Get the delay between node i and node j
                dij = lengths[i_n + j_node] * rec_speed_dt;
                dij = dij + 0.5;
                dij_i = (int)dij;
                    % else:
                // no delay specified
                dij_i = 0;
                % endif

                //***// Get the state of node j which is delayed by dij
                % for m in range(len(coupling)):
                    % for cp in (coupling[m].parameters):
                ${cp.name} = state(((t - dij_i + nh) % nh), j_node + ${cp.dimension} * n_node);
                    % endfor /
                %endfor

                // Sum it all together using the coupling function. Kuramoto coupling: (postsyn * presyn) == ((a) * (sin(xj - xi))) \
                % for ml in range(len(coupling)):
                    ## only do this if pre or post is specified
                    % if coupling[ml].dynamics.derived_variables and \
                        (coupling[ml].dynamics.derived_variables['pre'].value != 'None' or \
                         coupling[ml].dynamics.derived_variables['post'].value != 'None'):
                    % for cdp in (coupling[ml].derived_parameters):

                ${cdp.name} += wij * ${coupling[ml].dynamics.derived_variables['post'].value} * ${coupling[ml].dynamics.derived_variables['pre'].value};
                    %endfor
                    % endif
                % endfor /
            } // j_node */

            // global coupling handling, rec_n used to scale nodes
            % for m in range(len(coupling)):
                % for cdp in (coupling[m].derived_parameters):
                    % if cdp.value and (cdp.value !='None' and cdp.value !='none'):
            ${cdp.name} *= ${cdp.value};
                    % endif /
                % endfor
            % endfor \

            % if dynamics.derived_variables:
            // the dynamic derived variables declarations
            % for i, dv in enumerate(dynamics.derived_variables):
            ${dv.name} = ${dv.value};
            % endfor /
            % endif /

            % if dynamics.conditional_derived_variables:
            // The conditional variables
            % for con_der in dynamics.conditional_derived_variables:
                % for case in (con_der.cases):
                    % if (loop.first):
            if (${case.condition}) {
                ${con_der.name} = ${case.value};
                    % elif (not loop.last and not loop.first):
            } else if (${case.condition}) {
                ${con_der.name} = ${case.value};
                    % elif (loop.last):
            } else {
                ${con_der.name} = ${case.value};
            }
                    %endif
                % endfor
            % endfor
            % endif /

            // Integrate with forward euler
            % for i, tim_der in enumerate(dynamics.time_derivatives):
            ${tim_der.variable} = dt * (${tim_der.value});
            % endfor

            % if noisepresent:
            // Add noise because component_type Noise is present in model
            % for ds, td in zip(dynamics.state_variables, dynamics.time_derivatives):
            ${ds.name} += nsig * curand_normal(&crndst) + ${td.variable};
            % endfor /
            % else:
            // No noise is added because it is not present in model
            % for ds, td in zip(dynamics.state_variables, dynamics.time_derivatives):
            ${ds.name} += ${td.variable};
            % endfor /
            % endif

            // Wrap it within the limits of the model
            % for state_var in (dynamics.state_variables):
                % if state_var.exposure == 'PI':
            ${state_var.name} = wrap_it_${state_var.exposure}(${state_var.name});
                % elif (not state_var.exposure == "None" and not state_var.exposure == "none"):
            ${state_var.name} = wrap_it_${state_var.name}(${state_var.name});
                % endif
            % endfor /

            // Update the state
            % for i, state_var in enumerate(dynamics.state_variables):
            state((t + 1) % nh, i_node + ${i} * n_node) = ${state_var.name};
            % endfor /

            // Update the observable
            % for i, expo in enumerate(exposures):
            tavg(i_node + ${i} * n_node) += ${expo.name}/n_step;
            % endfor /

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
