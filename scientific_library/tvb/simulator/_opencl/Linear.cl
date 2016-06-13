__kernel void dfun(__global float *state, __global float *coupling,
                       __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        // this is boilerplate and could be generated
        float x = state[i];
        //gamma
        float c = coupling[i];
        float gamma = param[i];
        deriv[i] = gamma*x + x;
    }