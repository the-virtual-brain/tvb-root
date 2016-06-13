#define indexNum(cur,totalN) cur*n+i
__kernel void dfun(__global float *state, __global float *coupling,
                   __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        float xi = state[paraIndex(0,4)], eta=param[paraIndex(1,4)],alpha = state[paraIndex(2,4)], beta = param[paraIndex(3,4)];
        // tau a b K11 K12 K21 sigma mu

        float c_0 = coupling[i];

        float tau = param[indexNum(0,8)];
        float a = param[indexNum(1,8)];
        float b = param[indexNum(2,8)];
        float K11 = param[indexNum(3,8)];
        float K12 = param[indexNum(4,8)];
        float K21 = param[indexNum(5,8)];
        float sigma = param[indexNum(6,8)];
        float mu = param[indexNum(7,8)];


    }