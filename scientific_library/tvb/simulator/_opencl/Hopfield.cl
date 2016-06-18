#define indexNum(cur,totalN) cur*n+i
__kernel void dfun(__global float *state, __global float *coupling,
                   __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        // this is boilerplate and could be generated
        float x = state[indexNum(0,2)], theta=param[indexNum(1,2)];
        float c = coupling[i];
        // taux tauT dynamic
        float taux = param[indexNum(0,3)],tauT = param[indexNum(1,3)], dynamic = param[indexNum(2,3)];


        deriv[indexNum(0,2)] = (-x + c) / taux;
        deriv[indexNum(1,2)] = (-theta + c) /tauT;
    }