//Generic2dOscillator
#define indexNum(cur,totalN) cur*n+i
__kernel void dfun(__global float *state, __global float *coupling,
                       __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        // this is boilerplate and could be generated
        float V = state[i*2], W = state[i*2+1];
        //tau, a, b, c, I, d, e, f, g, alpha, beta, gamma
        float c_0 = coupling[i];
        float tau = param[n+i], a = param[2*n+i],
        b = param[3*n+i], c = param[4*n+i], I= param[5*n+i], d = param[6*n+i],
        e = param[7*n+i], f = param[8*n+i], g= param[9*n+i], alpha= param[10*n+i],
        beta = param[11*n+i], gamma = param[12*n+i];

        deriv[indexNum(0,2)] = d * tau * (alpha * W - f * V*V*V + e * V*V + g * V + gamma * I + gamma *c_0);
        deriv[indexNum(1,2)] = d * (a + b * V + c * V*V - beta * W) / tau;
    }