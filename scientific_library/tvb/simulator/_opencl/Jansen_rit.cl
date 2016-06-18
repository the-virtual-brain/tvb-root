#define indexNum(cur,totalN) cur*n+i // consider i*totalN+cur
__kernel void dfun(__global float *state, __global float *coupling,
                       __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        // this is boilerplate and could be generated
        float y0 = state[i*6], y1 = state[i*6+1],y2 = state[i*6+2], 
        y3 = state[i*6+3], y4 = state[i*6+4], y5 = state[i*6+5];
        float c = coupling[i]; ;
        // nu_max  r  v0  a  a_1  a_2  a_3  a_4  A  b  B  J  mu //total 13 parameters
        float nu_max = param[indexNum(0,13)];
        float  r = param[indexNum(1,13)];
        float  v0 = param[indexNum(2,13)];
        float  a = param[indexNum(3,13)];
        float  a_1 = param[indexNum(4,13)];
        float  a_2 = param[indexNum(5,13)];
        float  a_3 = param[indexNum(6,13)];
        float  a_4 = param[indexNum(7,13)];
        float  A = param[indexNum(8,13)];
        float  b = param[indexNum(9,13)];
        float  B = param[indexNum(10,13)];
        float  J = param[indexNum(11,13)];
        float  mu = param[indexNum(12,13)];

        float src = y1 - y2;

        float sigm_y1_y2 = 2.0 * nu_max / (1.0 + exp(r * (v0 - (y1 - y2))));
        float sigm_y0_1 = 2.0 * nu_max / (1.0 + exp(r * (v0 - (a_1 * J * y0))));
        float sigm_y0_3 = 2.0 * nu_max / (1.0 + exp(r * (v0 - (a_3 * J * y0))));
        deriv[indexNum(0,6)] = y3;
        deriv[indexNum(1,6)] = y4;
        deriv[indexNum(2,6)] = y5;
        deriv[indexNum(3,6)] = A * a * sigm_y1_y2 - 2.0 * a * y3 - a * a * y0;
        deriv[indexNum(4,6)] = A * a * (mu + a_2 * J * sigm_y0_1 + c + src) - 2.0 * a * y4 - a * a * y1;
        deriv[indexNum(5,6)] = B * b * (a_4 * J * sigm_y0_3) - 2.0 * b * y5 - b *b * y2;
    }
