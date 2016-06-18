#define indexNum(cur,totalN) cur*n+i
#define sigma_fun(sv) ( rho_1 *(rho_2 - sv) > 709 ? 0 : 2*e0 / (1+rho_1 *(rho_2 - sv)))


__kernel void dfun(__global float *state, __global float *coupling,
                       __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);
        int n_param = 12;
        // this is boilerplate and could be generated
        float v1 = state[indexNum(0,12)], y1 = state[indexNum(1,12)],v2 = state[indexNum(2,12)],
        y2 = state[indexNum(3,12)],v3 = state[indexNum(4,12)], y3 = state[indexNum(5,12)],
        v4 = state[indexNum(6,12)], y4 = state[indexNum(7,12)], v5 = state[indexNum(8,12)],
        y5 = state[indexNum(9,12)], v6 = state[indexNum(10,12)], v7 = state[indexNum(11,12)],
        c = coupling[i];
        //He Hi ke ki e0 rho_2 rho_1 gamma_1 gamma_2 gamma_3 gamma_4 gamma_5 P U Q Heke Hiki ke_2 ki_2 keke kiki gamma_1T gamma_2T gamma_3T
        float He = param[indexNum(0,24)];
        float Hi = param[indexNum(1,24)];
        float ke = param[indexNum(2,24)];
        float ki = param[indexNum(3,24)];
        float e0 = param[indexNum(4,24)];
        float rho_2 = param[indexNum(5,24)];
        float rho_1 = param[indexNum(6,24)];
        float gamma_1 = param[indexNum(7,24)];
        float gamma_2 = param[indexNum(8,24)];
        float gamma_3 = param[indexNum(9,24)];
        float gamma_4 = param[indexNum(10,24)];
        float gamma_5 = param[indexNum(11,24)];
        float P = param[indexNum(12,24)];
        float U = param[indexNum(13,24)];
        float Q = param[indexNum(14,24)];
        float Heke = param[indexNum(15,24)];
        float Hiki = param[indexNum(16,24)];
        float ke_2 = param[indexNum(17,24)];
        float ki_2 = param[indexNum(18,24)];
        float keke = param[indexNum(19,24)];
        float kiki = param[indexNum(20,24)];
        float gamma_1T = param[indexNum(21,24)];
        float gamma_2T = param[indexNum(22,24)];
        float gamma_3T = param[indexNum(23,24)];
        // TODO: implement local coupling
        float locol_coupling = 1;

        float coupled_input = c + 6*locol_coupling;

        deriv[indexNum(0,12)] = y1;
        deriv[indexNum(1,12)] = Heke * (gamma_1 * sigma_fun(v2 - v3) + gamma_1T * (U + coupled_input )) - ke_2 * y1 - keke * v1;
        // exc input to the pyramidal cells
        deriv[indexNum(2,12)] = y2;
        deriv[indexNum(3,12)] = Heke * (gamma_2 * sigma_fun(v1)      + gamma_2T * (P + coupled_input )) - ke_2 * y2 - keke * v2;
        // inh input to the pyramidal cells
        deriv[indexNum(4,12)] = y3;
        deriv[indexNum(5,12)] = Hiki * (gamma_4 * sigma_fun(v4 - v5)) - ki_2 * y3 - kiki * v3;
        deriv[indexNum(6,12)] = y4;
        // exc input to the inhibitory interneurons
        deriv[indexNum(7,12)] = Heke * (gamma_3 * sigma_fun(v2 - v3) + gamma_3T * (Q + coupled_input)) - ke_2 * y4 - keke * v4;
        deriv[indexNum(8,12)] = y5;
        // inh input to the inhibitory interneurons
        deriv[indexNum(9,12)] = Hiki * (gamma_5 * sigma_fun(v4 - v5)) - ki_2 * y5 - keke * v5;
        // aux variables (the sum gathering the postsynaptic inh & exc potentials)
        // pyramidal cells
        deriv[indexNum(10,12)] = y2 - y3;
        // inhibitory cells
        deriv[indexNum(11,12)] = y4 - y5;

    }
