#define indexNum(cur,totalN) cur*n+i
__kernel void dfun(__global float *state, __global float *coupling,
                   __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        float E = state[indexNum(0,2)], I = state[indexNum(1,2)];
        //c_ee c_ei c_ie c_ii tau_e tau_i a_e b_e c_e a_i b_i c_i r_e r_i k_e k_i P Q theta_e theta_i alpha_e alpha_i

        float c_0 = coupling[i];

        float c_ee = param[indexNum(0,22)];
        float c_ei = param[indexNum(1,22)];
        float c_ie = param[indexNum(2,22)];
        float c_ii = param[indexNum(3,22)];
        float tau_e = param[indexNum(4,22)];
        float tau_i = param[indexNum(5,22)];
        float a_e = param[indexNum(6,22)];
        float b_e = param[indexNum(7,22)];
        float c_e = param[indexNum(8,22)];
        float a_i = param[indexNum(9,22)];
        float b_i = param[indexNum(10,22)];
        float c_i = param[indexNum(11,22)];
        float r_e = param[indexNum(12,22)];
        float r_i = param[indexNum(13,22)];
        float k_e = param[indexNum(14,22)];
        float k_i = param[indexNum(15,22)];
        float P = param[indexNum(16,22)];
        float Q = param[indexNum(17,22)];
        float theta_e = param[indexNum(18,22)];
        float theta_i = param[indexNum(19,22)];
        float alpha_e = param[indexNum(20,22)];
        float alpha_i = param[indexNum(21,22)];

        //TODO: dummy local_coupling
        float local_coupling = 1;
        float lc_0 = local_coupling * E;
        float lc_1 = local_coupling * I;
        
        float x_e = alpha_e * (c_ee * E - c_ei * I + P  - theta_e +  c_0 + lc_0 + lc_1);
        float x_i = alpha_i * (c_ie * E - c_ii * I + Q  - theta_i + lc_0 + lc_1);

        float s_e = c_e / (1.0 + exp(-a_e * (x_e - b_e)));
        float s_i = c_i / (1.0 + exp(-a_i * (x_i - b_i)));

        deriv[indexNum(0,2)] = (-E + (k_e - r_e * E) * s_e) / tau_e;
        deriv[indexNum(1,2)] = (-I + (k_i - r_i * I) * s_i) / tau_i;
        
        

    }