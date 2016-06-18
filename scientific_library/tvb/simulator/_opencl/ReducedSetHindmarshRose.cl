define indexNum(cur,totalN) cur*n+i
__kernel void dfun(__global float *state, __global float *coupling,
                   __global float *params, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        float xi = state[indexNum(0,6)], eta=state[indexNum(1,6)], tau = state[indexNum(2,6)] ,  alpha = state[indexNum(3,6)], beta = state[indexNum(4,6)], gamma = state[indexNum(5,6)];
        // "r s K11 K12 K21 mu A_iK B_iK C_iK a_i b_i c_i d_i e_i f_i h_i p_i IE_i II_i m_i n_i gamma_1T gamma_2T gamma_3T

        float c_0 = coupling[i];

        float r = params[indexNum(0,24)];
        float s = params[indexNum(1,24)];
        float K11 = params[indexNum(2,24)];
        float K12 = params[indexNum(3,24)];
        float K21 = params[indexNum(4,24)];
        float mu = params[indexNum(5,24)];
        float A_iK = params[indexNum(6,24)];
        float B_iK = params[indexNum(7,24)];
        float C_iK = params[indexNum(8,24)];
        float a_i = params[indexNum(9,24)];
        float b_i = params[indexNum(10,24)];
        float c_i = params[indexNum(11,24)];
        float d_i = params[indexNum(12,24)];
        float e_i = params[indexNum(13,24)];
        float f_i = params[indexNum(14,24)];
        float h_i = params[indexNum(15,24)];
        float p_i = params[indexNum(16,24)];
        float IE_i = params[indexNum(17,24)];
        float II_i = params[indexNum(18,24)];
        float m_i = params[indexNum(19,24)];
        float n_i = params[indexNum(20,24)];
        float gamma_1T = params[indexNum(21,24)];
        float gamma_2T = params[indexNum(22,24)];
        float gamma_3T = params[indexNum(23,24)];

        float local_coupling = 0;
        //TODO Dot Product
            deriv[indexNum(0,6)] = (eta - a_i * pow(xi , 3) + b_i * pow(xi, 2) - tau +
                                    K11 * ((xi * A_iK) - xi) -
                                    K12 * ((alpha * B_iK) - xi) +
                                    IE_i + c_0 + local_coupling * xi);

            deriv[indexNum(1,6)] = c_i - d_i * pow(xi, 2) - eta;

            deriv[indexNum(2,6)] = r * s * xi - r * tau - m_i;

            deriv[indexNum(3,6)]= (beta - e_i * pow(alpha, 3) + f_i * pow(alpha , 2) - gamma +
                                    K21 * ((xi * C_iK) - alpha) +
                                    II_i + c_0 + local_coupling * xi);

            deriv[indexNum(4,6)] = h_i - p_i * pow(alpha, 2) - beta;

            deriv[indexNum(5,6)]= r * s * alpha - r * gamma - n_i;

    }