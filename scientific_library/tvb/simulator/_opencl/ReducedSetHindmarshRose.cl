#define indexNum(cur,totalN) cur*n+i
__kernel void dfun(__global float *state, __global float *coupling,
                   __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        float xi = state[paraIndex(0,5)], eta=param[paraIndex(1,5)],alpha = state[paraIndex(2,5)], beta = param[paraIndex(3,5)], gamma = param[paraIndex(4,5)];
        // tau a b K11 K12 K21 sigma mu Aik Bik Cik e_i f_i IE_i II_i m_i n_i

        float c_0 = coupling[i];

        float r = param[indexNum(0,27)];
        float a = param[indexNum(1,27)];
        float b = param[indexNum(2,27)];
        float c = param[indexNum(3,27)];
        float d = param[indexNum(4,27)];
        float s = param[indexNum(5,27)];
        float xo = param[indexNum(6,27)];
        float K11 = param[indexNum(7,27)];
        float K12 = param[indexNum(8,27)];
        float K21 = param[indexNum(9,27)];
        float sigma = param[indexNum(10,27)];
        float mu = param[indexNum(11,27)];
        float A_iK = param[indexNum(12,27)];
        float B_iK = param[indexNum(13,27)];
        float C_iK = param[indexNum(14,27)];
        float a_i = param[indexNum(15,27)];
        float b_i = param[indexNum(16,27)];
        float c_i = param[indexNum(17,27)];
        float d_i = param[indexNum(18,27)];
        float e_i = param[indexNum(19,27)];
        float f_i = param[indexNum(20,27)];
        float h_i = param[indexNum(21,27)];
        float p_i = param[indexNum(22,27)];
        float IE_i = param[indexNum(23,27)];
        float II_i = param[indexNum(24,27)];
        float m_i = param[indexNum(25,27)];
        float n_i = param[indexNum(26,27)];

        float local_coupling = 0;
        //TODO Dot Product
            deriv[paraIndex(0,6)] = (eta - a_i * pow(xi , 3) + b_i * pow(xi, 2) - tau +
                                    K11 * ((xi * A_ik) - xi) -
                                    K12 * ((alpha * B_ik) - xi) +
                                    IE_i + c_0 + local_coupling * xi);

            deriv[paraIndex(1,6)] = c_i - d_i * pow(xi, 2) - eta;

            deriv[paraIndex(2,6)] = r * s * xi - r * tau - m_i;

            deriv[paraIndex(3,6)]= (beta - e_i * pow(alpha, 3) + f_i * pow(alpha , 2) - gamma +
                                    K21 * ((xi * C_ik) - alpha) +
                                    II_i + c_0 + local_coupling * xi);

            deriv[paraIndex(4,6)] = h_i - p_i * pow(alpha, 2) - beta;

            deriv[paraIndex(5,6)]= r * s * alpha - r * gamma - n_i;

    }