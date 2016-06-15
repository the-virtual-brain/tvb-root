#define indexNum(cur,totalN) cur*n+i
__kernel void dfun(__global float *state, __global float *coupling,
                   __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        float xi = state[paraIndex(0,4)], eta=param[paraIndex(1,4)],alpha = state[paraIndex(2,4)], beta = param[paraIndex(3,4)];
        // tau a b K11 K12 K21 sigma mu Aik Bik Cik e_i f_i IE_i II_i m_i n_i

        float c_0 = coupling[i];

        float tau = param[indexNum(0,17)];
        float a = param[indexNum(1,17)];
        float b = param[indexNum(2,17)];
        float K11 = param[indexNum(3,17)];
        float K12 = param[indexNum(4,17)];
        float K21 = param[indexNum(5,17)];
        float sigma = param[indexNum(6,17)];
        float mu = param[indexNum(7,17)];
        float Aik = param[indexNum(8,17)];
        float Bik = param[indexNum(9,17)];
        float Cik = param[indexNum(10,17)];
        float e_i = param[indexNum(11,17)];
        float f_i = param[indexNum(12,17)];
        float IE_i = param[indexNum(13,17)];
        float II)i = param[indexNum(14,17)];
        float m_i = param[indexNum(15,17)];
        float n_i = param[indexNum(16,17)];

        float local_coupling = 0;
        //TODO Dot Product
        deriv[paraIndex(0,4)] = tau * (xi - e_i * pow(xi,3.0)-eta)+
                                K11 * (xi*Aik-xi)-K12*(alpha*Bik-xi)+
                                tau * (IE_i+c_0+local_coupling*xi);
        deriv[paraIndex(1,4)] = (xi -b*eta +m_i)/tau;
        deriv[paraIndex(2,4)] = tau * (alpha-f_i* pow(alpha,3.0)/3.0 -beta)+
                                K21 * (xi*Cik-alpha)+
                                tau * (II_i+c_0+local_coupling*xi);
        deriv[paraIndex(3,4)] = (alpha-b*beta+n_i)/tau;
    }