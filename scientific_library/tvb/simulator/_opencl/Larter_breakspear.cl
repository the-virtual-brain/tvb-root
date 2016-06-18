#define indexNum(cur,totalN) cur*n+i
__kernel void dfun(__global float *state, __global float *coupling,
                   __global float *param, __global float *deriv)
    {
    int i = get_global_id(0), n = get_global_size(0);
    float V = state[indexNum(0,3)], W = param[indexNum(1,3)],Z = param[indexNum(2,3)];
    
    float c_0 = coupling[i];

    float gCa = param[indexNum(0,32)];
    float gK = param[indexNum(1,32)];
    float gL = param[indexNum(2,32)];
    float phi = param[indexNum(3,32)];
    float gNa = param[indexNum(4,32)];
    float TK = param[indexNum(5,32)];
    float TCa = param[indexNum(6,32)];
    float TNa = param[indexNum(7,32)];
    float VCa = param[indexNum(8,32)];
    float VK = param[indexNum(9,32)];
    float VL = param[indexNum(10,32)];
    float VNa = param[indexNum(11,32)];
    float d_K = param[indexNum(12,32)];
    float tau_K = param[indexNum(13,32)];
    float d_Na = param[indexNum(14,32)];
    float d_Ca = param[indexNum(15,32)];
    float aei = param[indexNum(16,32)];
    float aie = param[indexNum(17,32)];
    float b = param[indexNum(18,32)];
    float C = param[indexNum(19,32)];
    float ane = param[indexNum(20,32)];
    float ani = param[indexNum(21,32)];
    float aee = param[indexNum(22,32)];
    float Iext = param[indexNum(23,32)];
    float rNMDA = param[indexNum(24,32)];
    float VT = param[indexNum(25,32)];
    float d_V = param[indexNum(26,32)];
    float ZT = param[indexNum(27,32)];
    float d_Z = param[indexNum(28,32)];
    float QV_max = param[indexNum(29,32)];
    float QZ_max = param[indexNum(30,32)];
    float t_scale = param[indexNum(31,32)];

    float local_coupling = 1;
    float m_Ca = 0.5 * (1 + tan((V - TCa) / d_Ca));
    float m_Na = 0.5 * (1 + tan((V - TNa) / d_Na));
    float m_K  = 0.5 * (1 + tan((V - TK )  / d_K));
    // voltage to firing rate
    float QV    = 0.5 * QV_max * (1 + tan((V - VT) / d_V));
    float QZ    = 0.5 * QZ_max * (1 + tan((Z - ZT) / d_Z));
    float lc_0  = local_coupling * QV;
    deriv[0] = t_scale * (- (gCa + (1.0 - C) * (rNMDA * aee) * (QV + lc_0)+ C * rNMDA * aee * c_0) * m_Ca * (V - VCa)
                     - gK * W * (V - VK)
                     - gL * (V - VL)
                     - (gNa * m_Na + (1.0 - C) * aee * (QV  + lc_0) + C * aee * c_0) * (V - VNa)
                     - aie * Z * QZ
                     + ane * Iext);
    deriv[1] = t_scale * phi * (m_K - W) / tau_K;
    deriv[2] = t_scale * b * (ani * Iext + aei * V * QV);

    }