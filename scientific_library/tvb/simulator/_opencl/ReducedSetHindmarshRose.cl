__kernel void dfun(__global float *state, __global float *coupling,
                   __global float *params, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        float3 xi = vload3(i,state), eta=vload3(i,state+3*n),
        tau = vload3(i,state+6*n), alpha = vload3(i,state+9*n),
        beta = vload3(i,state+12*n), gamma = vload3(i,state+15*n);

        // "r s K11 K12 K21   a_i b_i c_i d_i e_i f_i h_i p_i IE_i II_i m_i n_i   A_ik B_ik C_ik params length = 68

        float c_0 = coupling[i];

        float r = params[i];
        float s = params[n+i];
        float K11 = params[2*n+i];
        float K12 = params[3*n+i];
        float K21 = params[4*n+i];

        float3 a_i = vload3(i,params + 5*n);
        float3 b_i = vload3(i,params + 8*n);
        float3 c_i = vload3(i,params + 11*n);
        float3 d_i = vload3(i,params + 14*n);
        float3 e_i = vload3(i,params + 17*n);
        float3 f_i = vload3(i,params + 20*n);
        float3 h_i = vload3(i,params + 23*n);
        float3 p_i = vload3(i,params + 26*n);
        float3 IE_i = vload3(i,params + 29*n);
        float3 II_i = vload3(i,params + 32*n);
        float3 m_i = vload3(i,params + 35*n);
        float3 n_i = vload3(i,params + 38*n);



        float3 Aik_0 = vload3(i,params + 41*n);
        float3 Aik_1 = vload3(i,params + 41*n+3);
        float3 Aik_2 = vload3(i,params + 41*n+6);

        float3 Bik_0 = vload3(i,params + 50*n);
        float3 Bik_1 = vload3(i,params + 50*n+3);
        float3 Bik_2 = vload3(i,params + 50*n+6);

        float3 Cik_0 = vload3(i,params + 59*n);
        float3 Cik_1 = vload3(i,params + 59*n+3);
        float3 Cik_2 = vload3(i,params + 59*n+6);

        float local_coupling = 0;

        float3 deriv1 = (eta - a_i * pow(xi , 3) + b_i * pow(xi, 2) - tau +
                                K11 * ( (float3)(dot(xi , Aik_0),dot(xi , Aik_1),dot(xi , Aik_2)) - xi) -
                                K12 * ( (float3)(dot(alpha , Bik_0),dot(alpha , Bik_1),dot(alpha , Bik_2) )- xi) +
                                IE_i + c_0 + local_coupling * xi);

        float3 deriv2 = c_i - d_i * pow(xi, 2) - eta;

        float3 deriv3 = r * s * xi - r * tau - m_i;

        float3 deriv4 = beta - e_i * pow(alpha, 3) + f_i * pow(alpha , 2) - gamma +
                                K21 * ( (float3)(dot(xi , Cik_0),dot(xi , Cik_1), dot(xi , Cik_2)) - alpha) +
                                II_i + c_0 + local_coupling * xi;

        float3 deriv5 = h_i - p_i * pow(alpha, 2) - beta;

        float3 deriv6 = r * s * alpha - r * gamma - n_i;


        vstore3(deriv1,i,deriv);
        vstore3(deriv2,i,deriv+3*n);
        vstore3(deriv3,i,deriv+6*n);
        vstore3(deriv4,i,deriv+9*n);
        vstore3(deriv5,i,deriv+12*n);
        vstore3(deriv6,i,deriv+15*n);



    }