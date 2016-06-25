#define indexNum(cur,totalN) cur*n+i
#define getFloat3Vector(ptr) (float3)(*ptr,*(ptr+1),*(ptr+2))
__kernel void dfun(__global float *state, __global float *coupling,
                   __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        float3 xi = vload3(i,state), eta=vload3(i,state+3*n),
        alpha = vload3(i,state+6*n), beta = vload3(i,state+9*n);
        // tau b K11 K12 K21    e_i f_i IE_i II_i m_i n_i (1*3 vector)   Aik Bik Cik (3*3 matrix) params length = 50


        float c_0 = coupling[i];

        float tau = param[i];
        float b = param[n+i];
        float K11 = param[2*n+i];
        float K12 = param[3*n+i];
        float K21 = param[4*n+i];

        float3 e_i = vload3(i,param + 5*n);
        float3 f_i = vload3(i,param + 8*n);
        float3 IE_i = vload3(i,param + 11*n);
        float3 II_i = vload3(i,param +14*n);
        float3 m_i = vload3(i,param+ 17*n);
        float3 n_i = vload3(i,param + 20*n);

        float3 Aik_0 = vload3(i,param + 23*n);
        float3 Aik_1 = vload3(i,param + 23*n+3);
        float3 Aik_2 = vload3(i,param + 23*n+6);

        float3 Bik_0 = vload3(i,param + 32*n);
        float3 Bik_1 = vload3(i,param + 32*n+3);
        float3 Bik_2 = vload3(i,param + 32*n+6);

        float3 Cik_0 = vload3(i,param + 41*n);
        float3 Cik_1 = vload3(i,param + 41*n+3);
        float3 Cik_2 = vload3(i,param + 41*n+6);


        float local_coupling = 0;




        float3 deriv1 = tau * (xi - e_i * pow(xi,3)-eta)+
                                 K11 * ( (float3)(dot(xi,Aik_0),dot(xi,Aik_1),dot(xi,Aik_2)) - xi)-
                                 K12*( (float)(dot(alpha,Bik_0),dot(alpha,Bik_1),dot(alpha,Bik_2) )-xi)+
                                 tau * (IE_i+c_0+local_coupling*xi);

        float3 deriv2 = (xi - b*eta + m_i)/tau;
        float3 deriv3 = tau * (alpha-f_i*pow(alpha,3)/3 - beta)+
                                K21 * ((float3)(dot(xi,Cik_0),dot(xi,Cik_1),dot(xi,Cik_2)) -alpha)+
                                tau * (II_i+c_0+local_coupling*xi);
        float3 deriv4 = (alpha-b*beta+n_i)/tau;

        vstore3(deriv1,i,deriv);
        vstore3(deriv2,i,deriv+3*n);
        vstore3(deriv3,i,deriv+6*n);
        vstore3(deriv4,i,deriv+9*n);

    }
