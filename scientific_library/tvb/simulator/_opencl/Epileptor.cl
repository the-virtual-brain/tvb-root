#define indexNum(cur,totalN) cur*n+i
__kernel void dfun(__global float *state, __global float *coupling,
                   __global float *param, __global float *deriv)
    {
        int i = get_global_id(0), n = get_global_size(0);

        float y0 = state[indexNum(0,6)], y1=param[indexNum(1,6)],
        y2 = state[indexNum(2,6)], y3 = param[indexNum(4,6)],
        y4 = state[indexNum(4,6)], y5 = param[indexNum(5,6)];

        // x0 Iext Iext2 a b slope tt Kvf c d r Ks Kf aa tau ydot

        float c_pop1 = coupling[indexNum(0,2)];
        float c_pop2 = coupling[indexNum(1,2)];

        float x0 = param[indexNum(0,15)];
        float Iext = param[indexNum(1,15)];
        float Iext2 = param[indexNum(2,15)];
        float a = param[indexNum(3,15)];
        float b = param[indexNum(4,15)];
        float slope = param[indexNum(5,15)];
        float tt = param[indexNum(6,15)];
        float Kvf = param[indexNum(7,15)];
        float c = param[indexNum(8,15)];
        float d = param[indexNum(9,15)];
        float r = param[indexNum(10,15)];
        float Ks = param[indexNum(11,15)];
        float Kf = param[indexNum(12,15)];
        float aa = param[indexNum(13,15)];
        float tau = param[indexNum(14,15)];


        float temp_ydot0,temp_ydot2,temp_ydot4;
        if(y0 < 0.0){
            temp_ydot0 = -a*y0*y0+b*y0;
        }else{
            temp_ydot0 = slope-y3+0.6*(y2-4.0)*(y2-4.0);
        }

        deriv[0] = tt * (y1-y2+Iext + Kvf + c_pop1+temp_ydot0*y0);
        deriv[1] = tt * (c - d*y0*y0 - y1);

        if( y2 < 0.0){
            temp_ydot2 = -0.1*pow(y2,7);
        }else{
            temp_ydot2 = 0.0;
        }
        deriv[2] = tt* (r * (4*(y0-x0)-y2+temp_ydot2+Ks*c_pop1));

        deriv[3] = tt * (-y4 + y3 - pow(y3,3) + Iext2 + 2 * y5 - 0.3 * (y2 - 3.5) + Kf * c_pop2);

        if(y3<-0.25){
            temp_ydot4 = 0.0;
        }else{
            temp_ydot4 = aa*(y3+0.25);
        }
        deriv[4] = tt * ((-y4 + temp_ydot4) / tau);
        deriv[5] = tt * (-0.01 * (y5 - 0.1 * y0));
        

    }