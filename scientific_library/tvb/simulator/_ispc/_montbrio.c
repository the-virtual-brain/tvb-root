#define nn 96
#define nl 16
#define nc 6

#define pi 3.141592653589793f

float sq(float x) { return x * x; }

struct Data {
    float k, I, Delta, eta, tau, J, cr, cv, dt, r_sigma, V_sigma;
};

export void loop(
    uniform Data * uniform data,
    uniform float aff[],
    uniform float rh[],
    uniform float Vh[],
    uniform float wij[],
    uniform uint32 ih[],
    uniform float W[],
    uniform float r[],
    uniform float V[],
    uniform float nr[],
    uniform float nV[],
    uniform float tavg[]
    )
{
    uniform float o_tau = 1.0f / data->tau;
    uniform float sq_pi_tau = pi*pi * data->tau*data->tau;
    uniform float dt=data->dt;
    uniform float sqrt_dt = sqrt(dt);
    uniform float o_6 = 1.0f / 6.0f;
    uniform float o_16 = 1.0f / 16.0f;

    foreach (it = 0 ... nl)
    {
        // w/o rng 190k it/s,  5us call overhead
        // w/o rng 151k it/s,  7us t=1  w/o tavg
        // w/o rng 125k it/s,  8us t=1  w/ tavg
        // w/o rng  46k it/s, 21us t=16 w/ tavg
        // w/o rng  15k it/s, 68us t=64 w/ tavg
        //  -> loop body ~1us
        // w/  rng  18k it/s, 56us t=16 w/ tavg
        for (uniform int t=0; t<16; t++) // needs to match W.shape
        {
            // compute delayed state coupling
            for (uniform int j=0; j<nc; j++)
                aff[j*nl+it] = 0.0f;

            for (uniform int j=0; j<nn; j++)
                for (uniform int i=0; i<nc; i++)
                {
                    int ij = j*nn + i*nl + it;
                    float wij_ = wij[ij];
                    float rh_ = rh[j*nl + it];
                    int ih_ = ih[ij];
                    float inc = wij_ * shuffle(rh_, ih_);
                    aff[i*nl+it] += inc;
                }

            //continue;
            // update neural mass
            for (uniform int i=0; i<nc; i++) {
                int i_ = i*nl + it;
                float kr[4], kV[4], r_, V_; // RK4
                for (uniform int k=0; k<4; k++) {
                    r_ = r[i_];
                    V_ = V[i_];
                    if (k>0) {
                        uniform float kh = k==3 ? 1.0f : 0.5f;
                        r_ += dt*kh*kr[k-1];
                        V_ += dt*kh*kV[k-1];
                    }
                    kr[k] = o_tau * (data->Delta / (pi * data->tau) + 2 * V_ * r_);
                    kV[k] = o_tau * (sq(V_) - sq_pi_tau * sq(r_) + data->eta + data->J * data->tau * r_ + data->I + data->k * data->cr * aff[i_]);
                }
                nr[i_] = dt*o_6*(kr[0] + 2*kr[1] + 2*kr[2] + kr[3]) + sqrt_dt*data->r_sigma*W[t*nn+i_];
                nV[i_] = dt*o_6*(kV[0] + 2*kV[1] + 2*kV[2] + kV[3]) + sqrt_dt*data->V_sigma*W[16*nn + t*nn+i_];
            }

            for (uniform int i=0; i<nc; i++) {
                int i_ = i*nl + it;
                r[i_] += nr[i_];
                V[i_] += nV[i_];
            }

            // shift history
            for (uniform int i=0; i<nn; i++)
            {
                rh[i*nl+it] = rotate(rh[i*nl+it], -1);
                Vh[i*nl+it] = rotate(Vh[i*nl+it], -1);
            }

            // update history w/ current state
            if (it==0)
                for (uniform int i=0; i<nn; i++)
                {
                    rh[i*nl] = r[i];
                    Vh[i*nl] = V[i];
                }
        }

        // temporal average
        for (uniform int i=0; i<nn; i++)
        {
            uniform float tavg_i = reduce_add(rh[i*nl+it]);
            if (it==0) tavg[i] = tavg_i * o_16;
            tavg_i = reduce_add(Vh[i*nl+it]);
            if (it==0) tavg[nn+i] = tavg_i * o_16;
        }

        // TODO bold
    }
}