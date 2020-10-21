#define nn 96
#define nl 16
#define nc 6

#define I 1.0f
#define Delta 1.0f
#define eta -5.0f
#define tau 250.0f
#define J 15.0f
#define cr 0.01f
#define cv 0.0f
#define pi 3.141592653589793f

float sq(float x) { return x * x; }

export void loop(
    uniform float k,
    uniform float aff[],
    uniform float rh[],
    uniform float Vh[],
    uniform float wij[],
    uniform int ih[],
    uniform float W[],
    uniform float r[],
    uniform float V[],
    uniform float nr[],
    uniform float nV[],
    uniform float tavg[]
    )
{
    uniform float o_tau = 1.0f / tau;
    uniform float sq_pi_tau = pi*pi * tau*tau;
    uniform float sqrt_dt = sqrt(0.1f);
    foreach (it = 0 ... nl)
    {
        for (uniform int t=0; t<16; t++)
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
                    aff[i*nl+it] += wij_ * shuffle(rh_, ih_);
                }

            // update neural mass
            for (int i=0; i<nc; i++) {
                int i_ = i*nl + it;
                float dr = o_tau * (Delta / (pi * tau) + 2 * V[i_] * r[i_]);
                float dV = o_tau * (sq(V[i_]) - sq_pi_tau * sq(r[i_]) + eta + J * tau * r[i_] + I + cr * aff[i_]);
                nr[i_] = 0.1f*dr + sqrt_dt*1e-4f*W[t*nn+i_];
                nV[i_] = 0.1f*dV + sqrt_dt*1e-4f*W[nl*nn + t*nn+i_];
            }

            for (int i=0; i<nc; i++) {
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
            for (it==0)
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
            if (it==0) tavg[i] = tavg_i/16;
            tavg_i = reduce_add(Vh[i*nl+it]);
            if (it==0) tavg[nn+i] = tavg_i/16;
        }

        // TODO bold
    }
}