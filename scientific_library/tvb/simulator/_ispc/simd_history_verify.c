/* Single-threaded variant for older hardware */
export void loop0(uniform float aff[], uniform float rh[], uniform float wij[], uniform uint32 ih[])
{
    if (programIndex==0)
        for (uniform int t=0; t<16; t++)
        {
            for (uniform int i=0; i<nn; i++)
            {
                aff[i] = 0.0f;
                for (uniform int j=0; j<nn; j++)
                    aff[i] += wij[i+nn*j] * rh[j*nl + ih[i+nn*j]];
            }

            for (uniform int i=0; i<nn; i++)
            {
                for (uniform int d=nl-1; d>0; d--)
                    rh[i*nl + d] = rh[i*nl + d - 1];
                rh[i*nl] = aff[i];
            }
        }
}

/* 16-wide SIMD-variant for AVX2 or AVX512 */
export void loop1(uniform float aff[], uniform float rh[], uniform float wij[], uniform uint32 ih[])
{
    foreach (it = 0 ... nl)
        for (uniform int t=0; t<16; t++)
        {
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

            for (uniform int i=0; i<nn; i++)
                rh[i*nl+it] = rotate(rh[i*nl+it], -1);

            if (it==0)
                for (uniform int i=0; i<nn; i++)
                    rh[i*nl] = aff[i];
        }
}

/* 10 and 11 are same  as 0 and 1 but w/o delays */
export void loop10(uniform float aff[], uniform float rh[], uniform float wij[], uniform uint32 ih[])
{
    if (programIndex==0)
        for (uniform int t=0; t<16; t++)
        {
            for (uniform int i=0; i<nn; i++)
            {
                aff[i] = 0.0f;
                for (uniform int j=0; j<nn; j++)
                    aff[i] += wij[i+nn*j] * rh[j*nl];
            }

            for (uniform int i=0; i<nn; i++)
            {
                for (uniform int d=nl-1; d>0; d--)
                    rh[i*nl + d] = rh[i*nl + d - 1];
                rh[i*nl] = aff[i];
            }
        }
}


export void loop11(uniform float aff[], uniform float rh[], uniform float wij[], uniform uint32 ih[])
{
    foreach (it = 0 ... nl)
        for (uniform int t=0; t<16; t++)
        {
            for (uniform int j=0; j<nc; j++)
                aff[j*nl+it] = 0.0f;

            for (uniform int j=0; j<nn; j++)
                for (uniform int i=0; i<nc; i++)
                {
                    int ij = j*nn + i*nl + it;
                    float wij_ = wij[ij];
                    float rh_ = rh[j*nl + it];
                    int ih_ = ih[ij];
                    aff[i*nl+it] += wij_ * rh_;//shuffle(rh_, ih_);
                }

            for (uniform int i=0; i<nn; i++)
                rh[i*nl+it] = rotate(rh[i*nl+it], -1);

            if (it==0)
                for (uniform int i=0; i<nn; i++)
                    rh[i*nl] = aff[i];
        }
}

/* another SIMD using scatter, not verified */
/*
export void loop2(uniform int ni, uniform float aff[], uniform float rh[], uniform float wij[], uniform uint32 ih[])
{
    float *scatter[nc];

    foreach (it = 0 ... nl)
    {
        for (uniform int i=0; i<nc; i++)
            scatter[i] = rh + i*nl + it;

        for (uniform int i_ni=0; i_ni<ni; i_ni++)
        {
            for (uniform int t=0; t<1; t++)
            {
                for (int j=0; j<nc; j++)
                    aff[j*nl+it] = 0.0f;

                for (int j=0; j<nn; j++)
                    for (int i=0; i<nc; i++)
                        aff[i*nl+it] += wij[j*nc*nl + i*nl + it] * shuffle(rh[j*nl+it], ih[j*nc*nl + i*nl + it]);

                for (int i=0, ii=it; i<nn; i++, ii+=nl)
                    rh[ii] = rotate(rh[ii], -1);

                for (int i=0, ii=it; i<nc; i++, ii+=nl)
                    *scatter[i] = aff[ii];
            }
        }
    }
}
*/

// TODO 3rd variant w/  insert(rh, 0, extract(aff, i&(nl-1)))