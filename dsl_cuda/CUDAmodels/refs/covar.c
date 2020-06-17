#include <stdio.h>
#ifdef TEST_COVAR
#include <math.h>
#endif

// stable one-pass co-moment algo, cf wikipedia

#ifndef TEST_COVAR
__global__
#endif
void update_cov(
    unsigned int i_sample,
    unsigned int n_node,
    float * __restrict__ cov,
    float * __restrict__ means,
    const float * __restrict__ data
)
{
#ifdef TEST_COVAR
    const unsigned int it = 0;
    const unsigned int nt = 1;
#else
    const unsigned int it = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned int nt = blockDim.x * gridDim.x * gridDim.y;
#endif

    if (i_sample == 0)
    {
	for (int i_node = 0; i_node < n_node; i_node++)
	    means[i_node * nt + it] = data[i_node * nt + it];
	return;
    }

    const float recip_n = 1.0f / i_sample;

    // double buffer to avoid copying memory
    float *next_mean = means, *prev_mean = means;
    if ((i_sample % 2) == 0) {
	prev_mean += n_node * nt;
    } else {
	next_mean += n_node * nt;
    } 
 
    for (unsigned int i_node = threadIdx.y; i_node < n_node; i_node += blockDim.y)
    {
        if (i_node >= n_node) continue;

	int i_idx = i_node * nt + it;
	next_mean[i_idx] = prev_mean[i_idx] + (data[i_idx] - prev_mean[i_idx]) * recip_n;
    }
    
    // TODO shared mem useful here?
    for (unsigned int i_node = threadIdx.y; i_node < n_node; i_node += blockDim.y)
    {
        if (i_node >= n_node) continue;

	int i_idx = i_node * nt + it;
	float data_mean_i = data[i_idx] - prev_mean[i_idx];

	for (int j_node = 0; j_node < n_node; ++j_node)
	{
	    int j_idx = j_node * nt + it;
	    float data_mean_j = data[j_idx] - next_mean[j_idx];
	    int cij_idx = (j_node * n_node + i_node) * nt + it;
	    cov[cij_idx] += data_mean_j * data_mean_i;
	}
    }
}

#ifndef TEST_COVAR
__global__
#endif
void cov_to_corr(
    unsigned int n_sample,
    unsigned int n_node,
    float * __restrict__ cov,
    float * __restrict__ corr
)
{
#ifdef TEST_COVAR
    const unsigned int it = 0;
    const unsigned int nt = 1;
#else
    const unsigned int it = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned int nt = blockDim.x * gridDim.x * gridDim.y;
#endif

    float recip_n_samp = 1.0f / n_sample;

    // normalize comoment to covariance
    for (unsigned int ij = 0; ij < (n_node * n_node); ++ij)
	cov[ij*nt + it] *= recip_n_samp;

    // compute correlation coefficient
#define COV(i, j) cov[((i)*n_node + (j))*nt + it]
#define CORR(i, j) corr[((i)*n_node + (j))*nt + it]

    for (unsigned int i = threadIdx.y; i < n_node; i += blockDim.y)
    {
        if (i >= n_node) continue;

	float var_i = COV(i, i);
	for (unsigned int j = 0; j < n_node; ++j)
	{
	    float var_j = COV(j, j);
	    CORR(i, j) = COV(i, j) * rsqrtf(var_i * var_j);
	}
    }
}
