// defaults from Stefan 2007, cf tvb/analyzers/fmri_balloon.py
#define TAU_S 0.65f
#define TAU_F 0.41f
#define TAU_O 0.98f
#define ALPHA 0.32f
#define TE 0.04f
#define V0 4.0f
#define E0 0.4f
#define EPSILON 0.5f
#define NU_0 40.3f
#define R_0 25.0f

#define RECIP_TAU_S (1.0f / TAU_S)
#define RECIP_TAU_F (1.0f / TAU_F)
#define RECIP_TAU_O (1.0f / TAU_O)
#define RECIP_ALPHA (1.0f / ALPHA)
#define RECIP_E0 (1.0f / E0)

// "derived parameters"
#define k1 (4.3f * NU_0 * E0 * TE)
#define k2 (EPSILON * R_0 * E0 * TE)
#define k3 (1.0f - EPSILON)

__global__ void bold_update(int n_node, float dt,
                      // bold.shape = (4, n_nodes, n_threads)
		      float * __restrict__ bold_state, 
                      // nrl.shape = (n_nodes, n_threads)
		      float * __restrict__ neural_state,
                      // out.shape = (n_nodes, n_threads)
		      float * __restrict__ out)
{
  const unsigned int it = (blockIdx.x * blockDim.x) + threadIdx.x;
  const unsigned int nt = blockDim.x * gridDim.x;

  int var_stride = n_node * nt;
  for (int i_node=0; i_node < n_node; i_node++)
  {
      float *node_bold = bold_state + i_node * nt + it;

      float s = node_bold[0 * var_stride];
      float f = node_bold[1 * var_stride];
      float v = node_bold[2 * var_stride];
      float q = node_bold[3 * var_stride];

      float x = neural_state[i_node * nt + it];

      float ds = x - RECIP_TAU_S * s - RECIP_TAU_F * (f - 1.0f);
      float df = s;
      float dv = RECIP_TAU_O * (f - pow(v, RECIP_ALPHA));
      float dq = RECIP_TAU_O * (f * (1.0f - pow(1.0f - E0, 1.0f / f)) 
				* RECIP_E0 - pow(v, RECIP_ALPHA) * (q / v));

      s += dt * ds;
      f += dt * df;
      v += dt * dv;
      q += dt * dq;

      node_bold[0 * var_stride] = s;
      node_bold[1 * var_stride] = f;
      node_bold[2 * var_stride] = v;
      node_bold[3 * var_stride] = q;

      out[i_node * nt + it] = V0 * (   k1 * (1.0f - q    ) 
                                     + k2 * (1.0f - q / v) 
                                     + k3 * (1.0f -     v) );
  } // i_node
} // kernel
