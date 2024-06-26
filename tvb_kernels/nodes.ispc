// pointer is constant and same for all threads, as are elements pointed to
typedef const uniform   int *const uniform data_ints;
typedef const uniform float *const uniform data_floats;

// pointer is constant and same for all threads,
// elements pointed to are same for all threads, but modifiable
typedef uniform   int *const uniform work_ints;
typedef uniform float *const uniform work_floats;

struct params {
  const int count;
  data_floats values;
};

// connectivity model with csr format sparse connections & delay buffer
struct connectivity {
  const uniform int num_node;
  const uniform int num_nonzero;
  const uniform int num_cvar;
  // horizon must be power of two
  const int horizon;
  const int horizon_minus_1;
  data_floats weights; // (num_nonzero,)
  data_ints indices; // (num_nonzero,)
  data_ints indptr; // (num_nodes+1,)
  data_ints idelays; // (num_nonzero,)
  work_floats buf; // delay buffer (num_cvar, num_nodes, horizon)
  work_floats cx1;
  work_floats cx2;
};

struct sim {
  // keep invariant stuff at the top, per sim stuff below
  const int rng_seed;
  const int num_node;
  const int num_svar;
  const int num_time;
  const int num_params;
  const int num_spatial_params;
  const float dt;
  const int oversample; // TODO "oversample" for stability,
  const int num_skip; // skip per output sample
  work_floats z_scale; // (num_svar), sigma*sqrt(dt)

  // parameters
  const uniform params global_params;
  const uniform params spatial_params;

  work_floats state_trace; // (num_time//num_skip, num_svar, num_nodes)
  work_floats states; // full states (num_svar, num_nodes)

  const uniform connectivity conn;
};

typedef const uniform struct sim the_sim;
typedef const uniform struct connectivity the_conn;

// case of small number of nodes (90 floats is 360 bytes), dense, long horizon
// probably just hit a single node's horizon: random increments
export void cx_all_j(
    the_conn &c,
    uniform int t, uniform int j)
{
    uniform float * const buf = c.buf + j*c.horizon;
    uniform int th = t + c.horizon;
    // here assuming columns are targets
    foreach (l = c.indptr[j] ... c.indptr[j + 1])
    {
        int i = c.indices[l];
        float w = c.weights[l];
        int d = c.idelays[l];
        int p1 = (th - d) & c.horizon_minus_1;
        int p2 = (th - d + 1) & c.horizon_minus_1;
        c.cx1[i] += w * buf[p1];
        c.cx2[i] += w * buf[p2];
    }
}

export void cx_all(the_conn &c, uniform int t)
{
  foreach (i = 0 ... c.num_node)
    c.cx1[i] = c.cx2[i] = 0.0f;

  for (uniform int j=0; j<c.num_node; j++)
    cx_all_j(c, t, j);
}

export void cx_all2(the_conn &c, uniform int t)
{
  uniform int th = t + c.horizon;
  foreach (i = 0 ... c.num_node)
  {
    float cx1=0.f, cx2=0.f;
    for (int l = c.indptr[i]; l < c.indptr[i + 1]; l++)
    {
      int j = c.indices[l];
      float w = c.weights[l];
      int d = c.idelays[l];
      int p1 = (th - d) & c.horizon_minus_1;
      int p2 = (th - d + 1) & c.horizon_minus_1;
      cx1 += w * c.buf[j*c.horizon + p1];
      cx2 += w * c.buf[j*c.horizon + p2];
    }
    c.cx1[i] = cx1;
    c.cx2[i] = cx2;
  }
}

export void cx_all3(the_conn &c, uniform int t)
{
  uniform int th = t + c.horizon;
  foreach (i = 0 ... c.num_node)
  {
    float cx1=0.f, cx2=0.f;
    for (int l = c.indptr[i]; l < c.indptr[i + 1]; l++)
    {
      int j = c.indices[l];
      float w = c.weights[l];
      int d = c.idelays[l];
      int p1 = (th - d) & c.horizon_minus_1;
      int p2 = (th - d + 1) & c.horizon_minus_1;
      cx1 += w * c.buf[j + p1*c.num_node];
      cx2 += w * c.buf[j + p2*c.num_node];
    }
    c.cx1[i] = cx1;
    c.cx2[i] = cx2;
  }
}

export void cx_all_nop(the_conn &c, uniform int t)
{
  // just to test overhead of binding layer
  return;
}

void dfun(float &dx, float &dy, float x, float y, float cx, data_floats p)
{
  float a=p[0], tau=p[1], k=p[2];
  dx = tau*(x - x*x*x*0.333f + y);
  dy = (a + k*cx - x)*tau;
}

void save_trace(the_sim &s, const uniform int t_trace)
{
    uniform int i0 = t_trace * s.num_svar * s.num_node;
    work_floats out = s.state_trace + i0;
    foreach (i_node = 0 ... s.num_node)
    {
      out[i_node] = s.states[i_node];
      out[i_node+s.num_node] = s.states[i_node+s.num_node];
    }
}

export void run_batches(the_sim &s)
{
  return;
    RNGState rng;

    const uniform int num_trace = s.num_time / s.num_skip;
    for (uniform int t_trace=0; t_trace < num_trace; t_trace++)
    {
        const uniform int t_total = t_trace * s.num_skip;
        for (uniform int t_skip=0; t_skip<s.num_skip; t_skip++)
        {
          uniform int t = t_total + t_skip;
          uniform int wt = t & s.conn.horizon_minus_1;

          // per inter-domain connectivity
          cx_all(s.conn, t);

          // per domain
          foreach (i = 0 ... s.num_node)
          {
              float dx1, dy1, dx2, dy2;
              float x = s.states[i], y=s.states[i+s.num_node];
              dfun(dx1, dy1, x, y, s.conn.cx1[i], s.global_params.values);
              dfun(dx2, dy2, x+dx1*s.dt, y+dy1*s.dt, s.conn.cx2[i], s.global_params.values);
              s.states[i] = x + s.dt*0.5f*(dx1 + dx2);
              s.states[i+s.num_node] = y + s.dt*0.5f*(dy1 + dy2);

              s.conn.buf[wt + i*s.num_node] = s.states[i];
          }
        }

        save_trace(s, t_trace);
    }
}
