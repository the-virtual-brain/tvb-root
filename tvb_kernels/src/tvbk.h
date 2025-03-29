#pragma once

#include <stdint.h>

typedef struct tvbk_params tvbk_params;

struct tvbk_params {
  const uint32_t count;
  const float *const values;
};

/* a afferent coupling buffer into which the cx functions
   accumulate their results */
typedef struct tvbk_cx tvbk_cx;

struct tvbk_cx {
  /* values for 1st and 2nd Heun stage respectively.
     each shaped (num_node, ) */
  float *const cx1;
  float *const cx2;
  /* delay buffer (num_node, num_time)*/
  float *const buf;
  const uint32_t num_node;
  const uint32_t num_time; // horizon, power of 2
};

typedef struct tvbk_conn tvbk_conn;

struct tvbk_conn {
  const int num_node;
  const int num_nonzero;
  const int num_cvar;
  const float *const weights;    // (num_nonzero,)
  const uint32_t *const indices; // (num_nonzero,)
  const uint32_t *const indptr;  // (num_nodes+1,)
  const uint32_t *const idelays; // (num_nonzero,)
};

/* not currently used */
typedef struct tvbk_sim tvbk_sim;
struct tvbk_sim {
  // keep invariant stuff at the top, per sim stuff below
  const int rng_seed;
  const int num_node;
  const int num_svar;
  const int num_time;
  const int num_params;
  const int num_spatial_params;
  const float dt;
  const int oversample; // TODO "oversample" for stability,
  const int num_skip;   // skip per output sample
  float *z_scale;       // (num_svar), sigma*sqrt(dt)

  // parameters
  const tvbk_params global_params;
  const tvbk_params spatial_params;

  float *state_trace; // (num_time//num_skip, num_svar, num_nodes)
  float *states;      // full states (num_svar, num_nodes)

  const tvbk_conn conn;
};

void tvbk_cx_j(const tvbk_cx *cx, const tvbk_conn *conn, uint32_t t);
void tvbk_cx_i(const tvbk_cx *cx, const tvbk_conn *conn, uint32_t t);
void tvbk_cx_nop(const tvbk_cx *cx, const tvbk_conn *conn, uint32_t t);
