#pragma once

#include <stdint.h>

typedef struct tvbk_params tvbk_params;

struct tvbk_params {
  const int count;
  const float *values;
};

// conn model with csr format sparse connections & delay buffer
typedef struct tvbk_conn tvbk_conn;

struct tvbk_conn {
  const int num_node;
  const int num_nonzero;
  const int num_cvar;
  // horizon must be power of two
  const int horizon;
  const int horizon_minus_1;
  const float *weights; // (num_nonzero,)
  const uint32_t *indices;   // (num_nonzero,)
  const uint32_t *indptr;    // (num_nodes+1,)
  const uint32_t *idelays;   // (num_nonzero,)
  float *buf;           // delay buffer (num_cvar, num_nodes, horizon)
  float *cx1;
  float *cx2;
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

void tvbk_cx_j(const tvbk_conn *c, uint32_t t);
void tvbk_cx_i(const tvbk_conn *c, uint32_t t);
void tvbk_cx_nop(const tvbk_conn *c, uint32_t t);
