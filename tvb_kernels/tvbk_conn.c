#include "tvbk.h"

static void cx_all_j(const tvbk_conn *c, uint32_t t, uint32_t j) {
  float *const buf = c->buf + j * c->horizon;
  uint32_t th = t + c->horizon;
#pragma omp simd
  for (uint32_t l = c->indptr[j]; l < c->indptr[j + 1]; l++) {
    uint32_t i = c->indices[l];
    float w = c->weights[l];
    uint32_t d = c->idelays[l];
    uint32_t p1 = (th - d) & c->horizon_minus_1;
    uint32_t p2 = (th - d + 1) & c->horizon_minus_1;
    c->cx1[i] += w * buf[p1];
    c->cx2[i] += w * buf[p2];
  }
}

void tvbk_cx_j(const tvbk_conn *c, uint32_t t) {
#pragma omp simd
  for (int i = 0; i < c->num_node; i++)
    c->cx1[i] = c->cx2[i] = 0.0f;
  for (int j = 0; j < c->num_node; j++)
    cx_all_j(c, t, j);
}

void tvbk_cx_i(const tvbk_conn *c, uint32_t t) {
  uint32_t th = t + c->horizon;
#pragma omp simd
  for (int i = 0; i < c->num_node; i++) {
    float cx1 = 0.f, cx2 = 0.f;
    for (uint32_t l = c->indptr[i]; l < c->indptr[i + 1]; l++) {
      uint32_t j = c->indices[l];
      float w = c->weights[l];
      uint32_t d = c->idelays[l];
      uint32_t p1 = (th - d) & c->horizon_minus_1;
      uint32_t p2 = (th - d + 1) & c->horizon_minus_1;
      cx1 += w * c->buf[j * c->horizon + p1];
      cx2 += w * c->buf[j * c->horizon + p2];
    }
    c->cx1[i] = cx1;
    c->cx2[i] = cx2;
  }
}

void tvbk_cx_nop(const tvbk_conn *c, uint32_t t) {
  (void)c;
  (void)t;

  return;
}
