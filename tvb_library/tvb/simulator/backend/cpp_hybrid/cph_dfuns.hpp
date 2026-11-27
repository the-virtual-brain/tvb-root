#pragma once
// Hand-written SIMD model dfuns, generic dfun dispatch, state clamping, and
// subnetwork container (`cph::subnet<W>`) for tvb cpp_hybrid.
//
// Pure C++17 header (no nanobind headers). Compiled into the single _core.cpp
// LTO translation unit.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "cph_monitors.hpp"

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// Portability: GCC/Clang inline hints; MSVC ignores them (attributes are
// only optimization hints here, not required for correctness).
#if defined(_MSC_VER) && !defined(__clang__)
#ifndef CPH_INLINE_HINT
#define CPH_INLINE_HINT inline
#endif
#ifndef CPH_NOINLINE
#define CPH_NOINLINE
#endif
#else
#ifndef CPH_INLINE_HINT
#define CPH_INLINE_HINT __attribute__((always_inline)) inline
#endif
#ifndef CPH_NOINLINE
#define CPH_NOINLINE __attribute__((noinline))
#endif
#endif
#ifndef INLINE
#define INLINE CPH_INLINE_HINT
#endif

// ---- build-time generated ("built-in") generic dfuns -----------------------
//
// When the package build ran dfungen (CMakeLists.txt, TVB_CPP_GENERATE_MODELS),
// the generated kernels and their metadata table are compiled into this module
// and CPH_HAVE_BUILTIN_GEN is defined.  What follows redeclares the entry
// points dfungen.emit_sources emits; the declarations must stay identical to
// that emission.  cph_builtin_dfun returns 1 when it handled the model id and
// 0 when the id is not in the built-in table, which is what lets dfun_dispatch
// fall back to the runtime-injected dfun for models the build did not know
// about.  Without the macro nothing is declared and dispatch stays exactly as
// it was: the runtime pointer only.
#if defined(CPH_HAVE_BUILTIN_GEN) && !defined(CPH_GEN_ENTRY_DEFINED)
#define CPH_GEN_ENTRY_DEFINED 1
extern "C" {
typedef struct cph_gen_entry {
  const char *name;               /* model class name */
  int mid;                        /* model id (>= 100) */
  int n_parm, n_cvar, n_svar;     /* buffer shapes the kernel expects */
  const char *const *parm_names;  /* parr packing order */
  const char *signature;          /* dfun fingerprint (model_dfun_signature) */
} cph_gen_entry;
int cph_builtin_dfun(int model_id, float *dx, const float *x, int node,
                     int mode, int n_node, int n_modes, const float *c,
                     const float *p, int n_parm, int Wn);
int cph_builtin_count(void);
const cph_gen_entry *cph_builtin_entry(int idx);
}
#endif

namespace cph {

// Runtime-injected generic dfun (the ctypes fallback library compiled by
// dfungen.generate_lib).  int-returning to match the entry dfungen emits: 1
// when the id is in the runtime library's table, 0 when it is not (that is
// what lets dfun_dispatch hard-fail instead of silently never-evolving when
// neither table covers the id).
typedef int (*cph_generic_fn_t)(int, float *, const float *, int, int, int,
                                int, const float *, const float *, int, int);
inline cph_generic_fn_t cph_generic_fn = nullptr;

// ---- model dfuns -----------------------------------------------------------
// x layout (n_svar, n_node, W): x[(svar*n_node + node)*W + i]
// c: (n_cvar, n_node, W) coupling per cvar for this subnet
// p: (n_node, n_parm, W): p[(node*n_parm + k)*W + i]
// dx same layout as x. Clamping applied after integration.

static constexpr int MOD_MPR = 0;
static constexpr int MOD_G2D = 1;      // Generic2dOscillator
static constexpr int MOD_KURAMOTO = 2;
static constexpr int MOD_SUPHOPF = 3;
static constexpr int MOD_LINEAR = 4;
static constexpr int MOD_RWW = 5;      // ReducedWongWang
static constexpr int MOD_WC = 6;       // WilsonCowan
static constexpr int MOD_JR = 7;       // JansenRit
static constexpr int MOD_EPI = 8;      // Epileptor
static constexpr int MOD_EPI2D = 9;    // Epileptor2D
static constexpr int MOD_ZER1 = 10;    // ZerlautAdaptationFirstOrder
static constexpr int MOD_ZER2 = 11;    // ZerlautAdaptationSecondOrder
static constexpr int MOD_CRBL = 12;    // CerebellarMF

// n_svar / n_parm per model (param order = the nb_hybrid gufunc arg order)
INLINE static void model_dims(int model_id, int &n_svar, int &n_parm,
                              int &n_cvar) {
  switch (model_id) {
  case MOD_MPR:      n_svar = 2; n_parm = 7;  n_cvar = 2; break;
  case MOD_G2D:      n_svar = 2; n_parm = 12; n_cvar = 1; break;
  case MOD_KURAMOTO: n_svar = 1; n_parm = 1;  n_cvar = 1; break;
  case MOD_SUPHOPF:  n_svar = 2; n_parm = 2;  n_cvar = 2; break;
  case MOD_LINEAR:   n_svar = 1; n_parm = 1;  n_cvar = 1; break;
  case MOD_RWW:      n_svar = 1; n_parm = 8;  n_cvar = 1; break;
  case MOD_WC:       n_svar = 2; n_parm = 23; n_cvar = 2; break;
  case MOD_JR:       n_svar = 6; n_parm = 13; n_cvar = 2; break;
  case MOD_EPI:      n_svar = 6; n_parm = 17; n_cvar = 2; break;
  case MOD_EPI2D:    n_svar = 2; n_parm = 12; n_cvar = 1; break;
  case MOD_ZER1:     n_svar = 5; n_parm = 49; n_cvar = 1; break;
  case MOD_ZER2:     n_svar = 8; n_parm = 50; n_cvar = 1; break;
  case MOD_CRBL:     n_svar = 5; n_parm = 84; n_cvar = 2; break;
  default: n_svar = 2; n_parm = 6; n_cvar = 1; break;
  }
}

// state access helper: (svar, node, mode, lane) layout
#define XOFF(sv) ((size_t)((sv) * n_node * n_modes + node * n_modes + mode) * W)
#define COFF(cv) ((size_t)((cv) * n_node * n_modes + node * n_modes + mode) * W)

// MontbrioPazoRoxin: svars (r, V); params (tau, I, Delta, J, eta, cr[, cv])
template <int W>
CPH_NOINLINE static void dfun_mpr(float *dx, const float *x, int node, int mode,
                            int n_node, int n_modes, const float *c,
                            const float *p, int n_parm, int n_cvar) {
  const float *r0 = x + XOFF(0);
  const float *V0 = x + XOFF(1);
  float *dr = dx + XOFF(0);
  float *dV = dx + XOFF(1);
  const float *c0 = c + COFF(0);
  const float *c1 = (n_cvar >= 2) ? (c + COFF(1)) : nullptr;
  const float *pk = p + (size_t)node * n_parm * W;
  for (int i = 0; i < W; i++) {
    float tau = pk[0 * W + i], I = pk[1 * W + i], Delta = pk[2 * W + i],
          J = pk[3 * W + i], eta = pk[4 * W + i], cr = pk[5 * W + i],
          cv = (n_parm >= 7) ? pk[6 * W + i] : 0.f;
    // Match MontbrioPazoRoxin.dfun / state_variable_dfuns: do not clamp inside
    // dfun (post-stage clamp_mpr enforces r >= 0 on xi and x).
    float r = r0[i];
    float V = V0[i];
    dr[i] = (1.f / tau) * (Delta / (M_PI_F * tau) + 2.f * r * V);
    dV[i] = (1.f / tau) * (V * V + eta + J * tau * r + I + cr * c0[i] +
                           (c1 ? cv * c1[i] : 0.f) -
                           M_PI_F * M_PI_F * r * r * tau * tau);
  }
}

template <int W>
CPH_NOINLINE static void clamp_mpr(float *x, int node, int mode, int n_node,
                             int n_modes) {
  float *r = x + XOFF(0);
  for (int i = 0; i < W; i++) r[i] = r[i] * (r[i] > 0.f);
}

// Generic2dOscillator: svars (V, W); cvar 1; params
// (tau, I, a, b, c, d, e, f, g, beta, alpha, gamma)
template <int W>
CPH_NOINLINE static void dfun_g2d(float *dx, const float *x, int node, int mode,
                            int n_node, int n_modes, const float *c,
                            const float *p, int n_parm) {
  const float *V0 = x + XOFF(0);
  const float *W0 = x + XOFF(1);
  float *dV = dx + XOFF(0);
  float *dW = dx + XOFF(1);
  const float *c0 = c + COFF(0);
  const float *pk = p + (size_t)node * n_parm * W;
  for (int i = 0; i < W; i++) {
    float tau = pk[0 * W + i], I = pk[1 * W + i], a = pk[2 * W + i],
          b = pk[3 * W + i], cc = pk[4 * W + i], d = pk[5 * W + i],
          e = pk[6 * W + i], f = pk[7 * W + i], g = pk[8 * W + i],
          beta = pk[9 * W + i], alpha = pk[10 * W + i], gamma = pk[11 * W + i];
    float V = V0[i], w = W0[i], V2 = V * V;
    dV[i] = d * tau * (alpha * w - f * V2 * V + e * V2 + g * V +
                       gamma * I + gamma * c0[i]);
    dW[i] = d * (a + b * V + cc * V2 - beta * w) / tau;
  }
}

// Kuramoto: 1 svar (theta); cvar 1; params (omega)
template <int W>
CPH_NOINLINE static void dfun_kuramoto(float *dx, const float *x, int node, int mode,
                                 int n_node, int n_modes, const float *c,
                                 const float *p, int n_parm) {
  const float *t0 = x + XOFF(0);
  float *dt = dx + XOFF(0);
  const float *c0 = c + COFF(0);
  const float *pk = p + (size_t)node * n_parm * W;
  for (int i = 0; i < W; i++) dt[i] = pk[0 * W + i] + c0[i];
}

// SupHopf: svars (x, y); cvar 2; params (a, omega)
template <int W>
CPH_NOINLINE static void dfun_suphopf(float *dx, const float *x, int node, int mode,
                                int n_node, int n_modes, const float *c,
                                const float *p, int n_parm) {
  const float *x0 = x + XOFF(0);
  const float *y0 = x + XOFF(1);
  float *dx0 = dx + XOFF(0);
  float *dy0 = dx + XOFF(1);
  const float *c0 = c + COFF(0);
  const float *c1 = c + COFF(1);
  const float *pk = p + (size_t)node * n_parm * W;
  for (int i = 0; i < W; i++) {
    float a = pk[0 * W + i], om = pk[1 * W + i];
    float xx = x0[i], yy = y0[i];
    float r2 = a - xx * xx - yy * yy;
    dx0[i] = r2 * xx - om * yy + c0[i];
    dy0[i] = r2 * yy + om * xx + c1[i];
  }
}

// Linear: 1 svar (x); cvar 1; params (gamma)
template <int W>
CPH_NOINLINE static void dfun_linear(float *dx, const float *x, int node, int mode,
                               int n_node, int n_modes, const float *c,
                               const float *p, int n_parm) {
  const float *x0 = x + XOFF(0);
  float *dx0 = dx + XOFF(0);
  const float *c0 = c + COFF(0);
  const float *pk = p + (size_t)node * n_parm * W;
  for (int i = 0; i < W; i++) dx0[i] = pk[0 * W + i] * x0[i] + c0[i];
}

// ReducedWongWang: 1 svar (S); cvar 1;
// params (a, b, d, gamma, tau_s, w, J_N, I_o)
template <int W>
CPH_NOINLINE static void dfun_rww(float *dx, const float *x, int node, int mode,
                            int n_node, int n_modes, const float *c,
                            const float *p, int n_parm) {
  const float *S0 = x + XOFF(0);
  float *dS = dx + XOFF(0);
  const float *c0 = c + COFF(0);
  const float *pk = p + (size_t)node * n_parm * W;
  for (int i = 0; i < W; i++) {
    float a = pk[0 * W + i], b = pk[1 * W + i], d = pk[2 * W + i],
          g = pk[3 * W + i], ts = pk[4 * W + i], w = pk[5 * W + i],
          j = pk[6 * W + i], io = pk[7 * W + i];
    float S = S0[i];
    float xx = w * j * S + io + j * c0[i];
    float h = (a * xx - b) / (1.f - expf(-d * (a * xx - b)));
    dS[i] = -(S / ts) + (1.f - S) * h * g;
  }
}

// WilsonCowan: svars (E, I); cvar 2; params
// (c_ee, c_ei, c_ie, c_ii, tau_e, tau_i, a_e, b_e, c_e, theta_e,
//  a_i, b_i, c_i, theta_i, r_e, r_i, k_e, k_i, P, Q, alpha_e, alpha_i,
//  shift_sigmoid)
template <int W>
CPH_NOINLINE static void dfun_wc(float *dx, const float *x, int node, int mode,
                           int n_node, int n_modes, const float *c,
                           const float *p, int n_parm) {
  const float *E0 = x + XOFF(0);
  const float *I0 = x + XOFF(1);
  float *dE = dx + XOFF(0);
  float *dI = dx + XOFF(1);
  const float *c0 = c + COFF(0);
  const float *pk = p + (size_t)node * n_parm * W;
  for (int i = 0; i < W; i++) {
    float c_ee = pk[0 * W + i], c_ei = pk[1 * W + i], c_ie = pk[2 * W + i],
          c_ii = pk[3 * W + i], tau_e = pk[4 * W + i], tau_i = pk[5 * W + i],
          a_e = pk[6 * W + i], b_e = pk[7 * W + i], c_e = pk[8 * W + i],
          theta_e = pk[9 * W + i], a_i = pk[10 * W + i], b_i = pk[11 * W + i],
          c_i = pk[12 * W + i], theta_i = pk[13 * W + i], r_e = pk[14 * W + i],
          r_i = pk[15 * W + i], k_e = pk[16 * W + i], k_i = pk[17 * W + i],
          P = pk[18 * W + i], Q = pk[19 * W + i], alpha_e = pk[20 * W + i],
          alpha_i = pk[21 * W + i], shift = pk[22 * W + i];
    float E = E0[i], I = I0[i];
    float x_e = alpha_e * (c_ee * E - c_ei * I + P - theta_e + c0[i]);
    float x_i = alpha_i * (c_ie * E - c_ii * I + Q - theta_i);
    float s_e, s_i;
    if (shift > 0.5f) {
      s_e = c_e * (1.f / (1.f + expf(-a_e * (x_e - b_e))) -
                   1.f / (1.f + expf(-a_e * -b_e)));
      s_i = c_i * (1.f / (1.f + expf(-a_i * (x_i - b_i))) -
                   1.f / (1.f + expf(-a_i * -b_i)));
    } else {
      s_e = c_e / (1.f + expf(-a_e * (x_e - b_e)));
      s_i = c_i / (1.f + expf(-a_i * (x_i - b_i)));
    }
    dE[i] = (-E + (k_e - r_e * E) * s_e) / tau_e;
    dI[i] = (-I + (k_i - r_i * I) * s_i) / tau_i;
  }
}

// JansenRit: 6 svars (y0..y5); cvar 2 (slot 0 used);
// params (nu_max, r, v0, a, a_1, a_2, a_3, a_4, A, b, B, J, mu)
template <int W>
CPH_NOINLINE static void dfun_jr(float *dx, const float *x, int node, int mode,
                           int n_node, int n_modes, const float *c,
                           const float *p, int n_parm) {
  const float *c0 = c + COFF(0);
  const float *pk = p + (size_t)node * n_parm * W;
  for (int i = 0; i < W; i++) {
    float nu_max = pk[0 * W + i], r = pk[1 * W + i], v0 = pk[2 * W + i],
          a = pk[3 * W + i], a_1 = pk[4 * W + i], a_2 = pk[5 * W + i],
          a_3 = pk[6 * W + i], a_4 = pk[7 * W + i], A = pk[8 * W + i],
          b = pk[9 * W + i], B = pk[10 * W + i], J = pk[11 * W + i],
          mu = pk[12 * W + i];
    float y0 = x[XOFF(0) + i], y1 = x[XOFF(1) + i], y2 = x[XOFF(2) + i],
          y3 = x[XOFF(3) + i], y4 = x[XOFF(4) + i], y5 = x[XOFF(5) + i];
    float sig12 = 2.f * nu_max / (1.f + expf(r * (v0 - (y1 - y2))));
    float sig01 = 2.f * nu_max / (1.f + expf(r * (v0 - (a_1 * J * y0))));
    float sig03 = 2.f * nu_max / (1.f + expf(r * (v0 - (a_3 * J * y0))));
    dx[XOFF(0) + i] = y3;
    dx[XOFF(1) + i] = y4;
    dx[XOFF(2) + i] = y5;
    dx[XOFF(3) + i] = A * a * sig12 - 2.f * a * y3 - a * a * y0;
    dx[XOFF(4) + i] =
        A * a * (mu + a_2 * J * sig01 + c0[i]) - 2.f * a * y4 - a * a * y1;
    dx[XOFF(5) + i] = B * b * (a_4 * J * sig03) - 2.f * b * y5 - b * b * y2;
  }
}

// Epileptor: 6 svars; cvar 2; params
// (x0, Iext, Iext2, a, b, slope, tt, Kvf, c, d, r, Ks, Kf, aa, bb, tau,
//  modification)
template <int W>
CPH_NOINLINE static void dfun_epi(float *dx, const float *x, int node, int mode,
                            int n_node, int n_modes, const float *c,
                            const float *p, int n_parm) {
  const float *cp1 = c + COFF(0);
  const float *cp2 = c + COFF(1);
  const float *pk = p + (size_t)node * n_parm * W;
  for (int i = 0; i < W; i++) {
    float x0 = pk[0 * W + i], Iext = pk[1 * W + i], Iext2 = pk[2 * W + i],
          a = pk[3 * W + i], b = pk[4 * W + i], slope = pk[5 * W + i],
          tt = pk[6 * W + i], Kvf = pk[7 * W + i], cc = pk[8 * W + i],
          d = pk[9 * W + i], r = pk[10 * W + i], Ks = pk[11 * W + i],
          Kf = pk[12 * W + i], aa = pk[13 * W + i], bb = pk[14 * W + i],
          tau = pk[15 * W + i], modification = pk[16 * W + i];
    float y0 = x[XOFF(0) + i], y1 = x[XOFF(1) + i], y2 = x[XOFF(2) + i],
          y3 = x[XOFF(3) + i], y4 = x[XOFF(4) + i], y5 = x[XOFF(5) + i];
    float tmp = (y0 < 0.f) ? (-a * y0 * y0 + b * y0)
                           : (slope - y3 + 0.6f * (y2 - 4.f) * (y2 - 4.f));
    float d0 = tt * (y1 - y2 + Iext + Kvf * cp1[i] + tmp * y0);
    float d1 = tt * (cc - d * y0 * y0 - y1);
    float d2i = (y2 < 0.f) ? (-0.1f * y2 * y2 * y2 * y2 * y2 * y2 * y2) : 0.f;
    float h = (modification > 0.5f)
        ? (x0 + 3.f / (1.f + expf(-(y0 + 0.5f) / 0.1f)))
        : (4.f * (y0 - x0) + d2i);
    float d2 = tt * (r * (h - y2 + Ks * cp1[i]));
    float d3 = tt * (-y4 + y3 - y3 * y3 * y3 + Iext2 + bb * y5 -
                     0.3f * (y2 - 3.5f) + Kf * cp2[i]);
    float d4i = (y3 < -0.25f) ? 0.f : aa * (y3 + 0.25f);
    float d4 = tt * ((-y4 + d4i) / tau);
    float d5 = tt * (-0.01f * (y5 - 0.1f * y0));
    dx[XOFF(0) + i] = d0;
    dx[XOFF(1) + i] = d1;
    dx[XOFF(2) + i] = d2;
    dx[XOFF(3) + i] = d3;
    dx[XOFF(4) + i] = d4;
    dx[XOFF(5) + i] = d5;
  }
}

// Epileptor2D: 2 svars (x1, z); cvar 1; params
// (x0, Iext, a, b, slope, c, d, r, Kvf, Ks, tt, modification)
template <int W>
CPH_NOINLINE static void dfun_epi2d(float *dx, const float *x, int node, int mode,
                              int n_node, int n_modes, const float *c,
                              const float *p, int n_parm) {
  const float *cp = c + COFF(0);
  const float *pk = p + (size_t)node * n_parm * W;
  for (int i = 0; i < W; i++) {
    float x0 = pk[0 * W + i], Iext = pk[1 * W + i], a = pk[2 * W + i],
          b = pk[3 * W + i], slope = pk[4 * W + i], cc = pk[5 * W + i],
          d = pk[6 * W + i], r = pk[7 * W + i], Kvf = pk[8 * W + i],
          Ks = pk[9 * W + i], tt = pk[10 * W + i],
          modification = pk[11 * W + i];
    float yy0 = x[XOFF(0) + i], yy1 = x[XOFF(1) + i];
    float tmp = (yy0 < 0.f)
        ? (a * yy0 * yy0 + (d - b) * yy0)
        : (-slope - 0.6f * (yy1 - 4.f) * (yy1 - 4.f) + d * yy0);
    float d0 = tt * (cc - yy1 + Iext + Kvf * cp[i] - tmp * yy0);
    float d1i = (yy1 < 0.f) ? (-0.1f * yy1 * yy1 * yy1 * yy1 * yy1 * yy1 * yy1)
                            : 0.f;
    float h = (modification > 0.5f)
        ? (x0 + 3.f / (1.f + expf(-(yy0 + 0.5f) / 0.1f)))
        : (4.f * (yy0 - x0) + d1i);
    float d1 = tt * (r * (h - yy1 + Ks * cp[i]));
    dx[XOFF(0) + i] = d0;
    dx[XOFF(1) + i] = d1;
  }
}

// ---- Zerlaut transfer-function pipeline (scalar, per-lane) ----
INLINE static float z_fluct_muV(float gL, float Cm, float Qe, float te, float Ee,
    float Qi, float ti, float Ei, float Nt, float pce, float pci, float g,
    float Ke, float Ki, float Fe, float Fi, float Fe_ext, float Fi_ext,
    float wad, float E_L, float &muV, float &sigV, float &tV) {
  float fe = (Fe + 1e-6f) * (1.f - g) * pce * Nt + Fe_ext * Ke;
  float fi = (Fi + 1e-6f) * g * pci * Nt + Fi_ext * Ki;
  float muGe = Qe * te * fe, muGi = Qi * ti * fi;
  float muG = gL + muGe + muGi;
  float Tm = Cm / muG;
  muV = (muGe * Ee + muGi * Ei + gL * E_L - wad) / muG;
  float Ue = Qe / muG * (Ee - muV);
  float Ui = Qi / muG * (Ei - muV);
  sigV = std::sqrt(fe * (Ue * te) * (Ue * te) / (2.f * (te + Tm)) +
                   fi * (Ui * ti) * (Ui * ti) / (2.f * (ti + Tm)));
  float tvn = fe * (Ue * te) * (Ue * te) + fi * (Ui * ti) * (Ui * ti);
  float tvd = fe * (Ue * te) * (Ue * te) / (te + Tm) +
              fi * (Ui * ti) * (Ui * ti) / (ti + Tm);
  tV = (tvd != 0.f) ? (tvn / tvd) : 1.f;
  return muV;
}
INLINE static float z_threshold(float muV, float sigV, float TvN, const float P[10]) {
  float V = (muV + 60.f) / 10.f;   // (muV - muV0)/DmuV0, muV0=-60, DmuV0=10
  float S = (sigV - 4.f) / 6.f;
  float T = (TvN - 0.5f) / 1.f;
  return P[0] + P[1]*V + P[2]*S + P[3]*T + P[4]*V*V + P[5]*S*S +
         P[6]*T*T + P[7]*V*S + P[8]*V*T + P[9]*S*T;
}
INLINE static float z_TF(float gL, float Cm, float Qe, float te, float Ee,
    float Qi, float ti, float Ei, float Nt, float pce, float pci, float g,
    float Ke, float Ki, float Fe, float Fi, float Fe_ext, float Fi_ext,
    float wad, float E_L, const float P[10]) {
  float muV, sigV, tV;
  z_fluct_muV(gL, Cm, Qe, te, Ee, Qi, ti, Ei, Nt, pce, pci, g, Ke, Ki,
              Fe, Fi, Fe_ext, Fi_ext, wad, E_L, muV, sigV, tV);
  float TvN = tV * gL / Cm;
  float Vth = z_threshold(muV, sigV, TvN, P) * 1000.f;  // V -> mV
  return std::erfc((Vth - muV) / (1.4142135623730951f * sigV)) / (2.f * tV);
}

// ZerlautAdaptationFirstOrder: svars (E,I,W_e,W_i,ou_drift)
template <int W>
CPH_NOINLINE static void dfun_zerlaut1(float *dx, const float *x, int node,
    int mode, int n_node, int n_modes, const float *c, const float *p, int n_parm) {
  const float *E0 = x + XOFF(0), *I0 = x + XOFF(1), *We0 = x + XOFF(2),
              *Wi0 = x + XOFF(3), *ou0 = x + XOFF(4);
  float *dE0 = dx+XOFF(0), *dI0 = dx+XOFF(1), *dWe0 = dx+XOFF(2),
        *dWi0 = dx+XOFF(3), *dou0 = dx+XOFF(4);
  const float *c0 = c + COFF(0);
  const float *pk = p + (size_t)node * n_parm * W;
  for (int i = 0; i < W; i++) {
    float gL =   pk[0*W+i], Cm = pk[3*W+i], Qe = pk[12*W+i], te = pk[14*W+i],
          eE =   pk[10*W+i],Qi = pk[13*W+i],ti = pk[15*W+i], eI = pk[11*W+i],
          Nt =   pk[16*W+i],pce= pk[17*W+i],pci= pk[18*W+i],g  = pk[19*W+i],
          Ke =   pk[20*W+i],Ki = pk[21*W+i],T  = pk[22*W+i],
          be =   pk[4*W+i], ae = pk[5*W+i], bi = pk[6*W+i], ai = pk[7*W+i],
          tw_e = pk[8*W+i],tw_i= pk[9*W+i],ELe= pk[1*W+i],ELi= pk[2*W+i],
          exex = pk[23*W+i],exin= pk[24*W+i],inex= pk[25*W+i],inin= pk[26*W+i],
          tou =   pk[27*W+i],wn = pk[28*W+i];
    float Pe[10], Pi[10];
    for (int k = 0; k < 10; k++) { Pe[k] = pk[(29+k)*W+i]; Pi[k] = pk[(39+k)*W+i]; }
    float E = E0[i], I = I0[i], We = We0[i], Wi = Wi0[i], ou = ou0[i];
    float Fe_ext = c0[i] + wn * ou;
    if (Fe_ext * Ke < 0.f) Fe_ext = 0.f;
    float Fi_ext = 0.f;
    float tf_e = z_TF(gL,Cm,Qe,te,eE,Qi,ti,eI,Nt,pce,pci,g,Ke,Ki,
                      E,I, Fe_ext+exex, Fi_ext+exin, We, ELe, Pe);
    float tf_i = z_TF(gL,Cm,Qe,te,eE,Qi,ti,eI,Nt,pce,pci,g,Ke,Ki,
                      E,I, Fe_ext+inex, Fi_ext+inin, Wi, ELi, Pi);
    dE0[i] = (tf_e - E) / T;
    dI0[i] = (tf_i - I) / T;
    float muV_e, muV_i, d1;
    z_fluct_muV(gL,Cm,Qe,te,eE,Qi,ti,eI,Nt,pce,pci,g,Ke,Ki,
                E,I, Fe_ext+exex, Fi_ext+exin, We, ELe, muV_e, d1, d1);
    dWe0[i] = -We/tw_e + be*E + ae*(muV_e - ELe)/tw_e;
    z_fluct_muV(gL,Cm,Qe,te,eE,Qi,ti,eI,Nt,pce,pci,g,Ke,Ki,
                E,I, Fe_ext+inex, Fi_ext+inin, Wi, ELi, muV_i, d1, d1);
    dWi0[i] = -Wi/tw_i + bi*I + ai*(muV_i - ELi)/tw_i;
    dou0[i] = -ou / tou;
  }
}

// ZerlautAdaptationSecondOrder: svars (E,I,C_ee,C_ei,C_ii,W_e,W_i,ou_drift)
template <int W>
CPH_NOINLINE static void dfun_zerlaut2(float *dx, const float *x, int node,
    int mode, int n_node, int n_modes, const float *c, const float *p, int n_parm) {
  const float *E0=x+XOFF(0), *I0=x+XOFF(1), *Cee0=x+XOFF(2), *Cei0=x+XOFF(3),
              *Cii0=x+XOFF(4), *We0=x+XOFF(5), *Wi0=x+XOFF(6), *ou0=x+XOFF(7);
  float *dE0=dx+XOFF(0), *dI0=dx+XOFF(1), *dCee0=dx+XOFF(2), *dCei0=dx+XOFF(3),
        *dCii0=dx+XOFF(4), *dWe0=dx+XOFF(5), *dWi0=dx+XOFF(6), *dou0=dx+XOFF(7);
  const float *c0 = c + COFF(0);
  const float *pk = p + (size_t)node * n_parm * W;
  for (int i = 0; i < W; i++) {
    float gL=pk[0*W+i], Cm=pk[3*W+i], Qe=pk[12*W+i], te=pk[14*W+i],
          eE=pk[10*W+i], Qi=pk[13*W+i], ti=pk[15*W+i], eI=pk[11*W+i],
          Nt=pk[16*W+i], pce=pk[17*W+i], pci=pk[18*W+i], g=pk[19*W+i],
          Ke=pk[20*W+i], Ki=pk[21*W+i], T=pk[22*W+i],
          be=pk[4*W+i], ae=pk[5*W+i], bi=pk[6*W+i], ai=pk[7*W+i],
          tw_e=pk[8*W+i], tw_i=pk[9*W+i], ELe=pk[1*W+i], ELi=pk[2*W+i],
          exex=pk[23*W+i], exin=pk[24*W+i], inex=pk[25*W+i], inin=pk[26*W+i],
          tou=pk[27*W+i], wn=pk[28*W+i], Si=pk[29*W+i];
    float Ne = Nt * (1.f - g), Ni = Nt * g;
    float Pe[10], Pi[10];
    for (int k = 0; k < 10; k++) { Pe[k] = pk[(30+k)*W+i]; Pi[k] = pk[(40+k)*W+i]; }
    float E=E0[i], I=I0[i], Cee=Cee0[i], Cei=Cei0[i], Cii=Cii0[i];
    float We=We0[i], Wi=Wi0[i], ou=ou0[i];
    float Fe_ext = c0[i] + wn * ou;
    if (Fe_ext * Ke < 0.f) Fe_ext = 0.f;
    float Fi_ext = 0.f;
    // helper closures via local lambdas
    auto TF_e = [&](float fe, float fi, float fext, float fixt, float Wx) {
      return z_TF(gL,Cm,Qe,te,eE,Qi,ti,eI,Nt,pce,pci,g,Ke,Ki, fe,fi,fext,fixt, Wx, ELe, Pe); };
    auto TF_i = [&](float fe, float fi, float fext, float fixt, float Wx) {
      return z_TF(gL,Cm,Qe,te,eE,Qi,ti,eI,Nt,pce,pci,g,Ke,Ki, fe,fi,fext,fixt, Wx, ELi, Pi); };
    float E_in_ex = Fe_ext + exex;   // exc external fe_input
    float E_in_in = exin;            // exc internal (constant) fi_input
    float I_in_ex = Fe_ext*Si + inex + wn*ou;  // inh external
    if (I_in_ex < 0.f) I_in_ex = 0.f;
    if (E_in_ex < 0.f) E_in_ex = 0.f;
    float I_in_in = inin;            // inh internal (constant)
    float _TF_e = TF_e(E, I, E_in_ex, E_in_in, We);
    float _TF_i = TF_i(E, I, I_in_ex, I_in_in, Wi);
    const float df = 1e-4f, q = 1e3f;
    auto dfe = [&](auto TF, float f_e, float f_i, float fx, float fy, float Wx) {
      return (TF(f_e+df, f_i, fx, fy, Wx) - TF(f_e-df, f_i, fx, fy, Wx)) / (2.f*df*q); };
    auto dfi = [&](auto TF, float f_e, float f_i, float fx, float fy, float Wx) {
      return (TF(f_e, f_i+df, fx, fy, Wx) - TF(f_e, f_i-df, fx, fy, Wx)) / (2.f*df*q); };
    auto d2ee = [&](auto TF, float f_e, float f_i, float fx, float fy, float Wx, float base) {
      return (TF(f_e+df, f_i, fx, fy, Wx) - 2.f*base + TF(f_e-df, f_i, fx, fy, Wx)) / ((df*q)*(df*q)); };
    auto d2ii = [&](auto TF, float f_e, float f_i, float fx, float fy, float Wx, float base) {
      return (TF(f_e, f_i+df, fx, fy, Wx) - 2.f*base + TF(f_e, f_i-df, fx, fy, Wx)) / ((df*q)*(df*q)); };
    auto d2fife = [&](auto TF, float f_e, float f_i, float fx, float fy, float Wx) {
      float dp = (TF(f_e+df, f_i+df, fx, fy, Wx) - TF(f_e+df, f_i-df, fx, fy, Wx)) / (2.f*df*q);
      float dm = (TF(f_e-df, f_i+df, fx, fy, Wx) - TF(f_e-df, f_i-df, fx, fy, Wx)) / (2.f*df*q);
      return (dp - dm) / (2.f*df*q); };
    auto d2fefi = [&](auto TF, float f_e, float f_i, float fx, float fy, float Wx) {
      float dp = (TF(f_e+df, f_i+df, fx, fy, Wx) - TF(f_e-df, f_i+df, fx, fy, Wx)) / (2.f*df*q);
      float dm = (TF(f_e+df, f_i-df, fx, fy, Wx) - TF(f_e-df, f_i-df, fx, fy, Wx)) / (2.f*df*q);
      return (dp - dm) / (2.f*df*q); };
    float dfe_TF_e = dfe(TF_e, E,I,E_in_ex,E_in_in,We);
    float dfe_TF_i = dfe(TF_i, E,I,I_in_ex,I_in_in,Wi);
    float dfi_TF_e = dfi(TF_e, E,I,E_in_ex,E_in_in,We);
    float dfi_TF_i = dfi(TF_i, E,I,I_in_ex,I_in_in,Wi);
    float d2fefe_e = d2ee(TF_e, E,I,E_in_ex,E_in_in,We,_TF_e);
    float d2fefe_i = d2ee(TF_i, E,I,I_in_ex,I_in_in,Wi,_TF_i);
    float d2fifi_e = d2ii(TF_e, E,I,E_in_ex,E_in_in,We,_TF_e);
    float d2fifi_i = d2ii(TF_i, E,I,I_in_ex,I_in_in,Wi,_TF_i);
    float d2fefi_e = d2fefi(TF_e, E,I,E_in_ex,E_in_in,We);
    float d2fife_e = d2fife(TF_e, E,I,E_in_ex,E_in_in,We);
    float d2fefi_i = d2fefi(TF_i, E,I,I_in_ex,I_in_in,Wi);
    float d2fife_i = d2fife(TF_i, E,I,I_in_ex,I_in_in,Wi);
    dE0[i] = (_TF_e - E + 0.5f*Cee*d2fefe_e + 0.5f*Cei*d2fefi_e + 0.5f*Cei*d2fife_e + 0.5f*Cii*d2fifi_e)/T;
    dI0[i] = (_TF_i - I + 0.5f*Cee*d2fefe_i + 0.5f*Cei*d2fefi_i + 0.5f*Cei*d2fife_i + 0.5f*Cii*d2fifi_i)/T;
    dCee0[i] = (_TF_e*(1.f/T - _TF_e)/Ne + (E - _TF_e)*(E - _TF_e)
                + 2.f*Cee*dfe_TF_e + 2.f*Cei*dfi_TF_e - 2.f*Cee)/T;
    // dC_ei: matches zerlaut.py derivative[3] after the 2nd-order c_ei
    // correction (upstream PR tvb-root#800): the four population-derivative
    // terms were swapped relative to Carlu et al. 2020 Eq. 17
    dCei0[i] = ((_TF_e - E)*(_TF_i - I) + Cee*dfe_TF_i + Cei*dfe_TF_e
                + Cei*dfi_TF_i + Cii*dfi_TF_e - 2.f*Cei)/T;
    dCii0[i] = (_TF_i*(1.f/T - _TF_i)/Ni + (I - _TF_i)*(I - _TF_i)
                + 2.f*Cii*dfi_TF_i + 2.f*Cei*dfe_TF_i - 2.f*Cii)/T;
    float muV_e, muV_i, d1;
    z_fluct_muV(gL,Cm,Qe,te,eE,Qi,ti,eI,Nt,pce,pci,g,Ke,Ki, E,I,E_in_ex,E_in_in,We,ELe,muV_e,d1,d1);
    dWe0[i] = -We/tw_e + be*E + ae*(muV_e - ELe)/tw_e;
    z_fluct_muV(gL,Cm,Qe,te,eE,Qi,ti,eI,Nt,pce,pci,g,Ke,Ki, E,I,I_in_ex,I_in_in,Wi,ELi,muV_i,d1,d1);
    dWi0[i] = -Wi/tw_i + bi*I + ai*(muV_i - ELi)/tw_i;
    dou0[i] = -ou / tou;
  }
}

// CerebellarMF: svars (GrC, GoC, MLI, PC, noise); cvars (mossy, parallel).
// Port of backend/templates/nb-cerebellar-dfun.py.mako (nb_hybrid reference).
// Param order (see backend.py _MODEL_PARM_NAMES["CerebellarMF"]):
//   0..49   runtime scalars: g_L_*, E_L_*, C_m_*, E_e, E_i, Q_*, tau_*, K_*,
//           N_*, alpha_*, T
//   50..61  tau_OU, weight_noise, external_input_*, frac_*, mf_to_*, pf_to_*
//   62..66  P_grc[0..4]   67..71 P_goc[0..4]   72..76 P_mli[0..4]
//   77..81  P_pc[0..4]
//   82 use_legacy_goc_e_e   83 add_noise_mli_pc
#define CRBL(p, k) ((double)p[(k) * W + i])

static double crbl_fluct_2d(double Fe, double Fi, double Fe_ext, double Fi_ext,
                            double Q_e, double tau_e, double Ee, double Q_i,
                            double tau_i, double Ei, double Gl, double Cm,
                            double El, double Ke, double Ki,
                            double *mu_V_out, double *sigma_V_out,
                            double *T_V_out, double *muGn_out) {
  double fe = (Fe + 1.0e-6) + Fe_ext;
  double fi = (Fi + 1.0e-6) + Fi_ext;
  double mu_Ge = Q_e * tau_e * fe * Ke;
  double mu_Gi = Q_i * tau_i * fi * Ki;
  double mu_G = Gl + mu_Ge + mu_Gi;
  double mu_V = (2.718281828459045 * (mu_Ge * Ee + mu_Gi * Ei + Gl * El)) / mu_G;
  double muGn = mu_G / Gl;
  double Tm = Cm / mu_G;
  double Ue = Q_e / mu_G * (Ee - mu_V);
  double Ui = Q_i / mu_G * (Ei - mu_V);
  double sVe = (2.0 * Tm + tau_e) *
      pow((2.718281828459045 * Ue * tau_e) / (2.0 * (tau_e + Tm)), 2) * Ke * fe;
  double sVi = (2.0 * Tm + tau_i) *
      pow((2.718281828459045 * Ui * tau_i) / (2.0 * (tau_i + Tm)), 2) * Ki * fi;
  double sigma_V = std::sqrt(sVe + sVi);
  fe += 1.0e-9;
  fi += 1.0e-9;
  double Tv_num = (Ke * fe * Ue * Ue * tau_e * tau_e * 2.718281828459045 * 2.718281828459045
                   + Ki * fi * Ui * Ui * tau_i * tau_i * 2.718281828459045 * 2.718281828459045);
  double Tv_den = (sigma_V + 1.0e-20) * (sigma_V + 1.0e-20);
  double Tv = 0.5 * Tv_num / Tv_den;
  double T_V = Tv * Gl / Cm;
  *mu_V_out = mu_V; *sigma_V_out = sigma_V; *T_V_out = T_V; *muGn_out = muGn;
  return 0.0;
}

static double crbl_fluct_3d(double Fe, double Fi, double Fe_ext,
                            double Qe_gr, double Te_gr, double Ee, double Qi,
                            double Ti, double Ei, double Gl, double Cm,
                            double El, double Ke_grc, double Ki,
                            double Ke_ext, double Qe_ext, double Te_ext,
                            double *mu_V_out, double *sigma_V_out,
                            double *T_V_out, double *muGn_out) {
  double fe_g = Fe + 1.0e-6;
  double fe_m = Fe_ext;
  double fi = Fi + 1.0e-6;
  double muGe_g = Qe_gr * Ke_grc * Te_gr * fe_g;
  double muGe_m = Qe_ext * Ke_ext * Te_ext * fe_m;
  double muGi = Qi * Ki * Ti * fi;
  double mu_G = Gl + muGe_g + muGe_m + muGi;
  double mu_V = (2.718281828459045 *
                 (muGe_g * Ee + muGe_m * Ee + muGi * Ei + Gl * El)) / mu_G;
  double muGn = mu_G / Gl;
  double Tm = Cm / mu_G;
  double Ue_g = Qe_gr / mu_G * (Ee - mu_V);
  double Ue_m = Qe_ext / mu_G * (Ee - mu_V);
  double Ui = Qi / mu_G * (Ei - mu_V);
  double sVe_g = (2.0 * Tm + Te_gr) *
      pow((2.718281828459045 * Ue_g * Te_gr) / (2.0 * (Te_gr + Tm)), 2) * Ke_grc * fe_g;
  double sVe_m = (2.0 * Tm + Te_ext) *
      pow((2.718281828459045 * Ue_m * Te_ext) / (2.0 * (Te_ext + Tm)), 2) * Ke_ext * fe_m;
  double sVi = (2.0 * Tm + Ti) *
      pow((2.718281828459045 * Ui * Ti) / (2.0 * (Ti + Tm)), 2) * Ki * fi;
  double sigma_V = std::sqrt(sVe_g + sVe_m + sVi);
  fe_m += 1.0e-15;
  fe_g += 1.0e-15;
  fi += 1.0e-15;
  double Tv_num = (Ke_grc * fe_g * Ue_g * Ue_g * Te_gr * Te_gr * 2.718281828459045 * 2.718281828459045
                   + Ke_ext * fe_m * Ue_m * Ue_m * Te_ext * Te_ext * 2.718281828459045 * 2.718281828459045
                   + Ki * fi * Ui * Ui * Ti * Ti * 2.718281828459045 * 2.718281828459045);
  double Tv_den = (sigma_V + 1.0e-20) * (sigma_V + 1.0e-20);
  double Tv = 0.5 * Tv_num / Tv_den;
  double T_V = Tv * Gl / Cm;
  *mu_V_out = mu_V; *sigma_V_out = sigma_V; *T_V_out = T_V; *muGn_out = muGn;
  return 0.0;
}

static double crbl_threshold(double muV, double sigmaV, double TvN, double muGn,
                             const double *P) {
  double V = (muV - (-60.0)) / 10.0;
  double S = (sigmaV - 4.0) / 6.0;
  double T = (TvN - 0.5) / 1.0;
  return P[0] + P[1] * V + P[2] * S + P[3] * T + P[4] * std::log(muGn);
}

static double crbl_firing_rate(double muV, double sigmaV, double TvN,
                               double Vthre, double Gl, double Cm,
                               double alpha) {
  return 0.5 / TvN * Gl / Cm
      * std::erfc((Vthre - muV) / (1.4142135623730951 * sigmaV)) * alpha;
}

static double crbl_TF_2d(double Fe, double Fi, double Fe_ext, double Fi_ext,
                         double Q_e, double tau_e, double Ee, double Q_i,
                         double tau_i, double Ei, double Gl, double Cm,
                         double El, double Ke, double Ki, double alpha,
                         const double *P) {
  double mu_V, sigma_V, T_V, muGn;
  crbl_fluct_2d(Fe, Fi, Fe_ext, Fi_ext, Q_e, tau_e, Ee, Q_i, tau_i, Ei,
                Gl, Cm, El, Ke, Ki, &mu_V, &sigma_V, &T_V, &muGn);
  double V_thre = crbl_threshold(mu_V, sigma_V, T_V, muGn, P) * 1000.0;
  return crbl_firing_rate(mu_V, sigma_V, T_V, V_thre, Gl, Cm, alpha);
}

static double crbl_TF_3d(double Fe, double Fi, double Fe_ext,
                         double Qe_gr, double Te_gr, double Ee, double Qi,
                         double Ti, double Ei, double Gl, double Cm,
                         double El, double Ke_grc, double Ki, double Ke_ext,
                         double Qe_ext, double Te_ext, double alpha,
                         const double *P) {
  double mu_V, sigma_V, T_V, muGn;
  crbl_fluct_3d(Fe, Fi, Fe_ext, Qe_gr, Te_gr, Ee, Qi, Ti, Ei, Gl, Cm, El,
                Ke_grc, Ki, Ke_ext, Qe_ext, Te_ext,
                &mu_V, &sigma_V, &T_V, &muGn);
  double V_thre = crbl_threshold(mu_V, sigma_V, T_V, muGn, P) * 1000.0;
  return crbl_firing_rate(mu_V, sigma_V, T_V, V_thre, Gl, Cm, alpha);
}

// note: mu_V formula in the template subtracts XX (=0.0 in every call),
// hence it is omitted above.
template <int W>
CPH_NOINLINE static void dfun_crbl(float *dx, const float *x, int node, int mode,
                             int n_node, int n_modes, const float *c,
                             const float *p, int n_parm) {
  const float *GrC0 = x + XOFF(0);
  const float *GoC0 = x + XOFF(1);
  const float *MLI0 = x + XOFF(2);
  const float *PC0  = x + XOFF(3);
  const float *nz0  = x + XOFF(4);
  float *dGrC = dx + XOFF(0);
  float *dGoC = dx + XOFF(1);
  float *dMLI = dx + XOFF(2);
  float *dPC  = dx + XOFF(3);
  float *dnz  = dx + XOFF(4);
  const float *c_mossy = c + COFF(0);
  const float *c_par   = c + COFF(1);
  const float *pk = p + (size_t)node * n_parm * W;
  for (int i = 0; i < W; i++) {
    double GrC = GrC0[i], GoC = GoC0[i], MLI = MLI0[i], PC = PC0[i],
           noise = nz0[i];
    double mossy = c_mossy[i], parallel = c_par[i];
    bool legacy_ee = CRBL(pk, 82) != 0.0;
    bool add_nz    = CRBL(pk, 83) != 0.0;
    double P_grc[5], P_goc[5], P_mli[5], P_pc[5];
    for (int k = 0; k < 5; k++) {
      P_grc[k] = CRBL(pk, 62 + k);
      P_goc[k] = CRBL(pk, 67 + k);
      P_mli[k] = CRBL(pk, 72 + k);
      P_pc[k]  = CRBL(pk, 77 + k);
    }
    double gL_grc = CRBL(pk, 0), gL_goc = CRBL(pk, 1), gL_mli = CRBL(pk, 2),
           gL_pc = CRBL(pk, 3);
    double EL_grc = CRBL(pk, 4), EL_goc = CRBL(pk, 5), EL_mli = CRBL(pk, 6),
           EL_pc = CRBL(pk, 7);
    double Cm_grc = CRBL(pk, 8), Cm_goc = CRBL(pk, 9), Cm_mli = CRBL(pk, 10),
           Cm_pc = CRBL(pk, 11);
    double E_e = CRBL(pk, 12), E_i = CRBL(pk, 13);
    double Q_mf_grc = CRBL(pk, 14), Q_mf_goc = CRBL(pk, 15);
    double Q_grc_goc = CRBL(pk, 16), Q_grc_mli = CRBL(pk, 17),
           Q_grc_pc = CRBL(pk, 18);
    double Q_goc_grc = CRBL(pk, 19), Q_goc_goc = CRBL(pk, 20);
    double Q_mli_mli = CRBL(pk, 21), Q_mli_pc = CRBL(pk, 22);
    double tau_mf_grc = CRBL(pk, 23), tau_mf_goc = CRBL(pk, 24);
    double tau_grc_goc = CRBL(pk, 25), tau_grc_mli = CRBL(pk, 26),
           tau_grc_pc = CRBL(pk, 27);
    double tau_goc_grc = CRBL(pk, 28), tau_goc_goc = CRBL(pk, 29);
    double tau_mli_mli = CRBL(pk, 30), tau_mli_pc = CRBL(pk, 31);
    double K_mossy_grc = CRBL(pk, 32), K_mossy_goc = CRBL(pk, 33);
    double K_grc_goc = CRBL(pk, 34), K_grc_mli = CRBL(pk, 35),
           K_grc_pc = CRBL(pk, 36);
    double K_goc_goc = CRBL(pk, 37), K_mli_mli = CRBL(pk, 38),
           K_mli_pc = CRBL(pk, 39);
    double alpha_grc = CRBL(pk, 45), alpha_goc = CRBL(pk, 46),
           alpha_mli = CRBL(pk, 47), alpha_pc = CRBL(pk, 48);
    double T = CRBL(pk, 49);
    double tau_OU = CRBL(pk, 50), weight_noise = CRBL(pk, 51);
    double eie = CRBL(pk, 52), eiin = CRBL(pk, 53), ein_ex = CRBL(pk, 54);
    double fm = CRBL(pk, 55), fp = CRBL(pk, 56);
    double mf_to_grc = CRBL(pk, 57), mf_to_goc = CRBL(pk, 58);
    double pf_to_goc = CRBL(pk, 59), pf_to_mli = CRBL(pk, 60),
           pf_to_pc = CRBL(pk, 61);

    double wn = weight_noise * noise;
    double Fe_tod1 = mossy * fm * mf_to_grc + wn;
    double Fe_tod2 = mossy * fm * mf_to_goc + parallel * fp * pf_to_goc + wn;
    double nz_mli_pc = add_nz ? wn : 0.0;
    double Fe_tod3 = parallel * fp * pf_to_mli + nz_mli_pc;
    double Fe_tod4 = parallel * fp * pf_to_pc + nz_mli_pc;

    if (Fe_tod1 * K_mossy_grc < 0.0) Fe_tod1 = 0.0;
    if (Fe_tod2 * K_mossy_goc < 0.0) Fe_tod2 = 0.0;
    if (Fe_tod3 * K_grc_mli  < 0.0) Fe_tod3 = 0.0;
    if (Fe_tod4 * K_grc_pc   < 0.0) Fe_tod4 = 0.0;

    double Fi_ext = 0.0;
    double goc_Ee = legacy_ee ? E_i : E_e;

    dGrC[i] = (float)((crbl_TF_2d(
        Fe_tod1 + eie, GoC, 0.0, Fi_ext + eiin,
        Q_mf_grc, tau_mf_grc, E_e, Q_goc_grc, tau_goc_grc, E_i,
        gL_grc, Cm_grc, EL_grc, K_mossy_grc, K_mossy_goc, alpha_grc,
        P_grc) - GrC) / T);
    dGoC[i] = (float)((crbl_TF_3d(
        GrC, GoC, Fe_tod2 + ein_ex,
        Q_grc_goc, tau_grc_goc, goc_Ee, Q_goc_goc, tau_goc_goc, E_i,
        gL_goc, Cm_goc, EL_goc, K_grc_goc, K_goc_goc,
        K_mossy_goc, Q_mf_goc, tau_mf_goc, alpha_goc,
        P_goc) - GoC) / T);
    dMLI[i] = (float)((crbl_TF_2d(
        GrC, MLI, Fe_tod3, Fi_ext,
        Q_grc_mli, tau_grc_mli, E_e, Q_mli_mli, tau_mli_mli, E_i,
        gL_mli, Cm_mli, EL_mli, K_grc_mli, K_mli_mli, alpha_mli,
        P_mli) - MLI) / T);
    dPC[i] = (float)((crbl_TF_2d(
        GrC, MLI, Fe_tod4, Fi_ext,
        Q_grc_pc, tau_grc_pc, E_e, Q_mli_pc, tau_mli_pc, E_i,
        gL_pc, Cm_pc, EL_pc, K_grc_pc, K_mli_pc, alpha_pc,
        P_pc) - PC) / T);
    dnz[i] = (float)(-noise / tau_OU);
  }
}
#undef CRBL
#undef XOFF
#undef COFF

template <int W>
CPH_NOINLINE static void dfun_dispatch(int model_id, float *dx, const float *x,
                                 int node, int mode, int n_node, int n_modes,
                                 const float *c, const float *p, int n_parm,
                                 int n_cvar, cph_generic_fn_t generic_fn) {
  if (model_id >= 100) {
    // generic (expression-generated) dfuns: x/c laid out node-major
    // (n_svar, n_node, n_modes, W); works for single- and multi-mode.
    // The kernels compiled into the extension at build time serve the id
    // first; only when that table does not cover it (user-defined models,
    // added after the build) does the runtime-injected library get a turn.
#ifdef CPH_HAVE_BUILTIN_GEN
    if (::cph_builtin_dfun(model_id, dx, x, node, mode, n_node, n_modes,
                           c, p, n_parm, W))
      return;
#endif
    // Unbound low-level callers retain the enable_generic_dfuns default;
    // backend-created runtime subnets always supply their owning pointer.
    if (!generic_fn) generic_fn = cph_generic_fn;
    if (generic_fn &&
        generic_fn(model_id, dx, x, node, mode, n_node, n_modes,
                            c, p, n_parm, W))
      return;
    // Neither table covered the id, so dx was never written: letting the run
    // continue would silently never-evolve (Sim.add_subnet(model_id=...) with
    // a stale or out-of-range id).  Fail loudly, naming the id.
    throw std::runtime_error("unknown generic model id " +
                             std::to_string(model_id));
  }
  switch (model_id) {
  case MOD_MPR: dfun_mpr<W>(dx, x, node, mode, n_node, n_modes, c, p, n_parm, n_cvar); break;
  case MOD_G2D: dfun_g2d<W>(dx, x, node, mode, n_node, n_modes, c, p, n_parm); break;
  case MOD_KURAMOTO: dfun_kuramoto<W>(dx, x, node, mode, n_node, n_modes, c, p, n_parm); break;
  case MOD_SUPHOPF: dfun_suphopf<W>(dx, x, node, mode, n_node, n_modes, c, p, n_parm); break;
  case MOD_LINEAR: dfun_linear<W>(dx, x, node, mode, n_node, n_modes, c, p, n_parm); break;
  case MOD_RWW: dfun_rww<W>(dx, x, node, mode, n_node, n_modes, c, p, n_parm); break;
  case MOD_WC: dfun_wc<W>(dx, x, node, mode, n_node, n_modes, c, p, n_parm); break;
  case MOD_JR: dfun_jr<W>(dx, x, node, mode, n_node, n_modes, c, p, n_parm); break;
  case MOD_EPI: dfun_epi<W>(dx, x, node, mode, n_node, n_modes, c, p, n_parm); break;
  case MOD_EPI2D: dfun_epi2d<W>(dx, x, node, mode, n_node, n_modes, c, p, n_parm); break;
  case MOD_ZER1: dfun_zerlaut1<W>(dx, x, node, mode, n_node, n_modes, c, p, n_parm); break;
  case MOD_ZER2: dfun_zerlaut2<W>(dx, x, node, mode, n_node, n_modes, c, p, n_parm); break;
  case MOD_CRBL: dfun_crbl<W>(dx, x, node, mode, n_node, n_modes, c, p, n_parm); break;
  default: throw std::runtime_error("unknown model id");
  }
}

template <int W>
CPH_NOINLINE static void clamp_dispatch(int model_id, float *x, int node, int n_node,
                                  int n_modes) {
  for (int j = 0; j < n_node; j++)
    for (int m = 0; m < n_modes; m++) {
      switch (model_id) {
      case MOD_MPR: clamp_mpr<W>(x, j, m, n_node, n_modes); break;
      default: break;
      }
    }
}

// ---- subnetwork ------------------------------------------------------------
// voispec, KMonKind, KMonState, and kmon<W> live in cph_monitors.hpp.

template <int W> struct subnet {
  int n_node = 0, n_svar = 2, n_parm = 0, n_cvar = 1, n_modes = 1;
  int model_id = MOD_MPR;
  cph_generic_fn_t generic_fn = nullptr;  // owning runtime library, if bound
  int integ_id = -1;  // -1 = inherit simw::integ_id; 0 = euler, 1 = heun
  uint32_t K = 1;     // steps per master tick: dt == K * dt0 (multi-dt)
  float dt = 0.001f;  // this subnet's own integration step (float32 of dt_j)
  std::vector<voispec> voi_specs;
  std::vector<float> x, dx, xi, dxi;  // (n_svar, n_node, n_modes, W)
  std::vector<float> c;               // (n_cvar, n_node, n_modes, W)
  std::vector<float> p;               // (n_node, n_parm, W)
  std::vector<float> buf;             // (n_svar, n_node, n_modes, H, W)
  std::vector<float> sv_lo, sv_hi;    // optional (n_svar,) state bounds
  std::vector<int> clamp_idx;         // optional clamped svar indices
  std::vector<float> clamp_val;       // (n_clamps, n_node, n_modes)
  uint32_t H = 0;

  void alloc(int n_node_, int n_svar_, int n_parm_, int n_cvar_, int model_id_,
             uint32_t H_, int n_modes_) {
    n_node = n_node_; n_svar = n_svar_; n_parm = n_parm_; n_cvar = n_cvar_;
    model_id = model_id_; H = H_; n_modes = n_modes_;
    x.assign((size_t)n_svar * n_node * n_modes * W, 0.f);
    dx.assign((size_t)n_svar * n_node * n_modes * W, 0.f);
    xi.assign((size_t)n_svar * n_node * n_modes * W, 0.f);
    dxi.assign((size_t)n_svar * n_node * n_modes * W, 0.f);
    c.assign((size_t)n_cvar * n_node * n_modes * W, 0.f);
    p.assign((size_t)n_node * n_parm * W, 0.f);
    buf.assign((size_t)n_svar * n_node * n_modes * H * W, 0.f);
    sv_lo.clear();
    sv_hi.clear();
    clamp_idx.clear();
    clamp_val.clear();
  }

  void apply_constraints(float *target) {
    clamp_dispatch<W>(model_id, target, 0, n_node, n_modes);
    if (sv_lo.size() == (size_t)n_svar && sv_hi.size() == (size_t)n_svar) {
      const size_t stride = (size_t)n_node * n_modes * W;
      for (int sv = 0; sv < n_svar; sv++) {
        const float lo = sv_lo[sv];
        const float hi = sv_hi[sv];
        const bool has_lo = std::isfinite(lo);
        const bool has_hi = std::isfinite(hi);
        if (!has_lo && !has_hi) continue;
        float *base = target + (size_t)sv * stride;
        for (size_t idx = 0; idx < stride; idx++) {
          float v = base[idx];
          if (has_lo && v < lo) v = lo;
          if (has_hi && v > hi) v = hi;
          base[idx] = v;
        }
      }
    }
    if (!clamp_idx.empty()) {
      for (size_t cpos = 0; cpos < clamp_idx.size(); cpos++) {
        const int sv = clamp_idx[cpos];
        for (int j = 0; j < n_node; j++) {
          for (int m = 0; m < n_modes; m++) {
            const float cv = clamp_val[
                (cpos * (size_t)n_node + (size_t)j) * (size_t)n_modes +
                (size_t)m];
            float *dst = target +
                (((size_t)sv * (size_t)n_node + (size_t)j) * (size_t)n_modes +
                 (size_t)m) * W;
            for (int i = 0; i < W; i++) dst[i] = cv;
          }
        }
      }
    }
  }

  void push_state(int t) {
    // copy per-(svar,node,mode) rows into history slot t
    const size_t rows = (size_t)n_svar * n_node * n_modes;
    for (size_t row = 0; row < rows; row++) {
      const size_t svar = row / ((size_t)n_node * n_modes);
      const size_t rem = row % ((size_t)n_node * n_modes);
      const float *srcv = x.data() + row * W;
      float *dst = buf.data() +
          ((svar * (size_t)n_node * n_modes + rem) * H +
           ((uint32_t)t & (H - 1))) * W;
#pragma omp simd
      for (int i = 0; i < W; i++) dst[i] = srcv[i];
    }
  }

  void step_euler(float dt, const float *noise) {
    for (int j = 0; j < n_node; j++)
      for (int m = 0; m < n_modes; m++)
        dfun_dispatch<W>(model_id, dx.data(), x.data(), j, m, n_node, n_modes,
                         c.data(), p.data(), n_parm, n_cvar, generic_fn);
    const size_t n = x.size();
    if (noise) {
      // noise layout (n_svar, n_node, n_modes, W) for this step
#pragma omp simd
      for (size_t k = 0; k < n; k++)
        x[k] += dt * dx[k] + noise[k];
    } else {
#pragma omp simd
      for (size_t k = 0; k < n; k++) x[k] += dt * dx[k];
    }
    apply_constraints(x.data());
  }

  void step_heun(float dt, const float *noise) {
    for (int j = 0; j < n_node; j++)
      for (int m = 0; m < n_modes; m++)
        dfun_dispatch<W>(model_id, dx.data(), x.data(), j, m, n_node, n_modes,
                         c.data(), p.data(), n_parm, n_cvar, generic_fn);
    const size_t n = x.size();
    if (noise) {
#pragma omp simd
      for (size_t k = 0; k < n; k++) xi[k] = x[k] + dt * dx[k] + noise[k];
    } else {
#pragma omp simd
      for (size_t k = 0; k < n; k++) xi[k] = x[k] + dt * dx[k];
    }
    apply_constraints(xi.data());
    // coupling stays fixed across stages (matches nb_hybrid)
    for (int j = 0; j < n_node; j++)
      for (int m = 0; m < n_modes; m++)
        dfun_dispatch<W>(model_id, dxi.data(), xi.data(), j, m, n_node, n_modes,
                         c.data(), p.data(), n_parm, n_cvar, generic_fn);
    if (noise) {
#pragma omp simd
      for (size_t k = 0; k < n; k++)
        x[k] += dt * 0.5f * (dx[k] + dxi[k]) + noise[k];
    } else {
#pragma omp simd
      for (size_t k = 0; k < n; k++) x[k] += dt * 0.5f * (dx[k] + dxi[k]);
    }
    apply_constraints(x.data());
  }
};

}  // namespace cph
