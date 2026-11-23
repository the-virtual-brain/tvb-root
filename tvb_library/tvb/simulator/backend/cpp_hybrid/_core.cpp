// C++ hybrid simulator core for TVB.
//
// Runtime-tight SIMD kernels with control flow in C++ (no static codegen).
// Layout follows the vendored tvbk approach: the sweep/batch dimension is the
// SIMD lane dimension (W), so model dfuns and coupling math vectorize across
// batch members. State per subnet: (n_svar, n_node, W) float32.
//
// Coupling semantics match nb_hybrid: the weighted delayed sum is computed
// once per step from the source history (slot t-1-delay), the post-cfun is
// applied, and the same coupling is used for both Heun stages.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>  // std::vector<> caster
// gil release via scoped release below

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cph_features.hpp"
#include "cph_monitors.hpp"
#include "cph_rng.hpp"
#include "cph_stim.hpp"

namespace nb = nanobind;

// nanobind 3.x dropped the legacy `array_t<T>` / `unchecked<N>()` API:
// NumPy arrays are `nb::ndarray<nb::numpy, T>` (dtype enforced at the
// binding boundary) and typed access is `view<nb::ndim<N>>()` with the same
// (k, i, ...) indexing.  Output arrays are backed by a capsule-owned buffer
// so the numpy view handed to Python keeps the memory alive (same lifetime
// semantics as the former binding layer's shape-constructed numpy arrays).
template <typename T>
static nb::ndarray<nb::numpy, T> make_owned_array(
    const std::vector<size_t> &shape) {
  size_t n = 1;
  for (size_t s : shape) n *= s;
  T *buf = new T[n]();
  nb::capsule owner(buf,
                    [](void *p) noexcept { delete[] static_cast<T *>(p); });
  return nb::ndarray<nb::numpy, T>(buf, shape.size(), shape.data(), owner);
}

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// Portability: GCC/Clang inline hints; MSVC ignores them (attributes are
// only optimization hints here, not required for correctness).
#if defined(_MSC_VER) && !defined(__clang__)
#define CPH_INLINE_HINT inline
#define CPH_NOINLINE
#else
#define CPH_INLINE_HINT __attribute__((always_inline)) inline
#define CPH_NOINLINE __attribute__((noinline))
#endif
#define INLINE CPH_INLINE_HINT
#include <cstring>

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
#ifdef CPH_HAVE_BUILTIN_GEN
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
static cph_generic_fn_t cph_generic_fn = nullptr;

// ---- coupling functions ----------------------------------------------------
// cfun params are (n_parm, W) float32: p[k*W + i] = parameter k of lane i.
// ids: -1 passthrough, 0 Linear(post), 1 Scaling(post), 2 Sigmoidal(post),
// 3 Difference(pre v0-xi, post a*gx), 4 Kuramoto(pre sin(v0-xi), post a*gx),
// 5 HyperbolicTangent(pre, post id), 6 SigmoidalJansenRit(pre v0-v1, post a*gx)
// 7 PreSigmoidal static (pre H*(Q+tanh(G*(P*v0-theta))), post id)

template <int W> CPH_NOINLINE static void cfun_post(int id,
                                              float *cx, const float *p) {
  switch (id) {
  case 0:  // Linear: a*gx + b
#pragma omp simd
    for (int i = 0; i < W; i++) cx[i] = p[0 * W + i] * cx[i] + p[1 * W + i];
    break;
  case 1:  // Scaling: a*gx
#pragma omp simd
    for (int i = 0; i < W; i++) cx[i] = p[0 * W + i] * cx[i];
    break;
  case 2:  // Sigmoidal: cmin + (cmax-cmin)/(1+exp(-a*(gx-midpoint)/sigma))
#pragma omp simd
    for (int i = 0; i < W; i++)
      cx[i] = (float)((double)p[3 * W + i] +
              ((double)p[4 * W + i] - (double)p[3 * W + i]) /
                  (1.0 + exp(-(double)p[0 * W + i] *
                             ((double)cx[i] - (double)p[2 * W + i]) /
                             (double)p[1 * W + i])));
    break;
  case 3: case 6:  // Difference / SJR: a*gx
#pragma omp simd
    for (int i = 0; i < W; i++) cx[i] = p[0 * W + i] * cx[i];
    break;
  case 4:  // Kuramoto: a * inv_N * gx
#pragma omp simd
    for (int i = 0; i < W; i++) cx[i] = p[0 * W + i] * p[1 * W + i] * cx[i];
    break;
  case 5: case 7: case -1: default:  // identity post
    break;
  }
}

// per-edge pre transform of the source value(s) (v = src cvar values)
template <int W>
CPH_NOINLINE static void cfun_pre(int id, float *v,
                                               const float *p,
                                               const float *xi) {
  switch (id) {
  case 3:  // Difference: v0 - xi
#pragma omp simd
    for (int i = 0; i < W; i++) v[0 * W + i] = v[0 * W + i] - xi[i];
    break;
  case 4:  // Kuramoto: sin(v0 - xi)
#pragma omp simd
    for (int i = 0; i < W; i++)
      v[0 * W + i] = (float)sin((double)v[0 * W + i] - (double)xi[i]);
    break;
  case 5:  // HyperbolicTangent: a*(1+tanh((b*v0-midpoint)/sigma))
    for (int i = 0; i < W; i++)
      // param order (nb_hybrid attr map): a=0, midpoint=1, sigma=2, b=3
      v[0 * W + i] = (float)((double)p[0 * W + i] *
                     (1.0 + tanh(((double)p[3 * W + i] * (double)v[0 * W + i] -
                                  (double)p[1 * W + i]) /
                                 (double)p[2 * W + i])));
    break;
  case 6:  // SJR: cmin + (cmax-cmin)/(1+exp(r*(midpoint-(v0-v1))))
    for (int i = 0; i < W; i++)
      v[0 * W + i] = (float)((double)p[4 * W + i] +
          ((double)p[5 * W + i] - (double)p[4 * W + i]) /
              (1.0 + exp((double)p[2 * W + i] *
                         ((double)p[6 * W + i] -
                          ((double)v[0 * W + i] - (double)v[1 * W + i])))));
    break;
  case 7:  // PreSigmoidal static: H*(Q+tanh(G*(P*v0-theta)))
    for (int i = 0; i < W; i++)
      v[0 * W + i] = (float)((double)p[0 * W + i] *
                     ((double)p[1 * W + i] +
                      tanh((double)p[2 * W + i] *
                           ((double)p[3 * W + i] * (double)v[0 * W + i] -
                            (double)p[4 * W + i]))));
    break;
  case 8:  // SigmoidalJansenRit legacy: a*2e0/(1+exp(r*(v0 - x)))
    for (int i = 0; i < W; i++)
      v[0 * W + i] = (float)((double)p[0 * W + i] * 2.0 * (double)p[1 * W + i] /
          (1.0 + exp((double)p[2 * W + i] *
                     ((double)p[3 * W + i] - (double)v[0 * W + i]))));
    break;
  case 9:  // PreSigmoidal dynamic: H*(Q+tanh(G*(P*v0 - v1)))
           // (v1 already replaced by the globalT per-mode mean when globalT)
    for (int i = 0; i < W; i++)
      v[0 * W + i] = (float)((double)p[0 * W + i] *
                     ((double)p[1 * W + i] +
                      tanh((double)p[2 * W + i] *
                           ((double)p[3 * W + i] * (double)v[0 * W + i] -
                            (double)v[1 * W + i]))));
    break;
  default: break;
  }
}

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
static constexpr int MOD_ZER1 = 10;   // ZerlautAdaptationFirstOrder
static constexpr int MOD_ZER2 = 11;    // ZerlautAdaptationSecondOrder
static constexpr int MOD_CRBL = 12;    // CerebellarMF   // ZerlautAdaptationSecondOrder

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

// ---- projections -----------------------------------------------------------

template <int W> struct proj {
  int src_sn = 0, tgt_sn = 0;
  int src_cvar = 0, tgt_cvar = 0;
  int cfun_id = 0;
  float scale = 1.0f;
  float ts = 1.0f;          // target_scales[0]
  int tgt_state_cvar = 0;   // target state var for per-edge pre(x_i)
  bool has_pre = false;     // ids 3..9
  std::vector<int> src_cvars;  // state vars summed into the weighted input
  std::vector<float> w;      // (nnz,)
  std::vector<uint32_t> idx; // (nnz,)
  std::vector<uint32_t> ptr; // (n_tgt+1,)
  std::vector<uint32_t> del; // (nnz,) delays in steps
  std::vector<float> cfp;    // (n_cfun_parm, W)
  int n_cfun_parm = 2;

  // coupling into target subnet's c[tgt_cvar], reading source history
  // Multi-dt read rule (pinned in parity_audit.md §6, decisions 1/3): with
  // the source stepping every K master ticks, the read position in
  // source-step units at master tick t1 (1-based) is tau = (t1-1)/K - delay
  // (the state at the master time of the start of the target's step, the
  // k=1 read generalized), i0 = floor(tau) and alpha = frac(tau) =
  // ((t1-1) mod K)/K.  C++ slots: slot s holds state after source step s+1,
  // so s0 = (i0-1) & (H-1) and s1 = i0 & (H-1) read steps i0 and i0+1.
  // Zero-delay edges have i1 = m+1, not pushed yet (m = newest pushed
  // step), so alpha is zeroed for them: the blend holds x0 exactly
  // (zero-order hold, decision 3).  Negative i0 wraps onto IC-prefilled
  // slots (all slots are IC-prefilled; H >= max_delay+1 keeps negative
  // reads from aliasing pushed steps).  When K == 1 this reduces to the
  // legacy single-slot read exactly (degenerate gate).
  void apply(const subnet<W> &src, subnet<W> &tgt, int t) const {
    const uint32_t Hm1 = src.H - 1;
    const uint32_t Ksrc = src.K;
    // Decision 3 (parity_audit.md §6): intra-subnet projections never
    // interpolate — they read the exact source step (t1/K) - 1 - delay
    // (the state at the start of the own step being integrated) at every
    // master tick; only inter-subnet reads with k_src > 1 interpolate.
    const bool intra = (src_sn == tgt_sn);
    const bool interp = Ksrc > 1 && !intra;
    const uint32_t t1 = (uint32_t)t + 1u;  // 1-based master tick
    const int nn = tgt.n_node;
    const int nms = src.n_modes, nmt = tgt.n_modes;
    if ((int)mode_map.size() != nms * nmt) {
      // lazily build identity when sizes changed
      mode_map.assign((size_t)nms * nmt, 0.f);
      for (int m = 0; m < nms && m < nmt; m++) mode_map[m * nmt + m] = 1.f;
    }
    float *out = tgt.c.data() + (size_t)tgt_cvar * nn * nmt * W;
    // globalT (PreSigmoidal dynamic, globalT=1): mean of the delayed
    // threshold (second source cvar) over ALL projection edges, per source
    // mode — matches nb_hybrid's PreSigmoidal.pre() global-threshold path.
    // Check all SIMD lanes so mixed globalT sweeps (e.g. lane 0 = 0, lane k = 1)
    // compute gthr whenever any lane enables globalT.
    bool any_gT = false;
    if (cfun_id == 9 && n_cfun_parm > 5 && src_cvars.size() >= 2) {
      for (int i = 0; i < W; i++) {
        if (cfp[(size_t)5 * W + i] != 0.f) {
          any_gT = true;
          break;
        }
      }
    }
    if (any_gT) {
      if (gthr_buf.size() < (size_t)nms * W)
        gthr_buf.resize((size_t)nms * W);
      float *gthr = gthr_buf.data();
      std::fill_n(gthr, (size_t)nms * W, 0.f);
      const int nnz_total = ptr.empty() ? 0 : (int)ptr.back();
      if (nnz_total > 0) {
        const int cv1 = src_cvars[1];
        const float *sbuf1 = src.buf.data() +
            (size_t)cv1 * src.n_node * nms * src.H * W;
        if (interp) {
          for (int nz = 0; nz < nnz_total; nz++) {
            // i0 is the source-STEP index of x0 (slot s holds step s+1,
            // so s0 = (i0-1) and s1 = i0 read steps i0 and i0+1)
            const uint32_t i0 = (t1 - 1u) / Ksrc - del[nz];
            // zero-delay edges: i1 = m+1 not pushed; hold x0 (decision 3)
            const float alpha = (del[nz] == 0)
                ? 0.f : (float)((t1 - 1u) % Ksrc) / (float)Ksrc;
            const uint32_t s0 = (i0 - 1u) & Hm1;
            const uint32_t s1 = i0 & Hm1;
            for (int m = 0; m < nms; m++) {
              const float *b0 = sbuf1 +
                  (((size_t)idx[nz] * nms + m) * src.H + s0) * W;
              const float *b1 = sbuf1 +
                  (((size_t)idx[nz] * nms + m) * src.H + s1) * W;
              for (int i = 0; i < W; i++)
                gthr[(size_t)m * W + i] += b0[i] + alpha * (b1[i] - b0[i]);
            }
          }
        } else {
          for (int nz = 0; nz < nnz_total; nz++) {
            // exact single-slot read (legacy K == 1, or intra multi-dt —
            // decision 3); see the slow-path branch below for the derivation
            const uint32_t slot = (t1 / Ksrc - 2u - del[nz]) & Hm1;
            for (int m = 0; m < nms; m++) {
              const float *b = sbuf1 +
                  (((size_t)idx[nz] * nms + m) * src.H + slot) * W;
              for (int i = 0; i < W; i++) gthr[(size_t)m * W + i] += b[i];
            }
          }
        }
        const float inv = 1.0f / (float)nnz_total;
        for (int m = 0; m < nms; m++)
          for (int i = 0; i < W; i++) gthr[(size_t)m * W + i] *= inv;
      }
    }
    // fast path: single source mode, no pre transform, one source cvar,
    // matching target — the dominant case (Linear.a * matvec).  Avoids the
    // per-edge v[] copy / mode loops for W==1 and openmps for wider W.
    // Requires a single-dt source (K == 1): its exact legacy slot is the
    // only read the fast path performs, for intra and inter alike.
    if (nms == 1 && nmt == 1 && !has_pre && src_cvars.size() == 1 &&
        Ksrc == 1) {
      const int scv = src_cvars[0];
      const float *sbuf = src.buf.data() +
          (size_t)scv * src.n_node * src.H * W;
      const float ts_scale = scale * ts * (float)mode_map[0];
      for (int j = 0; j < nn; j++) {
        float cx[W];
        for (int i = 0; i < W; i++) cx[i] = 0.f;
        const int i0 = ptr[j], i1 = ptr[j + 1];
        for (int nz = i0; nz < i1; nz++) {
          const uint32_t slot = ((uint32_t)t - 1u - del[nz]) & Hm1;
          const float wgt = w[nz];
          const float *b = sbuf + ((size_t)idx[nz] * src.H + slot) * W;
#pragma omp simd
          for (int i = 0; i < W; i++) cx[i] += wgt * b[i];
        }
        cfun_post<W>(cfun_id, cx, cfp.data());
        float *dst = out + (size_t)j * W;
#pragma omp simd
        for (int i = 0; i < W; i++) dst[i] += ts_scale * cx[i];
      }
      return;
    }
    // Lazily size scratch buffers outside node/edge loops (zero steady-state allocs).
    if (xi_buf.size() < (size_t)nmt * W) xi_buf.resize((size_t)nmt * W);
    if (cx_buf.size() < (size_t)nms * W) cx_buf.resize((size_t)nms * W);
    const size_t n_v = std::max<size_t>(src_cvars.size(), 2);
    if (v_buf.size() < n_v * W) v_buf.resize(n_v * W);
    float *xi = xi_buf.data();
    float *cx = cx_buf.data();
    float *v = v_buf.data();
    const float *gthr = gthr_buf.data();

    // mode_map: (nms, nmt) row-major
    for (int j = 0; j < nn; j++) {
      // target state for per-edge pre(x_i): current state at tgt_state_cvar
      for (int m = 0; m < nmt; m++) {
        const float *xv = tgt.x.data() +
            ((size_t)tgt_state_cvar * nn * nmt + j * nmt + m) * W;
        for (int i = 0; i < W; i++) xi[(size_t)m * W + i] = xv[i];
      }
      std::fill_n(cx, (size_t)nms * W, 0.f);
      const int i0 = ptr[j], i1 = ptr[j + 1];
      for (int nz = i0; nz < i1; nz++) {
        // value written at step (t - 1 - delay); slots are (step & H-1).
        // Multi-dt: interpolate the two bracketing source samples per the
        // pinned read rule (tau = (t1-1)/K - delay) in the apply() comment;
        // zero-delay edges hold x0 (i1 not pushed, alpha zeroed).
        uint32_t s0, s1;
        float alpha = 0.f;
        if (interp) {
          // ii is the source-STEP index of x0 (slot s holds step s+1,
          // so s0 = (ii-1) and s1 = ii read steps ii and ii+1)
          const uint32_t ii = (t1 - 1u) / Ksrc - del[nz];
          s0 = (ii - 1u) & Hm1;
          s1 = ii & Hm1;
          alpha = (del[nz] == 0)
              ? 0.f : (float)((t1 - 1u) % Ksrc) / (float)Ksrc;
        } else {
          // exact single-slot read: the legacy K == 1 read and the
          // multi-dt intra read (decision 3, step (t1/K) - 1 - delay)
          // share one formula in the C++ slot convention (slot s holds
          // step s+1); unwrapped negatives land on IC-prefilled slots
          s0 = s1 = (t1 / Ksrc - 2u - del[nz]) & Hm1;
        }
        const float wgt = w[nz];
        for (int m = 0; m < nms; m++) {
          for (size_t cv = 0; cv < src_cvars.size(); cv++) {
            const float *sbuf = src.buf.data() +
                (size_t)src_cvars[cv] * src.n_node * nms * src.H * W;
            const float *b0 = sbuf +
                (((size_t)idx[nz] * nms + m) * src.H + s0) * W;
            const float *b1 = sbuf +
                (((size_t)idx[nz] * nms + m) * src.H + s1) * W;
            for (int i = 0; i < W; i++)
              v[cv * W + i] = b0[i] + alpha * (b1[i] - b0[i]);
          }
          if (has_pre) {
            if (any_gT) {
              for (int i = 0; i < W; i++) {
                if (cfp[(size_t)5 * W + i] != 0.f)
                  v[W + i] = gthr[(size_t)m * W + i];
              }
            }
            const int xi_m = (m < nmt) ? m : 0;
            cfun_pre<W>(cfun_id, v, cfp.data(), xi + (size_t)xi_m * W);
            for (int i = 0; i < W; i++) cx[(size_t)m * W + i] += wgt * v[i];
          } else {
            for (int i = 0; i < W; i++) {
              float acc = 0.f;
              for (size_t cv = 0; cv < src_cvars.size(); cv++)
                acc += v[cv * W + i];
              cx[(size_t)m * W + i] += wgt * acc;
            }
          }
        }
      }
      // post per source mode, then mode-mix into target modes

      for (int m = 0; m < nms; m++)
        cfun_post<W>(cfun_id, cx + (size_t)m * W, cfp.data());

      for (int mt = 0; mt < nmt; mt++) {
        float *dst = out + ((size_t)j * nmt + mt) * W;
        for (int i = 0; i < W; i++) {
          float acc = 0.f;
          for (int m = 0; m < nms; m++)
            acc += cx[(size_t)m * W + i] * (float)mode_map[(size_t)m * nmt + mt];
          dst[i] += scale * ts * acc;
        }
      }
    }
  }
  mutable std::vector<float> mode_map;  // (nms, nmt) row-major
  mutable std::vector<float> gthr_buf;  // (nms * W) globalT scratch
  mutable std::vector<float> xi_buf;    // (nmt * W) target state scratch
  mutable std::vector<float> cx_buf;    // (nms * W) per-mode weighted sum scratch
  mutable std::vector<float> v_buf;     // (max(src_cvars, 2) * W) edge cvar scratch
};

// ---- Python / nanobind marshaling helpers ----------------------------------

template <typename T>
static T dict_get_or(const nb::dict &d, const char *key, T defval) {
  if (!d.contains(key)) return defval;
  nb::object val = d[key];
  if (val.is_none()) return defval;
  return nb::cast<T>(val);
}

static std::vector<int> to_int_vec(nb::handle h) {
  std::vector<int> out;
  if (h.is_none()) return out;
  try {
    auto arr = nb::cast<nb::ndarray<nb::numpy, const int32_t, nb::c_contig>>(h);
    out.assign(arr.data(), arr.data() + arr.size());
    return out;
  } catch (const std::exception &) {
  }
  try {
    auto arr = nb::cast<nb::ndarray<nb::numpy, const int64_t, nb::c_contig>>(h);
    out.resize(arr.size());
    const int64_t *p = arr.data();
    for (size_t i = 0; i < arr.size(); i++) out[i] = static_cast<int>(p[i]);
    return out;
  } catch (const std::exception &) {
  }
  auto seq = nb::cast<nb::sequence>(h);
  for (auto item : seq) out.push_back(nb::cast<int>(item));
  return out;
}

// Legacy ndarray conversions may narrow before the caller can check the index.
static std::vector<int> to_checked_indices(nb::handle h, size_t upper_bound,
                                           const char *label, const char *what) {
  const auto invalid = [label, what]() {
    return std::runtime_error(std::string(label) + ": invalid " + what);
  };
  const auto numpy = nb::module_::import_("numpy");
  if (nb::isinstance(h, numpy.attr("ndarray"))) {
    const auto kind = nb::cast<std::string>(h.attr("dtype").attr("kind"));
    if ((kind != "i" && kind != "u") || nb::cast<int>(h.attr("ndim")) != 1)
      throw invalid();
  } else if (!nb::isinstance<nb::sequence>(h)) {
    throw invalid();
  }
  const auto numpy_bool = numpy.attr("bool_");
  const auto seq = nb::borrow<nb::sequence>(h);
  if (nb::len(seq) > (size_t)std::numeric_limits<int>::max()) throw invalid();
  std::vector<int> out;
  for (auto item : seq) {
    if (PyBool_Check(item.ptr()) || nb::isinstance(item, numpy_bool) || !PyIndex_Check(item.ptr()))
      throw invalid();
    auto index = nb::steal<nb::object>(PyNumber_Index(item.ptr()));
    if (!index.is_valid()) throw nb::python_error();
    int overflow = 0;
    const long value = PyLong_AsLongAndOverflow(index.ptr(), &overflow);
    if (overflow || value < 0 || value > std::numeric_limits<int>::max() ||
        (size_t)value >= upper_bound)
      throw invalid();
    out.push_back(static_cast<int>(value));
  }
  return out;
}

static std::vector<uint64_t> to_u64_vec(nb::handle h) {
  std::vector<uint64_t> out;
  if (h.is_none()) return out;
  try {
    auto arr = nb::cast<nb::ndarray<nb::numpy, const uint64_t, nb::c_contig>>(h);
    out.assign(arr.data(), arr.data() + arr.size());
    return out;
  } catch (const std::exception &) {
  }
  try {
    auto arr = nb::cast<nb::ndarray<nb::numpy, const int64_t, nb::c_contig>>(h);
    out.resize(arr.size());
    const int64_t *p = arr.data();
    for (size_t i = 0; i < arr.size(); i++)
      out[i] = static_cast<uint64_t>(p[i]);
    return out;
  } catch (const std::exception &) {
  }
  try {
    auto arr = nb::cast<nb::ndarray<nb::numpy, const uint32_t, nb::c_contig>>(h);
    out.resize(arr.size());
    const uint32_t *p = arr.data();
    for (size_t i = 0; i < arr.size(); i++)
      out[i] = static_cast<uint64_t>(p[i]);
    return out;
  } catch (const std::exception &) {
  }
  auto seq = nb::cast<nb::sequence>(h);
  for (auto item : seq) out.push_back(nb::cast<uint64_t>(item));
  return out;
}

static std::vector<float> to_float_vec(nb::handle h) {
  std::vector<float> out;
  if (h.is_none()) return out;
  try {
    auto arr = nb::cast<nb::ndarray<nb::numpy, const float, nb::c_contig>>(h);
    out.assign(arr.data(), arr.data() + arr.size());
    return out;
  } catch (const std::exception &) {
  }
  try {
    auto arr = nb::cast<nb::ndarray<nb::numpy, const double, nb::c_contig>>(h);
    out.resize(arr.size());
    const double *p = arr.data();
    for (size_t i = 0; i < arr.size(); i++) out[i] = static_cast<float>(p[i]);
    return out;
  } catch (const std::exception &) {
  }
  auto seq = nb::cast<nb::sequence>(h);
  for (auto item : seq) out.push_back(nb::cast<float>(item));
  return out;
}

static std::vector<double> to_double_vec(nb::handle h) {
  std::vector<double> out;
  if (h.is_none()) return out;
  try {
    auto arr = nb::cast<nb::ndarray<nb::numpy, const double, nb::c_contig>>(h);
    out.assign(arr.data(), arr.data() + arr.size());
    return out;
  } catch (const std::exception &) {
  }
  try {
    auto arr = nb::cast<nb::ndarray<nb::numpy, const float, nb::c_contig>>(h);
    out.resize(arr.size());
    const float *p = arr.data();
    for (size_t i = 0; i < arr.size(); i++) out[i] = static_cast<double>(p[i]);
    return out;
  } catch (const std::exception &) {
  }
  auto seq = nb::cast<nb::sequence>(h);
  for (auto item : seq) out.push_back(nb::cast<double>(item));
  return out;
}

static void extract_2d_shape(nb::handle h, int &rows, int &cols) {
  try {
    auto arr = nb::cast<nb::ndarray<nb::numpy, const float, nb::c_contig>>(h);
    if (arr.ndim() != 2) throw std::runtime_error("expected 2D float32 array");
    rows = static_cast<int>(arr.shape(0));
    cols = static_cast<int>(arr.shape(1));
    return;
  } catch (const std::exception &) {
  }
  auto arr = nb::cast<nb::ndarray<nb::numpy, const double, nb::c_contig>>(h);
  if (arr.ndim() != 2) throw std::runtime_error("expected 2D float64 array");
  rows = static_cast<int>(arr.shape(0));
  cols = static_cast<int>(arr.shape(1));
}

static nb::dict kmon_state_to_dict(const KMonState &st) {
  nb::dict d;
  d["slot"] = st.slot;
  d["kind"] = st.kind;
  d["width"] = st.width;
  d["n_out_v"] = st.n_out_v;
  d["n_out_space"] = st.n_out_space;
  d["n_out_modes"] = st.n_out_modes;
  d["m_step"] = st.m_step;
  d["run_start"] = st.run_start;

  auto acc = make_owned_array<double>({st.accum.size()});
  if (!st.accum.empty())
    std::memcpy(acc.data(), st.accum.data(), st.accum.size() * sizeof(double));
  d["accum"] = acc;
  d["accum_count"] = st.accum_count;

  auto itm = make_owned_array<float>({st.interim.size()});
  if (!st.interim.empty())
    std::memcpy(itm.data(), st.interim.data(),
                st.interim.size() * sizeof(float));
  d["interim"] = itm;
  d["interim_count"] = st.interim_count;

  auto stk = make_owned_array<float>({st.stock.size()});
  if (!st.stock.empty())
    std::memcpy(stk.data(), st.stock.data(), st.stock.size() * sizeof(float));
  d["stock"] = stk;
  d["stock_pos"] = st.stock_pos;

  auto bws = make_owned_array<float>({st.bw_state.size()});
  if (!st.bw_state.empty())
    std::memcpy(bws.data(), st.bw_state.data(),
                st.bw_state.size() * sizeof(float));
  d["bw_state"] = bws;
  return d;
}

static KMonState dict_to_kmon_state(const nb::dict &d) {
  KMonState st;
  st.slot = dict_get_or<int>(d, "slot", -1);
  st.kind = dict_get_or<int>(d, "kind", 0);
  st.width = dict_get_or<int>(d, "width", 1);
  st.n_out_v = dict_get_or<int>(d, "n_out_v", 0);
  st.n_out_space = dict_get_or<int>(d, "n_out_space", 0);
  st.n_out_modes = dict_get_or<int>(d, "n_out_modes", 1);
  st.m_step = dict_get_or<uint64_t>(d, "m_step", 0);
  st.run_start = dict_get_or<uint64_t>(d, "run_start", 0);
  if (d.contains("accum")) st.accum = to_double_vec(d["accum"]);
  st.accum_count = dict_get_or<uint32_t>(d, "accum_count", 0);
  if (d.contains("interim")) st.interim = to_float_vec(d["interim"]);
  st.interim_count = dict_get_or<uint32_t>(d, "interim_count", 0);
  if (d.contains("stock")) st.stock = to_float_vec(d["stock"]);
  st.stock_pos = dict_get_or<uint32_t>(d, "stock_pos", 0);
  if (d.contains("bw_state")) st.bw_state = to_float_vec(d["bw_state"]);
  return st;
}

static KMonState dict_to_staged_state(const nb::dict &d, bool bw,
                                      const std::string &label) {
  for (const char *key : {"slot", "kind", "width", "n_out_v", "n_out_space",
                          "n_out_modes", "accum", "accum_count", "m_step", "run_start"}) {
    if (!d.contains(key))
      throw std::runtime_error(label + "incomplete snapshot state");
  }
  // Staged snapshots may carry the empty legacy fields emitted by export.
  for (const char *key : {"interim", "stock", "bw_state"}) {
    if (d.contains(key) && nb::len(nb::handle(d[key].ptr())) != 0)
      throw std::runtime_error(label + "unexpected legacy snapshot state");
  }
  for (const char *key : {"interim_count", "stock_pos"}) {
    if (d.contains(key) && nb::cast<uint32_t>(d[key]) != 0)
      throw std::runtime_error(label + "unexpected legacy snapshot state");
  }
  KMonState st;
  st.slot = nb::cast<int>(d["slot"]);
  st.kind = nb::cast<int>(d["kind"]);
  st.width = nb::cast<int>(d["width"]);
  st.n_out_v = nb::cast<int>(d["n_out_v"]);
  st.n_out_space = nb::cast<int>(d["n_out_space"]);
  st.n_out_modes = nb::cast<int>(d["n_out_modes"]);
  if (bw) {
    if (nb::len(nb::handle(d["accum"].ptr())) != 0)
      throw std::runtime_error(label + "unexpected legacy snapshot accumulator");
  } else {
    st.accum = to_double_vec(d["accum"]);
  }
  st.accum_count = nb::cast<uint32_t>(d["accum_count"]);
  st.m_step = nb::cast<uint64_t>(d["m_step"]);
  st.run_start = nb::cast<uint64_t>(d["run_start"]);
  return st;
}

using BwParameters = balloon_windkessel::Parameters;
static constexpr std::pair<const char *, double BwParameters::*> bw_parameter_fields[] = {
    {"signal_decay_s", &BwParameters::signal_decay_s},
    {"flow_feedback_s2", &BwParameters::flow_feedback_s2},
    {"transit_time_s", &BwParameters::transit_time_s},
    {"grubb_exponent", &BwParameters::grubb_exponent},
    {"resting_extraction", &BwParameters::resting_extraction},
    {"intravascular_ratio", &BwParameters::intravascular_ratio},
    {"frequency_offset_hz", &BwParameters::frequency_offset_hz},
    {"relaxation_rate_per_s", &BwParameters::relaxation_rate_per_s},
    {"echo_time_s", &BwParameters::echo_time_s},
    {"resting_blood_volume", &BwParameters::resting_blood_volume},
};

static void export_bw_stream_state(nb::dict &state, const KMonState &st) {
  auto regions = make_owned_array<double>({st.bw_stream_regions.size(), 6});
  double *row = regions.data();
  for (const auto &r : st.bw_stream_regions) {
    const double values[] = {r.drive_sum, r.bold_sum, r.ode.signal,
                             r.ode.inflow, r.ode.volume, r.ode.deoxy};
    row = std::copy(std::begin(values), std::end(values), row);
  }
  state["bw_stream_regions"] = regions;
  state["bw_drive_count"] = st.bw_drive_count;
}

static void parse_bw_stream_state(KMonState &st, const nb::dict &d, size_t n_node) {
  if (!d.contains("bw_stream_regions") || !d.contains("bw_drive_count"))
    throw std::runtime_error("global BW stream: incomplete snapshot state");
  auto regions = nb::cast<nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>>(
      d["bw_stream_regions"], false);
  if (regions.shape(0) != n_node || regions.shape(1) != 6)
    throw std::runtime_error("global BW stream: invalid snapshot region shape");
  st.bw_drive_count = nb::cast<uint32_t>(d["bw_drive_count"]);
  st.bw_stream_regions.resize(n_node);
  const double *row = regions.data();
  for (auto &r : st.bw_stream_regions) {
    r = {row[0], row[1], {row[2], row[3], row[4], row[5]}};
    row += 6;
  }
}

static constexpr const char *projection_label = "global complete window projection";

static auto projection_array(nb::handle h, size_t ndim, const char *name) {
  const auto numpy = nb::module_::import_("numpy");
  const auto invalid = [name]() {
    return std::runtime_error(std::string(projection_label) + ": invalid " + name);
  };
  if (!nb::isinstance(h, numpy.attr("ndarray"))) throw invalid();
  const auto kind = nb::cast<std::string>(h.attr("dtype").attr("kind"));
  auto a = nb::cast<nb::ndarray<nb::numpy>>(h, false);
  if ((kind != "f" && kind != "i" && kind != "u") || a.ndim() != ndim)
    throw invalid();
  return a;
}

template <int W>
static void configure_window_projection(kmon<W> &m, const nb::dict &d, int n_nodes,
                                        size_t n_voi) {
  if (!d.contains("gain") || d["gain"].is_none())
    throw std::runtime_error("global complete window projection: preconfigured gain required");
  if (d.contains("obsnoise") && !d["obsnoise"].is_none())
    throw std::runtime_error("global complete window projection: observation noise is unsupported");
  const auto shape = projection_array(d["gain"], 2, "gain");
  const size_t n_sensors = shape.shape(0);
  if (n_sensors == 0 || n_sensors > (size_t)std::numeric_limits<int>::max() ||
      shape.shape(1) != (size_t)n_nodes || n_sensors > m.gain.max_size() / (size_t)n_nodes ||
      n_sensors > std::min(m.accum.max_size(), m.d_out.max_size()) / n_voi)
    throw std::runtime_error("global complete window projection: invalid gain or output extent");
  using FloatArray = nb::ndarray<nb::numpy, const float, nb::c_contig>;
  FloatArray ref;
  if (d.contains("ref_vec") && !d["ref_vec"].is_none()) {
    const auto ref_shape = projection_array(d["ref_vec"], 1, "ref_vec");
    if (ref_shape.shape(0) != n_sensors)
      throw std::runtime_error("global complete window projection: ref_vec length != sensor count");
    ref = nb::cast<FloatArray>(d["ref_vec"]);
    // Reference weights are prepared by the caller, including any masked
    // contribution. Do not rescale the effective operator here.
    if (std::any_of(ref.data(), ref.data() + n_sensors,
                    [](float x) { return !std::isfinite(x); }))
      throw std::runtime_error("global complete window projection: nonfinite ref_vec");
  }
  const auto gain = nb::cast<FloatArray>(d["gain"]);
  if (std::any_of(gain.data(), gain.data() + n_sensors * (size_t)n_nodes,
                  [](float x) { return !std::isfinite(x); }))
    throw std::runtime_error("global complete window projection: nonfinite gain");
  m.set_projection(gain.data(), (int)n_sensors, n_nodes, ref.is_valid() ? ref.data() : nullptr,
                   ref.is_valid() ? (int)n_sensors : 0);
  if (!m.has_ref_vec) m.ref_vec.clear();  // Canonical identity for no reference.
}

static constexpr const char *local_window_label = "Recorder complete window: ";
static constexpr int recorder_slot_base = 10000;  // Python _RECORDER_SLOT_BASE

static bool local_window_kind(int kind) {
  return kind == KMON_TAVG || kind == KMON_GLOBAL_AVG || kind == KMON_SPATIAL_AVG;
}

static bool valid_recorder_clock(double clock, float native) {
  return std::isfinite(clock) && clock > 0.0 && clock <= std::numeric_limits<float>::max() &&
         native > 0.0f && (float)clock == native;
}

static KMonState dict_to_local_window_state(const nb::dict &d, size_t extent) {
  for (const char *key : {"slot", "kind", "width", "n_out_v", "n_out_space", "n_out_modes",
                          "m_step", "run_start", "accum_count"}) {
    if (!d.contains(key) || !PyLong_CheckExact(d[key].ptr()))
      throw std::runtime_error("Recorder complete window: invalid snapshot integer field");
  }
  using Accumulator = nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig>;
  Accumulator acc;
  if (!d.contains("accum") || !nb::try_cast(d["accum"], acc, false) || acc.size() != extent)
    throw std::runtime_error("Recorder complete window: invalid snapshot accumulator extent or dtype");
  return dict_to_staged_state(d, false, local_window_label);
}

template <int W>
static void parse_kmon_spec(nb::handle item, int n_src_v, int n_space,
                            int n_cvar, int n_modes, double default_dt,
                            const std::vector<kmon<W>> &prev,
                            kmon<W> &out, bool global = false) {
  int slot = -1;
  int kind = 0;
  if (nb::isinstance<nb::dict>(item)) {
    nb::dict d = nb::cast<nb::dict>(item);
    slot = nb::cast<int>(d["slot"]);
    kind = nb::cast<int>(d["kind"]);
    if (dict_get_or<bool>(d, "local_window", false))
      throw std::runtime_error("Recorder complete window requires a local attachment");
    const bool complete_window = dict_get_or<bool>(d, "complete_window", false);
    if (kind == KMON_BOLD_BW_STREAM) {
      if (!global || W != 1 || complete_window ||
          dict_get_or<uint32_t>(d, "cadence_K", 1u) != 1)
        throw std::runtime_error("global BW stream requires a global width 1 attachment");
      for (const char *key : {"istep", "dt", "voi", "bw_drive_baseline", "bw_drive_scale",
                              "bw_stream_params"})
        if (!d.contains(key))
          throw std::runtime_error("global BW stream: incomplete attachment specification");
      const double clock = nb::cast<double>(d["dt"]);
      if (!std::isfinite(clock) || clock <= 0 || clock > std::numeric_limits<float>::max() ||
          (float)clock != default_dt)
        throw std::runtime_error("global BW stream: incompatible neural clock");
      const auto voi = to_checked_indices(d["voi"], (size_t)n_src_v, "global BW stream",
                                          "observed VOI");
      if (voi.size() != 1)
        throw std::runtime_error("global BW stream: exactly one valid observed VOI is required");
      const auto parameters = nb::cast<nb::dict>(d["bw_stream_params"]);
      BwParameters p;
      if (parameters.size() != std::size(bw_parameter_fields))
        throw std::runtime_error("global BW stream: expected one explicit scalar parameter set");
      for (const auto &[name, member] : bw_parameter_fields) {
        if (!parameters.contains(name))
          throw std::runtime_error("global BW stream: incomplete parameter set");
        p.*member = nb::cast<double>(parameters[name]);
      }
      out.configure_bw_stream(slot, nb::cast<int>(d["istep"]), default_dt, n_space, voi[0],
                              nb::cast<double>(d["bw_drive_baseline"]),
                              nb::cast<double>(d["bw_drive_scale"]), p);
      return;
    }
    if (complete_window &&
        (!global || W != 1 || (kind != KMON_TAVG && kind != KMON_GLOBAL_AVG &&
                               kind != KMON_SPATIAL_AVG && kind != KMON_PROJECTION)))
      throw std::runtime_error("complete_window requires a supported global monitor at width 1");
    auto it = std::find_if(prev.begin(), prev.end(), [&](const kmon<W> &e) {
      return !e.staged_global() && !e.local_window && e.slot == slot && e.kind == kind;
    });
    if (!complete_window && it != prev.end()) {
      out = *it;
      return;
    }
    const int istep = dict_get_or<int>(d, "istep", 1);
    const uint32_t cadence_K = dict_get_or<uint32_t>(d, "cadence_K", 1u);
    const bool midpoint_time = dict_get_or<bool>(d, "midpoint_time", false);
    double m_dt = default_dt;
    if (d.contains("dt") && !d["dt"].is_none())
      m_dt = nb::cast<double>(d["dt"]);
    else if (d.contains("dt0") && !d["dt0"].is_none())
      m_dt = nb::cast<double>(d["dt0"]);
    const bool window_projection = complete_window && kind == KMON_PROJECTION;
    std::vector<int> voi;
    if (d.contains("voi") && !d["voi"].is_none())
      voi = window_projection
          ? to_checked_indices(d["voi"], (size_t)n_src_v, projection_label, "observed VOI")
          : to_int_vec(d["voi"]);

    if (complete_window &&
        (istep < 1 || cadence_K != 1 || !std::isfinite(m_dt) || m_dt <= 0.0 ||
         (window_projection && m_dt > std::numeric_limits<float>::max()) ||
         (float)m_dt != (float)default_dt || n_src_v < 1 || n_space < 1 ||
         std::any_of(voi.begin(), voi.end(),
                     [n_src_v](int v) { return v < 0 || v >= n_src_v; })))
      throw std::runtime_error("global complete window: invalid period, clock, or VOI");
    out.complete_window = complete_window;

    if (kind == KMON_BOLD) {
      const int ii = dict_get_or<int>(d, "interim_istep", 1);
      const int S = dict_get_or<int>(d, "stock_steps", 1);
      const double k1v0 = dict_get_or<double>(d, "k1v0", 1.0);
      const bool fo = dict_get_or<bool>(d, "first_order", false);
      std::vector<double> hrf;
      if (d.contains("hrf") && !d["hrf"].is_none())
        hrf = to_double_vec(d["hrf"]);
      out.configure_bold(slot, istep, ii, S, k1v0, fo, m_dt, n_src_v, n_space,
                         voi, hrf, cadence_K);
    } else if (kind == KMON_BOLD_BW) {
      const float bw_dt = dict_get_or<float>(d, "bw_dt", 0.0f);
      std::vector<float> bwp;
      if (d.contains("bw_params") && !d["bw_params"].is_none())
        bwp = to_float_vec(d["bw_params"]);
      const float *bwp_ptr = (bwp.size() >= 10) ? bwp.data() : nullptr;
      out.configure_bold_bw(slot, istep, m_dt, n_src_v, n_space, voi, bw_dt,
                            bwp_ptr, cadence_K);
    } else if (kind == KMON_SPATIAL_AVG) {
      if (d.contains("spatial_mean") && !d["spatial_mean"].is_none()) {
        int n_areas = 0, n_sn = 0;
        extract_2d_shape(d["spatial_mean"], n_areas, n_sn);
        std::vector<float> sm = to_float_vec(d["spatial_mean"]);
        if (complete_window &&
            (n_areas < 1 || n_sn != n_space ||
             std::any_of(sm.begin(), sm.end(), [](float x) { return !std::isfinite(x); })))
          throw std::runtime_error("global complete window: invalid spatial_mean");
        out.set_spatial_mean(sm.data(), n_areas, n_sn);
      } else if (d.contains("spatial_area_idx") &&
                 !d["spatial_area_idx"].is_none()) {
        std::vector<int> aidx = to_int_vec(d["spatial_area_idx"]);
        std::vector<float> ascale = to_float_vec(d["spatial_scale"]);
        const int n_areas = dict_get_or<int>(d, "n_areas", 0);
        if (complete_window &&
            (n_areas < 1 || aidx.size() != (size_t)n_space ||
             ascale.size() != aidx.size() ||
             std::any_of(aidx.begin(), aidx.end(),
                         [n_areas](int a) { return a < 0 || a >= n_areas; }) ||
             std::any_of(ascale.begin(), ascale.end(),
                         [](float x) { return !std::isfinite(x); })))
          throw std::runtime_error("global complete window: invalid compact spatial weights");
        out.set_spatial_compact(aidx.data(), ascale.data(), (int)aidx.size(),
                                n_areas);
      } else if (complete_window) {
        throw std::runtime_error("global complete window: preconfigured spatial weights required");
      }
      out.configure_basic(slot, kind, istep, m_dt, n_src_v, n_space, 1, voi,
                          cadence_K, midpoint_time);
    } else if (kind == KMON_PROJECTION) {
      if (window_projection) {
        configure_window_projection(out, d, n_space, voi.empty() ? (size_t)n_src_v : voi.size());
      } else if (d.contains("gain") && !d["gain"].is_none()) {
        int n_sensors = 0, n_pn = 0;
        extract_2d_shape(d["gain"], n_sensors, n_pn);
        std::vector<float> g = to_float_vec(d["gain"]);
        std::vector<float> rv;
        if (d.contains("ref_vec") && !d["ref_vec"].is_none())
          rv = to_float_vec(d["ref_vec"]);
        out.set_projection(g.data(), n_sensors, n_pn,
                           rv.empty() ? nullptr : rv.data(), (int)rv.size());
      }
      out.configure_basic(slot, kind, istep, m_dt, n_src_v, n_space, 1, voi,
                          cadence_K, midpoint_time);
    } else if (kind == KMON_CTAVG) {
      out.configure_basic(slot, kind, istep, m_dt, n_cvar, n_space, n_modes,
                          voi, cadence_K, midpoint_time);
    } else {
      out.configure_basic(slot, kind, istep, m_dt, n_src_v, n_space, 1, voi,
                          cadence_K, midpoint_time);
    }
    return;
  }

  // Legacy 10-tuple: (slot, kind, istep, interim_istep, stock_steps, k1v0,
  //                   first_order, voi, hrf, dt)
  nb::tuple a = nb::cast<nb::tuple>(item);
  slot = nb::cast<int>(a[0]);
  kind = nb::cast<int>(a[1]);
  if (kind == KMON_BOLD_BW_STREAM)
    throw std::runtime_error("global BW stream requires an explicit native dictionary");
  auto it = std::find_if(prev.begin(), prev.end(), [&](const kmon<W> &e) {
    return !e.staged_global() && !e.local_window && e.slot == slot && e.kind == kind;
  });
  if (it != prev.end()) {
    out = *it;
    return;
  }
  const int istep = nb::cast<int>(a[2]);
  const int ii = nb::cast<int>(a[3]);
  const int S = nb::cast<int>(a[4]);
  const double k1v0 = nb::cast<double>(a[5]);
  const bool fo = nb::cast<bool>(a[6]);
  std::vector<int> voi = to_int_vec(a[7]);
  std::vector<double> hrf = to_double_vec(a[8]);
  const double m_dt = nb::cast<double>(a[9]);
  if (kind == KMON_BOLD) {
    out.configure_bold(slot, istep, ii, S, k1v0, fo, m_dt, n_src_v, n_space,
                       voi, hrf, 1u);
  } else if (kind == KMON_BOLD_BW) {
    out.configure_bold_bw(slot, istep, m_dt, n_src_v, n_space, voi, 0.0f,
                          nullptr, 1u);
  } else if (kind == KMON_CTAVG) {
    out.configure_basic(slot, kind, istep, m_dt, n_cvar, n_space, n_modes, voi,
                        1u, false);
  } else {
    out.configure_basic(slot, kind, istep, m_dt, n_src_v, n_space, 1, voi, 1u,
                        false);
  }
}

// ---- simulator (one per width instantiation) -------------------------------

template <int W> struct simw {
  static constexpr int W_CONST = W;
  std::vector<subnet<W>> sn;
  std::vector<proj<W>> pr;
  int integ_id = 1;  // 0 = euler, 1 = heun
  float dt = 0.001f;
  int tavg_period = 1;
  std::vector<std::vector<voispec>> voi_specs;       // per subnet
  std::vector<std::vector<float>> obs_row;           // per-subnet (n_voi, n_node, W)
  std::vector<std::vector<kmon<W>>> mons;            // per-subnet monitor engines
  std::vector<GlobalGatherSubnet> global_gather;     // per-subnet global gather spec
  std::vector<kmon<W>> global_mons;                  // global / merged monitor engines
  std::vector<float> global_obs_row;                 // (total_vois, total_nodes, W)
  int global_total_vois = 0;
  int global_total_nodes = 0;
  std::vector<stim_spec<W>> stim_specs;              // C++ stimulus specs
  std::vector<rng_engine<W>> sub_rng;                // per-subnet Philox RNG engines
  std::vector<std::vector<float>> rng_scratch;       // per-subnet step noise buffer
  std::vector<uint8_t> online_enabled;               // per-subnet online feature flag
  std::vector<OnlineFeatureAccum<W>> online_accums;  // per-subnet Tier-1 accumulators
  uint64_t t_abs = 0;

  void validate_local_window_source(size_t si, const kmon<W> &m) const {
    const auto &sub = sn[si];
    if (sub.K != m.cadence_K || !valid_recorder_clock(m.dt, sub.dt) ||
        !valid_recorder_clock(m.local_master_dt, dt))
      throw std::runtime_error("Recorder complete window: incompatible source or master clock");
  }

  void configure_local_window(size_t si, const nb::dict &d, kmon<W> &m) const {
    const auto &sub = sn[si];
    const auto &source = voi_specs[si];
    const int slot = nb::cast<int>(d["slot"]);
    const int kind = nb::cast<int>(d["kind"]);
    const int period = nb::cast<int>(d["istep"]);
    const uint32_t cadence = nb::cast<uint32_t>(d["cadence_K"]);
    const double clock = nb::cast<double>(d["dt"]);
    const double master_clock = nb::cast<double>(d["master_dt"]);
    const double scheduled_dt = cadence * master_clock;
    const size_t max_int = (size_t)std::numeric_limits<int>::max();
    if (W != 1 || sub.n_modes != 1 || sub.n_node < 1 ||
        source.empty() || source.size() > max_int ||
        source.size() > obs_row[si].max_size() / (size_t)sub.n_node ||
        slot < recorder_slot_base || !local_window_kind(kind) || period < 1 || cadence == 0 ||
        dict_get_or<bool>(d, "complete_window", false))
      throw std::runtime_error("Recorder complete window: invalid attachment geometry, kind or period");
    // Match the existing NetworkSet cadence tolerance, and check both stored native clocks.
    if (cadence != sub.K || !valid_recorder_clock(clock, sub.dt) ||
        !valid_recorder_clock(master_clock, dt) ||
        std::abs(clock - scheduled_dt) > 1e-12 + 1e-9 * scheduled_dt)
      throw std::runtime_error("Recorder complete window: incompatible source or master clock");
    for (const auto &v : source) {
      if (v.kind < 0 || v.kind > 1 || v.a < 0 || v.a >= sub.n_svar ||
          v.b < 0 || v.b >= sub.n_svar)
        throw std::runtime_error("Recorder complete window: invalid observed VOI source");
    }
    const auto voi = to_checked_indices(d["voi"], source.size(), "Recorder complete window", "observed VOI");
    const size_t n_voi = voi.empty() ? source.size() : voi.size();
    const size_t max_space = std::min(m.accum.max_size(), m.d_out.max_size()) / n_voi;
    m.local_window = true;
    m.local_master_dt = master_clock;
    m.local_source_name = nb::cast<std::string>(d["source_name"]);
    m.local_node_indices = nb::cast<std::vector<int64_t>>(d["node_indices"]);
    if (!m.local_node_indices.empty() && m.local_node_indices.size() != (size_t)sub.n_node)
      throw std::runtime_error("Recorder complete window: incompatible node order extent");
    if (kind == KMON_SPATIAL_AVG) {
      using SpatialMean = nb::ndarray<nb::numpy, const float, nb::ndim<2>, nb::c_contig>;
      SpatialMean weights;
      if (!d.contains("spatial_mean") || !nb::try_cast(d["spatial_mean"], weights, false) ||
          weights.shape(0) == 0 || weights.shape(1) != (size_t)sub.n_node ||
          weights.shape(0) > max_space || weights.shape(0) > max_int / weights.shape(1))
        throw std::runtime_error("Recorder complete window: invalid spatial_mean extent or dtype");
      if (std::any_of(weights.data(), weights.data() + weights.size(),
                      [](float w) { return !std::isfinite(w); }))
        throw std::runtime_error("Recorder complete window: nonfinite spatial_mean");
      m.set_spatial_mean(weights.data(), (int)weights.shape(0), sub.n_node);
    } else if ((kind == KMON_GLOBAL_AVG ? 1u : (size_t)sub.n_node) > max_space) {
      throw std::runtime_error("Recorder complete window: output extent out of range");
    }
    m.configure_basic(slot, kind, period, clock, (int)source.size(), sub.n_node, 1, voi, cadence);
  }

  void set_monitors(int si, nb::list atts) {
    if (si < 0 || (size_t)si >= sn.size())
      throw std::runtime_error("bad subnet index in set_monitors");
    if (mons.size() < sn.size()) mons.resize(sn.size());
    auto &cur = mons[si];
    const int n_voi = (int)voi_specs[si].size();
    const int n_node = sn[si].n_node;
    const int n_cvar = sn[si].n_cvar;
    const int n_modes = sn[si].n_modes;
    const double def_dt = (double)dt;
    std::vector<kmon<W>> next;
    next.reserve(atts.size());
    for (auto item : atts) {
      kmon<W> m;
      if (nb::isinstance<nb::dict>(item) &&
          dict_get_or<bool>(nb::cast<nb::dict>(item), "local_window", false)) {
        configure_local_window((size_t)si, nb::cast<nb::dict>(item), m);
      } else {
        parse_kmon_spec<W>(item, n_voi, n_node, n_cvar, n_modes, def_dt, cur, m);
      }
      if (std::any_of(next.begin(), next.end(), [&](const auto &other) {
            return (m.local_window || other.local_window) && other.slot == m.slot;
          }))
        throw std::runtime_error("Recorder complete window: duplicate attachment slot");
      next.push_back(std::move(m));
    }
    cur = std::move(next);
  }

  void set_global_monitors(nb::list gather_specs, nb::list mon_specs) {
    if (gather_specs.size() != sn.size())
      throw std::runtime_error(
          "set_global_monitors: gather_specs length != subnet count");
    bool has_window = false;
    bool has_projection_window = false;
    bool has_bw_stream = false;
    for (auto item : mon_specs) {
      if (!nb::isinstance<nb::dict>(item)) continue;
      const auto d = nb::cast<nb::dict>(item);
      const bool complete = dict_get_or<bool>(d, "complete_window", false);
      const int kind = dict_get_or<int>(d, "kind", 0);
      has_window |= complete;
      has_projection_window |= complete && kind == KMON_PROJECTION;
      has_bw_stream |= kind == KMON_BOLD_BW_STREAM;
    }
    if (has_bw_stream) {
      if (W != 1 || sn.size() != 1 || sn[0].K != 1 || sn[0].dt != dt || sn[0].n_modes != 1)
        throw std::runtime_error("global BW stream requires width 1, one regional subnet, clock and mode");
      validate_single_source_observations("global BW stream");
    }
    if (has_projection_window) {
      if (W != 1 || sn.size() != 1 || sn[0].K != 1 || sn[0].dt != dt || sn[0].n_modes != 1)
        throw std::runtime_error("global complete window projection requires width 1, one regional subnet, clock and mode");
      if (!std::isfinite(dt) || dt <= 0.0f)
        throw std::runtime_error("global complete window projection: invalid stored neural clock");
      validate_single_source_observations(projection_label);
    }
    const bool staged = has_window || has_bw_stream;
    std::vector<GlobalGatherSubnet> gather;
    gather.reserve(gather_specs.size());
    int tot_v = 0;
    int tot_n = 0;
    const auto parse_node_indices = [&](nb::handle h) {
      return has_bw_stream || has_projection_window
          ? to_checked_indices(h, (size_t)sn[0].n_node,
                               has_bw_stream ? "global BW stream" : projection_label,
                               "node gather index")
          : to_int_vec(h);
    };
    for (auto item : gather_specs) {
      GlobalGatherSubnet gs;
      if (nb::isinstance<nb::dict>(item)) {
        nb::dict d = nb::cast<nb::dict>(item);
        gs.n_voi = nb::cast<int>(d["n_voi"]);
        gs.n_node = nb::cast<int>(d["n_node"]);
        gs.voi_offset = dict_get_or<int>(d, "voi_offset", 0);
        gs.node_offset = dict_get_or<int>(d, "node_offset", 0);
        if (d.contains("node_indices") && !d["node_indices"].is_none())
          gs.node_indices = parse_node_indices(d["node_indices"]);
      } else {
        nb::tuple t = nb::cast<nb::tuple>(item);
        gs.n_voi = nb::cast<int>(t[0]);
        gs.n_node = nb::cast<int>(t[1]);
        gs.voi_offset = nb::cast<int>(t[2]);
        gs.node_offset = nb::cast<int>(t[3]);
        if (t.size() > 4 && !t[4].is_none())
          gs.node_indices = parse_node_indices(t[4]);
      }
      if (!staged) {
        if (gs.voi_offset + gs.n_voi > tot_v) tot_v = gs.voi_offset + gs.n_voi;
        if (gs.node_indices.empty()) {
          if (gs.node_offset + gs.n_node > tot_n)
            tot_n = gs.node_offset + gs.n_node;
        } else {
          for (int gn : gs.node_indices)
            if (gn + 1 > tot_n) tot_n = gn + 1;
        }
      }
      gather.push_back(std::move(gs));
    }
    if (staged) {
      const auto [n_voi, n_node] = validate_window_gather(gather);
      tot_v = n_voi;
      tot_n = n_node;
    }
    if (has_projection_window) {
      const auto &g = gather.front();
      for (int j = 0; j < g.n_node; ++j)
        if (g.node_indices.empty() ? g.node_offset != 0 : g.node_indices[j] != j)
          throw std::runtime_error("global complete window projection requires local node order");
    }

    std::vector<kmon<W>> next;
    next.reserve(mon_specs.size());
    const double def_dt = (double)dt;
    for (auto item : mon_specs) {
      kmon<W> m;
      parse_kmon_spec<W>(item, tot_v, tot_n, tot_v, 1, def_dt, global_mons, m,
                         true);
      if (m.bw_stream()) m.bw_source_voi = voi_specs[0][(size_t)m.voi[0]];
      if (m.staged_global()) m.begin_run(t_abs);
      next.push_back(std::move(m));
    }
    if (staged) {
      for (const auto &m : next) {
        if (!m.staged_global()) continue;
        if (m.slot < 0 || std::count_if(next.begin(), next.end(), [&](const auto &other) {
              return other.slot == m.slot;
            }) != 1)
          throw std::runtime_error(m.bw_stream() ? "global BW stream: monitor slot must be unique"
                                                 : "global complete window: monitor slot must be unique");
      }
    }
    global_obs_row.assign(
        (size_t)std::max(0, tot_v) * (size_t)std::max(0, tot_n) * (size_t)W,
        0.0f);
    global_total_vois = tot_v;
    global_total_nodes = tot_n;
    global_gather = std::move(gather);
    global_mons = std::move(next);
  }

  std::pair<int, int> validate_window_gather(
      const std::vector<GlobalGatherSubnet> &gather) const {
    const size_t n_voi = voi_specs.empty() ? 0 : voi_specs.front().size();
    constexpr int max_extent = std::numeric_limits<int>::max();
    if (gather.empty() || n_voi == 0 || n_voi > (size_t)max_extent)
      throw std::runtime_error("global complete window: incompatible gather shape");
    // Derive the allocation extents from the sources, never from offsets.
    int64_t n_node = 0;
    for (size_t i = 0; i < gather.size(); ++i) {
      const auto &g = gather[i];
      const auto &sub = sn[i];
      if (sub.K != 1 || sub.dt != dt || sub.n_modes != 1)
        throw std::runtime_error("global complete window requires one regional clock and mode");
      if (g.n_node != sub.n_node || sub.n_node <= 0 || g.n_voi != (int)n_voi ||
          voi_specs[i].size() != n_voi || g.voi_offset != 0 ||
          (!g.node_indices.empty() && g.node_indices.size() != (size_t)sub.n_node))
        throw std::runtime_error("global complete window: incompatible gather shape");
      n_node += sub.n_node;
      if (n_node > max_extent)
        throw std::runtime_error("global complete window: gather extent out of range");
    }
    const size_t max_elems = std::min(global_obs_row.max_size(),
                                     std::vector<double>().max_size());
    if (n_voi > max_elems / (size_t)W / (size_t)n_node)
      throw std::runtime_error("global complete window: gather extent out of range");
    std::vector<bool> seen((size_t)n_node, false);
    for (const auto &g : gather) {
      for (int j = 0; j < g.n_node; ++j) {
        const int64_t node = g.node_indices.empty() ? (int64_t)g.node_offset + j
                                                    : g.node_indices[j];
        if (node < 0 || node >= n_node || seen[(size_t)node])
          throw std::runtime_error("global complete window requires a complete, disjoint node gather");
        seen[(size_t)node] = true;
      }
    }
    return {(int)n_voi, (int)n_node};
  }

  bool projection_window_armed() const {
    return std::any_of(global_mons.begin(), global_mons.end(),
                       [](const auto &m) { return m.projection_window(); });
  }

  void set_observed_rows(int si, std::vector<voispec> rows) {
    auto &current = voi_specs[si];
    if (!std::equal(current.begin(), current.end(), rows.begin(), rows.end(),
                    [](const auto &a, const auto &b) {
                      return a.kind == b.kind && a.a == b.a && a.b == b.b;
                    })) {
      if (projection_window_armed())
        throw std::runtime_error("global complete window projection: detach before changing observed rows");
      if (std::any_of(mons[si].begin(), mons[si].end(), [](const auto &m) { return m.local_window; }))
        throw std::runtime_error("Recorder complete window: detach before changing observed rows");
    }
    current = std::move(rows);
  }

  void validate_single_source_observations(const char *label) const {
    for (const auto &v : voi_specs[0]) {
      if (v.kind < 0 || v.kind > 1 || v.a < 0 || v.a >= sn[0].n_svar ||
          v.b < 0 || v.b >= sn[0].n_svar)
        throw std::runtime_error(std::string(label) + ": invalid observed VOI source");
    }
  }

  void validate_bw_live_config(const kmon<W> &m) const {
    if (sn.size() != 1 || sn[0].K != 1 || sn[0].n_modes != 1 ||
        sn[0].dt != dt || dt != (float)m.dt || sn[0].n_node != m.n_out_space ||
        voi_specs[0].size() != (size_t)global_total_vois)
      throw std::runtime_error("global BW stream: changed simulator geometry or clock; reattach monitor");
    const auto &v = voi_specs[0][(size_t)m.voi[0]];
    if (v.kind != m.bw_source_voi.kind || v.a != m.bw_source_voi.a || v.b != m.bw_source_voi.b)
      throw std::runtime_error("global BW stream: changed selected observation; reattach monitor");
    validate_single_source_observations("global BW stream");
  }

  nb::list window_source_config() const {
    nb::list sources;
    for (size_t i = 0; i < global_gather.size(); ++i) {
      const auto &g = global_gather[i];
      nb::list source_voi;
      for (const auto &v : voi_specs[i])
        source_voi.append(nb::make_tuple(v.kind, v.a, v.b));
      sources.append(nb::make_tuple(g.n_voi, g.n_node, g.voi_offset, g.node_offset,
                                    g.node_indices, source_voi));
    }
    return sources;
  }

  nb::tuple window_config(const kmon<W> &m, const nb::list &sources) const {
    // The projection's live source definitions and clock cannot change while armed.
    if (m.projection_window())
      return nb::make_tuple(m.istep, m.dt, m.voi, sources, m.n_sensors, m.n_proj_nodes,
                            m.gain, m.ref_vec);
    // Exact effective configuration, including weights and input row identity.
    return nb::make_tuple(m.istep, m.dt, m.voi, sources, m.spatial_compact,
                          m.spatial_area_idx, m.spatial_scale, m.sm_row_ptr,
                          m.sm_col_idx, m.sm_val);
  }

  nb::tuple bw_stream_config(const kmon<W> &m) const {
    nb::list params;
    for (const auto &field : bw_parameter_fields)
      params.append(m.bw_stream_params.*field.second);
    return nb::make_tuple(m.istep, m.dt, m.bw_drive_steps, m.bw_drive_ms, m.bw_substeps,
                          m.voi, window_source_config(), m.bw_drive_baseline, m.bw_drive_scale,
                          params);
  }

  nb::list take_global_monitor_outputs() {
    nb::list out;
    for (auto &gm : global_mons) {
      const size_t n = gm.t_out.size();
      nb::ndarray<nb::numpy, double> mt = make_owned_array<double>({n});
      if (n > 0)
        std::copy(gm.t_out.begin(), gm.t_out.end(),
                  static_cast<double *>(mt.data()));
      nb::ndarray<nb::numpy, float> md = make_owned_array<float>(
          {n, (size_t)gm.n_out_v, (size_t)gm.n_out_space,
           (size_t)gm.n_out_modes * (size_t)W});
      if (!gm.d_out.empty())
        std::copy(gm.d_out.begin(), gm.d_out.end(),
                  static_cast<float *>(md.data()));
      gm.begin_call();
      out.append(nb::make_tuple(mt, md));
    }
    return out;
  }

  nb::tuple local_window_config(size_t si, const kmon<W> &m) const {
    nb::list observations;
    for (const auto &v : voi_specs[si])
      observations.append(nb::make_tuple(v.kind, v.a, v.b));
    const auto &sub = sn[si];
    return nb::make_tuple(1, si, m.slot, m.kind, m.local_source_name, m.dt, m.local_master_dt,
                          m.cadence_K, m.istep, m.voi, observations,
                          sub.n_svar, sub.n_node, sub.n_modes, m.local_node_indices,
                          m.sm_row_ptr, m.sm_col_idx, m.sm_val);
  }

  nb::list get_monitor_states() const {
    nb::list per_sn;
    for (size_t si = 0; si < sn.size(); si++) {
      nb::list ms;
      if (si < mons.size()) {
        for (const auto &m : mons[si]) {
          if (m.local_window) validate_local_window_source(si, m);
          KMonState st = m.export_state();
          nb::dict d = kmon_state_to_dict(st);
          if (m.local_window) {
            m.validate_local_window_state(st, t_abs);
            d["local_window_config"] = local_window_config(si, m);
            d["local_window_phase"] = t_abs % m.cadence_K;
          }
          ms.append(d);
        }
      }
      per_sn.append(ms);
    }
    return per_sn;
  }

  void set_monitor_states(nb::list states) {
    std::vector<std::pair<kmon<W> *, KMonState>> staged;
    for (size_t si = 0; si < states.size(); ++si) {
      nb::list ms = nb::cast<nb::list>(states[si]);
      for (auto item : ms) {
        nb::dict d = nb::cast<nb::dict>(item);
        const int slot = dict_get_or<int>(d, "slot", -1);
        const int kind = dict_get_or<int>(d, "kind", 0);
        if (!d.contains("local_window_config")) {
          // The reserved Recorder slots distinguish historical local windows from global
          // monitors routed through the legacy per-subnet collector.
          if (d.contains("local_window_phase") || (slot >= recorder_slot_base && local_window_kind(kind)))
            throw std::runtime_error("Recorder complete window: historical snapshot monitor");
          continue;
        }
        if (si >= mons.size())
          throw std::runtime_error("Recorder complete window: unmatched snapshot subnet");
        auto it = std::find_if(mons[si].begin(), mons[si].end(), [&](const auto &m) {
          return m.local_window && m.slot == slot && m.kind == kind;
        });
        if (it == mons[si].end())
          throw std::runtime_error("Recorder complete window: unmatched snapshot monitor");
        if (std::any_of(staged.begin(), staged.end(), [&](const auto &entry) { return entry.first == &*it; }))
          throw std::runtime_error("Recorder complete window: duplicate snapshot monitor");
        validate_local_window_source(si, *it);
        const int equal = PyObject_RichCompareBool(d["local_window_config"].ptr(),
                                                    local_window_config(si, *it).ptr(), Py_EQ);
        if (equal < 0) throw nb::python_error();
        if (!equal)
          throw std::runtime_error("Recorder complete window: incompatible snapshot configuration");
        if (!d.contains("local_window_phase") || !PyLong_CheckExact(d["local_window_phase"].ptr()) ||
            nb::cast<uint32_t>(d["local_window_phase"]) != t_abs % it->cadence_K)
          throw std::runtime_error("Recorder complete window: invalid snapshot master cadence phase");
        KMonState st = dict_to_local_window_state(d, it->accum.size());
        it->validate_local_window_state(st, t_abs);
        staged.emplace_back(&*it, std::move(st));
      }
    }
    for (const auto &monitors : mons) {
      for (const auto &m : monitors) {
        if (m.local_window && std::none_of(staged.begin(), staged.end(), [&](const auto &entry) {
              return entry.first == &m;
            }))
          throw std::runtime_error("Recorder complete window: missing compatible snapshot monitor");
      }
    }

    // Preserve legacy matching and position fallback, but never across the staged boundary.
    const size_t n = std::min(states.size(), mons.size());
    for (size_t si = 0; si < n; si++) {
      nb::list ms = nb::cast<nb::list>(states[si]);
      for (size_t mi = 0; mi < ms.size(); mi++) {
        nb::dict d = nb::cast<nb::dict>(ms[mi]);
        if (d.contains("local_window_config")) continue;
        KMonState st = dict_to_kmon_state(d);
        auto it = std::find_if(mons[si].begin(), mons[si].end(),
                               [&](const kmon<W> &m) {
                                 return !m.local_window && m.slot == st.slot && m.kind == st.kind;
                               });
        if (it != mons[si].end()) {
          it->import_state(st);
        } else if (mi < mons[si].size() && !mons[si][mi].local_window) {
          mons[si][mi].import_state(st);
        }
      }
    }
    for (auto &[monitor, state] : staged) monitor->commit_validated_state(state);
  }

  nb::list get_global_monitor_states() const {
    nb::list out;
    for (const auto &gm : global_mons) {
      const auto st = gm.export_state();
      nb::dict state = kmon_state_to_dict(st);
      if (gm.complete_window) {
        state["window_config"] = window_config(gm, window_source_config());
      } else if (gm.bw_stream()) {
        validate_bw_live_config(gm);
        state["bw_stream_config"] = bw_stream_config(gm);
        export_bw_stream_state(state, st);
      }
      out.append(state);
    }
    return out;
  }

  void set_global_monitor_states(nb::list states) {
    std::vector<bool> restored(global_mons.size(), false);
    std::vector<std::pair<size_t, KMonState>> staged_states;
    bool has_bw_stream = false;
    const bool has_projection_window = projection_window_armed();
    for (auto item : states) {
      nb::dict d = nb::cast<nb::dict>(item);
      const bool bw = d.contains("bw_stream_config");
      if (!bw && !d.contains("window_config")) {
        if (has_projection_window && dict_get_or<int>(d, "kind", 0) == KMON_PROJECTION) {
          const int slot = dict_get_or<int>(d, "slot", -1);
          if (std::any_of(global_mons.begin(), global_mons.end(), [&](const auto &m) {
                return m.projection_window() && m.slot == slot;
              }))
            throw std::runtime_error("global complete window projection: historical snapshot monitor");
        }
        continue;
      }
      const char *label = bw ? "global BW stream: " : "global complete window: ";
      if (bw && d.contains("window_config"))
        throw std::runtime_error("global BW stream: ambiguous snapshot identity");
      KMonState st = dict_to_staged_state(d, bw, label);
      auto it = std::find_if(global_mons.begin(), global_mons.end(),
                             [&](const auto &m) {
                               return (bw ? m.bw_stream() : m.complete_window) &&
                                      m.slot == st.slot && m.kind == st.kind;
                             });
      if (it == global_mons.end())
        throw std::runtime_error(std::string(label) + "unmatched snapshot monitor");
      const size_t i = (size_t)(it - global_mons.begin());
      if (bw) validate_bw_live_config(*it);
      const nb::tuple config = bw ? bw_stream_config(*it)
                                  : window_config(*it, window_source_config());
      const int equal = PyObject_RichCompareBool(
          d[bw ? "bw_stream_config" : "window_config"].ptr(), config.ptr(), Py_EQ);
      if (equal < 0) throw nb::python_error();
      if (restored[i] || !equal)
        throw std::runtime_error(std::string(label) + "incompatible snapshot configuration");
      if (bw) {
        parse_bw_stream_state(st, d, (size_t)it->n_out_space);
        it->validate_bw_stream_state(st, t_abs);
        has_bw_stream = true;
      } else {
        it->validate_window_state(st, t_abs);
      }
      restored[i] = true;
      staged_states.emplace_back(i, std::move(st));
    }
    for (size_t i = 0; i < global_mons.size(); ++i) {
      if (!global_mons[i].staged_global() || restored[i]) continue;
      throw std::runtime_error(global_mons[i].bw_stream()
          ? "global BW stream: missing compatible snapshot monitor"
          : "global complete window: missing compatible snapshot monitor");
    }

    // With BW or staged Projection, the checkpoint must cover the entire attached set exactly once.
    // Keep strict matching, validation, and output clearing out of the historical restore path.
    const bool strict_siblings = has_bw_stream || has_projection_window;
    const char *sibling_label = has_bw_stream ? "global BW stream: "
                                             : "global complete window projection: ";
    for (size_t mi = 0; mi < states.size(); mi++) {
      nb::dict d = nb::cast<nb::dict>(states[mi]);
      if (d.contains("window_config") || d.contains("bw_stream_config")) continue;
      if (strict_siblings) {
        for (const char *key : {"slot", "kind", "width", "n_out_v", "n_out_space", "n_out_modes"})
          if (!d.contains(key) || d[key].is_none())
            throw std::runtime_error(std::string(sibling_label) + "incomplete sibling snapshot identity or geometry");
      }
      KMonState st = dict_to_kmon_state(d);
      const auto matches = [&](const auto &m) {
        return !m.staged_global() && m.slot == st.slot && m.kind == st.kind;
      };
      auto it = std::find_if(global_mons.begin(), global_mons.end(), matches);
      if (strict_siblings) {
        if (it == global_mons.end() || std::find_if(std::next(it), global_mons.end(), matches) != global_mons.end())
          throw std::runtime_error(std::string(sibling_label) + "unmatched or ambiguous sibling snapshot monitor");
        const size_t i = (size_t)(it - global_mons.begin());
        if (restored[i])
          throw std::runtime_error(std::string(sibling_label) + "duplicate sibling snapshot monitor");
        it->validate_legacy_sibling_state(st, sibling_label);
        restored[i] = true;
        staged_states.emplace_back(i, std::move(st));
      } else {
        if (it == global_mons.end() && mi < global_mons.size() && !global_mons[mi].staged_global())
          it = global_mons.begin() + mi;
        if (it != global_mons.end()) it->import_state(st);
      }
    }
    if (strict_siblings && std::find(restored.begin(), restored.end(), false) != restored.end())
      throw std::runtime_error(std::string(sibling_label) + "missing sibling snapshot monitor");
    for (auto &[i, st] : staged_states)
      global_mons[i].commit_validated_state(st);
  }

  void set_stim_specs(int si, nb::list specs) {
    if (si < 0 || (size_t)si >= sn.size())
      throw std::runtime_error("bad subnet index in set_stim_specs");
    stim_specs.erase(
        std::remove_if(stim_specs.begin(), stim_specs.end(),
                       [si](const stim_spec<W> &s) {
                         return s.subnet_idx == si;
                       }),
        stim_specs.end());
    for (auto item : specs) {
      nb::dict d = nb::cast<nb::dict>(item);
      stim_spec<W> sp;
      sp.subnet_idx = si;
      sp.n_node = dict_get_or<int>(d, "n_node", sn[si].n_node);
      if (d.contains("target_cvars") && !d["target_cvars"].is_none())
        sp.target_cvars = to_int_vec(d["target_cvars"]);
      if (d.contains("spatial") && !d["spatial"].is_none()) {
        sp.spatial = to_float_vec(d["spatial"]);
        if (d.contains("lane_varying") && !d["lane_varying"].is_none()) {
          sp.lane_varying = nb::cast<bool>(d["lane_varying"]);
        } else {
          sp.lane_varying =
              (W > 1 && sp.spatial.size() == (size_t)sp.n_node * (size_t)W);
        }
      }
      sp.eq_kind = dict_get_or<int>(d, "eq_kind", STIM_DISCRETE_1D);
      sp.onset_t0 = dict_get_or<double>(d, "onset_t0", 0.0);
      sp.dt0 = dict_get_or<float>(d, "dt0", dt);
      if (d.contains("eq_params") && !d["eq_params"].is_none()) {
        std::vector<double> ep = to_double_vec(d["eq_params"]);
        for (size_t k = 0; k < ep.size() && k < sp.eq_params.size(); k++)
          sp.eq_params[k] = ep[k];
      }
      if (d.contains("temporal_1d") && !d["temporal_1d"].is_none())
        sp.temporal_1d = to_float_vec(d["temporal_1d"]);
      sp.step_offset = dict_get_or<int64_t>(d, "step_offset", 0);
      sp.scale = dict_get_or<float>(d, "scale", 1.0f);
      if (d.contains("lane_scale") && !d["lane_scale"].is_none())
        sp.lane_scale = to_float_vec(d["lane_scale"]);
      stim_specs.push_back(std::move(sp));
    }
  }

  void clear_stim_specs() { stim_specs.clear(); }

  void set_subnet_rng(int si, nb::object lane_keys_obj, nb::object gfun_obj,
                      double ntau = 0.0, bool is_multiplicative = false,
                      nb::object mul_a_obj = nb::none(),
                      nb::object mul_b_obj = nb::none()) {
    if (si < 0 || (size_t)si >= sn.size())
      throw std::runtime_error("bad subnet index in set_subnet_rng");
    if (sub_rng.size() < sn.size()) sub_rng.resize(sn.size());
    if (rng_scratch.size() < sn.size()) rng_scratch.resize(sn.size());
    auto &eng = sub_rng[si];
    eng.subnet_id = (uint32_t)si;
    eng.dt = sn[si].dt;
    eng.ntau = (float)ntau;
    eng.step_counter = 0;
    eng.ou_state.clear();
    if (lane_keys_obj.is_none()) {
      eng.enabled = false;
      return;
    }
    if (nb::isinstance<nb::int_>(lane_keys_obj)) {
      eng.set_uniform_seed(nb::cast<uint64_t>(lane_keys_obj));
    } else {
      std::vector<uint64_t> seeds = to_u64_vec(lane_keys_obj);
      if (seeds.empty()) {
        eng.enabled = false;
        return;
      }
      eng.set_lane_seeds(seeds.data(), seeds.size());
    }
    eng.gfun = to_float_vec(gfun_obj);
    eng.is_multiplicative = is_multiplicative;
    eng.mul_a = mul_a_obj.is_none() ? 1.0f : nb::cast<float>(mul_a_obj);
    eng.mul_b = mul_b_obj.is_none() ? 0.0f : nb::cast<float>(mul_b_obj);
    eng.enabled = true;
  }

  nb::list get_rng_states() const {
    nb::list out;
    for (size_t si = 0; si < sn.size(); si++) {
      nb::dict d;
      if (si < sub_rng.size()) {
        const auto &eng = sub_rng[si];
        auto st = eng.export_state();
        d["enabled"] = eng.enabled;
        auto lk = make_owned_array<uint64_t>({(size_t)W});
        for (int w = 0; w < W; w++) lk.data()[w] = st.lane_keys[(size_t)w];
        d["lane_keys"] = lk;
        d["step_counter"] = st.step_counter;
        auto ou = make_owned_array<float>({st.ou_state.size()});
        if (!st.ou_state.empty())
          std::memcpy(ou.data(), st.ou_state.data(),
                      st.ou_state.size() * sizeof(float));
        d["ou_state"] = ou;
      } else {
        d["enabled"] = false;
      }
      out.append(d);
    }
    return out;
  }

  void set_rng_states(nb::list states) {
    if (sub_rng.size() < sn.size()) sub_rng.resize(sn.size());
    const size_t n = std::min(states.size(), sub_rng.size());
    for (size_t si = 0; si < n; si++) {
      nb::dict d = nb::cast<nb::dict>(states[si]);
      sub_rng[si].enabled = dict_get_or<bool>(d, "enabled", false);
      if (d.contains("lane_keys") && !d["lane_keys"].is_none()) {
        rng_state<W> st;
        std::vector<uint64_t> lk = to_u64_vec(d["lane_keys"]);
        for (int w = 0; w < W && (size_t)w < lk.size(); w++)
          st.lane_keys[(size_t)w] = lk[(size_t)w];
        st.step_counter = dict_get_or<uint64_t>(d, "step_counter", 0);
        if (d.contains("ou_state") && !d["ou_state"].is_none())
          st.ou_state = to_float_vec(d["ou_state"]);
        sub_rng[si].import_state(st);
      }
    }
  }

  void enable_online_features(int si, double dt_sample = 1.0) {
    if (si < 0 || (size_t)si >= sn.size())
      throw std::runtime_error("bad subnet index in enable_online_features");
    if (online_enabled.size() < sn.size()) online_enabled.resize(sn.size(), 0);
    if (online_accums.size() < sn.size()) online_accums.resize(sn.size());
    online_accums[si].reset((int)voi_specs[si].size(), sn[si].n_node,
                            dt_sample);
    online_enabled[si] = 1;
  }

  void disable_online_features() {
    std::fill(online_enabled.begin(), online_enabled.end(), 0);
  }

  nb::dict take_online_features(int si, const std::vector<std::string> &names,
                                int order = 4) const {
    if (si < 0 || (size_t)si >= online_accums.size())
      throw std::runtime_error("bad subnet index in take_online_features");
    const auto &acc = online_accums[si];
    nb::dict out;
    for (const auto &name : names) {
      const int n_comp = OnlineFeatureAccum<W>::feature_components(name, order);
      std::vector<size_t> shape;
      if (name == "moments") {
        shape = {(size_t)n_comp, (size_t)acc.n_voi, (size_t)acc.n_node,
                 (size_t)W};
      } else {
        shape = {(size_t)acc.n_voi, (size_t)acc.n_node, (size_t)W};
      }
      auto arr = make_owned_array<double>(shape);
      acc.finalize_into(name, arr.data(), order);
      out[nb::str(name.c_str())] = arr;
    }
    return out;
  }

  void monitor_begin_run() {
    for (auto &mons_s : mons)
      for (auto &m : mons_s) m.begin_run(t_abs);
    for (auto &gm : global_mons) gm.begin_run(t_abs);
  }

  void monitor_begin_call() {
    for (auto &mons_s : mons)
      for (auto &m : mons_s) m.begin_call();
    for (auto &gm : global_mons) gm.begin_call();
  }

  // Run nstep steps. Output per chunk of `chunk_size` steps:
  // tavg (n_chunks, n_voi, n_node, W) or None when !need_tavg,
  // ctavg (n_chunks, n_cvar, n_node, n_modes * W) or None when !need_ctavg,
  // followed by per-subnet monitor engine streams (times, data).
  nb::list run(int nstep, int chunk_size, nb::object noise_obj,
               nb::object stim_obj, bool need_tavg = true,
               bool need_ctavg = true) {
    for (const auto &gm : global_mons) {
      if (!gm.staged_global()) continue;
      if (gm.bw_stream()) validate_bw_live_config(gm);
      if (gm.m_step != t_abs)
        throw std::runtime_error(gm.bw_stream()
            ? "global BW stream: reset or restore at the simulator clock before run"
            : "global complete window: reset or restore at the simulator clock before run");
      if (nstep > 0 && (uint64_t)nstep > std::numeric_limits<uint64_t>::max() - t_abs)
        throw std::runtime_error(gm.bw_stream() ? "global BW stream: run exceeds remaining tick range"
                                              : "global complete window: run exceeds remaining tick range");
    }
    for (size_t si = 0; si < mons.size(); ++si) {
      for (const auto &m : mons[si]) {
        if (!m.local_window) continue;
        validate_local_window_source(si, m);
        if (m.run_start != 0 || m.m_step != t_abs / m.cadence_K ||
            m.accum_count != m.m_step % (uint64_t)m.istep)
          throw std::runtime_error("Recorder complete window: reset or restore at the simulator clock before run");
        if (nstep < 0 || (uint64_t)nstep > std::numeric_limits<uint64_t>::max() - t_abs)
          throw std::runtime_error("Recorder complete window: run exceeds tick range");
      }
    }
    const int cs = chunk_size > 0 ? chunk_size : tavg_period;
    const int rem = nstep % cs;
    const int n_chunks = nstep / cs + (rem > 0);
    const size_t n_sn = sn.size();
    monitor_begin_call();
    for (size_t s = 0; s < mons.size(); s++) {
      for (auto &mon : mons[s]) {
        if (mon.local_window) mon.reserve_local_window(t_abs, t_abs + (uint64_t)nstep);
        else mon.reserve_for_steps(nstep);
      }
    }
    for (auto &gm : global_mons) gm.reserve_for_steps(nstep);

    // Ensure per-subnet scratch buffers are sized
    if (obs_row.size() < n_sn) obs_row.resize(n_sn);
    if (rng_scratch.size() < n_sn) rng_scratch.resize(n_sn);
    for (size_t s = 0; s < n_sn; s++) {
      const size_t need_obs =
          (size_t)voi_specs[s].size() * (size_t)sn[s].n_node * (size_t)W;
      if (obs_row[s].size() < need_obs) obs_row[s].resize(need_obs);
    }
    std::vector<const float *> global_src_rows(n_sn, nullptr);
    if (!global_mons.empty()) {
      if (global_gather.size() != n_sn)
        throw std::runtime_error(
            "simw::run: global_gather size != subnet count");
      const size_t gsz =
          (size_t)std::max(0, global_total_vois) *
          (size_t)std::max(0, global_total_nodes) * (size_t)W;
      if (global_obs_row.size() < gsz) global_obs_row.resize(gsz);
    }

    std::vector<nb::ndarray<nb::numpy, float>> keep_alive;
    auto resolve_per_sn = [&](nb::object obj) -> std::vector<const float *> {
      std::vector<const float *> ps(n_sn, nullptr);
      if (obj.is_none()) return ps;
      if (nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj)) {
        auto seq = nb::cast<nb::sequence>(obj);
        size_t idx = 0;
        for (auto item : seq) {
          if (idx >= n_sn)
            throw std::runtime_error(
                "noise/stim sequence longer than number of subnetworks");
          if (!item.is_none()) {
            keep_alive.push_back(
                nb::cast<nb::ndarray<nb::numpy, float>>(item));
            ps[idx] = keep_alive.back().data();
          }
          idx++;
        }
      } else {
        keep_alive.push_back(nb::cast<nb::ndarray<nb::numpy, float>>(obj));
        const float *d = keep_alive.back().data();
        for (size_t s = 0; s < n_sn; s++) ps[s] = d;
      }
      return ps;
    };
    const std::vector<const float *> noise_ps = resolve_per_sn(noise_obj);
    const std::vector<const float *> stim_ps = resolve_per_sn(stim_obj);

    std::vector<std::vector<float>> tacc(n_sn), cacc(n_sn);
    std::vector<std::vector<int>> counts(n_sn);
    for (size_t s = 0; s < n_sn; s++) {
      const int n_voi = (int)voi_specs[s].size();
      if (need_tavg) {
        tacc[s].assign((size_t)n_chunks * n_voi * sn[s].n_node * W, 0.f);
      }
      if (need_ctavg) {
        cacc[s].assign((size_t)n_chunks * sn[s].n_cvar * sn[s].n_node *
                           sn[s].n_modes * W,
                       0.f);
      }
      counts[s].assign(n_chunks, 0);
    }

    std::exception_ptr pend = nullptr;
    {
      nb::gil_scoped_release nogil;
      try {
        for (int step = 0; step < nstep; step++) {
          const size_t chunk = (size_t)(step / cs);
          for (auto &sub : sn) std::fill(sub.c.begin(), sub.c.end(), 0.f);
          for (auto &p : pr) p.apply(sn[p.src_sn], sn[p.tgt_sn], (int)t_abs);

          for (size_t s = 0; s < n_sn; s++) {
            auto &sub = sn[s];
            counts[s][chunk]++;

            if (const float *stim = stim_ps[s]) {
              apply_legacy_dense_stim<W>(sub.c.data(), stim, step, sub.n_cvar,
                                         sub.n_node, sub.n_modes);
            }
            if (!stim_specs.empty()) {
              apply_subnet_stims<W>(stim_specs, (int)s, sub.c.data(),
                                    sub.n_modes, (int64_t)t_abs + 1,
                                    (double)(t_abs + 1u) * (double)dt);
            }

            if (need_ctavg) {
              const size_t n_cv =
                  (size_t)sub.n_cvar * sub.n_node * sub.n_modes * W;
              const float *cv = sub.c.data();
              float *dst = cacc[s].data() + chunk * n_cv;
#pragma omp simd
              for (size_t k = 0; k < n_cv; k++) dst[k] += cv[k];
            }

            if (((t_abs + 1) % sub.K) == 0) {
              const float *nz = nullptr;
              if (const float *nptr = noise_ps[s]) {
                nz = nptr + ((size_t)step) * (size_t)sub.n_svar * sub.n_node *
                                sub.n_modes * W;
              } else if (s < sub_rng.size() && sub_rng[s].enabled) {
                const size_t need_nz =
                    (size_t)sub.n_svar * sub.n_node * sub.n_modes * W;
                if (rng_scratch[s].size() < need_nz)
                  rng_scratch[s].resize(need_nz);
                sub_rng[s].fill_next_step_noise(rng_scratch[s].data(),
                                                sub.n_svar, sub.n_node,
                                                sub.n_modes, sub.x.data());
                nz = rng_scratch[s].data();
              }
              const int sub_integ = (sub.integ_id >= 0) ? sub.integ_id : integ_id;
              if (sub_integ == 1)
                sub.step_heun(sub.dt, nz);
              else
                sub.step_euler(sub.dt, nz);
              sub.push_state((int)((((t_abs + 1) / sub.K) - 1) & (sub.H - 1)));
            }

            const int n_voi = (int)voi_specs[s].size();
            const bool has_sn_mons = (s < mons.size() && !mons[s].empty());
            const bool has_online =
                (s < online_enabled.size() && online_enabled[s] != 0);
            const bool need_obs =
                need_tavg || has_sn_mons || !global_mons.empty() || has_online;
            if (need_obs && n_voi > 0) {
              observe_voi_row<W>(sub.x.data(), sub.n_node, sub.n_modes,
                                 voi_specs[s], obs_row[s].data());
            }
            if (need_tavg && n_voi > 0) {
              const size_t ntv = (size_t)n_voi * sub.n_node * W;
              float *dst = tacc[s].data() + chunk * ntv;
              const float *src_obs = obs_row[s].data();
#pragma omp simd
              for (size_t k = 0; k < ntv; k++) dst[k] += src_obs[k];
            }
            if (has_sn_mons) {
              for (auto &mon : mons[s]) {
                if (mon.kind == KMON_CTAVG) {
                  mon.step_one(sub.c.data(), sub.n_cvar, sub.n_node,
                               sub.n_modes, t_abs);
                } else {
                  mon.step_one(obs_row[s].data(), n_voi, sub.n_node, 1, t_abs);
                }
              }
            }
            if (has_online) {
              online_accums[s].update(obs_row[s].data());
            }
          }

          if (!global_mons.empty()) {
            for (size_t s = 0; s < n_sn; s++)
              global_src_rows[s] = obs_row[s].data();
            gather_global_voi_row<W>(global_src_rows, global_gather,
                                     global_total_vois, global_total_nodes,
                                     global_obs_row.data());
            for (auto &gm : global_mons) {
              gm.step_one(global_obs_row.data(), global_total_vois,
                          global_total_nodes, 1, t_abs);
            }
          }
          t_abs++;
        }
      } catch (...) {
        pend = std::current_exception();
      }
    }
    if (pend) std::rethrow_exception(pend);

    if (nstep > 0 && rem != 0) {
      const size_t chunk = (size_t)(nstep / cs);
      for (size_t s = 0; s < n_sn; s++) {
        counts[s][chunk] = rem;
      }
    }

    nb::list outs;
    for (size_t s = 0; s < n_sn; s++) {
      const int n_voi = (int)voi_specs[s].size();
      if (need_tavg) {
        nb::ndarray<nb::numpy, float> tarr = make_owned_array<float>(
            {(size_t)n_chunks, (size_t)n_voi, (size_t)sn[s].n_node, (size_t)W});
        float *td = tarr.data();
        const size_t ntv = (size_t)n_voi * sn[s].n_node * W;
        for (size_t chunk = 0; chunk < (size_t)n_chunks; chunk++) {
          const float tinv =
              1.f / (float)(counts[s][chunk] ? counts[s][chunk] : 1);
#pragma omp simd
          for (size_t k = 0; k < ntv; k++)
            td[chunk * ntv + k] = tacc[s][chunk * ntv + k] * tinv;
        }
        outs.append(tarr);
      } else {
        outs.append(nb::none());
      }

      if (need_ctavg) {
        nb::ndarray<nb::numpy, float> carr = make_owned_array<float>(
            {(size_t)n_chunks, (size_t)sn[s].n_cvar, (size_t)sn[s].n_node,
             (size_t)sn[s].n_modes * W});
        float *cd = carr.data();
        const size_t ncv =
            (size_t)sn[s].n_cvar * sn[s].n_node * sn[s].n_modes * W;
        for (size_t chunk = 0; chunk < (size_t)n_chunks; chunk++) {
          const float cinv =
              1.f / (float)(counts[s][chunk] ? counts[s][chunk] : 1);
#pragma omp simd
          for (size_t k = 0; k < ncv; k++)
            cd[chunk * ncv + k] = cacc[s][chunk * ncv + k] * cinv;
        }
        outs.append(carr);
      } else {
        outs.append(nb::none());
      }

      if (mons.size() == n_sn) {
        for (auto &mon : mons[s]) {
          const size_t n = mon.t_out.size();
          nb::ndarray<nb::numpy, double> mt = make_owned_array<double>({n});
          if (n > 0)
            std::copy(mon.t_out.begin(), mon.t_out.end(),
                      static_cast<double *>(mt.data()));
          nb::ndarray<nb::numpy, float> md = make_owned_array<float>(
              {n, (size_t)mon.n_out_v, (size_t)mon.n_out_space,
               (size_t)mon.n_out_modes * (size_t)W});
          if (!mon.d_out.empty())
            std::copy(mon.d_out.begin(), mon.d_out.end(),
                      static_cast<float *>(md.data()));
          outs.append(mt);
          outs.append(md);
        }
      }
    }
    return outs;
  }
};

// ---- width-erasing wrapper (W in {1, 4, 8, 16}) -----------------------------

struct sim {
  int width = 8;
  std::unique_ptr<simw<1>> s1;
  std::unique_ptr<simw<4>> s4;
  std::unique_ptr<simw<8>> s8;
  std::unique_ptr<simw<16>> s16;

  explicit sim(int width_ = 8) : width(width_) {
    if (width_ != 1 && width_ != 4 && width_ != 8 && width_ != 16) {
      throw std::invalid_argument(
          "Sim width must be one of {1, 4, 8, 16}, got " +
          std::to_string(width_));
    }
  }

  void ensure_sim() {
    switch (width) {
      case 1:  if (!s1)  s1  = std::make_unique<simw<1>>();  break;
      case 4:  if (!s4)  s4  = std::make_unique<simw<4>>();  break;
      case 8:  if (!s8)  s8  = std::make_unique<simw<8>>();  break;
      case 16: if (!s16) s16 = std::make_unique<simw<16>>(); break;
      default: throw std::invalid_argument("unsupported Sim width");
    }
  }

  template <typename F> void dispatch(F &&f) {
    switch (width) {
      case 1:  if (!s1)  throw std::runtime_error("sim not configured"); f(*s1);  break;
      case 4:  if (!s4)  throw std::runtime_error("sim not configured"); f(*s4);  break;
      case 8:  if (!s8)  throw std::runtime_error("sim not configured"); f(*s8);  break;
      case 16: if (!s16) throw std::runtime_error("sim not configured"); f(*s16); break;
      default: throw std::invalid_argument("unsupported Sim width");
    }
  }

  template <typename F> void dispatch(F &&f) const {
    switch (width) {
      case 1:  if (!s1)  throw std::runtime_error("sim not configured"); f(*s1);  break;
      case 4:  if (!s4)  throw std::runtime_error("sim not configured"); f(*s4);  break;
      case 8:  if (!s8)  throw std::runtime_error("sim not configured"); f(*s8);  break;
      case 16: if (!s16) throw std::runtime_error("sim not configured"); f(*s16); break;
      default: throw std::invalid_argument("unsupported Sim width");
    }
  }

  void add_subnet(int n_node, int n_svar, int n_parm, int n_cvar,
                  int model_id, int horizon, int n_modes, uint32_t k = 1,
                  float dt = 0.001f, int integ_id = -1, uint64_t generic_dfun = 0) {
    uint32_t H = 1;
    while (H < (uint32_t)(horizon + 2)) H <<= 1;
    ensure_sim();
    dispatch([&](auto &sw) {
      if (sw.projection_window_armed())
        throw std::runtime_error("global complete window projection: detach before adding a subnet");
      constexpr int W = std::remove_reference_t<decltype(sw)>::W_CONST;
      subnet<W> s;
      s.alloc(n_node, n_svar, n_parm, n_cvar, model_id, H, n_modes);
      s.generic_fn = reinterpret_cast<cph_generic_fn_t>(generic_dfun);
      s.K = k;
      s.dt = dt;
      s.integ_id = integ_id;
      sw.sn.push_back(std::move(s));
      sw.voi_specs.push_back({});
      sw.obs_row.push_back({});
      sw.mons.push_back({});
      sw.sub_rng.push_back(rng_engine<W>{});
      sw.rng_scratch.push_back({});
      sw.online_enabled.push_back(0);
      sw.online_accums.push_back(OnlineFeatureAccum<W>{});
    });
  }

  void add_projection(int src_sn, int tgt_sn, nb::ndarray<nb::numpy, float> w,
                      nb::ndarray<nb::numpy, uint32_t> idx,
                      nb::ndarray<nb::numpy, uint32_t> ptr,
                      nb::ndarray<nb::numpy, uint32_t> del, int cfun_id,
                      nb::ndarray<nb::numpy, int> src_cvars,
                      int tgt_cvar, int n_cfun_parm, float scale,
                      float ts, int tgt_state_cvar,
                      nb::object mode_map_obj) {
    ensure_sim();
    dispatch([&](auto &sw) {
      constexpr int W = std::remove_reference_t<decltype(sw)>::W_CONST;
      proj<W> p;
      p.src_sn = src_sn;
      p.tgt_sn = tgt_sn;
      _fill_proj(p, w, idx, ptr, del, cfun_id, src_cvars, tgt_cvar,
                 n_cfun_parm, scale, ts, tgt_state_cvar, mode_map_obj);
      sw.pr.push_back(std::move(p));
    });
  }

  template <int W>
  void _fill_proj(proj<W> &p, nb::ndarray<nb::numpy, float> w,
                  nb::ndarray<nb::numpy, uint32_t> idx,
                  nb::ndarray<nb::numpy, uint32_t> ptr,
                  nb::ndarray<nb::numpy, uint32_t> del, int cfun_id,
                  nb::ndarray<nb::numpy, int> src_cvars,
                  int tgt_cvar, int n_cfun_parm, float scale,
                  float ts, int tgt_state_cvar,
                  nb::object mode_map_obj) {
    p.ts = ts;
    p.tgt_state_cvar = tgt_state_cvar;
    p.has_pre = (cfun_id >= 3 && cfun_id <= 9);
    if (mode_map_obj.is_none()) {
      p.mode_map.assign(1, 1.0f);
    } else {
      nb::ndarray<nb::numpy, float> mm =
          nb::cast<nb::ndarray<nb::numpy, float>>(mode_map_obj);
      auto M = mm.view<nb::ndim<2>>();
      p.mode_map.resize((size_t)M.shape(0) * M.shape(1));
      for (size_t r = 0; r < M.shape(0); r++)
        for (size_t cq = 0; cq < M.shape(1); cq++)
          p.mode_map[r * M.shape(1) + cq] = M(r, cq);
    }
    p.scale = scale;
    p.cfun_id = cfun_id;
    auto S = src_cvars.view<nb::ndim<1>>();
    for (size_t k = 0; k < S.shape(0); k++) p.src_cvars.push_back(S(k));
    p.tgt_cvar = tgt_cvar;
    p.n_cfun_parm = n_cfun_parm;
    auto W_ = w.view<nb::ndim<1>>();
    auto I_ = idx.view<nb::ndim<1>>();
    auto T_ = ptr.view<nb::ndim<1>>();
    auto D_ = del.view<nb::ndim<1>>();
    p.w.resize(W_.shape(0));
    for (size_t k = 0; k < W_.shape(0); k++) p.w[k] = W_(k);
    p.idx.resize(I_.shape(0));
    for (size_t k = 0; k < I_.shape(0); k++) p.idx[k] = I_(k);
    p.ptr.resize(T_.shape(0));
    for (size_t k = 0; k < T_.shape(0); k++) p.ptr[k] = T_(k);
    p.del.resize(D_.shape(0));
    for (size_t k = 0; k < D_.shape(0); k++) p.del[k] = D_(k);
    p.cfp.assign((size_t)n_cfun_parm * W, 0.f);
  }

  void set_subnet_params(int si, nb::ndarray<nb::numpy, float> v) {
    dispatch([&](auto &s) {
      auto &sn = s.sn[si];
      auto V = v.view<nb::ndim<3>>();
      if ((int)V.shape(0) != sn.n_node || (int)V.shape(1) != sn.n_parm ||
          (int)V.shape(2) != (int)std::remove_reference_t<decltype(s)>::W_CONST)
        throw std::runtime_error("bad params shape");
      const int W = std::remove_reference_t<decltype(s)>::W_CONST;
      for (int j = 0; j < sn.n_node; j++)
        for (int k = 0; k < sn.n_parm; k++)
          for (int i = 0; i < W; i++)
            sn.p[((size_t)j * sn.n_parm + k) * W + i] = V(j, k, i);
    });
  }

  void set_subnet_state(int si, nb::ndarray<nb::numpy, float> v) {
    dispatch([&](auto &s) {
      auto &sn = s.sn[si];
      const int W = std::remove_reference_t<decltype(s)>::W_CONST;
      auto V = v.view<nb::ndim<4>>();
      if ((int)V.shape(0) != sn.n_svar || (int)V.shape(1) != sn.n_node ||
          (int)V.shape(2) != sn.n_modes || (int)V.shape(3) != W)
        throw std::runtime_error("bad state shape");
      std::memcpy(sn.x.data(), v.data(), sn.x.size() * sizeof(float));
      const size_t rows = (size_t)sn.n_svar * sn.n_node * sn.n_modes;
      for (size_t row = 0; row < rows; row++) {
        const size_t svar = row / ((size_t)sn.n_node * sn.n_modes);
        const size_t rem = row % ((size_t)sn.n_node * sn.n_modes);
        for (uint32_t hh = 0; hh < sn.H; hh++) {
          float *dst = sn.buf.data() +
              ((svar * (size_t)sn.n_node * sn.n_modes + rem) * sn.H + hh) * W;
          const float *srcv = sn.x.data() + row * W;
#pragma omp simd
          for (int i = 0; i < W; i++) dst[i] = srcv[i];
        }
      }
      s.t_abs = 0;
    });
  }

  uint64_t t_abs() const {
    if (s1) return s1->t_abs;
    if (s4) return s4->t_abs;
    if (s8) return s8->t_abs;
    if (s16) return s16->t_abs;
    return 0;
  }

  void set_t_abs(uint64_t t) {
    dispatch([&](auto &s) { s.t_abs = t; });
  }

  nb::ndarray<nb::numpy, float> get_subnet_state(int si) {
    nb::ndarray<nb::numpy, float> out;
    dispatch([&](auto &s) {
      auto &sn = s.sn[si];
      const int W = std::remove_reference_t<decltype(s)>::W_CONST;
      out = make_owned_array<float>({(size_t)sn.n_svar, (size_t)sn.n_node,
                                     (size_t)sn.n_modes, (size_t)W});
      std::memcpy(out.data(), sn.x.data(), sn.x.size() * sizeof(float));
    });
    return out;
  }

  nb::ndarray<nb::numpy, float> get_subnet_buf(int si) {
    nb::ndarray<nb::numpy, float> out;
    dispatch([&](auto &s) {
      if (si < 0 || (size_t)si >= s.sn.size())
        throw std::runtime_error("bad subnet index");
      auto &sn = s.sn[si];
      const int W = std::remove_reference_t<decltype(s)>::W_CONST;
      out = make_owned_array<float>({(size_t)sn.n_svar, (size_t)sn.n_node,
                                     (size_t)sn.n_modes, (size_t)sn.H,
                                     (size_t)W});
      std::memcpy(out.data(), sn.buf.data(), sn.buf.size() * sizeof(float));
    });
    return out;
  }

  void set_subnet_buf(int si, nb::ndarray<nb::numpy, float> v) {
    dispatch([&](auto &s) {
      if (si < 0 || (size_t)si >= s.sn.size())
        throw std::runtime_error("bad subnet index");
      auto &sn = s.sn[si];
      const int W = std::remove_reference_t<decltype(s)>::W_CONST;
      auto V = v.view<nb::ndim<5>>();
      if ((int)V.shape(0) != sn.n_svar || (int)V.shape(1) != sn.n_node ||
          (int)V.shape(2) != sn.n_modes || (uint32_t)V.shape(3) != sn.H ||
          (int)V.shape(4) != W)
        throw std::runtime_error("bad buffer shape");
      size_t pos = 0;
      for (int sv = 0; sv < sn.n_svar; sv++)
        for (int j = 0; j < sn.n_node; j++)
          for (int m = 0; m < sn.n_modes; m++)
            for (uint32_t h = 0; h < sn.H; h++)
              for (int i = 0; i < W; i++)
                sn.buf[pos++] = V(sv, j, m, h, i);
    });
  }

  void set_subnet_bounds(int si, nb::ndarray<nb::numpy, float> lo,
                         nb::ndarray<nb::numpy, float> hi) {
    dispatch([&](auto &s) {
      if (si < 0 || (size_t)si >= s.sn.size())
        throw std::runtime_error("bad subnet index");
      auto &sn = s.sn[si];
      auto L = lo.view<nb::ndim<1>>();
      auto H = hi.view<nb::ndim<1>>();
      if ((int)L.shape(0) != sn.n_svar || (int)H.shape(0) != sn.n_svar)
        throw std::runtime_error("bad subnet bounds shape");
      sn.sv_lo.resize(sn.n_svar);
      sn.sv_hi.resize(sn.n_svar);
      for (int k = 0; k < sn.n_svar; k++) {
        sn.sv_lo[k] = L(k);
        sn.sv_hi[k] = H(k);
      }
    });
  }

  void set_subnet_clamps(int si, nb::ndarray<nb::numpy, int> indices,
                         nb::ndarray<nb::numpy, float> values) {
    dispatch([&](auto &s) {
      if (si < 0 || (size_t)si >= s.sn.size())
        throw std::runtime_error("bad subnet index");
      auto &sn = s.sn[si];
      auto I = indices.view<nb::ndim<1>>();
      auto V = values.view<nb::ndim<3>>();
      const size_t nc = I.shape(0);
      if (V.shape(0) != nc || (int)V.shape(1) != sn.n_node ||
          (int)V.shape(2) != sn.n_modes)
        throw std::runtime_error("bad clamp shape");
      sn.clamp_idx.resize(nc);
      for (size_t c = 0; c < nc; c++) {
        const int sv = I(c);
        if (sv < 0 || sv >= sn.n_svar)
          throw std::runtime_error("clamp state variable index out of range");
        sn.clamp_idx[c] = sv;
      }
      sn.clamp_val.resize(nc * (size_t)sn.n_node * (size_t)sn.n_modes);
      for (size_t c = 0; c < nc; c++)
        for (int j = 0; j < sn.n_node; j++)
          for (int m = 0; m < sn.n_modes; m++)
            sn.clamp_val[(c * (size_t)sn.n_node + (size_t)j) *
                             (size_t)sn.n_modes +
                         (size_t)m] = V(c, j, m);
    });
  }

  void set_cfun_params(int pi, nb::ndarray<nb::numpy, float> v) {
    dispatch([&](auto &s) {
      auto &p = s.pr[pi];
      const int W = std::remove_reference_t<decltype(s)>::W_CONST;
      auto V = v.view<nb::ndim<2>>();
      if ((int)V.shape(1) != W) throw std::runtime_error("cfun lane dim");
      for (int k = 0; k < (int)V.shape(0); k++)
        for (int i = 0; i < W; i++) p.cfp[k * W + i] = V(k, i);
    });
  }

  void set_voi(int si, std::vector<int> v) {
    dispatch([&](auto &s) {
      std::vector<voispec> rows;
      for (int idx : v) rows.push_back({0, idx, 0});
      s.set_observed_rows(si, std::move(rows));
    });
  }

  void set_voi_specs(int si, std::vector<int> flat) {
    dispatch([&](auto &s) {
      std::vector<voispec> rows;
      for (size_t k = 0; k + 2 < flat.size(); k += 3)
        rows.push_back({flat[k], flat[k + 1], flat[k + 2]});
      s.set_observed_rows(si, std::move(rows));
    });
  }

  void set_opts(int integ_id, float dt, int tavg_period) {
    ensure_sim();
    dispatch([&](auto &s) {
      if (s.projection_window_armed() && s.dt != dt)
        throw std::runtime_error("global complete window projection: detach before changing the clock");
      s.integ_id = integ_id;
      s.dt = dt;
      s.tavg_period = tavg_period;
    });
  }

  void set_monitors(int si, nb::list atts) {
    dispatch([&](auto &s) { s.set_monitors(si, atts); });
  }

  void set_global_monitors(nb::list gather_specs, nb::list mon_specs) {
    dispatch([&](auto &s) { s.set_global_monitors(gather_specs, mon_specs); });
  }

  nb::list take_global_monitor_outputs() {
    nb::list out;
    dispatch([&](auto &s) { out = s.take_global_monitor_outputs(); });
    return out;
  }

  nb::list get_monitor_states() const {
    nb::list out;
    dispatch([&](const auto &s) { out = s.get_monitor_states(); });
    return out;
  }

  void set_monitor_states(nb::list states) {
    dispatch([&](auto &s) { s.set_monitor_states(states); });
  }

  nb::list get_global_monitor_states() const {
    nb::list out;
    dispatch([&](const auto &s) { out = s.get_global_monitor_states(); });
    return out;
  }

  void set_global_monitor_states(nb::list states) {
    dispatch([&](auto &s) { s.set_global_monitor_states(states); });
  }

  void set_stim_specs(int si, nb::list specs) {
    dispatch([&](auto &s) { s.set_stim_specs(si, specs); });
  }

  void clear_stim_specs() {
    dispatch([&](auto &s) { s.clear_stim_specs(); });
  }

  void set_subnet_rng(int si, nb::object lane_keys, nb::object gfun,
                      double ntau = 0.0, bool is_multiplicative = false,
                      nb::object mul_a = nb::none(),
                      nb::object mul_b = nb::none()) {
    dispatch([&](auto &s) {
      s.set_subnet_rng(si, lane_keys, gfun, ntau, is_multiplicative, mul_a,
                       mul_b);
    });
  }

  nb::list get_rng_states() const {
    nb::list out;
    dispatch([&](const auto &s) { out = s.get_rng_states(); });
    return out;
  }

  void set_rng_states(nb::list states) {
    dispatch([&](auto &s) { s.set_rng_states(states); });
  }

  void enable_online_features(int si, double dt_sample = 1.0) {
    dispatch([&](auto &s) { s.enable_online_features(si, dt_sample); });
  }

  void disable_online_features() {
    dispatch([&](auto &s) { s.disable_online_features(); });
  }

  nb::dict take_online_features(int si, const std::vector<std::string> &names,
                                int order = 4) const {
    nb::dict out;
    dispatch([&](const auto &s) {
      out = s.take_online_features(si, names, order);
    });
    return out;
  }

  void monitor_begin_run() {
    dispatch([&](auto &s) { s.monitor_begin_run(); });
  }

  nb::list run(int nstep, int chunk_size, nb::object noise, nb::object stim,
               bool need_tavg = true, bool need_ctavg = true) {
    nb::list outs;
    dispatch([&](auto &s) {
      outs = s.run(nstep, chunk_size, noise, stim, need_tavg, need_ctavg);
    });
    return outs;
  }
};

// ---- Feature extraction bindings (cph_features.hpp) ------------------------

template <typename T>
static nb::dict compute_features_impl(
    nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> ts,
    const std::vector<std::string> &names, double fs, int bins, int order,
    nb::object lo, nb::object hi) {
  const int n_series = static_cast<int>(ts.shape(0));
  const int n_time = static_cast<int>(ts.shape(1));
  FeatureParams p;
  p.fs = fs;
  p.bins = bins;
  p.order = order;
  if (!lo.is_none() && !hi.is_none()) {
    p.has_band = true;
    p.lo = nb::cast<double>(lo);
    p.hi = nb::cast<double>(hi);
  }
  const T *ptr = ts.data();
  std::vector<BatchFeatureOutput> res;
  {
    nb::gil_scoped_release nogil;
    res = compute_batch_features<T>(ptr, n_series, n_time, names, p);
  }

  nb::dict out;
  for (const auto &bo : res) {
    if (bo.name == "psd_raw") {
      auto freqs_arr = make_owned_array<double>({(size_t)bo.n_freq});
      if (!bo.freqs.empty())
        std::memcpy(freqs_arr.data(), bo.freqs.data(),
                    bo.freqs.size() * sizeof(double));
      auto psd_arr =
          make_owned_array<double>({(size_t)n_series, (size_t)bo.n_freq});
      if (!bo.data.empty())
        std::memcpy(psd_arr.data(), bo.data.data(),
                    bo.data.size() * sizeof(double));
      out[nb::str(bo.name.c_str())] = nb::make_tuple(freqs_arr, psd_arr);
    } else if (bo.name == "moments" || bo.name == "spectrum_stats" ||
               bo.name == "spectrum_moments") {
      auto arr = make_owned_array<double>(
          {(size_t)bo.n_components, (size_t)n_series});
      if (!bo.data.empty())
        std::memcpy(arr.data(), bo.data.data(),
                    bo.data.size() * sizeof(double));
      out[nb::str(bo.name.c_str())] = arr;
    } else {
      auto arr = make_owned_array<double>({(size_t)n_series});
      if (!bo.data.empty())
        std::memcpy(arr.data(), bo.data.data(),
                    bo.data.size() * sizeof(double));
      out[nb::str(bo.name.c_str())] = arr;
    }
  }
  return out;
}

struct py_online_feature_accum {
  int width = 8;
  std::unique_ptr<OnlineFeatureAccum<1>> a1;
  std::unique_ptr<OnlineFeatureAccum<4>> a4;
  std::unique_ptr<OnlineFeatureAccum<8>> a8;
  std::unique_ptr<OnlineFeatureAccum<16>> a16;

  explicit py_online_feature_accum(int width_ = 8) : width(width_) {
    switch (width_) {
      case 1:  a1  = std::make_unique<OnlineFeatureAccum<1>>();  break;
      case 4:  a4  = std::make_unique<OnlineFeatureAccum<4>>();  break;
      case 8:  a8  = std::make_unique<OnlineFeatureAccum<8>>();  break;
      case 16: a16 = std::make_unique<OnlineFeatureAccum<16>>(); break;
      default:
        throw std::invalid_argument(
            "OnlineFeatureAccum width must be one of {1, 4, 8, 16}");
    }
  }

  template <typename F> void dispatch(F &&f) {
    switch (width) {
      case 1:  f(*a1);  break;
      case 4:  f(*a4);  break;
      case 8:  f(*a8);  break;
      case 16: f(*a16); break;
    }
  }

  template <typename F> void dispatch(F &&f) const {
    switch (width) {
      case 1:  f(*a1);  break;
      case 4:  f(*a4);  break;
      case 8:  f(*a8);  break;
      case 16: f(*a16); break;
    }
  }

  void reset(int n_voi, int n_node, double dt_sample = 1.0) {
    dispatch([&](auto &a) { a.reset(n_voi, n_node, dt_sample); });
  }

  template <typename T>
  void update_typed(nb::ndarray<nb::numpy, const T, nb::c_contig> obs_row) {
    dispatch([&](auto &a) {
      const size_t expected = a.num_series();
      if (obs_row.size() != expected)
        throw std::invalid_argument("OnlineFeatureAccum::update: size mismatch");
      a.update(obs_row.data());
    });
  }

  nb::dict finalize(const std::vector<std::string> &names,
                    int order = 4) const {
    nb::dict out;
    dispatch([&](const auto &a) {
      constexpr int W = std::remove_reference_t<decltype(a)>::WIDTH;
      for (const auto &name : names) {
        const int n_comp = OnlineFeatureAccum<W>::feature_components(name, order);
        std::vector<size_t> shape;
        if (name == "moments") {
          shape = {(size_t)n_comp, (size_t)a.n_voi, (size_t)a.n_node, (size_t)W};
        } else {
          shape = {(size_t)a.n_voi, (size_t)a.n_node, (size_t)W};
        }
        auto arr = make_owned_array<double>(shape);
        a.finalize_into(name, arr.data(), order);
        out[nb::str(name.c_str())] = arr;
      }
    });
    return out;
  }
};

// Model metadata for the dfuns compiled into this module at build time:
// class name -> {mid, n_parm, n_cvar, n_svar, parm_names}, i.e. the same
// shape dfungen's meta dict has.  Read straight from the generated table, so
// Python and the kernels cannot disagree about ids or parameter order.
// Empty when the extension was built without the generated TU.
static nb::dict generic_model_table() {
  nb::dict out;
#ifdef CPH_HAVE_BUILTIN_GEN
  const int n = cph_builtin_count();
  for (int i = 0; i < n; i++) {
    const cph_gen_entry *e = cph_builtin_entry(i);
    if (e == nullptr || e->name == nullptr) continue;
    nb::list parm;
    for (int k = 0; k < e->n_parm; k++)
      parm.append(nb::str(e->parm_names[k] ? e->parm_names[k] : ""));
    nb::dict rec;
    rec["mid"] = e->mid;
    rec["n_parm"] = e->n_parm;
    rec["n_cvar"] = e->n_cvar;
    rec["n_svar"] = e->n_svar;
    rec["parm_names"] = parm;
    rec["signature"] = nb::str(e->signature ? e->signature : "");
    out[nb::str(e->name)] = rec;
  }
#endif
  return out;
}

// debug helper: single MPR dfun evaluation (1 node, 1 lane)
std::vector<float> dbg_mpr_dfun(float r, float V, float c,
                                nb::ndarray<nb::numpy, float> parr) {
  float x[2] = {r, V}, dx[2] = {0.f, 0.f};
  auto P = parr.view<nb::ndim<1>>();
  float p[6];
  for (int k = 0; k < 6; k++) p[k] = P(k);
  float cc[1] = {c};
  dfun_mpr<1>(dx, x, 0, 0, 1, 1, cc, p, 6, 1);
  return {dx[0], dx[1]};
}

// debug helper: full heun step replicating subnet::step_heun (1 node)
std::vector<float> dbg_mpr_heun(float r, float V, float c,
                                nb::ndarray<nb::numpy, float> parr,
                                float dt) {
  float x[2] = {r, V}, dx[2] = {0.f, 0.f}, xi[2], dxi[2] = {0.f, 0.f};
  auto P = parr.view<nb::ndim<1>>();
  float p[6];
  for (int k = 0; k < 6; k++) p[k] = P(k);
  float cc[1] = {c};
  dfun_mpr<1>(dx, x, 0, 0, 1, 1, cc, p, 6, 1);
  for (int k = 0; k < 2; k++) xi[k] = x[k] + dt * dx[k];
  xi[0] = xi[0] * (xi[0] > 0.f);
  dfun_mpr<1>(dxi, xi, 0, 0, 1, 1, cc, p, 6, 1);
  for (int k = 0; k < 2; k++) x[k] += dt * 0.5f * (dx[k] + dxi[k]);
  x[0] = x[0] * (x[0] > 0.f);
  return {x[0], x[1], dx[0], dx[1], dxi[0], dxi[1]};
}

}  // namespace cph

NB_MODULE(_cpp_hybrid, m) {
  m.doc() = "C++ hybrid simulator core (runtime SIMD kernels, no codegen)";
  m.def("enable_generic_dfuns", [](nb::int_ fn) {
    cph::cph_generic_fn = reinterpret_cast<cph::cph_generic_fn_t>(
        static_cast<uint64_t>(fn));
  }, nb::arg("fn"));
  m.def("dbg_mpr_dfun", &cph::dbg_mpr_dfun);
  m.def("dbg_mpr_heun", &cph::dbg_mpr_heun);
  m.def("generic_model_table", &cph::generic_model_table);
  m.def("compute_features", &cph::compute_features_impl<float>,
        nb::arg("ts"), nb::arg("names"), nb::arg("fs") = 1.0,
        nb::arg("bins") = 20, nb::arg("order") = 4,
        nb::arg("lo") = nb::none(), nb::arg("hi") = nb::none());
  m.def("compute_features", &cph::compute_features_impl<double>,
        nb::arg("ts"), nb::arg("names"), nb::arg("fs") = 1.0,
        nb::arg("bins") = 20, nb::arg("order") = 4,
        nb::arg("lo") = nb::none(), nb::arg("hi") = nb::none());

  nb::class_<cph::py_online_feature_accum>(m, "OnlineFeatureAccum")
      .def(nb::init<int>(), nb::arg("width") = 8)
      .def("reset", &cph::py_online_feature_accum::reset,
           nb::arg("n_voi"), nb::arg("n_node"), nb::arg("dt_sample") = 1.0)
      .def("update", &cph::py_online_feature_accum::update_typed<float>,
           nb::arg("obs_row"))
      .def("update", &cph::py_online_feature_accum::update_typed<double>,
           nb::arg("obs_row"))
      .def("finalize", &cph::py_online_feature_accum::finalize,
           nb::arg("names"), nb::arg("order") = 4);

  nb::class_<cph::sim>(m, "Sim")
      .def(nb::init<int>(), nb::arg("width") = 8)
      .def("add_subnet", &cph::sim::add_subnet, nb::arg("n_node"),
           nb::arg("n_svar"), nb::arg("n_parm"), nb::arg("n_cvar"),
           nb::arg("model_id"), nb::arg("horizon"), nb::arg("n_modes"),
           nb::arg("k") = 1, nb::arg("dt") = 0.001f, nb::arg("integ_id") = -1,
           nb::arg("generic_dfun") = 0)
      .def("add_projection", &cph::sim::add_projection, nb::arg("src_sn"),
           nb::arg("tgt_sn"), nb::arg("w"), nb::arg("idx"), nb::arg("ptr"),
           nb::arg("del"), nb::arg("cfun_id"), nb::arg("src_cvars"),
           nb::arg("tgt_cvar"), nb::arg("n_cfun_parm"), nb::arg("scale"),
           nb::arg("ts") = 1.0f, nb::arg("tgt_state_cvar") = 0,
           nb::arg("mode_map") = nb::none())
      .def("set_subnet_params", &cph::sim::set_subnet_params)
      .def("set_subnet_state", &cph::sim::set_subnet_state)
      .def("set_subnet_bounds", &cph::sim::set_subnet_bounds, nb::arg("si"),
           nb::arg("lo"), nb::arg("hi"))
      .def("set_subnet_clamps", &cph::sim::set_subnet_clamps, nb::arg("si"),
           nb::arg("indices"), nb::arg("values"))
      .def("set_cfun_params", &cph::sim::set_cfun_params)
      .def("set_voi", &cph::sim::set_voi)
      .def("set_voi_specs", &cph::sim::set_voi_specs)
      .def("set_opts", &cph::sim::set_opts)
      .def("set_monitors", &cph::sim::set_monitors, nb::arg("si"),
           nb::arg("atts"))
      .def("set_global_monitors", &cph::sim::set_global_monitors,
           nb::arg("gather_specs"), nb::arg("mon_specs"))
      .def("take_global_monitor_outputs",
           &cph::sim::take_global_monitor_outputs)
      .def("get_monitor_states", &cph::sim::get_monitor_states)
      .def("set_monitor_states", &cph::sim::set_monitor_states,
           nb::arg("states"))
      .def("get_global_monitor_states",
           &cph::sim::get_global_monitor_states)
      .def("set_global_monitor_states",
           &cph::sim::set_global_monitor_states, nb::arg("states"))
      .def("monitor_begin_run", &cph::sim::monitor_begin_run)
      .def("set_stim_specs", &cph::sim::set_stim_specs, nb::arg("si"),
           nb::arg("specs"))
      .def("clear_stim_specs", &cph::sim::clear_stim_specs)
      .def("set_subnet_rng", &cph::sim::set_subnet_rng, nb::arg("si"),
           nb::arg("lane_keys"), nb::arg("gfun"), nb::arg("ntau") = 0.0,
           nb::arg("is_multiplicative") = false,
           nb::arg("mul_a") = nb::none(), nb::arg("mul_b") = nb::none())
      .def("get_rng_states", &cph::sim::get_rng_states)
      .def("set_rng_states", &cph::sim::set_rng_states, nb::arg("states"))
      .def("enable_online_features", &cph::sim::enable_online_features,
           nb::arg("si"), nb::arg("dt_sample") = 1.0)
      .def("disable_online_features", &cph::sim::disable_online_features)
      .def("take_online_features", &cph::sim::take_online_features,
           nb::arg("si"), nb::arg("names"), nb::arg("order") = 4)
      .def("get_subnet_state", &cph::sim::get_subnet_state)
      .def("get_subnet_buf", &cph::sim::get_subnet_buf, nb::arg("si"))
      .def("set_subnet_buf", &cph::sim::set_subnet_buf, nb::arg("si"),
           nb::arg("v"))
      .def("t_abs", &cph::sim::t_abs)
      .def("set_t_abs", &cph::sim::set_t_abs, nb::arg("t"))
      .def("run", &cph::sim::run, nb::arg("nstep"), nb::arg("chunk_size") = 0,
           nb::arg("noise") = nb::none(), nb::arg("stim") = nb::none(),
           nb::arg("need_tavg") = true, nb::arg("need_ctavg") = true);
}
