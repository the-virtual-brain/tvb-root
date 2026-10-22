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
#include <nanobind/stl/vector.h>  // std::vector<> caster
// gil release via scoped release below

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

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
static bool cph_have_generic = false;

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
    for (int i = 0; i < W; i++) cx[i] = p[0 * W + i] * cx[i] + p[1 * W + i];
    break;
  case 1:  // Scaling: a*gx
    for (int i = 0; i < W; i++) cx[i] = p[0 * W + i] * cx[i];
    break;
  case 2:  // Sigmoidal: cmin + (cmax-cmin)/(1+exp(-a*(gx-midpoint)/sigma))
    for (int i = 0; i < W; i++)
      cx[i] = (float)((double)p[3 * W + i] +
              ((double)p[4 * W + i] - (double)p[3 * W + i]) /
                  (1.0 + exp(-(double)p[0 * W + i] *
                             ((double)cx[i] - (double)p[2 * W + i]) /
                             (double)p[1 * W + i])));
    break;
  case 3: case 4: case 6:  // Difference / Kuramoto / SJR: a*gx
    for (int i = 0; i < W; i++) cx[i] = p[0 * W + i] * cx[i];
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
    for (int i = 0; i < W; i++) v[0 * W + i] = v[0 * W + i] - xi[i];
    break;
  case 4:  // Kuramoto: sin(v0 - xi)
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
  case MOD_MPR:      n_svar = 2; n_parm = 6;  n_cvar = 2; break;
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

// MontbrioPazoRoxin: svars (r, V); params (tau, I, Delta, J, eta, cr)
template <int W>
CPH_NOINLINE static void dfun_mpr(float *dx, const float *x, int node, int mode,
                            int n_node, int n_modes, const float *c,
                            const float *p, int n_parm) {
  const float *r0 = x + XOFF(0);
  const float *V0 = x + XOFF(1);
  float *dr = dx + XOFF(0);
  float *dV = dx + XOFF(1);
  const float *c0 = c + COFF(0);
  const float *pk = p + (size_t)node * n_parm * W;
  for (int i = 0; i < W; i++) {
    float tau = pk[0 * W + i], I = pk[1 * W + i], Delta = pk[2 * W + i],
          J = pk[3 * W + i], eta = pk[4 * W + i], cr = pk[5 * W + i];
    float r = r0[i] * (r0[i] > 0.f);
    float V = V0[i];
    dr[i] = (1.f / tau) * (Delta / (M_PI_F * tau) + 2.f * r * V);
    dV[i] = (1.f / tau) * (V * V + eta + J * tau * r + I + cr * c0[i] -
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
    bool legacy_ee = CRBL(p, 82) != 0.0;
    bool add_nz    = CRBL(p, 83) != 0.0;
    double P_grc[5], P_goc[5], P_mli[5], P_pc[5];
    for (int k = 0; k < 5; k++) {
      P_grc[k] = CRBL(p, 62 + k);
      P_goc[k] = CRBL(p, 67 + k);
      P_mli[k] = CRBL(p, 72 + k);
      P_pc[k]  = CRBL(p, 77 + k);
    }
    double gL_grc = CRBL(p, 0), gL_goc = CRBL(p, 1), gL_mli = CRBL(p, 2),
           gL_pc = CRBL(p, 3);
    double EL_grc = CRBL(p, 4), EL_goc = CRBL(p, 5), EL_mli = CRBL(p, 6),
           EL_pc = CRBL(p, 7);
    double Cm_grc = CRBL(p, 8), Cm_goc = CRBL(p, 9), Cm_mli = CRBL(p, 10),
           Cm_pc = CRBL(p, 11);
    double E_e = CRBL(p, 12), E_i = CRBL(p, 13);
    double Q_mf_grc = CRBL(p, 14), Q_mf_goc = CRBL(p, 15);
    double Q_grc_goc = CRBL(p, 16), Q_grc_mli = CRBL(p, 17),
           Q_grc_pc = CRBL(p, 18);
    double Q_goc_grc = CRBL(p, 19), Q_goc_goc = CRBL(p, 20);
    double Q_mli_mli = CRBL(p, 21), Q_mli_pc = CRBL(p, 22);
    double tau_mf_grc = CRBL(p, 23), tau_mf_goc = CRBL(p, 24);
    double tau_grc_goc = CRBL(p, 25), tau_grc_mli = CRBL(p, 26),
           tau_grc_pc = CRBL(p, 27);
    double tau_goc_grc = CRBL(p, 28), tau_goc_goc = CRBL(p, 29);
    double tau_mli_mli = CRBL(p, 30), tau_mli_pc = CRBL(p, 31);
    double K_mossy_grc = CRBL(p, 32), K_mossy_goc = CRBL(p, 33);
    double K_grc_goc = CRBL(p, 34), K_grc_mli = CRBL(p, 35),
           K_grc_pc = CRBL(p, 36);
    double K_goc_goc = CRBL(p, 37), K_mli_mli = CRBL(p, 38),
           K_mli_pc = CRBL(p, 39);
    double alpha_grc = CRBL(p, 45), alpha_goc = CRBL(p, 46),
           alpha_mli = CRBL(p, 47), alpha_pc = CRBL(p, 48);
    double T = CRBL(p, 49);
    double tau_OU = CRBL(p, 50), weight_noise = CRBL(p, 51);
    double eie = CRBL(p, 52), eiin = CRBL(p, 53), ein_ex = CRBL(p, 54);
    double fm = CRBL(p, 55), fp = CRBL(p, 56);
    double mf_to_grc = CRBL(p, 57), mf_to_goc = CRBL(p, 58);
    double pf_to_goc = CRBL(p, 59), pf_to_mli = CRBL(p, 60),
           pf_to_pc = CRBL(p, 61);

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
                                 const float *c, const float *p, int n_parm) {
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
    if (cph::cph_have_generic && cph::cph_generic_fn &&
        cph::cph_generic_fn(model_id, dx, x, node, mode, n_node, n_modes,
                            c, p, n_parm, W))
      return;
    // Neither table covered the id, so dx was never written: letting the run
    // continue would silently never-evolve (Sim.add_subnet(model_id=...) with
    // a stale or out-of-range id).  Fail loudly, naming the id.
    throw std::runtime_error("unknown generic model id " +
                             std::to_string(model_id));
  }
  switch (model_id) {
  case MOD_MPR: dfun_mpr<W>(dx, x, node, mode, n_node, n_modes, c, p, n_parm); break;
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

struct voispec { int kind, a, b; };  // kind 0 = state a; 1 = diff a-b

// ---- kernel-side monitor engines -------------------------------------------
// Step-level monitor collection running inside the GIL-released step loop.
// The engines mirror the shared Python reference implementation in
// nb_hybrid.py::_apply_monitors (which drives tvb.simulator.monitors
// sample() methods); the Python path stays in place as the parity reference
// for the numba backend and as the fallback when the engines are disabled.
// Each engine consumes the same per-step observed row the kernel already
// accumulates into tacc (voi-spec sums, modes summed), so input values are
// bit-identical to what the Python reference sees.
enum { KMON_RAW = 1, KMON_SUBSAMPLE = 2, KMON_BOLD = 3 };

template <int W> struct kmon {
  int slot = -1;          // monitor list index (state identity)
  int kind = 0;           // KMON_*
  int istep = 1;          // steps per sample (subsample / bold period)
  int interim_istep = 1;  // bold: steps per interim-stock write
  int stock_steps = 0;    // bold: outer stock length S
  int n_out_v = 0;        // output rows per sample (full row or voi subset)
  double k1v0 = 1.0;      // bold: (dot - 1) * k1v0 for FirstOrderVolterra
  bool first_order = false;
  double dt = 0.1;        // master step dt (float64, matches the reference)
  std::vector<int> voi;         // bold: indices into the per-step row
  std::vector<double> hrf;      // bold: (S,) G[::-1]
  // persistent state (mirrors the Python Bold monitor's _interim_stock,
  // _stock and step counter; survives across run() calls)
  std::vector<float> interim;   // (interim_istep, n_out_v, n_node, W)
  std::vector<float> stock;     // (stock_steps, n_out_v, n_node, W)
  uint64_t m_step = 0;          // 1-based global monitor step counter
  uint64_t run_start = 0;       // m_step at the start of the current run()
  // per-run() call outputs (cleared at each simw::run entry)
  std::vector<double> t_out;
  std::vector<float> d_out;

  void begin_run() { run_start = m_step; }
  void begin_call() { t_out.clear(); d_out.clear(); }

  // rowbase: the per-step observed row (n_voi, n_node, W) for this subnet.
  // step_one is invoked once per master step (same cadence the Python
  // path drives the reference monitor with).
  void step_one(const float *rowbase, int n_voi, int n_node) {
    m_step++;
    const uint64_t within = m_step - run_start;  // 1-based within this run
    if (kind == KMON_RAW) {
      // reference stamp: the per-chunk time the Python path emits for
      // chunk_size == 1 is (within) * float32-quantized dt (backend.py's
      // master-grid times), so the engine quantizes dt to float once
      t_out.push_back((double)within * (double)(float)dt);
      const size_t n = (size_t)n_voi * n_node * W;
      for (size_t k = 0; k < n; k++) d_out.push_back(rowbase[k]);
      return;
    }
    if (kind == KMON_SUBSAMPLE) {
      if (within % (uint64_t)istep == 0) {
        // same float32-quantized master-grid stamp as the reference path
        t_out.push_back((double)within * (double)(float)dt);
        const size_t n = (size_t)n_voi * n_node * W;
        for (size_t k = 0; k < n; k++) d_out.push_back(rowbase[k]);
      }
      return;
    }
    if (kind == KMON_BOLD) {
      const int L = (int)voi.size();
      const int ii = interim_istep;
      const int S = stock_steps;
      const size_t plane = (size_t)L * n_node * W;
      // interim stock write: Python index ((step % ii) - 1) wraps to ii-1
      {
        const int slot = (int)((m_step % (uint64_t)ii) + (uint64_t)ii - 1u) % ii;
        for (int v = 0; v < L; v++) {
          const int rv = voi[v];
          if (rv < 0 || rv >= n_voi)
            throw std::runtime_error(
                "bold monitor voi index out of range for subnet row");
          const float *src = rowbase + (size_t)rv * n_node * W;
          float *dst = interim.data() + ((size_t)slot * L + v) * n_node * W;
          std::memcpy(dst, src, (size_t)n_node * W * sizeof(float));
        }
      }
      // interim mean -> outer stock at Python index ((step//ii % S) - 1)
      if (m_step % (uint64_t)ii == 0) {
        const int srow =
            (int)((m_step / (uint64_t)ii) % (uint64_t)S) - 1;
        const size_t sroww = (srow < 0) ? (size_t)(srow + S) : (size_t)srow;
        for (size_t k = 0; k < plane; k++) {
          double acc = 0.0;
          for (int q = 0; q < ii; q++)
            acc += (double)interim[(size_t)q * plane + k];
          stock[sroww * plane + k] = (float)(acc / (double)ii);
        }
      }
      // monitor period: hrf dot over the outer stock, rolled by r
      if (m_step % (uint64_t)istep == 0) {
        const int r = (int)((m_step / (uint64_t)ii) % (uint64_t)S) - 1;
        t_out.push_back((double)m_step * dt);
        for (size_t k = 0; k < plane; k++) {
          double acc = 0.0;
          for (int q = 0; q < S; q++) {
            int hk = (q - r) % S;
            if (hk < 0) hk += S;
            acc += hrf[(size_t)hk] * (double)stock[(size_t)q * plane + k];
          }
          const double out = first_order ? (acc - 1.0) * k1v0 : acc;
          d_out.push_back((float)out);
        }
      }
      return;
    }
  }
};

template <int W> struct subnet {
  int n_node = 0, n_svar = 2, n_parm = 0, n_cvar = 1, n_modes = 1;
  int model_id = MOD_MPR;
  uint32_t K = 1;     // steps per master tick: dt == K * dt0 (multi-dt)
  float dt = 0.001f;  // this subnet's own integration step (float32 of dt_j)
  std::vector<voispec> voi_specs;
  std::vector<float> x, dx, xi, dxi;  // (n_svar, n_node, n_modes, W)
  std::vector<float> c;               // (n_cvar, n_node, n_modes, W)
  std::vector<float> p;               // (n_node, n_parm, W)
  std::vector<float> buf;             // (n_svar, n_node, n_modes, H, W)
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
      for (int i = 0; i < W; i++) dst[i] = srcv[i];
    }
  }

  void step_euler(float dt, const float *noise) {
    for (int j = 0; j < n_node; j++)
      for (int m = 0; m < n_modes; m++)
        dfun_dispatch<W>(model_id, dx.data(), x.data(), j, m, n_node, n_modes,
                         c.data(), p.data(), n_parm);
    if (noise) {
      // noise layout (n_svar, n_node, n_modes, W) for this step
      for (size_t k = 0; k < x.size(); k++)
        x[k] += dt * dx[k] + noise[k];
    } else {
      for (size_t k = 0; k < x.size(); k++) x[k] += dt * dx[k];
    }
    clamp_dispatch<W>(model_id, x.data(), 0, n_node, n_modes);
  }

  void step_heun(float dt, const float *noise) {
    for (int j = 0; j < n_node; j++)
      for (int m = 0; m < n_modes; m++)
        dfun_dispatch<W>(model_id, dx.data(), x.data(), j, m, n_node, n_modes,
                         c.data(), p.data(), n_parm);
    if (noise) {
      for (size_t k = 0; k < x.size(); k++) xi[k] = x[k] + dt * dx[k] + noise[k];
    } else {
      for (size_t k = 0; k < x.size(); k++) xi[k] = x[k] + dt * dx[k];
    }
    clamp_dispatch<W>(model_id, xi.data(), 0, n_node, n_modes);
    // coupling stays fixed across stages (matches nb_hybrid)
    for (int j = 0; j < n_node; j++)
      for (int m = 0; m < n_modes; m++)
        dfun_dispatch<W>(model_id, dxi.data(), xi.data(), j, m, n_node, n_modes,
                         c.data(), p.data(), n_parm);
    if (noise) {
      for (size_t k = 0; k < x.size(); k++)
        x[k] += dt * 0.5f * (dx[k] + dxi[k]) + noise[k];
    } else {
      for (size_t k = 0; k < x.size(); k++) x[k] += dt * 0.5f * (dx[k] + dxi[k]);
    }
    clamp_dispatch<W>(model_id, x.data(), 0, n_node, n_modes);
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
    float gthr[8 * W];
    bool use_gT = (cfun_id == 9 && n_cfun_parm > 5 &&
                   cfp[(size_t)5 * W] != 0.f);
    if (use_gT) {
      const int nnz_total = (int)ptr[ptr.size() - 1];
      const int cv1 = src_cvars[1];
      const float *sbuf1 = src.buf.data() +
          (size_t)cv1 * src.n_node * nms * src.H * W;
      for (int m = 0; m < nms && m < 8; m++)
        for (int i = 0; i < W; i++) gthr[m * W + i] = 0.f;
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
          for (int m = 0; m < nms && m < 8; m++) {
            const float *b0 = sbuf1 +
                (((size_t)idx[nz] * nms + m) * src.H + s0) * W;
            const float *b1 = sbuf1 +
                (((size_t)idx[nz] * nms + m) * src.H + s1) * W;
            for (int i = 0; i < W; i++)
              gthr[m * W + i] += b0[i] + alpha * (b1[i] - b0[i]);
          }
        }
      } else {
        for (int nz = 0; nz < nnz_total; nz++) {
          // exact single-slot read (legacy K == 1, or intra multi-dt —
          // decision 3); see the slow-path branch below for the derivation
          const uint32_t slot = (t1 / Ksrc - 2u - del[nz]) & Hm1;
          for (int m = 0; m < nms && m < 8; m++) {
            const float *b = sbuf1 +
                (((size_t)idx[nz] * nms + m) * src.H + slot) * W;
            for (int i = 0; i < W; i++) gthr[m * W + i] += b[i];
          }
        }
      }
      const float inv = 1.0f / (float)nnz_total;
      for (int m = 0; m < nms && m < 8; m++)
        for (int i = 0; i < W; i++) gthr[m * W + i] *= inv;
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
      const float ts_scale = scale * ts;
      for (int j = 0; j < nn; j++) {
        float cx[8 * W];
        for (int i = 0; i < W; i++) cx[i] = 0.f;
        const int i0 = ptr[j], i1 = ptr[j + 1];
        for (int nz = i0; nz < i1; nz++) {
          const uint32_t slot = ((uint32_t)t - 1u - del[nz]) & Hm1;
          const float wgt = w[nz];
          const float *b = sbuf + ((size_t)idx[nz] * src.H + slot) * W;
          for (int i = 0; i < W; i++) cx[i] += wgt * b[i];
        }
        cfun_post<W>(cfun_id, cx, cfp.data());
        float *dst = out + (size_t)j * W;
        for (int i = 0; i < W; i++) dst[i] += ts_scale * cx[i];
      }
      return;
    }
    // mode_map: (nms, nmt) row-major
    for (int j = 0; j < nn; j++) {
      // target state for per-edge pre(x_i): current state at tgt_state_cvar
      float xi[8 * W];
      for (int m = 0; m < nmt && m < 8; m++) {
        const float *xv = tgt.x.data() +
            ((size_t)tgt_state_cvar * nn * nmt + j * nmt + m) * W;
        for (int i = 0; i < W; i++) xi[m * W + i] = xv[i];
      }
      float cx[8 * W];
      for (int m = 0; m < nms && m < 8; m++) {
        for (int i = 0; i < W; i++) cx[m * W + i] = 0.f;
      }
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
        for (int m = 0; m < nms && m < 8; m++) {
          float v[4 * W];  // up to 4 source cvars
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
            if (use_gT)
              for (int i = 0; i < W; i++) v[W + i] = gthr[m * W + i];
            cfun_pre<W>(cfun_id, v, cfp.data(), xi + (nmt < 8 ? 0 : 0) * W);
            for (int i = 0; i < W; i++) cx[m * W + i] += wgt * v[i];
          } else {
            for (int i = 0; i < W; i++) {
              float acc = 0.f;
              for (size_t cv = 0; cv < src_cvars.size(); cv++)
                acc += v[cv * W + i];
              cx[m * W + i] += wgt * acc;
            }
          }
        }
      }
      // post per source mode, then mode-mix into target modes

      for (int m = 0; m < nms && m < 8; m++)
        cfun_post<W>(cfun_id, cx + m * W, cfp.data());

      for (int mt = 0; mt < nmt && mt < 8; mt++) {
        float *dst = out + ((size_t)j * nmt + mt) * W;
        for (int i = 0; i < W; i++) {
          float acc = 0.f;
          for (int m = 0; m < nms && m < 8; m++)
            acc += cx[m * W + i] * (float)mode_map[m * nmt + mt];
          dst[i] += scale * ts * acc;
        }
      }
    }
  }
  mutable std::vector<float> mode_map;  // (nms, nmt) row-major
};

// ---- simulator (one per width instantiation) -------------------------------

template <int W> struct simw {
  static constexpr int W_CONST = W;
  std::vector<subnet<W>> sn;
  std::vector<proj<W>> pr;
  int integ_id = 1;  // 0 = euler, 1 = heun
  float dt = 0.001f;
  int tavg_period = 1;
  std::vector<std::vector<voispec>> voi_specs;  // per subnet
  std::vector<std::vector<kmon<W>>> mons;       // per-subnet monitor engines
  uint64_t t_abs = 0;

  // Replace subnet si's armed monitor engines with `atts`.  Engines whose
  // (slot, kind) match an existing engine are kept with their state intact
  // (matching the Python reference, which keys monitor runtimes by
  // (kind, monitor index, subnet)); everything else is recreated; engines no
  // longer armed are dropped.  atts entries are tuples:
  //   (slot, kind, istep, interim_istep, stock_steps, k1v0, first_order,
  //    voi (int32 ndarray), hrf (float64 ndarray or None), dt)
  void set_monitors(int si, nb::list atts) {
    if ((size_t)si >= sn.size())
      throw std::runtime_error("bad subnet index in set_monitors");
    if (mons.size() < sn.size()) mons.resize(sn.size());
    auto &cur = mons[si];
    const int n_voi = (int)voi_specs[si].size();
    const int n_node = sn[si].n_node;
    const double dt64 = (double)dt;
    std::vector<kmon<W>> next;
    next.reserve(atts.size());
    for (auto item : atts) {
      nb::tuple a = nb::cast<nb::tuple>(item);
      const int slot = nb::cast<int>(a[0]);
      const int kind = nb::cast<int>(a[1]);
      auto it = std::find_if(cur.begin(), cur.end(), [&](const kmon<W> &e) {
        return e.slot == slot && e.kind == kind;
      });
      if (it != cur.end()) {  // same (slot, kind): keep state (Python semantics)
        next.push_back(std::move(*it));
        continue;
      }
      kmon<W> m;
      m.slot = slot;
      m.kind = kind;
      m.istep = nb::cast<int>(a[2]);
      if (m.istep < 1) m.istep = 1;
      m.interim_istep = nb::cast<int>(a[3]);
      if (m.interim_istep < 1) m.interim_istep = 1;
      m.stock_steps = nb::cast<int>(a[4]);
      if (m.stock_steps < 1) m.stock_steps = 1;
      m.k1v0 = nb::cast<double>(a[5]);
      m.first_order = nb::cast<bool>(a[6]);
      if (kind == KMON_BOLD) {
        auto V = nb::cast<nb::ndarray<nb::numpy, int>>(a[7]);
        auto Vv = V.view<nb::ndim<1>>();
        m.voi.resize(Vv.shape(0));
        for (size_t k = 0; k < Vv.shape(0); k++) m.voi[k] = Vv(k);
        if (!a[8].is_none()) {
          auto H = nb::cast<nb::ndarray<nb::numpy, double>>(a[8]);
          auto Hv = H.view<nb::ndim<1>>();
          m.hrf.assign(Hv.shape(0), 0.0);
          for (size_t k = 0; k < Hv.shape(0); k++) m.hrf[k] = Hv(k);
        }
        m.n_out_v = (int)m.voi.size();
        const size_t plane = (size_t)m.n_out_v * n_node * W;
        m.interim.assign((size_t)m.interim_istep * plane, 0.f);
        m.stock.assign((size_t)m.stock_steps * plane, 0.f);
      } else {
        m.n_out_v = n_voi;  // raw / subsample emit the full observed row
      }
      m.dt = nb::cast<double>(a[9]);
      (void)dt64;
      next.push_back(std::move(m));
    }
    cur = std::move(next);
  }

  void monitor_begin_run() {
    for (auto &mons_s : mons)
      for (auto &m : mons_s) m.begin_run();
  }

  void monitor_begin_call() {
    for (auto &mons_s : mons)
      for (auto &m : mons_s) m.begin_call();
  }

  // Run nstep steps. Output per chunk of `chunk_size` steps:
  // tavg (n_chunks, n_voi, n_node, W) and ctavg (n_chunks, n_cvar, n_node, W),
  // each the mean of the per-step values within the chunk (nb_hybrid semantics).
  // noise: optional (n_svar, n_node, W, nstep) float32; stim: optional
  // (n_cvar, n_node, W, nstep) float32 added to c before integration.
  nb::list run(int nstep, int chunk_size, nb::object noise_obj,
               nb::object stim_obj) {
    const int cs = chunk_size > 0 ? chunk_size : tavg_period;
    const int n_chunks = (nstep + cs - 1) / cs;
    const size_t n_sn = sn.size();
    monitor_begin_call();

    // per-step noise/stim pointers (lane-major innermost)
    // Each of noise_obj / stim_obj is either a single array (single-subnet
    // layout, applied to every subnet) or a sequence with one entry per
    // subnetwork (None = no input for that subnet).
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
      tacc[s].assign((size_t)n_chunks * n_voi * sn[s].n_node * W, 0.f);
      // ctavg layout matches the returned array and sub.c: (n_chunks,
      // n_cvar, n_node, n_modes, W).  The n_modes factor was missing here
      // while the accumulation below (and the output copy) indexed with
      // it, so any subnet with n_modes > 1 wrote and read past the end of
      // the accumulator (heap corruption, garbage ctavg).
      cacc[s].assign((size_t)n_chunks * sn[s].n_cvar * sn[s].n_node *
                         sn[s].n_modes * W, 0.f);
      counts[s].assign(n_chunks, 0);
    }

    std::exception_ptr pend = nullptr;
    {
      nb::gil_scoped_release nogil;
      try {
      for (int step = 0; step < nstep; step++) {
        const size_t chunk = (size_t)(step / cs);
        // zero the per-step coupling scratch (nb_hybrid semantics)
        for (auto &sub : sn) std::fill(sub.c.begin(), sub.c.end(), 0.f);
        for (auto &p : pr) p.apply(sn[p.src_sn], sn[p.tgt_sn], (int)t_abs);
        for (size_t s = 0; s < n_sn; s++) {
          auto &sub = sn[s];
          const float *stim = stim_ps[s];
          counts[s][chunk]++;
          if (stim) {
            // stim layout (n_cvar, n_node, W, nstep)
            const size_t base = ((size_t)step) * (size_t)sub.n_cvar *
                                sub.n_node * sub.n_modes * W;
            for (int cv = 0; cv < sub.n_cvar; cv++)
              for (int j = 0; j < sub.n_node; j++)
                for (int m = 0; m < sub.n_modes; m++) {
                const float *sv = stim + base +
                    (((size_t)cv * sub.n_node + j) * sub.n_modes + m) * W;
                float *dst = sub.c.data() +
                    (((size_t)cv * sub.n_node + j) * sub.n_modes + m) * W;
                for (int i = 0; i < W; i++) dst[i] += sv[i];
                }
          }
          // accumulate ctavg (c for this step, including stimulus)
          {
            const size_t n_cv = (size_t)sub.n_cvar * sub.n_node * sub.n_modes * W;
            const float *cv = sub.c.data();
            float *dst = cacc[s].data() + chunk * n_cv;
            for (size_t k = 0; k < n_cv; k++) dst[k] += cv[k];
          }
          // multi-dt: the subnet integrates (and pushes) only on its own
          // ticks; t_abs is the 0-based iteration counter, so master tick
          // (t_abs + 1) is due when (t_abs + 1) % K == 0.
          if (((t_abs + 1) % sub.K) == 0) {
            const float *nz = nullptr;
            if (const float *nptr = noise_ps[s])
              nz = nptr + ((size_t)step) * (size_t)sub.n_svar * sub.n_node * sub.n_modes * W;
            if (integ_id == 1)
              sub.step_heun(sub.dt, nz);
            else
              sub.step_euler(sub.dt, nz);
            // slot = (subnet's own step count) - 1 in C++ convention:
            // after master tick (t_abs+1) the subnet has executed
            // (t_abs+1)/K steps, and slot s holds state after step s+1.
            sub.push_state((int)((((t_abs + 1) / sub.K) - 1) & (sub.H - 1)));
          }
          // accumulate tavg (voi specs: 0 = state var, 1 = var difference);
          // values summed over modes (matches hybrid observe)
            {
              const int n_voi = (int)voi_specs[s].size();
              for (int v = 0; v < n_voi; v++) {
                const voispec &vs = voi_specs[s][v];
                float *dst = tacc[s].data() + chunk * n_voi * sub.n_node * W +
                             (size_t)v * sub.n_node * W;
                // layout is (n_svar, n_node, n_modes, W): node-major.  Sum
                // values over modes to match hybrid observe.
                const size_t ba = (size_t)vs.a * sub.n_node * sub.n_modes * W;
                const size_t bb = (size_t)vs.b * sub.n_node * sub.n_modes * W;
                for (size_t k = 0; k < (size_t)sub.n_node; k++) {
                  const size_t off = k * (size_t)sub.n_modes * W;
                  for (int m = 0; m < sub.n_modes; m++) {
                    const size_t mo = off + (size_t)m * W;
                    const float *sva = sub.x.data() + ba + mo;
                    const float *svb = sub.x.data() + bb + mo;
                    for (size_t w = 0; w < (size_t)W; w++) {
                      const size_t dk = k * (size_t)W + w;
                      if (vs.kind == 0) dst[dk] += sva[w];
                      else dst[dk] += sva[w] - svb[w];
                    }
                  }
                }
              }
            }
          // kernel-side monitor engines consume this step's observed row
          // (only valid at chunk_size == 1: the per-chunk sums are then the
          // per-step rows, bit-identical to what the Python reference sees)
          if (!mons.empty() && !mons[s].empty()) {
            if (cs != 1)
              throw std::runtime_error(
                  "kernel monitor engines require chunk_size == 1");
            const int n_voi = (int)voi_specs[s].size();
            const size_t ntv = (size_t)n_voi * sn[s].n_node * W;
            const float *rowbase = tacc[s].data() + (size_t)step * ntv;
            for (auto &mon : mons[s])
              mon.step_one(rowbase, n_voi, sn[s].n_node);
          }
        }
        t_abs++;
      }
      } catch (...) {
        pend = std::current_exception();
      }
    }
    if (pend) std::rethrow_exception(pend);

    // final partial chunk: emit mean over the steps actually run
    const int rem = nstep % cs;
    if (nstep > 0 && rem != 0) {
      const size_t chunk = (size_t)(nstep / cs);
      for (size_t s = 0; s < n_sn; s++) {
        // counts[s][chunk] was already incremented per step in the loop
        // (it equals rem); recorded once more to be explicit.
        counts[s][chunk] = rem;
      }
    }

    nb::list outs;
    for (size_t s = 0; s < n_sn; s++) {
      const int n_voi = (int)voi_specs[s].size();
      nb::ndarray<nb::numpy, float> tarr = make_owned_array<float>(
          {(size_t)n_chunks, (size_t)n_voi, (size_t)sn[s].n_node, (size_t)W});
      nb::ndarray<nb::numpy, float> carr = make_owned_array<float>(
          {(size_t)n_chunks, (size_t)sn[s].n_cvar, (size_t)sn[s].n_node,
           (size_t)sn[s].n_modes * W});
      float *td = tarr.data();
      float *cd = carr.data();
      const size_t ntv = (size_t)n_voi * sn[s].n_node * W;
      const size_t ncv = (size_t)sn[s].n_cvar * sn[s].n_node * sn[s].n_modes * W;
      for (size_t chunk = 0; chunk < (size_t)n_chunks; chunk++) {
        const float tinv = 1.f / (float)(counts[s][chunk] ? counts[s][chunk] : 1);
        // ctavg normalization: the accumulated step count, not the fixed
        // chunk size (nb_hybrid divides by tavg_count; parity_audit.md
        // section 6 decision 6).  Identical on full chunks, correct on the
        // partial final chunk.
        const float cinv = tinv;
        for (size_t k = 0; k < ntv; k++) td[chunk * ntv + k] = tacc[s][chunk * ntv + k] * tinv;
        for (size_t k = 0; k < ncv; k++) cd[chunk * ncv + k] = cacc[s][chunk * ncv + k] * cinv;
      }
      outs.append(tarr);
      outs.append(carr);
      // per-subnet monitor engine streams, in arm order:
      // (times float64, data float32 (n, n_out_v, n_node, W)) each
      if (mons.size() == n_sn) {
        for (auto &mon : mons[s]) {
          const size_t n = mon.t_out.size();
          nb::ndarray<nb::numpy, double> mt = make_owned_array<double>({n});
          std::copy(mon.t_out.begin(), mon.t_out.end(),
                    static_cast<double *>(mt.data()));
          nb::ndarray<nb::numpy, float> md = make_owned_array<float>(
              {n, (size_t)mon.n_out_v, (size_t)sn[s].n_node, (size_t)W});
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

// ---- width-erasing wrapper --------------------------------------------------

struct sim {
  int width = 8;
  std::unique_ptr<simw<1>> s1;
  std::unique_ptr<simw<8>> s8;

  explicit sim(int width_ = 8) : width(width_) {}

  template <typename F> void dispatch(F &&f) {
    if (width == 8) {
      if (!s8) throw std::runtime_error("sim not configured");
      f(*s8);
    } else {
      if (!s1) throw std::runtime_error("sim not configured");
      f(*s1);
    }
  }

  void add_subnet(int n_node, int n_svar, int n_parm, int n_cvar,
                  int model_id, int horizon, int n_modes, uint32_t k = 1,
                  float dt = 0.001f) {
    // round the history length up to a power of two >= horizon+2 so that
    // slot masking with (H-1) is valid
    uint32_t H = 1;
    while (H < (uint32_t)(horizon + 2)) H <<= 1;
    if (width == 8) {
      if (!s8) s8 = std::make_unique<simw<8>>();
      subnet<8> s;
      s.alloc(n_node, n_svar, n_parm, n_cvar, model_id, H, n_modes);
      s.K = k;
      s.dt = dt;
      s8->sn.push_back(std::move(s));
      s8->voi_specs.push_back({});
    } else {
      if (!s1) s1 = std::make_unique<simw<1>>();
      subnet<1> s;
      s.alloc(n_node, n_svar, n_parm, n_cvar, model_id, H, n_modes);
      s.K = k;
      s.dt = dt;
      s1->sn.push_back(std::move(s));
      s1->voi_specs.push_back({});
    }
  }

  void add_projection(int src_sn, int tgt_sn, nb::ndarray<nb::numpy, float> w,
                      nb::ndarray<nb::numpy, uint32_t> idx,
                      nb::ndarray<nb::numpy, uint32_t> ptr,
                      nb::ndarray<nb::numpy, uint32_t> del, int cfun_id,
                      nb::ndarray<nb::numpy, int> src_cvars,
                      int tgt_cvar, int n_cfun_parm, float scale,
                      float ts, int tgt_state_cvar,
                      nb::object mode_map_obj) {
    if (width == 8) {
      proj<8> p;
      p.src_sn = src_sn;
      p.tgt_sn = tgt_sn;
      _fill_proj(p, w, idx, ptr, del, cfun_id, src_cvars, tgt_cvar,
                 n_cfun_parm, scale, ts, tgt_state_cvar, mode_map_obj);
      s8->pr.push_back(std::move(p));
    } else {
      proj<1> p;
      p.src_sn = src_sn;
      p.tgt_sn = tgt_sn;
      _fill_proj(p, w, idx, ptr, del, cfun_id, src_cvars, tgt_cvar,
                 n_cfun_parm, scale, ts, tgt_state_cvar, mode_map_obj);
      s1->pr.push_back(std::move(p));
    }
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
      // identity handled by caller-provided sizes; fill lazily in apply is
      // not possible for const apply — fill with 1x1 identity here and
      // resize when the sim is run (see simw::run)
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
      // state layout (n_svar, n_node, n_modes, W)
      auto V = v.view<nb::ndim<4>>();
      if ((int)V.shape(0) != sn.n_svar || (int)V.shape(1) != sn.n_node ||
          (int)V.shape(2) != sn.n_modes || (int)V.shape(3) != W)
        throw std::runtime_error("bad state shape");
      std::memcpy(sn.x.data(), v.data(), sn.x.size() * sizeof(float));
      // prefill history with the initial state
      const size_t rows = (size_t)sn.n_svar * sn.n_node * sn.n_modes;
      for (size_t row = 0; row < rows; row++) {
        const size_t svar = row / ((size_t)sn.n_node * sn.n_modes);
        const size_t rem = row % ((size_t)sn.n_node * sn.n_modes);
        for (uint32_t hh = 0; hh < sn.H; hh++) {
          float *dst = sn.buf.data() +
              ((svar * (size_t)sn.n_node * sn.n_modes + rem) * sn.H + hh) * W;
          const float *srcv = sn.x.data() + row * W;
          for (int i = 0; i < W; i++) dst[i] = srcv[i];
        }
      }
      s.t_abs = 0;
    });
  }

  uint64_t t_abs() const {
    if (s8) return s8->t_abs;
    if (s1) return s1->t_abs;
    return 0;
  }

  nb::ndarray<nb::numpy, float> get_subnet_state(int si) {
    nb::ndarray<nb::numpy, float> out;
    dispatch([&](auto &s) {
      auto &sn = s.sn[si];
      const int W = std::remove_reference_t<decltype(s)>::W_CONST;
      out = make_owned_array<float>({(size_t)sn.n_svar, (size_t)sn.n_node,
                                     (size_t)sn.n_modes, (size_t)W});
      float *dst = out.data();
      std::memcpy(dst, sn.x.data(), sn.x.size() * sizeof(float));
    });
    return out;
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
    // simple state-variable indices
    dispatch([&](auto &s) {
      s.voi_specs[si].clear();
      for (int idx : v) s.voi_specs[si].push_back({0, idx, 0});
    });
  }

  void set_voi_specs(int si, std::vector<int> flat) {
    // flat triplets (kind, a, b)
    dispatch([&](auto &s) {
      s.voi_specs[si].clear();
      for (size_t k = 0; k + 2 < flat.size(); k += 3)
        s.voi_specs[si].push_back({flat[k], flat[k + 1], flat[k + 2]});
    });
  }

  void set_opts(int integ_id, float dt, int tavg_period) {
    dispatch([&](auto &s) {
      s.integ_id = integ_id;
      s.dt = dt;
      s.tavg_period = tavg_period;
    });
  }

  void set_monitors(int si, nb::list atts) {
    dispatch([&](auto &s) { s.set_monitors(si, atts); });
  }

  void monitor_begin_run() {
    dispatch([&](auto &s) { s.monitor_begin_run(); });
  }

  nb::list run(int nstep, int chunk_size, nb::object noise, nb::object stim) {
    nb::list outs;
    dispatch([&](auto &s) { outs = s.run(nstep, chunk_size, noise, stim); });
    return outs;
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
  dfun_mpr<1>(dx, x, 0, 0, 1, 1, cc, p, 6);
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
  dfun_mpr<1>(dx, x, 0, 0, 1, 1, cc, p, 6);
  for (int k = 0; k < 2; k++) xi[k] = x[k] + dt * dx[k];
  xi[0] = xi[0] * (xi[0] > 0.f);
  dfun_mpr<1>(dxi, xi, 0, 0, 1, 1, cc, p, 6);
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
    cph::cph_have_generic = true;
  }, nb::arg("fn"));
  m.def("dbg_mpr_dfun", &cph::dbg_mpr_dfun);
  m.def("dbg_mpr_heun", &cph::dbg_mpr_heun);
  m.def("generic_model_table", &cph::generic_model_table);
  m.doc() = "C++ hybrid simulator core (runtime SIMD kernels, no codegen)";
  nb::class_<cph::sim>(m, "Sim")
      .def(nb::init<int>(), nb::arg("width") = 8)
      .def("add_subnet", &cph::sim::add_subnet, nb::arg("n_node"),
           nb::arg("n_svar"), nb::arg("n_parm"), nb::arg("n_cvar"),
           nb::arg("model_id"), nb::arg("horizon"), nb::arg("n_modes"),
           nb::arg("k") = 1, nb::arg("dt") = 0.001f)
      .def("add_projection", &cph::sim::add_projection, nb::arg("src_sn"),
           nb::arg("tgt_sn"), nb::arg("w"), nb::arg("idx"), nb::arg("ptr"),
           nb::arg("del"), nb::arg("cfun_id"), nb::arg("src_cvars"),
           nb::arg("tgt_cvar"), nb::arg("n_cfun_parm"), nb::arg("scale"),
           nb::arg("ts") = 1.0f, nb::arg("tgt_state_cvar") = 0,
           nb::arg("mode_map") = nb::none())
      .def("set_subnet_params", &cph::sim::set_subnet_params)
      .def("set_subnet_state", &cph::sim::set_subnet_state)
      .def("set_cfun_params", &cph::sim::set_cfun_params)
      .def("set_voi", &cph::sim::set_voi)
      .def("set_voi_specs", &cph::sim::set_voi_specs)

      .def("set_opts", &cph::sim::set_opts)
      .def("set_monitors", &cph::sim::set_monitors, nb::arg("si"),
           nb::arg("atts"))
      .def("monitor_begin_run", &cph::sim::monitor_begin_run)
      .def("get_subnet_state", &cph::sim::get_subnet_state)
      .def("t_abs", &cph::sim::t_abs)
      .def("run", &cph::sim::run, nb::arg("nstep"), nb::arg("chunk_size") = 0,
           nb::arg("noise") = nb::none(), nb::arg("stim") = nb::none());
}
