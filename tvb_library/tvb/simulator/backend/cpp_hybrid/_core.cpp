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

namespace cph {

typedef void (*cph_generic_fn_t)(int, float *, const float *, int, int, int,
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
static constexpr int MOD_ZER2 = 11;   // ZerlautAdaptationSecondOrder

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
    dCei0[i] = ((_TF_e - E)*(_TF_i - I) + Cee*dfe_TF_e + Cei*dfe_TF_i
                + Cei*dfi_TF_e + Cii*dfi_TF_i - 2.f*Cei)/T;
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

template <int W>
CPH_NOINLINE static void dfun_dispatch(int model_id, float *dx, const float *x,
                                 int node, int mode, int n_node, int n_modes,
                                 const float *c, const float *p, int n_parm) {
  if (model_id >= 100) {
    // generic (expression-generated) dfuns: x/c laid out node-major
    // (n_svar, n_node, n_modes, W); works for single- and multi-mode.
    if (cph::cph_have_generic && cph::cph_generic_fn)
      cph::cph_generic_fn(model_id, dx, x, node, mode, n_node, n_modes,
                          c, p, n_parm, W);
    return;
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

template <int W> struct subnet {
  int n_node = 0, n_svar = 2, n_parm = 0, n_cvar = 1, n_modes = 1;
  int model_id = MOD_MPR;
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
  bool has_pre = false;     // ids 3..6
  std::vector<int> src_cvars;  // state vars summed into the weighted input
  std::vector<float> w;      // (nnz,)
  std::vector<uint32_t> idx; // (nnz,)
  std::vector<uint32_t> ptr; // (n_tgt+1,)
  std::vector<uint32_t> del; // (nnz,) delays in steps
  std::vector<float> cfp;    // (n_cfun_parm, W)
  int n_cfun_parm = 2;

  // coupling into target subnet's c[tgt_cvar], reading source history
  void apply(const subnet<W> &src, subnet<W> &tgt, int t) const {
    const uint32_t Hm1 = src.H - 1;
    const int nn = tgt.n_node;
    const int nms = src.n_modes, nmt = tgt.n_modes;
    if ((int)mode_map.size() != nms * nmt) {
      // lazily build identity when sizes changed
      mode_map.assign((size_t)nms * nmt, 0.f);
      for (int m = 0; m < nms && m < nmt; m++) mode_map[m * nmt + m] = 1.f;
    }
    float *out = tgt.c.data() + (size_t)tgt_cvar * nn * nmt * W;
    // fast path: single source mode, no pre transform, one source cvar,
    // matching target — the dominant case (Linear.a * matvec).  Avoids the
    // per-edge v[] copy / mode loops for W==1 and openmps for wider W.
    if (nms == 1 && nmt == 1 && !has_pre && src_cvars.size() == 1) {
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
        // value written at step (t - 1 - delay); slots are (step & H-1)
        const uint32_t slot = ((uint32_t)t - 1u - del[nz]) & Hm1;
        const float wgt = w[nz];
        for (int m = 0; m < nms && m < 8; m++) {
          float v[4 * W];  // up to 4 source cvars
          for (size_t cv = 0; cv < src_cvars.size(); cv++) {
            const float *sbuf = src.buf.data() +
                (size_t)src_cvars[cv] * src.n_node * nms * src.H * W;
            const float *b = sbuf +
                (((size_t)idx[nz] * nms + m) * src.H + slot) * W;
            for (int i = 0; i < W; i++) v[cv * W + i] = b[i];
          }
          if (has_pre) {
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
  uint64_t t_abs = 0;

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

    // per-step noise/stim pointers (lane-major innermost)
    const float *noise = nullptr, *stim = nullptr;
    nb::ndarray<nb::numpy, float> noise_arr, stim_arr;
    if (!noise_obj.is_none()) {
      noise_arr = nb::cast<nb::ndarray<nb::numpy, float>>(noise_obj);
      noise = noise_arr.data();
    }
    if (!stim_obj.is_none()) {
      stim_arr = nb::cast<nb::ndarray<nb::numpy, float>>(stim_obj);
      stim = stim_arr.data();
    }

    std::vector<std::vector<float>> tacc(n_sn), cacc(n_sn);
    std::vector<std::vector<int>> counts(n_sn);
    for (size_t s = 0; s < n_sn; s++) {
      const int n_voi = (int)voi_specs[s].size();
      tacc[s].assign((size_t)n_chunks * n_voi * sn[s].n_node * W, 0.f);
      cacc[s].assign((size_t)n_chunks * sn[s].n_cvar * sn[s].n_node * W, 0.f);
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
          const float *nz = nullptr;
          if (noise)
            nz = noise + ((size_t)step) * (size_t)sub.n_svar * sub.n_node * sub.n_modes * W;
          if (integ_id == 1)
            sub.step_heun(dt, nz);
          else
            sub.step_euler(dt, nz);
          sub.push_state((int)(t_abs & (sub.H - 1)));
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
        const int n_voi = (int)voi_specs[s].size();
        const size_t nv = (size_t)n_voi * sn[s].n_node * W;
        float *dst = tacc[s].data() + chunk * nv;
        (void)dst;
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
        const float cinv = 1.f / (float)cs;
        for (size_t k = 0; k < ntv; k++) td[chunk * ntv + k] = tacc[s][chunk * ntv + k] * tinv;
        for (size_t k = 0; k < ncv; k++) cd[chunk * ncv + k] = cacc[s][chunk * ncv + k] * cinv;
      }
      outs.append(tarr);
      outs.append(carr);
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
                  int model_id, int horizon, int n_modes) {
    // round the history length up to a power of two >= horizon+2 so that
    // slot masking with (H-1) is valid
    uint32_t H = 1;
    while (H < (uint32_t)(horizon + 2)) H <<= 1;
    if (width == 8) {
      if (!s8) s8 = std::make_unique<simw<8>>();
      subnet<8> s;
      s.alloc(n_node, n_svar, n_parm, n_cvar, model_id, H, n_modes);
      s8->sn.push_back(std::move(s));
      s8->voi_specs.push_back({});
    } else {
      if (!s1) s1 = std::make_unique<simw<1>>();
      subnet<1> s;
      s.alloc(n_node, n_svar, n_parm, n_cvar, model_id, H, n_modes);
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
    p.has_pre = (cfun_id >= 3 && cfun_id <= 7);
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

  nb::list run(int nstep, int chunk_size, nb::object noise, nb::object stim) {
    nb::list outs;
    dispatch([&](auto &s) { outs = s.run(nstep, chunk_size, noise, stim); });
    return outs;
  }
};

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
  m.doc() = "C++ hybrid simulator core (runtime SIMD kernels, no codegen)";
  nb::class_<cph::sim>(m, "Sim")
      .def(nb::init<int>(), nb::arg("width") = 8)
      .def("add_subnet", &cph::sim::add_subnet)
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
      .def("get_subnet_state", &cph::sim::get_subnet_state)
      .def("t_abs", &cph::sim::t_abs)
      .def("run", &cph::sim::run, nb::arg("nstep"), nb::arg("chunk_size") = 0,
           nb::arg("noise") = nb::none(), nb::arg("stim") = nb::none());
}
