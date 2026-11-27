#pragma once
// Coupling functions (`cfun_post<W>`, `cfun_pre<W>`) and sparse delayed
// projection (`proj<W>`) for tvb cpp_hybrid.
//
// Pure C++17 header (no nanobind headers). Compiled into the single _core.cpp
// LTO translation unit.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "cph_dfuns.hpp"

namespace cph {

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
  void apply(const subnet<W> &src, subnet<W> &tgt, uint64_t t) const {
    const uint64_t Hm1 = (uint64_t)(src.H - 1u);
    const uint32_t Ksrc = src.K;
    // Decision 3 (parity_audit.md §6): intra-subnet projections never
    // interpolate — they read the exact source step (t1/K) - 1 - delay
    // (the state at the start of the own step being integrated) at every
    // master tick; only inter-subnet reads with k_src > 1 interpolate.
    const bool intra = (src_sn == tgt_sn);
    const bool interp = Ksrc > 1 && !intra;
    const uint64_t t1 = t + 1u;  // 1-based master tick
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
            const uint64_t i0 = (t1 - 1u) / Ksrc - (uint64_t)del[nz];
            // zero-delay edges: i1 = m+1 not pushed; hold x0 (decision 3)
            const float alpha = (del[nz] == 0)
                ? 0.f : (float)((t1 - 1u) % Ksrc) / (float)Ksrc;
            const uint32_t s0 = (uint32_t)((i0 - 1u) & Hm1);
            const uint32_t s1 = (uint32_t)(i0 & Hm1);
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
            const uint32_t slot =
                (uint32_t)((t1 / Ksrc - 2u - (uint64_t)del[nz]) & Hm1);
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
          const uint32_t slot = (uint32_t)((t - 1u - (uint64_t)del[nz]) & Hm1);
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
          const uint64_t ii = (t1 - 1u) / Ksrc - (uint64_t)del[nz];
          s0 = (uint32_t)((ii - 1u) & Hm1);
          s1 = (uint32_t)(ii & Hm1);
          alpha = (del[nz] == 0)
              ? 0.f : (float)((t1 - 1u) % Ksrc) / (float)Ksrc;
        } else {
          // exact single-slot read: the legacy K == 1 read and the
          // multi-dt intra read (decision 3, step (t1/K) - 1 - delay)
          // share one formula in the C++ slot convention (slot s holds
          // step s+1); unwrapped negatives land on IC-prefilled slots
          s0 = s1 = (uint32_t)((t1 / Ksrc - 2u - (uint64_t)del[nz]) & Hm1);
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

}  // namespace cph
