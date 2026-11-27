// Separable & analytic C++17 stimulus engine for TVB cpp_hybrid.
//
// Pure C++17 header: no nanobind / Python headers; compiles cleanly under
// g++ 13, Clang, and Emscripten em++ 4.0.9 (-msimd128), templated over
// SIMD lane width W in {1, 4, 8, 16}.
//
// ============================================================================
// Wave 2 Integration Contract (_core.cpp & backend.py)
// ============================================================================
//
// Include:
//   #include "cph_stim.hpp"
//
// 1. Enum `cph::StimEqKind`:
//      STIM_DISCRETE_1D     = 0  // 1D lookup in `temporal_1d` by master step
//      STIM_LINEAR          = 1  // eq_params: [0]=a, [1]=b
//      STIM_SINUSOID        = 2  // eq_params: [0]=amp, [1]=frequency
//      STIM_COSINE          = 3  // eq_params: [0]=amp, [1]=frequency
//      STIM_PULSETRAIN      = 4  // eq_params: [0]=T, [1]=tau, [2]=amp, [3]=onset
//      STIM_ALPHA           = 5  // eq_params: [0]=onset, [1]=alpha, [2]=beta
//      STIM_GENSIGMOID      = 6  // eq_params: [0]=low, [1]=high, [2]=midpoint, [3]=sigma
//      STIM_GAUSSIAN        = 7  // eq_params: [0]=amp, [1]=sigma, [2]=midpoint, [3]=offset
//      STIM_DOUBLE_GAUSSIAN = 8  // eq_params: [0]=amp_1, [1]=sigma_1, [2]=midpoint_1,
//                                //            [3]=amp_2, [4]=sigma_2, [5]=midpoint_2
//      STIM_SIGMOID         = 9  // eq_params: [0]=amp, [1]=radius, [2]=sigma, [3]=offset
//
//    All analytic equations match `tvb/datatypes/equations.py` and are evaluated
//    in float64 at `double var = t_ms - onset_t0`, then cast to float32 on write.
//    Leave `onset_t0 = 0.0` for standard TVB behavior (where Alpha and
//    PulseTrain use their own `onset` inside `eq_params`).
//
// 2. Struct `cph::stim_spec<W>`:
//      int subnet_idx;              // target subnetwork index
//      std::vector<int> target_cvars; // target coupling variable slots
//      int n_node;                  // target subnet node count
//      std::vector<float> spatial;  // (n_node,) if !lane_varying, else (n_node, W)
//      bool lane_varying = false;   // true when spatial varies across W lanes
//      int eq_kind;                 // StimEqKind
//      double onset_t0 = 0.0;       // optional time-origin shift (ms)
//      float dt0 = 0.001f;          // master step dt (ms)
//      std::array<double, 12> eq_params{};
//      std::vector<float> temporal_1d; // 1D samples for STIM_DISCRETE_1D
//      int64_t step_offset = 0;     // base step index corresponding to temporal_1d[0]
//      float scale = 1.0f;          // global projection_scale
//      std::vector<float> lane_scale; // optional (W,) per-lane multiplier
//
//    Key methods:
//      double evaluate_temporal(double t_ms) const;
//      double evaluate_step(int64_t step_idx_1based, double t_ms) const;
//      void apply_step(float *sub_c, int n_modes,
//                      int64_t step_idx_1based, double t_ms) const;
//
//    `sub_c` layout: (n_cvar, n_node, n_modes, W) float32, accumulating:
//      sub_c[(((cv * n_node + j) * n_modes + m) * W + w)] +=
//          spatial(j, w) * temporal(t) * scale * lane_scale(w)
//
//    Multi-dt / global step convention (matching backend.py & nb_hybrid):
//      At master iteration `t_abs` (0-based), `step_idx_1based = t_abs + 1`,
//      and `t_ms = (double)step_idx_1based * (double)dt0_f64`.
//      For `STIM_DISCRETE_1D`, `idx = step_idx_1based - step_offset` indexes
//      `temporal_1d` (set `step_offset = 0` when `temporal_1d[k]` holds step `k`).
//
// 3. Legacy 5D dense buffer fallback (cleanly separated):
//      cph::apply_legacy_dense_stim_step<W>(sub_c, stim_step_ptr,
//                                           n_cvar, n_node, n_modes);
//      cph::apply_legacy_dense_stim<W>(sub_c, stim_base, step,
//                                      n_cvar, n_node, n_modes);
//    where `stim_base` has layout (nstep, n_cvar, n_node, n_modes, W) float32.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cph {

enum StimEqKind {
  STIM_DISCRETE_1D     = 0,
  STIM_LINEAR          = 1,
  STIM_SINUSOID        = 2,
  STIM_COSINE          = 3,
  STIM_PULSETRAIN      = 4,
  STIM_ALPHA           = 5,
  STIM_GENSIGMOID      = 6,
  STIM_GAUSSIAN        = 7,
  STIM_DOUBLE_GAUSSIAN = 8,
  STIM_SIGMOID         = 9
};

template <int W>
struct stim_spec {
  static_assert(W == 1 || W == 4 || W == 8 || W == 16,
                "stim_spec<W> supports W in {1, 4, 8, 16}");

  int subnet_idx = 0;
  std::vector<int> target_cvars;
  int n_node = 0;

  // Spatial weights: (n_node,) when lane_varying == false,
  // or (n_node, W) row-major when lane_varying == true.
  std::vector<float> spatial;
  bool lane_varying = false;

  int eq_kind = STIM_DISCRETE_1D;
  double onset_t0 = 0.0;
  float dt0 = 0.001f;
  std::array<double, 12> eq_params{};

  // For STIM_DISCRETE_1D: 1D temporal values indexed by (step_idx_1based - step_offset).
  std::vector<float> temporal_1d;
  int64_t step_offset = 0;

  float scale = 1.0f;
  // Optional per-lane scale factor of length W (empty => 1.0f for all lanes).
  std::vector<float> lane_scale;

  // Evaluate analytic temporal equations in float64 at time `t_ms` (ms),
  // matching `tvb/datatypes/equations.py`.
  double evaluate_temporal(double t_ms) const {
    const double var = t_ms - onset_t0;
    switch (eq_kind) {
      case STIM_DISCRETE_1D: {
        if (temporal_1d.empty() || dt0 <= 0.0f) return 0.0;
        const int64_t step_approx =
            static_cast<int64_t>(std::llround(t_ms / static_cast<double>(dt0)));
        const int64_t idx = step_approx - step_offset;
        if (idx < 0 || idx >= static_cast<int64_t>(temporal_1d.size())) {
          return 0.0;
        }
        return static_cast<double>(temporal_1d[static_cast<size_t>(idx)]);
      }
      case STIM_LINEAR:
        // a * var + b
        return eq_params[0] * var + eq_params[1];

      case STIM_SINUSOID:
        // amp * sin(2 * pi * frequency * var)
        return eq_params[0] * std::sin(6.283185307179586 * eq_params[1] * var);

      case STIM_COSINE:
        // amp * cos(2 * pi * frequency * var)
        return eq_params[0] * std::cos(6.283185307179586 * eq_params[1] * var);

      case STIM_PULSETRAIN: {
        // where((var > onset) & (((var - onset) % T) < tau), amp, 0)
        const double T = eq_params[0];
        const double tau = eq_params[1];
        const double amp = eq_params[2];
        const double onset = eq_params[3];
        if (var > onset && T > 0.0) {
          double phase = std::fmod(var - onset, T);
          if (phase < 0.0) phase += T;
          if (phase < tau) return amp;
        }
        return 0.0;
      }

      case STIM_ALPHA: {
        // where((var - onset) > 0,
        //       (alpha * beta) / (beta - alpha) * (exp(-alpha*(var-onset)) - exp(-beta*(var-onset))),
        //       0)
        const double onset = eq_params[0];
        const double alpha = eq_params[1];
        const double beta = eq_params[2];
        const double dt = var - onset;
        if (dt > 0.0) {
          return (alpha * beta) / (beta - alpha) *
                 (std::exp(-alpha * dt) - std::exp(-beta * dt));
        }
        return 0.0;
      }

      case STIM_GENSIGMOID: {
        // low + (high - low) / (1.0 + exp(-1.8137993642342178 * (var - midpoint) / sigma))
        const double low = eq_params[0];
        const double high = eq_params[1];
        const double midpoint = eq_params[2];
        const double sigma = eq_params[3];
        return low + (high - low) /
               (1.0 + std::exp(-1.8137993642342178 * (var - midpoint) / sigma));
      }

      case STIM_GAUSSIAN: {
        // (amp * exp(-((var - midpoint)^2 / (2.0 * sigma^2)))) + offset
        const double amp = eq_params[0];
        const double sigma = eq_params[1];
        const double midpoint = eq_params[2];
        const double offset = eq_params[3];
        const double d = var - midpoint;
        return (amp * std::exp(-(d * d) / (2.0 * sigma * sigma))) + offset;
      }

      case STIM_DOUBLE_GAUSSIAN: {
        // (amp_1 * exp(-((var - midpoint_1)^2 / (2.0 * sigma_1^2))))
        // - (amp_2 * exp(-((var - midpoint_2)^2 / (2.0 * sigma_2^2))))
        const double amp1 = eq_params[0];
        const double sig1 = eq_params[1];
        const double mid1 = eq_params[2];
        const double amp2 = eq_params[3];
        const double sig2 = eq_params[4];
        const double mid2 = eq_params[5];
        const double d1 = var - mid1;
        const double d2 = var - mid2;
        return (amp1 * std::exp(-(d1 * d1) / (2.0 * sig1 * sig1))) -
               (amp2 * std::exp(-(d2 * d2) / (2.0 * sig2 * sig2)));
      }

      case STIM_SIGMOID: {
        // (amp / (1.0 + exp(-1.8137993642342178 * (radius - var) / sigma))) + offset
        const double amp = eq_params[0];
        const double radius = eq_params[1];
        const double sigma = eq_params[2];
        const double offset = eq_params[3];
        return (amp / (1.0 + std::exp(-1.8137993642342178 * (radius - var) / sigma))) +
               offset;
      }

      default:
        return 0.0;
    }
  }

  // Evaluate temporal factor for a given master step (exact index lookup for
  // STIM_DISCRETE_1D, float64 analytic evaluation at t_ms otherwise).
  double evaluate_step(int64_t step_idx_1based, double t_ms) const {
    if (eq_kind == STIM_DISCRETE_1D) {
      const int64_t idx = step_idx_1based - step_offset;
      if (idx < 0 || idx >= static_cast<int64_t>(temporal_1d.size())) {
        return 0.0;
      }
      return static_cast<double>(temporal_1d[static_cast<size_t>(idx)]);
    }
    return evaluate_temporal(t_ms);
  }

  // Accumulate stimulus contribution into target subnet coupling buffer `sub_c`
  // of shape (n_cvar, n_node, n_modes, W).
  void apply_step(float *sub_c, int n_modes, int64_t step_idx_1based,
                  double t_ms) const {
    if (!sub_c || n_node <= 0 || n_modes <= 0 || target_cvars.empty()) return;

    const double tv = evaluate_step(step_idx_1based, t_ms);
    if (tv == 0.0) return;

    const bool has_lane_scale =
        (lane_scale.size() >= static_cast<size_t>(W));
    const double base_t = tv * static_cast<double>(scale);
    if (base_t == 0.0) return;

    std::array<float, W> t_lane{};
    if (has_lane_scale) {
      for (int w = 0; w < W; w++) {
        t_lane[w] = static_cast<float>(base_t * static_cast<double>(lane_scale[w]));
      }
    } else {
      const float tf = static_cast<float>(base_t);
      for (int w = 0; w < W; w++) {
        t_lane[w] = tf;
      }
    }

    if (lane_varying) {
      if (spatial.size() < static_cast<size_t>(n_node) * W) return;
      for (int cv : target_cvars) {
        if (cv < 0) continue;
        float *c_cv = sub_c + static_cast<size_t>(cv) * n_node * n_modes * W;
        for (int j = 0; j < n_node; j++) {
          const float *sp = spatial.data() + static_cast<size_t>(j) * W;
          std::array<float, W> val{};
          #pragma omp simd
          for (int w = 0; w < W; w++) {
            val[w] = sp[w] * t_lane[w];
          }
          for (int m = 0; m < n_modes; m++) {
            float *dst = c_cv + (static_cast<size_t>(j) * n_modes + m) * W;
            #pragma omp simd
            for (int w = 0; w < W; w++) {
              dst[w] += val[w];
            }
          }
        }
      }
    } else {
      if (spatial.size() < static_cast<size_t>(n_node)) return;
      for (int cv : target_cvars) {
        if (cv < 0) continue;
        float *c_cv = sub_c + static_cast<size_t>(cv) * n_node * n_modes * W;
        for (int j = 0; j < n_node; j++) {
          const float sp_j = spatial[static_cast<size_t>(j)];
          if (sp_j == 0.0f) continue;
          std::array<float, W> val{};
          #pragma omp simd
          for (int w = 0; w < W; w++) {
            val[w] = sp_j * t_lane[w];
          }
          for (int m = 0; m < n_modes; m++) {
            float *dst = c_cv + (static_cast<size_t>(j) * n_modes + m) * W;
            #pragma omp simd
            for (int w = 0; w < W; w++) {
              dst[w] += val[w];
            }
          }
        }
      }
    }
  }
};

// Apply all stimulus specs targeting `subnet_idx` in a single pass.
template <int W>
inline void apply_subnet_stims(const std::vector<stim_spec<W>> &stims,
                               int subnet_idx, float *sub_c, int n_modes,
                               int64_t step_idx_1based, double t_ms) {
  for (const auto &st : stims) {
    if (st.subnet_idx == subnet_idx) {
      st.apply_step(sub_c, n_modes, step_idx_1based, t_ms);
    }
  }
}

// ---- Legacy / fallback external 5D dense stimulus buffer --------------------
// Layout of `stim_step_ptr`: (n_cvar, n_node, n_modes, W) float32.
template <int W>
inline void apply_legacy_dense_stim_step(float *sub_c,
                                         const float *stim_step_ptr,
                                         int n_cvar, int n_node, int n_modes) {
  if (!sub_c || !stim_step_ptr || n_cvar <= 0 || n_node <= 0 || n_modes <= 0) {
    return;
  }
  const size_t total =
      static_cast<size_t>(n_cvar) * n_node * n_modes * static_cast<size_t>(W);
  #pragma omp simd
  for (size_t k = 0; k < total; k++) {
    sub_c[k] += stim_step_ptr[k];
  }
}

// Layout of `stim_base`: (nstep, n_cvar, n_node, n_modes, W) float32.
template <int W>
inline void apply_legacy_dense_stim(float *sub_c, const float *stim_base,
                                    int64_t step_0based, int n_cvar, int n_node,
                                    int n_modes) {
  if (!stim_base || step_0based < 0) return;
  const size_t stride =
      static_cast<size_t>(n_cvar) * n_node * n_modes * static_cast<size_t>(W);
  apply_legacy_dense_stim_step<W>(
      sub_c, stim_base + static_cast<size_t>(step_0based) * stride,
      n_cvar, n_node, n_modes);
}

}  // namespace cph
