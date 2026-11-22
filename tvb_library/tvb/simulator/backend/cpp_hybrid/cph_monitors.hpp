// C++17 all-in-C++ monitor and observation engine for tvb cpp_hybrid.
//
// Pure header-only implementation (#pragma once, namespace cph, NO nanobind
// headers). Compiles cleanly under g++ 13 (-O3 -fopenmp-simd -ffp-contract=off),
// Clang, and Emscripten em++ 4.0.9 (-msimd128 -Wno-unknown-pragmas).
// Instantiated for template <int W> with W = 1, 4, 8, 16.
//
// ============================================================================
// Wave 2 Integration API Summary (for _core.cpp & backend.py)
// ============================================================================
//
// 1. VOI Observation (replaces inline tacc accumulation; works for any chunk_size):
//      struct voispec { int kind; int a; int b; }; // kind 0: x[a]; kind 1: x[a]-x[b]
//      template <int W>
//      void observe_voi_row(const float *x, int n_node, int n_modes,
//                           const std::vector<voispec> &voi_specs,
//                           float *out_row);
//    - Reads subnet state x of shape (n_svar, n_node, n_modes, W) float32.
//    - Writes summed-over-modes observed row (n_voi, n_node, W) into out_row.
//
// 2. Multi-Subnet Global Observation Gather:
//      struct GlobalGatherSubnet {
//        int n_voi, n_node;
//        int voi_offset, node_offset;      // used when node_indices is empty
//        std::vector<int> node_indices;    // non-empty => merged connectome order
//      };
//      template <int W>
//      void gather_global_voi_row(const std::vector<const float *> &src_rows,
//                                 const std::vector<GlobalGatherSubnet> &subnets,
//                                 int total_vois, int total_nodes,
//                                 float *global_row);
//    - Maps merged homogeneous subnets (via node_indices) or unmerged
//      block-diagonal / concatenated subnets (via voi_offset, node_offset)
//      into one contiguous (total_vois, total_nodes, W) buffer for global
//      GlobalAverage / SpatialAverage / Projection / Bold monitors.
//
// 3. Monitor Kinds:
//      enum KMonKind {
//        KMON_RAW = 1,           // Raw / RawVoi
//        KMON_SUBSAMPLE = 2,     // SubSample
//        KMON_BOLD = 3,          // Bold (exact TVB HRF convolution)
//        KMON_TAVG = 4,          // TemporalAverage
//        KMON_CTAVG = 5,         // AfferentCoupling (istep==1) / AfferentCouplingTemporalAverage
//        KMON_GLOBAL_AVG = 6,    // GlobalAverage (node mean + istep temporal avg)
//        KMON_SPATIAL_AVG = 7,   // SpatialAverage (dense/CSR or compact + istep temporal avg)
//        KMON_PROJECTION = 8,    // Projection / EEG / MEG / iEEG (gain @ x - ref_vec + istep avg)
//        KMON_BOLD_BW = 9        // SIMD Balloon-Windkessel ODE [s, f, v, q] across W lanes
//      };
//
// 4. Per-Monitor Engine (template <int W> struct kmon):
//    - Configuration methods:
//        configure_basic(slot, kind, istep, dt, n_src_v, n_space, n_modes=1,
//                        voi={}, cadence_K=1, midpoint_time=false)
//        configure_bold(slot, istep, interim_istep, stock_steps, k1v0,
//                       first_order, dt, n_src_v, n_node, voi, hrf, cadence_K=1)
//        configure_bold_bw(slot, istep, dt, n_src_v, n_node, voi={},
//                          bw_dt=0.0f, bw_params_10=nullptr, cadence_K=1)
//        set_spatial_mean(sm, n_areas, n_nodes)           // dense -> CSR
//        set_spatial_compact(area_idx, scale, n_nodes, n_areas)
//        set_projection(gain, n_sensors, n_nodes, ref_vec=nullptr, ref_len=0)
//    - Run lifecycle:
//        begin_run(tick)               // reanchors complete windows; legacy sets run_start
//        begin_call()                  // clears t_out / d_out
//        reserve_for_steps(nstep)      // pre-reserves t_out / d_out capacity
//        step_one(src, n_src_v, n_node, n_modes=1, t_abs=0)
//          * Skips step when cadence_K > 1 && ((t_abs + 1) % cadence_K) != 0
//          * For KMON_CTAVG: pass sub.c.data(), sub.n_cvar, sub.n_node, sub.n_modes, t_abs
//          * For all others: pass observed row (n_voi, n_node, W), n_voi, n_node, 1, t_abs
//          * Emits timestamps into t_out (double) and samples of shape
//            (n_out_v, n_out_space, n_out_modes * W) into d_out (float)
//    - State serialization (snapshot / resume without Python fallback):
//        KMonState export_state() const
//        void import_state(const KMonState &st)
// ============================================================================

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "cph_balloon_windkessel.hpp"

namespace cph {

// ---- VOI specification & per-step observation ------------------------------

struct voispec {
  int kind = 0;  // 0 = state variable `a`; 1 = difference `a - b`
  int a = 0;
  int b = 0;
};

// Observe `n_voi` variables of interest from subnet state `x` of shape
// (n_svar, n_node, n_modes, W) into `out_row` of shape (n_voi, n_node, W),
// summing across `n_modes`. Overwrites `out_row`.
template <int W>
inline void observe_voi_row(const float *x, int n_node, int n_modes,
                            const voispec *specs, int n_voi, float *out_row) {
  if (n_voi <= 0 || n_node <= 0) return;
  if (n_modes <= 0) {
    std::memset(out_row, 0,
                (size_t)n_voi * (size_t)n_node * (size_t)W * sizeof(float));
    return;
  }
  const size_t stride_sv = (size_t)n_node * (size_t)n_modes * (size_t)W;
  for (int v = 0; v < n_voi; v++) {
    const voispec &vs = specs[v];
    float *dst = out_row + (size_t)v * (size_t)n_node * (size_t)W;
    const float *xa = x + (size_t)vs.a * stride_sv;
    const float *xb = x + (size_t)vs.b * stride_sv;
    if (n_modes == 1) {
      const size_t nw = (size_t)n_node * (size_t)W;
      if (vs.kind == 0) {
        std::memcpy(dst, xa, nw * sizeof(float));
      } else {
#pragma omp simd
        for (size_t idx = 0; idx < nw; idx++) {
          dst[idx] = xa[idx] - xb[idx];
        }
      }
    } else {
      for (int k = 0; k < n_node; k++) {
        const size_t off = (size_t)k * (size_t)n_modes * (size_t)W;
        float *dk = dst + (size_t)k * (size_t)W;
        {
          const float *sva = xa + off;
          const float *svb = xb + off;
          if (vs.kind == 0) {
#pragma omp simd
            for (int w = 0; w < W; w++) dk[w] = sva[w];
          } else {
#pragma omp simd
            for (int w = 0; w < W; w++) dk[w] = sva[w] - svb[w];
          }
        }
        for (int m = 1; m < n_modes; m++) {
          const size_t mo = off + (size_t)m * (size_t)W;
          const float *sva = xa + mo;
          const float *svb = xb + mo;
          if (vs.kind == 0) {
#pragma omp simd
            for (int w = 0; w < W; w++) dk[w] += sva[w];
          } else {
#pragma omp simd
            for (int w = 0; w < W; w++) dk[w] += sva[w] - svb[w];
          }
        }
      }
    }
  }
}

template <int W>
inline void observe_voi_row(const float *x, int n_node, int n_modes,
                            const std::vector<voispec> &voi_specs,
                            float *out_row) {
  observe_voi_row<W>(x, n_node, n_modes, voi_specs.data(),
                     (int)voi_specs.size(), out_row);
}

// ---- Multi-subnet global observation gather --------------------------------

struct GlobalGatherSubnet {
  int n_voi = 0;
  int n_node = 0;
  int voi_offset = 0;             // used when node_indices is empty (or heterogeneous)
  int node_offset = 0;            // used when node_indices is empty
  std::vector<int> node_indices;  // non-empty => merged connectome node mapping
};

// Gather per-subnet observed rows (n_voi, n_node, W) into a single contiguous
// global row (total_vois, total_nodes, W). Zero-fills `global_row` first so
// block-diagonal off-blocks and sparse node_indices gaps stay 0.
template <int W>
inline void gather_global_voi_row(const float *const *src_rows,
                                  const GlobalGatherSubnet *subnets,
                                  int n_subnets, int total_vois,
                                  int total_nodes, float *global_row) {
  if (total_vois <= 0 || total_nodes <= 0) return;
  const size_t total_elems =
      (size_t)total_vois * (size_t)total_nodes * (size_t)W;
  std::memset(global_row, 0, total_elems * sizeof(float));

  for (int s = 0; s < n_subnets; s++) {
    const float *src = src_rows[s];
    const GlobalGatherSubnet &gs = subnets[s];
    if (src == nullptr || gs.n_voi <= 0 || gs.n_node <= 0) continue;
    const bool use_map = !gs.node_indices.empty();
    if (use_map && (int)gs.node_indices.size() != gs.n_node) {
      throw std::runtime_error(
          "gather_global_voi_row: node_indices size mismatch");
    }
    for (int v = 0; v < gs.n_voi; v++) {
      const int gv = gs.voi_offset + v;
      if (gv < 0 || gv >= total_vois) {
        throw std::runtime_error(
            "gather_global_voi_row: global voi index out of range");
      }
      const float *src_v = src + (size_t)v * (size_t)gs.n_node * (size_t)W;
      float *dst_v =
          global_row + (size_t)gv * (size_t)total_nodes * (size_t)W;
      if (!use_map) {
        if (gs.node_offset < 0 || gs.node_offset + gs.n_node > total_nodes) {
          throw std::runtime_error(
              "gather_global_voi_row: global node range out of bounds");
        }
        float *dst_block = dst_v + (size_t)gs.node_offset * (size_t)W;
        std::memcpy(dst_block, src_v,
                    (size_t)gs.n_node * (size_t)W * sizeof(float));
      } else {
        for (int j = 0; j < gs.n_node; j++) {
          const int gn = gs.node_indices[j];
          if (gn < 0 || gn >= total_nodes) {
            throw std::runtime_error(
                "gather_global_voi_row: mapped node index out of range");
          }
          const float *sj = src_v + (size_t)j * (size_t)W;
          float *dj = dst_v + (size_t)gn * (size_t)W;
#pragma omp simd
          for (int w = 0; w < W; w++) dj[w] = sj[w];
        }
      }
    }
  }
}

template <int W>
inline void gather_global_voi_row(const std::vector<const float *> &src_rows,
                                  const std::vector<GlobalGatherSubnet> &subnets,
                                  int total_vois, int total_nodes,
                                  float *global_row) {
  if (src_rows.size() != subnets.size()) {
    throw std::runtime_error(
        "gather_global_voi_row: src_rows/subnets size mismatch");
  }
  gather_global_voi_row<W>(src_rows.data(), subnets.data(),
                           (int)subnets.size(), total_vois, total_nodes,
                           global_row);
}

// ---- Monitor kinds & state serialization -----------------------------------

enum KMonKind {
  KMON_RAW = 1,
  KMON_SUBSAMPLE = 2,
  KMON_BOLD = 3,
  KMON_TAVG = 4,
  KMON_CTAVG = 5,
  KMON_GLOBAL_AVG = 6,
  KMON_SPATIAL_AVG = 7,
  KMON_PROJECTION = 8,
  KMON_BOLD_BW = 9,
  KMON_BOLD_BW_STREAM = 10  // staged global-only stream, not the endpoint monitor
};

struct BwStreamRegion {
  double drive_sum = 0.0;
  double bold_sum = 0.0;
  balloon_windkessel::State ode;
};

struct KMonState {
  int slot = -1;
  int kind = 0;
  int width = 1;
  int n_out_v = 0;
  int n_out_space = 0;
  int n_out_modes = 1;
  uint64_t m_step = 0;
  uint64_t run_start = 0;
  std::vector<double> accum;
  uint32_t accum_count = 0;
  std::vector<float> interim;
  uint32_t interim_count = 0;
  std::vector<float> stock;
  uint32_t stock_pos = 0;
  std::vector<float> bw_state;
  std::vector<BwStreamRegion> bw_stream_regions;
  uint32_t bw_drive_count = 0;
};

// ---- Kernel-side monitor engine --------------------------------------------

template <int W>
struct kmon {
  static constexpr int W_CONST = W;

  int slot = -1;                // monitor list index (state identity)
  int kind = 0;                 // KMonKind
  int istep = 1;                // steps per sample
  uint32_t cadence_K = 1;       // per-subnet Recorder cadence: step when (t_abs+1)%cadence_K==0
  int interim_istep = 1;        // bold: steps per interim-stock write
  int stock_steps = 0;          // bold: outer stock length S
  int n_out_v = 0;              // output VOI count
  int n_out_space = 0;          // spatial output size (n_node, 1, n_areas, n_sensors)
  int n_out_modes = 1;          // 1 for observed monitors; n_modes for KMON_CTAVG
  double k1v0 = 1.0;            // bold: (dot - 1) * k1v0 for FirstOrderVolterra
  bool first_order = false;
  bool complete_window = false; // staged native global window policy
  bool midpoint_time = false;   // force (m_step - 0.5*istep)*dt (e.g. CTAVG tavg / Recorder)
  double dt = 0.1;              // effective monitor step dt (dt0 for global, sn.dt for Recorder)

  std::vector<int> voi;         // indices into source row / sub.c; empty => all
  std::vector<double> hrf;      // bold: (S,) G[::-1]

  // KMON_BOLD persistent state
  std::vector<float> interim;   // (interim_istep, n_out_v, n_node, W)
  uint32_t interim_count = 0;
  std::vector<float> stock;     // (stock_steps, n_out_v, n_node, W)
  uint32_t stock_pos = 0;

  // KMON_BOLD_BW persistent state & parameters
  // bw_params layout: [rtau_s, rtau_f, rtau_o, ra, e0, re0, k1, k2, k3, V0]
  float bw_dt = 0.0f;           // if <= 0, uses (float)dt
  float bw_params[10] = {
      1.0f / 0.65f,                  // rtau_s
      1.0f / 0.41f,                  // rtau_f
      1.0f / 0.98f,                  // rtau_o
      1.0f / 0.32f,                  // ra
      0.4f,                          // e0
      1.0f / 0.4f,                   // re0
      4.3f * 40.3f * 0.4f * 0.04f,   // k1
      0.5f * 25.0f * 0.4f * 0.04f,   // k2
      1.0f - 0.5f,                   // k3
      4.0f                           // V0
  };
  std::vector<float> bw_state;  // (n_out_v, 4, n_node, W): [s, f, v, q]

  static constexpr int bw_drive_ms = 10;
  static constexpr int bw_substeps = 2;
  int bw_drive_steps = 0;
  uint32_t bw_output_endpoints = 0;
  double bw_drive_baseline = 0.0;
  double bw_drive_scale = 0.0;
  voispec bw_source_voi;
  balloon_windkessel::Parameters bw_stream_params;
  std::vector<BwStreamRegion> bw_stream_regions;
  std::vector<BwStreamRegion> bw_stream_next;  // bounded staging for an all-region drive close
  uint32_t bw_drive_count = 0;

  bool bw_stream() const { return kind == KMON_BOLD_BW_STREAM; }
  bool staged_global() const { return complete_window || bw_stream(); }

  // KMON_SPATIAL_AVG weights (compact per-node OR sparse CSR from dense matrix)
  int n_areas = 0;
  int n_spatial_nodes = 0;
  bool spatial_compact = false;
  std::vector<int> spatial_area_idx;
  std::vector<float> spatial_scale;
  std::vector<int> sm_row_ptr;
  std::vector<int> sm_col_idx;
  std::vector<float> sm_val;

  // KMON_PROJECTION weights & optional EEG reference vector
  int n_sensors = 0;
  int n_proj_nodes = 0;
  std::vector<float> gain;      // (n_sensors, n_proj_nodes) row-major
  std::vector<float> ref_vec;   // (n_sensors,) or empty
  bool has_ref_vec = false;

  // Running temporal-average accumulator (TAVG, CTAVG, GLOBAL_AVG, SPATIAL_AVG, PROJECTION)
  std::vector<double> accum;    // (n_out_v, n_out_space, n_out_modes, W)
  uint32_t accum_count = 0;

  // Global & per-run counters
  uint64_t m_step = 0;
  uint64_t run_start = 0;

  // Per-call outputs (cleared at begin_call)
  std::vector<double> t_out;
  std::vector<float> d_out;

  void begin_run(uint64_t tick) {
    if (staged_global()) {
      m_step = tick;
      std::fill(bw_stream_regions.begin(), bw_stream_regions.end(), BwStreamRegion{});
      bw_drive_count = 0;
      std::fill(accum.begin(), accum.end(), 0.0);
      accum_count = 0;
      begin_call();
    }
    run_start = m_step;
  }

  void begin_call() {
    if (bw_stream()) {
      // Do not retain a previous call's BOLD output capacity as working memory.
      std::vector<double>().swap(t_out);
      std::vector<float>().swap(d_out);
    } else {
      t_out.clear();
      d_out.clear();
    }
  }

  void configure_bw_stream(int slot_, int period, double neural_dt, int n_node,
                           int observed_voi, double baseline, double scale,
                           const balloon_windkessel::Parameters &params) {
    // The simulator stores its clock in binary32. Require the correctly rounded
    // representation of a 10 ms divisor with an unambiguous integer tick count.
    if (!std::isfinite(neural_dt) || neural_dt <= 0)
      throw std::runtime_error("global BW stream: invalid neural clock");
    const double ticks = std::round(bw_drive_ms / neural_dt);
    if (ticks < 1 ||
        ticks > std::numeric_limits<int>::max() ||
        (float)(bw_drive_ms / ticks) != (float)neural_dt ||
        (float)(bw_drive_ms / (ticks + 1)) == (float)neural_dt ||
        (ticks > 1 && (float)(bw_drive_ms / (ticks - 1)) == (float)neural_dt))
      throw std::runtime_error("global BW stream: neural clock must divide 10 ms");
    const int drive_steps = (int)ticks;
    if (period <= 0 || period % drive_steps != 0)
      throw std::runtime_error("global BW stream: output period must be whole 10 ms windows");
    const uint64_t endpoints = (uint64_t)(period / drive_steps) * bw_substeps;
    if (endpoints > std::numeric_limits<uint32_t>::max())
      throw std::runtime_error("global BW stream: output endpoint count out of range");
    if (n_node < 1 || (size_t)n_node > bw_stream_regions.max_size() ||
        !std::isfinite(baseline) || !std::isfinite(scale) ||
        !balloon_windkessel::detail::valid_parameters(params))
      throw std::runtime_error("global BW stream: invalid region extent, calibration or parameters");
    slot = slot_;
    kind = KMON_BOLD_BW_STREAM;
    istep = period;
    dt = (double)bw_drive_ms / drive_steps;
    n_out_v = 1;
    n_out_space = n_node;
    voi = {observed_voi};
    bw_drive_steps = drive_steps;
    bw_output_endpoints = (uint32_t)endpoints;
    bw_drive_baseline = baseline;
    bw_drive_scale = scale;
    bw_stream_params = params;
    bw_stream_regions.resize((size_t)n_node);
    bw_stream_next.resize((size_t)n_node);
  }

  void init_default_bw_params() {
    const float tau_s = 0.65f, tau_f = 0.41f, tau_o = 0.98f, alpha = 0.32f;
    const float e0 = 0.4f, eps = 0.5f, nu_0 = 40.3f, r_0 = 25.0f, te = 0.04f;
    bw_params[0] = 1.0f / tau_s;
    bw_params[1] = 1.0f / tau_f;
    bw_params[2] = 1.0f / tau_o;
    bw_params[3] = 1.0f / alpha;
    bw_params[4] = e0;
    bw_params[5] = 1.0f / e0;
    bw_params[6] = 4.3f * nu_0 * e0 * te;
    bw_params[7] = eps * r_0 * e0 * te;
    bw_params[8] = 1.0f - eps;
    bw_params[9] = 4.0f;
  }

  void set_bw_params(const float *params_10) {
    if (params_10 == nullptr) {
      init_default_bw_params();
    } else {
      for (int i = 0; i < 10; i++) bw_params[i] = params_10[i];
    }
  }

  void set_spatial_mean(const float *sm, int n_areas_, int n_nodes_) {
    if (n_areas_ < 0 || n_nodes_ < 0)
      throw std::runtime_error("kmon: negative spatial_mean dimension");
    n_areas = n_areas_;
    n_spatial_nodes = n_nodes_;
    n_out_space = n_areas_;
    spatial_compact = false;
    spatial_area_idx.clear();
    spatial_scale.clear();
    sm_row_ptr.assign((size_t)n_areas_ + 1, 0);
    sm_col_idx.clear();
    sm_val.clear();
    for (int a = 0; a < n_areas_; a++) {
      for (int j = 0; j < n_nodes_; j++) {
        const float w = sm[(size_t)a * (size_t)n_nodes_ + (size_t)j];
        if (w != 0.0f) {
          sm_col_idx.push_back(j);
          sm_val.push_back(w);
        }
      }
      sm_row_ptr[(size_t)a + 1] = (int)sm_col_idx.size();
    }
    resize_accum_if_needed();
  }

  void set_spatial_compact(const int *area_idx, const float *scale,
                           int n_nodes_, int n_areas_) {
    if (n_areas_ < 0 || n_nodes_ < 0)
      throw std::runtime_error("kmon: negative compact spatial dimension");
    n_areas = n_areas_;
    n_spatial_nodes = n_nodes_;
    n_out_space = n_areas_;
    spatial_compact = true;
    sm_row_ptr.clear();
    sm_col_idx.clear();
    sm_val.clear();
    spatial_area_idx.assign(area_idx, area_idx + n_nodes_);
    spatial_scale.assign(scale, scale + n_nodes_);
    resize_accum_if_needed();
  }

  void set_projection(const float *gain_data, int n_sensors_, int n_nodes_,
                      const float *ref_vec_data = nullptr, int ref_len = 0) {
    if (n_sensors_ < 0 || n_nodes_ < 0)
      throw std::runtime_error("kmon: negative projection dimension");
    n_sensors = n_sensors_;
    n_proj_nodes = n_nodes_;
    n_out_space = n_sensors_;
    gain.assign(gain_data,
                gain_data + (size_t)n_sensors_ * (size_t)n_nodes_);
    ref_vec.clear();
    has_ref_vec = false;
    if (ref_vec_data != nullptr && ref_len > 0) {
      if (ref_len != n_sensors_)
        throw std::runtime_error("kmon: ref_vec length != n_sensors");
      ref_vec.assign(ref_vec_data, ref_vec_data + ref_len);
      for (float rv : ref_vec) {
        if (rv != 0.0f) {
          has_ref_vec = true;
          break;
        }
      }
    }
    resize_accum_if_needed();
  }

  // Configure non-BOLD monitors (RAW, SUBSAMPLE, TAVG, CTAVG, GLOBAL_AVG,
  // SPATIAL_AVG, PROJECTION). For SPATIAL_AVG / PROJECTION, call
  // set_spatial_* / set_projection either before or after (accum resizes).
  void configure_basic(int slot_, int kind_, int istep_, double dt_,
                       int n_src_v, int n_space, int n_modes_ = 1,
                       const std::vector<int> &voi_ = {},
                       uint32_t cadence_K_ = 1, bool midpoint_time_ = false) {
    slot = slot_;
    kind = kind_;
    istep = std::max(1, istep_);
    cadence_K = std::max<uint32_t>(1u, cadence_K_);
    dt = dt_;
    midpoint_time = midpoint_time_;
    voi = voi_;
    if (voi.empty() && n_src_v > 0) {
      voi.resize((size_t)n_src_v);
      for (int i = 0; i < n_src_v; i++) voi[(size_t)i] = i;
    }
    n_out_v = voi.empty() ? n_src_v : (int)voi.size();
    n_out_modes = (kind == KMON_CTAVG) ? std::max(1, n_modes_) : 1;

    if (kind == KMON_GLOBAL_AVG) {
      n_out_space = 1;
    } else if (kind == KMON_SPATIAL_AVG) {
      n_out_space = (n_areas > 0) ? n_areas : n_space;
    } else if (kind == KMON_PROJECTION) {
      n_out_space = (n_sensors > 0) ? n_sensors : n_space;
    } else {
      n_out_space = n_space;
    }

    resize_accum_if_needed();
  }

  void configure_bold(int slot_, int istep_, int interim_istep_,
                      int stock_steps_, double k1v0_, bool first_order_,
                      double dt_, int n_src_v, int n_node,
                      const std::vector<int> &voi_,
                      const std::vector<double> &hrf_,
                      uint32_t cadence_K_ = 1) {
    slot = slot_;
    kind = KMON_BOLD;
    istep = std::max(1, istep_);
    interim_istep = std::max(1, interim_istep_);
    stock_steps = std::max(1, stock_steps_);
    k1v0 = k1v0_;
    first_order = first_order_;
    cadence_K = std::max<uint32_t>(1u, cadence_K_);
    dt = dt_;
    voi = voi_;
    if (voi.empty() && n_src_v > 0) {
      voi.resize((size_t)n_src_v);
      for (int i = 0; i < n_src_v; i++) voi[(size_t)i] = i;
    }
    n_out_v = voi.empty() ? n_src_v : (int)voi.size();
    n_out_space = n_node;
    n_out_modes = 1;
    hrf = hrf_;
    const size_t plane = (size_t)n_out_v * (size_t)n_node * (size_t)W;
    interim.assign((size_t)interim_istep * plane, 0.0f);
    stock.assign((size_t)stock_steps * plane, 0.0f);
    interim_count = 0;
    stock_pos = 0;
    accum.clear();
    accum_count = 0;
  }

  void configure_bold_bw(int slot_, int istep_, double dt_, int n_src_v,
                         int n_node, const std::vector<int> &voi_ = {},
                         float bw_dt_ = 0.0f,
                         const float *bw_params_10 = nullptr,
                         uint32_t cadence_K_ = 1) {
    slot = slot_;
    kind = KMON_BOLD_BW;
    istep = std::max(1, istep_);
    cadence_K = std::max<uint32_t>(1u, cadence_K_);
    dt = dt_;
    bw_dt = bw_dt_;
    set_bw_params(bw_params_10);
    voi = voi_;
    if (voi.empty() && n_src_v > 0) {
      voi.resize((size_t)n_src_v);
      for (int i = 0; i < n_src_v; i++) voi[(size_t)i] = i;
    }
    n_out_v = voi.empty() ? n_src_v : (int)voi.size();
    n_out_space = n_node;
    n_out_modes = 1;
    bw_state.resize((size_t)n_out_v * 4u * (size_t)n_node * (size_t)W);
    reset_bw_state(n_node);
    accum.clear();
    accum_count = 0;
  }

  void reset_bw_state(int n_node) {
    const size_t nw = (size_t)n_node * (size_t)W;
    bw_state.resize((size_t)n_out_v * 4u * nw);
    for (int v = 0; v < n_out_v; v++) {
      float *base = bw_state.data() + (size_t)v * 4u * nw;
      std::fill(base + 0u * nw, base + 1u * nw, 0.0f);  // s = 0
      std::fill(base + 1u * nw, base + 4u * nw, 1.0f);  // f = 1, v = 1, q = 1
    }
  }

  // Pre-reserve t_out and d_out capacity for `nstep` master steps to avoid
  // reallocation churn during long runs.
  void reserve_for_steps(int nstep) {
    if (nstep <= 0) return;
    if (bw_stream()) {
      const uint64_t phase = (m_step - run_start) % (uint64_t)istep;
      const size_t rows = (size_t)((phase + (uint64_t)nstep) / (uint64_t)istep);
      if (rows > t_out.max_size() || rows > d_out.max_size() / (size_t)n_out_space)
        throw std::runtime_error("global BW stream: output extent out of range");
      t_out.reserve(rows);
      d_out.reserve(rows * (size_t)n_out_space);
      return;
    }
    const size_t ck = (cadence_K > 1u) ? (size_t)cadence_K : 1u;
    const size_t eff_steps = ((size_t)nstep + ck - 1u) / ck;
    size_t max_samples = 0;
    if (kind == KMON_RAW ||
        (kind == KMON_CTAVG && istep <= 1 && !midpoint_time)) {
      max_samples = eff_steps;
    } else {
      const size_t p = (istep > 1) ? (size_t)istep : 1u;
      max_samples = (eff_steps + p - 1u) / p + 1u;
    }
    t_out.reserve(t_out.size() + max_samples);
    const size_t space = (size_t)std::max(0, n_out_space);
    const size_t modes = (size_t)std::max(1, n_out_modes);
    const size_t ov = (size_t)std::max(0, n_out_v);
    const size_t sample_elems = ov * space * modes * (size_t)W;
    if (sample_elems > 0) {
      d_out.reserve(d_out.size() + max_samples * sample_elems);
    }
  }

  // Advance monitor by one step.
  // - For KMON_CTAVG: `src` is sub.c of shape (n_cvar, n_node, n_modes, W),
  //   `n_src_v = n_cvar`, `n_modes = sub.n_modes`.
  // - For all other kinds: `src` is observed row of shape (n_voi, n_node, W),
  //   `n_src_v = n_voi`, `n_modes = 1`.
  void step_one(const float *src, int n_src_v, int n_node, int n_modes = 1,
                uint64_t t_abs = 0) {
    if (bw_stream()) {
      step_bw_stream(src + (size_t)voi[0] * (size_t)n_node);
      return;
    }
    if (cadence_K > 1u && ((t_abs + 1u) % (uint64_t)cadence_K) != 0u) {
      return;
    }
    m_step++;
    const uint64_t within = m_step - run_start;
    const bool all_voi = voi.empty();
    const int L = all_voi ? n_src_v : (int)voi.size();
    if (n_out_v != L) n_out_v = L;

    // Validate VOI indices up front (outside SIMD loops)
    if (!all_voi) {
      for (int v = 0; v < L; v++) {
        if (voi[(size_t)v] < 0 || voi[(size_t)v] >= n_src_v) {
          throw std::runtime_error(
              "kmon::step_one: voi index out of range for source row");
        }
      }
    }

    switch (kind) {
      case KMON_RAW: {
        n_out_space = n_node;
        n_out_modes = 1;
        t_out.push_back((double)within * (double)(float)dt);
        emit_voi_slice(src, n_node, 1, L, all_voi);
        return;
      }
      case KMON_SUBSAMPLE: {
        n_out_space = n_node;
        n_out_modes = 1;
        if (within % (uint64_t)istep == 0u) {
          t_out.push_back((double)within * (double)(float)dt);
          emit_voi_slice(src, n_node, 1, L, all_voi);
        }
        return;
      }
      case KMON_TAVG: {
        n_out_space = n_node;
        n_out_modes = 1;
        const size_t nw = (size_t)n_node * (size_t)W;
        ensure_accum((size_t)L * nw);
        for (int v = 0; v < L; v++) {
          const int rv = all_voi ? v : voi[(size_t)v];
          const float *sv = src + (size_t)rv * nw;
          double *da = accum.data() + (size_t)v * nw;
#pragma omp simd
          for (size_t k = 0; k < nw; k++) da[k] += (double)sv[k];
        }
        finish_average_step((size_t)L * nw);
        return;
      }
      case KMON_CTAVG: {
        n_out_space = n_node;
        n_out_modes = std::max(1, n_modes);
        if (istep <= 1 && !midpoint_time) {
          t_out.push_back((double)within * (double)(float)dt);
          emit_voi_slice(src, n_node, n_out_modes, L, all_voi);
          return;
        }
        const size_t nmw = (size_t)n_node * (size_t)n_out_modes * (size_t)W;
        ensure_accum((size_t)L * nmw);
        for (int v = 0; v < L; v++) {
          const int rv = all_voi ? v : voi[(size_t)v];
          const float *sv = src + (size_t)rv * nmw;
          double *da = accum.data() + (size_t)v * nmw;
#pragma omp simd
          for (size_t k = 0; k < nmw; k++) da[k] += (double)sv[k];
        }
        accum_count++;
        if (m_step % (uint64_t)istep == 0u) {
          t_out.push_back(((double)m_step - 0.5 * (double)istep) * dt);
          flush_accum_to_d_out((size_t)L * nmw);
        }
        return;
      }
      case KMON_GLOBAL_AVG: {
        n_out_space = 1;
        n_out_modes = 1;
        ensure_accum((size_t)L * (size_t)W);
        const double inv_n = (n_node > 0) ? (1.0 / (double)n_node) : 0.0;
        for (int v = 0; v < L; v++) {
          const int rv = all_voi ? v : voi[(size_t)v];
          const float *sv = src + (size_t)rv * (size_t)n_node * (size_t)W;
          double lane_sum[W];
          for (int w = 0; w < W; w++) lane_sum[w] = 0.0;
          for (int j = 0; j < n_node; j++) {
            const float *sj = sv + (size_t)j * (size_t)W;
#pragma omp simd
            for (int w = 0; w < W; w++) lane_sum[w] += (double)sj[w];
          }
          double *da = accum.data() + (size_t)v * (size_t)W;
#pragma omp simd
          for (int w = 0; w < W; w++) da[w] += lane_sum[w] * inv_n;
        }
        finish_average_step((size_t)L * (size_t)W);
        return;
      }
      case KMON_SPATIAL_AVG: {
        if (n_spatial_nodes > 0 && n_node != n_spatial_nodes) {
          throw std::runtime_error(
              "kmon::step_one: n_node mismatch with spatial_mean");
        }
        n_out_space = n_areas;
        n_out_modes = 1;
        const size_t total_elems = (size_t)L * (size_t)n_areas * (size_t)W;
        ensure_accum(total_elems);
        for (int v = 0; v < L; v++) {
          const int rv = all_voi ? v : voi[(size_t)v];
          const float *sv = src + (size_t)rv * (size_t)n_node * (size_t)W;
          double *acc_v = accum.data() + (size_t)v * (size_t)n_areas * (size_t)W;
          if (spatial_compact) {
            for (int j = 0; j < n_node; j++) {
              const int a = spatial_area_idx[(size_t)j];
              if (a < 0 || a >= n_areas) continue;
              const double sc = (double)spatial_scale[(size_t)j];
              const float *sj = sv + (size_t)j * (size_t)W;
              double *da = acc_v + (size_t)a * (size_t)W;
#pragma omp simd
              for (int w = 0; w < W; w++) da[w] += sc * (double)sj[w];
            }
          } else {
            for (int a = 0; a < n_areas; a++) {
              const int beg = sm_row_ptr[(size_t)a];
              const int end = sm_row_ptr[(size_t)a + 1u];
              double sum_w[W];
              for (int w = 0; w < W; w++) sum_w[w] = 0.0;
              for (int p = beg; p < end; p++) {
                const int j = sm_col_idx[(size_t)p];
                const double sc = (double)sm_val[(size_t)p];
                const float *sj = sv + (size_t)j * (size_t)W;
#pragma omp simd
                for (int w = 0; w < W; w++) sum_w[w] += sc * (double)sj[w];
              }
              double *da = acc_v + (size_t)a * (size_t)W;
#pragma omp simd
              for (int w = 0; w < W; w++) da[w] += sum_w[w];
            }
          }
        }
        finish_average_step(total_elems);
        return;
      }
      case KMON_PROJECTION: {
        if (n_proj_nodes > 0 && n_node != n_proj_nodes) {
          throw std::runtime_error(
              "kmon::step_one: n_node mismatch with projection gain");
        }
        n_out_space = n_sensors;
        n_out_modes = 1;
        const size_t total_elems = (size_t)L * (size_t)n_sensors * (size_t)W;
        ensure_accum(total_elems);
        for (int v = 0; v < L; v++) {
          const int rv = all_voi ? v : voi[(size_t)v];
          const float *sv = src + (size_t)rv * (size_t)n_node * (size_t)W;
          double *acc_v =
              accum.data() + (size_t)v * (size_t)n_sensors * (size_t)W;
          for (int s = 0; s < n_sensors; s++) {
            const float *gs = gain.data() + (size_t)s * (size_t)n_node;
            double sum_w[W];
            for (int w = 0; w < W; w++) sum_w[w] = 0.0;
            for (int j = 0; j < n_node; j++) {
              const float g = gs[j];
              if (g == 0.0f) continue;
              const double gd = (double)g;
              const float *sj = sv + (size_t)j * (size_t)W;
#pragma omp simd
              for (int w = 0; w < W; w++) sum_w[w] += gd * (double)sj[w];
            }
            double *da = acc_v + (size_t)s * (size_t)W;
#pragma omp simd
            for (int w = 0; w < W; w++) da[w] += sum_w[w];
          }
        }
        accum_count++;
        if (m_step % (uint64_t)istep == 0u) {
          t_out.push_back(period_emit_time());
          const double inv_cnt =
              1.0 / (double)(accum_count ? accum_count : 1u);
          const size_t old_sz = d_out.size();
          d_out.resize(old_sz + total_elems);
          float *out_ptr = d_out.data() + old_sz;
          if (!has_ref_vec) {
#pragma omp simd
            for (size_t k = 0; k < total_elems; k++) {
              out_ptr[k] = (float)(accum[k] * inv_cnt);
            }
          } else {
            for (int v = 0; v < L; v++) {
              const double *acc_v =
                  accum.data() + (size_t)v * (size_t)n_sensors * (size_t)W;
              float *out_v =
                  out_ptr + (size_t)v * (size_t)n_sensors * (size_t)W;
              double ref_w[W];
              for (int w = 0; w < W; w++) ref_w[w] = 0.0;
              for (int s = 0; s < n_sensors; s++) {
                const double rvs = (double)ref_vec[(size_t)s];
                if (rvs == 0.0) continue;
                const double *as = acc_v + (size_t)s * (size_t)W;
#pragma omp simd
                for (int w = 0; w < W; w++) ref_w[w] += rvs * (as[w] * inv_cnt);
              }
              for (int s = 0; s < n_sensors; s++) {
                const double *as = acc_v + (size_t)s * (size_t)W;
                float *os = out_v + (size_t)s * (size_t)W;
#pragma omp simd
                for (int w = 0; w < W; w++) {
                  os[w] = (float)(as[w] * inv_cnt - ref_w[w]);
                }
              }
            }
          }
          std::fill(accum.begin(), accum.end(), 0.0);
          accum_count = 0;
        }
        return;
      }
      case KMON_BOLD: {
        n_out_space = n_node;
        n_out_modes = 1;
        const int ii = interim_istep;
        const int S = stock_steps;
        if (ii <= 0 || S <= 0) {
          throw std::runtime_error("kmon::step_one: invalid BOLD stock config");
        }
        if ((int)hrf.size() != S) {
          throw std::runtime_error("kmon::step_one: BOLD hrf size mismatch");
        }
        const size_t plane = (size_t)L * (size_t)n_node * (size_t)W;
        if (interim.size() != (size_t)ii * plane) {
          interim.assign((size_t)ii * plane, 0.0f);
        }
        if (stock.size() != (size_t)S * plane) {
          stock.assign((size_t)S * plane, 0.0f);
        }
        // interim stock write: Python index ((step % ii) - 1) wraps to ii - 1
        {
          const int slot_i =
              (int)((m_step % (uint64_t)ii) + (uint64_t)ii - 1u) % ii;
          interim_count = (uint32_t)(slot_i + 1);
          for (int v = 0; v < L; v++) {
            const int rv = all_voi ? v : voi[(size_t)v];
            const float *sv = src + (size_t)rv * (size_t)n_node * (size_t)W;
            float *dst = interim.data() +
                         ((size_t)slot_i * (size_t)L + (size_t)v) *
                             (size_t)n_node * (size_t)W;
            std::memcpy(dst, sv, (size_t)n_node * (size_t)W * sizeof(float));
          }
        }
        // interim mean -> outer stock at Python index ((step // ii % S) - 1)
        if (m_step % (uint64_t)ii == 0u) {
          const int srow =
              (int)((m_step / (uint64_t)ii) % (uint64_t)S) - 1;
          const size_t sroww = (srow < 0) ? (size_t)(srow + S) : (size_t)srow;
          stock_pos = (uint32_t)sroww;
          const size_t vn_count = (size_t)L * (size_t)n_node;
          float *stock_dst = stock.data() + sroww * plane;
          for (size_t vn = 0; vn < vn_count; vn++) {
            double acc[W];
            for (int w = 0; w < W; w++) acc[w] = 0.0;
            for (int q = 0; q < ii; q++) {
              const float *iq =
                  interim.data() + (size_t)q * plane + vn * (size_t)W;
#pragma omp simd
              for (int w = 0; w < W; w++) acc[w] += (double)iq[w];
            }
            float *sd = stock_dst + vn * (size_t)W;
#pragma omp simd
            for (int w = 0; w < W; w++) sd[w] = (float)(acc[w] / (double)ii);
          }
        }
        // monitor period: rolled hrf dot over outer stock
        if (m_step % (uint64_t)istep == 0u) {
          const int r = (int)((m_step / (uint64_t)ii) % (uint64_t)S) - 1;
          t_out.push_back((double)m_step * dt);
          const size_t vn_count = (size_t)L * (size_t)n_node;
          const size_t old_sz = d_out.size();
          d_out.resize(old_sz + plane);
          float *out_ptr = d_out.data() + old_sz;
          for (size_t vn = 0; vn < vn_count; vn++) {
            double acc[W];
            for (int w = 0; w < W; w++) acc[w] = 0.0;
            for (int q = 0; q < S; q++) {
              int hk = (q - r) % S;
              if (hk < 0) hk += S;
              const double h = hrf[(size_t)hk];
              const float *sq =
                  stock.data() + (size_t)q * plane + vn * (size_t)W;
#pragma omp simd
              for (int w = 0; w < W; w++) acc[w] += h * (double)sq[w];
            }
            float *od = out_ptr + vn * (size_t)W;
            if (first_order) {
#pragma omp simd
              for (int w = 0; w < W; w++)
                od[w] = (float)((acc[w] - 1.0) * k1v0);
            } else {
#pragma omp simd
              for (int w = 0; w < W; w++) od[w] = (float)acc[w];
            }
          }
        }
        return;
      }
      case KMON_BOLD_BW: {
        n_out_space = n_node;
        n_out_modes = 1;
        const size_t nw = (size_t)n_node * (size_t)W;
        if (bw_state.size() != (size_t)L * 4u * nw) {
          reset_bw_state(n_node);
        }
        const float h = (bw_dt > 0.0f) ? bw_dt : (float)dt;
        const float rtau_s = bw_params[0];
        const float rtau_f = bw_params[1];
        const float rtau_o = bw_params[2];
        const float ra     = bw_params[3];
        const float e0     = bw_params[4];
        const float re0    = bw_params[5];
        const float k1     = bw_params[6];
        const float k2     = bw_params[7];
        const float k3     = bw_params[8];
        const float V0     = bw_params[9];
        const float one_m_e0 = 1.0f - e0;

        for (int v = 0; v < L; v++) {
          const int rv = all_voi ? v : voi[(size_t)v];
          const float *sv = src + (size_t)rv * nw;
          float *bbase = bw_state.data() + (size_t)v * 4u * nw;
          float *bs = bbase + 0u * nw;
          float *bf = bbase + 1u * nw;
          float *bv = bbase + 2u * nw;
          float *bq = bbase + 3u * nw;
#pragma omp simd
          for (size_t k = 0; k < nw; k++) {
            const float x_in = sv[k];
            const float s = bs[k];
            const float f = bf[k];
            const float vv = bv[k];
            const float q = bq[k];
            const float v_ra = std::pow(vv, ra);
            const float ds = x_in - rtau_s * s - rtau_f * (f - 1.0f);
            const float df = s;
            const float dv = rtau_o * (f - v_ra);
            const float dq =
                rtau_o * (f * (1.0f - std::pow(one_m_e0, 1.0f / f)) * re0 -
                          v_ra * (q / vv));
            bs[k] = s + h * ds;
            bf[k] = f + h * df;
            bv[k] = vv + h * dv;
            bq[k] = q + h * dq;
          }
        }
        if (m_step % (uint64_t)istep == 0u) {
          t_out.push_back((double)m_step * dt);
          const size_t plane = (size_t)L * nw;
          const size_t old_sz = d_out.size();
          d_out.resize(old_sz + plane);
          float *out_ptr = d_out.data() + old_sz;
          for (int v = 0; v < L; v++) {
            const float *bbase = bw_state.data() + (size_t)v * 4u * nw;
            const float *bv = bbase + 2u * nw;
            const float *bq = bbase + 3u * nw;
            float *od = out_ptr + (size_t)v * nw;
#pragma omp simd
            for (size_t k = 0; k < nw; k++) {
              const float vv = bv[k];
              const float q = bq[k];
              od[k] = V0 * (k1 * (1.0f - q) + k2 * (1.0f - q / vv) +
                            k3 * (1.0f - vv));
            }
          }
        }
        return;
      }
      default:
        throw std::runtime_error("kmon::step_one: unknown monitor kind");
    }
  }

  KMonState export_state() const {
    KMonState st;
    st.slot = slot;
    st.kind = kind;
    st.width = W;
    st.n_out_v = n_out_v;
    st.n_out_space = n_out_space;
    st.n_out_modes = n_out_modes;
    st.m_step = m_step;
    st.run_start = run_start;
    st.accum = accum;
    st.accum_count = accum_count;
    st.interim = interim;
    st.interim_count = interim_count;
    st.stock = stock;
    st.stock_pos = stock_pos;
    st.bw_state = bw_state;
    st.bw_stream_regions = bw_stream_regions;
    st.bw_drive_count = bw_drive_count;
    return st;
  }

  void validate_window_state(const KMonState &st, uint64_t tick) const {
    if (st.slot != slot || st.kind != kind || st.width != W ||
        st.n_out_v != n_out_v || st.n_out_space != n_out_space ||
        st.n_out_modes != n_out_modes || st.accum.size() != accum.size() ||
        st.m_step != tick || st.run_start > st.m_step ||
        st.accum_count != (st.m_step - st.run_start) % (uint64_t)istep ||
        (st.accum_count == 0 && std::any_of(st.accum.begin(), st.accum.end(),
                                          [](double x) { return x != 0.0; })))
      throw std::runtime_error("global complete window: invalid snapshot state or clock");
  }

  void validate_bw_stream_state(const KMonState &st, uint64_t tick) const {
    if (st.slot != slot || st.kind != kind || st.width != W ||
        st.n_out_v != n_out_v || st.n_out_space != n_out_space ||
        st.n_out_modes != n_out_modes || !st.accum.empty() ||
        st.bw_stream_regions.size() != bw_stream_regions.size() ||
        st.m_step != tick || st.run_start > st.m_step)
      throw std::runtime_error("global BW stream: invalid snapshot shape or clock");
    const uint64_t elapsed = st.m_step - st.run_start;
    if (st.bw_drive_count != elapsed % (uint64_t)bw_drive_steps ||
        st.accum_count != (elapsed % (uint64_t)istep) / bw_drive_steps * bw_substeps)
      throw std::runtime_error("global BW stream: invalid snapshot phase");
    for (const auto &r : st.bw_stream_regions) {
      if (!std::isfinite(r.drive_sum) || !std::isfinite(r.bold_sum) ||
          std::abs(r.drive_sum) > (double)st.bw_drive_count * std::numeric_limits<float>::max() ||
          (st.accum_count == 0 && r.bold_sum != 0.0) ||
          !balloon_windkessel::detail::valid_state(r.ode) ||
          (elapsed < (uint64_t)bw_drive_steps &&
           (r.ode.signal != 0 || r.ode.inflow != 1 || r.ode.volume != 1 || r.ode.deoxy != 1)))
        throw std::runtime_error("global BW stream: invalid snapshot regional state");
    }
  }

  void validate_legacy_sibling_state(const KMonState &st) const {
    if (st.width != W || st.n_out_v != n_out_v || st.n_out_space != n_out_space ||
        st.n_out_modes != n_out_modes)
      throw std::runtime_error("global BW stream: incompatible sibling snapshot geometry");
    if (st.accum.size() != accum.size() || st.interim.size() != interim.size() ||
        st.stock.size() != stock.size() || st.bw_state.size() != bw_state.size())
      throw std::runtime_error("global BW stream: incompatible sibling snapshot buffers");
  }

  // Commit without allocating. Pending rows are not checkpoint data.
  void commit_validated_state(KMonState &st) noexcept {
    m_step = st.m_step;
    run_start = st.run_start;
    accum.swap(st.accum);
    accum_count = st.accum_count;
    interim.swap(st.interim);
    interim_count = st.interim_count;
    stock.swap(st.stock);
    stock_pos = st.stock_pos;
    bw_state.swap(st.bw_state);
    bw_stream_regions.swap(st.bw_stream_regions);
    bw_drive_count = st.bw_drive_count;
    begin_call();
  }

  void import_state(const KMonState &st) {
    if (st.width != W)
      throw std::runtime_error("kmon::import_state: width mismatch");
    if (st.kind != kind)
      throw std::runtime_error("kmon::import_state: kind mismatch");
    if (!accum.empty() && st.accum.size() != accum.size())
      throw std::runtime_error("kmon::import_state: accum size mismatch");
    if (!interim.empty() && st.interim.size() != interim.size())
      throw std::runtime_error("kmon::import_state: interim size mismatch");
    if (!stock.empty() && st.stock.size() != stock.size())
      throw std::runtime_error("kmon::import_state: stock size mismatch");
    if (!bw_state.empty() && st.bw_state.size() != bw_state.size())
      throw std::runtime_error("kmon::import_state: bw_state size mismatch");

    n_out_v = st.n_out_v;
    n_out_space = st.n_out_space;
    n_out_modes = st.n_out_modes;
    m_step = st.m_step;
    run_start = st.run_start;
    accum = st.accum;
    accum_count = st.accum_count;
    interim = st.interim;
    interim_count = st.interim_count;
    stock = st.stock;
    stock_pos = st.stock_pos;
    bw_state = st.bw_state;
  }

 private:
  void step_bw_stream(const float *observed) {
    const size_t n = bw_stream_regions.size();
    for (size_t r = 0; r < n; ++r)
      if (!std::isfinite(observed[r]))
        throw std::runtime_error("global BW stream: nonfinite observed drive");
    const uint32_t drive_count = bw_drive_count + 1;
    if (drive_count < (uint32_t)bw_drive_steps) {
      for (size_t r = 0; r < n; ++r) bw_stream_regions[r].drive_sum += observed[r];
      bw_drive_count = drive_count;
      ++m_step;
      return;
    }
    const uint32_t endpoints = accum_count + bw_substeps;
    const bool ready = endpoints == bw_output_endpoints;
    for (size_t r = 0; r < n; ++r) {
      auto &next = bw_stream_next[r];
      next = bw_stream_regions[r];
      const double mean = (next.drive_sum + observed[r]) / drive_count;
      const double drive = (mean - bw_drive_baseline) * bw_drive_scale;
      if (balloon_windkessel::advance_10ms(next.ode, bw_stream_params, drive,
                                          bw_substeps, next.bold_sum) !=
          balloon_windkessel::AdvanceResult::ok)
        throw std::runtime_error("global BW stream: invalid calibrated drive or RK4 domain failure");
      next.drive_sum = 0;
      if (ready && std::abs(next.bold_sum / endpoints) > std::numeric_limits<float>::max())
        throw std::runtime_error("global BW stream: BOLD output exceeds float range");
    }
    // The entire drive close succeeded, including representable output.
    // All rows for this call were reserved before integration.
    ++m_step;
    if (ready) {
      const uint64_t closed = (m_step - run_start) / (uint64_t)bw_drive_steps;
      t_out.push_back((double)run_start * dt +
                      ((double)closed - 0.5 * (istep / bw_drive_steps)) * bw_drive_ms);
      for (auto &r : bw_stream_next) {
        d_out.push_back((float)(r.bold_sum / endpoints));
        r.bold_sum = 0;
      }
    }
    bw_stream_regions.swap(bw_stream_next);
    bw_drive_count = 0;
    accum_count = ready ? 0 : endpoints;
  }

  void resize_accum_if_needed() {
    const bool needs_accum =
        (kind == KMON_TAVG || kind == KMON_GLOBAL_AVG ||
         kind == KMON_SPATIAL_AVG || kind == KMON_PROJECTION ||
         (kind == KMON_CTAVG && (istep > 1 || midpoint_time)));
    if (needs_accum && n_out_v > 0 && n_out_space > 0) {
      accum.assign((size_t)n_out_v * (size_t)n_out_space *
                       (size_t)std::max(1, n_out_modes) * (size_t)W,
                   0.0);
    } else {
      accum.clear();
    }
    accum_count = 0;
  }

  void ensure_accum(size_t n) {
    if (accum.size() != n) accum.assign(n, 0.0);
  }

  void finish_average_step(size_t total_elems) {
    ++accum_count;
    const bool ready = complete_window ? accum_count == (uint32_t)istep
                                      : m_step % (uint64_t)istep == 0u;
    if (!ready) return;
    if (complete_window || kind == KMON_TAVG)
      t_out.push_back(((double)m_step - 0.5 * (double)istep) * dt);
    else
      t_out.push_back(period_emit_time());
    flush_accum_to_d_out(total_elems);
  }

  double period_emit_time() const {
    if (midpoint_time) {
      return ((double)m_step - 0.5 * (double)istep) * dt;
    }
    return ((double)m_step - 0.5 * (double)(istep - 1)) * (double)(float)dt;
  }

  void emit_voi_slice(const float *src, int n_node, int n_modes, int L,
                      bool all_voi) {
    const size_t slice_elems = (size_t)n_node * (size_t)n_modes * (size_t)W;
    const size_t total = (size_t)L * slice_elems;
    const size_t old_sz = d_out.size();
    d_out.resize(old_sz + total);
    float *dst = d_out.data() + old_sz;
    if (all_voi) {
      std::memcpy(dst, src, total * sizeof(float));
    } else {
      for (int v = 0; v < L; v++) {
        const int rv = voi[(size_t)v];
        std::memcpy(dst + (size_t)v * slice_elems,
                    src + (size_t)rv * slice_elems,
                    slice_elems * sizeof(float));
      }
    }
  }

  void flush_accum_to_d_out(size_t total_elems) {
    const double inv_cnt = 1.0 / (double)(accum_count ? accum_count : 1u);
    const size_t old_sz = d_out.size();
    d_out.resize(old_sz + total_elems);
    float *out_ptr = d_out.data() + old_sz;
#pragma omp simd
    for (size_t k = 0; k < total_elems; k++) {
      out_ptr[k] = (float)(accum[k] * inv_cnt);
    }
    std::fill(accum.begin(), accum.end(), 0.0);
    accum_count = 0;
  }
};

}  // namespace cph
