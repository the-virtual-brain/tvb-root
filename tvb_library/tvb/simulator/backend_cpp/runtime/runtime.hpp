#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tvb::hybrid::runtime {

struct SimulationMetadata {
  const char* module_name;
  const char* cache_key;
  const char* backend_version;
  const char* user_source_hint;
  double dt;
  std::size_t num_subnetworks;
  std::size_t num_inter_projections;
  std::size_t num_intra_projections;
  std::size_t num_monitors;
};

struct SimulationResult {
  std::vector<double> times;
  std::vector<double> data;        // (num_chunks, n_voi, n_nodes, n_modes)
  std::vector<double> ctavg_data;  // (num_chunks, n_coupling_vars, n_nodes, 1)
  std::size_t num_chunks;
  std::size_t num_voi;
  std::size_t num_nodes;
  std::size_t num_modes;
  std::size_t num_coupling_vars;
};

// ---------------------------------------------------------------------------
// Coupling function type codes — must stay in sync with _CFUN_TYPE_TO_INT in Python
// ---------------------------------------------------------------------------

static constexpr uint8_t kCfunNone         = 0;
static constexpr uint8_t kCfunLinear       = 1;
static constexpr uint8_t kCfunScaling      = 2;
static constexpr uint8_t kCfunSigmoidal    = 3;
static constexpr uint8_t kCfunSigmoidalJr  = 4;  // legacy single-cvar
static constexpr uint8_t kCfunTanh         = 5;
static constexpr uint8_t kCfunPreSigmoidal = 6;  // static threshold single-cvar
static constexpr uint8_t kCfunKuramoto     = 7;

// ---------------------------------------------------------------------------
// ProjectionArrays — lightweight view over CSR projection data owned by Python
// ---------------------------------------------------------------------------

// All pointers point directly into numpy array buffers; no copy is made.
// n_target_nodes must equal weights_indptr.size() - 1.
// Layout of coupling output: coupling[target_cvar_slot * coupling_n_nodes + j]
struct ProjectionArrays {
  const double*   weights_data;
  const int32_t*  weights_indices;  // source node per edge
  const int32_t*  weights_indptr;   // row pointer (target nodes)
  const int32_t*  idelays;          // delay in steps per edge (0 = previous step)
  std::size_t     n_target_nodes;
  std::size_t     source_svar;      // state-variable index to read from source history
  std::size_t     target_cvar_slot; // index into target subnet's coupling array
  double          scale;            // global projection scale
  uint8_t         cfun_type;        // one of kCfun* constants
  float           cfun_params[8];   // layout per cfun_type (see _cfun_params in Python)
};

// Apply per-edge pre-summation coupling to a single source value x_j.
// For cfun types whose pre() is the identity, returns x_j unchanged.
// Parameter layout:
//   kCfunTanh:         [a, midpoint, sigma, b]  →  a*(1+tanh((b*x-midpoint)/sigma))
//   kCfunSigmoidalJr:  [a, e0, r, v0]           →  a*2*e0/(1+exp(r*(v0-x)))
//   kCfunPreSigmoidal: [H, Q, G, P, theta]       →  H*(Q+tanh(G*(P*x-theta)))
inline double cfun_pre_single(uint8_t type, const float* p, double x_j) noexcept {
  switch (type) {
    case kCfunTanh: {
      const double a = p[0], midpoint = p[1], sigma = p[2], b = p[3];
      return a * (1.0 + std::tanh((b * x_j - midpoint) / sigma));
    }
    case kCfunSigmoidalJr: {
      const double a = p[0], e0 = p[1], r = p[2], v0 = p[3];
      return a * 2.0 * e0 / (1.0 + std::exp(r * (v0 - x_j)));
    }
    case kCfunPreSigmoidal: {
      const double H = p[0], Q = p[1], G = p[2], P = p[3], theta = p[4];
      return H * (Q + std::tanh(G * (P * x_j - theta)));
    }
    default:
      return x_j;
  }
}

// Apply post-summation coupling to the already-scaled weighted sum.
// Parameter layout:
//   kCfunLinear:    [a, b]                        →  a*x + b
//   kCfunScaling:   [a]                            →  a*x
//   kCfunKuramoto:  [a]                            →  a*x
//   kCfunSigmoidal: [a, sigma, midpoint, cmin, cmax] → cmin+(cmax-cmin)/(1+exp(-a*(x-midpoint)/sigma))
//   others: identity (pre-only cfuns)
inline double cfun_post(uint8_t type, const float* p, double wsum) noexcept {
  switch (type) {
    case kCfunLinear:
      return double(p[0]) * wsum + double(p[1]);
    case kCfunScaling:
    case kCfunKuramoto:
      return double(p[0]) * wsum;
    case kCfunSigmoidal: {
      const double a = p[0], sigma = p[1], midpoint = p[2], cmin = p[3], cmax = p[4];
      return cmin + (cmax - cmin) / (1.0 + std::exp(-a * (wsum - midpoint) / sigma));
    }
    default:
      return wsum;  // kCfunNone, kCfunTanh, kCfunSigmoidalJr, kCfunPreSigmoidal
  }
}

// Accumulate one CSR projection into a flat coupling buffer.
// coupling layout: coupling[cvar_slot * coupling_n_nodes + target_node]
inline void accumulate_projection(
    const ProjectionArrays& proj,
    const class HistoryBuffer& src_history,
    double* coupling,
    std::size_t coupling_n_nodes);

// ---------------------------------------------------------------------------
// MonitorBuffer
// ---------------------------------------------------------------------------

class MonitorBuffer {
 public:
  MonitorBuffer(std::size_t n_voi, std::size_t n_nodes, std::size_t n_modes)
      : n_voi_(n_voi),
        n_nodes_(n_nodes),
        n_modes_(n_modes),
        accum_(n_voi * n_nodes * n_modes, 0.0) {}

  double& accum(std::size_t voi, std::size_t node, std::size_t mode) {
    return accum_[offset(voi, node, mode)];
  }

  const double& accum(
      std::size_t voi,
      std::size_t node,
      std::size_t mode) const {
    return accum_[offset(voi, node, mode)];
  }

  void clear_accum() { std::fill(accum_.begin(), accum_.end(), 0.0); }

  void write_chunk_average_into(
      std::vector<double>& out,
      std::size_t chunk_index,
      std::size_t steps_in_chunk) const {
    for (std::size_t voi = 0; voi < n_voi_; ++voi) {
      for (std::size_t node = 0; node < n_nodes_; ++node) {
        for (std::size_t mode = 0; mode < n_modes_; ++mode) {
          const std::size_t out_idx =
              ((chunk_index * n_voi_ + voi) * n_nodes_ + node) * n_modes_ + mode;
          out[out_idx] = accum(voi, node, mode) / static_cast<double>(steps_in_chunk);
        }
      }
    }
  }

  void write_chunk_average(
      SimulationResult& result,
      std::size_t chunk_index,
      std::size_t steps_in_chunk) const {
    write_chunk_average_into(result.data, chunk_index, steps_in_chunk);
  }

 private:
  std::size_t offset(
      std::size_t voi,
      std::size_t node,
      std::size_t mode) const {
    if (voi >= n_voi_ || node >= n_nodes_ || mode >= n_modes_) {
      throw std::runtime_error("MonitorBuffer index out of range.");
    }
    return ((voi * n_nodes_) + node) * n_modes_ + mode;
  }

  std::size_t n_voi_;
  std::size_t n_nodes_;
  std::size_t n_modes_;
  std::vector<double> accum_;
};

// ---------------------------------------------------------------------------
// StateBuffer
// ---------------------------------------------------------------------------

class StateBuffer {
 public:
  StateBuffer(
      std::size_t n_state_vars,
      std::size_t n_nodes,
      std::size_t n_modes,
      std::vector<double> values)
      : n_state_vars_(n_state_vars),
        n_nodes_(n_nodes),
        n_modes_(n_modes),
        values_(std::move(values)) {
    const std::size_t expected_size = n_state_vars_ * n_nodes_ * n_modes_;
    if (values_.size() != expected_size) {
      throw std::runtime_error("StateBuffer size does not match declared shape.");
    }
  }

  StateBuffer(
      std::size_t n_state_vars,
      std::size_t n_nodes,
      std::size_t n_modes)
      : StateBuffer(
            n_state_vars,
            n_nodes,
            n_modes,
            std::vector<double>(n_state_vars * n_nodes * n_modes, 0.0)) {}

  double& operator()(std::size_t svar, std::size_t node, std::size_t mode) {
    return values_[offset(svar, node, mode)];
  }

  const double& operator()(
      std::size_t svar,
      std::size_t node,
      std::size_t mode) const {
    return values_[offset(svar, node, mode)];
  }

  std::vector<double>& raw() { return values_; }

  const std::vector<double>& raw() const { return values_; }

  std::size_t size() const { return values_.size(); }

 private:
  std::size_t offset(
      std::size_t svar,
      std::size_t node,
      std::size_t mode) const {
    if (svar >= n_state_vars_ || node >= n_nodes_ || mode >= n_modes_) {
      throw std::runtime_error("StateBuffer index out of range.");
    }
    return ((svar * n_nodes_) + node) * n_modes_ + mode;
  }

  std::size_t n_state_vars_;
  std::size_t n_nodes_;
  std::size_t n_modes_;
  std::vector<double> values_;
};

// ---------------------------------------------------------------------------
// HistoryBuffer — flat ring buffer of simulation frames
//
// Layout: data_[slot * frame_stride + svar * (n_nodes * n_modes) + node * n_modes + mode]
//
// Slots are written in round-robin order.  push() copies one full frame
// contiguously.  read_value() computes a direct index.
// ---------------------------------------------------------------------------

class HistoryBuffer {
 public:
  HistoryBuffer(
      std::size_t capacity,
      std::size_t n_state_vars,
      std::size_t n_nodes,
      std::size_t n_modes)
      : capacity_(std::max<std::size_t>(1, capacity)),
        n_state_vars_(n_state_vars),
        n_nodes_(n_nodes),
        n_modes_(n_modes),
        frame_stride_(n_state_vars * n_nodes * n_modes),
        next_slot_(0),
        filled_(0),
        data_(capacity_ * frame_stride_, 0.0) {}

  void push(const StateBuffer& state) {
    const std::size_t offset = next_slot_ * frame_stride_;
    std::copy(
        state.raw().begin(),
        state.raw().end(),
        data_.begin() + static_cast<std::ptrdiff_t>(offset));
    next_slot_ = (next_slot_ + 1) % capacity_;
    filled_ = std::min(filled_ + 1, capacity_);
  }

  double read_value(
      std::size_t delay_steps,
      std::size_t svar,
      std::size_t node,
      std::size_t mode) const {
    if (filled_ == 0) {
      throw std::runtime_error("HistoryBuffer is empty.");
    }
    if (delay_steps >= filled_) {
      throw std::runtime_error("Requested delay exceeds available history.");
    }
    const std::size_t latest_slot = (next_slot_ + capacity_ - 1) % capacity_;
    const std::size_t slot = (latest_slot + capacity_ - delay_steps) % capacity_;
    return data_[slot * frame_stride_ + svar * (n_nodes_ * n_modes_) + node * n_modes_ + mode];
  }

  std::size_t capacity() const { return capacity_; }
  std::size_t size() const { return filled_; }

 private:
  std::size_t capacity_;
  std::size_t n_state_vars_;
  std::size_t n_nodes_;
  std::size_t n_modes_;
  std::size_t frame_stride_;
  std::size_t next_slot_;
  std::size_t filled_;
  std::vector<double> data_;
};

// ---------------------------------------------------------------------------
// accumulate_projection — defined after HistoryBuffer is complete
// ---------------------------------------------------------------------------

inline void accumulate_projection(
    const ProjectionArrays& proj,
    const HistoryBuffer& src_history,
    double* coupling,
    std::size_t coupling_n_nodes) {
  const uint8_t  type = proj.cfun_type;
  const float*   p    = proj.cfun_params;
  for (std::size_t j = 0; j < proj.n_target_nodes; ++j) {
    double wsum = 0.0;
    for (std::ptrdiff_t ptr = proj.weights_indptr[j];
         ptr < proj.weights_indptr[j + 1];
         ++ptr) {
      const std::size_t src_node =
          static_cast<std::size_t>(proj.weights_indices[ptr]);
      const std::size_t delay =
          static_cast<std::size_t>(proj.idelays[ptr]);
      const double x_j =
          src_history.read_value(delay, proj.source_svar, src_node, 0);
      wsum += proj.weights_data[ptr] * cfun_pre_single(type, p, x_j);
    }
    coupling[proj.target_cvar_slot * coupling_n_nodes + j] +=
        cfun_post(type, p, proj.scale * wsum);
  }
}

// ---------------------------------------------------------------------------
// describe<Generated>
// ---------------------------------------------------------------------------

template <typename Generated>
inline SimulationMetadata describe() {
  return SimulationMetadata{
      Generated::kModuleName,
      Generated::kCacheKey,
      Generated::kBackendVersion,
      Generated::kUserSourceHint,
      Generated::kDt,
      Generated::kNumSubnetworks,
      Generated::kNumInterProjections,
      Generated::kNumIntraProjections,
      Generated::kNumMonitors,
  };
}

// ---------------------------------------------------------------------------
// heun_step<Generated>
//
// coupling: flat array (kNumCouplingVars * kNumNodes) — pre-accumulated by
// accumulate_projection before this step.  coupling[slot * kNumNodes + node]
// gives coupling term `slot` at `node`.
// ---------------------------------------------------------------------------

template <typename Generated>
inline void heun_step(StateBuffer& state, const double* coupling) {
  StateBuffer predictor = state;
  std::array<std::array<double, Generated::kNumStateVars>, Generated::kNumNodes> dx0_cache{};
  for (std::size_t node = 0; node < Generated::kNumNodes; ++node) {
    Generated::compute_dfun(state, coupling, node, dx0_cache[node]);
    for (std::size_t svar = 0; svar < Generated::kNumStateVars; ++svar) {
      predictor(svar, node, 0) =
          state(svar, node, 0) + Generated::kDt * dx0_cache[node][svar];
    }
    Generated::apply_state_constraints(predictor, node);
  }

  for (std::size_t node = 0; node < Generated::kNumNodes; ++node) {
    std::array<double, Generated::kNumStateVars> dx1{};
    Generated::compute_dfun(predictor, coupling, node, dx1);
    for (std::size_t svar = 0; svar < Generated::kNumStateVars; ++svar) {
      state(svar, node, 0) +=
          0.5 * Generated::kDt * (dx0_cache[node][svar] + dx1[svar]);
    }
    Generated::apply_state_constraints(state, node);
  }
}

// ---------------------------------------------------------------------------
// heun_step_stochastic<Generated>
//
// Mirrors TVB HeunStochastic: the same Wiener increment w is added to both
// the predictor and the corrector (additive noise on state transitions).
//
// noise layout: (kNumStateVars, kNumNodes, 1, nstep) C-contiguous float64.
// Access: noise[svar * kNumNodes * nstep + node * nstep + step_0idx]
// ---------------------------------------------------------------------------

template <typename Generated>
inline void heun_step_stochastic(
    StateBuffer& state,
    const double* coupling,
    const double* noise,
    std::size_t step_0idx,
    std::size_t nstep) {
  StateBuffer predictor = state;
  std::array<std::array<double, Generated::kNumStateVars>, Generated::kNumNodes> dx0_cache{};
  for (std::size_t node = 0; node < Generated::kNumNodes; ++node) {
    Generated::compute_dfun(state, coupling, node, dx0_cache[node]);
    for (std::size_t svar = 0; svar < Generated::kNumStateVars; ++svar) {
      const double w =
          noise[svar * Generated::kNumNodes * nstep + node * nstep + step_0idx];
      predictor(svar, node, 0) =
          state(svar, node, 0) + Generated::kDt * dx0_cache[node][svar] + w;
    }
    Generated::apply_state_constraints(predictor, node);
  }

  for (std::size_t node = 0; node < Generated::kNumNodes; ++node) {
    std::array<double, Generated::kNumStateVars> dx1{};
    Generated::compute_dfun(predictor, coupling, node, dx1);
    for (std::size_t svar = 0; svar < Generated::kNumStateVars; ++svar) {
      const double w =
          noise[svar * Generated::kNumNodes * nstep + node * nstep + step_0idx];
      state(svar, node, 0) +=
          0.5 * Generated::kDt * (dx0_cache[node][svar] + dx1[svar]) + w;
    }
    Generated::apply_state_constraints(state, node);
  }
}

// ---------------------------------------------------------------------------
// euler_step<Generated>
// ---------------------------------------------------------------------------

template <typename Generated>
inline void euler_step(StateBuffer& state, const double* coupling) {
  for (std::size_t node = 0; node < Generated::kNumNodes; ++node) {
    std::array<double, Generated::kNumStateVars> dx{};
    Generated::compute_dfun(state, coupling, node, dx);
    for (std::size_t svar = 0; svar < Generated::kNumStateVars; ++svar) {
      state(svar, node, 0) += Generated::kDt * dx[svar];
    }
    Generated::apply_state_constraints(state, node);
  }
}

// ---------------------------------------------------------------------------
// euler_step_stochastic<Generated>
// ---------------------------------------------------------------------------

template <typename Generated>
inline void euler_step_stochastic(
    StateBuffer& state,
    const double* coupling,
    const double* noise,
    std::size_t step_0idx,
    std::size_t nstep) {
  for (std::size_t node = 0; node < Generated::kNumNodes; ++node) {
    std::array<double, Generated::kNumStateVars> dx{};
    Generated::compute_dfun(state, coupling, node, dx);
    for (std::size_t svar = 0; svar < Generated::kNumStateVars; ++svar) {
      const double w =
          noise[svar * Generated::kNumNodes * nstep + node * nstep + step_0idx];
      state(svar, node, 0) += Generated::kDt * dx[svar] + w;
    }
    Generated::apply_state_constraints(state, node);
  }
}

}  // namespace tvb::hybrid::runtime
