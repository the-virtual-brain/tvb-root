#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
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
  std::vector<double> data;  // shape: (num_chunks, n_voi, n_nodes, n_modes)
  std::size_t num_chunks;
  std::size_t num_voi;
  std::size_t num_nodes;
  std::size_t num_modes;
};

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

template <typename Generated>
inline void heun_step(StateBuffer& state) {
  StateBuffer predictor = state;
  for (std::size_t node = 0; node < Generated::kNumNodes; ++node) {
    std::array<double, Generated::kNumStateVars> dx0{};
    Generated::compute_dfun(state, node, dx0);
    predictor(0, node, 0) = state(0, node, 0) + Generated::kDt * dx0[0];
    predictor(1, node, 0) = state(1, node, 0) + Generated::kDt * dx0[1];
    Generated::apply_state_constraints(predictor, node);
  }

  for (std::size_t node = 0; node < Generated::kNumNodes; ++node) {
    std::array<double, Generated::kNumStateVars> dx0{};
    std::array<double, Generated::kNumStateVars> dx1{};
    Generated::compute_dfun(state, node, dx0);
    Generated::compute_dfun(predictor, node, dx1);
    state(0, node, 0) += 0.5 * Generated::kDt * (dx0[0] + dx1[0]);
    state(1, node, 0) += 0.5 * Generated::kDt * (dx0[1] + dx1[1]);
    Generated::apply_state_constraints(state, node);
  }
}

template <typename Generated>
inline SimulationResult run_simulation(
    const std::vector<double>& initial_state,
    std::size_t nstep,
    std::size_t chunk_size) {
  if (Generated::kNumSubnetworks != 1 || Generated::kNumInterProjections != 0 ||
      Generated::kNumIntraProjections != 0) {
    throw std::runtime_error(
        "run_simulation currently supports exactly one subnetwork and no projections.");
  }
  if (Generated::kNumModes != 1) {
    throw std::runtime_error(
        "run_simulation currently supports only single-mode subnetworks.");
  }
  if (initial_state.size() !=
      Generated::kNumStateVars * Generated::kNumNodes * Generated::kNumModes) {
    throw std::runtime_error("initial_state size does not match generated spec.");
  }
  if (chunk_size == 0) {
    throw std::runtime_error("chunk_size must be >= 1.");
  }

  StateBuffer state(
      Generated::kNumStateVars,
      Generated::kNumNodes,
      Generated::kNumModes,
      initial_state);
  const std::size_t num_chunks = (nstep + chunk_size - 1) / chunk_size;
  SimulationResult result;
  result.num_chunks = num_chunks;
  result.num_voi = Generated::kNumVoi;
  result.num_nodes = Generated::kNumNodes;
  result.num_modes = Generated::kNumModes;
  result.times.resize(num_chunks);
  result.data.resize(
      num_chunks * Generated::kNumVoi * Generated::kNumNodes * Generated::kNumModes,
      0.0);

  std::vector<double> accum(Generated::kNumVoi * Generated::kNumNodes, 0.0);
  std::size_t current_chunk = 0;
  std::size_t steps_in_chunk = 0;
  std::size_t chunk_start_step = 1;

  for (std::size_t step = 1; step <= nstep; ++step) {
    heun_step<Generated>(state);
    for (std::size_t ivoi = 0; ivoi < Generated::kNumVoi; ++ivoi) {
      const std::size_t svar =
          static_cast<std::size_t>(Generated::kVoiIndices[ivoi]);
      for (std::size_t node = 0; node < Generated::kNumNodes; ++node) {
        accum[ivoi * Generated::kNumNodes + node] += state(svar, node, 0);
      }
    }
    ++steps_in_chunk;

    const bool close_chunk = (steps_in_chunk == chunk_size) || (step == nstep);
    if (!close_chunk) {
      continue;
    }

    const double mid_step = static_cast<double>(chunk_start_step) +
                            (static_cast<double>(steps_in_chunk) - 1.0) / 2.0;
    result.times[current_chunk] = mid_step * Generated::kDt;

    for (std::size_t ivoi = 0; ivoi < Generated::kNumVoi; ++ivoi) {
      for (std::size_t node = 0; node < Generated::kNumNodes; ++node) {
        const std::size_t out_idx =
            ((current_chunk * Generated::kNumVoi + ivoi) * Generated::kNumNodes +
             node) *
            Generated::kNumModes;
        result.data[out_idx] = accum[ivoi * Generated::kNumNodes + node] /
                               static_cast<double>(steps_in_chunk);
      }
    }

    std::fill(accum.begin(), accum.end(), 0.0);
    ++current_chunk;
    chunk_start_step = step + 1;
    steps_in_chunk = 0;
  }

  return result;
}

}  // namespace tvb::hybrid::runtime
