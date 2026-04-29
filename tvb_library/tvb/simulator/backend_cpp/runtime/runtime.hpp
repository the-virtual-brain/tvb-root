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

  void write_chunk_average(
      SimulationResult& result,
      std::size_t chunk_index,
      std::size_t steps_in_chunk) const {
    for (std::size_t voi = 0; voi < n_voi_; ++voi) {
      for (std::size_t node = 0; node < n_nodes_; ++node) {
        for (std::size_t mode = 0; mode < n_modes_; ++mode) {
          const std::size_t out_idx =
              ((chunk_index * n_voi_ + voi) * n_nodes_ + node) * n_modes_ + mode;
          result.data[out_idx] =
              accum(voi, node, mode) / static_cast<double>(steps_in_chunk);
        }
      }
    }
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

class HistoryBuffer {
 public:
  HistoryBuffer(
      std::size_t capacity,
      std::size_t n_state_vars,
      std::size_t n_nodes,
      std::size_t n_modes)
      : capacity_(std::max<std::size_t>(1, capacity)),
        next_slot_(0),
        filled_(0) {
    snapshots_.reserve(capacity_);
    for (std::size_t i = 0; i < capacity_; ++i) {
      snapshots_.emplace_back(n_state_vars, n_nodes, n_modes);
    }
  }

  void push(const StateBuffer& state) {
    snapshots_[next_slot_] = state;
    next_slot_ = (next_slot_ + 1) % capacity_;
    filled_ = std::min(filled_ + 1, capacity_);
  }

  const StateBuffer& read(std::size_t delay_steps) const {
    if (filled_ == 0) {
      throw std::runtime_error("HistoryBuffer is empty.");
    }
    if (delay_steps >= filled_) {
      throw std::runtime_error("Requested delay exceeds available history.");
    }
    const std::size_t latest_slot = (next_slot_ + capacity_ - 1) % capacity_;
    const std::size_t slot =
        (latest_slot + capacity_ - (delay_steps % capacity_)) % capacity_;
    return snapshots_[slot];
  }

  double read_value(
      std::size_t delay_steps,
      std::size_t svar,
      std::size_t node,
      std::size_t mode) const {
    return read(delay_steps)(svar, node, mode);
  }

  std::size_t capacity() const { return capacity_; }

  std::size_t size() const { return filled_; }

 private:
  std::size_t capacity_;
  std::size_t next_slot_;
  std::size_t filled_;
  std::vector<StateBuffer> snapshots_;
};

inline double delayed_state_value(
    const HistoryBuffer& history,
    std::size_t delay_steps,
    std::size_t svar,
    std::size_t node,
    std::size_t mode) {
  return history.read_value(delay_steps, svar, node, mode);
}

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
inline void heun_step(StateBuffer& state, const HistoryBuffer& history) {
  StateBuffer predictor = state;
  for (std::size_t node = 0; node < Generated::kNumNodes; ++node) {
    std::array<double, Generated::kNumStateVars> dx0{};
    Generated::compute_dfun(state, history, node, dx0);
    predictor(0, node, 0) = state(0, node, 0) + Generated::kDt * dx0[0];
    predictor(1, node, 0) = state(1, node, 0) + Generated::kDt * dx0[1];
    Generated::apply_state_constraints(predictor, node);
  }

  for (std::size_t node = 0; node < Generated::kNumNodes; ++node) {
    std::array<double, Generated::kNumStateVars> dx0{};
    std::array<double, Generated::kNumStateVars> dx1{};
    Generated::compute_dfun(state, history, node, dx0);
    Generated::compute_dfun(predictor, history, node, dx1);
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
  HistoryBuffer history(
      Generated::kSourceHistoryHorizon,
      Generated::kNumStateVars,
      Generated::kNumNodes,
      Generated::kNumModes);
  for (std::size_t i = 0; i < history.capacity(); ++i) {
    history.push(state);
  }
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

  MonitorBuffer monitor(
      Generated::kNumVoi,
      Generated::kNumNodes,
      Generated::kNumModes);
  std::size_t current_chunk = 0;
  std::size_t steps_in_chunk = 0;
  std::size_t chunk_start_step = 1;

  for (std::size_t step = 1; step <= nstep; ++step) {
    heun_step<Generated>(state, history);
    history.push(state);
    for (std::size_t ivoi = 0; ivoi < Generated::kNumVoi; ++ivoi) {
      const std::size_t svar =
          static_cast<std::size_t>(Generated::kVoiIndices[ivoi]);
      for (std::size_t node = 0; node < Generated::kNumNodes; ++node) {
        for (std::size_t mode = 0; mode < Generated::kNumModes; ++mode) {
          monitor.accum(ivoi, node, mode) += state(svar, node, mode);
        }
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

    monitor.write_chunk_average(result, current_chunk, steps_in_chunk);
    monitor.clear_accum();
    ++current_chunk;
    chunk_start_step = step + 1;
    steps_in_chunk = 0;
  }

  return result;
}

}  // namespace tvb::hybrid::runtime
