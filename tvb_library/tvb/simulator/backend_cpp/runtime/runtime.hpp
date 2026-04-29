#pragma once

#include <algorithm>
#include <array>
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
  std::vector<double> data;  // shape: (num_chunks, n_voi, n_nodes, n_modes)
  std::size_t num_chunks;
  std::size_t num_voi;
  std::size_t num_nodes;
  std::size_t num_modes;
};

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
};

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
  for (std::size_t j = 0; j < proj.n_target_nodes; ++j) {
    double wsum = 0.0;
    for (std::ptrdiff_t ptr = proj.weights_indptr[j];
         ptr < proj.weights_indptr[j + 1];
         ++ptr) {
      const std::size_t src_node =
          static_cast<std::size_t>(proj.weights_indices[ptr]);
      const std::size_t delay =
          static_cast<std::size_t>(proj.idelays[ptr]);
      wsum += proj.weights_data[ptr] *
              src_history.read_value(delay, proj.source_svar, src_node, 0);
    }
    coupling[proj.target_cvar_slot * coupling_n_nodes + j] +=
        proj.scale * wsum;
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
  for (std::size_t node = 0; node < Generated::kNumNodes; ++node) {
    std::array<double, Generated::kNumStateVars> dx0{};
    Generated::compute_dfun(state, coupling, node, dx0);
    for (std::size_t svar = 0; svar < Generated::kNumStateVars; ++svar) {
      predictor(svar, node, 0) = state(svar, node, 0) + Generated::kDt * dx0[svar];
    }
    Generated::apply_state_constraints(predictor, node);
  }

  for (std::size_t node = 0; node < Generated::kNumNodes; ++node) {
    std::array<double, Generated::kNumStateVars> dx0{};
    std::array<double, Generated::kNumStateVars> dx1{};
    Generated::compute_dfun(state, coupling, node, dx0);
    Generated::compute_dfun(predictor, coupling, node, dx1);
    for (std::size_t svar = 0; svar < Generated::kNumStateVars; ++svar) {
      state(svar, node, 0) += 0.5 * Generated::kDt * (dx0[svar] + dx1[svar]);
    }
    Generated::apply_state_constraints(state, node);
  }
}

// ---------------------------------------------------------------------------
// run_simulation<Generated>
//
// projections: runtime-provided CSR projection data (may be empty).
// For intra-projections the source history is this subnet's own history.
// ---------------------------------------------------------------------------

template <typename Generated>
inline SimulationResult run_simulation(
    const std::vector<double>& initial_state,
    const std::vector<ProjectionArrays>& projections,
    std::size_t nstep,
    std::size_t chunk_size) {
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

  // Coupling buffer: coupling[cvar_slot * kNumNodes + node]
  std::vector<double> coupling(Generated::kNumCouplingVars * Generated::kNumNodes, 0.0);

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
    // Accumulate coupling from all projections (reads history from previous step).
    std::fill(coupling.begin(), coupling.end(), 0.0);
    for (const auto& proj : projections) {
      accumulate_projection(proj, history, coupling.data(), Generated::kNumNodes);
    }

    heun_step<Generated>(state, coupling.data());
    history.push(state);

    for (std::size_t node = 0; node < Generated::kNumNodes; ++node) {
      std::array<double, Generated::kNumVoi> voi_vals{};
      Generated::compute_voi(state, node, voi_vals);
      for (std::size_t ivoi = 0; ivoi < Generated::kNumVoi; ++ivoi) {
        monitor.accum(ivoi, node, 0) += voi_vals[ivoi];
      }
    }
    ++steps_in_chunk;

    const bool close_chunk = (steps_in_chunk == chunk_size) || (step == nstep);
    if (!close_chunk) {
      continue;
    }

    // Timestamp convention (shared with NbHybridBackend): midpoint of the step
    // range [chunk_start_step, chunk_start_step + steps_in_chunk - 1], with
    // steps counted from 1.  The first chunk midpoint is therefore
    // (1 + (chunk_size-1)/2) * dt = 1.5 * dt for chunk_size == 2.
    //
    // Python's TemporalAverage counts from step 0, placing its first midpoint
    // at (chunk_size-1)/2 * dt = 0.5 * dt for chunk_size == 2.  The constant
    // offset is: python_time == native_time - 0.5 * dt.
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
