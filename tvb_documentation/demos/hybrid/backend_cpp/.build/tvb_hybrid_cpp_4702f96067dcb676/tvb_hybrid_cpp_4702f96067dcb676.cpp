// Auto-generated C++ translation unit for the TVB hybrid backend.
// 1 subnetwork(s) / 0 intra / 0 inter projection(s)

#include <array>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "runtime/runtime.hpp"

namespace tvb::hybrid::generated {

static constexpr double kDt = 0.10000000000000001;

// ============================================================
// Per-subnet model structs
// ============================================================


// Subnet 0: subnet (MontbrioPazoRoxin)
struct SubnetModel_0 {
  static constexpr double      kDt              = 0.10000000000000001;
  static constexpr std::size_t kNumNodes        = 3;
  static constexpr std::size_t kNumModes        = 1;
  static constexpr std::size_t kNumStateVars    = 2;
  static constexpr std::size_t kNumCouplingVars = 2;
  static constexpr std::size_t kNumVoi          = 2;
  static constexpr std::size_t kSourceHistoryHorizon = 1;
  static constexpr bool        kIsStochastic         = false;

  static constexpr std::array<double, 1> kParam_Delta = { 1 };
  static constexpr std::array<double, 1> kParam_I = { 2 };
  static constexpr std::array<double, 1> kParam_J = { 15 };
  static constexpr std::array<double, 1> kParam_cr = { 1 };
  static constexpr std::array<double, 1> kParam_cv = { 0 };
  static constexpr std::array<double, 1> kParam_eta = { -5 };
  static constexpr std::array<double, 1> kParam_tau = { 1 };

  template <std::size_t N>
  static inline double param_at(const std::array<double, N>& values, std::size_t node) {
    if constexpr (N == 1) { return values[0]; }
    else { return values[node]; }
  }

  // coupling layout: coupling[cvar_slot * kNumNodes + node]
  static inline void compute_dfun(
      const tvb::hybrid::runtime::StateBuffer& state,
      const double* coupling,
      std::size_t node,
      std::array<double, kNumStateVars>& dx) {
    const double r = state(0, node, 0);
    const double V = state(1, node, 0);
    const double tau = param_at(kParam_tau, node);
    const double Delta = param_at(kParam_Delta, node);
    const double eta = param_at(kParam_eta, node);
    const double J = param_at(kParam_J, node);
    const double I = param_at(kParam_I, node);
    const double cr = param_at(kParam_cr, node);
    const double cv = param_at(kParam_cv, node);
    const double Coupling_Term_r = coupling[0 * kNumNodes + node];
    const double Coupling_Term_V = coupling[1 * kNumNodes + node];
    dx[0] = ((1.0 / tau) * ((Delta / (M_PI * tau)) + ((2.0 * V) * r)));
    dx[1] = ((1.0 / tau) * (((((((V * V) - (((((M_PI * M_PI) * tau) * tau) * r) * r)) + eta) + ((J * tau) * r)) + I) + (cr * Coupling_Term_r)) + (cv * Coupling_Term_V)));
  }

  static inline void compute_voi(
      const tvb::hybrid::runtime::StateBuffer& state,
      std::size_t node,
      std::array<double, kNumVoi>& voi) {
    voi[0] = state(0, node, 0);
    voi[1] = state(1, node, 0);
  }

  static inline void apply_state_constraints(
      tvb::hybrid::runtime::StateBuffer& state,
      std::size_t node) {
    state(0, node, 0) = std::max(0.0, state(0, node, 0));
    state(0, node, 0) = std::min(1.7976931348623157e+308, state(0, node, 0));
  }
};

// subnetworks:
//   subnet: model=MontbrioPazoRoxin, integrator=HeunDeterministic, nodes=3, modes=1, state_vars=2, coupling_vars=2
// projections: none

// ============================================================
// Simulation metadata
// ============================================================

using SimulationMetadata = tvb::hybrid::runtime::SimulationMetadata;
using SimulationResult   = tvb::hybrid::runtime::SimulationResult;

inline SimulationMetadata describe() {
  return SimulationMetadata{
      "tvb_hybrid_cpp_4702f96067dcb676",
      "4702f96067dcb676defbad2f62d76fcc65c677340d2ce133898f99104a4dc70f",
      "0.3",
      "visualize_cpp_models_timeseries_MontbrioPazoRoxin",
      kDt,
      1,
      0,
      0,
      1,
  };
}

// ============================================================
// Multi-subnet simulation loop
//
// The loop follows the same phase order as the Numba backend:
//   1. Zero all coupling arrays.
//   2. Accumulate all intra-projection coupling (each subnet reads its own history).
//   3. Accumulate all inter-projection coupling (each reads the *source* subnet's
//      history, which was last written at the END of the previous step — so all
//      reads see consistent t-1 state regardless of subnet ordering).
//   4. Integrate all subnets (Heun step, using the now-complete coupling arrays).
//   5. Push all updated states into their history buffers.
//   6. Accumulate monitor data.
//
// initial_states[i]        : flat (n_state_vars * n_nodes * n_modes) for subnet i
// intra_projections        : CSR projections within a single subnet (source == target)
// inter_projections        : CSR projections between different subnets;
//                            scale already incorporates mode_map[0,0] for single-mode
// ============================================================

inline std::vector<SimulationResult> run_simulation(
    const std::vector<std::vector<double>>& initial_states,
    const std::vector<tvb::hybrid::runtime::ProjectionArrays>& intra_projections,
    const std::vector<tvb::hybrid::runtime::ProjectionArrays>& inter_projections,
    std::size_t nstep,
    std::size_t chunk_size,
    const std::vector<const double*>& noise_ptrs,
    const std::vector<const double*>& stim_ptrs) {
  // noise_ptrs[i]: nullptr for deterministic subnets, else (n_vars, n_nodes, n_modes, nstep).
  // stim_ptrs[i]:  nullptr for no-stimulus subnets, else (n_cvar, n_nodes, nstep) — already
  //               accumulated to target_cvar slots on the Python side.
  // (kNumStateVars, kNumNodes, 1, nstep) C-contiguous float64 array.

  using namespace tvb::hybrid::runtime;

  if (initial_states.size() != 1) {
    throw std::runtime_error(
        "run_simulation: initial_states.size() != num_subnetworks (1)");
  }
  if (intra_projections.size() != 0) {
    throw std::runtime_error(
        "run_simulation: intra_projections.size() != num_intra_projections (0)");
  }
  if (inter_projections.size() != 0) {
    throw std::runtime_error(
        "run_simulation: inter_projections.size() != num_inter_projections (0)");
  }
  if (chunk_size == 0) {
    throw std::runtime_error("chunk_size must be >= 1.");
  }

  // --- Per-subnet state / history / coupling / monitor ---
  StateBuffer state_0(
      SubnetModel_0::kNumStateVars,
      SubnetModel_0::kNumNodes,
      SubnetModel_0::kNumModes,
      initial_states[0]);
  HistoryBuffer history_0(
      SubnetModel_0::kSourceHistoryHorizon,
      SubnetModel_0::kNumStateVars,
      SubnetModel_0::kNumNodes,
      SubnetModel_0::kNumModes);
  for (std::size_t _i = 0; _i < history_0.capacity(); ++_i) {
    history_0.push(state_0);
  }
  std::vector<double> coupling_0(
      SubnetModel_0::kNumCouplingVars * SubnetModel_0::kNumNodes, 0.0);
  MonitorBuffer monitor_0(
      SubnetModel_0::kNumVoi,
      SubnetModel_0::kNumNodes,
      SubnetModel_0::kNumModes);
  MonitorBuffer ctavg_0(
      SubnetModel_0::kNumCouplingVars,
      SubnetModel_0::kNumNodes,
      1);  // coupling has no mode dimension

  // --- Result storage ---
  const std::size_t num_chunks = (nstep + chunk_size - 1) / chunk_size;
  SimulationResult result_0;
  result_0.num_chunks      = num_chunks;
  result_0.num_voi         = SubnetModel_0::kNumVoi;
  result_0.num_nodes       = SubnetModel_0::kNumNodes;
  result_0.num_modes       = SubnetModel_0::kNumModes;
  result_0.num_coupling_vars = SubnetModel_0::kNumCouplingVars;
  result_0.times.resize(num_chunks);
  result_0.data.resize(
      num_chunks *
      SubnetModel_0::kNumVoi *
      SubnetModel_0::kNumNodes *
      SubnetModel_0::kNumModes,
      0.0);
  result_0.ctavg_data.resize(
      num_chunks * SubnetModel_0::kNumCouplingVars * SubnetModel_0::kNumNodes,
      0.0);

  std::size_t current_chunk    = 0;
  std::size_t steps_in_chunk   = 0;
  std::size_t chunk_start_step = 1;

  for (std::size_t step = 1; step <= nstep; ++step) {

    // ---- Phase 1: zero all coupling arrays ----
    std::fill(coupling_0.begin(), coupling_0.end(), 0.0);

    // ---- Phase 2a: intra-projection coupling ----
    // Each subnet reads its own history (state at end of previous step).
    // === Subnet 0: subnet (MontbrioPazoRoxin) ===
    std::fill(coupling_0.begin(), coupling_0.end(), 0.0);


    // ---- Phase 2b: inter-projection coupling ----
    // All reads happen before any history push, so every subnet sees the
    // consistent t-1 state of every source regardless of traversal order.

    // ---- Phase 2c: apply pre-computed stimulus to coupling ----
    // Mirrors Numba: stimulus is added after projections, before ctavg accumulation,
    // so the ctavg monitor reflects the total input (projection + stimulus).
    // stim layout: (n_cvar, n_nodes, nstep), target_cvar already applied Python-side.
    if (stim_ptrs[0] != nullptr) {
      const std::size_t step_0idx_s = step - 1;
      for (std::size_t cv = 0; cv < SubnetModel_0::kNumCouplingVars; ++cv) {
        for (std::size_t node = 0; node < SubnetModel_0::kNumNodes; ++node) {
          coupling_0[cv * SubnetModel_0::kNumNodes + node] +=
              stim_ptrs[0][cv * SubnetModel_0::kNumNodes * nstep +
                               node * nstep + step_0idx_s];
        }
      }
    }

    // ---- Phase 2d: accumulate afferent coupling for AfferentCoupling monitor ----
    // Mirrors Numba template: coupling is sampled after full accumulation,
    // before integration, so ctavg reflects the actual input each node received.
    for (std::size_t cv = 0; cv < SubnetModel_0::kNumCouplingVars; ++cv) {
      for (std::size_t node = 0; node < SubnetModel_0::kNumNodes; ++node) {
        ctavg_0.accum(cv, node, 0) +=
            coupling_0[cv * SubnetModel_0::kNumNodes + node];
      }
    }

    // ---- Phase 3: integrate all subnets ----

    // Subnet 0: subnet (MontbrioPazoRoxin)
    heun_step<SubnetModel_0>(state_0, coupling_0.data());

    // ---- Phase 4: push all updated states into history buffers ----
    // All integrations are complete before any push, so all inter-projection
    // reads in the next step will consistently see this step's final states.
    history_0.push(state_0);

    // ---- Monitor accumulation ----

    for (std::size_t node = 0; node < SubnetModel_0::kNumNodes; ++node) {
      std::array<double, SubnetModel_0::kNumVoi> voi_vals{};
      SubnetModel_0::compute_voi(state_0, node, voi_vals);
      for (std::size_t ivoi = 0; ivoi < SubnetModel_0::kNumVoi; ++ivoi) {
        monitor_0.accum(ivoi, node, 0) += voi_vals[ivoi];
      }
    }

    ++steps_in_chunk;
    const bool close_chunk = (steps_in_chunk == chunk_size) || (step == nstep);
    if (!close_chunk) { continue; }

    // Timestamp: midpoint of [chunk_start_step, chunk_start_step + steps_in_chunk - 1]
    const double mid_step = static_cast<double>(chunk_start_step) +
                            (static_cast<double>(steps_in_chunk) - 1.0) / 2.0;
    result_0.times[current_chunk] = mid_step * kDt;
    monitor_0.write_chunk_average(result_0, current_chunk, steps_in_chunk);
    monitor_0.clear_accum();
    ctavg_0.write_chunk_average_into(result_0.ctavg_data, current_chunk, steps_in_chunk);
    ctavg_0.clear_accum();
    ++current_chunk;
    chunk_start_step = step + 1;
    steps_in_chunk = 0;
  }

  return {result_0};
}

}  // namespace tvb::hybrid::generated
