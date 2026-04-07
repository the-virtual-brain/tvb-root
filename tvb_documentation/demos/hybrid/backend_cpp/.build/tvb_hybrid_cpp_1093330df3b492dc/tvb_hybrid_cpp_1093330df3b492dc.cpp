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


// Subnet 0: subnet (JansenRit)
struct SubnetModel_0 {
  static constexpr double      kDt              = 0.10000000000000001;
  static constexpr std::size_t kNumNodes        = 3;
  static constexpr std::size_t kNumModes        = 1;
  static constexpr std::size_t kNumStateVars    = 6;
  static constexpr std::size_t kNumCouplingVars = 1;
  static constexpr std::size_t kNumVoi          = 4;
  static constexpr std::size_t kSourceHistoryHorizon = 1;
  static constexpr bool        kIsStochastic         = false;

  static constexpr std::array<double, 1> kParam_A = { 3.25 };
  static constexpr std::array<double, 1> kParam_B = { 22 };
  static constexpr std::array<double, 1> kParam_J = { 135 };
  static constexpr std::array<double, 1> kParam_a = { 0.10000000000000001 };
  static constexpr std::array<double, 1> kParam_a_1 = { 1 };
  static constexpr std::array<double, 1> kParam_a_2 = { 0.80000000000000004 };
  static constexpr std::array<double, 1> kParam_a_3 = { 0.25 };
  static constexpr std::array<double, 1> kParam_a_4 = { 0.25 };
  static constexpr std::array<double, 1> kParam_b = { 0.050000000000000003 };
  static constexpr std::array<double, 1> kParam_mu = { 0.22 };
  static constexpr std::array<double, 1> kParam_nu_max = { 0.0025000000000000001 };
  static constexpr std::array<double, 1> kParam_r = { 0.56000000000000005 };
  static constexpr std::array<double, 1> kParam_v0 = { 5.5199999999999996 };

  template <std::size_t N>
  static inline double param_at(const std::array<double, N>& values, std::size_t node) {
    if constexpr (N == 1) { return values[0]; }
    else { return values[node]; }
  }

  static inline double sigm_jr(double x, double nu_max, double r, double v0) {
    return ((2 * nu_max) / (1 + std::exp((r * (v0 - x)))));
  }

  // coupling layout: coupling[cvar_slot * kNumNodes + node]
  static inline void compute_dfun(
      const tvb::hybrid::runtime::StateBuffer& state,
      const double* coupling,
      std::size_t node,
      std::array<double, kNumStateVars>& dx) {
    const double y0 = state(0, node, 0);
    const double y1 = state(1, node, 0);
    const double y2 = state(2, node, 0);
    const double y3 = state(3, node, 0);
    const double y4 = state(4, node, 0);
    const double y5 = state(5, node, 0);
    const double A = param_at(kParam_A, node);
    const double B = param_at(kParam_B, node);
    const double a = param_at(kParam_a, node);
    const double b = param_at(kParam_b, node);
    const double v0 = param_at(kParam_v0, node);
    const double nu_max = param_at(kParam_nu_max, node);
    const double r = param_at(kParam_r, node);
    const double J = param_at(kParam_J, node);
    const double a_1 = param_at(kParam_a_1, node);
    const double a_2 = param_at(kParam_a_2, node);
    const double a_3 = param_at(kParam_a_3, node);
    const double a_4 = param_at(kParam_a_4, node);
    const double mu = param_at(kParam_mu, node);
    const double Coupling_Term = coupling[0 * kNumNodes + node];
    const double sigm_y1_y2 = sigm_jr((y1 - y2), nu_max, r, v0);
    const double sigm_y0_1 = sigm_jr(((a_1 * J) * y0), nu_max, r, v0);
    const double sigm_y0_3 = sigm_jr(((a_3 * J) * y0), nu_max, r, v0);
    dx[0] = y3;
    dx[1] = y4;
    dx[2] = y5;
    dx[3] = ((((A * a) * sigm_y1_y2) - ((2 * a) * y3)) - (std::pow(a, 2.0) * y0));
    dx[4] = ((((A * a) * ((mu + ((a_2 * J) * sigm_y0_1)) + Coupling_Term)) - ((2 * a) * y4)) - (std::pow(a, 2.0) * y1));
    dx[5] = ((((B * b) * ((a_4 * J) * sigm_y0_3)) - ((2 * b) * y5)) - (std::pow(b, 2.0) * y2));
  }

  static inline void compute_voi(
      const tvb::hybrid::runtime::StateBuffer& state,
      std::size_t node,
      std::array<double, kNumVoi>& voi) {
    voi[0] = state(0, node, 0);
    voi[1] = state(1, node, 0);
    voi[2] = state(2, node, 0);
    voi[3] = state(3, node, 0);
  }

  static inline void apply_state_constraints(
      tvb::hybrid::runtime::StateBuffer& state,
      std::size_t node) {
    (void)state; (void)node;
  }
};

// subnetworks:
//   subnet: model=JansenRit, integrator=HeunDeterministic, nodes=3, modes=1, state_vars=6, coupling_vars=2
// projections: none

// ============================================================
// Simulation metadata
// ============================================================

using SimulationMetadata = tvb::hybrid::runtime::SimulationMetadata;
using SimulationResult   = tvb::hybrid::runtime::SimulationResult;

inline SimulationMetadata describe() {
  return SimulationMetadata{
      "tvb_hybrid_cpp_1093330df3b492dc",
      "1093330df3b492dc1c66f47ce539516a5ea26b2a12e0741cd4103fef36c74313",
      "0.3",
      "visualize_cpp_models_timeseries_JansenRit",
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
    // === Subnet 0: subnet (JansenRit) ===
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

    // Subnet 0: subnet (JansenRit)
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
