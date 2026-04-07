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


// Subnet 0: subnet (LarterBreakspear)
struct SubnetModel_0 {
  static constexpr double      kDt              = 0.10000000000000001;
  static constexpr std::size_t kNumNodes        = 3;
  static constexpr std::size_t kNumModes        = 1;
  static constexpr std::size_t kNumStateVars    = 3;
  static constexpr std::size_t kNumCouplingVars = 1;
  static constexpr std::size_t kNumVoi          = 1;
  static constexpr std::size_t kSourceHistoryHorizon = 1;
  static constexpr bool        kIsStochastic         = false;

  static constexpr std::array<double, 1> kParam_C = { 0.10000000000000001 };
  static constexpr std::array<double, 1> kParam_Iext = { 0.29999999999999999 };
  static constexpr std::array<double, 1> kParam_QV_max = { 1 };
  static constexpr std::array<double, 1> kParam_QZ_max = { 1 };
  static constexpr std::array<double, 1> kParam_TCa = { -0.01 };
  static constexpr std::array<double, 1> kParam_TK = { 0 };
  static constexpr std::array<double, 1> kParam_TNa = { 0.29999999999999999 };
  static constexpr std::array<double, 1> kParam_VCa = { 1 };
  static constexpr std::array<double, 1> kParam_VK = { -0.69999999999999996 };
  static constexpr std::array<double, 1> kParam_VL = { -0.5 };
  static constexpr std::array<double, 1> kParam_VNa = { 0.53000000000000003 };
  static constexpr std::array<double, 1> kParam_VT = { 0 };
  static constexpr std::array<double, 1> kParam_ZT = { 0 };
  static constexpr std::array<double, 1> kParam_aee = { 0.40000000000000002 };
  static constexpr std::array<double, 1> kParam_aei = { 2 };
  static constexpr std::array<double, 1> kParam_aie = { 2 };
  static constexpr std::array<double, 1> kParam_ane = { 1 };
  static constexpr std::array<double, 1> kParam_ani = { 0.40000000000000002 };
  static constexpr std::array<double, 1> kParam_b = { 0.10000000000000001 };
  static constexpr std::array<double, 1> kParam_d_Ca = { 0.14999999999999999 };
  static constexpr std::array<double, 1> kParam_d_K = { 0.29999999999999999 };
  static constexpr std::array<double, 1> kParam_d_Na = { 0.14999999999999999 };
  static constexpr std::array<double, 1> kParam_d_V = { 0.65000000000000002 };
  static constexpr std::array<double, 1> kParam_d_Z = { 0.69999999999999996 };
  static constexpr std::array<double, 1> kParam_gCa = { 1.1000000000000001 };
  static constexpr std::array<double, 1> kParam_gK = { 2 };
  static constexpr std::array<double, 1> kParam_gL = { 0.5 };
  static constexpr std::array<double, 1> kParam_gNa = { 6.7000000000000002 };
  static constexpr std::array<double, 1> kParam_phi = { 0.69999999999999996 };
  static constexpr std::array<double, 1> kParam_rNMDA = { 0.25 };
  static constexpr std::array<double, 1> kParam_t_scale = { 1 };
  static constexpr std::array<double, 1> kParam_tau_K = { 1 };

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
    const double V = state(0, node, 0);
    const double W = state(1, node, 0);
    const double Z = state(2, node, 0);
    const double gCa = param_at(kParam_gCa, node);
    const double gK = param_at(kParam_gK, node);
    const double gL = param_at(kParam_gL, node);
    const double phi = param_at(kParam_phi, node);
    const double gNa = param_at(kParam_gNa, node);
    const double TK = param_at(kParam_TK, node);
    const double TCa = param_at(kParam_TCa, node);
    const double TNa = param_at(kParam_TNa, node);
    const double VCa = param_at(kParam_VCa, node);
    const double VK = param_at(kParam_VK, node);
    const double VL = param_at(kParam_VL, node);
    const double VNa = param_at(kParam_VNa, node);
    const double d_K = param_at(kParam_d_K, node);
    const double d_Na = param_at(kParam_d_Na, node);
    const double d_Ca = param_at(kParam_d_Ca, node);
    const double tau_K = param_at(kParam_tau_K, node);
    const double aei = param_at(kParam_aei, node);
    const double aie = param_at(kParam_aie, node);
    const double b = param_at(kParam_b, node);
    const double C = param_at(kParam_C, node);
    const double ane = param_at(kParam_ane, node);
    const double ani = param_at(kParam_ani, node);
    const double aee = param_at(kParam_aee, node);
    const double Iext = param_at(kParam_Iext, node);
    const double rNMDA = param_at(kParam_rNMDA, node);
    const double VT = param_at(kParam_VT, node);
    const double d_V = param_at(kParam_d_V, node);
    const double ZT = param_at(kParam_ZT, node);
    const double d_Z = param_at(kParam_d_Z, node);
    const double QV_max = param_at(kParam_QV_max, node);
    const double QZ_max = param_at(kParam_QZ_max, node);
    const double t_scale = param_at(kParam_t_scale, node);
    const double Coupling_Term = coupling[0 * kNumNodes + node];
    const double m_Ca = (0.5 * (1 + std::tanh(((V - TCa) / d_Ca))));
    const double m_Na = (0.5 * (1 + std::tanh(((V - TNa) / d_Na))));
    const double m_K = (0.5 * (1 + std::tanh(((V - TK) / d_K))));
    const double QV = ((0.5 * QV_max) * (1 + std::tanh(((V - VT) / d_V))));
    const double QZ = ((0.5 * QZ_max) * (1 + std::tanh(((Z - ZT) / d_Z))));
    dx[0] = (t_scale * ((((((((-((gCa + ((((1 - C) * rNMDA) * aee) * QV)) + (((C * rNMDA) * aee) * Coupling_Term))) * m_Ca) * (V - VCa)) - ((gK * W) * (V - VK))) - (gL * (V - VL))) - ((((gNa * m_Na) + (((1 - C) * aee) * QV)) + ((C * aee) * Coupling_Term)) * (V - VNa))) - ((aie * Z) * QZ)) + (ane * Iext)));
    dx[1] = (((t_scale * phi) * (m_K - W)) / tau_K);
    dx[2] = ((t_scale * b) * ((ani * Iext) + ((aei * V) * QV)));
  }

  static inline void compute_voi(
      const tvb::hybrid::runtime::StateBuffer& state,
      std::size_t node,
      std::array<double, kNumVoi>& voi) {
    voi[0] = state(0, node, 0);
  }

  static inline void apply_state_constraints(
      tvb::hybrid::runtime::StateBuffer& state,
      std::size_t node) {
    (void)state; (void)node;
  }
};

// subnetworks:
//   subnet: model=LarterBreakspear, integrator=HeunDeterministic, nodes=3, modes=1, state_vars=3, coupling_vars=1
// projections: none

// ============================================================
// Simulation metadata
// ============================================================

using SimulationMetadata = tvb::hybrid::runtime::SimulationMetadata;
using SimulationResult   = tvb::hybrid::runtime::SimulationResult;

inline SimulationMetadata describe() {
  return SimulationMetadata{
      "tvb_hybrid_cpp_aca1869bed5daf04",
      "aca1869bed5daf04999197b845786bfbfacf5d7675235c43aaa428d4477afaba",
      "0.3",
      "visualize_cpp_models_timeseries_LarterBreakspear",
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
    // === Subnet 0: subnet (LarterBreakspear) ===
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

    // Subnet 0: subnet (LarterBreakspear)
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
