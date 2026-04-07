// Auto-generated C++ translation unit for the TVB hybrid backend.
// 2 subnetwork(s) / 0 intra / 1 inter projection(s)

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


// Subnet 0: cortex (JansenRit)
struct SubnetModel_0 {
  static constexpr double      kDt              = 0.10000000000000001;
  static constexpr std::size_t kNumNodes        = 38;
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


// Subnet 1: thalamus (ReducedSetFitzHughNagumo)
struct SubnetModel_1 {
  static constexpr double      kDt              = 0.10000000000000001;
  static constexpr std::size_t kNumNodes        = 38;
  static constexpr std::size_t kNumModes        = 3;
  static constexpr std::size_t kNumStateVars    = 4;
  static constexpr std::size_t kNumCouplingVars = 2;
  static constexpr std::size_t kNumVoi          = 2;
  static constexpr std::size_t kSourceHistoryHorizon = 1;
  static constexpr bool        kIsStochastic         = false;

  static constexpr std::array<double, 1> kParam_K11 = { 0.5 };
  static constexpr std::array<double, 1> kParam_K12 = { 0.14999999999999999 };
  static constexpr std::array<double, 1> kParam_K21 = { 0.14999999999999999 };
  static constexpr std::array<double, 1> kParam_a = { 0.45000000000000001 };
  static constexpr std::array<double, 1> kParam_b = { 0.90000000000000002 };
  static constexpr std::array<double, 1> kParam_mu = { 0 };
  static constexpr std::array<double, 1> kParam_sigma = { 0.34999999999999998 };
  static constexpr std::array<double, 1> kParam_tau = { 3 };
  static constexpr std::array<double, 9> kDerived_Aik = { 0.33281658111902224, 0.18525241779532714, 0.33281658111902407, 0.59845379335385518, 0.33311144470274784, 0.5984537933538584, 0.33281658111902046, 0.18525241779532614, 0.33281658111902224 };
  static constexpr std::array<double, 9> kDerived_Bik = { 0.33281658111902224, 0.59845379335385518, 0.33281658111902046, 0.18525241779532714, 0.33311144470274784, 0.18525241779532614, 0.33281658111902407, 0.5984537933538584, 0.33281658111902224 };
  static constexpr std::array<double, 9> kDerived_Cik = { 0.33281658111902224, 0.18525241779532714, 0.33281658111902407, 0.59845379335385518, 0.33311144470274784, 0.5984537933538584, 0.33281658111902046, 0.18525241779532614, 0.33281658111902224 };
  static constexpr std::array<double, 3> kDerived_e_i = { 1.0283127964183962, 3.3190009791326016, 1.0283127964183838 };
  static constexpr std::array<double, 3> kDerived_f_i = { 1.0283127964183962, 3.3190009791326016, 1.0283127964183838 };
  static constexpr std::array<double, 3> kDerived_IE_i = { -0.6280522659370894, 6.9388939039072284e-18, 0.62805226593709818 };
  static constexpr std::array<double, 3> kDerived_II_i = { -0.6280522659370894, 6.9388939039072284e-18, 0.62805226593709818 };
  static constexpr std::array<double, 3> kDerived_m_i = { 0.44376177872964995, 0.2470067511613051, 0.44376177872965239 };
  static constexpr std::array<double, 3> kDerived_n_i = { 0.44376177872964995, 0.2470067511613051, 0.44376177872965239 };

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
  }

  static inline void compute_voi(
      const tvb::hybrid::runtime::StateBuffer& state,
      std::size_t node,
      std::array<double, kNumVoi>& voi) {
  }

  static inline void apply_state_constraints(
      tvb::hybrid::runtime::StateBuffer& state,
      std::size_t node) {
    (void)state; (void)node;
  }
};

// subnetworks:
//   cortex: model=JansenRit, integrator=HeunDeterministic, nodes=38, modes=1, state_vars=6, coupling_vars=2
//   thalamus: model=ReducedSetFitzHughNagumo, integrator=HeunDeterministic, nodes=38, modes=3, state_vars=4, coupling_vars=2
// projections:
//   cortex_to_thalamus: cortex -> thalamus, cfun=none, mapping=1_to_1, horizon=1, nnz=1444

// ============================================================
// Simulation metadata
// ============================================================

using SimulationMetadata = tvb::hybrid::runtime::SimulationMetadata;
using SimulationResult   = tvb::hybrid::runtime::SimulationResult;

inline SimulationMetadata describe() {
  return SimulationMetadata{
      "tvb_hybrid_cpp_c4c40aac667cf6b1",
      "c4c40aac667cf6b15ed5d16d1c7fb2d6f9e6be63734b0b147724b0e2571cec83",
      "0.3",
      "cpp_hybrid_getting_started",
      kDt,
      2,
      1,
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

  if (initial_states.size() != 2) {
    throw std::runtime_error(
        "run_simulation: initial_states.size() != num_subnetworks (2)");
  }
  if (intra_projections.size() != 0) {
    throw std::runtime_error(
        "run_simulation: intra_projections.size() != num_intra_projections (0)");
  }
  if (inter_projections.size() != 1) {
    throw std::runtime_error(
        "run_simulation: inter_projections.size() != num_inter_projections (1)");
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
  StateBuffer state_1(
      SubnetModel_1::kNumStateVars,
      SubnetModel_1::kNumNodes,
      SubnetModel_1::kNumModes,
      initial_states[1]);
  HistoryBuffer history_1(
      SubnetModel_1::kSourceHistoryHorizon,
      SubnetModel_1::kNumStateVars,
      SubnetModel_1::kNumNodes,
      SubnetModel_1::kNumModes);
  for (std::size_t _i = 0; _i < history_1.capacity(); ++_i) {
    history_1.push(state_1);
  }
  std::vector<double> coupling_1(
      SubnetModel_1::kNumCouplingVars * SubnetModel_1::kNumNodes, 0.0);
  MonitorBuffer monitor_1(
      SubnetModel_1::kNumVoi,
      SubnetModel_1::kNumNodes,
      SubnetModel_1::kNumModes);
  MonitorBuffer ctavg_1(
      SubnetModel_1::kNumCouplingVars,
      SubnetModel_1::kNumNodes,
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
  SimulationResult result_1;
  result_1.num_chunks      = num_chunks;
  result_1.num_voi         = SubnetModel_1::kNumVoi;
  result_1.num_nodes       = SubnetModel_1::kNumNodes;
  result_1.num_modes       = SubnetModel_1::kNumModes;
  result_1.num_coupling_vars = SubnetModel_1::kNumCouplingVars;
  result_1.times.resize(num_chunks);
  result_1.data.resize(
      num_chunks *
      SubnetModel_1::kNumVoi *
      SubnetModel_1::kNumNodes *
      SubnetModel_1::kNumModes,
      0.0);
  result_1.ctavg_data.resize(
      num_chunks * SubnetModel_1::kNumCouplingVars * SubnetModel_1::kNumNodes,
      0.0);

  std::size_t current_chunk    = 0;
  std::size_t steps_in_chunk   = 0;
  std::size_t chunk_start_step = 1;

  for (std::size_t step = 1; step <= nstep; ++step) {

    // ---- Phase 1: zero all coupling arrays ----
    std::fill(coupling_0.begin(), coupling_0.end(), 0.0);
    std::fill(coupling_1.begin(), coupling_1.end(), 0.0);

    // ---- Phase 2a: intra-projection coupling ----
    // Each subnet reads its own history (state at end of previous step).
    // === Subnet 0: cortex (JansenRit) ===
    std::fill(coupling_0.begin(), coupling_0.end(), 0.0);

    // === Subnet 1: thalamus (ReducedSetFitzHughNagumo) ===
    std::fill(coupling_1.begin(), coupling_1.end(), 0.0);


    // ---- Phase 2b: inter-projection coupling ----
    // All reads happen before any history push, so every subnet sees the
    // consistent t-1 state of every source regardless of traversal order.
    accumulate_projection(
        inter_projections[0], history_0,
        coupling_1.data(), SubnetModel_1::kNumNodes);

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
    if (stim_ptrs[1] != nullptr) {
      const std::size_t step_0idx_s = step - 1;
      for (std::size_t cv = 0; cv < SubnetModel_1::kNumCouplingVars; ++cv) {
        for (std::size_t node = 0; node < SubnetModel_1::kNumNodes; ++node) {
          coupling_1[cv * SubnetModel_1::kNumNodes + node] +=
              stim_ptrs[1][cv * SubnetModel_1::kNumNodes * nstep +
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
    for (std::size_t cv = 0; cv < SubnetModel_1::kNumCouplingVars; ++cv) {
      for (std::size_t node = 0; node < SubnetModel_1::kNumNodes; ++node) {
        ctavg_1.accum(cv, node, 0) +=
            coupling_1[cv * SubnetModel_1::kNumNodes + node];
      }
    }

    // ---- Phase 3: integrate all subnets ----

    // Subnet 0: cortex (JansenRit)
    heun_step<SubnetModel_0>(state_0, coupling_0.data());

    // Subnet 1: thalamus (ReducedSetFitzHughNagumo)
    for (std::size_t node = 0; node < SubnetModel_1::kNumNodes; ++node) {
      // Parameter and coupling reads (same for all modes)
      const double tau = SubnetModel_1::param_at(SubnetModel_1::kParam_tau, node);
      const double a = SubnetModel_1::param_at(SubnetModel_1::kParam_a, node);
      const double b = SubnetModel_1::param_at(SubnetModel_1::kParam_b, node);
      const double K11 = SubnetModel_1::param_at(SubnetModel_1::kParam_K11, node);
      const double K12 = SubnetModel_1::param_at(SubnetModel_1::kParam_K12, node);
      const double K21 = SubnetModel_1::param_at(SubnetModel_1::kParam_K21, node);
      const double sigma = SubnetModel_1::param_at(SubnetModel_1::kParam_sigma, node);
      const double mu = SubnetModel_1::param_at(SubnetModel_1::kParam_mu, node);
      const double c_xi = coupling_1[0 * SubnetModel_1::kNumNodes + node];
      const double c_alpha = coupling_1[1 * SubnetModel_1::kNumNodes + node];

      // Cross-mode matrix-vector products from current state
      double Axi[3] = {};
      for (std::size_t mi = 0; mi < 3; ++mi) {
        for (std::size_t mk = 0; mk < 3; ++mk) {
          Axi[mi] += SubnetModel_1::kDerived_Aik[mi * 3 + mk]
                                * state_1(0, node, mk);
        }
      }
      double Balpha[3] = {};
      for (std::size_t mi = 0; mi < 3; ++mi) {
        for (std::size_t mk = 0; mk < 3; ++mk) {
          Balpha[mi] += SubnetModel_1::kDerived_Bik[mi * 3 + mk]
                                * state_1(2, node, mk);
        }
      }
      double Cxi[3] = {};
      for (std::size_t mi = 0; mi < 3; ++mi) {
        for (std::size_t mk = 0; mk < 3; ++mk) {
          Cxi[mi] += SubnetModel_1::kDerived_Cik[mi * 3 + mk]
                                * state_1(0, node, mk);
        }
      }

      // k1: derivatives for all modes
      double k1[12] = {};
      for (std::size_t m = 0; m < 3; ++m) {
        const double xi_m = state_1(0, node, m);
        const double eta_m = state_1(1, node, m);
        const double alpha_m = state_1(2, node, m);
        const double beta_m = state_1(3, node, m);
        const double e_i_m = SubnetModel_1::kDerived_e_i[m];
        const double f_i_m = SubnetModel_1::kDerived_f_i[m];
        const double IE_i_m = SubnetModel_1::kDerived_IE_i[m];
        const double II_i_m = SubnetModel_1::kDerived_II_i[m];
        const double m_i_m = SubnetModel_1::kDerived_m_i[m];
        const double n_i_m = SubnetModel_1::kDerived_n_i[m];
        const double Axi_m = Axi[m];
        const double Balpha_m = Balpha[m];
        const double Cxi_m = Cxi[m];
        k1[0 * 3 + m] = ((((tau * ((xi_m - ((e_i_m * std::pow(xi_m, 3.0)) / 3)) - eta_m)) + (K11 * (Axi_m - xi_m))) - (K12 * (Balpha_m - xi_m))) + (tau * (IE_i_m + c_xi)));
        k1[1 * 3 + m] = ((1 / tau) * ((xi_m - (b * eta_m)) + m_i_m));
        k1[2 * 3 + m] = (((tau * ((alpha_m - ((f_i_m * std::pow(alpha_m, 3.0)) / 3)) - beta_m)) + (K21 * (Cxi_m - alpha_m))) + (tau * (II_i_m + (c_alpha * mu))));
        k1[3 * 3 + m] = ((1 / tau) * ((alpha_m - (b * beta_m)) + n_i_m));
      }

      // Predictor: state + dt * k1 [+ noise for stochastic]
      // noise layout: (n_vars, n_nodes, n_modes, nstep) — same index used in corrector.
      double pred[12] = {};
      for (std::size_t sv = 0; sv < 4; ++sv) {
        for (std::size_t m = 0; m < 3; ++m) {
          pred[sv * 3 + m] = state_1(sv, node, m) + kDt * k1[sv * 3 + m];
        }
      }

      // Cross-mode matrix-vector products from predictor
      double Axi_p[3] = {};
      for (std::size_t mi = 0; mi < 3; ++mi) {
        for (std::size_t mk = 0; mk < 3; ++mk) {
          Axi_p[mi] += SubnetModel_1::kDerived_Aik[mi * 3 + mk]
                                  * pred[0 * 3 + mk];
        }
      }
      double Balpha_p[3] = {};
      for (std::size_t mi = 0; mi < 3; ++mi) {
        for (std::size_t mk = 0; mk < 3; ++mk) {
          Balpha_p[mi] += SubnetModel_1::kDerived_Bik[mi * 3 + mk]
                                  * pred[2 * 3 + mk];
        }
      }
      double Cxi_p[3] = {};
      for (std::size_t mi = 0; mi < 3; ++mi) {
        for (std::size_t mk = 0; mk < 3; ++mk) {
          Cxi_p[mi] += SubnetModel_1::kDerived_Cik[mi * 3 + mk]
                                  * pred[0 * 3 + mk];
        }
      }

      // k2: derivatives at predictor states
      double k2[12] = {};
      for (std::size_t m = 0; m < 3; ++m) {
        const double xi_m = pred[0 * 3 + m];
        const double eta_m = pred[1 * 3 + m];
        const double alpha_m = pred[2 * 3 + m];
        const double beta_m = pred[3 * 3 + m];
        const double e_i_m = SubnetModel_1::kDerived_e_i[m];
        const double f_i_m = SubnetModel_1::kDerived_f_i[m];
        const double IE_i_m = SubnetModel_1::kDerived_IE_i[m];
        const double II_i_m = SubnetModel_1::kDerived_II_i[m];
        const double m_i_m = SubnetModel_1::kDerived_m_i[m];
        const double n_i_m = SubnetModel_1::kDerived_n_i[m];
        const double Axi_m = Axi_p[m];
        const double Balpha_m = Balpha_p[m];
        const double Cxi_m = Cxi_p[m];
        k2[0 * 3 + m] = ((((tau * ((xi_m - ((e_i_m * std::pow(xi_m, 3.0)) / 3)) - eta_m)) + (K11 * (Axi_m - xi_m))) - (K12 * (Balpha_m - xi_m))) + (tau * (IE_i_m + c_xi)));
        k2[1 * 3 + m] = ((1 / tau) * ((xi_m - (b * eta_m)) + m_i_m));
        k2[2 * 3 + m] = (((tau * ((alpha_m - ((f_i_m * std::pow(alpha_m, 3.0)) / 3)) - beta_m)) + (K21 * (Cxi_m - alpha_m))) + (tau * (II_i_m + (c_alpha * mu))));
        k2[3 * 3 + m] = ((1 / tau) * ((alpha_m - (b * beta_m)) + n_i_m));
      }

      // Corrector: state += 0.5 * dt * (k1 + k2) [+ same noise for stochastic]
      for (std::size_t sv = 0; sv < 4; ++sv) {
        for (std::size_t m = 0; m < 3; ++m) {
          state_1(sv, node, m) +=
              0.5 * kDt * (k1[sv * 3 + m] + k2[sv * 3 + m]);
        }
      }
    }  // end node loop (combined-mode)

    // ---- Phase 4: push all updated states into history buffers ----
    // All integrations are complete before any push, so all inter-projection
    // reads in the next step will consistently see this step's final states.
    history_0.push(state_0);
    history_1.push(state_1);

    // ---- Monitor accumulation ----

    for (std::size_t node = 0; node < SubnetModel_0::kNumNodes; ++node) {
      std::array<double, SubnetModel_0::kNumVoi> voi_vals{};
      SubnetModel_0::compute_voi(state_0, node, voi_vals);
      for (std::size_t ivoi = 0; ivoi < SubnetModel_0::kNumVoi; ++ivoi) {
        monitor_0.accum(ivoi, node, 0) += voi_vals[ivoi];
      }
    }

    for (std::size_t node = 0; node < SubnetModel_1::kNumNodes; ++node) {
      for (std::size_t m = 0; m < SubnetModel_1::kNumModes; ++m) {
        monitor_1.accum(0, node, m) += state_1(0, node, m);
        monitor_1.accum(1, node, m) += state_1(2, node, m);
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
    result_1.times[current_chunk] = mid_step * kDt;
    monitor_1.write_chunk_average(result_1, current_chunk, steps_in_chunk);
    monitor_1.clear_accum();
    ctavg_1.write_chunk_average_into(result_1.ctavg_data, current_chunk, steps_in_chunk);
    ctavg_1.clear_accum();
    ++current_chunk;
    chunk_start_step = step + 1;
    steps_in_chunk = 0;
  }

  return {result_0, result_1};
}

}  // namespace tvb::hybrid::generated
