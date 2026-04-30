// Auto-generated C++ translation unit for the TVB hybrid backend.
// ${num_subnetworks} subnetwork(s) / ${num_intra_projections} intra / ${num_inter_projections} inter projection(s)

#include <array>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "runtime/runtime.hpp"

namespace tvb::hybrid::generated {

static constexpr double kDt = ${f'{dt:.17g}'};

// ============================================================
// Per-subnet model structs
// ============================================================
<%
import sys
%>
% for si, sc in enumerate(subnets_ctx):
<%
  subnet   = sc['subnet']
  dfun_ctx = sc['dfun_ctx']
  horizon  = sc['horizon']
  params   = sorted(subnet.parameter_values.items())
%>
// Subnet ${si}: ${subnet.name} (${subnet.model_type})
struct SubnetModel_${si} {
  static constexpr double      kDt              = ${f'{dt:.17g}'};
  static constexpr std::size_t kNumNodes        = ${subnet.n_nodes};
  static constexpr std::size_t kNumModes        = ${subnet.n_modes};
  static constexpr std::size_t kNumStateVars    = ${subnet.n_state_vars};
  static constexpr std::size_t kNumCouplingVars = ${dfun_ctx['n_coupling_vars']};
  static constexpr std::size_t kNumVoi          = ${len(subnet.variables_of_interest)};
  static constexpr std::size_t kSourceHistoryHorizon = ${horizon};
  static constexpr bool        kIsStochastic         = ${'true' if sc['is_stochastic'] else 'false'};

% for param_name, param_values in params:
  static constexpr std::array<double, ${len(param_values)}> kParam_${param_name} = { ${', '.join(f'{float(v):.17g}' for v in param_values)} };
% endfor

  template <std::size_t N>
  static inline double param_at(const std::array<double, N>& values, std::size_t node) {
    if constexpr (N == 1) { return values[0]; }
    else { return values[node]; }
  }

% for decl in dfun_ctx['dfun_helper_decls']:
${decl}

% endfor
  // coupling layout: coupling[cvar_slot * kNumNodes + node]
  static inline void compute_dfun(
      const tvb::hybrid::runtime::StateBuffer& state,
      const double* coupling,
      std::size_t node,
      std::array<double, kNumStateVars>& dx) {
% for stmt in dfun_ctx['dfun_state_reads']:
    ${stmt}
% endfor
% for stmt in dfun_ctx['dfun_param_reads']:
    ${stmt}
% endfor
% for stmt in dfun_ctx['dfun_coupling_reads']:
    ${stmt}
% endfor
% for stmt in dfun_ctx['dfun_intermediate_decls']:
    ${stmt}
% endfor
% for stmt in dfun_ctx['dfun_dx_assignments']:
    ${stmt}
% endfor
  }

  static inline void compute_voi(
      const tvb::hybrid::runtime::StateBuffer& state,
      std::size_t node,
      std::array<double, kNumVoi>& voi) {
% for stmt in dfun_ctx['dfun_voi_assignments']:
    ${stmt}
% endfor
  }

  static inline void apply_state_constraints(
      tvb::hybrid::runtime::StateBuffer& state,
      std::size_t node) {
% if dfun_ctx['dfun_constraint_stmts']:
% for stmt in dfun_ctx['dfun_constraint_stmts']:
    ${stmt}
% endfor
% else:
    (void)state; (void)node;
% endif
  }
};

% endfor
${subnetwork_summary}
${projection_summary}

// ============================================================
// Simulation metadata
// ============================================================

using SimulationMetadata = tvb::hybrid::runtime::SimulationMetadata;
using SimulationResult   = tvb::hybrid::runtime::SimulationResult;

inline SimulationMetadata describe() {
  return SimulationMetadata{
      "${module_name}",
      "${cache_key}",
      "${backend_version}",
      "${user_source_hint}",
      kDt,
      ${num_subnetworks},
      ${num_inter_projections},
      ${num_intra_projections},
      ${num_monitors},
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

  if (initial_states.size() != ${num_subnetworks}) {
    throw std::runtime_error(
        "run_simulation: initial_states.size() != num_subnetworks (${num_subnetworks})");
  }
  if (intra_projections.size() != ${num_intra_projections}) {
    throw std::runtime_error(
        "run_simulation: intra_projections.size() != num_intra_projections (${num_intra_projections})");
  }
  if (inter_projections.size() != ${num_inter_projections}) {
    throw std::runtime_error(
        "run_simulation: inter_projections.size() != num_inter_projections (${num_inter_projections})");
  }
  if (chunk_size == 0) {
    throw std::runtime_error("chunk_size must be >= 1.");
  }

  // --- Per-subnet state / history / coupling / monitor ---
% for si, sc in enumerate(subnets_ctx):
  StateBuffer state_${si}(
      SubnetModel_${si}::kNumStateVars,
      SubnetModel_${si}::kNumNodes,
      SubnetModel_${si}::kNumModes,
      initial_states[${si}]);
  HistoryBuffer history_${si}(
      SubnetModel_${si}::kSourceHistoryHorizon,
      SubnetModel_${si}::kNumStateVars,
      SubnetModel_${si}::kNumNodes,
      SubnetModel_${si}::kNumModes);
  for (std::size_t _i = 0; _i < history_${si}.capacity(); ++_i) {
    history_${si}.push(state_${si});
  }
  std::vector<double> coupling_${si}(
      SubnetModel_${si}::kNumCouplingVars * SubnetModel_${si}::kNumNodes, 0.0);
  MonitorBuffer monitor_${si}(
      SubnetModel_${si}::kNumVoi,
      SubnetModel_${si}::kNumNodes,
      SubnetModel_${si}::kNumModes);
  MonitorBuffer ctavg_${si}(
      SubnetModel_${si}::kNumCouplingVars,
      SubnetModel_${si}::kNumNodes,
      1);  // coupling has no mode dimension
% endfor

  // --- Result storage ---
  const std::size_t num_chunks = (nstep + chunk_size - 1) / chunk_size;
% for si, sc in enumerate(subnets_ctx):
  SimulationResult result_${si};
  result_${si}.num_chunks      = num_chunks;
  result_${si}.num_voi         = SubnetModel_${si}::kNumVoi;
  result_${si}.num_nodes       = SubnetModel_${si}::kNumNodes;
  result_${si}.num_modes       = SubnetModel_${si}::kNumModes;
  result_${si}.num_coupling_vars = SubnetModel_${si}::kNumCouplingVars;
  result_${si}.times.resize(num_chunks);
  result_${si}.data.resize(
      num_chunks *
      SubnetModel_${si}::kNumVoi *
      SubnetModel_${si}::kNumNodes *
      SubnetModel_${si}::kNumModes,
      0.0);
  result_${si}.ctavg_data.resize(
      num_chunks * SubnetModel_${si}::kNumCouplingVars * SubnetModel_${si}::kNumNodes,
      0.0);
% endfor

  std::size_t current_chunk    = 0;
  std::size_t steps_in_chunk   = 0;
  std::size_t chunk_start_step = 1;

  for (std::size_t step = 1; step <= nstep; ++step) {

    // ---- Phase 1: zero all coupling arrays ----
% for si in range(num_subnetworks):
    std::fill(coupling_${si}.begin(), coupling_${si}.end(), 0.0);
% endfor

    // ---- Phase 2a: intra-projection coupling ----
    // Each subnet reads its own history (state at end of previous step).
% for si, sc in enumerate(subnets_ctx):
    // === Subnet ${si}: ${sc['subnet'].name} (${sc['subnet'].model_type}) ===
    std::fill(coupling_${si}.begin(), coupling_${si}.end(), 0.0);
<%
    intra_indices = sc['intra_proj_indices']
%>
% for pi in intra_indices:
    accumulate_projection(
        intra_projections[${pi}], history_${si},
        coupling_${si}.data(), SubnetModel_${si}::kNumNodes);
% endfor
% endfor

    // ---- Phase 2b: inter-projection coupling ----
    // All reads happen before any history push, so every subnet sees the
    // consistent t-1 state of every source regardless of traversal order.
% for pi, src_si, tgt_si in all_inter_proj_routes:
    accumulate_projection(
        inter_projections[${pi}], history_${src_si},
        coupling_${tgt_si}.data(), SubnetModel_${tgt_si}::kNumNodes);
% endfor

    // ---- Phase 2c: apply pre-computed stimulus to coupling ----
    // Mirrors Numba: stimulus is added after projections, before ctavg accumulation,
    // so the ctavg monitor reflects the total input (projection + stimulus).
    // stim layout: (n_cvar, n_nodes, nstep), target_cvar already applied Python-side.
% for si in range(num_subnetworks):
    if (stim_ptrs[${si}] != nullptr) {
      const std::size_t step_0idx_s = step - 1;
      for (std::size_t cv = 0; cv < SubnetModel_${si}::kNumCouplingVars; ++cv) {
        for (std::size_t node = 0; node < SubnetModel_${si}::kNumNodes; ++node) {
          coupling_${si}[cv * SubnetModel_${si}::kNumNodes + node] +=
              stim_ptrs[${si}][cv * SubnetModel_${si}::kNumNodes * nstep +
                               node * nstep + step_0idx_s];
        }
      }
    }
% endfor

    // ---- Phase 2d: accumulate afferent coupling for AfferentCoupling monitor ----
    // Mirrors Numba template: coupling is sampled after full accumulation,
    // before integration, so ctavg reflects the actual input each node received.
% for si in range(num_subnetworks):
    for (std::size_t cv = 0; cv < SubnetModel_${si}::kNumCouplingVars; ++cv) {
      for (std::size_t node = 0; node < SubnetModel_${si}::kNumNodes; ++node) {
        ctavg_${si}.accum(cv, node, 0) +=
            coupling_${si}[cv * SubnetModel_${si}::kNumNodes + node];
      }
    }
% endfor

    // ---- Phase 3: integrate all subnets ----
% for si, sc in enumerate(subnets_ctx):
<%
    subnet   = sc['subnet']
    dfun_ctx = sc['dfun_ctx']
    is_comb  = dfun_ctx['is_combined']
    n_modes  = subnet.n_modes
    n_sv     = subnet.n_state_vars
%>
    // Subnet ${si}: ${subnet.name} (${subnet.model_type})
% if is_comb:
    for (std::size_t node = 0; node < SubnetModel_${si}::kNumNodes; ++node) {
      // Parameter and coupling reads (same for all modes)
% for stmt in dfun_ctx['param_reads']:
      ${stmt}
% endfor
% for stmt in dfun_ctx['coupling_reads']:
      ${stmt}
% endfor

      // Cross-mode matrix-vector products from current state
% for op in dfun_ctx['dm_ops_info']:
      double ${op['result']}[${n_modes}] = {};
      for (std::size_t mi = 0; mi < ${n_modes}; ++mi) {
        for (std::size_t mk = 0; mk < ${n_modes}; ++mk) {
          ${op['result']}[mi] += SubnetModel_${si}::kDerived_${op['matrix']}[mi * ${n_modes} + mk]
                                * state_${si}(${op['svar_idx']}, node, mk);
        }
      }
% endfor

      // k1: derivatives for all modes
      double k1[${n_sv * n_modes}] = {};
      for (std::size_t m = 0; m < ${n_modes}; ++m) {
% for stmt in dfun_ctx['state_reads_k1']:
        ${stmt}
% endfor
% for stmt in dfun_ctx['derived_scalar_reads']:
        ${stmt}
% endfor
% for stmt in dfun_ctx['cross_mode_reads_k1']:
        ${stmt}
% endfor
% for stmt in dfun_ctx['k1_assignments']:
        ${stmt}
% endfor
      }

% if sc['is_euler']:
      // Euler: state += dt * k1 [+ noise for stochastic]
      for (std::size_t sv = 0; sv < ${n_sv}; ++sv) {
        for (std::size_t m = 0; m < ${n_modes}; ++m) {
          state_${si}(sv, node, m) += kDt * k1[sv * ${n_modes} + m];
% if sc['is_stochastic']:
          state_${si}(sv, node, m) += noise_ptrs[${si}][
              sv * SubnetModel_${si}::kNumNodes * ${n_modes} * nstep +
              node * ${n_modes} * nstep +
              m * nstep + (step - 1)];
% endif
        }
      }
% else:
      // Predictor: state + dt * k1 [+ noise for stochastic]
      // noise layout: (n_vars, n_nodes, n_modes, nstep) — same index used in corrector.
      double pred[${n_sv * n_modes}] = {};
      for (std::size_t sv = 0; sv < ${n_sv}; ++sv) {
        for (std::size_t m = 0; m < ${n_modes}; ++m) {
          pred[sv * ${n_modes} + m] = state_${si}(sv, node, m) + kDt * k1[sv * ${n_modes} + m];
% if sc['is_stochastic']:
          pred[sv * ${n_modes} + m] += noise_ptrs[${si}][
              sv * SubnetModel_${si}::kNumNodes * ${n_modes} * nstep +
              node * ${n_modes} * nstep +
              m * nstep + (step - 1)];
% endif
        }
      }

      // Cross-mode matrix-vector products from predictor
% for op in dfun_ctx['dm_ops_info']:
      double ${op['result']}_p[${n_modes}] = {};
      for (std::size_t mi = 0; mi < ${n_modes}; ++mi) {
        for (std::size_t mk = 0; mk < ${n_modes}; ++mk) {
          ${op['result']}_p[mi] += SubnetModel_${si}::kDerived_${op['matrix']}[mi * ${n_modes} + mk]
                                  * pred[${op['svar_idx']} * ${n_modes} + mk];
        }
      }
% endfor

      // k2: derivatives at predictor states
      double k2[${n_sv * n_modes}] = {};
      for (std::size_t m = 0; m < ${n_modes}; ++m) {
% for stmt in dfun_ctx['state_reads_k2']:
        ${stmt}
% endfor
% for stmt in dfun_ctx['derived_scalar_reads']:
        ${stmt}
% endfor
% for stmt in dfun_ctx['cross_mode_reads_k2']:
        ${stmt}
% endfor
% for stmt in dfun_ctx['k2_assignments']:
        ${stmt}
% endfor
      }

      // Corrector: state += 0.5 * dt * (k1 + k2) [+ same noise for stochastic]
      for (std::size_t sv = 0; sv < ${n_sv}; ++sv) {
        for (std::size_t m = 0; m < ${n_modes}; ++m) {
          state_${si}(sv, node, m) +=
              0.5 * kDt * (k1[sv * ${n_modes} + m] + k2[sv * ${n_modes} + m]);
% if sc['is_stochastic']:
          state_${si}(sv, node, m) += noise_ptrs[${si}][
              sv * SubnetModel_${si}::kNumNodes * ${n_modes} * nstep +
              node * ${n_modes} * nstep +
              m * nstep + (step - 1)];
% endif
        }
      }
% endif
    }  // end node loop (combined-mode)
% else:
% if sc['is_euler']:
% if sc['is_stochastic']:
    euler_step_stochastic<SubnetModel_${si}>(
        state_${si}, coupling_${si}.data(), noise_ptrs[${si}], step - 1, nstep);
% else:
    euler_step<SubnetModel_${si}>(state_${si}, coupling_${si}.data());
% endif
% else:
% if sc['is_stochastic']:
    heun_step_stochastic<SubnetModel_${si}>(
        state_${si}, coupling_${si}.data(), noise_ptrs[${si}], step - 1, nstep);
% else:
    heun_step<SubnetModel_${si}>(state_${si}, coupling_${si}.data());
% endif
% endif
% endif
% endfor

    // ---- Phase 4: push all updated states into history buffers ----
    // All integrations are complete before any push, so all inter-projection
    // reads in the next step will consistently see this step's final states.
% for si in range(num_subnetworks):
    history_${si}.push(state_${si});
% endfor

    // ---- Monitor accumulation ----
% for si, sc in enumerate(subnets_ctx):
<%
    dfun_ctx = sc['dfun_ctx']
    subnet   = sc['subnet']
    is_comb  = dfun_ctx['is_combined']
    svar_to_idx = {sv: i for i, sv in enumerate(subnet.state_variables)}
%>
% if is_comb:
    for (std::size_t node = 0; node < SubnetModel_${si}::kNumNodes; ++node) {
      for (std::size_t m = 0; m < SubnetModel_${si}::kNumModes; ++m) {
% for ivoi, voi in enumerate(subnet.variables_of_interest):
        monitor_${si}.accum(${ivoi}, node, m) += state_${si}(${svar_to_idx.get(voi, 0)}, node, m);
% endfor
      }
    }
% else:
    for (std::size_t node = 0; node < SubnetModel_${si}::kNumNodes; ++node) {
      std::array<double, SubnetModel_${si}::kNumVoi> voi_vals{};
      SubnetModel_${si}::compute_voi(state_${si}, node, voi_vals);
      for (std::size_t ivoi = 0; ivoi < SubnetModel_${si}::kNumVoi; ++ivoi) {
        monitor_${si}.accum(ivoi, node, 0) += voi_vals[ivoi];
      }
    }
% endif
% endfor

    ++steps_in_chunk;
    const bool close_chunk = (steps_in_chunk == chunk_size) || (step == nstep);
    if (!close_chunk) { continue; }

    // Timestamp: midpoint of [chunk_start_step, chunk_start_step + steps_in_chunk - 1]
    const double mid_step = static_cast<double>(chunk_start_step) +
                            (static_cast<double>(steps_in_chunk) - 1.0) / 2.0;
% for si, sc in enumerate(subnets_ctx):
    result_${si}.times[current_chunk] = mid_step * kDt;
    monitor_${si}.write_chunk_average(result_${si}, current_chunk, steps_in_chunk);
    monitor_${si}.clear_accum();
    ctavg_${si}.write_chunk_average_into(result_${si}.ctavg_data, current_chunk, steps_in_chunk);
    ctavg_${si}.clear_accum();
% endfor
    ++current_chunk;
    chunk_start_step = step + 1;
    steps_in_chunk = 0;
  }

  return {${', '.join(f'result_{si}' for si in range(num_subnetworks))}};
}

}  // namespace tvb::hybrid::generated
