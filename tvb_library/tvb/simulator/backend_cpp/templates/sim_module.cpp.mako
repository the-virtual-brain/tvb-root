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
    std::size_t chunk_size) {

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
% endfor

  // --- Result storage ---
  const std::size_t num_chunks = (nstep + chunk_size - 1) / chunk_size;
% for si, sc in enumerate(subnets_ctx):
  SimulationResult result_${si};
  result_${si}.num_chunks = num_chunks;
  result_${si}.num_voi    = SubnetModel_${si}::kNumVoi;
  result_${si}.num_nodes  = SubnetModel_${si}::kNumNodes;
  result_${si}.num_modes  = SubnetModel_${si}::kNumModes;
  result_${si}.times.resize(num_chunks);
  result_${si}.data.resize(
      num_chunks *
      SubnetModel_${si}::kNumVoi *
      SubnetModel_${si}::kNumNodes *
      SubnetModel_${si}::kNumModes,
      0.0);
% endfor

  std::size_t current_chunk    = 0;
  std::size_t steps_in_chunk   = 0;
  std::size_t chunk_start_step = 1;

  for (std::size_t step = 1; step <= nstep; ++step) {

% for si, sc in enumerate(subnets_ctx):
    // === Subnet ${si}: ${sc['subnet'].name} (${sc['subnet'].model_type}) ===
    std::fill(coupling_${si}.begin(), coupling_${si}.end(), 0.0);
<%
    intra_indices = sc['intra_proj_indices']
    inter_targets = sc['inter_proj_targets']
%>
% if intra_indices:
    // intra-projections
% for pi in intra_indices:
    accumulate_projection(
        intra_projections[${pi}], history_${si},
        coupling_${si}.data(), SubnetModel_${si}::kNumNodes);
% endfor
% endif
% if inter_targets:
    // inter-projections targeting this subnet
% for pi, src_si in inter_targets:
    accumulate_projection(
        inter_projections[${pi}], history_${src_si},
        coupling_${si}.data(), SubnetModel_${si}::kNumNodes);
% endfor
% endif
    heun_step<SubnetModel_${si}>(state_${si}, coupling_${si}.data());
    history_${si}.push(state_${si});

% endfor
    // Monitor accumulation
% for si, sc in enumerate(subnets_ctx):
    for (std::size_t node = 0; node < SubnetModel_${si}::kNumNodes; ++node) {
      std::array<double, SubnetModel_${si}::kNumVoi> voi_vals{};
      SubnetModel_${si}::compute_voi(state_${si}, node, voi_vals);
      for (std::size_t ivoi = 0; ivoi < SubnetModel_${si}::kNumVoi; ++ivoi) {
        monitor_${si}.accum(ivoi, node, 0) += voi_vals[ivoi];
      }
    }
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
% endfor
    ++current_chunk;
    chunk_start_step = step + 1;
    steps_in_chunk = 0;
  }

  return {${', '.join(f'result_{si}' for si in range(num_subnetworks))}};
}

}  // namespace tvb::hybrid::generated
