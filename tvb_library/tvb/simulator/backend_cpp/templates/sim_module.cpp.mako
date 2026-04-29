// Auto-generated C++ translation unit for the TVB hybrid backend.
// Model: ${subnet.model_type} / Integrator: ${subnet.integrator.type_name}

#include <array>
#include <cstddef>
#include <cmath>
#include <vector>

#include "runtime/runtime.hpp"

namespace tvb::hybrid::generated {

struct GeneratedModel {
  static constexpr const char* kModuleName     = "${module_name}";
  static constexpr const char* kCacheKey       = "${cache_key}";
  static constexpr const char* kBackendVersion = "${backend_version}";
  static constexpr const char* kUserSourceHint = "${user_source_hint}";
  static constexpr double      kDt             = ${f'{dt:.17g}'};

  static constexpr std::size_t kNumSubnetworks      = ${num_subnetworks};
  static constexpr std::size_t kNumInterProjections = ${num_inter_projections};
  static constexpr std::size_t kNumIntraProjections = ${num_intra_projections};
  static constexpr std::size_t kNumMonitors         = ${num_monitors};

  static constexpr std::size_t kNumNodes            = ${subnet.n_nodes};
  static constexpr std::size_t kNumModes            = ${subnet.n_modes};
  static constexpr std::size_t kNumStateVars        = ${subnet.n_state_vars};
  static constexpr std::size_t kNumCouplingVars     = ${n_coupling_vars};
  static constexpr std::size_t kNumVoi              = ${len(subnet.variables_of_interest)};
  static constexpr std::size_t kSourceHistoryHorizon = ${source_history_horizon};

  // Model parameters (${subnet.model_type})
<%
  params = sorted(subnet.parameter_values.items())
%>
% for param_name, param_values in params:
  static constexpr std::array<double, ${len(param_values)}> kParam_${param_name} = { ${', '.join(f'{float(v):.17g}' for v in param_values)} };
% endfor

  template <std::size_t N>
  static inline double param_at(const std::array<double, N>& values, std::size_t node) {
    if constexpr (N == 1) { return values[0]; }
    else { return values[node]; }
  }

  // Helper functions
% for decl in dfun_helper_decls:
${decl}

% endfor
  // coupling layout: coupling[cvar_slot * kNumNodes + node]
  static inline void compute_dfun(
      const tvb::hybrid::runtime::StateBuffer& state,
      const double* coupling,
      std::size_t node,
      std::array<double, kNumStateVars>& dx) {
    // state variables
% for stmt in dfun_state_reads:
    ${stmt}
% endfor
    // model parameters
% for stmt in dfun_param_reads:
    ${stmt}
% endfor
    // coupling inputs (zero until projection support is wired in;
    // delayed self-feedback, if enabled, overrides specific terms)
% for stmt in dfun_coupling_reads:
    ${stmt}
% endfor
% for stmt in dfun_intermediate_decls:
    ${stmt}
% endfor
    // derivatives
% for stmt in dfun_dx_assignments:
    ${stmt}
% endfor
  }

  static inline void compute_voi(
      const tvb::hybrid::runtime::StateBuffer& state,
      std::size_t node,
      std::array<double, kNumVoi>& voi) {
% for stmt in dfun_voi_assignments:
    ${stmt}
% endfor
  }

  static inline void apply_state_constraints(
      tvb::hybrid::runtime::StateBuffer& state,
      std::size_t node) {
% if dfun_constraint_stmts:
% for stmt in dfun_constraint_stmts:
    ${stmt}
% endfor
% else:
    (void)state; (void)node;
% endif
  }
};

${subnetwork_summary}
${projection_summary}

static constexpr std::size_t kNumNodes     = GeneratedModel::kNumNodes;
static constexpr std::size_t kNumModes     = GeneratedModel::kNumModes;
static constexpr std::size_t kNumStateVars = GeneratedModel::kNumStateVars;
static constexpr std::size_t kNumVoi       = GeneratedModel::kNumVoi;

using SimulationMetadata = tvb::hybrid::runtime::SimulationMetadata;
using SimulationResult   = tvb::hybrid::runtime::SimulationResult;

inline SimulationMetadata describe() {
  return tvb::hybrid::runtime::describe<GeneratedModel>();
}

inline SimulationResult run_simulation(
    const std::vector<double>& initial_state,
    std::size_t nstep,
    std::size_t chunk_size) {
  return tvb::hybrid::runtime::run_simulation<GeneratedModel>(
      initial_state, nstep, chunk_size);
}

}  // namespace tvb::hybrid::generated
