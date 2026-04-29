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
  static constexpr std::size_t kNumVoi              = ${len(subnet.variables_of_interest)};
  static constexpr std::size_t kSourceHistoryHorizon = ${source_history_horizon};

  static constexpr bool        kDelayedSelfFeedbackEnabled    = ${'true' if delayed_enabled else 'false'};
  static constexpr std::size_t kDelayedSelfFeedbackSteps      = ${delayed_steps};
  static constexpr double      kDelayedSelfFeedbackGain       = ${f'{delayed_gain:.17g}'};
  static constexpr std::size_t kDelayedSelfFeedbackSourceSvar = ${delayed_source_svar};
  static constexpr std::size_t kDelayedSelfFeedbackTargetSvar = ${delayed_target_svar};

  static constexpr std::array<int, kNumVoi> kVoiIndices = { ${', '.join(str(i) for i in voi_indices)} };

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

% if subnet.model_type == 'MontbrioPazoRoxin':
  static inline void compute_dfun(
      const tvb::hybrid::runtime::StateBuffer& state,
      const tvb::hybrid::runtime::HistoryBuffer& history,
      std::size_t node,
      std::array<double, kNumStateVars>& dx) {
    const double r     = std::max(0.0, state(0, node, 0));
    const double V     = state(1, node, 0);
    const double tau   = param_at(kParam_tau,   node);
    const double Delta = param_at(kParam_Delta, node);
    const double eta   = param_at(kParam_eta,   node);
    const double J     = param_at(kParam_J,     node);
    const double I     = param_at(kParam_I,     node);
    const double cr    = param_at(kParam_cr,    node);
    const double cv    = param_at(kParam_cv,    node);
    const double coupling_r = 0.0;
    double coupling_V = 0.0;
    if constexpr (kDelayedSelfFeedbackEnabled) {
      coupling_V += kDelayedSelfFeedbackGain *
                    tvb::hybrid::runtime::delayed_state_value(
                        history, kDelayedSelfFeedbackSteps,
                        kDelayedSelfFeedbackSourceSvar, node, 0);
    }
    dx[0] = (1.0 / tau) * (Delta / (M_PI * tau) + 2.0 * V * r);
    dx[1] = (1.0 / tau) *
            (V * V - (M_PI * M_PI) * tau * tau * r * r + eta + J * tau * r + I +
             cr * coupling_r + cv * coupling_V);
  }

  static inline void apply_state_constraints(
      tvb::hybrid::runtime::StateBuffer& state,
      std::size_t node) {
    state(0, node, 0) = std::max(0.0, state(0, node, 0));
  }
% else:
#error "Model type '${subnet.model_type}' does not have a generated dfun in this backend version."
% endif
};

${subnetwork_summary}
${projection_summary}

static constexpr std::size_t kNumNodes     = GeneratedModel::kNumNodes;
static constexpr std::size_t kNumModes     = GeneratedModel::kNumModes;
static constexpr std::size_t kNumStateVars = GeneratedModel::kNumStateVars;
static constexpr std::size_t kNumVoi       = GeneratedModel::kNumVoi;
static constexpr auto        kVoiIndices   = GeneratedModel::kVoiIndices;

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
