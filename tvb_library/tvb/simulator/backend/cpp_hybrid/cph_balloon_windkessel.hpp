#pragma once

#include <cmath>

namespace cph::balloon_windkessel {

struct Parameters {
  double signal_decay_s{};
  double flow_feedback_s2{};  // Feedback denominator, not a rate.
  double transit_time_s{};
  double grubb_exponent{};
  double resting_extraction{};
  double intravascular_ratio{};
  double frequency_offset_hz{};
  double relaxation_rate_per_s{};
  double echo_time_s{};
  double resting_blood_volume{};
};

inline constexpr Parameters kNominalParameters{
    0.65, 0.41, 0.98, 0.32, 0.4, 0.5, 40.3, 25.0, 0.04, 0.04,
};

struct State {
  double signal = 0.0;  // s^-1
  double inflow = 1.0;
  double volume = 1.0;
  double deoxy = 1.0;
};

enum class AdvanceResult { ok, invalid_input, domain_failure };

namespace detail {

inline bool positive_finite(double x) {
  return std::isfinite(x) && x > 0.0;
}

inline bool valid_state(const State &x) {
  return std::isfinite(x.signal) && positive_finite(x.inflow) &&
         positive_finite(x.volume) && positive_finite(x.deoxy);
}

inline bool valid_parameters(const Parameters &p) {
  return positive_finite(p.signal_decay_s) && positive_finite(p.flow_feedback_s2) &&
         positive_finite(p.transit_time_s) && positive_finite(p.grubb_exponent) &&
         p.grubb_exponent <= 1.0 && positive_finite(p.resting_extraction) &&
         p.resting_extraction < 1.0 && positive_finite(p.intravascular_ratio) &&
         positive_finite(p.frequency_offset_hz) && positive_finite(p.relaxation_rate_per_s) &&
         positive_finite(p.echo_time_s) && positive_finite(p.resting_blood_volume) &&
         p.resting_blood_volume < 1.0;
}

inline State derivative(const State &x, const Parameters &p, double drive_per_s2,
                        double outflow_exponent, double log_unextracted) {
  const double outflow = std::pow(x.volume, outflow_exponent);
  const double extraction = -std::expm1(log_unextracted / x.inflow);
  return {drive_per_s2 - x.signal / p.signal_decay_s - (x.inflow - 1.0) / p.flow_feedback_s2,
          x.signal,
          (x.inflow - outflow) / p.transit_time_s,
          (x.inflow * extraction / p.resting_extraction - outflow * x.deoxy / x.volume) /
              p.transit_time_s};
}

inline State shift(const State &x, double dt_s, const State &slope) {
  return {x.signal + dt_s * slope.signal, x.inflow + dt_s * slope.inflow,
          x.volume + dt_s * slope.volume, x.deoxy + dt_s * slope.deoxy};
}

inline State rk4_slopes(const State &a, const State &b, const State &c, const State &d) {
  return {a.signal + 2.0 * (b.signal + c.signal) + d.signal,
          a.inflow + 2.0 * (b.inflow + c.inflow) + d.inflow,
          a.volume + 2.0 * (b.volume + c.volume) + d.volume,
          a.deoxy + 2.0 * (b.deoxy + c.deoxy) + d.deoxy};
}

}  // namespace detail

// Replay the just-closed interval with constant, already calibrated drive. An
// integer count declares commensurate ODE steps of 0.010 / substeps seconds.
// Add fractional BOLD at every right endpoint to bold_sum (which cannot alias
// state). Hold the grid fixed across an output window for an arithmetic mean.
// Neither caller-owned value is changed on failure; no clipping or retry.
inline AdvanceResult advance_10ms(State &state, const Parameters &p, double drive_per_s2,
                                  int substeps, double &bold_sum) {
  if (substeps <= 0 || !std::isfinite(drive_per_s2) || !std::isfinite(bold_sum) ||
      !detail::valid_state(state) || !detail::valid_parameters(p)) {
    return AdvanceResult::invalid_input;
  }

  const double h = 0.010 / substeps;
  const double outflow_exponent = 1.0 / p.grubb_exponent;
  const double log_unextracted = std::log1p(-p.resting_extraction);
  const double extraction_echo = p.resting_extraction * p.echo_time_s;
  const double k1 = 4.3 * p.frequency_offset_hz * extraction_echo;
  const double k2 = p.intravascular_ratio * p.relaxation_rate_per_s * extraction_echo;
  const double k3 = 1.0 - p.intravascular_ratio;
  if (!std::isfinite(outflow_exponent) || !std::isfinite(k1) || !std::isfinite(k2)) {
    return AdvanceResult::domain_failure;
  }

  State x = state;
  double sum = bold_sum;
  const auto slope = [&](const State &stage) {
    return detail::derivative(stage, p, drive_per_s2, outflow_exponent, log_unextracted);
  };
  for (int i = 0; i < substeps; ++i) {
    // Classical RK4 for smooth dynamics within each held-drive interval.
    const State a = slope(x);
    const State at_b = detail::shift(x, 0.5 * h, a);
    if (!detail::valid_state(at_b)) return AdvanceResult::domain_failure;
    const State b = slope(at_b);
    const State at_c = detail::shift(x, 0.5 * h, b);
    if (!detail::valid_state(at_c)) return AdvanceResult::domain_failure;
    const State c = slope(at_c);
    const State at_d = detail::shift(x, h, c);
    if (!detail::valid_state(at_d)) return AdvanceResult::domain_failure;
    const State d = slope(at_d);
    x = detail::shift(x, h / 6.0, detail::rk4_slopes(a, b, c, d));
    if (!detail::valid_state(x)) return AdvanceResult::domain_failure;

    // Stephan et al. (2007), revised nonlinear readout (Eq. 12).
    sum += p.resting_blood_volume * (k1 * (1.0 - x.deoxy) +
                                    k2 * (1.0 - x.deoxy / x.volume) +
                                    k3 * (1.0 - x.volume));
    if (!std::isfinite(sum)) return AdvanceResult::domain_failure;
  }
  state = x;
  bold_sum = sum;
  return AdvanceResult::ok;
}

}  // namespace cph::balloon_windkessel
