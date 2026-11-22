// Standalone driver for check_balloon_windkessel_native.py; not a TVB binding.
#include "cph_balloon_windkessel.hpp"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>

namespace bw = cph::balloon_windkessel;

void require(bool ok, const char *message) {
  if (!ok) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

void check_failures() {
  const bw::Parameters p{0.65, 0.41, 0.98, 0.32, 0.4, 0.5, 40.3, 25.0, 0.04, 0.04};
  const double huge = std::numeric_limits<double>::max();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto check = [&](bw::AdvanceResult expected, bw::State initial,
                         const bw::Parameters &params, double drive, int steps,
                         double initial_sum = 0.125) {
    bw::State x = initial;
    double sum = initial_sum;
    require(bw::advance_10ms(x, params, drive, steps, sum) == expected, "wrong failure result");
    require(x.signal == initial.signal && x.inflow == initial.inflow &&
                x.volume == initial.volume && x.deoxy == initial.deoxy &&
                (sum == initial_sum || (std::isnan(sum) && std::isnan(initial_sum))),
            "failed call modified state or sum");
  };
  const auto invalid = bw::AdvanceResult::invalid_input;
  const auto domain = bw::AdvanceResult::domain_failure;
  check(invalid, {}, p, 0.0, 0);
  check(invalid, {}, p, 0.0, -1);
  check(invalid, {}, p, nan, 2);
  check(invalid, {}, p, 0.0, 2, nan);
  check(invalid, {std::numeric_limits<double>::infinity(), 1, 1, 1}, p, 0.0, 2);
  check(invalid, {0, 0, 1, 1}, p, 0.0, 2);
  check(invalid, {0, 1, -1, 1}, p, 0.0, 2);
  check(invalid, {0, 1, 1, 0}, p, 0.0, 2);
  auto bad = p;
  bad.resting_extraction = 1.0;
  check(invalid, {}, bad, 0.0, 2);
  bad = p;
  bad.transit_time_s = 0.0;
  check(invalid, {}, bad, 0.0, 2);
  bad = p;
  bad.resting_blood_volume = 4.0;
  check(invalid, {}, bad, 0.0, 2);
  bad = p;
  bad.frequency_offset_hz = huge;
  check(domain, {}, bad, 0.0, 2);  // Readout coefficient overflow.
  bad.frequency_offset_hz = 1e307;
  check(domain, {0, 1, 1, 0.5}, bad, 0.0, 2, huge);  // BOLD sum overflow.
  check(domain, {}, p, huge, 2);  // Nonfinite RK slope combination.
  check(domain, {-3.0, 0.02, 1.0, 1.0}, p, 0.0, 2);  // Crosses f=0 in the second step.

  bw::State x, with_seed;
  double sum = 0.0, seeded = 0.125;
  require(bw::advance_10ms(x, p, 0.5, 2, sum) == bw::AdvanceResult::ok, "unseeded step failed");
  require(bw::advance_10ms(with_seed, p, 0.5, 2, seeded) == bw::AdvanceResult::ok,
          "seeded step failed");
  require(sum != 0.0 && std::abs((seeded - 0.125) - sum) < 4e-17, "BOLD sum not additive");
}

int main() {
  check_failures();
  int substeps;
  bw::Parameters p{};
  bw::State x;
  require(static_cast<bool>(std::cin >> substeps >> p.signal_decay_s >> p.flow_feedback_s2 >>
                            p.transit_time_s >> p.grubb_exponent >> p.resting_extraction >>
                            p.intravascular_ratio >> p.frequency_offset_hz >>
                            p.relaxation_rate_per_s >> p.echo_time_s >> p.resting_blood_volume >>
                            x.signal >> x.inflow >> x.volume >> x.deoxy), "invalid driver header");
  std::cout << std::setprecision(17);
  for (double drive; std::cin >> drive;) {
    double sum = 0.0;
    require(bw::advance_10ms(x, p, drive, substeps, sum) == bw::AdvanceResult::ok,
            "trajectory step failed");
    std::cout << x.signal << ' ' << x.inflow << ' ' << x.volume << ' ' << x.deoxy << ' '
              << sum << '\n';
  }
  require(std::cin.eof(), "invalid driver input");
}
