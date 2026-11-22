"""Compare the standalone C++ seam with the reviewed float64 held-drive oracle.

Run manually with the path to the C++17 driver; not collected by ordinary pytest.
"""

from dataclasses import astuple, replace
from pathlib import Path
import subprocess
import sys

import numpy as np

from balloon_windkessel_reference import (
    DRIVE_WINDOW_MS, NOMINAL_PARAMETERS, REST_STATE, bold_fraction, solve_held_drive,
)


def native(executable, substeps, parameters, initial_state, drive):
    tokens = (substeps, *astuple(parameters), *initial_state, *drive)
    result = subprocess.run([str(executable)], input=" ".join(map(str, tokens)),
                            text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"standalone driver: {result.stderr}")
    return np.fromstring(result.stdout, sep=" ").reshape(len(drive), 5)


def check_case(executable, name, drive, parameters=NOMINAL_PARAMETERS, initial_state=REST_STATE):
    errors = []
    reference_means = []
    for substeps in (1, 2, 4):
        rows = native(executable, substeps, parameters, initial_state, drive)
        oracle = solve_held_drive(drive, neural_dt_ms=DRIVE_WINDOW_MS,
                                  hemo_dt_ms=DRIVE_WINDOW_MS / substeps,
                                  parameters=parameters, initial_state=initial_state)
        # Both state comparisons land at window ends. The readout comparison
        # includes exactly the same right endpoints in each summed contribution.
        states, means = rows[:, :4], rows[:, 4] / substeps
        expected_states = oracle.states[substeps::substeps]
        expected_means = bold_fraction(oracle.states[1:], parameters).reshape(-1, substeps).mean(1)
        assert np.all(np.isfinite(rows)) and np.all(states[:, 1:] > 0)
        error = np.r_[np.max(np.abs(states - expected_states), axis=0),
                      np.max(np.abs(means - expected_means))]
        errors.append(error)
        reference_means.append(expected_means)
        print(f"{name:14} {10 / substeps:4g} ms absolute [s,f,v,q,BOLD]:",
              " ".join(f"{e:.3e}" for e in error), flush=True)
        if substeps == 2:
            lo, hi = states[:, 1:].min(0), states[:, 1:].max(0)
            print(f"  5 ms grid, 10 ms boundary f/v/q ranges: {lo} to {hi}; "
                  f"peak |BOLD|={np.max(np.abs(means)):.3e}", flush=True)
        if name == "rest":
            np.testing.assert_allclose(states, np.broadcast_to(REST_STATE, states.shape),
                                       atol=2e-14, rtol=0)
            np.testing.assert_allclose(means, 0.0, atol=2e-15, rtol=0)
        if name == "onset-pulse":
            np.testing.assert_allclose(states[0], REST_STATE, atol=2e-14, rtol=0)
            assert abs(means[0]) < 2e-15
            assert states[1, 0] > 0 and means[1] < -1e-11
            if substeps == 2:
                final_readout = bold_fraction(expected_states[1], parameters)
                assert abs(means[1] - final_readout) > 1e-10
                assert abs(means[1] - expected_means[1]) < 1e-12
                print(f"  onset mean={means[1]:.12e}; endpoint only={final_readout:.12e}")

    if name not in ("rest", "onset-pulse"):
        # Only compare integrator errors on matched grids. Coarser right-endpoint
        # quadrature itself has a different target even with an exact ODE solve.
        assert np.all(errors[1] < errors[0] / 10.0)
        assert np.all(errors[2] < errors[1] / 10.0)
        quadrature_delta = np.max(np.abs(reference_means[1] - reference_means[2]))
        print(f"  oracle-only 5 vs 2.5 ms right-grid mean delta: {quadrature_delta:.3e}")
    return errors[1]


def main(executable):
    i = np.arange(2000)
    varied = 0.25 * np.sin(i * 0.031) + 0.15 * np.cos(i * 0.083) + 0.1 * (-1.0) ** i
    # Align the held-drive signs against the damped flow impulse kernel
    # at the end of the run, to stress minimum flow beyond a negative step.
    p = NOMINAL_PARAMETERS
    omega = np.sqrt(1 / p.flow_feedback_s2 - (0.5 / p.signal_decay_s) ** 2)
    lag_s = (len(i) - i - 0.5) * (DRIVE_WINDOW_MS / 1000)
    reversal = np.where(np.sin(omega * lag_s) > 0.0, -0.5, 0.5)
    unlike = replace(NOMINAL_PARAMETERS, signal_decay_s=0.48, flow_feedback_s2=0.55,
                     transit_time_s=0.60, grubb_exponent=0.25, resting_extraction=0.32,
                     intravascular_ratio=1.15, echo_time_s=0.03, resting_blood_volume=0.025)
    errors = [
        check_case(executable, "rest", np.zeros(200)),
        check_case(executable, "onset-pulse", np.r_[0.0, 0.5, np.zeros(28)]),
        check_case(executable, "positive-step", np.where(i < 50, 0.0, 0.5)),
        check_case(executable, "negative-step", np.where(i < 50, 0.0, -0.5)),
        check_case(executable, "varied", varied),
        check_case(executable, "flow-reversal", reversal),
        check_case(executable, "perturbed", 2.0 * varied, unlike, (0.15, 0.9, 1.12, 0.85)),
    ]
    maximum = np.max(errors, axis=0)
    print("5 ms grid max absolute errors [s,f,v,q,BOLD]:", maximum)
    # Absolute regression budgets for these fixtures, not a public envelope.
    budget = np.array([1e-10, 1e-10, 3e-9, 3e-9, 1e-10])
    assert np.all(maximum < budget), (maximum, budget)
    print("PASS: seven matched-oracle trajectories on three grids, plus native failure checks")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
