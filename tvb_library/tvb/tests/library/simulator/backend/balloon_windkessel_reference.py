"""Test-side oracle for a proposed Balloon–Windkessel design, not public API.

Flow and balloon dynamics follow Friston et al. (2000),
https://doi.org/10.1006/nimg.2000.0630. BOLD uses the revised nonlinear
readout (Eq. 12) of Stephan et al. (2007),
https://doi.org/10.1016/j.neuroimage.2007.07.040.
No TVB monitor or native code is imported.
"""

from dataclasses import dataclass, fields

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class BalloonParameters:
    """Nominal fixture parameters, not population estimates or API defaults."""

    signal_decay_s: float = 0.65
    flow_feedback_s2: float = 0.41
    transit_time_s: float = 0.98
    grubb_exponent: float = 0.32
    resting_extraction: float = 0.4
    intravascular_ratio: float = 0.5
    frequency_offset_hz: float = 40.3
    relaxation_rate_per_s: float = 25.0
    echo_time_s: float = 0.04
    resting_blood_volume: float = 0.04

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{field.name} must be finite and positive")
        for name in ("resting_extraction", "resting_blood_volume"):
            if getattr(self, name) >= 1:
                raise ValueError(f"{name} must be less than one")
        if self.grubb_exponent > 1:
            raise ValueError("grubb_exponent must be at most one")


NOMINAL_PARAMETERS = BalloonParameters()
REST_STATE = (0.0, 1.0, 1.0, 1.0)  # s, f, v, q


def _check_state(state):
    if not np.all(np.isfinite(state)) or np.any(state[..., 1:] <= 0):
        raise ValueError("expected finite [s, f, v, q] with positive f, v, q")


def bold_fraction(state, parameters=NOMINAL_PARAMETERS):
    """Compute instantaneous fractional ΔS/S (multiply by 100 for percent)."""
    state = np.asarray(state, dtype=np.float64)
    _check_state(state)
    _, _, volume, deoxy = np.moveaxis(state, -1, 0)
    p = parameters
    extraction_echo = p.resting_extraction * p.echo_time_s
    k1 = 4.3 * p.frequency_offset_hz * extraction_echo
    k2 = p.intravascular_ratio * p.relaxation_rate_per_s * extraction_echo
    k3 = 1.0 - p.intravascular_ratio
    return p.resting_blood_volume * (
        k1 * (1.0 - deoxy) + k2 * (1.0 - deoxy / volume) + k3 * (1.0 - volume)
    )


def _exact_step_count(interval_ms, step_ms):
    if not all(np.isfinite(x) and x > 0 for x in (interval_ms, step_ms)):
        raise ValueError("clock intervals must be finite and positive")
    ratio = interval_ms / step_ms
    count = round(ratio)
    if count < 1 or not np.isclose(ratio, count, rtol=0, atol=1e-9):
        raise ValueError("expected commensurate clocks (a positive integer step count)")
    return count


@dataclass(frozen=True)
class BalloonTrajectory:
    states: np.ndarray  # (hemodynamic checkpoints including t=0, [s, f, v, q])
    neural_dt_ms: float
    hemo_dt_ms: float
    parameters: BalloonParameters

    @property
    def times_ms(self):
        return np.arange(len(self.states), dtype=np.float64) * self.hemo_dt_ms

    def windowed_bold(self, period_ms):
        """Return (midpoints_ms, release_times_ms, BOLD means) for full windows."""
        _exact_step_count(period_ms, self.neural_dt_ms)
        steps = _exact_step_count(period_ms, self.hemo_dt_ms)
        complete = (len(self.states) - 1) // steps
        # Right hemodynamic endpoints, without the initial t=0 point or the tail.
        readouts = bold_fraction(self.states[1:complete * steps + 1], self.parameters)
        ends = np.arange(1, complete + 1, dtype=np.float64) * period_ms
        return ends - 0.5 * period_ms, ends, readouts.reshape(complete, steps).mean(axis=1)


def solve_held_drive(drive, *, neural_dt_ms, hemo_dt_ms, parameters=NOMINAL_PARAMETERS,
                     initial_state=REST_STATE):
    """Integrate one stream; drive[i] (s⁻²) holds on [i*dt_n, (i+1)*dt_n).

    The drive has already been baseline-subtracted and scaled. The hemodynamic
    clock sets checkpoints and an upper bound on solver steps, not the amount
    of physical time advanced per neural sample. The solver lands exactly at
    drive discontinuities instead of integrating through them.
    """
    substeps = _exact_step_count(neural_dt_ms, hemo_dt_ms)
    drive = np.asarray(drive, dtype=np.float64)
    if drive.ndim != 1 or not np.all(np.isfinite(drive)):
        raise ValueError("expected a finite one-dimensional held drive")
    initial_state = np.asarray(initial_state, dtype=np.float64)
    if initial_state.shape != (4,):
        raise ValueError("expected initial state [s, f, v, q]")
    _check_state(initial_state)
    states = np.empty((len(drive) * substeps + 1, 4), dtype=np.float64)
    states[0] = initial_state
    checkpoint_s = np.arange(len(states), dtype=np.float64) * (hemo_dt_ms / 1000.0)
    p = parameters
    log_unextracted = np.log1p(-p.resting_extraction)

    def dynamics(_t, state, held_drive):
        _check_state(state)
        signal, inflow, volume, deoxy = state
        outflow = volume ** (1.0 / p.grubb_exponent)
        extraction = -np.expm1(log_unextracted / inflow)
        return (
            held_drive - signal / p.signal_decay_s - (inflow - 1) / p.flow_feedback_s2,
            signal,
            (inflow - outflow) / p.transit_time_s,
            (inflow * extraction / p.resting_extraction - outflow * deoxy / volume)
            / p.transit_time_s,
        )

    if drive.size:
        # Merge equal adjacent holds, but split at every actual change in drive.
        boundaries = np.r_[0, np.flatnonzero(drive[1:] != drive[:-1]) + 1, len(drive)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            left, right = start * substeps, stop * substeps
            solution = solve_ivp(
                dynamics, (checkpoint_s[left], checkpoint_s[right]), states[left],
                args=(drive[start],), method="DOP853", rtol=2e-12, atol=2e-14,
                max_step=hemo_dt_ms / 1000.0, t_eval=checkpoint_s[left + 1:right + 1],
            )
            if not solution.success:
                raise ValueError(f"hemodynamic solve failed: {solution.message}")
            states[left + 1:right + 1] = solution.y.T
    _check_state(states)
    return BalloonTrajectory(states, neural_dt_ms, hemo_dt_ms, parameters)
