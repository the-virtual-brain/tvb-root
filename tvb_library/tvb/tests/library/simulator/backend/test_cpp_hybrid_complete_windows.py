"""The staged policy is tested on actual native global output, not Python lowering."""

import numpy as np
import pytest

from tvb.simulator.backend.cpp_hybrid import _cpp_hybrid


DT = 0.125
INITIAL = np.array([2, 10, 40], dtype=np.float32)
DELTA = np.array([1, 3, -2], dtype=np.float32)
WEIGHTS = np.array([[0.25, 0.75, 0], [0, 0, 1]], dtype=np.float32)
GATHER = [dict(n_voi=2, n_node=3)]


def _sim(width=1, k=1):
    sim = _cpp_hybrid.Sim(width)
    sim.set_opts(0, DT, 1)
    # Native Linear (model id 4), zero drift; explicit per-step increments.
    sim.add_subnet(3, 1, 1, 1, 4, 0, 1, k=k, dt=k * DT)
    sim.set_subnet_params(0, np.zeros((3, 1, width), dtype=np.float32))
    sim.set_voi_specs(0, [0, 0, 0, 1, 0, 0])  # x, x-x
    sim.set_subnet_state(0, np.broadcast_to(INITIAL.reshape(1, 3, 1, 1),
                                           (1, 3, 1, width)).copy())
    return sim


def _specs(period=4, weights=WEIGHTS, *, compact=False):
    spatial = {"spatial_mean": weights}
    if compact:
        spatial = dict(spatial_area_idx=[0, 0, 1],
                       spatial_scale=[0.25, 0.75, 1.0], n_areas=2)
    return [dict(slot=i, kind=kind, istep=period, dt=DT, voi=[1, 0],
                 complete_window=True, **(spatial if kind == 7 else {}))
            for i, kind in enumerate((4, 6, 7))]


def _armed(specs=None):
    sim = _sim()
    sim.set_global_monitors(GATHER, _specs() if specs is None else specs)
    sim.monitor_begin_run()
    return sim


def _run(sim, nstep, chunk=1):
    noise = np.broadcast_to(DELTA.reshape(1, 1, 3, 1, 1),
                            (nstep, 1, 3, 1, 1)).copy()
    sim.run(nstep, chunk, noise, need_tavg=False, need_ctavg=False)
    return sim.take_global_monitor_outputs()


def _assert_streams(actual, expected):
    assert len(actual) == len(expected)
    for (times, values), (ref_times, ref_values) in zip(actual, expected):
        np.testing.assert_array_equal(times, ref_times)
        np.testing.assert_array_equal(values, ref_values)


def _window_reference(nstep, period, initial=INITIAL, origin=0):
    closed = nstep // period
    times = (origin + np.arange(1, closed + 1) * period - period / 2) * DT
    # For this ramp the arithmetic mean equals the ramp at the mean of the
    # included *post-step* indices; the timestamps above are interval centres.
    mean_steps = np.arange(closed) * period + (period + 1) / 2
    regional = initial + mean_steps[:, None] * DELTA
    values = (regional, regional.mean(axis=1, keepdims=True), regional @ WEIGHTS.T)
    return [(times, np.stack((np.zeros_like(x), x), axis=1)[..., None].astype(np.float32))
            for x in values]


@pytest.mark.parametrize("period", [1, 3, 4])
@pytest.mark.parametrize("compact", [False, True], ids=["dense", "compact"])
def test_native_complete_windows_emit_values_and_interval_centres(period, compact):
    sim = _armed(_specs(period, compact=compact))
    _assert_streams(_run(sim, 9, chunk=5), _window_reference(9, period))


def test_native_point_monitors_keep_post_step_sample_times_and_values():
    sim = _armed(_specs() + [dict(slot=3, kind=1, dt=DT),
                            dict(slot=4, kind=1, dt=DT, voi=[0]),
                            dict(slot=5, kind=2, dt=DT, istep=4, voi=[0])])
    steps = np.arange(1, 10)
    times = steps * DT
    x = (INITIAL + steps[:, None] * DELTA).astype(np.float32)
    points = [(times, np.stack((x, np.zeros_like(x)), axis=1)[..., None]),
              (times, x[:, None, :, None]),
              (times[3::4], x[3::4, None, :, None])]
    _assert_streams(_run(sim, 9, chunk=5), _window_reference(9, 4) + points)


@pytest.mark.parametrize("restore", [False, True], ids=["same-sim", "native-restore"])
def test_native_complete_windows_preserve_open_window_across_calls(restore):
    full = _run(_armed(), 11, chunk=7)
    _assert_streams(full, _window_reference(11, 4))

    split = _armed()
    first = _run(split, 6, chunk=5)  # one emitted window, two samples open
    _assert_streams(first, _window_reference(6, 4))
    states = split.get_global_monitor_states()
    assert [s["accum_count"] for s in states] == [2, 2, 2]
    if restore:
        resumed = _armed()
        resumed.set_subnet_state(0, split.get_subnet_state(0))
        resumed.set_subnet_buf(0, split.get_subnet_buf(0))
        resumed.set_t_abs(split.t_abs())
        resumed.set_global_monitor_states(states)
        split = resumed
    second = _run(split, 5, chunk=2)
    _assert_streams([(np.concatenate((t1, t2)), np.concatenate((x1, x2)))
                     for (t1, x1), (t2, x2) in zip(first, second)], full)
    assert [s["accum_count"] for s in split.get_global_monitor_states()] == [3, 3, 3]


def test_native_complete_windows_reset_open_accumulator_and_phase():
    sim = _armed()
    _assert_streams(_run(sim, 3), _window_reference(3, 4))
    sim.monitor_begin_run()  # start a fresh window at tick 3, without rewinding dynamics
    _assert_streams(_run(sim, 4), _window_reference(4, 4, INITIAL + 3 * DELTA, origin=3))
    _run(sim, 2)  # another open window, then rewind both dynamics and monitor
    sim.set_subnet_state(0, INITIAL.reshape(1, 3, 1, 1))
    sim.monitor_begin_run()
    _assert_streams(_run(sim, 9), _window_reference(9, 4))


def test_native_complete_window_restore_rejects_changed_same_shape_spatial_weights():
    sim = _armed()
    _run(sim, 3)
    states = sim.get_global_monitor_states()
    changed = WEIGHTS.copy()
    changed[0, :2] = [0.75, 0.25]  # same shape and sparsity pattern
    sim.set_global_monitors(GATHER, _specs(weights=changed))
    with pytest.raises(RuntimeError, match="incompatible snapshot configuration"):
        sim.set_global_monitor_states(states)
    assert [s["accum_count"] for s in sim.get_global_monitor_states()] == [0, 0, 0]


@pytest.mark.parametrize("change", ["period", "missing", "legacy", "count", "clock", "source_voi"])
def test_native_complete_window_restore_rejects_incompatible_state(change):
    sim = _armed()
    _run(sim, 3)
    states = sim.get_global_monitor_states()
    target = _armed(_specs(period=3 if change == "period" else 4))
    target.set_t_abs(2 if change == "clock" else 3)
    if change == "missing":
        states.pop()
    elif change == "legacy":
        states[0].pop("window_config")
    elif change == "count":
        states[1]["accum_count"] = 1
    elif change == "source_voi":
        target.set_voi_specs(0, [1, 0, 0, 0, 0, 0])
        target.set_global_monitors(GATHER, _specs())
    with pytest.raises(RuntimeError, match="global complete window:"):
        target.set_global_monitor_states(states)


def test_native_complete_window_rejects_out_of_scope_attachments():
    with pytest.raises(RuntimeError, match="width 1"):
        _sim(width=4).set_global_monitors(GATHER, _specs())
    with pytest.raises(RuntimeError, match="one regional clock"):
        _sim(k=2).set_global_monitors(GATHER, _specs())
    with pytest.raises(RuntimeError, match="supported global monitor"):
        _sim().set_monitors(0, _specs()[:1])
