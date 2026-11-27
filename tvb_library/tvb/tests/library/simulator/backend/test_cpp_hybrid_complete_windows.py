"""The staged policy is tested on actual native global output, not Python lowering."""

import numpy as np
import pytest

from tvb.simulator.backend.cpp_hybrid import _cpp_hybrid


DT = 0.125
INITIAL = np.array([2, 10, 40], dtype=np.float32)
DELTA = np.array([1, 3, -2], dtype=np.float32)
WEIGHTS = np.array([[0.25, 0.75, 0], [0, 0, 1]], dtype=np.float32)
GATHER = [dict(n_voi=2, n_node=3)]


def _sim(width=1, k=1, dt=DT):
    sim = _cpp_hybrid.Sim(width)
    sim.set_opts(0, dt, 1)
    # Native Linear (model id 4), zero drift; explicit per-step increments.
    sim.add_subnet(3, 1, 1, 1, 4, 0, 1, k=k, dt=k * dt)
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


def _run_pending(sim, nstep, chunk=1):
    noise = np.broadcast_to(DELTA.reshape(1, 1, 3, 1, 1),
                            (nstep, 1, 3, 1, 1)).copy()
    sim.run(nstep, chunk, noise, need_tavg=False, need_ctavg=False)


def _run(sim, nstep, chunk=1):
    _run_pending(sim, nstep, chunk)
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
        target.set_global_monitors(GATHER, [])
        target.set_voi_specs(0, [1, 0, 0, 0, 0, 0])
        target.set_global_monitors(GATHER, _specs())
    with pytest.raises(RuntimeError, match="global complete window:"):
        target.set_global_monitor_states(states)


@pytest.mark.parametrize("field, value", [
    ("interim", [1.0]), ("stock", [1.0]), ("bw_state", [1.0]),
    ("interim_count", 1), ("stock_pos", 1), ("accum_count", 2),
])
def test_native_complete_window_rejected_restore_preserves_staged_set(field, value):
    sim = _armed()
    _run_pending(sim, 5)  # pending full window and one open sample per monitor
    before = sim.get_global_monitor_states()
    candidate = sim.get_global_monitor_states()
    candidate[0]["accum"] += 1.0  # valid, observably different earlier record
    candidate[1][field] = value
    with pytest.raises(RuntimeError, match="global complete window:"):
        sim.set_global_monitor_states(candidate)
    np.testing.assert_equal(sim.get_global_monitor_states(), before)
    _assert_streams(sim.take_global_monitor_outputs(), _window_reference(5, 4))


@pytest.mark.parametrize("change, error", [
    ({"node_offset": 4}, "complete, disjoint node gather"),
    ({"node_indices": [0, 0, 2]}, "complete, disjoint node gather"),
    ({"voi_offset": 1}, "incompatible gather shape"),
    ({"n_node": 2}, "incompatible gather shape"),
])
def test_native_complete_window_validates_gather_before_monitor_configuration(change, error):
    sim = _armed()
    _run_pending(sim, 5)
    before = sim.get_global_monitor_states()
    # Invalid period verifies that a bad gather is rejected before monitor
    # configuration. All candidate dimensions are deliberately small.
    with pytest.raises(RuntimeError, match=error):
        sim.set_global_monitors([dict(GATHER[0], **change)], _specs(period=0))
    np.testing.assert_equal(sim.get_global_monitor_states(), before)
    _assert_streams(sim.take_global_monitor_outputs(), _window_reference(5, 4))
    _assert_streams(_run(sim, 3), _window_reference(4, 4, INITIAL + 4 * DELTA, origin=4))


def test_native_regional_windows_guard_source_clock_and_topology_until_detached():
    sim = _armed()
    with pytest.raises(RuntimeError, match="global complete window: detach before changing observed rows"):
        sim.set_voi(0, [0])  # reject a shrink before the observed buffer is allocated
    _run_pending(sim, 5)
    before = sim.get_global_monitor_states()
    with pytest.raises(RuntimeError, match="global complete window: detach before changing observed rows"):
        sim.set_voi_specs(0, [1, 0, 0, 0, 0, 0])  # same extent, different identity
    with pytest.raises(RuntimeError, match="global complete window: detach before changing the clock"):
        sim.set_opts(0, 2 * DT, 1)
    with pytest.raises(RuntimeError, match="global complete window: detach before adding a subnet"):
        sim.add_subnet(3, 1, 1, 1, 4, 0, 1, dt=DT)
    sim.set_voi_specs(0, [0, 0, 0, 1, 0, 0])  # identical rows and clock remain valid
    sim.set_opts(0, DT, 3)
    np.testing.assert_equal(sim.get_global_monitor_states(), before)
    _assert_streams(sim.take_global_monitor_outputs(), _window_reference(5, 4))

    sim.set_global_monitors(GATHER, [])
    sim.set_voi(0, [0])
    gather = [dict(n_voi=1, n_node=3)]
    sim.set_global_monitors(gather, [dict(_specs()[0], voi=[0])])
    times, values = _run(sim, 4)[0]
    np.testing.assert_array_equal(times, [7 * DT])
    np.testing.assert_array_equal(values[0, 0, :, 0], INITIAL + 7.5 * DELTA)
    sim.set_global_monitors(gather, [])
    sim.set_opts(0, 2 * DT, 1)
    sim.add_subnet(3, 1, 1, 1, 4, 0, 1, dt=2 * DT)


def test_native_regional_window_checks_indices_before_narrowing():
    sim = _sim()
    spec = _specs()[0]
    with pytest.raises(RuntimeError, match="invalid observed VOI"):
        sim.set_global_monitors(GATHER, [dict(spec, voi=np.array([2**32], dtype=np.int64))])
    with pytest.raises(RuntimeError, match="invalid node gather index"):
        sim.set_global_monitors([dict(GATHER[0], node_indices=np.array([2**32, 1, 2], dtype=np.int64))],
                                [spec])
    gather = [dict(GATHER[0], node_indices=np.array([2, 0, 1], dtype=np.int64))]
    sim.set_global_monitors(gather, [dict(spec, voi=np.array([1, 0], dtype=np.int64))])
    times, values = _window_reference(4, 4)[0]
    _assert_streams(_run(sim, 4), [(times, values[:, :, [1, 2, 0], :])])


def test_native_complete_window_rejects_tick_wrap_before_integration_or_output_clear():
    sim = _armed(_specs(period=2))
    tick_max = (1 << 64) - 1
    sim.set_t_abs(tick_max - 3)
    sim.monitor_begin_run()
    _run_pending(sim, 2)
    before = sim.get_global_monitor_states()
    model_before = sim.get_subnet_state(0).copy()
    assert sim.t_abs() == tick_max - 1
    with pytest.raises(RuntimeError, match="run exceeds remaining tick range"):
        _run_pending(sim, 2)
    assert sim.t_abs() == tick_max - 1
    np.testing.assert_array_equal(sim.get_subnet_state(0), model_before)
    np.testing.assert_equal(sim.get_global_monitor_states(), before)
    midpoint = np.array([((tick_max - 1) - 1) * DT])
    _assert_streams(sim.take_global_monitor_outputs(),
                    [(midpoint, x) for _, x in _window_reference(2, 2)])
    _run_pending(sim, 1)  # exactly the remaining range is allowed
    assert sim.t_abs() == tick_max


def test_native_complete_window_rejects_out_of_scope_attachments():
    with pytest.raises(RuntimeError, match="width 1"):
        _sim(width=4).set_global_monitors(GATHER, _specs())
    with pytest.raises(RuntimeError, match="one regional clock"):
        _sim(k=2).set_global_monitors(GATHER, _specs())
    with pytest.raises(RuntimeError, match="supported global monitor"):
        _sim().set_monitors(0, _specs()[:1])


GAIN = np.array([[1, -2, 0.5], [0.25, 3, -1], [0, 0, 0], [-1, 0, 2]], dtype=np.float32)
REFERENCE = np.array([0.25, 0.25, 0, 0.25], dtype=np.float32)


def _projection_spec(**changes):
    return dict(dict(slot=0, kind=8, complete_window=True, dt=DT, istep=4,
                     voi=np.array([1, 0], dtype=np.int64), gain=GAIN, ref_vec=REFERENCE),
                **changes)


def _projection_oracle(nstep, period=4, *, origin=0, gain=GAIN, ref=REFERENCE):
    closed = nstep // period
    steps = origin + np.arange(1, closed * period + 1)
    regional_rows = INITIAL + steps[:, None] * DELTA
    means = regional_rows.reshape(closed, period, 3).mean(axis=1)
    sensors = means @ gain.T
    if ref is not None:
        sensors -= (sensors @ ref)[:, None]
    # The selected observed row order is difference, then direct observation.
    values = np.stack((np.zeros_like(sensors), sensors), axis=1)[..., None]
    times = (origin + period * (np.arange(closed) + 0.5)) * DT
    return [(times, values.astype(np.float32))]


@pytest.mark.parametrize("ref", [None, np.array([0, 1, 0, 0.0]), REFERENCE],
                         ids=["no-reference", "named", "masked-average"])
@pytest.mark.parametrize("period, chunk", [(1, 8), (3, 5), (4, 1), (4, 3), (4, 8)])
def test_native_projection_window_samples_post_step_voi_order_and_sensor_reference(ref, period, chunk):
    sim = _armed([_projection_spec(istep=period, ref_vec=ref)])
    _assert_streams(_run(sim, 9, chunk), _projection_oracle(9, period, ref=ref))
    assert sim.get_global_monitor_states()[0]["accum_count"] == 9 % period


def test_native_projection_window_drains_resume_and_fresh_operator_reset():
    source = _armed([_projection_spec()])
    first = _run(source, 5, 3)
    _assert_streams(first, _projection_oracle(5))
    _assert_streams(_run(source, 1), _projection_oracle(0))  # drain with no output
    before, = source.get_global_monitor_states()
    assert before["m_step"] == 6 and before["run_start"] == 0 and before["accum_count"] == 2
    assert "window_config" in before
    resumed = _armed([_projection_spec()])
    resumed.set_subnet_state(0, source.get_subnet_state(0))
    resumed.set_subnet_buf(0, source.get_subnet_buf(0))
    resumed.set_t_abs(6)
    resumed.set_global_monitor_states([before])
    second = _run(resumed, 5, 8)
    _assert_streams(second, _run(source, 5, 2))
    _assert_streams([tuple(np.concatenate((a, b)) for a, b in zip(first[0], second[0]))],
                    _projection_oracle(11))
    # Fresh attachment discards the three samples open at tick 11 and anchors
    # the replacement operator at 11 rather than the previous phase or tick 0.
    changed_gain = GAIN.copy()
    changed_gain[1, 0] = 5
    resumed.set_global_monitors(GATHER, [_projection_spec(gain=changed_gain)])
    _assert_streams(_run(resumed, 4, 3), _projection_oracle(4, origin=11, gain=changed_gain))
    _run(resumed, 2)
    resumed.monitor_begin_run()
    _assert_streams(_run(resumed, 4), _projection_oracle(4, origin=17, gain=changed_gain))


@pytest.mark.parametrize("change", ["gain", "ref_vec", "voi", "source_rows", "gather", "period",
                                    "missing", "historical", "duplicate", "count", "nan"])
def test_native_projection_window_rejected_restore_preserves_partial_state_and_output(change):
    spec = _projection_spec()
    gather = GATHER
    if change == "gain":
        spec["gain"] = GAIN * 2
    elif change == "ref_vec":
        spec["ref_vec"] = np.array([0, 1, 0, 0], dtype=np.float32)
    elif change == "voi":
        spec["voi"] = [0, 1]
    elif change == "period":
        spec["istep"] = 5
    elif change == "gather":
        gather = [dict(GATHER[0], node_indices=[0, 1, 2])]
    source = _armed([_projection_spec()])
    _run(source, 6, 3)
    candidate = source.get_global_monitor_states()
    if change == "missing":
        candidate.clear()
    elif change == "historical":
        # A genuine old native projection record has no complete-window identity.
        old = _armed([_projection_spec(complete_window=False)])
        _run(old, 6)
        candidate = old.get_global_monitor_states()
        assert "window_config" not in candidate[0]
    elif change == "duplicate":
        candidate.append(candidate[0])
    elif change == "count":
        candidate[0]["accum_count"] = 1
    elif change == "nan":
        candidate[0]["accum"][0] = np.nan
    target = _sim()
    if change == "source_rows":
        target.set_voi_specs(0, [1, 0, 0, 0, 0, 0])
    target.set_global_monitors(gather, [spec])
    control = _sim()
    if change == "source_rows":
        control.set_voi_specs(0, [1, 0, 0, 0, 0, 0])
    control.set_global_monitors(gather, [spec])
    _run_pending(target, 6)
    expected = _run(control, 6)
    before = target.get_global_monitor_states()
    with pytest.raises(RuntimeError, match="global complete window"):
        target.set_global_monitor_states(candidate)
    np.testing.assert_equal(target.get_global_monitor_states(), before)
    _assert_streams(target.take_global_monitor_outputs(), expected)


def _projection_with_subsamples():
    return [_projection_spec(), dict(slot=1, kind=2, istep=2, dt=DT, voi=[0]),
            dict(slot=2, kind=2, istep=3, dt=DT, voi=[0])]


@pytest.mark.parametrize("field, value, error", [
    ("n_out_space", 1, "geometry"), ("stock", [1.0], "buffers"), ("slot", 9, "unmatched"),
])
def test_native_projection_window_rejected_sibling_preserves_state_and_pending_rows(field, value, error):
    specs = _projection_with_subsamples()
    target, control = _armed(specs), _armed(specs)
    _run_pending(target, 6)
    expected = _run(control, 6)
    before = target.get_global_monitor_states()
    candidate = target.get_global_monitor_states()
    candidate[0]["accum"] += 1.0  # valid change to staged Projection
    candidate[1]["run_start"] += 1  # valid change to first SubSample
    candidate[2][field] = value
    with pytest.raises(RuntimeError, match=f"global complete window projection: .*{error}"):
        target.set_global_monitor_states(candidate)
    np.testing.assert_equal(target.get_global_monitor_states(), before)
    assert all(times.size for times, _ in expected)
    _assert_streams(target.take_global_monitor_outputs(), expected)


@pytest.mark.parametrize("field, value", [("width", 2), ("n_out_space", 1)], ids=["width", "space"])
def test_native_regional_window_restore_is_transactional_with_point_siblings(field, value):
    specs = [_specs()[0], dict(slot=1, kind=2, istep=2, dt=DT, voi=[0]),
             dict(slot=2, kind=2, istep=3, dt=DT, voi=[0])]
    sim = _armed(specs)
    _run_pending(sim, 6)
    before = sim.get_global_monitor_states()
    candidate = sim.get_global_monitor_states()
    candidate[0]["accum"] += 1.0
    candidate[1]["run_start"] += 1  # valid legacy change before the malformed record
    candidate[2][field] = value
    with pytest.raises(RuntimeError, match="global complete window: incompatible sibling snapshot geometry"):
        sim.set_global_monitor_states(candidate)
    # Assert rejection and unchanged geometry before any drain.
    np.testing.assert_equal(sim.get_global_monitor_states(), before)
    expected = _run(_armed(specs), 6)
    assert all(times.size for times, _ in expected)
    _assert_streams(sim.take_global_monitor_outputs(), expected)

    _run_pending(sim, 6)  # requeue at least one output for every attachment
    valid = sim.get_global_monitor_states()
    valid[1]["run_start"] += 1
    sim.set_global_monitor_states(valid[::-1])
    np.testing.assert_equal(sim.get_global_monitor_states(), valid)
    assert all(times.size == values.size == 0 for times, values in sim.take_global_monitor_outputs())


def test_native_legacy_only_global_restore_retains_counter_defaults():
    sim = _armed([dict(slot=0, kind=2, istep=3, dt=DT, voi=[0])])
    _run_pending(sim, 4)
    states = sim.get_global_monitor_states()
    assert states[0]["m_step"] == 4
    for field in ("m_step", "run_start", "accum_count", "interim_count", "stock_pos"):
        states[0].pop(field)
    sim.set_global_monitor_states(states)
    assert sim.get_global_monitor_states()[0]["m_step"] == 0


def test_native_projection_window_restores_reordered_subsamples_and_continues():
    specs = _projection_with_subsamples()
    source, target = _armed(specs), _armed(specs)
    first = _run(source, 6)
    states = source.get_global_monitor_states()
    target.set_subnet_state(0, (INITIAL + 16).reshape(1, 3, 1, 1))
    _run_pending(target, 6)
    target.set_subnet_state(0, source.get_subnet_state(0))
    target.set_subnet_buf(0, source.get_subnet_buf(0))
    target.set_t_abs(source.t_abs())
    target.set_global_monitor_states(states[::-1])
    np.testing.assert_equal(target.get_global_monitor_states(), states)
    assert all(times.size == values.size == 0 for times, values in target.take_global_monitor_outputs())
    tail = _run(target, 5, chunk=3)
    _assert_streams([tuple(np.concatenate((a, b)) for a, b in zip(head, rest))
                     for head, rest in zip(first, tail)], _run(_armed(specs), 11))


@pytest.mark.parametrize("stored_dt, spec_dt", [(0.0, 1e-100), (-DT, DT), (np.inf, DT)],
                         ids=["zero", "negative", "nonfinite"])
def test_native_projection_window_rejects_invalid_stored_neural_clock(stored_dt, spec_dt):
    with pytest.raises(RuntimeError, match="invalid stored neural clock"):
        _sim(dt=stored_dt).set_global_monitors(GATHER, [_projection_spec(dt=spec_dt)])


@pytest.mark.parametrize("change, error", [
    ({"gain": None}, "preconfigured gain"),
    ({"gain": np.ones(3)}, "invalid gain"),
    ({"gain": np.empty((0, 3))}, "invalid gain"),
    ({"gain": np.ones((4, 2))}, "invalid gain"),
    ({"gain": np.full((4, 3), np.nan)}, "nonfinite gain"),
    ({"gain": np.full((4, 3), np.inf)}, "nonfinite gain"),
    ({"ref_vec": np.ones((1, 4))}, "invalid ref_vec"),
    ({"ref_vec": np.ones(3)}, "ref_vec length"),
    ({"ref_vec": np.full(4, np.nan)}, "nonfinite ref_vec"),
    ({"voi": np.array([2**32], dtype=np.int64)}, "invalid observed VOI"),
    ({"voi": np.array([2**64 - 1], dtype=np.uint64)}, "invalid observed VOI"),
    ({"voi": [0.0]}, "invalid observed VOI"),
    ({"istep": 0}, "invalid period"),
    ({"dt": 2 * DT}, "invalid period"),
    ({"dt": 1e100}, "invalid period"),
    ({"obsnoise": object()}, "observation noise is unsupported"),
])
def test_native_projection_window_validates_attachment_without_replacing_pending_state(change, error):
    sim = _armed([_projection_spec()])
    _run_pending(sim, 5)
    before = sim.get_global_monitor_states()
    with pytest.raises(RuntimeError, match=error):
        sim.set_global_monitors(GATHER, [_projection_spec(**change)])
    np.testing.assert_equal(sim.get_global_monitor_states(), before)
    _assert_streams(sim.take_global_monitor_outputs(), _projection_oracle(5))


def test_native_projection_window_rejects_nonlocal_geometry_and_invalid_source():
    spec = [_projection_spec()]
    with pytest.raises(RuntimeError, match="width 1"):
        _sim(width=4).set_global_monitors(GATHER, spec)
    with pytest.raises(RuntimeError, match="clock and mode"):
        _sim(k=2).set_global_monitors(GATHER, spec)
    with pytest.raises(RuntimeError, match="supported global monitor"):
        _sim().set_monitors(0, spec)
    sim = _sim()
    with pytest.raises(RuntimeError, match="local node order"):
        sim.set_global_monitors([dict(GATHER[0], node_indices=[1, 0, 2])], spec)
    with pytest.raises(RuntimeError, match="invalid node gather index"):
        sim.set_global_monitors([dict(GATHER[0], node_indices=np.array([2**32, 1, 2]))], spec)
    sim.set_voi_specs(0, [0, 1, 0, 1, 0, 0])
    with pytest.raises(RuntimeError, match="invalid observed VOI source"):
        sim.set_global_monitors(GATHER, spec)
    sim.add_subnet(3, 1, 1, 1, 4, 0, 1, dt=DT)
    with pytest.raises(RuntimeError, match="one regional subnet"):
        sim.set_global_monitors(GATHER * 2, spec)


def test_native_projection_window_guards_live_source_and_clock_until_detached():
    sim = _armed([_projection_spec()])
    _run_pending(sim, 5)
    before = sim.get_global_monitor_states()
    with pytest.raises(RuntimeError, match="detach before changing observed rows"):
        sim.set_voi(0, [0, 0])
    with pytest.raises(RuntimeError, match="detach before changing observed rows"):
        sim.set_voi_specs(0, [1, 0, 0, 0, 0, 0])
    with pytest.raises(RuntimeError, match="detach before changing the clock"):
        sim.set_opts(0, 2 * DT, 1)
    with pytest.raises(RuntimeError, match="detach before adding a subnet"):
        sim.add_subnet(3, 1, 1, 1, 4, 0, 1, dt=DT)
    sim.set_voi_specs(0, [0, 0, 0, 1, 0, 0])  # same source identity is allowed
    sim.set_opts(0, DT, 3)  # same clock is allowed
    np.testing.assert_equal(sim.get_global_monitor_states(), before)
    _assert_streams(sim.take_global_monitor_outputs(), _projection_oracle(5))
    _assert_streams(_run(sim, 3), _projection_oracle(4, origin=4))
    sim.set_global_monitors(GATHER, [])
    sim.set_voi(0, [0, 0])


@pytest.mark.parametrize("compact", [False, True], ids=["dense", "compact"])
@pytest.mark.parametrize("change", [
    "bool-slot", "bool-width", "bool-accum-count", "bool-interim-count",
    "bool-stock-pos", "bool-spatial-compact", "float32-accum", "list-accum",
    "scalar-eq", "container-eq",
])
def test_native_regional_window_restore_requires_strict_schema(compact, change):
    from copy import deepcopy

    sim = _armed(_specs(compact=compact))
    _run_pending(sim, 5)
    before = sim.get_global_monitor_states()
    candidate = deepcopy(before)
    spatial_state = candidate[2]
    comparisons = []

    class EqualScalar(int):
        def __eq__(self, other):
            comparisons.append(other)
            return True

    class EqualContainer(tuple):
        def __eq__(self, other):
            comparisons.append(other)
            return True

    if change == "bool-slot":
        candidate[0]["slot"] = False  # False == 0 in Python
    elif change == "bool-width":
        spatial_state["width"] = True  # True == 1 in Python
    elif change == "bool-accum-count":
        spatial_state["accum_count"] = True  # 5 % 4 == 1
    elif change == "bool-interim-count":
        spatial_state["interim_count"] = False
    elif change == "bool-stock-pos":
        spatial_state["stock_pos"] = False
    elif change == "bool-spatial-compact":
        cfg = list(spatial_state["window_config"])
        cfg[4] = int(cfg[4])  # bool spatial_compact must not accept int 0/1
        spatial_state["window_config"] = tuple(cfg)
    elif change == "float32-accum":
        spatial_state["accum"] = spatial_state["accum"].astype(np.float32)
    elif change == "list-accum":
        spatial_state["accum"] = spatial_state["accum"].tolist()
    elif change == "scalar-eq":
        cfg = list(spatial_state["window_config"])
        cfg[0] = EqualScalar(cfg[0])
        spatial_state["window_config"] = tuple(cfg)
    elif change == "container-eq":
        spatial_state["window_config"] = EqualContainer(spatial_state["window_config"])

    with pytest.raises(RuntimeError, match="global complete window:"):
        sim.set_global_monitor_states(candidate)
    assert not comparisons
    np.testing.assert_equal(sim.get_global_monitor_states(), before)
    _assert_streams(sim.take_global_monitor_outputs(), _window_reference(5, 4))
