"""Exercise the staged BW stream through the installed native binding only."""

from dataclasses import asdict

import numpy as np
import pytest

from tvb.simulator.backend.cpp_hybrid import _cpp_hybrid
from .balloon_windkessel_reference import NOMINAL_PARAMETERS, solve_averaged_drive


DT = 2.5
BASELINE = 2.0
SCALE = 0.25
GATHER = dict(n_voi=2, n_node=3, node_indices=[2, 0, 1])  # local node -> output region
INITIAL = np.full(3, BASELINE, dtype=np.float32)


def _sim(*, width=1, dt=DT, subnet_dt=None, k=1, modes=1, subnet_count=1):
    sim = _cpp_hybrid.Sim(width)
    sim.set_opts(0, dt, 1)
    for sn in range(subnet_count):
        # Linear model, zero drift: the noise argument is an explicit increment.
        sim.add_subnet(3, 1, 1, 1, 4, 0, modes, k=k,
                       dt=k * dt if subnet_dt is None else subnet_dt)
        sim.set_subnet_params(sn, np.zeros((3, 1, width), dtype=np.float32))
        sim.set_voi_specs(sn, [1, 0, 0, 0, 0, 0])  # x-x, x
        sim.set_subnet_state(sn, np.broadcast_to(INITIAL.reshape(1, 3, 1, 1),
                                                (1, 3, modes, width)).copy())
    return sim


def _spec(**changes):
    return dict(dict(slot=0, kind=10, istep=4, dt=DT, voi=[1],
                     bw_drive_baseline=BASELINE, bw_drive_scale=SCALE,
                     bw_stream_params=asdict(NOMINAL_PARAMETERS)), **changes)


def _armed(*, period_ms=10, gather=GATHER, spec=None, extra=()):
    sim = _sim()
    sim.set_global_monitors([gather], [_spec(istep=int(period_ms / DT)) if spec is None else spec,
                                       *extra])
    sim.monitor_begin_run()
    return sim


def _neural_fixture(windows=60, tail=3):
    # Deliberately asymmetric region means and nonconstant samples within each
    # bin; both point-sampling and pre-step averaging give the wrong drive.
    means = np.resize([[3.0, 0.0, 2.5], [1.25, 3.5, 2.75], [3.25, 1.5, 2.625]],
                      (windows, 3))
    means = np.vstack([INITIAL, means])
    jitter = np.array([0.5, -0.25, 0.125, -0.375])
    obs = (means[:, None, :] + jitter[None, :, None]).reshape(-1, 3)
    return obs[:windows * 4 + tail].astype(np.float32)


def _increments(obs, initial=INITIAL):
    return np.diff(np.vstack([initial, obs]), axis=0).astype(np.float32)


def _run(sim, delta, chunk=3, *, drain=True):
    sim.run(len(delta), chunk, delta[:, None, :, None, None].copy(),
            need_tavg=False, need_ctavg=False)
    if drain:
        return sim.take_global_monitor_outputs()[0]


def _oracle(obs):
    # This source ordering is independently specified, not read from GATHER or
    # from the output of the monitor under test.
    regional_obs = np.column_stack([obs[:, 1], obs[:, 2], obs[:, 0]])
    return solve_averaged_drive(regional_obs, neural_dt_ms=DT, hemo_dt_ms=5,
                                baseline=BASELINE, gain_per_s2_per_observation=SCALE)


def _assert_oracle(stream, reference, period_ms, origin=0):
    times, values = stream
    t, _, bold = reference.windowed_bold(period_ms)
    np.testing.assert_array_equal(times, t + origin)
    assert values.dtype == np.float32
    assert values.shape == (len(t), 1, 3, 1)
    np.testing.assert_allclose(values[:, 0, :, 0], bold, rtol=1e-7, atol=1e-10)


def _assert_stream(actual, expected):
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def _resume(source, *, period_ms):
    resumed = _armed(period_ms=period_ms)
    resumed.set_subnet_state(0, source.get_subnet_state(0))
    resumed.set_subnet_buf(0, source.get_subnet_buf(0))
    resumed.set_t_abs(source.t_abs())
    resumed.set_global_monitor_states(source.get_global_monitor_states())
    return resumed


def test_native_bw_stream_closes_at_20ms_not_17_5ms():
    obs = _neural_fixture(2, 1)
    delta = _increments(obs)
    ref = _oracle(obs)
    np.testing.assert_array_equal(ref.held_drive_per_s2, [[0, 0, 0], [-0.5, 0.125, 0.25]])
    sim = _armed()
    _assert_oracle(_run(sim, delta[:4]), _oracle(obs[:4]), 10)
    times, values = _run(sim, delta[4:7])
    assert times.size == values.size == 0
    state, = sim.get_global_monitor_states()
    assert state["bw_drive_count"] == 3 and state["accum_count"] == 0
    np.testing.assert_array_equal(state["bw_stream_regions"][:, 0],
                                  obs[4:7, [1, 2, 0]].sum(axis=0))
    np.testing.assert_array_equal(state["bw_stream_regions"][:, 2:], np.tile([0, 1, 1, 1], (3, 1)))
    times, values = _run(sim, delta[7:8])
    np.testing.assert_array_equal(times, [15])
    t_ref, _, bold_ref = ref.windowed_bold(10)
    assert t_ref[-1] == 15 and np.all(bold_ref[-1] != 0)
    np.testing.assert_allclose(values[0, 0, :, 0], bold_ref[-1], atol=1e-10, rtol=0)
    assert _run(sim, delta[8:])[0].size == 0  # no trailing flush


@pytest.mark.parametrize("restore", [False, True], ids=["chunked", "restored"])
def test_native_bw_stream_oracle_and_simultaneously_open_windows(restore):
    obs = _neural_fixture()
    delta = _increments(obs)
    ref = _oracle(obs)
    full = _armed(period_ms=30)
    whole = _run(full, delta, 17)
    _assert_oracle(whole, ref, 30)
    sim = _armed(period_ms=30)
    outputs = []
    start = 0
    for end, chunk in [(7, 2), (12, 4), (13, 5), (22, 1), (68, 19), (len(delta), 11)]:
        outputs.append(_run(sim, delta[start:end], chunk))
        snapshot, = sim.get_global_monitor_states()
        assert snapshot["bw_drive_count"] == end % 4
        assert snapshot["accum_count"] == (end % 12 // 4) * 2
        data = snapshot["bw_stream_regions"]
        assert data.shape == (3, 6) and data.dtype == np.float64 and data.nbytes == 144
        for key in ("accum", "interim", "stock", "bw_state"):
            assert snapshot[key].size == 0
        np.testing.assert_array_equal(data[:, 0], obs[end - end % 4:end, [1, 2, 0]].sum(axis=0))
        np.testing.assert_allclose(data[:, 2:], ref.states[end // 4 * 2], rtol=0, atol=3e-9)
        if restore:
            sim = _resume(sim, period_ms=30)
        start = end
    _assert_stream(tuple(np.concatenate(parts) for parts in zip(*outputs)), whole)
    assert sim.take_global_monitor_outputs()[0][0].size == 0


def test_native_bw_stream_fresh_reset_reanchors_both_phases_and_rest_state():
    obs = _neural_fixture(7)
    sim = _armed(period_ms=30)
    _run(sim, _increments(obs[:19]), drain=False)
    before, = sim.get_global_monitor_states()
    assert before["bw_drive_count"] == 3 and before["accum_count"] == 2
    sim.monitor_begin_run()
    assert sim.take_global_monitor_outputs()[0][0].size == 0
    after, = sim.get_global_monitor_states()
    assert after["m_step"] == after["run_start"] == 19
    assert after["bw_drive_count"] == after["accum_count"] == 0
    np.testing.assert_array_equal(after["bw_stream_regions"], np.tile([0, 0, 0, 1, 1, 1], (3, 1)))
    ref = _oracle(obs[19:])
    _assert_oracle(_run(sim, _increments(obs[19:], obs[18])), ref, 30, origin=19 * DT)


@pytest.mark.parametrize("change", ["period", "baseline", "scale", "params", "voi", "source",
                                    "layout", "clock", "tick", "missing", "legacy", "duplicate"])
def test_native_bw_stream_rejects_incompatible_snapshot_configuration(change):
    sim = _armed()
    _run(sim, _increments(_neural_fixture(2)[:7]))
    saved = sim.get_global_monitor_states()
    configs = {"period": dict(istep=8), "baseline": dict(bw_drive_baseline=1),
               "scale": dict(bw_drive_scale=-0.5), "voi": dict(voi=[0]),
               "params": dict(bw_stream_params=dict(asdict(NOMINAL_PARAMETERS), echo_time_s=0.03))}
    target = _sim(dt=5 if change == "clock" else DT)
    if change == "source":
        target.set_voi_specs(0, [0, 0, 0, 1, 0, 0])
    target.set_t_abs(6 if change == "tick" else 7)
    gather = dict(GATHER, node_indices=[0, 1, 2]) if change == "layout" else GATHER
    spec = _spec(**configs.get(change, {})) if change != "clock" else _spec(dt=5, istep=2)
    target.set_global_monitors([gather], [spec])
    before = target.get_global_monitor_states()
    if change == "missing":
        saved.clear()
    elif change == "legacy":
        saved[0].pop("bw_stream_config")
    elif change == "duplicate":
        saved.append(saved[0])
    with pytest.raises(RuntimeError, match="global BW stream:"):
        target.set_global_monitor_states(saved)
    np.testing.assert_equal(target.get_global_monitor_states(), before)


@pytest.mark.parametrize("field,value", [("bw_drive_count", 1), ("accum_count", 4),
                                         ("regional_shape", None), ("nonfinite", np.inf),
                                         ("invalid_flow", -1.0)])
def test_native_bw_stream_rejected_restore_preserves_staged_monitors_and_pending_output(field, value):
    sim = _armed(period_ms=30, extra=[_spec(slot=1, istep=12)])
    obs = _neural_fixture(5)[:19]
    _run(sim, _increments(obs), drain=False)
    before = sim.get_global_monitor_states()
    candidate = sim.get_global_monitor_states()
    candidate[0]["bw_stream_regions"][0, 0] += 1  # legal change to first staged state
    if field == "regional_shape":
        candidate[1]["bw_stream_regions"] = np.zeros((2, 6))
    elif field == "nonfinite":
        candidate[1]["bw_stream_regions"][0, 1] = value
    elif field == "invalid_flow":
        candidate[1]["bw_stream_regions"][0, 3] = value
    else:
        candidate[1][field] = value
    with pytest.raises(RuntimeError, match="global BW stream:"):
        sim.set_global_monitor_states(candidate)
    np.testing.assert_equal(sim.get_global_monitor_states(), before)
    first, second = sim.take_global_monitor_outputs()
    _assert_oracle(first, _oracle(obs), 30)
    _assert_stream(first, second)


@pytest.mark.parametrize("field,value,error", [
    ("width", 4, "geometry"), ("n_out_space", 2, "geometry"),
    ("n_out_v", 1, "geometry"), ("n_out_modes", 2, "geometry"),
    ("accum", np.empty(0), "buffers"), ("stock", np.zeros(1), "buffers"),
    ("slot", 9, "unmatched"), ("kind", 1, "unmatched"),
    ("missing", None, "missing"), ("duplicate", None, "duplicate"),
    ("missing_identity", None, "identity"),
])
def test_native_bw_stream_rejected_sibling_preserves_global_monitors_and_pending_output(field, value, error):
    companions = [dict(kind=1, slot=1), dict(kind=4, slot=2, istep=4)]
    sim = _armed(extra=companions)
    reference = _armed(extra=companions)
    delta = _increments(_neural_fixture(5)[:19])
    for target in (sim, reference):
        _run(target, delta, drain=False)
    before = sim.get_global_monitor_states()
    candidate = sim.get_global_monitor_states()
    candidate[0]["bw_stream_regions"][0, 0] += 1  # legal change to staged state
    candidate[1]["run_start"] += 1  # legal change to first sibling
    if field == "missing":
        candidate.pop()
    elif field == "duplicate":
        candidate.append(candidate[-1])
    elif field == "missing_identity":
        candidate[-1].pop("slot")
    else:
        candidate[-1][field] = value
    with pytest.raises(RuntimeError, match=f"global BW stream: .*{error}"):
        sim.set_global_monitor_states(candidate)
    np.testing.assert_equal(sim.get_global_monitor_states(), before)
    for actual, expected in zip(sim.take_global_monitor_outputs(), reference.take_global_monitor_outputs()):
        assert expected[0].size and expected[1].size
        _assert_stream(actual, expected)


@pytest.mark.parametrize("changes,error", [
    ({"voi": []}, "observed VOI"), ({"voi": [0, 1]}, "observed VOI"),
    ({"voi": [2]}, "observed VOI"), ({"istep": 6}, "output period"),
    ({"istep": 0}, "output period"), ({"istep": -1}, "output period"),
    ({"dt": 5}, "neural clock"), ({"dt": np.nan}, "neural clock"),
    ({"bw_drive_baseline": np.nan}, "calibration"), ({"bw_drive_scale": np.inf}, "calibration"),
    ({"bw_stream_params": {}}, "parameter set"),
    ({"bw_stream_params": dict(asdict(NOMINAL_PARAMETERS), resting_blood_volume=1.0)}, "parameters"),
    ({"cadence_K": 2}, "global width 1"), ({"complete_window": True}, "global width 1"),
])
def test_native_bw_stream_rejects_invalid_spec(changes, error):
    sim = _sim()
    with pytest.raises(RuntimeError, match=error):
        sim.set_global_monitors([GATHER], [_spec(**changes)])
    assert sim.get_global_monitor_states() == []


@pytest.mark.parametrize("geometry,error", [({"width": 4}, "width 1"), ({"k": 2}, "clock"),
    ({"dt": 3}, "divide 10"), ({"dt": 1e-10}, "divide 10"), ({"dt": 5e-7}, "divide 10"),
    ({"subnet_dt": 0.1}, "clock"), ({"modes": 2}, "mode"), ({"subnet_count": 2}, "one regional subnet")])
def test_native_bw_stream_rejects_unsupported_geometry(geometry, error):
    sim = _sim(**geometry)
    with pytest.raises(RuntimeError, match=error):
        sim.set_global_monitors([GATHER] * geometry.get("subnet_count", 1),
                                [_spec(dt=geometry.get("dt", DT))])


def test_native_bw_stream_explicit_attachment_and_checked_live_vois():
    sim = _sim()
    with pytest.raises(RuntimeError, match="global width 1"):
        sim.set_monitors(0, [_spec()])
    with pytest.raises(RuntimeError, match="explicit native dictionary"):
        sim.set_global_monitors([GATHER], [(0, 10, 4, 1, 1, 1, False, [1], [], DT)])
    with pytest.raises(RuntimeError, match="complete, disjoint node gather"):
        sim.set_global_monitors([dict(GATHER, node_indices=[0, 0, 1])], [_spec()])
    with pytest.raises(RuntimeError, match="slot must be unique"):
        sim.set_global_monitors([GATHER], [_spec(), dict(slot=0, kind=9)])
    sim.set_voi_specs(0, [0, 6, 0, 0, 0, 0])  # invalid even in the unselected row
    with pytest.raises(RuntimeError, match="invalid observed VOI source"):
        sim.set_global_monitors([GATHER], [_spec()])
    sim = _armed()
    sim.set_voi_specs(0, [1, 0, 0, 1, 0, 0])
    with pytest.raises(RuntimeError, match="changed selected observation"):
        _run(sim, np.zeros((1, 3), np.float32))
    assert sim.t_abs() == 0


@pytest.mark.parametrize("voi", [np.array([2**32 + 1], dtype=np.int64),
                                  np.array([2**64 - 1], dtype=np.uint64),
                                  np.array([1.0]), np.array([True]), [1.0], [np.bool_(True)]])
def test_native_bw_stream_checks_voi_indices_before_narrowing(voi):
    with pytest.raises(RuntimeError, match="observed VOI"):
        _armed(spec=_spec(voi=voi))


@pytest.mark.parametrize("as_tuple", [False, True], ids=["dict", "tuple"])
@pytest.mark.parametrize("nodes", [np.array([2, 0, 2**32 + 1], dtype=np.int64),
                                    np.array([2, 0, 2**64 - 1], dtype=np.uint64),
                                    np.array([2.0, 0.0, 1.0]), [2, 0, True]])
def test_native_bw_stream_checks_node_gather_indices_before_narrowing(as_tuple, nodes):
    gather = (2, 3, 0, 0, nodes) if as_tuple else dict(GATHER, node_indices=nodes)
    with pytest.raises(RuntimeError, match="node gather index"):
        _armed(gather=gather)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint64, np.dtype(">i4")])
def test_native_bw_stream_accepts_integer_ndarray_voi_and_strided_gather(dtype):
    nodes = np.array([2, 99, 0, 99, 1], dtype=dtype)[::2]
    sim = _armed(gather=dict(GATHER, node_indices=nodes), spec=_spec(voi=np.array([1], dtype=dtype)))
    obs = _neural_fixture(3)
    _assert_oracle(_run(sim, _increments(obs)), _oracle(obs), 10)


def test_native_bw_stream_numerical_failure_does_not_commit_closing_step():
    sim = _armed(spec=_spec(bw_drive_scale=1e308))
    # Gather order [node 1, node 2, node 0] -> two valid regions before failing.
    delta = np.zeros((4, 3), dtype=np.float32)
    _run(sim, delta[:3])
    before = sim.get_global_monitor_states()
    delta[-1, 0] = 100
    with pytest.raises(RuntimeError, match="calibrated drive or RK4 domain failure"):
        _run(sim, delta[-1:])
    np.testing.assert_equal(sim.get_global_monitor_states(), before)
    assert sim.take_global_monitor_outputs()[0][0].size == 0
    # The neural state advanced before the monitor failed; the simulator run
    # itself is not transactional and cannot safely retry from this state.
    assert sim.t_abs() == 3
    assert sim.get_subnet_state(0).ravel()[0] == 102


def test_native_bw_stream_large_commensurate_period_has_no_window_sized_working_buffer():
    sim = _sim(dt=10)
    sim.set_global_monitors([GATHER], [_spec(dt=10, istep=(1 << 31) - 1)])
    assert _run(sim, np.zeros((8, 3), dtype=np.float32))[0].size == 0
    state, = sim.get_global_monitor_states()
    assert state["bw_drive_count"] == 0 and state["accum_count"] == 16
    assert state["bw_stream_regions"].nbytes == 144


def test_native_bw_stream_accepts_a_binary32_representation_of_a_10ms_divisor():
    dt = 10 / 3
    sim = _sim(dt=dt)
    sim.set_global_monitors([GATHER], [_spec(dt=dt, istep=3)])
    assert _run(sim, np.zeros((2, 3), dtype=np.float32))[0].size == 0
    times, values = _run(sim, np.zeros((4, 3), dtype=np.float32))
    np.testing.assert_array_equal(times, [5, 15])
    np.testing.assert_array_equal(values, np.zeros((2, 1, 3, 1), dtype=np.float32))


def test_native_bw_stream_coexists_with_complete_window_hrf_and_endpoint_bw():
    companions = [dict(kind=4, slot=1, complete_window=True, istep=4, dt=DT, voi=[1]),
                  dict(kind=3, slot=2, istep=4, interim_istep=4, stock_steps=3,
                       hrf=[1, 0.25, 0.5], dt=DT, voi=[1]),
                  dict(kind=9, slot=3, istep=4, bw_dt=0.0025, dt=DT, voi=[0])]
    sim = _armed(extra=companions)
    obs = _neural_fixture(5, tail=0)
    delta = _increments(obs)
    _run(sim, delta[:7], drain=False)
    parts = [sim.take_global_monitor_outputs()]
    # Restore with an open interim bin, exactly at a close on row 1,
    # and exactly at the first wrapped close on row 0.
    for split, stop, interim_count, stock_pos in [(7, 8, 3, 0), (8, 16, 4, 1), (16, 20, 4, 0)]:
        states = sim.get_global_monitor_states()
        hrf_state = states[2]
        assert (hrf_state["m_step"], hrf_state["interim_count"], hrf_state["stock_pos"]) == (
            split, interim_count, stock_pos,
        )
        resumed = _armed(extra=companions)
        _run(resumed, np.zeros_like(delta[:split]), drain=False)
        resumed.set_subnet_state(0, sim.get_subnet_state(0))
        resumed.set_subnet_buf(0, sim.get_subnet_buf(0))
        resumed.set_t_abs(sim.t_abs())
        # Reversed records and pending output in every target monitor.
        resumed.set_global_monitor_states(states[::-1])
        np.testing.assert_equal(resumed.get_global_monitor_states(), states)
        for times, values in resumed.take_global_monitor_outputs():
            assert times.size == values.size == 0
        for target in (sim, resumed):
            _run(target, delta[split:stop], drain=False)
        parts.append(resumed.take_global_monitor_outputs())
        for actual, expected in zip(parts[-1], sim.take_global_monitor_outputs()):
            _assert_stream(actual, expected)
        np.testing.assert_equal(resumed.get_global_monitor_states(), sim.get_global_monitor_states())

    bw, average, hrf, endpoint = [tuple(np.concatenate(rows) for rows in zip(*streams))
                                  for streams in zip(*parts)]
    _assert_oracle(bw, _oracle(obs), 10)
    np.testing.assert_array_equal(average[0], [5, 15, 25, 35, 45])
    np.testing.assert_array_equal(hrf[0], [10, 20, 30, 40, 50])
    np.testing.assert_array_equal(endpoint[0], hrf[0])
    expected = obs[:, [1, 2, 0]].reshape(5, 4, 3).mean(axis=1)[:, None, :, None]
    expected_hrf = expected.copy()
    expected_hrf[1:] += 0.5 * expected[:-1]
    expected_hrf[2:] += 0.25 * expected[:-2]
    np.testing.assert_array_equal(average[1], expected)
    np.testing.assert_array_equal(hrf[1], expected_hrf)
    np.testing.assert_array_equal(endpoint[1], np.zeros_like(expected))


def test_native_bw_stream_and_projection_window_restore_an_exact_mixed_set():
    obs = _neural_fixture(5, 1)
    delta = _increments(obs)
    gain = np.array([[1, 0.5, -1], [-1, 0, 2]], dtype=np.float32)
    projection = dict(slot=1, kind=8, complete_window=True, istep=3, dt=DT,
                      voi=[1], gain=gain)
    gather = dict(n_voi=2, n_node=3)
    ordinary = dict(slot=2, kind=2, istep=4, dt=DT, voi=[0])

    def armed():
        return _armed(gather=gather, extra=(projection, ordinary))

    full = armed()
    _run(full, delta, 11, drain=False)
    whole = full.take_global_monitor_outputs()
    p_times, p_values = whole[1]
    np.testing.assert_array_equal(p_times, (np.arange(7) * 3 + 1.5) * DT)
    expected = obs.astype(np.float64).reshape(7, 3, 3).mean(axis=1) @ gain.T
    np.testing.assert_array_equal(p_values[:, 0, :, 0], expected.astype(np.float32))

    split = armed()
    _run(split, delta[:5], 4, drain=False)
    first = split.take_global_monitor_outputs()
    bw, proj, other = split.get_global_monitor_states()
    assert bw["bw_drive_count"] == 1 and proj["accum_count"] == 2
    assert "bw_stream_config" in bw and "window_config" in proj
    resumed = armed()
    _run(resumed, delta[:5], 2, drain=False)
    pending = resumed.get_global_monitor_states()
    historical = {key: value for key, value in proj.items() if key != "window_config"}
    for candidate in ([bw, other], [bw, historical, other], [bw, proj],
                      [bw, proj, other, historical], [bw, proj, other, other]):
        with pytest.raises(RuntimeError, match="global (BW stream|complete window)"):
            resumed.set_global_monitor_states(candidate)
        np.testing.assert_equal(resumed.get_global_monitor_states(), pending)
    for actual, expected in zip(resumed.take_global_monitor_outputs(), first):
        _assert_stream(actual, expected)

    resumed = armed()
    resumed.set_subnet_state(0, split.get_subnet_state(0))
    resumed.set_subnet_buf(0, split.get_subnet_buf(0))
    resumed.set_t_abs(split.t_abs())
    resumed.set_global_monitor_states([other, proj, bw])  # identity, not record order
    _run(resumed, delta[5:], 8, drain=False)
    for head, tail, expected in zip(first, resumed.take_global_monitor_outputs(), whole):
        _assert_stream(tuple(np.concatenate((a, b)) for a, b in zip(head, tail)), expected)


def test_native_projection_before_bw_stream_recovers_from_full_checkpoint_after_run_failure():
    projection = dict(slot=1, kind=8, complete_window=True, istep=4, dt=DT,
                      voi=[1], gain=np.ones((1, 3), dtype=np.float32))

    def armed():
        sim = _sim()
        sim.set_global_monitors([dict(n_voi=2, n_node=3)],
                                [projection, _spec(bw_drive_scale=1e308)])
        sim.monitor_begin_run()
        return sim

    sim = armed()
    quiet = np.zeros((4, 3), dtype=np.float32)
    _run(sim, quiet[:3], drain=False)
    state = sim.get_subnet_state(0)
    history = sim.get_subnet_buf(0)
    tick = sim.t_abs()
    monitors = sim.get_global_monitor_states()
    bad = np.array([[100, 0, 0]], dtype=np.float32)
    with pytest.raises(RuntimeError, match="calibrated drive or RK4 domain failure"):
        _run(sim, bad, drain=False)

    # Projection runs first. Nothing from the failed run is safe to reuse;
    # restore the full checkpoint before attempting another run.
    sim.set_subnet_state(0, state)
    sim.set_subnet_buf(0, history)
    sim.set_t_abs(tick)
    sim.set_global_monitor_states(monitors)
    assert all(t.size == d.size == 0 for t, d in sim.take_global_monitor_outputs())
    _run(sim, quiet[3:], drain=False)
    control = armed()
    _run(control, quiet, drain=False)
    for actual, expected in zip(sim.take_global_monitor_outputs(), control.take_global_monitor_outputs()):
        assert expected[0].size == 1
        _assert_stream(actual, expected)
