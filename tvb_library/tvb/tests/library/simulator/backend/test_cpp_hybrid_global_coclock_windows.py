"""Native master-clock complete windows across indexed regional sources."""

import numpy as np
import pytest

from tvb.simulator import integrators, models, monitors, noise
from tvb.simulator.hybrid import NetworkSet, Simulator, Subnetwork


DT = 0.125
NODE_INDICES = ((3, 0), (4, 1, 2))
CANONICAL = [1, 3, 4, 0, 2]  # global order indexing the concatenated local rows
INITIAL = tuple(np.asarray(x, dtype=float)[..., None] for x in (
    [[2, 10], [-3, 5]], [[21, -12, 4], [8, -9, 17]],
))
SLOPE = tuple(np.asarray(x, dtype=float)[..., None] for x in (
    [[0.5, -2], [1, 4]], [[-5, 0.25, 3], [2, -1, 0.5]],
))
WEIGHTS = np.array([[0.5, 0, 0.125, 0.25, 0.125], [0, 0.5, 0, 0, 0.5]])


@pytest.fixture
def cpp():
    return pytest.importorskip("tvb.simulator.backend.cpp_hybrid.backend")


def _network(*, stochastic=False, maps=NODE_INDICES, cadence=(1, 1),
             observed=(("W", "V"), ("W", "V")), names=("small", "large"), dt0=DT):
    sources = []
    for i, (indices, k, terms, name) in enumerate(zip(maps, cadence, observed, names)):
        # With all nonlinear/linear coefficients zero, V'=I and W'=a.
        params = dict.fromkeys(("alpha", "beta", "b", "c", "e", "f", "g"), [0.0])
        params.update(d=[1.0], tau=[1.0], gamma=[1.0], I=SLOPE[i][0, :, 0], a=SLOPE[i][1, :, 0])
        model = models.Generic2dOscillator(variables_of_interest=terms,
                                          **{key: np.array(value) for key, value in params.items()})
        dt = k * dt0
        scheme = (integrators.EulerStochastic(dt=dt, noise=noise.Additive(nsig=np.zeros(2)))
                  if stochastic else integrators.EulerDeterministic(dt=dt))
        sources.append(Subnetwork(name=name, nnodes=INITIAL[i].shape[1], model=model, scheme=scheme,
                                  node_indices=None if indices is None else np.asarray(indices)))
    return NetworkSet(subnets=sources).configure()


def _windows(period=4 * DT):
    result = [kind(period=period, variables_of_interest=np.array([1, 0]))
              for kind in (monitors.TemporalAverage, monitors.GlobalAverage, monitors.SpatialAverage)]
    result[-1].spatial_mean = WEIGHTS.copy()
    return result


def _post_update(i, steps):
    return INITIAL[i] + np.asarray(steps)[..., None, None, None] * DT * SLOPE[i]


def _regional(steps):
    return np.concatenate([_post_update(i, steps) for i in range(2)], axis=-2)[..., CANONICAL, :]


def _oracle(stop, *, start=0, m=4, weights=WEIGHTS):
    closes = np.arange(start // m + 1, stop // m + 1) * m
    times = (closes - m / 2) * DT
    means = _regional(closes[:, None] - np.arange(m)).mean(axis=1)
    return [(times, means), (times, means.mean(axis=2, keepdims=True)),
            (times, np.einsum("an,tvnm->tvam", weights, means))]


def _assert_streams(actual, expected, *, atol=2e-6):
    assert len(actual) == len(expected)
    for (times, data), (ref_times, ref_data) in zip(actual, expected):
        np.testing.assert_array_equal(times, ref_times)
        assert data.shape == ref_data.shape
        np.testing.assert_allclose(data, ref_data, rtol=0, atol=atol)


def _unpack_global(streams):
    return [stream for (stream,) in streams]


@pytest.mark.parametrize("cadence", [(1, 1), (1, 3)], ids=["co-clock", "mixed-rate"])
@pytest.mark.parametrize("stochastic", [False, True], ids=["one-native-call", "numpy-chunks"])
def test_gathered_global_windows_chunking_and_no_tail(cpp, monkeypatch, stochastic, cadence):
    real_sim, calls = cpp.Sim, []

    class TracedSim:
        def __init__(self, width):
            self.sim = real_sim(width)

        def __getattr__(self, name):
            return getattr(self.sim, name)

        def run(self, nstep, *args, **kwargs):
            calls.append(nstep)
            assert not kwargs["need_tavg"] and not kwargs["need_ctavg"]
            return self.sim.run(nstep, *args, **kwargs)

    monkeypatch.setattr(cpp, "Sim", TracedSim)
    compiled = cpp.CppHybridBackend().compile(_network(stochastic=stochastic, cadence=cadence))
    actual, checkpoint = compiled.run(11, chunk_size=7, monitors=_windows(), initial_states=INITIAL,
                                      return_snapshot=True)
    assert calls == ([7, 4] if stochastic else [11])
    expected = _oracle(11) if cadence == (1, 1) else _mixed_rate_oracle(11)
    _assert_streams(_unpack_global(actual), expected)
    assert [state["accum_count"] for state in checkpoint["kmon_global_states"]] == [3] * 3


def test_coclock_global_windows_release_at_boundaries_not_midpoints(cpp):
    compiled = cpp.CppHybridBackend().compile(_network())
    first = compiled.run(1, monitors=_windows(), initial_states=INITIAL)
    _assert_streams(_unpack_global(first), _oracle(1))  # includes shaped empty data
    releases = []
    for tick in range(2, 12):
        compiled._sim.run(1, 1, None, None, need_tavg=False, need_ctavg=False)
        outputs = compiled._sim.take_global_monitor_outputs()
        _assert_streams(outputs, _oracle(tick, start=tick - 1))
        if outputs[0][0].size:
            releases.append(tick)
    assert releases == [4, 8]
    assert [state["accum_count"] for state in compiled._sim.get_global_monitor_states()] == [3] * 3


def test_coclock_global_windows_public_points_and_local_recorder(cpp):
    net = _network()
    net.subnets[0].add_monitor(monitors.TemporalAverage(period=3 * DT))
    public = Simulator(nets=net, backend="cpp", simulation_length=11 * DT,
                       monitors=_windows() + [monitors.Raw(), monitors.SubSample(period=4 * DT)])
    public.configure()
    output = public.run(initial_conditions=INITIAL)
    steps = np.arange(1, 12)
    _assert_streams(output, _oracle(11) + [(steps * DT, _regional(steps)[:, [1, 0]]),
                                          (steps[3::4] * DT, _regional(steps[3::4])[:, [1, 0]])])
    recorder = net.subnets[0].monitors[0]
    close = np.array([3, 6, 9])
    local = _post_update(0, close[:, None] - np.arange(3)).mean(axis=1)[:, [1, 0]]
    _assert_streams([recorder.to_arrays()], [((close - 1.5) * DT, local)])


def test_coclock_global_windows_checked_resume_and_fresh_reset(cpp):
    be = cpp.CppHybridBackend()
    compiled = be.compile(_network())
    first, snapshot = compiled.run(7, monitors=_windows(), initial_states=INITIAL, return_snapshot=True)
    _assert_streams(_unpack_global(first), _oracle(7))
    states = snapshot["kmon_global_states"]
    assert all("coclock_window_config" in state for state in states)
    assert [state["accum_count"] for state in states] == [3] * 3

    last = be.compile(_network()).resume(snapshot, 4, monitors=_windows())
    _assert_streams(_unpack_global(last), _oracle(11, start=7))
    _, neural = be.compile(_network()).resume(snapshot, 4, return_snapshot=True)
    assert neural["step"] == 11 and neural["kmon_global_states"] == []
    for i, state in enumerate(neural["states"]):
        np.testing.assert_allclose(state, _post_update(i, 11), rtol=0, atol=1e-6)

    changed = _windows(period=DT)
    changed[-1].spatial_mean = WEIGHTS[:, ::-1].copy()
    fresh = compiled.run(5, monitors=changed, initial_states=INITIAL)
    _assert_streams(_unpack_global(fresh), _oracle(5, m=1, weights=changed[-1].spatial_mean))


PROJECTION_GAIN = np.array([[0.25, 0, -0.5, 0.125, 0.25], [0, -0.5, 0, 0.5, 0.25]])
PROJECTION_REF = np.array([0.25, 0.75])


def _mixed_windows():
    projection = monitors.Projection(period=4 * DT, variables_of_interest=np.array([1, 0]),
                                     obsnoise=None)
    projection.gain = PROJECTION_GAIN.copy()
    projection._ref_vec = PROJECTION_REF.copy()
    return [_windows()[0], projection]


def _mixed_oracle(stop, *, start=0, gain=PROJECTION_GAIN):
    closes = np.arange(start // 4 + 1, stop // 4 + 1) * 4
    means = _regional(closes[:, None] - np.arange(4)).mean(axis=1)
    projected = np.einsum("an,tvnm->tvam", gain, means)
    projected -= np.einsum("a,tvam->tvm", PROJECTION_REF, projected)[:, :, None, :]
    # Generic Projection retains the legacy mean-of-post-step timestamp.
    return [((closes - 2) * DT, means), ((closes - 1.5) * DT, projected)]


def test_coclock_global_projection_rearms_changed_gain_and_resumes_open_window(cpp):
    be = cpp.CppHybridBackend()
    compiled, requested = be.compile(_network()), _mixed_windows()
    first, snapshot = compiled.run(7, monitors=requested, initial_states=INITIAL, return_snapshot=True)
    _assert_streams(_unpack_global(first), _mixed_oracle(7))
    window, projection = snapshot["kmon_global_states"]
    assert window["accum_count"] == projection["accum_count"] == 3
    assert set(snapshot["coclock_projection_identities"]) == {1}
    assert "bw_projection_identities" not in snapshot

    native = compiled._sim
    before = native.get_global_monitor_states()
    for field, value, error in (
        ("accum_count", None, "invalid sibling snapshot counter: accum_count"),
        ("accum_count", 3.0, "invalid sibling snapshot counter: accum_count"),
        ("run_start", False, "invalid sibling snapshot counter: run_start"),
        ("accum_count", 2, "incompatible sibling snapshot counter phase"),
    ):
        candidate = native.get_global_monitor_states()
        candidate[0]["accum"] += 1.0  # valid, but must not commit on sibling rejection
        if value is None:
            candidate[1].pop(field)
        else:
            candidate[1][field] = value
        with pytest.raises(RuntimeError, match=f"global gathered complete window: {error}"):
            native.set_global_monitor_states(candidate)
        np.testing.assert_equal(native.get_global_monitor_states(), before)

    requested[1].gain *= -2
    fresh = compiled.run(11, monitors=requested, initial_states=INITIAL)
    _assert_streams(_unpack_global(fresh), _mixed_oracle(11, gain=-2 * PROJECTION_GAIN))
    for target in (be.compile(_network()), compiled):
        continued = target.resume(snapshot, 4, monitors=_mixed_windows())
        _assert_streams(_unpack_global(continued), _mixed_oracle(11, start=7))

    _, neural = compiled.resume(snapshot, 4, return_snapshot=True)
    assert neural["step"] == 11 and neural["kmon_global_states"] == []
    assert "coclock_projection_identities" not in neural
    for i, state in enumerate(neural["states"]):
        np.testing.assert_allclose(state, _post_update(i, 11), rtol=0, atol=1e-6)


@pytest.mark.parametrize("resuming", [False, True], ids=["fresh", "resume"])
def test_coclock_global_projection_companion_rejects_observation_noise(cpp, resuming):
    be, net = cpp.CppHybridBackend(), _network()
    requested = _mixed_windows()
    if resuming:
        _, snapshot = be.compile(net).run(7, monitors=requested, initial_states=INITIAL,
                                          return_snapshot=True)
    requested[1].obsnoise = noise.Additive(nsig=np.array([0.25]))
    target = be.compile(net)
    with pytest.raises(NotImplementedError, match="Projection companions require obsnoise=None"):
        if resuming:
            target.resume(snapshot, 4, monitors=requested)
        else:
            target.run(11, monitors=requested, initial_states=INITIAL)
    assert target._sim.t_abs() == (7 if resuming else 0)
    assert target._sim.get_global_monitor_states() == []


@pytest.mark.parametrize("change", ["gain", "reference", "period", "voi-order", "missing-sidecar", "mapping"])
def test_coclock_global_projection_rejects_changed_requested_resume(cpp, change):
    be = cpp.CppHybridBackend()
    _, snapshot = be.compile(_network()).run(7, monitors=_mixed_windows(),
                                             initial_states=INITIAL, return_snapshot=True)
    requested = _mixed_windows()
    projection = requested[1]
    net = _network()
    if change == "gain":
        projection.gain[0, 0] += 0.5
    elif change == "reference":
        projection._ref_vec[:] = [0.75, 0.25]
    elif change == "period":
        projection.period = 3 * DT
    elif change == "voi-order":
        projection.variables_of_interest = np.array([0, 1])
    elif change == "missing-sidecar":
        snapshot.pop("coclock_projection_identities")
    elif change == "mapping":
        net = _network(maps=((0, 3), NODE_INDICES[1]))

    target = be.compile(net)
    error, message = ((RuntimeError, "global gathered complete window: incompatible snapshot configuration")
                      if change == "mapping" else
                      (ValueError, "co-clock global complete windows resume requires matching Projection"))
    with pytest.raises(error, match=message):
        target.resume(snapshot, 4, monitors=requested)
    assert target._sim.t_abs() == 7  # initialization, but no integration
    if change != "mapping":
        assert target._sim.get_global_monitor_states() == []  # no attachment/import on a fresh target


@pytest.mark.parametrize("change", ["mapping", "source-name", "source-order", "observed-order",
                                    "missing-mapping", "operator", "voi", "period", "missing-request", "point-only"])
def test_coclock_global_windows_reject_changed_requested_resume(cpp, change):
    be = cpp.CppHybridBackend()
    _, snapshot = be.compile(_network()).run(7, monitors=_windows(), initial_states=INITIAL, return_snapshot=True)
    target, requested = _network(), _windows()
    if change == "mapping":
        target = _network(maps=((0, 3), NODE_INDICES[1]))
    elif change == "source-name":
        target = _network(names=("other", "large"))
        snapshot["buffers"]["other"] = snapshot["buffers"].pop("small")
    elif change == "source-order":
        target = NetworkSet(subnets=target.subnets[::-1]).configure()
        snapshot["states"].reverse()
    elif change == "observed-order":
        target = _network(observed=(("V", "W"), ("V", "W")))
    elif change == "missing-mapping":
        target = _network(maps=(None, None))
    elif change == "operator":
        requested[-1].spatial_mean = WEIGHTS[:, ::-1].copy()
    elif change == "voi":
        requested[1].variables_of_interest = np.array([0, 1])
    elif change == "period":
        requested[0].period = 3 * DT
    elif change == "missing-request":
        requested.pop()
    elif change == "point-only":
        requested = [monitors.Raw()]
    compiled = be.compile(target)
    with pytest.raises(RuntimeError, match="global gathered complete window"):
        compiled.resume(snapshot, 4, monitors=requested)
    assert compiled._sim.t_abs() == 7  # no integration on a rejected request


@pytest.mark.parametrize("cadence", [(1, 1), (1, 3)], ids=["co-clock", "mixed-rate"])
@pytest.mark.parametrize("change", ["missing", "duplicate", "legacy", "origin", "phase", "nonfinite", "cadence"])
def test_gathered_global_windows_native_restore_is_transactional(cpp, cadence, change):
    compiled = cpp.CppHybridBackend().compile(_network(cadence=cadence))
    compiled.run(6, monitors=_windows(), initial_states=INITIAL)
    sim = compiled._sim
    sim.run(3, 1, None, None, need_tavg=False, need_ctavg=False)  # pending tick-8 release and one open row
    before = sim.get_global_monitor_states()
    candidate = sim.get_global_monitor_states()
    candidate[0]["accum"] += 1.0  # valid but distinct earlier state; must not be committed
    if change == "missing":
        candidate.pop()
    elif change == "duplicate":
        candidate.append(candidate[-1])
    elif change == "legacy":
        candidate[-1].pop("coclock_window_config")
    elif change == "origin":
        candidate[-1]["run_start"] = 4  # same phase but not a compatible origin
    elif change == "phase":
        candidate[-1]["accum_count"] = 2
    elif change == "nonfinite":
        candidate[-1]["accum"][0] = np.nan
    elif change == "cadence":
        declared_sources = candidate[-1]["coclock_window_config"][6]
        source = declared_sources[1]
        declared_sources[1] = (*source[:3], source[3] + 1, *source[4:])
    with pytest.raises(RuntimeError, match="global gathered complete window"):
        sim.set_global_monitor_states(candidate)
    for orig, current in zip(before, sim.get_global_monitor_states()):
        np.testing.assert_array_equal(current["accum"], orig["accum"])
    expected = _oracle(9, start=6) if cadence == (1, 1) else _mixed_rate_oracle(9, start=6)
    _assert_streams(sim.take_global_monitor_outputs(), expected)


@pytest.mark.parametrize("maps", [((0, 0), (4, 1, 2)), ((3, 0), (4, 1, 7)),
                                  ((2**32 + 3, 0), (4, 1, 2))])
def test_coclock_global_windows_reject_invalid_node_indices(cpp, maps):
    compiled = cpp.CppHybridBackend().compile(_network(maps=maps))
    with pytest.raises(ValueError, match="node_indices"):
        compiled.run(4, monitors=_windows(), initial_states=INITIAL)
    assert compiled._sim.t_abs() == 0


def test_coclock_global_windows_check_native_source_and_narrowing(cpp):
    compiled = cpp.CppHybridBackend().compile(_network())
    gather = compiled._gathered_window_sources(DT)
    sim = compiled._sim
    window = dict(slot=0, kind=4, istep=4, dt=DT, complete_window=True)
    with pytest.raises(RuntimeError, match="observed VOI"):
        sim.set_global_monitors(gather, [dict(window, voi=np.array([2**32], dtype=np.int64))])
    with pytest.raises(RuntimeError, match="node gather index"):
        sim.set_global_monitors([dict(gather[0], node_indices=np.array([2**32 + 3, 0], dtype=np.int64)),
                                 gather[1]], [window])
    with pytest.raises(RuntimeError, match="common declared clock"):
        sim.set_global_monitors([gather[0], dict(gather[1], source_dt=np.nextafter(DT, 1.0))], [window])
    sim.set_global_monitors(gather, [window])
    with pytest.raises(RuntimeError, match="detach before changing observed rows"):
        sim.set_voi(1, [0, 1])


@pytest.mark.parametrize("cadence", [(1, 1), (1, 3)], ids=["co-clock", "mixed-rate"])
@pytest.mark.parametrize("profile", ["missing-indices", "heterogeneous-order", "host", "unprepared-spatial"])
def test_gathered_global_windows_leave_other_routes_unstaged(cpp, cadence, profile):
    be = cpp.CppHybridBackend()
    requested = [monitors.TemporalAverage(period=4 * DT)]
    if profile == "missing-indices":
        net = _network(cadence=cadence, maps=(None, None))
    elif profile == "heterogeneous-order":
        net = _network(cadence=cadence, observed=(("W", "V"), ("V", "W")))
    else:
        net = _network(cadence=cadence)
        if profile == "host":
            be.enable_kernel_monitors = False
        else:
            requested = [monitors.SpatialAverage(period=4 * DT)]
    outputs, snapshot = be.compile(net).run(8, monitors=requested, initial_states=INITIAL, return_snapshot=True)
    assert not any("coclock_window_config" in state for state in snapshot["kmon_global_states"])
    assert len(outputs[0]) == (1 if profile == "unprepared-spatial" and cadence == (1, 1) else 2)


def test_coclock_global_windows_reject_noncommensurate_period_and_invalid_operator(cpp):
    compiled = cpp.CppHybridBackend().compile(_network())
    with pytest.raises(ValueError, match="commensurate"):
        compiled.run(4, monitors=_windows(period=3.5 * DT), initial_states=INITIAL)
    windows = _windows()
    windows[-1].spatial_mean[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite preconfigured spatial_mean"):
        compiled.run(4, monitors=windows, initial_states=INITIAL)
    assert compiled._sim.t_abs() == 0


MIXED_CADENCE = (1, 3)


def _held_source(i, ticks, *, cadence=MIXED_CADENCE, dt0=DT):
    count = np.asarray(ticks) // cadence[i]
    return INITIAL[i] + count[..., None, None, None] * (cadence[i] * dt0) * SLOPE[i]


def _mixed_rate_oracle(stop, *, start=0, m=4, cadence=MIXED_CADENCE, dt0=DT, weights=WEIGHTS):
    closes = np.arange(start // m + 1, stop // m + 1) * m
    # Sample every master tick, irrespective of which subnet stepped on that tick.
    ticks = closes[:, None] + np.arange(1 - m, 1)
    held_rows = [_held_source(i, ticks, cadence=cadence, dt0=dt0) for i in range(2)]
    rows = np.concatenate(held_rows, axis=3)[:, :, :, CANONICAL, :].mean(axis=1)
    times = (closes - m / 2) * dt0
    return [(times, rows), (times, rows.sum(axis=2, keepdims=True) / len(CANONICAL)),
            (times, np.einsum("an,tvnm->tvam", weights, rows))]


@pytest.mark.parametrize("split", [1, 7], ids=["before-first-slow-update", "off-cadence"])
def test_mixed_rate_global_windows_checked_resume_and_rearm(cpp, split):
    be = cpp.CppHybridBackend()
    # The slow source comes first; the master clock (0.1) is not float32-exact.
    cadence, dt0 = (3, 1), 0.1
    compiled = be.compile(_network(cadence=cadence, dt0=dt0))
    windows = _windows(period=4 * dt0)
    first, snapshot = compiled.run(split, monitors=windows, initial_states=INITIAL, return_snapshot=True)
    _assert_streams(_unpack_global(first), _mixed_rate_oracle(split, cadence=cadence, dt0=dt0))
    states = snapshot["kmon_global_states"]
    assert [state["accum_count"] for state in states] == [split % 4] * 3
    assert all([source[3] for source in state["coclock_window_config"][6]] == [3, 1] for state in states)

    if split == 1:
        # No window was released, but the initial slow rows were already accumulated.
        rows = np.concatenate([INITIAL[0], _held_source(1, 1, cadence=cadence, dt0=dt0)], axis=1)
        np.testing.assert_allclose(states[0]["accum"], rows[:, CANONICAL, :].ravel(), rtol=0, atol=2e-6)

    changed = _windows(period=dt0)
    changed[-1].spatial_mean = WEIGHTS[:, ::-1].copy()
    fresh = compiled.run(5, monitors=changed, initial_states=INITIAL)
    _assert_streams(_unpack_global(fresh), _mixed_rate_oracle(
        5, m=1, cadence=cadence, dt0=dt0, weights=changed[-1].spatial_mean), atol=6e-6)
    slow_columns = NODE_INDICES[0]
    slow = fresh[0][0][1][:, :, slow_columns, :]
    np.testing.assert_allclose(slow[:2], np.broadcast_to(INITIAL[0], slow[:2].shape))
    np.testing.assert_array_equal(slow[2:], np.broadcast_to(slow[2], slow[2:].shape))
    assert not np.array_equal(slow[1], slow[2])

    for target in (compiled, be.compile(_network(cadence=cadence, dt0=dt0))):
        last = target.resume(snapshot, 11 - split, monitors=windows)
        _assert_streams(_unpack_global(last), _mixed_rate_oracle(11, start=split, cadence=cadence, dt0=dt0), atol=6e-6)


def test_mixed_rate_global_windows_release_and_local_recorder_points(cpp):
    compiled = cpp.CppHybridBackend().compile(_network(cadence=MIXED_CADENCE))
    first = compiled.run(1, monitors=_windows(), initial_states=INITIAL)
    _assert_streams(_unpack_global(first), _mixed_rate_oracle(1))  # shaped empties
    release = []
    for tick in range(2, 12):
        compiled._sim.run(1, 1, None, None, need_tavg=False, need_ctavg=False)
        outputs = compiled._sim.take_global_monitor_outputs()
        _assert_streams(outputs, _mixed_rate_oracle(tick, start=tick - 1))
        if outputs[0][0].size:
            release.append(tick)
    assert release == [4, 8]

    requested = _windows() + [monitors.Raw(), monitors.SubSample(period=4 * DT)]
    net = _network(cadence=MIXED_CADENCE)
    net.subnets[1].add_monitor(monitors.TemporalAverage(period=6 * DT))
    public = Simulator(nets=net, backend="cpp", simulation_length=11 * DT, monitors=requested)
    public.configure()
    rows = np.concatenate([_held_source(i, np.arange(1, 12)) for i in range(2)], axis=2)
    rows = rows[:, [1, 0]][:, :, CANONICAL, :]
    ticks = np.arange(1, 12)
    points = [(ticks * DT, rows), (ticks[3::4] * DT, rows[3::4])]
    _assert_streams(public.run(initial_conditions=INITIAL), _mixed_rate_oracle(11) + points)
    # The slow Recorder uses only its eligible ticks 3 and 6 (not held master-grid rows).
    expected_local = _held_source(1, np.array([3, 6])).mean(axis=0, keepdims=True)[:, [1, 0]]
    _assert_streams([net.subnets[1].monitors[0].to_arrays()], [(np.array([3 * DT]), expected_local)])
    point_only = compiled.run(11, monitors=requested[-2:], initial_states=INITIAL)
    together = compiled.run(11, monitors=requested, initial_states=INITIAL)
    assert [len(streams) for streams in together] == [1, 1, 1, 2, 2]
    for actual, expected in zip(together[-2:], point_only):
        _assert_streams(actual, expected)


@pytest.mark.parametrize("change", ["cadence", "clock", "mapping", "observations", "period", "voi", "weights"])
def test_mixed_rate_global_windows_reject_changed_resume_before_stepping(cpp, change):
    be = cpp.CppHybridBackend()
    source, windows = _network(cadence=MIXED_CADENCE), _windows()
    _, snapshot = be.compile(source).run(7, monitors=windows, initial_states=INITIAL, return_snapshot=True)
    if change == "cadence":
        source = _network(cadence=(1, 2))
    elif change == "clock":
        source = _network(cadence=MIXED_CADENCE, dt0=np.nextafter(DT, 1.0))
        for window in windows:
            window.period = 4 * np.nextafter(DT, 1.0)
    elif change == "mapping":
        source = _network(cadence=MIXED_CADENCE, maps=((0, 3), NODE_INDICES[1]))
    elif change == "observations":
        source = _network(cadence=MIXED_CADENCE, observed=(("V", "W"), ("V", "W")))
    elif change == "period":
        windows[0].period = 5 * DT
    elif change == "voi":
        windows[1].variables_of_interest = np.array([0, 1])
    elif change == "weights":
        windows[-1].spatial_mean[0, 0] += 0.5
    target = be.compile(source)
    with pytest.raises(RuntimeError, match="global gathered complete window: incompatible snapshot configuration"):
        target.resume(snapshot, 4, monitors=windows)
    assert target._sim.t_abs() == 7


def test_mixed_rate_global_windows_native_gather_checks_and_scope(cpp):
    compiled = cpp.CppHybridBackend().compile(_network(cadence=MIXED_CADENCE))
    gather = compiled._gathered_window_sources(DT)
    sim = compiled._sim
    window = dict(slot=0, kind=4, istep=4, dt=DT, complete_window=True)
    # The perturbed source clock rounds to the native float32 but does not match K*dt.
    with pytest.raises(RuntimeError, match="declared source clock or cadence"):
        sim.set_global_monitors([gather[0], dict(gather[1], source_dt=3 * DT + 1e-8)], [window])
    with pytest.raises(RuntimeError, match="invalid period, kind or clock"):
        sim.set_global_monitors(gather, [dict(window, cadence_K=3)])
    with pytest.raises(RuntimeError, match="invalid period, kind or clock"):
        sim.set_global_monitors(gather, [dict(window, dt=3 * DT)])
    with pytest.raises(RuntimeError, match="homogeneous observed VOIs"):
        sim.set_global_monitors([gather[0], dict(gather[1], observed_terms=["V", "W"])], [window])
    with pytest.raises(RuntimeError, match="complete, disjoint node gather"):
        sim.set_global_monitors([gather[0], dict(gather[1], node_indices=[0, 1, 2])], [window])
    with pytest.raises(RuntimeError, match="mixed-rate gather requires complete-window monitors"):
        sim.set_global_monitors(gather, [window, dict(slot=1, kind=8)])  # no generic global Projection
    sim.set_global_monitors(gather, [window])
    with pytest.raises(RuntimeError, match="detach before changing observed rows"):
        sim.set_voi(1, [0, 1])
    with pytest.raises(RuntimeError, match="detach before changing the clock"):
        sim.set_opts(0, 2 * DT, 1)
    with pytest.raises(ValueError, match="commensurate with master dt"):
        compiled.run(4, monitors=_windows(period=3.5 * DT), initial_states=INITIAL)
    assert sim.t_abs() == 0
