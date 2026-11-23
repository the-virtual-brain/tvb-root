"""Native complete windows on one clock across indexed regional sources."""

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
             observed=(("W", "V"), ("W", "V")), names=("small", "large")):
    sources = []
    for i, (indices, k, terms, name) in enumerate(zip(maps, cadence, observed, names)):
        # With all nonlinear/linear coefficients zero, V'=I and W'=a.
        params = dict.fromkeys(("alpha", "beta", "b", "c", "e", "f", "g"), [0.0])
        params.update(d=[1.0], tau=[1.0], gamma=[1.0], I=SLOPE[i][0, :, 0], a=SLOPE[i][1, :, 0])
        model = models.Generic2dOscillator(variables_of_interest=terms,
                                          **{key: np.array(value) for key, value in params.items()})
        dt = k * DT
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


def _assert_streams(actual, expected):
    assert len(actual) == len(expected)
    for (times, data), (ref_times, ref_data) in zip(actual, expected):
        np.testing.assert_array_equal(times, ref_times)
        assert data.shape == ref_data.shape
        np.testing.assert_allclose(data, ref_data, rtol=0, atol=2e-6)


def _unpack_global(streams):
    return [stream for (stream,) in streams]


@pytest.mark.parametrize("stochastic", [False, True], ids=["one-native-call", "numpy-chunks"])
def test_coclock_global_windows_gather_unequal_sources_across_chunks(cpp, monkeypatch, stochastic):
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
    actual = cpp.CppHybridBackend().run_network(_network(stochastic=stochastic), 11, chunk_size=7,
                                                monitors=_windows(), initial_states=INITIAL)
    assert calls == ([7, 4] if stochastic else [11])
    _assert_streams(_unpack_global(actual), _oracle(11))


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
    with pytest.raises(RuntimeError, match="global co-clock complete window"):
        compiled.resume(snapshot, 4, monitors=requested)
    assert compiled._sim.t_abs() == 7  # no integration on a rejected request


@pytest.mark.parametrize("change", ["missing", "duplicate", "legacy", "origin", "phase", "nonfinite"])
def test_coclock_global_windows_native_restore_is_transactional(cpp, change):
    compiled = cpp.CppHybridBackend().compile(_network())
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
    with pytest.raises(RuntimeError, match="global co-clock complete window"):
        sim.set_global_monitor_states(candidate)
    for orig, current in zip(before, sim.get_global_monitor_states()):
        np.testing.assert_array_equal(current["accum"], orig["accum"])
    _assert_streams(sim.take_global_monitor_outputs(), _oracle(9, start=6))


@pytest.mark.parametrize("maps", [((0, 0), (4, 1, 2)), ((3, 0), (4, 1, 7)),
                                  ((2**32 + 3, 0), (4, 1, 2))])
def test_coclock_global_windows_reject_invalid_node_indices(cpp, maps):
    compiled = cpp.CppHybridBackend().compile(_network(maps=maps))
    with pytest.raises(ValueError, match="node_indices"):
        compiled.run(4, monitors=_windows(), initial_states=INITIAL)
    assert compiled._sim.t_abs() == 0


def test_coclock_global_windows_check_native_source_and_narrowing(cpp):
    compiled = cpp.CppHybridBackend().compile(_network())
    gather = compiled._coclock_window_gather(DT)
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


@pytest.mark.parametrize("profile", ["mixed-rate", "missing-indices", "heterogeneous-order",
                                     "host", "unprepared-spatial"])
def test_coclock_global_windows_leave_other_routes_unstaged(cpp, profile):
    be = cpp.CppHybridBackend()
    requested = [monitors.TemporalAverage(period=4 * DT)]
    if profile == "mixed-rate":
        net = _network(cadence=(1, 2))
    elif profile == "missing-indices":
        net = _network(maps=(None, None))
    elif profile == "heterogeneous-order":
        net = _network(observed=(("W", "V"), ("V", "W")))
    else:
        net = _network()
        if profile == "host":
            be.enable_kernel_monitors = False
        else:
            requested = [monitors.SpatialAverage(period=4 * DT)]
    outputs, snapshot = be.compile(net).run(8, monitors=requested, initial_states=INITIAL, return_snapshot=True)
    assert not any("coclock_window_config" in state for state in snapshot["kmon_global_states"])
    assert len(outputs[0]) == (1 if profile == "unprepared-spatial" else 2)


def test_coclock_global_windows_reject_noncommensurate_period_and_invalid_operator(cpp):
    compiled = cpp.CppHybridBackend().compile(_network())
    with pytest.raises(ValueError, match="commensurate"):
        compiled.run(4, monitors=_windows(period=3.5 * DT), initial_states=INITIAL)
    windows = _windows()
    windows[-1].spatial_mean[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite preconfigured spatial_mean"):
        compiled.run(4, monitors=windows, initial_states=INITIAL)
    assert compiled._sim.t_abs() == 0
