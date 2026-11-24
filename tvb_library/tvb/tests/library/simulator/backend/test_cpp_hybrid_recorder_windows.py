"""Native Recorder windows use each subnet's post-update source clock."""

from copy import deepcopy

import numpy as np
import pytest

from tvb.datatypes.equations import Linear
from tvb.datatypes.patterns import StimuliRegion
from tvb.simulator import integrators, models, monitors, noise
from tvb.simulator.hybrid import NetworkSet, Simulator, Subnetwork
from tvb.simulator.hybrid.recorder import Recorder
from tvb.simulator.hybrid.stimulus import Stim


DT = 0.125
NAMES = ("fast", "slow")
CADENCE = (1, 2)
PERIODS = (3, 2)
INITIAL = np.array([[[0.25, -0.5, 1.0], [2.0, -1.0, 0.5]],
                    [[4.0, -2.0, 0.5], [-1.0, 3.0, 4.0]]])[..., None]
SLOPE = np.array([[[0.0, 1.0, -0.5], [0.5, -0.25, 2.0]],
                  [[2.0, 0.25, -1.0], [1.0, -2.0, 0.5]]])[..., None]
WEIGHTS = np.array([[0.25, 0.0, 0.75], [0.0, 0.625, 0.375]])
WINDOWS = (monitors.TemporalAverage, monitors.GlobalAverage, monitors.SpatialAverage)


@pytest.fixture
def cpp():
    return pytest.importorskip("tvb.simulator.backend.cpp_hybrid.backend")


def _network(*, stochastic=False, master_dt=DT, slow_voi_order=("W", "V"), slow_nodes=(5, 3, 4)):
    subnets = []
    for i, name in enumerate(NAMES):
        # With all other terms zero, Generic2dOscillator advances V by dt*I and W by dt*a.
        params = dict.fromkeys(("alpha", "beta", "b", "c", "e", "f", "g"), [0.0])
        params.update(d=[1.0], tau=[1.0], gamma=[1.0], I=SLOPE[i, 0, :, 0], a=SLOPE[i, 1, :, 0])
        model = models.Generic2dOscillator(
            variables_of_interest=("W", "V") if i == 0 else slow_voi_order,
            **{key: np.array(value) for key, value in params.items()},
        )
        dt = master_dt * CADENCE[i]
        scheme = (integrators.EulerStochastic(dt=dt, noise=noise.Additive(nsig=np.zeros(2)))
                  if stochastic else integrators.EulerDeterministic(dt=dt))
        subnets.append(Subnetwork(name=name, nnodes=3, model=model, scheme=scheme,
                                  node_indices=np.array((2, 0, 1) if i == 0 else slow_nodes)))
    return NetworkSet(subnets=subnets).configure()


def _specs(*, periods=PERIODS, points=False, master_dt=DT):
    specs = {}
    for name, k, m in zip(NAMES, CADENCE, periods):
        dt = master_dt * k
        windows = [cls(period=m * dt, variables_of_interest=np.array([1, 0])) for cls in WINDOWS]
        windows[-1].spatial_mean = WEIGHTS.copy()
        if points:
            windows += [monitors.Raw(), monitors.RawVoi(variables_of_interest=np.array([1])),
                        monitors.SubSample(period=2 * dt, variables_of_interest=np.array([1, 0]))]
        specs[name] = windows
    return specs


def _oracle(i, stop, *, start=0, m=None, points=False, master_dt=DT):
    k = CADENCE[i]
    dt = k * master_dt
    m = PERIODS[i] if m is None else m

    def post_update(j):
        return INITIAL[i] + np.asarray(j)[..., None, None, None] * dt * SLOPE[i]

    j = np.arange(start // k + 1, stop // k + 1)
    closing = j[j % m == 0]
    times = (closing - m / 2) * dt
    means = post_update(closing[:, None] - np.arange(m)).mean(axis=1)
    result = [(times, means), (times, means.mean(axis=2, keepdims=True)),
              (times, np.einsum("tvnm,an->tvam", means, WEIGHTS))]
    if points:
        point_times = j * dt
        src = post_update(j)
        take = j % 2 == 0
        result += [(point_times, src[:, [1, 0]]), (point_times, src[:, [0]]),
                   (point_times[take], src[take])]
    return result


def _assert_streams(streams, expected):
    assert len(streams) == len(expected)
    for (t, data), (exp_t, exp_data) in zip(streams, expected):
        assert t.shape == exp_t.shape and data.shape == exp_data.shape
        np.testing.assert_array_equal(t, exp_t)
        np.testing.assert_allclose(data, exp_data, atol=2e-6, rtol=0)


def _trace_native_runs(cpp, monkeypatch):
    real_sim, calls = cpp.Sim, []

    class TracedSim:
        def __init__(self, width):
            self.sim = real_sim(width)

        def __getattr__(self, name):
            return getattr(self.sim, name)

        def run(self, nstep, *args, **kwargs):
            calls.append(nstep)
            return self.sim.run(nstep, *args, **kwargs)

        def take_global_monitor_outputs(self):
            pytest.fail("local Recorder streams must be delivered in the packed native result")

    monkeypatch.setattr(cpp, "Sim", TracedSim)
    return calls


@pytest.mark.parametrize("mode", ["deterministic", "numpy", "philox"])
def test_recorder_windows_post_update_values_through_native_dispatch(cpp, monkeypatch, mode):
    calls = _trace_native_runs(cpp, monkeypatch)
    be = cpp.CppHybridBackend()
    _, streams = be.run_network(_network(stochastic=mode != "deterministic"), 11, chunk_size=7,
                                initial_states=INITIAL, subnet_monitors=_specs(points=True),
                                rng_mode="philox" if mode == "philox" else "numpy")
    assert calls == ([7, 4] if mode == "numpy" else [11])
    for i, name in enumerate(NAMES):
        _assert_streams(streams[name], _oracle(i, 11, points=True))
    np.testing.assert_array_equal(streams["fast"][0][0], [0.1875, 0.5625, 0.9375])
    np.testing.assert_array_equal(streams["slow"][0][0], [0.25, 0.75])


def test_recorder_windows_release_only_on_closing_master_ticks(cpp):
    cc = cpp.CppHybridBackend().compile(_network())
    _, first = cc.run(1, initial_states=INITIAL, subnet_monitors=_specs())
    for i, name in enumerate(NAMES):
        _assert_streams(first[name], _oracle(i, 1))
    releases = [[], []]
    for tick in range(2, 12):
        packed = cc._sim.run(1, 1, None, None, need_tavg=False, need_ctavg=False)
        for i in range(2):
            # Each subnet packs two chunk outputs then three (time,data) pairs.
            start = i * 8 + 2
            local = list(zip(packed[start:start + 6:2], packed[start + 1:start + 6:2]))
            _assert_streams(local, _oracle(i, tick, start=tick - 1))
            if local[0][0].size:
                releases[i].append(tick)
    assert releases == [[3, 6, 9], [4, 8]]
    assert [[s["accum_count"] for s in local] for local in cc._sim.get_monitor_states()] == [[2] * 3, [1] * 3]


@pytest.mark.parametrize("split,counts", [(7, (1, 1)), (1, (1, 0))])
def test_recorder_windows_restore_on_fresh_compiled_sim(cpp, split, counts):
    be, net, specs = cpp.CppHybridBackend(), _network(), _specs(points=True)
    _, snap, head = be.compile(net).run(split, chunk_size=7, initial_states=INITIAL,
                                        subnet_monitors=specs, return_snapshot=True)
    assert snap["step"] == split
    for i, (k, count) in enumerate(zip(CADENCE, counts)):
        windows = snap["kmon_states"][i][:3]
        assert [(s["m_step"], s["run_start"], s["accum_count"]) for s in windows] == [(split // k, 0, count)] * 3
        assert all(s["local_window_phase"] == split % k for s in windows)
        # Restore uses identity rather than checkpoint list position.
        snap["kmon_states"][i].reverse()
    _, tail = be.compile(net).resume(snap, 11 - split, chunk_size=3, subnet_monitors=specs)
    for i, name in enumerate(NAMES):
        _assert_streams(head[name], _oracle(i, split, points=True))
        _assert_streams(tail[name], _oracle(i, 11, start=split, points=True))
        whole = [tuple(np.concatenate(x) for x in zip(h, t)) for h, t in zip(head[name], tail[name])]
        _assert_streams(whole, _oracle(i, 11, points=True))


def test_recorder_windows_same_slot_rebuild_origin_and_one_tick_midpoints(cpp):
    # Choose a decimal clock not exactly representable in the native float32 integrator.
    master_dt = 0.1
    sim = cpp.CppHybridBackend().compile(_network(master_dt=master_dt))
    sim.run(7, initial_states=INITIAL, subnet_monitors=_specs(master_dt=master_dt))
    specs = _specs(periods=(1, 1), master_dt=master_dt)
    specs["slow"][2].spatial_mean = 2 * WEIGHTS
    _, streams = sim.run(5, initial_states=INITIAL, subnet_monitors=specs)
    for i, name in enumerate(NAMES):
        expected = _oracle(i, 5, m=1, master_dt=master_dt)
        if i == 1:
            t, values = expected[2]
            expected[2] = t, 2 * values
        _assert_streams(streams[name], expected)
        assert streams[name][0][0][0] == master_dt * CADENCE[i] / 2


@pytest.mark.parametrize("change", ["period", "voi", "weights", "source", "node-order", "missing",
                                    "duplicates", "without-recorders", "point-instead", "historical",
                                    "phase", "count", "origin", "nan", "extent", "rank", "extra-subnet",
                                    "missing-container"])
def test_recorder_window_resume_requires_exact_valid_window_set(cpp, monkeypatch, change):
    calls = _trace_native_runs(cpp, monkeypatch)
    be, net, specs = cpp.CppHybridBackend(), _network(), _specs()
    _, snap, _ = be.compile(net).run(7, initial_states=INITIAL, subnet_monitors=specs, return_snapshot=True)
    calls.clear()
    slow = snap["kmon_states"][1]
    if change == "period":
        specs["slow"][0].period += 2 * DT
    elif change == "voi":
        specs["slow"][0].variables_of_interest = np.array([0, 1])
    elif change == "weights":
        specs["slow"][2].spatial_mean[0, 1] = 0.25
    elif change == "source":
        net = _network(slow_voi_order=("V", "W"))
    elif change == "node-order":
        net = _network(slow_nodes=(3, 4, 5))
    elif change == "missing":
        slow.pop()
    elif change == "duplicates":
        slow.append(deepcopy(slow[-1]))
    elif change == "without-recorders":
        specs = None
    elif change == "point-instead":
        specs["slow"][0] = monitors.Raw()
    elif change == "historical":
        del slow[0]["local_window_config"]
    elif change == "phase":
        slow[-1]["m_step"] += 2
    elif change == "count":
        slow[-1]["accum_count"] = 0
    elif change == "origin":
        slow[-1]["run_start"] = 2
    elif change == "nan":
        slow[-1]["accum"][0] = np.nan
    elif change == "extent":
        slow[-1]["accum"] = slow[-1]["accum"][1:]
    elif change == "rank":
        slow[-1]["accum"] = slow[-1]["accum"][None, :]
    elif change == "extra-subnet":
        snap["kmon_states"].append(deepcopy(slow))
    elif change == "missing-container":
        del snap["kmon_states"]
    with pytest.raises((ValueError, RuntimeError), match="Recorder complete window"):
        be.compile(net).resume(snap, 4, subnet_monitors=specs)
    assert calls == []


@pytest.mark.parametrize("altered_step,error", [(0, "master cadence phase"), (1.5, "integer snapshot master tick")])
def test_slow_recorder_restore_checks_master_phase_before_first_update(cpp, monkeypatch, altered_step, error):
    calls = _trace_native_runs(cpp, monkeypatch)
    be, net, specs = cpp.CppHybridBackend(), _network(), {"slow": _specs()["slow"]}
    _, snap, _ = be.compile(net).run(1, initial_states=INITIAL, subnet_monitors=specs, return_snapshot=True)
    calls.clear()
    snap["step"] = altered_step
    with pytest.raises((ValueError, RuntimeError), match=error):
        be.compile(net).resume(snap, 10, subnet_monitors=specs)
    assert calls == []


def test_recorder_window_native_restore_validates_all_before_commit(cpp):
    sim = cpp.CppHybridBackend().compile(_network())
    sim.run(8, initial_states=INITIAL, subnet_monitors=_specs())
    state = sim._sim.get_monitor_states()
    corrupted = deepcopy(state)
    corrupted[0][0]["accum"][0] += 50  # valid nonempty buffer; must not commit it
    corrupted[1][-1]["accum"][0] = 1  # a closed window must have a zero buffer
    with pytest.raises(RuntimeError, match="invalid snapshot state or clock"):
        sim._sim.set_monitor_states(corrupted)
    np.testing.assert_equal(sim._sim.get_monitor_states(), state)
    assert sim._sim.t_abs() == 8


@pytest.mark.parametrize("change,error", [
    ("width", "incompatible sibling snapshot geometry"),
    ("space", "incompatible sibling snapshot geometry"),
    ("buffer", "incompatible sibling snapshot buffers"),
    ("slot", "unmatched or ambiguous snapshot monitor"),
    ("duplicate", "duplicate snapshot monitor"),
    ("missing", "missing compatible snapshot monitor"),
    pytest.param("missing-step", "invalid sibling snapshot counter: m_step", id="missing-step"),
    pytest.param("float-step", "invalid sibling snapshot counter: m_step", id="float-step"),
    pytest.param("bool-origin", "invalid sibling snapshot counter: run_start", id="bool-origin"),
    pytest.param("late-origin", "incompatible sibling snapshot counter phase", id="late-origin"),
])
def test_recorder_window_native_restore_rejects_bad_legacy_sibling_without_committing(cpp, change, error):
    sim = cpp.CppHybridBackend().compile(_network())
    sim.run(7, initial_states=INITIAL, subnet_monitors=_specs(points=True))
    state = sim._sim.get_monitor_states()
    corrupted = deepcopy(state)
    corrupted[0][0]["accum"][0] += 50  # valid open-window change
    point = corrupted[0].pop(3)
    point["m_step"] += 2  # valid legacy counter change, before the bad sibling
    corrupted[0].insert(0, point)
    slow = corrupted[1]
    if change == "width":
        slow[-1]["width"] += 1
    elif change == "space":
        slow[-1]["n_out_space"] += 1
    elif change == "buffer":
        slow[-1]["accum"] = np.zeros(1)  # point monitors have no accumulator
    elif change == "slot":
        slow[-1]["slot"] += 100  # the same-kind positional fallback must not run
    elif change == "duplicate":
        slow.append(deepcopy(slow[-1]))
    elif change == "missing":
        slow.pop()
    elif change == "missing-step":
        slow[-1].pop("m_step")
    elif change == "float-step":
        slow[-1]["m_step"] = float(slow[-1]["m_step"])
    elif change == "bool-origin":
        slow[-1]["run_start"] = False
    elif change == "late-origin":
        slow[-1]["run_start"] = slow[-1]["m_step"] + 1
    with pytest.raises(RuntimeError, match=error):
        sim._sim.set_monitor_states(corrupted)
    np.testing.assert_equal(sim._sim.get_monitor_states(), state)
    assert sim._sim.t_abs() == 7


def test_recorder_window_native_restore_commits_reordered_mixed_set(cpp):
    sim = cpp.CppHybridBackend().compile(_network())
    sim.run(7, initial_states=INITIAL, subnet_monitors=_specs(points=True))
    expected = sim._sim.get_monitor_states()
    expected[0][0]["accum"][0] += 50
    expected[0][3]["m_step"] += 2
    sim._sim.set_monitor_states([list(reversed(local)) for local in expected])
    np.testing.assert_equal(sim._sim.get_monitor_states(), expected)


@pytest.mark.parametrize("change", ["bool-version", "float-version", "nested-tuple", "nested-int",
                                    "nested-float", "container-equality", "scalar-equality"])
def test_recorder_window_native_restore_requires_builtin_config_schema(cpp, change):
    sim = cpp.CppHybridBackend().compile(_network())
    sim.run(7, initial_states=INITIAL, subnet_monitors=_specs())
    state = sim._sim.get_monitor_states()
    corrupted = deepcopy(state)
    saved = corrupted[1][-1]
    config = list(saved["local_window_config"])
    comparisons = []

    class EqualScalar(int):
        def __eq__(self, other):
            comparisons.append(other)
            return True

    class EqualContainer(tuple):
        def __eq__(self, other):
            comparisons.append(other)
            return True

    if change == "bool-version":
        config[0] = True
    elif change == "float-version":
        config[0] = 1.0
    elif change == "nested-tuple":
        config[10][0] = list(config[10][0])  # an observation is a tuple
    elif change == "nested-int":
        config[9][0] = bool(config[9][0])  # the VOI index is an int
    elif change == "nested-float":
        config[-1][0] = np.float64(config[-1][0])  # weights are built-in floats
    elif change == "scalar-equality":
        config[9][0] = EqualScalar(config[9][0])
    saved["local_window_config"] = EqualContainer(config) if change == "container-equality" else tuple(config)
    with pytest.raises(RuntimeError, match="incompatible snapshot configuration"):
        sim._sim.set_monitor_states(corrupted)
    assert not comparisons
    np.testing.assert_equal(sim._sim.get_monitor_states(), state)


def test_recorder_window_rejects_live_observation_and_clock_changes(cpp):
    sim = cpp.CppHybridBackend().compile(_network())
    sim.run(7, initial_states=INITIAL, subnet_monitors=_specs())
    before = sim._sim.get_monitor_states()
    with pytest.raises(RuntimeError, match="detach before changing observed rows"):
        sim._sim.set_voi(1, [0, 1])
    np.testing.assert_equal(sim._sim.get_monitor_states(), before)
    sim._sim.set_opts(0, DT / 2, 1)
    with pytest.raises(RuntimeError, match="incompatible source or master clock"):
        sim._sim.get_monitor_states()
    with pytest.raises(RuntimeError, match="incompatible source or master clock"):
        sim._sim.run(1, 1, None, None)
    assert sim._sim.t_abs() == 7


@pytest.mark.parametrize("overrides", [dict(cadence_K=1), dict(dt=DT), dict(master_dt=DT / 2)])
def test_recorder_window_native_attachment_checks_actual_cadence_and_clock(cpp, overrides):
    cc = cpp.CppHybridBackend().compile(_network())
    spec = cc._recorder_window_spec(cc._analysis.subnets[1], 0, _specs()["slow"][0])
    with pytest.raises(RuntimeError, match="incompatible source or master clock"):
        cc._sim.set_monitors(1, [dict(spec, **overrides)])


def test_recorder_window_native_checked_master_tick_range(cpp):
    cc = cpp.CppHybridBackend().compile(_network())
    cc.run(3, initial_states=INITIAL, subnet_monitors=_specs(periods=(1, 1)))
    states = cc._sim.get_monitor_states()
    tick = (1 << 64) - 2
    cc._sim.set_t_abs(tick)
    for k, entries in zip(CADENCE, states):
        for state in entries:
            state["m_step"] = tick // k
            state["local_window_phase"] = tick % k
    cc._sim.set_monitor_states(states)
    with pytest.raises(RuntimeError, match="run exceeds tick range"):
        cc._sim.run(2, 1, None, None)
    assert cc._sim.t_abs() == tick
    np.testing.assert_equal(cc._sim.get_monitor_states(), states)


@pytest.mark.parametrize("change,message", [("period", "commensurate with source dt"),
                                           ("spatial", "preconfigured spatial_mean"),
                                           ("disabled", "require native collection")])
def test_recorder_windows_ineligible_requests_do_not_fall_back(cpp, monkeypatch, change, message):
    calls = _trace_native_runs(cpp, monkeypatch)
    be, specs = cpp.CppHybridBackend(), _specs()
    if change == "period":
        specs["slow"][0].period = 3 * DT  # commensurate with master, not its source
    elif change == "spatial":
        del specs["slow"][2].spatial_mean
    elif change == "disabled":
        be.enable_kernel_monitors = False
    with pytest.raises((ValueError, NotImplementedError), match=message):
        be.compile(_network()).run(7, initial_states=INITIAL, subnet_monitors=specs)
    assert calls == []


@pytest.mark.parametrize("period, midpoint", [(1, False), (1, True), (2, True)],
                         ids=["direct-point", "period-one-midpoint", "multi-tick-average"])
def test_recorder_strict_restore_continues_legacy_ctavg(cpp, period, midpoint):
    be, net = cpp.CppHybridBackend(), _network()
    slow = net.subnets[1]
    source_dt = slow.scheme.dt
    spatial = np.array([1.0, 2.0, 4.0])
    # Make the open accumulator observable after restore: coupling ramps on
    # the master grid, and this Recorder's CTAVG sees only slow source ticks.
    pattern = StimuliRegion.from_weights(spatial, Linear(parameters={"a": 1.0, "b": 0.0}))
    slow.stimuli = [Stim(target=slow, stimulus=pattern, target_cvar=np.array([0]))]
    coupling = (monitors.AfferentCouplingTemporalAverage(period=period * source_dt)
                if midpoint else monitors.AfferentCoupling())
    requested = {slow.name: [monitors.TemporalAverage(period=2 * source_dt), coupling]}
    split, stop = 7, 13
    _, snap, head = be.compile(net).run(split, initial_states=INITIAL, subnet_monitors=requested,
                                        return_snapshot=True)
    window, state = snap["kmon_states"][1]
    assert window["accum_count"] == 1 and "local_window_config" in window
    assert state["kind"] == 5 and "local_window_config" not in state
    assert (state["m_step"], state["run_start"], state["accum_count"]) == (3, 0, 3 % period)
    if period > 1:
        np.testing.assert_array_equal(state["accum"], 3 * source_dt * spatial)

    snap["kmon_states"][1].reverse()
    _, tail = be.compile(net).resume(snap, stop - split, subnet_monitors=requested)
    _, whole = be.compile(net).run(stop, initial_states=INITIAL, subnet_monitors=requested)
    for (tail_t, tail_x), (head_t, _), (whole_t, whole_x) in zip(
            tail[slow.name], head[slow.name], whole[slow.name]):
        np.testing.assert_array_equal(tail_t, whole_t[head_t.size:])
        np.testing.assert_array_equal(tail_x, whole_x[head_t.size:])

    closes = np.arange(4, 7, period)
    times, values = tail[slow.name][1]
    np.testing.assert_array_equal(times, (closes - (period / 2 if midpoint else 0)) * source_dt)
    mean_coupling = (closes - (period - 1) / 2)[:, None] * source_dt * spatial
    np.testing.assert_array_equal(values, mean_coupling[:, None, :, None])


def test_recorder_legacy_coupling_average_and_neural_only_resume_are_unchanged(cpp):
    be, net = cpp.CppHybridBackend(), _network()
    monitors_by_name = {"slow": [monitors.AfferentCouplingTemporalAverage(period=3 * DT)]}
    _, snap, streams = be.compile(net).run(7, initial_states=INITIAL, return_snapshot=True,
                                         subnet_monitors=monitors_by_name)
    [state] = snap["kmon_states"][1]
    assert state["kind"] == 5 and "local_window_config" not in state
    assert streams["slow"][0][0].size == 1  # legacy period is rounded
    _, tail = be.compile(net).resume(snap, 4, return_snapshot=True)
    assert tail["step"] == 11 and tail["kmon_states"] == [[], []]


def test_facade_recorder_windows_replace_buffers_and_collect_in_one_pass(cpp, monkeypatch):
    calls = _trace_native_runs(cpp, monkeypatch)
    net = _network()
    specs = _specs()
    for sn in net.subnets:
        sn.monitors = [Recorder(monitor=m) for m in specs[sn.name]]
    sim = Simulator(nets=net, backend="cpp", simulation_length=10.5 * DT)
    assert sim.run(initial_conditions=list(INITIAL)) == []
    assert calls == [11]
    for i, sn in enumerate(net.subnets):
        streams = [rec.to_arrays() for rec in sn.monitors]
        _assert_streams(streams, _oracle(i, 11))
        assert all(times.dtype == np.float32 for times, _ in streams)
    sim.simulation_length = DT
    assert sim.run(initial_conditions=list(INITIAL)) == []
    assert calls == [11, 1]
    for sn in net.subnets:
        for rec in sn.monitors:
            assert rec._current_idx == 0 and rec.times is None and rec.samples is None
