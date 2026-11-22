"""Regional C++ EEG/MEG/iEEG use the native complete-window stream."""

import numpy as np
import pytest

from tvb.datatypes.projections import ProjectionSurfaceEEG
from tvb.datatypes.sensors import SensorsEEG, SensorsInternal, SensorsMEG
from tvb.simulator import integrators, models, monitors, noise
from tvb.simulator.hybrid import NetworkSet, Simulator, Subnetwork
from tvb.simulator.hybrid.recorder import Recorder


DT = 0.125
INITIAL = np.array([[0.25, -0.5, 1.0], [2.0, -1.0, 0.5]])[..., None]
SLOPE = np.array([[0.0, 1.0, -0.5], [0.5, -0.25, 2.0]])[..., None]
GAIN = np.array([[1., -0.5, 2.], [-2., 0.25, 0.5], [0.5, 0., -1.], [2., 1., 0.25]])


@pytest.fixture
def cpp():
    return pytest.importorskip("tvb.simulator.backend.cpp_hybrid.backend")


def _subnet(name="regions", *, stochastic=False, indices=(0, 1, 2), voi_order=("W", "V")):
    params = dict.fromkeys(("alpha", "beta", "b", "c", "e", "f", "g"), [0.0])
    params.update(d=[1.0], tau=[1.0], gamma=[1.0], I=SLOPE[0, :, 0], a=SLOPE[1, :, 0])
    model = models.Generic2dOscillator(
        variables_of_interest=voi_order, **{key: np.array(v) for key, v in params.items()},
    )
    scheme = (integrators.EulerStochastic(dt=DT, noise=noise.Additive(nsig=np.zeros(2)))
              if stochastic else integrators.EulerDeterministic(dt=DT))
    return Subnetwork(name=name, nnodes=3, model=model, scheme=scheme,
                      node_indices=None if indices is None else np.array(indices, dtype=np.int64))


def _network(**opts):
    return NetworkSet(subnets=[_subnet(**opts)]).configure()


def _projection(period=4, modality=monitors.EEG, reference=None):
    sensor_class = {monitors.EEG: SensorsEEG, monitors.MEG: SensorsMEG, monitors.iEEG: SensorsInternal}
    sensors = sensor_class[modality](labels=np.array(["z", "b", "a", "c"]),
                                     locations=np.zeros((4, 3)),
                                     usable=np.array([True, True, False, True]))
    mon = modality(period=period * DT, sensors=sensors, obsnoise=None,
                   variables_of_interest=np.array([1, 0]))
    mon.gain = GAIN.copy()
    if modality == monitors.EEG:
        mon.reference = reference
        # This is not the effective reference for any of the requested tests.
        mon._ref_vec = np.array([1., 0., 0., 0.])
        mon._ref_vec_mask = np.ones(4, dtype=bool)
    return mon


def _post_step(steps):
    return INITIAL + np.asarray(steps)[:, None, None, None] * DT * SLOPE


def _oracle(period, nstep, reference=None):
    ends = np.arange(period, nstep + 1, period)
    means = _post_step(np.arange(1, len(ends) * period + 1)).reshape(-1, period, 2, 3, 1).mean(axis=1)
    # The model observes W,V; the monitor selects rows 1,0 (V,W).
    projected = np.einsum("tvrm,sr->tvsm", means, GAIN)
    projected[:, :, 2] = 0.0
    if reference == "average":
        projected -= projected.sum(axis=2, keepdims=True) / 4
    elif reference == "b":
        projected -= projected[:, :, 1:2].copy()
    # A named but unusable channel ("a") contributes zero, not another row.
    return (ends - period / 2) * DT, projected


def _assert_stream(actual, expected):
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1].shape == expected[1].shape
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-6, rtol=0)


def _trace(cpp, monkeypatch, *, data_offset=0):
    real_sim, events = cpp.Sim, []

    class TracedSim:
        def __init__(self, width):
            self.sim = real_sim(width)

        def __getattr__(self, name):
            return getattr(self.sim, name)

        def set_global_monitors(self, gather, attachments):
            events.append(("attach", attachments))
            self.sim.set_global_monitors(gather, attachments)

        def run(self, nstep, *args, **kwargs):
            events.append(("run", nstep))
            return self.sim.run(nstep, *args, **kwargs)

        def take_global_monitor_outputs(self):
            streams = self.sim.take_global_monitor_outputs()
            events.append(("drain", streams))
            return [(times, data + data_offset) for times, data in streams]

    monkeypatch.setattr(cpp, "Sim", TracedSim)
    return events


@pytest.mark.parametrize("period", [1, 3, 4])
@pytest.mark.parametrize("modality,reference", [
    (monitors.EEG, None), (monitors.EEG, "b"), (monitors.EEG, "average"), (monitors.EEG, "a"),
    (monitors.MEG, None), (monitors.iEEG, None),
])
def test_hybrid_projection_consumes_native_complete_windows(cpp, monkeypatch, period, modality, reference):
    events = _trace(cpp, monkeypatch, data_offset=8)
    nstep = 2 * period + 1
    mon = _projection(period, modality, reference)
    sim = Simulator(nets=_network(), backend="cpp", monitors=[mon], simulation_length=nstep * DT)
    assert mon.voi == slice(None)  # lowering must use the ordered variables_of_interest instead
    [public] = sim.run(initial_conditions=[INITIAL])
    assert [name for name, _ in events] == ["attach", "run", "drain"]
    [spec] = events[0][1]
    assert spec["kind"] == 8 and spec["complete_window"] is True
    assert spec["voi"] == [1, 0] and spec["istep"] == period
    [native] = events[-1][1]
    expected_t, expected_x = _oracle(period, nstep, reference)
    _assert_stream(native, (expected_t, expected_x))
    _assert_stream(public, (expected_t, expected_x + 8))


@pytest.mark.parametrize("chunk,drain_sizes", [(1, [0, 0, 0, 1, 0, 0, 0, 1, 0]),
                                                (3, [0, 1, 1]), (8, [2, 0])])
def test_projection_chunks_and_numpy_drains_preserve_windows_and_drop_tail(cpp, monkeypatch, chunk, drain_sizes):
    events = _trace(cpp, monkeypatch)
    [[stream]] = cpp.CppHybridBackend().run_network(
        _network(stochastic=True), 9, chunk_size=chunk,
        monitors=[_projection()], initial_states=[INITIAL],
    )
    assert [kind for kind, _ in events] == ["attach"] + ["run", "drain"] * len(drain_sizes)
    assert [len(streams[0][0]) for kind, streams in events if kind == "drain"] == drain_sizes
    _assert_stream(stream, _oracle(4, 9))


def test_projection_with_raw_and_implicit_local_node_order(cpp, monkeypatch):
    events = _trace(cpp, monkeypatch)
    [raw], [projected] = cpp.CppHybridBackend().run_network(
        _network(indices=None), 9, monitors=[monitors.Raw(), _projection()], initial_states=[INITIAL],
    )
    steps = np.arange(1, 10)
    _assert_stream(raw, (steps * DT, _post_step(steps)[:, [1, 0]]))
    _assert_stream(projected, _oracle(4, 9))
    assert events[0][1][0]["slot"] == 1


def test_projection_accepts_reference_free_regional_gain_without_sensor_metadata(cpp):
    mon = monitors.EEG(period=DT, obsnoise=None)
    mon.gain = GAIN.copy()
    [[stream]] = cpp.CppHybridBackend().run_network(
        _network(), 1, monitors=[mon], initial_states=[INITIAL],
    )
    expected = np.einsum("tvrm,sr->tvsm", _post_step([1])[:, [1, 0]], GAIN)
    _assert_stream(stream, ([DT / 2], expected))


def test_projection_rejects_default_observation_noise_before_native_step(cpp, monkeypatch):
    events = _trace(cpp, monkeypatch)
    mon = monitors.MEG(period=DT)
    mon.gain = GAIN.copy()
    with pytest.raises(NotImplementedError, match="obsnoise=None"):
        cpp.CppHybridBackend().run_network(_network(), 1, monitors=[mon])
    assert events == []


def test_projection_resume_with_open_window_and_no_legacy_sidecar(cpp):
    be, net = cpp.CppHybridBackend(), _network()
    mon = _projection(reference="average")
    [[part]], snapshot = be.compile(net).run(
        5, chunk_size=3, monitors=[mon], initial_states=[INITIAL], return_snapshot=True,
    )
    state, = snapshot["kmon_global_states"]
    assert state["kind"] == 8 and state["accum_count"] == 1 and "window_config" in state
    assert "bw_projection_identities" not in snapshot
    [[tail]] = be.compile(net).resume(snapshot, 4, chunk_size=8, monitors=[mon])
    _assert_stream(tuple(np.concatenate(items) for items in zip(part, tail)), _oracle(4, 9, "average"))

    # Omitting monitors deliberately requests neural state only.
    [(times, data, _)] = be.compile(net).resume(snapshot, 4, chunk_size=1)
    steps = np.arange(6, 10)
    _assert_stream((times, data), (steps * DT, _post_step(steps)[:, [1, 0]]))


@pytest.mark.parametrize("change", ["gain", "reference", "period", "voi", "source", "historical",
                                    "missing_identity", "missing_companion", "missing_projection"])
def test_projection_resume_rejects_incompatible_requested_set(cpp, monkeypatch, change):
    events = _trace(cpp, monkeypatch)
    be, net, mon = cpp.CppHybridBackend(), _network(), _projection(reference="b")
    mons = [mon, monitors.GlobalAverage(period=6 * DT)]
    _, snapshot = be.compile(net).run(5, monitors=mons, initial_states=[INITIAL], return_snapshot=True)
    events.clear()
    if change == "gain":
        mon.gain[0, 0] += 1
    elif change == "reference":
        mon.reference = "average"
    elif change == "period":
        mon.period += DT
    elif change == "voi":
        mon.variables_of_interest = np.array([0, 1])
    elif change == "source":
        net = _network(voi_order=("V", "W"))
    elif change == "historical":
        del snapshot["kmon_global_states"][0]["window_config"]
    elif change == "missing_identity":
        snapshot["kmon_global_states"].pop(0)
    elif change == "missing_companion":
        mons.pop()
    else:
        mons.pop(0)
    with pytest.raises(RuntimeError, match="snapshot"):
        be.compile(net).resume(snapshot, 4, monitors=mons)
    assert all(kind != "run" for kind, _ in events)


@pytest.mark.parametrize("changes,message", [
    ({"obsnoise": noise.Additive(nsig=np.zeros(1))}, "obsnoise=None"),
    ({"gain": None}, "preconfigured"), ({"gain": np.zeros(3)}, "preconfigured"),
    ({"gain": np.zeros((0, 3))}, "preconfigured"), ({"gain": np.ones((4, 2))}, "preconfigured"),
    ({"gain": np.full((4, 3), np.nan)}, "preconfigured"), ({"gain": np.full((4, 3), 1e50)}, "preconfigured"),
    ({"gain": np.full((4, 3), 1j)}, "preconfigured"),
    ({"variables_of_interest": np.array([2**32], dtype=np.int64)}, "observed VOI"),
    ({"variables_of_interest": np.array([-1])}, "observed VOI"),
    ({"variables_of_interest": np.array([[1]])}, "observed VOI"),
    ({"reference": "missing"}, "must name one sensor"),
    ({"reference": "b", "sensors": None}, "ordered sensor labels"),
    ({"sensors": SensorsEEG(labels=np.array(["z"]))}, "labels must align"),
    ({"sensors": SensorsEEG(labels=np.array(["z", "b", "a", "c"]), usable=np.ones(2, dtype=bool))},
     "mask must align"),
])
def test_projection_rejects_invalid_effective_operator_before_native_step(cpp, monkeypatch, changes, message):
    events = _trace(cpp, monkeypatch)
    mon = _projection()
    # A file matrix of the same shape does not make it a prepared regional gain.
    mon.projection = ProjectionSurfaceEEG(projection_data=GAIN.copy())
    if "sensors" in changes and changes["sensors"] is None:
        del mon.__dict__["sensors"]
    for name, value in changes.items():
        if name != "sensors" or value is not None:
            setattr(mon, name, value)
    with pytest.raises((ValueError, NotImplementedError), match=message):
        cpp.CppHybridBackend().run_network(_network(), 5, monitors=[mon], initial_states=[INITIAL])
    assert events == []


@pytest.mark.parametrize("layout,message", [
    ("host", "native monitor collection"), ("compiled_width", "width 1"), ("requested_width", "width 1"),
    ("multiple_subnets", "one regional subnetwork"), ("permutation", "identity node_indices"),
    ("fractional_period", "commensurate"), ("recorder", "local or Recorder"),
    ("local_monitor", "local or Recorder"), ("local_recorder", "local or Recorder"),
    ("sweep", "Unsupported sweep monitor"), ("legacy_sweep", "not supported in sweeps"),
    ("legacy_sweep_key", "not supported in sweeps"), ("legacy_sweep_name", "not supported in sweeps"),
])
def test_projection_rejects_out_of_profile_requests_before_native_step(cpp, monkeypatch, layout, message):
    events = _trace(cpp, monkeypatch)
    be, net, mon = cpp.CppHybridBackend(), _network(), _projection()
    with pytest.raises((ValueError, NotImplementedError), match=message):
        if layout == "sweep":
            be.sweep(net, {}, monitor="eeg")
        elif layout == "legacy_sweep":
            be.run_sweep(net, np.zeros(1), monitors=[mon])
        elif layout == "legacy_sweep_key":
            be.run_sweep(net, np.zeros(1), eeg=True)
        elif layout == "legacy_sweep_name":
            be.run_sweep(net, np.zeros(1), monitor="iEEG")
        elif layout == "compiled_width":
            be.compile(net, width=4).run(5, monitors=[mon])
        elif layout == "recorder":
            sn = _subnet()
            sn.monitors = [Recorder(monitor=mon)]
            Simulator(nets=NetworkSet(subnets=[sn]), backend="cpp")
        elif layout in ("local_monitor", "local_recorder"):
            local = mon if layout == "local_monitor" else Recorder(monitor=mon)
            be.run_network(net, 5, subnet_monitors={"regions": [local]})
        else:
            if layout == "host":
                be.enable_kernel_monitors = False
            elif layout == "multiple_subnets":
                net = NetworkSet(subnets=[_subnet(), _subnet("other", indices=(3, 4, 5))]).configure()
            elif layout == "permutation":
                net = _network(indices=(2, 0, 1))
            elif layout == "fractional_period":
                mon.period = 3.5 * DT
            opts = {"width": 4} if layout == "requested_width" else {}
            be.run_network(net, 5, monitors=[mon], **opts)
    assert events == []
