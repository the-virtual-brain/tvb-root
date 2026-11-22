"""Public C++ regional window policy, with an independent post-step oracle."""

import numpy as np
import pytest

from tvb.simulator import integrators, models, monitors, noise
from tvb.simulator.hybrid import NetworkSet, Simulator, Subnetwork


DT = 0.125
INITIAL = np.array([[0.25, -0.5, 1.0], [2.0, -1.0, 0.5]])[..., None]
SLOPE = np.array([1.0, -0.5])[:, None, None]
SPATIAL = np.array([[0.75, 0.0, 0.25], [0.0, 1.0, 0.0]])
KINDS = ("TemporalAverage", "GlobalAverage", "SpatialAverage")
NOISE_SEED = 127
NSIG = np.array([0.02, 0.03])


@pytest.fixture
def backend():
    return pytest.importorskip("tvb.simulator.backend.cpp_hybrid.backend")


def _network(dt=DT, stochastic=False):
    params = dict.fromkeys(("alpha", "beta", "b", "c", "e", "f", "g"), 0.0)
    params.update(d=1.0, tau=1.0, gamma=1.0, I=1.0, a=-0.5)
    model = models.Generic2dOscillator(
        variables_of_interest=("V", "W"),
        **{k: np.array([v]) for k, v in params.items()},
    )
    scheme = (
        integrators.HeunStochastic(dt=dt, noise=noise.Additive(nsig=NSIG, noise_seed=NOISE_SEED))
        if stochastic else integrators.HeunDeterministic(dt=dt)
    )
    return NetworkSet(subnets=[Subnetwork(
        name="regional_input", model=model, scheme=scheme, nnodes=3,
        node_indices=np.array([2, 0, 1]),
    )]).configure()


def _monitor(kind, period=4 * DT):
    mon = getattr(monitors, kind)(period=period, variables_of_interest=np.array([1, 0]))
    if kind == "SpatialAverage":
        mon.spatial_mean = SPATIAL.copy()
    return mon


def _trajectory(nstep, dt=DT):
    return INITIAL + np.arange(1, nstep + 1)[:, None, None, None] * dt * SLOPE


def _oracle(kind, states, window):
    complete = len(states) // window
    values = states[:complete * window].reshape(complete, window, *INITIAL.shape).mean(axis=1)
    values = values[:, [1, 0]]
    if kind == "GlobalAverage":
        return values.mean(axis=2, keepdims=True)
    if kind == "SpatialAverage":
        return np.einsum("an,tvnm->tvam", SPATIAL, values)
    return values


def _assert_values(data, expected):
    assert data.shape == expected.shape
    np.testing.assert_allclose(data, expected, atol=1e-6, rtol=0)


def _trace_native_sim(backend, monkeypatch, *, add_to_values=0):
    real_sim = backend.Sim
    runs, drains = [], []

    class TracedSim:
        def __init__(self, width):
            self.sim = real_sim(width)

        def __getattr__(self, attr):
            return getattr(self.sim, attr)

        def run(self, *args, **kwargs):
            runs.append(args[0])
            return self.sim.run(*args, **kwargs)

        def take_global_monitor_outputs(self):
            outputs = self.sim.take_global_monitor_outputs()
            drains.append(outputs)
            if add_to_values:
                return [(ts, data + add_to_values) for ts, data in outputs]
            return outputs

    monkeypatch.setattr(backend, "Sim", TracedSim)
    return runs, drains


def test_hybrid_simulator_returns_native_global_stream_and_point_companions(backend, monkeypatch):
    runs, drains = _trace_native_sim(backend, monkeypatch, add_to_values=16)
    kinds = ("SpatialAverage", "TemporalAverage", "GlobalAverage")
    sim = Simulator(nets=_network(), backend="cpp", simulation_length=9 * DT, monitors=[
        _monitor(kinds[0]), monitors.RawVoi(variables_of_interest=np.array([1])),
        _monitor(kinds[1]), monitors.SubSample(period=3 * DT), _monitor(kinds[2]),
    ])
    sim.configure()
    outputs = sim.run(initial_conditions=[INITIAL.copy()])
    states = _trajectory(9)
    assert runs == [9] and len(drains) == 1 and len(drains[0]) == 3
    for kind, public, native in zip(kinds, outputs[::2], drains[0]):
        native_t, native_x = native
        np.testing.assert_array_equal(native_t, DT * np.array([2, 6]))
        _assert_values(native_x, _oracle(kind, states, 4))
        np.testing.assert_array_equal(public[0], native_t)
        np.testing.assert_array_equal(public[1], native_x + 16)
    raw_t, raw_x = outputs[1]
    np.testing.assert_array_equal(raw_t, DT * np.arange(1, 10))
    _assert_values(raw_x, states[:, [1]])
    sub_t, sub_x = outputs[3]
    np.testing.assert_array_equal(sub_t, DT * np.array([3, 6, 9]))
    _assert_values(sub_x, states[2::3])


@pytest.mark.parametrize("chunk", [None, 1, 3, 9])
@pytest.mark.parametrize("window", [1, 4])
def test_run_network_windows_independent_of_chunks(backend, chunk, window):
    outputs = backend.CppHybridBackend().run_network(
        _network(), 10, chunk_size=chunk, initial_states=[INITIAL],
        monitors=[_monitor(kind, window * DT) for kind in KINDS],
    )
    for kind, [(times, data)] in zip(KINDS, outputs):
        np.testing.assert_array_equal(times, DT * (np.arange(window, 11, window) - window / 2))
        _assert_values(data, _oracle(kind, _trajectory(10), window))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("period", [0.0, 0.4375, 4 * DT + 1e-10, np.inf])
def test_run_network_rejects_invalid_window_period(backend, kind, period):
    with pytest.raises(ValueError, match=r"commensurate with source dt.*positive integer"):
        backend.CppHybridBackend().run_network(_network(), 8, monitors=[_monitor(kind, period)])


@pytest.mark.parametrize("kind, n_space", zip(KINDS, (3, 1, 2)))
def test_run_network_accepts_max_window_with_short_default_chunk_run(backend, kind, n_space):
    period = int(np.iinfo(np.int32).max) * DT
    [[(times, data)]] = backend.CppHybridBackend().run_network(
        _network(), 2, chunk_size=None, initial_states=[INITIAL],
        monitors=[_monitor(kind, period)],
    )
    assert times.shape == (0,)
    assert data.shape == (0, 2, n_space, 1)


def test_run_network_accepts_decimal_roundoff(backend):
    [[(times, data)]] = backend.CppHybridBackend().run_network(
        _network(dt=0.1), 8, chunk_size=5, initial_states=[INITIAL],
        monitors=[_monitor("TemporalAverage", 0.3)],
    )
    np.testing.assert_allclose(times, [0.15, 0.45], atol=1e-15, rtol=0)
    _assert_values(data, _oracle("TemporalAverage", _trajectory(8, dt=0.1), 3))


@pytest.mark.parametrize("weights", [None, np.ones(3), np.ones((2, 2)), np.full((2, 3), np.nan)])
def test_run_network_requires_preconfigured_spatial_weights(backend, weights):
    mon = _monitor("SpatialAverage")
    mon.spatial_mean = weights
    with pytest.raises(ValueError, match="preconfigured spatial_mean"):
        backend.CppHybridBackend().run_network(_network(), 8, monitors=[mon])


def test_compiled_run_rejects_host_disabled_native_and_wide_lanes(backend):
    be = backend.CppHybridBackend()
    mon = _monitor("TemporalAverage")
    with pytest.raises(NotImplementedError, match="width 1 and one source clock and mode"):
        be.compile(_network(), width=4).run(8, monitors=[mon])
    be.enable_kernel_monitors = False
    with pytest.raises(NotImplementedError, match="require native monitor collection"):
        be.run_network(_network(), 8, monitors=[mon])


def test_compiled_fresh_run_drops_prior_partial_window(backend):
    compiled = backend.CppHybridBackend().compile(_network())
    mons = [_monitor(kind) for kind in KINDS]
    compiled.run(9, monitors=mons, initial_states=[2 * INITIAL])
    for kind, [(times, data)] in zip(KINDS, compiled.run(3, monitors=mons, initial_states=[INITIAL])):
        assert not times.size
        _assert_values(data, _oracle(kind, _trajectory(3), 4))
    for kind, [(times, data)] in zip(KINDS, compiled.run(5, monitors=mons, initial_states=[INITIAL])):
        np.testing.assert_array_equal(times, [2 * DT])
        _assert_values(data, _oracle(kind, _trajectory(5), 4))


def test_compiled_resume_restores_open_window_at_snapshot_clock(backend):
    be = backend.CppHybridBackend()
    net = _network()
    mons = [_monitor(kind) for kind in KINDS]
    before, snapshot = be.compile(net).run(
        5, chunk_size=3, monitors=mons, initial_states=[INITIAL], return_snapshot=True,
    )
    assert snapshot["step"] == 5
    assert [state["kind"] for state in snapshot["kmon_global_states"] if "window_config" in state] == [4, 6, 7]
    after = be.compile(net).resume(snapshot, 6, chunk_size=2, monitors=mons)
    for kind, [(first_t, first_x)], [(last_t, last_x)] in zip(KINDS, before, after):
        np.testing.assert_array_equal(first_t, [2 * DT])
        np.testing.assert_array_equal(last_t, [6 * DT])
        _assert_values(np.concatenate([first_x, last_x]), _oracle(kind, _trajectory(11), 4))


@pytest.mark.parametrize("monitor_kwargs", [{}, {"monitors": []}], ids=["omitted", "empty"])
def test_compiled_resume_continues_neural_state_without_global_monitors(backend, monitor_kwargs):
    be = backend.CppHybridBackend()
    net = _network()
    _, snapshot = be.compile(net).run(
        5, monitors=[_monitor("TemporalAverage")], initial_states=[INITIAL],
        return_snapshot=True,
    )
    assert snapshot["step"] == 5 and "window_config" in snapshot["kmon_global_states"][0]
    _, continued = be.compile(net).resume(snapshot, 3, return_snapshot=True, **monitor_kwargs)
    assert continued["step"] == 8
    _assert_values(continued["states"][0], _trajectory(8)[-1])
    assert not continued["kmon_global_states"]


@pytest.mark.parametrize("missing", ["key", "record", "config"])
def test_compiled_resume_requires_staged_monitor_snapshot(backend, missing):
    compiled = backend.CppHybridBackend().compile(_network())
    mons = [_monitor("TemporalAverage")]
    _, snapshot = compiled.run(3, monitors=mons, return_snapshot=True)
    if missing == "key":
        del snapshot["kmon_global_states"]
        error, match = ValueError, "require global monitor snapshot state"
    else:
        if missing == "record":
            snapshot["kmon_global_states"].clear()
        else:
            del snapshot["kmon_global_states"][0]["window_config"]
        error, match = RuntimeError, "missing compatible snapshot monitor"
    with pytest.raises(error, match=match):
        compiled.resume(snapshot, 1, monitors=mons)


def test_compiled_resume_rejects_changed_weights_and_allows_next_fresh_run(backend):
    compiled = backend.CppHybridBackend().compile(_network())
    mon = _monitor("SpatialAverage")
    _, snapshot = compiled.run(3, monitors=[mon], return_snapshot=True)
    changed = _monitor("SpatialAverage")
    changed.spatial_mean[0, [0, 2]] = [0.25, 0.75]  # Same shape and sparsity.
    with pytest.raises(RuntimeError, match="incompatible snapshot configuration"):
        compiled.resume(snapshot, 1, monitors=[changed])
    [[(times, data)]] = compiled.run(5, monitors=[mon], initial_states=[INITIAL])
    np.testing.assert_array_equal(times, [2 * DT])
    _assert_values(data, _oracle("SpatialAverage", _trajectory(5), 4))


def test_numpy_stochastic_run_collects_global_stream_after_every_native_call(backend, monkeypatch):
    runs, drains = _trace_native_sim(backend, monkeypatch)
    nstep = 14
    outputs = backend.CppHybridBackend().run_network(
        _network(stochastic=True), nstep, chunk_size=3, initial_states=[INITIAL],
        monitors=[_monitor(kind) for kind in KINDS], rng_mode="numpy",
    )
    assert runs == [3, 3, 3, 3, 2] and len(drains) == len(runs)
    draws = np.random.RandomState(NOISE_SEED).randn(nstep, *INITIAL.shape)
    states = _trajectory(nstep) + np.cumsum(draws * np.sqrt(2 * DT * NSIG)[None, :, None, None], axis=0)
    for kind, [(times, data)], calls in zip(KINDS, outputs, zip(*drains)):
        assert [len(call[0]) for call in calls] == [0, 1, 1, 1, 0]
        np.testing.assert_array_equal(times, [2 * DT, 6 * DT, 10 * DT])
        _assert_values(data, _oracle(kind, states, 4))
        np.testing.assert_array_equal(times, np.concatenate([call[0] for call in calls]))
        np.testing.assert_array_equal(data, np.concatenate([call[1] for call in calls]))
