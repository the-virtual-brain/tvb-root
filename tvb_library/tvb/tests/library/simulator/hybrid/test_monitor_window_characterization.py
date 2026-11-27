"""Historical public monitor behavior, not a parity or future-policy contract.

Two independent state variables have constant, different derivatives and
three distinct node offsets. The analytical post-step state, not any other
backend or a Raw output, is the oracle for every value assertion.

Superseded single-subnet C++ bug characterizations are kept as historical
cases (baseline 2958e4f7, Linux Slurm 5213862). Current regional C++ policy
acceptance is in test_cpp_hybrid_public_windows.py and
test_cpp_hybrid_public_projection.py (replacing the former C++ EEG sample-tick
midpoint, rounded period, host fallback and dropped-noise expectations).
"""

import numpy as np
import pytest

from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.projections import ProjectionSurfaceEEG
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.sensors import SensorsEEG
from tvb.simulator import integrators, models, monitors, noise
from tvb.simulator.hybrid import NetworkSet, Simulator as HybridSimulator, Subnetwork
from tvb.simulator.simulator import Simulator as ClassicSimulator


DT = 0.125
INITIAL = np.array([[0.25, -0.5, 1.0], [2.0, -1.0, 0.5]])[..., None]
SLOPE = np.array([1.0, -0.5])[:, None, None]
SPATIAL = np.array([[0.75, 0.0, 0.25], [0.0, 1.0, 0.0]])
GAIN = np.array([[1.0, -0.5, 2.0], [-2.0, 0.25, 0.5], [0.5, 0.0, -1.0]])
ROUTES = ("classic", "python", "numba", "cpp-native", "cpp-host")
REGIONAL_WINDOWS = ("TemporalAverage", "GlobalAverage", "SpatialAverage")
HISTORICAL_CPP_REGIONAL = "superseded regional C++ behavior; baseline 2958e4f7 (Slurm 5213862)"


def _model(constant=False, harmonic=None):
    params = dict.fromkeys(("alpha", "beta", "b", "c", "e", "f", "g"), 0.0)
    params.update(d=1.0, tau=1.0, gamma=1.0,
                  I=0.0 if constant else 1.0, a=0.0 if constant else -0.5)
    if harmonic is not None:
        params.update(I=0.0, a=0.0, alpha=-harmonic, b=harmonic)
    return models.Generic2dOscillator(variables_of_interest=("V", "W"),
                                      **{k: np.array([v]) for k, v in params.items()})


def _trajectory(steps, constant=False):
    return INITIAL + np.asarray(steps)[:, None, None, None] * DT * (0 if constant else SLOPE)


def _classic(mons, constant, harmonic):
    conn = Connectivity(weights=np.zeros((3, 3)), tract_lengths=np.zeros((3, 3)),
                        centres=np.eye(3), orientations=np.eye(3),
                        cortical=np.ones(3, dtype=bool),
                        region_labels=np.array(["a", "b", "c"]))
    return ClassicSimulator(connectivity=conn, model=_model(constant, harmonic),
                            integrator=integrators.HeunDeterministic(dt=DT),
                            initial_conditions=INITIAL[None, ...].copy(),
                            monitors=mons).configure()


def _network(constant=False, merged=True, harmonic=None):
    return NetworkSet(subnets=[Subnetwork(
        name="window_input", model=_model(constant, harmonic), scheme=integrators.HeunDeterministic(dt=DT),
        nnodes=3, node_indices=np.arange(3) if merged else None)])


def _set_cpp_monitor_route(route, monkeypatch):
    if not route.startswith("cpp-"):
        return
    cpp = pytest.importorskip("tvb.simulator.backend.cpp_hybrid")
    cls = cpp.CppHybridBackend
    if route == "cpp-host":
        original_init = cls.__init__

        def host_init(self):
            original_init(self)
            self.enable_kernel_monitors = False

        monkeypatch.setattr(cls, "__init__", host_init)

    # Verify routing at spec lowering, not by assuming numerical agreement.
    original_spec = cls._monitor_attach_spec
    lowered = []

    def record_spec(self, slot, monitor, *args, **kwargs):
        lowered.append(type(monitor).__name__)
        return original_spec(self, slot, monitor, *args, **kwargs)

    monkeypatch.setattr(cls, "_monitor_attach_spec", record_spec)
    return lowered


def _run(route, mons, nstep, monkeypatch, *, constant=False, merged=True, harmonic=None):
    if route == "classic" or (not route.startswith("cpp-")
                              and any(isinstance(m, monitors.EEG) for m in mons)):
        # Hybrid has no connectome/sensor config_for_sim hook; give EEG a
        # properly configured regional lead field from the classic initializer.
        classic = _classic(mons, constant, harmonic)
    for mon in mons:
        if isinstance(mon, monitors.EEG) and route.startswith("cpp-"):
            mon.gain = GAIN.copy()  # already regional; no classic lifecycle or reference cache
        if isinstance(mon, monitors.SpatialAverage):
            # Classic's labels configure uniform group weights; supply the same
            # deliberately asymmetric weighted operator to all routes.
            mon.spatial_mean = SPATIAL.copy()
    if route == "classic":
        assert classic.current_step == 0
        return classic.run(simulation_length=nstep * DT)

    lowered = _set_cpp_monitor_route(route, monkeypatch)
    backend = "cpp" if route.startswith("cpp-") else route
    sim = HybridSimulator(nets=_network(constant, merged, harmonic), backend=backend,
                          monitors=mons, simulation_length=nstep * DT)
    sim.configure()
    for mon in mons:
        if isinstance(mon, monitors.EEG) and not route.startswith("cpp-"):
            # Hybrid's constructor overwrites the configured index array with
            # slice(None); EEG.sample uses .size. Restore the configured array
            # so the actual record/sample path is reachable.
            mon.voi = (np.arange(2) if mon.variables_of_interest is None
                       else mon.variables_of_interest.copy())
    result = sim.run(initial_conditions=[INITIAL.copy()])
    if lowered is not None:
        assert lowered == ([type(m).__name__ for m in mons] if route == "cpp-native" else [])
    return result


def _eeg(period, obsnoise=None, **kwargs):
    sensors = SensorsEEG(labels=np.array(["x", "y", "z"]), locations=np.eye(3))
    return monitors.EEG(period=period, sensors=sensors, reference="x",
                        region_mapping=RegionMapping(array_data=np.arange(3)),
                        projection=ProjectionSurfaceEEG(projection_data=GAIN.copy()),
                        obsnoise=obsnoise, **kwargs)


def _monitor(kind, period, **kwargs):
    if kind == "EEG":
        return _eeg(period, **kwargs)
    if kind == "SpatialAverage":
        kwargs["spatial_mask"] = np.array([0, 1, 0])
    return getattr(monitors, kind)(period=period, **kwargs)


def _reduce(kind, state, reference=True):
    if kind == "GlobalAverage":
        return state.mean(axis=2, keepdims=True)
    if kind in ("SpatialAverage", "EEG"):
        matrix = SPATIAL if kind == "SpatialAverage" else GAIN
        reduced = np.einsum("an,tvnm->tvam", matrix, state)
        if kind == "EEG" and reference:
            reduced -= reduced[:, :, :1, :].copy()
        return reduced
    return state


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("kind", ("Raw", "RawVoi", "SubSample", "TemporalAverage",
                                 "GlobalAverage", "SpatialAverage", "EEG"))
@pytest.mark.parametrize("window", (1, 3, 4))
def test_public_monitor_value_and_time(route, kind, window, monkeypatch):
    if route == "cpp-host" and kind in REGIONAL_WINDOWS:
        pytest.skip(HISTORICAL_CPP_REGIONAL)
    monitor = _monitor(kind, window * DT)
    if route == "cpp-host" and kind == "EEG":
        with pytest.raises(NotImplementedError, match="native monitor collection"):
            _run(route, [monitor], 2 * window, monkeypatch)
        return
    [(times, data)] = _run(route, [monitor], 2 * window, monkeypatch)
    ends = np.arange(window, 2 * window + 1, window)
    if kind in ("Raw", "RawVoi"):
        ends = np.arange(1, 2 * window + 1)
    sample_steps = ends.astype(float)
    represented_steps = sample_steps.copy()
    if kind == "TemporalAverage" or (kind == "EEG" and route != "numba"):
        sample_steps -= (window - 1) / 2
        represented_steps -= window / 2
    elif kind in ("GlobalAverage", "SpatialAverage", "EEG") and route not in ("classic", "python"):
        sample_steps -= (window - 1) / 2
        represented_steps -= (window / 2 if route == "cpp-native" and kind in REGIONAL_WINDOWS
                              else (window - 1) / 2)

    np.testing.assert_array_equal(times, represented_steps * DT)
    expected = _reduce(kind, _trajectory(sample_steps), reference=route not in ("numba", "cpp-host"))
    assert data.shape == expected.shape
    np.testing.assert_allclose(data, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("route", ROUTES)
def test_public_spatial_and_eeg_preserve_constant_values(route, monkeypatch):
    if route == "cpp-host":
        with pytest.raises(NotImplementedError, match="native monitor collection"):
            _run(route, [_eeg(4 * DT)], 8, monkeypatch, constant=True)
        return
    kinds = ("SpatialAverage", "EEG")
    mons = [_monitor(kind, 4 * DT) for kind in kinds]
    outputs = _run(route, mons, 8, monkeypatch, constant=True)
    assert len(outputs) == len(kinds)
    for kind, (times, data) in zip(kinds, outputs):
        if route in ("classic", "python"):
            expected_times = np.array([4, 8] if kind == "SpatialAverage" else [2, 6])
        elif route == "cpp-native":
            expected_times = np.array([2, 6])
        else:
            expected_times = np.array([2.5, 6.5])
        np.testing.assert_array_equal(times, expected_times * DT)
        expected = _reduce(kind, _trajectory([0, 0], constant=True),
                           reference=route not in ("numba", "cpp-host"))
        assert data.shape == expected.shape
        np.testing.assert_allclose(data, expected, atol=1e-6, rtol=0)


class _FixedObservationNoise(noise.Additive):
    calls = 0

    def generate(self, shape, **kwargs):
        self.calls += 1
        return np.broadcast_to(np.array([1.0, 4.0, -2.0])[:, None], shape)


@pytest.mark.parametrize("route", ROUTES)
def test_public_eeg_observation_noise_is_at_emit_before_reference_when_supported(route, monkeypatch):
    obsnoise = _FixedObservationNoise()
    if route.startswith("cpp-"):
        reason = "obsnoise=None" if route == "cpp-native" else "native monitor collection"
        with pytest.raises(NotImplementedError, match=reason):
            _run(route, [_eeg(4 * DT, obsnoise=obsnoise)], 8, monkeypatch)
        assert obsnoise.calls == 0
        return
    [(times, data)] = _run(route, [_eeg(4 * DT, obsnoise=obsnoise)], 8, monkeypatch)
    reference = route not in ("numba", "cpp-host")
    expected = _reduce("EEG", _trajectory([2.5, 6.5]), reference=reference)
    if route in ("classic", "python"):
        assert obsnoise.calls == 2
        expected += np.array([0.0, 3.0, -3.0])[None, None, :, None]
        np.testing.assert_array_equal(times, np.array([2, 6]) * DT)
    else:
        assert obsnoise.calls == 0
        np.testing.assert_array_equal(times, np.array([2.5, 6.5]) * DT)
    assert data.shape == expected.shape
    np.testing.assert_allclose(data, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("kind", ("GlobalAverage", "SpatialAverage", "EEG"))
@pytest.mark.parametrize("merged", (True, False), ids=("merged", "unmerged"))
def test_public_raw_companion_and_trailing_step(route, kind, merged, monkeypatch):
    if route == "cpp-host" and kind in REGIONAL_WINDOWS:
        pytest.skip(HISTORICAL_CPP_REGIONAL)
    # Raw forces the compiled execution chunk to 1. Some legacy host paths
    # consume that chunk grid directly instead of the requested monitor grid.
    mons = [monitors.Raw(), _monitor(kind, 4 * DT)]
    if route == "cpp-host" and kind == "EEG":
        with pytest.raises(NotImplementedError, match="native monitor collection"):
            _run(route, mons, 9, monkeypatch, merged=merged)
        return
    if route != "classic" and not merged:
        with pytest.warns(UserWarning, match="do not have node_indices"):
            raw, (times, data) = _run(route, mons, 9, monkeypatch, merged=merged)
    else:
        raw, (times, data) = _run(route, mons, 9, monkeypatch, merged=merged)
    np.testing.assert_array_equal(raw[0], np.arange(1, 10) * DT)
    np.testing.assert_allclose(raw[1], _trajectory(np.arange(1, 10)), atol=1e-6, rtol=0)

    if route in ("classic", "python") and kind != "EEG":
        value_steps = time_steps = [4, 8]
    elif route in ("classic", "python"):
        value_steps, time_steps = [2.5, 6.5], [2, 6]
    elif route == "cpp-native":
        value_steps = [2.5, 6.5]
        time_steps = [2, 6]
    elif kind == "GlobalAverage" or merged:
        value_steps = time_steps = np.arange(1, 10)
    else:
        value_steps, time_steps = [2.5, 6.5], [2, 6]
    np.testing.assert_array_equal(times, np.asarray(time_steps) * DT)
    expected = _reduce(kind, _trajectory(value_steps), reference=route not in ("numba", "cpp-host"))
    assert data.shape == expected.shape
    np.testing.assert_allclose(data, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("route", ROUTES)
def test_public_noncommensurate_projection_period(route, monkeypatch):
    if route == "cpp-native":
        with pytest.raises(ValueError, match="commensurate"):
            _run(route, [_eeg(3.5 * DT)], 12, monkeypatch)
        return
    if route == "cpp-host":
        with pytest.raises(NotImplementedError, match="native monitor collection"):
            _run(route, [_eeg(3.5 * DT)], 12, monkeypatch)
        return
    [(times, data)] = _run(route, [_eeg(3.5 * DT)], 12, monkeypatch)
    if route in ("classic", "python"):
        time_steps = np.array([1.5, 4.5, 7.5, 10.5])
        value_steps = time_steps + 0.5
    else:
        value_steps = time_steps = np.array([2.5, 6.5, 10.5])
    np.testing.assert_array_equal(times, time_steps * DT)
    expected = _reduce("EEG", _trajectory(value_steps), reference=route not in ("numba", "cpp-host"))
    assert data.shape == expected.shape
    np.testing.assert_allclose(data, expected, atol=1e-6, rtol=0)


def _cpp_backend(use_kernel):
    cpp = pytest.importorskip("tvb.simulator.backend.cpp_hybrid")
    backend = cpp.CppHybridBackend()
    backend.enable_kernel_monitors = use_kernel
    return backend


@pytest.mark.parametrize("use_kernel", (True, False), ids=("native", "host"))
@pytest.mark.parametrize("chunk, centres", ((None, (2.5, 6.5)), (1, tuple(range(1, 10))),
                                            (2, (1.5, 3.5, 5.5, 7.5)), (8, (4.5,))))
@pytest.mark.skip(reason=HISTORICAL_CPP_REGIONAL)
def test_cpp_backend_global_average_explicit_chunk_can_override_monitor_period(use_kernel, chunk, centres):
    network = _network()
    network.configure()
    [[(times, data)]] = _cpp_backend(use_kernel).run_network(
        network, nstep=9, chunk_size=chunk, monitors=[monitors.GlobalAverage(period=4 * DT)],
        initial_states=[INITIAL.copy()])
    if not use_kernel and chunk != 1:
        centres += (9,)
    np.testing.assert_array_equal(times, np.asarray(centres) * DT)
    expected = _reduce("GlobalAverage", _trajectory(centres))
    assert data.shape == expected.shape
    np.testing.assert_allclose(data, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("use_kernel", (True, False), ids=("native", "host"))
@pytest.mark.parametrize("kind", ("TemporalAverage", "GlobalAverage"))
def test_cpp_backend_resume_with_an_open_period(use_kernel, kind):
    if not use_kernel:
        pytest.skip(HISTORICAL_CPP_REGIONAL)
    backend = _cpp_backend(use_kernel)
    network = _network()
    network.configure()
    compiled = backend.compile(network)
    mon = _monitor(kind, 4 * DT)
    [[(full_t, full_x)]] = compiled.run(9, monitors=[mon], initial_states=[INITIAL.copy()])
    [[(first_t, first_x)]], snapshot = backend.compile(network).run(
        3, monitors=[mon], initial_states=[INITIAL.copy()], return_snapshot=True)
    assert snapshot["step"] == 3
    # Use a fresh kernel so in-memory accumulators cannot mask missing state.
    [[(last_t, last_x)]] = backend.compile(network).resume(snapshot, 6, monitors=[mon])
    split_t = np.concatenate((first_t, last_t))
    split_x = np.concatenate((first_x, last_x))
    if not use_kernel and kind == "GlobalAverage":
        # No host period accumulator: the first call emits an incomplete chunk,
        # and the second call starts a new window on the next absolute step.
        np.testing.assert_array_equal(full_t, np.array([2.5, 6.5, 9]) * DT)
        np.testing.assert_array_equal(split_t, np.array([2, 5.5, 8.5]) * DT)
        np.testing.assert_allclose(full_x, _reduce(kind, _trajectory([2.5, 6.5, 9])), atol=1e-6, rtol=0)
        np.testing.assert_allclose(split_x, _reduce(kind, _trajectory([2, 5.5, 8.5])), atol=1e-6, rtol=0)
    else:
        assert first_t.size == 0
        time_steps = [2, 6]
        np.testing.assert_array_equal(full_t, np.asarray(time_steps) * DT)
        np.testing.assert_array_equal(split_t, full_t)
        np.testing.assert_allclose(full_x, _reduce(kind, _trajectory([2.5, 6.5])), atol=1e-6, rtol=0)
        np.testing.assert_array_equal(split_x, full_x)


@pytest.mark.parametrize("route", ROUTES)
def test_public_global_average_selected_voi(route, monkeypatch):
    if route == "cpp-host":
        pytest.skip(HISTORICAL_CPP_REGIONAL)
    mon = monitors.GlobalAverage(period=4 * DT, variables_of_interest=np.array([1]))
    [(times, data)] = _run(route, [mon], 8, monkeypatch)
    sample_steps = [4, 8] if route in ("classic", "python") else [2.5, 6.5]
    time_steps = [2, 6] if route == "cpp-native" else sample_steps
    np.testing.assert_array_equal(times, np.asarray(time_steps) * DT)
    expected = _reduce("GlobalAverage", _trajectory(sample_steps))
    if route in ("classic", "cpp-native"):
        expected = expected[:, [1]]
    assert data.shape == expected.shape
    np.testing.assert_allclose(data, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("theta, alias", ((np.pi / 12, np.pi / 3), (5 * np.pi / 12, -np.pi / 3)),
                         ids=("below-output-nyquist", "above-output-nyquist"))
def test_public_temporal_average_boxcar_gain_phase_and_alias(route, theta, alias, monkeypatch):
    if route == "cpp-host":
        pytest.skip(HISTORICAL_CPP_REGIONAL)
    # Use both quadratures of a linear oscillator, V + iW. Heun integrates it
    # with multiplier z; choose h so arg(z) is the requested discrete frequency.
    # Reference the boxcar to the endpoint and divide out Heun's amplitude drift
    # to isolate the observation operator from the integrator.
    tangent = np.tan(theta)
    h = (np.sqrt(1 + 2 * tangent**2) - 1) / tangent
    z = 1 + 1j * h - 0.5 * h**2
    window = 4
    [(times, data)] = _run(route, [monitors.TemporalAverage(period=window * DT)],
                           12, monkeypatch, harmonic=h / DT)
    np.testing.assert_array_equal(times, DT * np.array([2, 6, 10]))
    assert data.shape == (3, 2, 3, 1)
    complex_out = data[:, 0, :, 0] + 1j * data[:, 1, :, 0]
    start = INITIAL[0, :, 0] + 1j * INITIAL[1, :, 0]
    end_state = z**np.array([4, 8, 12])[:, None] * start
    response = (1 - z**(-window)) / (window * (1 - 1 / z))
    measured = complex_out / end_state
    np.testing.assert_allclose(np.abs(measured), abs(response), atol=2e-6, rtol=0)
    np.testing.assert_allclose(np.angle(measured), np.angle(response), atol=2e-6, rtol=0)
    np.testing.assert_allclose(np.angle(complex_out[1:] / complex_out[:-1]), alias, atol=2e-6, rtol=0)


@pytest.mark.parametrize("route", ("cpp-native", "cpp-host"))
def test_public_cpp_global_and_recorder_clocks_differ_for_slow_subnet(route, monkeypatch):
    lowered = _set_cpp_monitor_route(route, monkeypatch)
    subnets = []
    for index, cadence in enumerate((1, 2)):
        subnet = Subnetwork(name=f"subnet_{index}", model=_model(),
                            scheme=integrators.HeunDeterministic(dt=cadence * DT),
                            nnodes=3, node_indices=np.arange(3) + 3 * index)
        subnet.add_monitor(monitors.GlobalAverage(period=4 * DT))
        subnets.append(subnet)
    sim = HybridSimulator(nets=NetworkSet(subnets=subnets), backend="cpp", simulation_length=9 * DT,
                          monitors=[monitors.TemporalAverage(period=4 * DT)])
    sim.configure()
    if route == "cpp-host":
        with pytest.raises(NotImplementedError, match="Recorder complete windows require native collection"):
            sim.run(initial_conditions=[INITIAL.copy(), INITIAL.copy()])
        assert lowered == []
        return

    [(times, data)] = sim.run(initial_conditions=[INITIAL.copy(), INITIAL.copy()])
    assert sorted(lowered) == ["GlobalAverage", "GlobalAverage", "TemporalAverage"]
    np.testing.assert_array_equal(times, DT * np.array([2, 6]))
    # The global monitor sees held slow states on master ticks 1, 2, 3, 4:
    # integrated elapsed steps 0, 2, 2, 4. A local Recorder sees only 2, 4.
    expected = np.concatenate((_trajectory([2.5, 6.5]), _trajectory([2, 6])), axis=2)
    assert data.shape == expected.shape
    np.testing.assert_allclose(data, expected, atol=1e-6, rtol=0)
    for index, subnet in enumerate(subnets):
        local_times, local_data = subnet.monitors[0].to_arrays()
        np.testing.assert_array_equal(local_times, np.array([2, 6]) * DT)
        local_mean_steps = [2.5, 6.5] if index == 0 else [3, 7]
        expected = _reduce("GlobalAverage", _trajectory(local_mean_steps))
        assert local_data.shape == expected.shape
        np.testing.assert_allclose(local_data, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("use_kernel", (True, False), ids=("native", "host"))
@pytest.mark.skip(reason=HISTORICAL_CPP_REGIONAL)
def test_cpp_compiled_fresh_run_reuses_temporal_average_accumulator(use_kernel):
    network = _network()
    network.configure()
    compiled = _cpp_backend(use_kernel).compile(network)
    mon = monitors.TemporalAverage(period=4 * DT)
    compiled.run(9, monitors=[mon], initial_states=[INITIAL.copy()])
    [[(times, data)]] = compiled.run(3, monitors=[mon], initial_states=[INITIAL.copy()])
    # The integration restarts but the monitor does not: it completes the
    # old window using old step 9 plus new steps 1..3 and emits on the old clock.
    np.testing.assert_array_equal(times, np.array([10 * DT]))
    np.testing.assert_allclose(data, _trajectory([3.75]), atol=1e-6, rtol=0)


@pytest.mark.skip(reason=HISTORICAL_CPP_REGIONAL)
def test_cpp_native_resume_accepts_changed_spatial_weights_with_same_shape():
    network = _network()
    network.configure()
    backend = _cpp_backend(True)
    previous = monitors.SpatialAverage(period=4 * DT)
    previous.spatial_mean = SPATIAL.copy()
    [[(first_t, _)]], snapshot = backend.compile(network).run(
        3, monitors=[previous], initial_states=[INITIAL.copy()], return_snapshot=True)
    assert first_t.size == 0
    changed = monitors.SpatialAverage(period=4 * DT)
    changed.spatial_mean = SPATIAL[:, ::-1].copy()
    [[(times, data)]] = backend.compile(network).resume(snapshot, 1, monitors=[changed])
    np.testing.assert_array_equal(times, np.array([2.5 * DT]))
    # This is a mixed-operator result, not a valid window under either matrix.
    expected = (3 * _reduce("SpatialAverage", _trajectory([2])) +
                np.einsum("an,tvnm->tvam", changed.spatial_mean, _trajectory([4]))) / 4
    np.testing.assert_allclose(data, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("kind", ("Raw", "RawVoi", "SubSample"))
def test_cpp_backend_rejects_explicit_coarse_chunk_for_point_samples(kind):
    network = _network()
    network.configure()
    with pytest.raises(ValueError, match="requires chunk_size=1"):
        _cpp_backend(True).run_network(network, nstep=8, chunk_size=2,
                                       monitors=[_monitor(kind, 4 * DT)],
                                       initial_states=[INITIAL.copy()])
