"""Historical public monitor behavior, not a parity or future-policy contract.

Two independent state variables have constant, different derivatives and
three distinct node offsets. The analytical post-step state, not any other
backend or a Raw output, is the oracle for every value assertion.
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


def _model(constant=False):
    params = dict.fromkeys(("alpha", "beta", "b", "c", "e", "f", "g"), 0.0)
    params.update(d=1.0, tau=1.0, gamma=1.0,
                  I=0.0 if constant else 1.0, a=0.0 if constant else -0.5)
    return models.Generic2dOscillator(variables_of_interest=("V", "W"),
                                      **{k: np.array([v]) for k, v in params.items()})


def _trajectory(steps, constant=False):
    return INITIAL + np.asarray(steps)[:, None, None, None] * DT * (0 if constant else SLOPE)


def _classic(mons, constant):
    conn = Connectivity(weights=np.zeros((3, 3)), tract_lengths=np.zeros((3, 3)),
                        centres=np.eye(3), orientations=np.eye(3),
                        cortical=np.ones(3, dtype=bool),
                        region_labels=np.array(["a", "b", "c"]))
    return ClassicSimulator(connectivity=conn, model=_model(constant),
                            integrator=integrators.HeunDeterministic(dt=DT),
                            initial_conditions=INITIAL[None, ...].copy(),
                            monitors=mons).configure()


def _network(constant=False, merged=True):
    return NetworkSet(subnets=[Subnetwork(
        name="ramp", model=_model(constant), scheme=integrators.HeunDeterministic(dt=DT),
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


def _run(route, mons, nstep, monkeypatch, *, constant=False, merged=True):
    if route == "classic" or any(isinstance(m, monitors.EEG) for m in mons):
        # Hybrid has no connectome/sensor config_for_sim hook; give EEG a
        # properly configured regional lead field from the classic initializer.
        classic = _classic(mons, constant)
    for mon in mons:
        if isinstance(mon, monitors.SpatialAverage):
            # Classic's labels configure uniform group weights; supply the same
            # deliberately asymmetric weighted operator to all routes.
            mon.spatial_mean = SPATIAL.copy()
    if route == "classic":
        assert classic.current_step == 0
        return classic.run(simulation_length=nstep * DT)

    lowered = _set_cpp_monitor_route(route, monkeypatch)
    backend = "cpp" if route.startswith("cpp-") else route
    sim = HybridSimulator(nets=_network(constant, merged), backend=backend,
                          monitors=mons, simulation_length=nstep * DT)
    sim.configure()
    for mon in mons:
        if isinstance(mon, monitors.EEG):
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
    monitor = _monitor(kind, window * DT)
    [(times, data)] = _run(route, [monitor], 2 * window, monkeypatch)
    ends = np.arange(window, 2 * window + 1, window)
    if kind in ("Raw", "RawVoi"):
        ends = np.arange(1, 2 * window + 1)
    sample_steps = ends.astype(float)
    represented_steps = sample_steps.copy()
    if kind == "TemporalAverage" or (kind == "EEG" and route in ("classic", "python")):
        sample_steps -= (window - 1) / 2
        represented_steps -= window / 2
    elif kind in ("GlobalAverage", "SpatialAverage", "EEG") and route not in ("classic", "python"):
        sample_steps -= (window - 1) / 2
        represented_steps -= (window - 1) / 2

    np.testing.assert_array_equal(times, represented_steps * DT)
    expected = _reduce(kind, _trajectory(sample_steps), reference=route not in ("numba", "cpp-host"))
    assert data.shape == expected.shape
    np.testing.assert_allclose(data, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("route", ROUTES)
def test_public_spatial_and_eeg_preserve_constant_values(route, monkeypatch):
    mons = [_monitor(kind, 4 * DT) for kind in ("SpatialAverage", "EEG")]
    outputs = _run(route, mons, 8, monkeypatch, constant=True)
    for kind, (times, data) in zip(("SpatialAverage", "EEG"), outputs):
        if route in ("classic", "python"):
            expected_times = np.array([4, 8] if kind == "SpatialAverage" else [2, 6])
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
