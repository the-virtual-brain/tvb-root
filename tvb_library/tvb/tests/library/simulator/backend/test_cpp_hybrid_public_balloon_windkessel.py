"""Public opt-in BW lowering and consumption of the native regional stream."""

from dataclasses import asdict

import numpy as np
import pytest

from tvb.simulator import integrators, models, monitors, noise
from tvb.simulator.hybrid import NetworkSet, Simulator, Subnetwork
from tvb.simulator.hybrid.recorder import Recorder

from .balloon_windkessel_reference import NOMINAL_PARAMETERS, solve_averaged_drive


DT = 2.5
INITIAL = np.array([[0.25, -0.5, 1.0], [2.0, -1.0, 0.5]])[..., None]
SLOPE = np.array([[0.0, 1 / 256, -1 / 128], [1 / 128, -1 / 256, 1 / 64]])[..., None]
BASELINE = 1.25
SCALE = -0.25


@pytest.fixture
def cpp():
    return pytest.importorskip("tvb.simulator.backend.cpp_hybrid.backend")


def _network(dt=DT, *, stochastic=False, voi_order=("W", "V"), indices=(2, 0, 1)):
    params = dict.fromkeys(("alpha", "beta", "b", "c", "e", "f", "g"), [0.0])
    params.update(d=[1.0], tau=[1.0], gamma=[1.0], I=SLOPE[0, :, 0], a=SLOPE[1, :, 0])
    model = models.Generic2dOscillator(
        variables_of_interest=voi_order, **{k: np.array(v) for k, v in params.items()},
    )
    scheme = (
        integrators.EulerStochastic(dt=dt, noise=noise.Additive(nsig=np.zeros(2), noise_seed=127))
        if stochastic else integrators.EulerDeterministic(dt=dt)
    )
    return NetworkSet(subnets=[Subnetwork(
        name="regions", model=model, scheme=scheme, nnodes=3,
        node_indices=None if indices is None else np.array(indices, dtype=np.int64),
    )]).configure()


def _monitor(period=20.0, **changes):
    return monitors.BalloonWindkessel(**dict(dict(
        period=period, variables_of_interest=np.array([0]),
        drive_baseline=BASELINE, drive_scale=SCALE,
    ), **changes))


def _oracle(nstep, period=20.0):
    # The observed ordering is W,V, not the raw order V,W. The observed W
    # evolves asymmetrically across regions and within each 10 ms window.
    post_step = INITIAL[1, :, 0] + DT * np.arange(1, nstep + 1)[:, None] * SLOPE[1, :, 0]
    return solve_averaged_drive(
        post_step, neural_dt_ms=DT, hemo_dt_ms=5, baseline=BASELINE,
        gain_per_s2_per_observation=SCALE,
    ).windowed_bold(period)


def test_hybrid_consumes_native_bw_stream_with_observed_order_and_timing(cpp, monkeypatch):
    real_sim = cpp.Sim
    specs, drains = [], []

    class TracedSim:
        def __init__(self, width):
            self.sim = real_sim(width)

        def __getattr__(self, name):
            return getattr(self.sim, name)

        def set_global_monitors(self, gather, attachments):
            specs.extend(attachments)
            self.sim.set_global_monitors(gather, attachments)

        def take_global_monitor_outputs(self):
            [(times, data)] = self.sim.take_global_monitor_outputs()
            drains.append((times, data))
            # Demonstrate the public return consumes this binding's actual rows
            # rather than recomputing BOLD or regenerating times on the host.
            return [(times + 0.25, data + 4)]

    monkeypatch.setattr(cpp, "Sim", TracedSim)
    nstep = 243
    sim = Simulator(nets=_network(), backend="cpp", simulation_length=nstep * DT,
                    monitors=[_monitor()])
    sim.configure()
    [(times, data)] = sim.run(initial_conditions=[INITIAL])
    assert specs == [dict(slot=0, kind=10, dt=DT, istep=8, voi=[0],
                          bw_drive_baseline=BASELINE, bw_drive_scale=SCALE,
                          bw_stream_params=asdict(NOMINAL_PARAMETERS))]
    [(native_t, native_x)] = drains
    expected_t, releases, expected_x = _oracle(nstep)
    assert times.shape == native_t.shape == (30,)
    assert data.shape == native_x.shape == (30, 1, 3, 1)
    assert times.dtype == np.float64 and data.dtype == np.float32
    np.testing.assert_array_equal(expected_t, releases - 10.0)
    np.testing.assert_array_equal(native_t, expected_t)
    np.testing.assert_allclose(native_x[:, 0, :, 0], expected_x, atol=1e-10, rtol=1e-7)
    np.testing.assert_array_equal(times, native_t + 0.25)
    np.testing.assert_array_equal(data, native_x + 4)


def test_bw_causal_close_and_resume_with_both_open_windows(cpp):
    be = cpp.CppHybridBackend()
    net, bw = _network(), _monitor()
    # Tick 7 has one completed drive bin, 3/4 of the next, and no BOLD output.
    [[(before_t, before_x)]], snapshot = be.compile(net).run(
        7, chunk_size=3, monitors=[bw], initial_states=[INITIAL], return_snapshot=True,
    )
    assert before_t.shape == (0,)
    assert before_x.shape == (0, 1, 3, 1)
    state, = snapshot["kmon_global_states"]
    assert state["kind"] == 10 and state["bw_drive_count"] == 3 and state["accum_count"] == 2
    [[(closed_t, closed_x)]], snapshot = be.compile(net).resume(
        snapshot, 1, monitors=[bw], return_snapshot=True,
    )
    np.testing.assert_array_equal(closed_t, [10.0])
    assert closed_x.shape == (1, 1, 3, 1)
    [[(tail_t, tail_x)]] = be.compile(net).resume(snapshot, 235, chunk_size=5, monitors=[bw])
    [[(whole_t, whole_x)]] = be.compile(net).run(243, monitors=[bw], initial_states=[INITIAL])
    np.testing.assert_array_equal(np.concatenate((closed_t, tail_t)), whole_t)
    np.testing.assert_array_equal(np.concatenate((closed_x, tail_x)), whole_x)


def test_bw_drains_each_native_call_and_keeps_existing_point_window_and_hrf(cpp, monkeypatch):
    real_sim = cpp.Sim
    calls = []

    class TracedSim:
        def __init__(self, width):
            self.sim = real_sim(width)

        def __getattr__(self, name):
            return getattr(self.sim, name)

        def run(self, *args, **kwargs):
            calls.append("run")
            return self.sim.run(*args, **kwargs)

        def take_global_monitor_outputs(self):
            calls.append("drain")
            return self.sim.take_global_monitor_outputs()

    monkeypatch.setattr(cpp, "Sim", TracedSim)
    be = cpp.CppHybridBackend()
    companions = [monitors.RawVoi(variables_of_interest=np.array([1])),
                  monitors.SubSample(period=3 * DT), monitors.TemporalAverage(period=4 * DT),
                  monitors.Bold(period=40.0)]
    steps = 41
    [[(times, data)], *others] = be.run_network(
        _network(stochastic=True), steps, monitors=[_monitor(), *companions],
        initial_states=[INITIAL],
    )
    assert calls == ["run", "drain"] * steps
    expected_t, _, expected_x = _oracle(steps)
    np.testing.assert_array_equal(times, expected_t)
    np.testing.assert_allclose(data[:, 0, :, 0], expected_x, atol=1e-10, rtol=1e-7)
    standalone = be.run_network(_network(stochastic=True), steps, monitors=companions,
                                initial_states=[INITIAL])
    np.testing.assert_array_equal(others[0][0][0], DT * np.arange(1, steps + 1))
    np.testing.assert_array_equal(others[1][0][0], DT * np.arange(3, steps + 1, 3))
    assert others[-1][0][0].size == 2  # two actual HRF rows, not just equal empty outputs
    for [(t, x)], [(ref_t, ref_x)] in zip(others, standalone):
        np.testing.assert_array_equal(t, ref_t)
        np.testing.assert_array_equal(x, ref_x)


def test_bw_fresh_run_does_not_inherit_partial_windows_or_hemodynamics(cpp):
    compiled = cpp.CppHybridBackend().compile(_network())
    bw = _monitor()
    compiled.run(13, monitors=[bw], initial_states=[2 * INITIAL])
    [[(t, x)]] = compiled.run(9, monitors=[bw], initial_states=[INITIAL])
    expected_t, _, expected_x = _oracle(9)
    np.testing.assert_array_equal(t, expected_t)
    np.testing.assert_allclose(x[:, 0, :, 0], expected_x, atol=1e-10, rtol=1e-7)


@pytest.mark.parametrize("change", ["missing", "omit_monitor", "period", "voi", "source", "calibration", "sibling"])
def test_bw_resume_rejects_incompatible_full_snapshot(cpp, change):
    be = cpp.CppHybridBackend()
    mons = [_monitor(), monitors.TemporalAverage(period=4 * DT)]
    net = _network()
    _, snapshot = be.compile(net).run(7, monitors=mons, return_snapshot=True,
                                      initial_states=[INITIAL])
    error, match = RuntimeError, "snapshot"
    if change == "missing":
        del snapshot["kmon_global_states"]
        error, match = ValueError, "requires global monitor snapshot state"
    elif change == "omit_monitor":
        mons = None
        error, match = ValueError, "requires the compatible global monitors"
    elif change == "period":
        mons[0] = _monitor(period=30.0)
    elif change == "voi":
        mons[0] = _monitor(variables_of_interest=np.array([1]))
    elif change == "source":
        net = _network(voi_order=("V", "W"))
    elif change == "calibration":
        mons[0] = _monitor(drive_scale=0.0)
    else:
        snapshot["kmon_global_states"].pop()
    with pytest.raises(error, match=match):
        be.compile(net).resume(snapshot, 1, monitors=mons)


@pytest.mark.parametrize("period", [0.0, 15.0, 20.0 + 1e-10, np.inf, 10.0 * 2**31])
def test_bw_rejects_invalid_period(cpp, period):
    with pytest.raises(ValueError, match="period must be a positive multiple of 10 ms"):
        cpp.CppHybridBackend().run_network(_network(), 1, monitors=[_monitor(period=period)])


@pytest.mark.parametrize("dt", [3.0, 0.1000001, float(np.float32(10.0 / 2**25))])
def test_bw_rejects_nondivisor_or_ambiguous_clock(cpp, dt):
    with pytest.raises(ValueError, match="source dt must divide 10 ms"):
        cpp.CppHybridBackend().run_network(_network(dt), 1, monitors=[_monitor()])


@pytest.mark.parametrize("dt", [0.1, float(np.float32(0.1)), 10 / 3])
def test_bw_accepts_rounded_clock_and_zero_scale(cpp, dt):
    steps = round(10 / dt)
    [[(t, x)]] = cpp.CppHybridBackend().run_network(
        _network(dt), steps, monitors=[_monitor(period=10.0, drive_scale=0.0)],
        initial_states=[INITIAL],
    )
    np.testing.assert_array_equal(t, [5.0])
    np.testing.assert_array_equal(x, np.zeros((1, 1, 3, 1)))


@pytest.mark.parametrize("voi", [None, [], [0, 1], [[0]], [-1], [2], [2**32]])
def test_bw_rejects_missing_or_invalid_observed_voi(cpp, voi):
    bw = _monitor() if voi is not None else monitors.BalloonWindkessel(
        period=20.0, drive_baseline=BASELINE, drive_scale=SCALE,
    )
    if voi is not None:
        bw.variables_of_interest = np.array(voi, dtype=np.int64)
    with pytest.raises(ValueError, match="exactly one valid observed VOI index"):
        cpp.CppHybridBackend().run_network(_network(), 1, monitors=[bw])


@pytest.mark.parametrize("calibration", [{}, {"drive_baseline": BASELINE},
                                         {"drive_baseline": np.inf, "drive_scale": SCALE},
                                         {"drive_baseline": BASELINE, "drive_scale": -np.inf}])
def test_bw_rejects_missing_or_nonfinite_calibration(cpp, calibration):
    bw = monitors.BalloonWindkessel(period=20.0, variables_of_interest=np.array([0]), **calibration)
    with pytest.raises(ValueError, match="explicit finite drive_baseline and drive_scale"):
        cpp.CppHybridBackend().run_network(_network(), 1, monitors=[bw])


@pytest.mark.parametrize("indices", [(1, 1, 0), (0, 1, 3), (0, 1, 2**32 + 2)])
def test_bw_rejects_invalid_regional_node_indices(cpp, indices):
    with pytest.raises(ValueError, match="full regional node permutation"):
        cpp.CppHybridBackend().run_network(_network(indices=indices), 1, monitors=[_monitor()])


def test_bw_rejects_multiple_subnets_multimode_and_wide_lanes(cpp):
    be, bw = cpp.CppHybridBackend(), _monitor()
    net = _network()
    other = _network().subnets[0]
    other.name = "other_regions"
    two = NetworkSet(subnets=[net.subnets[0], other]).configure()
    with pytest.raises(NotImplementedError, match="one regional subnetwork"):
        be.run_network(two, 1, monitors=[bw])
    net.subnets[0].model = models.ReducedSetFitzHughNagumo()
    net.configure()
    with pytest.raises(NotImplementedError, match="one source clock and mode"):
        be.run_network(net, 1, monitors=[bw])
    with pytest.raises(NotImplementedError, match="width 1"):
        be.compile(_network(), width=4).run(1, monitors=[bw])


def test_bw_rejects_native_monitor_disable_and_local_cpp(cpp):
    be, net, bw = cpp.CppHybridBackend(), _network(), _monitor()
    with pytest.raises(NotImplementedError, match="local or Recorder"):
        be.run_network(net, 1, subnet_monitors={"regions": [bw]})
    be.enable_kernel_monitors = False
    with pytest.raises(NotImplementedError, match="requires native monitor collection"):
        be.run_network(net, 1, monitors=[bw])


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_bw_rejects_non_cpp_hybrid_simulator(backend):
    with pytest.raises(NotImplementedError, match="explicit C\\+\\+ hybrid backend"):
        Simulator(nets=_network(), backend=backend, simulation_length=20, monitors=[_monitor()])


def test_bw_rejects_classic_and_recorder_attach():
    from tvb.simulator.simulator import Simulator as ClassicSimulator

    bw = _monitor()
    assert not isinstance(bw, monitors.Bold)
    with pytest.raises(NotImplementedError, match="global monitor on the explicit C\\+\\+ hybrid backend"):
        bw.config_for_sim(ClassicSimulator())
    with pytest.raises(NotImplementedError, match="local or Recorder"):
        _network().subnets[0].add_monitor(bw)
    with pytest.raises(NotImplementedError, match="local or Recorder"):
        Recorder(monitor=bw)


def test_bw_rejects_direct_numba_and_legacy_numba_sweep():
    from tvb.simulator.backend.nb_hybrid import NbHybridBackend

    be, net, bw = NbHybridBackend(), _network(), _monitor()
    with pytest.raises(NotImplementedError, match="not Numba"):
        be.run_network(net, 1, monitors=[bw])
    with pytest.raises(NotImplementedError, match="not supported in sweeps"):
        be.run_sweep(net, [0.0], nstep=1, monitors=[bw])
    with pytest.raises(NotImplementedError, match="not supported in sweeps"):
        be.sweep(net, params={}, nstep=1, monitor=bw)


def test_bw_rejects_cpp_sweeps(cpp):
    be, bw = cpp.CppHybridBackend(), _monitor()
    with pytest.raises(NotImplementedError, match="not supported in sweeps"):
        be.sweep(_network(), params={}, monitor=bw)
    with pytest.raises(NotImplementedError, match="not supported in sweeps"):
        be.run_sweep(_network(), [0.0], nstep=1, monitors=[bw])
