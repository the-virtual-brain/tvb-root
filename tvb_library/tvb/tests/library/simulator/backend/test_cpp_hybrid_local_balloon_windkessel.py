"""Public local Balloon-Windkessel Recorders on the C++ hybrid backend.

The contract under test:

* ``Recorder(monitor=BalloonWindkessel(...))`` on a ``Subnetwork`` of a hybrid
  ``Simulator(backend="cpp")`` accumulates the subnetwork's eligible
  post-update observations into complete 10 ms held-drive windows on that
  subnetwork's own physical clock. Each 10 ms window holds as many eligible
  source updates as its clock dictates (two at 5 ms, four at 2.5 ms); only
  eligible updates (``sn.k`` cadence) are observed, the held drive is the
  arithmetic mean of the window's post-step observations,
  ``(mean - drive_baseline) * drive_scale``, and two 5 ms RK4 steps advance that
  region's hemodynamics over the same elapsed 10 ms.
* Observations and hemodynamics are independent per region and per SIMD lane.
  Widths 1, 4, 8 and 16 are supported; width one is not a product limit.
* Output covers complete ``period`` windows only (a positive multiple of 10 ms),
  released at the closing window boundary and stamped with the window midpoint
  on the physical local clock. No incomplete tail is flushed and a fresh run
  starts from hemodynamic rest.
* With matching drive rows and clocks the local Recorder reproduces the existing
  global ``BalloonWindkessel`` monitor exactly. Genuinely different mixed-rate
  sampling semantics are not claimed to agree.
* A compatible compiled snapshot restores the partially accumulated drive window
  and the partially accumulated BOLD output window plus the regional ODE state.
  Incompatible recorder sets are rejected before any state is committed, and a
  rejected all-region/all-lane window close commits nothing. These are native
  recorder-set and window-close atomicity statements, not whole-simulator
  rollback: neural state, delays and RNG may already have advanced.
* Classic Simulator execution, the Numba backend and BW sweeps keep their
  explicit rejection.

``NOMINAL_PARAMETERS`` is the shared nominal hemodynamic set, not a fitted
population estimate. Oracle values come from the independent float64 reference
in ``balloon_windkessel_reference``, which imports neither TVB nor that native
code.
"""

from copy import copy, deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from tvb.simulator import integrators, models, monitors
from tvb.simulator.hybrid import NetworkSet, Simulator, Subnetwork
from tvb.simulator.hybrid.recorder import Recorder

from .balloon_windkessel_reference import NOMINAL_PARAMETERS, solve_averaged_drive


DT = 2.5  # master clock in ms
CADENCE = 2  # a slow subnetwork updates every second master tick
TICKS_PER_WINDOW = int(round(10.0 / DT))  # eligible updates per 10 ms drive window
BASELINE = 1.25
SCALE = -0.25
PERIOD = 20.0
RECORDER_SLOT_BASE = 10000
# The observed rows are W (index 0 of the ("W", "V") ordering) and V. With every
# other Generic2dOscillator term zero the scheme advances V by dt * I and W by
# dt * a, so these slopes make each region's observed W strictly predictable.
INITIAL = np.array([[0.25, -0.5, 1.0], [2.0, -1.0, 0.5]])[..., None]
SLOPE = np.array([[0.0, 1 / 256, -1 / 128], [1 / 128, -1 / 256, 1 / 64]])[..., None]


@pytest.fixture
def cpp():
    return pytest.importorskip("tvb.simulator.backend.cpp_hybrid.backend")


def _subnet(name="regions", *, cadence=1, master_dt=DT, regions=(0, 1, 2),
            voi_order=("W", "V"), node_indices=None, model=None):
    regions = list(regions)
    if model is None:
        params = dict.fromkeys(("alpha", "beta", "b", "c", "e", "f", "g"), [0.0])
        params.update(d=[1.0], tau=[1.0], gamma=[1.0],
                      I=SLOPE[0, regions, 0], a=SLOPE[1, regions, 0])
        model = models.Generic2dOscillator(
            variables_of_interest=voi_order,
            **{key: np.array(value) for key, value in params.items()},
        )
    return Subnetwork(
        name=name, model=model, nnodes=len(regions),
        scheme=integrators.EulerDeterministic(dt=cadence * master_dt),
        node_indices=None if node_indices is None else np.asarray(node_indices, dtype=np.int64),
    )


def _network(*subnets):
    return NetworkSet(subnets=list(subnets) or [_subnet()]).configure()


def _clocks_network():
    """A fast companion fixes the 2.5 ms master clock; the slow subnet keeps k=2.

    A lone 5 ms subnetwork would itself become the 5 ms master and run at k=1, so
    the nontrivial cadence only exists alongside a faster source.
    """
    return _network(_subnet("fast"), _subnet("slow", cadence=CADENCE))


def _two_subnet_ics():
    return [_initial(), _initial()]


def _initial(regions=(0, 1, 2)):
    return INITIAL[:, list(regions)]


def _monitor(period=PERIOD, **changes):
    return monitors.BalloonWindkessel(**dict(dict(
        period=period, variables_of_interest=np.array([0]),
        drive_baseline=BASELINE, drive_scale=SCALE,
    ), **changes))


def _oracle(*, source_dt=DT, n_updates, period=PERIOD, regions=(0, 1, 2)):
    """Complete-window BOLD for the known post-step observed rows of one region set."""
    regions = list(regions)
    post_step = (INITIAL[1, regions, 0]
                 + source_dt * np.arange(1, n_updates + 1)[:, None] * SLOPE[1, regions, 0])
    return solve_averaged_drive(
        post_step, neural_dt_ms=source_dt, hemo_dt_ms=5.0,
        baseline=BASELINE, gain_per_s2_per_observation=SCALE,
    ).windowed_bold(period)


def _assert_stream(stream, expected, *, regions=(0, 1, 2)):
    times, values = stream
    exp_t, _, exp_x = expected
    assert times.dtype == np.float64 and values.dtype == np.float32
    np.testing.assert_array_equal(times, exp_t)
    assert values.shape == (len(exp_t), 1, len(regions), 1)
    np.testing.assert_allclose(values[:, 0, :, 0], exp_x, rtol=1e-7, atol=1e-10)


def _run_local(be, net, nstep, bw, *, name="regions", initial_states=None, **kwargs):
    """Backend-level single-recorder run; returns that subnetwork's stream list."""
    _, streams = be.run_network(
        net, nstep, subnet_monitors={name: [bw]},
        initial_states=initial_states or [_initial()], **kwargs,
    )
    return streams[name]


def _trace_native_runs(cpp, monkeypatch):
    """Record native calls; local BW streams must not use the global drain."""
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
            pytest.fail("local BW streams are delivered in the packed native result")

    monkeypatch.setattr(cpp, "Sim", TracedSim)
    return calls


def _local_spec(period_ticks, *, cadence=1, dt=DT, master_dt=DT, **changes):
    return dict(dict(
        local_window=True, slot=RECORDER_SLOT_BASE, kind=11, istep=period_ticks,
        dt=dt, master_dt=master_dt, cadence_K=cadence, source_name="regions",
        node_indices=[], voi=[0], bw_drive_baseline=BASELINE, bw_drive_scale=SCALE,
        bw_stream_params=asdict(NOMINAL_PARAMETERS),
    ), **changes)


# --------------------------------------------------------------------------
# Parity with the existing global monitor and the independent oracle
# --------------------------------------------------------------------------

def test_local_bw_recorder_matches_global_monitor_under_matching_drives(cpp, monkeypatch):
    be, net, bw = cpp.CppHybridBackend(), _network(), _monitor()
    # 35 master ticks = 87.5 ms: eight complete 10 ms windows, four output
    # periods, and the window plus tick still open at the end.
    nstep = 2 * 4 * TICKS_PER_WINDOW + 3
    # The global reference is taken before tracing: the global route drains
    # through take_global_monitor_outputs, the local route must not. Backend
    # monitor outputs are list[monitor][subnet] of (times, data).
    global_out, = be.run_network(net, nstep, monitors=[bw], initial_states=[_initial()])
    assert isinstance(global_out, list) and len(global_out) == 1
    global_t, global_x = global_out[0]
    calls = _trace_native_runs(cpp, monkeypatch)
    local_out = _run_local(be, net, nstep, bw)
    assert isinstance(local_out, list) and len(local_out) == 1
    local_t, local_x = local_out[0]
    assert local_t.shape == global_t.shape and local_x.shape == global_x.shape
    np.testing.assert_array_equal(local_t, global_t)
    np.testing.assert_array_equal(local_x, global_x)
    _assert_stream((local_t, local_x), _oracle(n_updates=nstep))
    assert calls == [nstep]


def test_local_bw_uses_observed_voi_order_and_calibration(cpp):
    be = cpp.CppHybridBackend()
    nstep = 2 * TICKS_PER_WINDOW
    # Observed index 0 is V here; its post-step row advances by dt * I.
    [stream] = _run_local(be, _network(_subnet(voi_order=("V", "W"))), nstep, _monitor())
    post_step = INITIAL[0, :, 0] + DT * np.arange(1, nstep + 1)[:, None] * SLOPE[0, :, 0]
    _assert_stream(stream, solve_averaged_drive(
        post_step, neural_dt_ms=DT, hemo_dt_ms=5.0, baseline=BASELINE,
        gain_per_s2_per_observation=SCALE,
    ).windowed_bold(PERIOD))

    # Zero scale is a valid explicit calibration: rest state, identically zero BOLD.
    [flat] = _run_local(be, _network(), int(PERIOD / DT), _monitor(drive_scale=0.0))
    np.testing.assert_array_equal(flat[0], [PERIOD / 2])
    np.testing.assert_array_equal(flat[1], np.zeros((1, 1, 3, 1)))
    [signed] = _run_local(be, _network(), int(PERIOD / DT), _monitor(drive_scale=-SCALE))
    [plain] = _run_local(be, _network(), int(PERIOD / DT), _monitor())
    assert not np.array_equal(signed[1], plain[1])


# --------------------------------------------------------------------------
# Source cadence, physical clock and independent per-subnetwork clocks
# --------------------------------------------------------------------------

def test_local_bw_counts_only_eligible_source_updates(cpp):
    be = cpp.CppHybridBackend()
    net = _clocks_network()
    compiled = be.compile(net)
    fast, slow = compiled._analysis.subnets
    assert (fast.k, slow.k) == (1, CADENCE)
    assert compiled._analysis.dt == DT and slow.dt == CADENCE * DT
    bw = _monitor()
    mons, ics = {"slow": [bw]}, _two_subnet_ics()
    # The first master tick arms the monitors; the remaining ticks use the same
    # compiled clock instead of restarting the run.
    [stream] = compiled.run(1, subnet_monitors=mons, initial_states=ics)[1]["slow"]
    streams = [stream]
    for tick in range(1, 18):
        if tick > 1:
            # Fast subnet 0 has no monitors; slow subnet 1 packs (times, data).
            packed = compiled._sim.run(1, 1, None, None, need_tavg=False, need_ctavg=False)
            stream = (np.asarray(packed[4]), np.asarray(packed[5]))
            streams.append(stream)
        [state] = compiled._sim.get_monitor_states()[1]
        updates = tick // CADENCE
        assert state["kind"] == 11 and state["m_step"] == updates
        # 5 ms source clock: two eligible updates per 10 ms drive window.
        assert state["bw_drive_count"] == updates % 2
        if tick % CADENCE:
            # An ineligible master tick neither observes nor advances the drive.
            assert stream[0].size == 0 and stream[1].size == 0
    assert updates == 8
    expected = _oracle(source_dt=CADENCE * DT, n_updates=updates)
    np.testing.assert_array_equal(expected[0], [10.0, 30.0])
    _assert_stream(tuple(np.concatenate(parts) for parts in zip(*streams)), expected)


def test_local_bw_supports_independent_physical_source_clocks(cpp):
    be = cpp.CppHybridBackend()
    net = _clocks_network()
    nstep = 3 * 4 * TICKS_PER_WINDOW  # 120 ms of physical time for both clocks
    _, streams = be.run_network(
        net, nstep, initial_states=_two_subnet_ics(),
        subnet_monitors={"fast": [_monitor()], "slow": [_monitor()]},
    )
    _assert_stream(streams["fast"][0], _oracle(source_dt=DT, n_updates=nstep))
    _assert_stream(streams["slow"][0],
                   _oracle(source_dt=CADENCE * DT, n_updates=nstep // CADENCE))
    # Equal physical runtime and output period give the same release grid and row
    # count; only the sampled drive differs because each drive window holds a
    # different number of eligible updates.
    np.testing.assert_array_equal(streams["fast"][0][0], streams["slow"][0][0])
    assert streams["fast"][0][1].shape == streams["slow"][0][1].shape
    assert not np.array_equal(streams["fast"][0][1], streams["slow"][0][1])
    # This layout has no canonical global node mapping and remains unsupported.
    with pytest.raises(ValueError, match="compatible regional sources"):
        be.run_network(net, nstep, monitors=[_monitor()], initial_states=_two_subnet_ics())


# --------------------------------------------------------------------------
# Complete windows, chunk independence and partial tails
# --------------------------------------------------------------------------

@pytest.mark.parametrize("chunk", [1, 3, 7, 13])
def test_local_bw_chunks_and_partial_tails_are_chunk_independent(cpp, chunk):
    be, net, bw = cpp.CppHybridBackend(), _network(), _monitor()
    nstep = 15 * TICKS_PER_WINDOW + 1  # 15 drive windows: one open output window
    [whole] = _run_local(be, net, nstep, bw)
    [part] = _run_local(be, net, nstep, bw, chunk_size=chunk)
    np.testing.assert_array_equal(part[0], whole[0])
    np.testing.assert_array_equal(part[1], whole[1])
    # Seven complete 20 ms periods; the open window and the partial tick are dropped.
    assert whole[0].size == 7
    _assert_stream(whole, _oracle(n_updates=nstep // TICKS_PER_WINDOW * TICKS_PER_WINDOW))
    [tail] = _run_local(be, net, 3, bw)
    assert tail[0].size == 0 and tail[1].shape == (0, 1, 3, 1)


def test_local_bw_fresh_run_rearms_windows_and_hemodynamics(cpp):
    be = cpp.CppHybridBackend()
    compiled = be.compile(_network())
    bw = _monitor()
    first = compiled.run(13, subnet_monitors={"regions": [bw]},
                         initial_states=[2 * _initial()])[1]["regions"][0]
    # 32.5 ms holds one complete 20 ms period; the rest stays open.
    assert first[0].size == 1
    [warm] = compiled._sim.get_monitor_states()[0]
    assert (warm["m_step"], warm["bw_drive_count"], warm["accum_count"]) == (13, 1, 2)
    # That run had already advanced the hemodynamics off rest.
    assert np.any(warm["bw_local_regions"][:, :, 3:] != 1.0)
    # A second, fresh run must not inherit either window or the ODE state.
    second = compiled.run(9, subnet_monitors={"regions": [bw]},
                          initial_states=[_initial()])[1]["regions"][0]
    _assert_stream(second, _oracle(n_updates=9))
    [state] = compiled._sim.get_monitor_states()[0]
    assert (state["m_step"], state["bw_drive_count"], state["accum_count"]) == (9, 1, 0)


# --------------------------------------------------------------------------
# Independent regions and lanes
# --------------------------------------------------------------------------

def test_local_bw_regions_advance_independently(cpp):
    be, nstep = cpp.CppHybridBackend(), 2 * 4
    [joint] = _run_local(be, _network(), nstep, _monitor())
    for region in range(3):
        [solo] = _run_local(
            be, _network(_subnet(regions=(region,))), nstep, _monitor(),
            initial_states=[_initial((region,))],
        )
        np.testing.assert_array_equal(joint[0], solo[0])
        np.testing.assert_allclose(joint[1][:, 0, region, 0], solo[1][:, 0, 0, 0],
                                   rtol=0, atol=0)
        _assert_stream(solo, _oracle(n_updates=nstep, regions=(region,)), regions=(region,))


def test_local_bw_node_order_does_not_change_local_output(cpp):
    be, nstep = cpp.CppHybridBackend(), 2 * 4
    [plain] = _run_local(be, _network(_subnet(node_indices=None)), nstep, _monitor())
    [permuted] = _run_local(be, _network(_subnet(node_indices=(2, 0, 1))), nstep, _monitor())
    # Local Recorder output stays in subnetwork-local region order.
    np.testing.assert_array_equal(permuted[0], plain[0])
    np.testing.assert_array_equal(permuted[1], plain[1])


@pytest.mark.parametrize("width", [1, 4, 8, 16])
def test_local_bw_supports_simd_widths_without_a_width_one_limit(cpp, width):
    be, net, nstep = cpp.CppHybridBackend(), _network(), 2 * 4
    [stream] = be.compile(net, width=width).run(
        2 * TICKS_PER_WINDOW, subnet_monitors={"regions": [_monitor()]},
        initial_states=[_initial()],
    )[1]["regions"]
    times, values = stream
    expected = _oracle(n_updates=2 * TICKS_PER_WINDOW)
    assert values.shape == (len(times), 1, 3, width)
    np.testing.assert_array_equal(times, expected[0])
    for lane in range(width):
        np.testing.assert_allclose(values[:, 0, :, lane], expected[2], rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("width", [1, 4, 8, 16])
def test_local_bw_lane_permutation_matches_independent_width_one_controls(width):
    """Lanes hold independent regional hemodynamics; permutation only permutes output.

    The public backend has no per-lane drive argument outside a sweep (a later
    phase), so lanes receive distinct explicit increments through the same
    native ``Sim`` entry points the backend uses.
    """
    from tvb.simulator.backend.cpp_hybrid import _cpp_hybrid

    window_ticks = TICKS_PER_WINDOW
    native_steps = 6 * window_ticks  # six 10 ms drive windows, three output periods
    lanes = np.array([0.5, -0.25, 0.125, 0.375, -0.5, 0.25, 0.75, -0.125,
                      0.0625, -0.375, 0.625, 0.1875, -0.1875, 0.3125,
                      -0.4375, 0.875])[:width]
    per_node = np.array([1.0, 2.0, -1.0])
    increments = (lanes[None, None, None, None, :]
                  * per_node[None, None, :, None, None]
                  * np.ones((native_steps, 1, 1, 1, 1))).astype(np.float32)
    assert increments.shape == (native_steps, 1, 3, 1, width)

    def _armed(lane_count):
        sim = _cpp_hybrid.Sim(lane_count)
        sim.set_opts(0, DT, 1)
        # Native Linear (model id 4), zero drift: the noise argument is an
        # explicit increment, so post-step rows are exactly controlled.
        sim.add_subnet(3, 1, 1, 1, 4, 0, 1, k=1, dt=DT)
        sim.set_subnet_params(0, np.zeros((3, 1, lane_count), dtype=np.float32))
        sim.set_voi_specs(0, [0, 0, 0, 1, 0, 0])  # x, x-x; BW observes index 0 = x
        sim.set_subnet_state(0, np.broadcast_to(
            INITIAL[1][None, :, :, None], (1, 3, 1, lane_count)).copy())
        sim.set_monitors(0, [_local_spec(2 * window_ticks)])
        sim.monitor_begin_run()
        return sim

    def _run(sim, noise):
        # The native noise argument is consumed as a flat buffer, so only
        # contiguous float32 increments are passed here.
        assert noise.dtype == np.float32 and noise.flags["C_CONTIGUOUS"]
        packed = sim.run(noise.shape[0], 3, noise, need_tavg=False, need_ctavg=False)
        times, values = np.asarray(packed[2]), np.asarray(packed[3])
        return times, values

    times, values = _run(_armed(width), increments)
    assert values.shape == (3, 1, 3, width)
    for lane in range(width):
        lane_noise = np.ascontiguousarray(increments[..., lane:lane + 1], dtype=np.float32)
        control_t, control_x = _run(_armed(1), lane_noise)
        np.testing.assert_array_equal(times, control_t)
        np.testing.assert_array_equal(values[:, 0, :, lane], control_x[:, 0, :, 0])
    if width > 1:
        assert len({values[:, 0, 0, lane].tobytes() for lane in range(width)}) == width
    permuted_t, permuted_x = _run(
        _armed(width), np.ascontiguousarray(increments[..., ::-1], dtype=np.float32))
    np.testing.assert_array_equal(permuted_t, times)
    np.testing.assert_array_equal(permuted_x, values[..., ::-1])


def test_local_bw_late_lane_drive_close_failure_commits_nothing():
    """A rejected all-lane window close leaves the recorder state untouched.

    Only the monitor is unchanged: the neural state may already have advanced,
    so this is not a whole-simulator rollback claim.
    """
    from tvb.simulator.backend.cpp_hybrid import _cpp_hybrid

    width, window_ticks = 4, TICKS_PER_WINDOW
    increments = np.zeros((window_ticks, 1, 3, 1, width), np.float32)
    increments[..., :width - 1] = 1e-3
    increments[..., width - 1] = 1e9  # drives the last lane out of its domain
    sim = _cpp_hybrid.Sim(width)
    sim.set_opts(0, DT, 1)
    sim.add_subnet(3, 1, 1, 1, 4, 0, 1, k=1, dt=DT)
    sim.set_subnet_params(0, np.zeros((3, 1, width), dtype=np.float32))
    sim.set_voi_specs(0, [0, 0, 0, 1, 0, 0])
    sim.set_subnet_state(0, np.broadcast_to(
        INITIAL[1][None, :, :, None], (1, 3, 1, width)).copy())
    sim.set_monitors(0, [_local_spec(2 * window_ticks)])
    sim.monitor_begin_run()
    # Three eligible updates accumulate normally; the fourth closes the window.
    sim.run(window_ticks - 1, 1, np.ascontiguousarray(increments[:-1]),
            need_tavg=False, need_ctavg=False)
    before = sim.get_monitor_states()
    [opened] = before[0]
    assert opened["m_step"] == window_ticks - 1
    assert opened["bw_drive_count"] == window_ticks - 1
    with pytest.raises(RuntimeError, match="Recorder complete window"):
        sim.run(1, 1, np.ascontiguousarray(increments[-1:]),
                need_tavg=False, need_ctavg=False)
    # The rejected close leaves the master clock and every recorder counter where
    # they were, so the state is still readable; no neural rollback is claimed.
    assert sim.t_abs() == window_ticks - 1
    np.testing.assert_equal(sim.get_monitor_states(), before)
    [state] = sim.get_monitor_states()[0]
    # The closing observation was not folded into any lane, no endpoint was
    # counted and no region or lane committed a partially integrated close.
    assert state["accum_count"] == 0
    np.testing.assert_array_equal(state["bw_local_regions"][:, :, 3:], 1.0)


# --------------------------------------------------------------------------
# Compiled checkpoint / restore of both open windows
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cadence", [1, CADENCE])
@pytest.mark.parametrize("split", [7, 11, 13])
def test_local_bw_restore_continues_both_partial_windows(cpp, cadence, split):
    be, bw = cpp.CppHybridBackend(), _monitor()
    source_ticks = int(round(10.0 / (cadence * DT)))  # eligible updates per 10 ms
    period_ticks = int(round(PERIOD / (cadence * DT)))
    if cadence == 1:
        net, name, ics = _network(), "regions", [_initial()]
    else:
        net, name, ics = _clocks_network(), "slow", _two_subnet_ics()
    # 80 master ticks = 200 ms = ten 20 ms output periods for both source clocks.
    stop = 80
    mons = {name: [bw]}
    compiled = be.compile(net)
    source = compiled._analysis.subnets[-1]
    assert source.k == cadence and source.dt == cadence * compiled._analysis.dt
    _, snap, head = compiled.run(split, subnet_monitors=mons, initial_states=ics,
                                 return_snapshot=True)
    state, = snap["kmon_states"][-1]
    assert state["kind"] == 11 and state["local_window_config"][0] == 2
    assert state["m_step"] == split // cadence
    assert state["bw_drive_count"] == state["m_step"] % source_ticks
    assert state["accum_count"] == (state["m_step"] % period_ticks // source_ticks) * 2
    assert state["bw_local_regions"].shape == (3, 1, 6)
    assert state["bw_local_regions"].dtype == np.float64
    # Stored flow, volume and deoxyhemoglobin states stay in their physical domain.
    assert np.all(state["bw_local_regions"][:, :, 3:5] > 0)
    assert np.all(np.isfinite(state["bw_local_regions"]))
    _, tail = be.compile(net).resume(
        snap, stop - split, chunk_size=3, subnet_monitors=mons,
    )
    _, whole = be.compile(net).run(stop, subnet_monitors=mons, initial_states=ics)
    expected = _oracle(source_dt=cadence * DT, n_updates=stop // cadence)
    _assert_stream(
        tuple(np.concatenate(parts) for parts in zip(head[name][0], tail[name][0])),
        expected,
    )
    released = head[name][0][0].size
    np.testing.assert_array_equal(tail[name][0][0], whole[name][0][0][released:])
    np.testing.assert_array_equal(tail[name][0][1], whole[name][0][1][released:])


def test_local_bw_restore_state_records_both_open_windows(cpp):
    be, bw = cpp.CppHybridBackend(), _monitor()
    sim = be.compile(_network())
    _, snap, _ = sim.run(
        7, subnet_monitors={"regions": [bw]}, initial_states=[_initial()],
        return_snapshot=True,
    )
    state, = snap["kmon_states"][0]
    # One closed drive window already integrated, three observations open on the
    # next one, and a half-open 20 ms output window.
    assert (state["m_step"], state["bw_drive_count"], state["accum_count"]) == (7, 3, 2)
    pending = INITIAL[1, :, 0] + DT * np.arange(5, 8)[:, None] * SLOPE[1, :, 0]
    np.testing.assert_allclose(state["bw_local_regions"][:, 0, 0], pending.sum(axis=0),
                               rtol=0, atol=1e-6)
    # The open output window already holds the two BOLD endpoints of its first
    # 10 ms window, so its running sum is non-zero and finite.
    assert np.all(state["bw_local_regions"][:, 0, 1] != 0.0)
    assert np.all(np.isfinite(state["bw_local_regions"]))
    # The exported state round-trips through the same armed native monitor set.
    before = sim._sim.get_monitor_states()
    sim._sim.set_monitor_states(deepcopy(before))
    np.testing.assert_equal(sim._sim.get_monitor_states(), before)


def _atomic_monitors():
    return {"fast": [_monitor(), monitors.TemporalAverage(period=2 * DT)],
            "slow": [_monitor(period=3 * 10.0)]}


@pytest.mark.parametrize("change", [
    "period", "baseline", "scale", "params", "voi", "source", "cadence", "clock",
    "node-order", "historical", "missing", "duplicate", "phase", "m_step",
    "drive-count", "endpoint-count", "origin", "nonfinite", "extent", "rank",
    "missing-container", "bool-origin", "bool-width", "accum-list",
    "accum-float32", "accum-rank",
])
def test_local_bw_restore_rejects_incompatible_sets_atomically(cpp, change):
    be = cpp.CppHybridBackend()
    net = _clocks_network()
    sim = be.compile(net)
    mons, ics = _atomic_monitors(), _two_subnet_ics()
    sim.run(7, subnet_monitors=mons, initial_states=ics)
    before = sim._sim.get_monitor_states()
    corrupted = deepcopy(before)
    # A valid, different state for earlier monitors must not be committed when a
    # later monitor of the same participating set is rejected.
    corrupted[0][0]["bw_local_regions"][0, 0, 0] += 50.0
    corrupted[0][1]["accum"][0] += 25.0
    state = corrupted[1][0]  # the slow local BW recorder validates last
    config = list(state["local_window_config"])
    if change == "period":
        config[8] += 6
    elif change == "baseline":
        config[20] += 1.0
    elif change == "scale":
        config[21] = -config[21]
    elif change == "params":
        config[22][-1] += 0.01
    elif change == "voi":
        config[9] = [1]
    elif change == "source":
        config[4] = "other"
    elif change == "cadence":
        config[7] += 1
    elif change == "clock":
        config[5] *= 2
    elif change == "node-order":
        config[14] = [0, 1, 2]
    elif change == "historical":
        state.pop("local_window_config")
    elif change == "missing":
        corrupted[1].pop()
    elif change == "duplicate":
        corrupted[1].append(deepcopy(state))
    elif change == "phase":
        state["local_window_phase"] = (state["local_window_phase"] + 1) % CADENCE
    elif change == "m_step":
        state["m_step"] += 1
    elif change == "drive-count":
        state["bw_drive_count"] = (state["bw_drive_count"] + 1) % 2
    elif change == "endpoint-count":
        state["accum_count"] += 1
    elif change == "origin":
        state["run_start"] = 1
    elif change == "nonfinite":
        state["bw_local_regions"][0, 0, 2] = np.nan
    elif change == "extent":
        state["bw_local_regions"] = state["bw_local_regions"][:2]
    elif change == "rank":
        state["bw_local_regions"] = state["bw_local_regions"][:, :, 0]
    elif change == "missing-container":
        del corrupted[1]
    elif change == "bool-origin":
        state["run_start"] = False  # bool is not an exact snapshot integer
    elif change == "bool-width":
        state["width"] = True
    elif change == "accum-list":
        state["accum"] = []  # not a float64 one-dimensional array
    elif change == "accum-float32":
        state["accum"] = np.zeros(0, dtype=np.float32)
    elif change == "accum-rank":
        state["accum"] = np.zeros((1, 0), dtype=np.float64)
    if change != "historical":
        state["local_window_config"] = tuple(config)
    with pytest.raises((ValueError, RuntimeError), match="Recorder complete window"):
        sim._sim.set_monitor_states(corrupted)
    np.testing.assert_equal(sim._sim.get_monitor_states(), before)
    assert sim._sim.t_abs() == 7


@pytest.mark.parametrize("width", [4, 16])
def test_local_bw_restore_rejects_late_lane_corruption_without_committing(cpp, width):
    be, bw = cpp.CppHybridBackend(), _monitor()
    sim = be.compile(_network(), width=width)
    sim.run(7, subnet_monitors={"regions": [bw]}, initial_states=[_initial()])
    before = sim._sim.get_monitor_states()
    corrupted = deepcopy(before)
    # A valid change on the first lane/region, an invalid domain state on the
    # last lane/region of the same recorder.
    corrupted[0][0]["bw_local_regions"][0, 0, 0] += 50.0
    corrupted[0][0]["bw_local_regions"][-1, -1, 4] = -1.0  # volume must be positive
    with pytest.raises(RuntimeError, match="Recorder complete window"):
        sim._sim.set_monitor_states(corrupted)
    np.testing.assert_equal(sim._sim.get_monitor_states(), before)


def test_local_bw_restore_commits_a_reordered_valid_set(cpp):
    be = cpp.CppHybridBackend()
    sim = be.compile(_clocks_network())
    mons, ics = _atomic_monitors(), _two_subnet_ics()
    sim.run(11, subnet_monitors=mons, initial_states=ics)
    expected = sim._sim.get_monitor_states()
    assert [len(per_subnet) for per_subnet in expected] == [2, 1]
    sim._sim.set_monitor_states([list(reversed(expected[0])), list(expected[1])])
    np.testing.assert_equal(sim._sim.get_monitor_states(), expected)


# --------------------------------------------------------------------------
# Invalid configuration and retained guards
# --------------------------------------------------------------------------

@pytest.mark.parametrize("period", [0.0, 15.0, np.inf, 10.0 * 2**31])
def test_local_bw_rejects_invalid_period(cpp, period):
    with pytest.raises(ValueError, match="period must be a positive multiple of 10 ms"):
        _run_local(cpp.CppHybridBackend(), _network(), 4, _monitor(period=period))


@pytest.mark.parametrize("dt", [3.0, 0.1000001, float(np.float32(10.0 / 2**25))])
def test_local_bw_rejects_nondivisor_or_ambiguous_source_clock(cpp, dt):
    with pytest.raises(ValueError, match="source dt must divide 10 ms"):
        _run_local(cpp.CppHybridBackend(), _network(_subnet(master_dt=dt)), 4, _monitor())


@pytest.mark.parametrize("voi", [None, [], [0, 1], [[0]], [-1], [2], [2**32]])
def test_local_bw_rejects_missing_or_invalid_observed_voi(cpp, voi):
    bw = _monitor() if voi is not None else monitors.BalloonWindkessel(
        period=PERIOD, drive_baseline=BASELINE, drive_scale=SCALE,
    )
    if voi is not None:
        bw.variables_of_interest = np.array(voi, dtype=np.int64)
    with pytest.raises(ValueError, match="exactly one valid observed VOI index"):
        _run_local(cpp.CppHybridBackend(), _network(), 4, bw)


@pytest.mark.parametrize("calibration", [{}, {"drive_baseline": BASELINE},
                                         {"drive_baseline": np.inf, "drive_scale": SCALE},
                                         {"drive_baseline": BASELINE, "drive_scale": -np.inf}])
def test_local_bw_rejects_missing_or_nonfinite_calibration(cpp, calibration):
    bw = monitors.BalloonWindkessel(period=PERIOD, variables_of_interest=np.array([0]),
                                    **calibration)
    with pytest.raises(ValueError, match="explicit finite drive_baseline and drive_scale"):
        _run_local(cpp.CppHybridBackend(), _network(), 4, bw)


def test_local_bw_rejects_multimode_source_at_the_spec_seam(cpp):
    """Unit seam: the local Recorder spec refuses a multimode analysis entry.

    A public multimode model cannot stand in here: swapping one into a configured
    subnetwork trips the existing dfun fingerprint guard during analysis, before
    any monitor code runs. The C++ geometry guard is covered separately below.
    """
    be = cpp.CppHybridBackend()
    compiled = be.compile(_network())
    source = copy(compiled._analysis.subnets[0])
    source.n_modes = 2
    with pytest.raises(NotImplementedError, match="one source mode"):
        compiled._recorder_window_spec(source, 0, _monitor())


def test_local_bw_native_geometry_rejects_multimode_subnet():
    from tvb.simulator.backend.cpp_hybrid import _cpp_hybrid

    sim = _cpp_hybrid.Sim(1)
    sim.set_opts(0, DT, 1)
    sim.add_subnet(3, 1, 1, 1, 4, 0, 2, k=1, dt=DT)  # n_modes = 2
    sim.set_subnet_params(0, np.zeros((3, 1, 1), dtype=np.float32))
    sim.set_voi_specs(0, [0, 0, 0, 1, 0, 0])
    sim.set_subnet_state(0, np.zeros((1, 3, 2, 1), dtype=np.float32))
    with pytest.raises(RuntimeError,
                       match="Recorder complete window: invalid attachment geometry"):
        sim.set_monitors(0, [_local_spec(2 * TICKS_PER_WINDOW)])


def test_local_bw_rejects_disabled_native_collection(cpp):
    be = cpp.CppHybridBackend()
    be.enable_kernel_monitors = False
    with pytest.raises(NotImplementedError, match="requires native monitor collection"):
        _run_local(be, _network(), 4, _monitor())


def test_local_bw_rejects_invalid_configuration_before_native_stepping(cpp, monkeypatch):
    calls = _trace_native_runs(cpp, monkeypatch)
    be = cpp.CppHybridBackend()
    net = _network(_subnet(master_dt=3.0))
    with pytest.raises(ValueError, match="source dt must divide 10 ms"):
        _run_local(be, net, 4, _monitor())
    with pytest.raises(ValueError, match="period must be a positive multiple of 10 ms"):
        _run_local(be, _network(), 4, _monitor(period=15.0))
    assert calls == []


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_local_bw_rejects_non_cpp_hybrid_simulator(backend):
    net = _network()
    net.subnets[0].monitors = [Recorder(monitor=_monitor())]
    with pytest.raises(NotImplementedError, match="explicit C\\+\\+ hybrid backend"):
        Simulator(nets=net, backend=backend, simulation_length=PERIOD)


def test_local_bw_recorder_construction_is_supported_but_classic_bw_is_not():
    from tvb.simulator.simulator import Simulator as ClassicSimulator

    rec = Recorder(monitor=_monitor())
    assert isinstance(rec.monitor, monitors.BalloonWindkessel)
    assert not isinstance(rec.monitor, monitors.Bold)
    with pytest.raises(NotImplementedError,
                       match="global monitor on the explicit C\\+\\+ hybrid backend"):
        rec.monitor.config_for_sim(ClassicSimulator())


def test_local_bw_keeps_global_sweep_and_topology_guards(cpp):
    be, bw = cpp.CppHybridBackend(), _monitor()
    two = _network(_subnet(node_indices=(0, 1, 2)),
                   _subnet("other", node_indices=(2, 4, 5)))
    with pytest.raises(ValueError, match="disjoint"):
        be.run_network(two, 1, monitors=[bw])
    with pytest.raises(NotImplementedError, match="not supported in sweeps"):
        be.sweep(_network(), params={}, monitor=bw)
    with pytest.raises(NotImplementedError, match="not supported in sweeps"):
        be.run_sweep(_network(), [0.0], nstep=1, monitors=[bw])
    from tvb.simulator.backend.nb_hybrid import NbHybridBackend

    with pytest.raises(NotImplementedError, match="not Numba"):
        NbHybridBackend().run_network(_network(), 1, monitors=[bw])


# --------------------------------------------------------------------------
# Facade Recorder population and mixed siblings
# --------------------------------------------------------------------------

def test_local_bw_facade_populates_recorder_in_one_native_pass(cpp, monkeypatch):
    calls = _trace_native_runs(cpp, monkeypatch)
    net, bw = _network(), _monitor()
    net.subnets[0].monitors = [Recorder(monitor=bw)]
    sim = Simulator(nets=net, backend="cpp", simulation_length=13 * DT)
    assert sim.run(initial_conditions=[_initial()]) == []
    assert calls == [13]
    times, samples = net.subnets[0].monitors[0].to_arrays()
    assert times.dtype == np.float32 and samples.dtype == np.float32
    expected = _oracle(n_updates=13)
    np.testing.assert_array_equal(times, expected[0].astype(np.float32))
    assert samples.shape == (len(times), 1, 3, 1)
    np.testing.assert_allclose(samples[:, 0, :, 0], expected[2], rtol=1e-7, atol=1e-10)


def test_local_bw_coexists_with_local_and_global_siblings(cpp):
    be, net, bw = cpp.CppHybridBackend(), _network(_subnet(node_indices=(2, 0, 1))), _monitor()
    tavg = monitors.TemporalAverage(period=2 * DT)
    raw = monitors.Raw()
    global_tavg = monitors.TemporalAverage(period=4 * DT)
    hrf_bold = monitors.Bold(period=8 * DT)
    nstep = 33

    global_only = be.run_network(net, nstep, monitors=[global_tavg, hrf_bold],
                                 initial_states=[_initial()])
    globals_mixed, mixed = be.run_network(
        net, nstep, initial_states=[_initial()], monitors=[global_tavg, hrf_bold],
        subnet_monitors={"regions": [bw, tavg, raw]},
    )
    # Global outputs are list[monitor][subnet] of (times, data).
    assert len(globals_mixed) == len(global_only) == 2
    for mixed_per_monitor, expected_per_monitor in zip(globals_mixed, global_only):
        assert len(mixed_per_monitor) == len(expected_per_monitor) == 1
        mixed_t, mixed_x = mixed_per_monitor[0]
        expected_t, expected_x = expected_per_monitor[0]
        assert mixed_t.shape == expected_t.shape and mixed_x.shape == expected_x.shape
        np.testing.assert_array_equal(mixed_t, expected_t)
        np.testing.assert_array_equal(mixed_x, expected_x)

    assert isinstance(mixed["regions"], list) and len(mixed["regions"]) == 3
    standalone = [_run_local(be, net, nstep, mon)[0] for mon in (bw, tavg, raw)]
    for (stream_t, stream_x), (expected_t, expected_x) in zip(mixed["regions"], standalone):
        assert stream_t.shape == expected_t.shape and stream_x.shape == expected_x.shape
        np.testing.assert_array_equal(stream_t, expected_t)
        np.testing.assert_array_equal(stream_x, expected_x)
