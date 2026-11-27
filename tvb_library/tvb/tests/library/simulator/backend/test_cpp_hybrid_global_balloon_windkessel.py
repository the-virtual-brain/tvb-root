"""Global BW gathers canonical regional observations on the master clock."""

from copy import deepcopy

import numpy as np
import pytest

from .balloon_windkessel_reference import solve_averaged_drive
from .test_cpp_hybrid_local_balloon_windkessel import (
    BASELINE, DT, INITIAL, SCALE, SLOPE, _initial, _monitor, _network, _subnet,
)


@pytest.fixture
def cpp():
    return pytest.importorskip("tvb.simulator.backend.cpp_hybrid.backend")


def _sources(cadence=3):
    return _network(_subnet("fast", regions=(2, 0), node_indices=(2, 0)),
                    _subnet("slow", cadence=cadence, regions=(1,), node_indices=(1,)))


def _initials():
    return [_initial((2, 0)), _initial((1,))]


def _reference(ticks, cadence, *, pre_update=False, always_update=False):
    t = np.arange(1, ticks + 1) - int(pre_update)
    slow = t if always_update else (t // cadence) * cadence
    updates = np.stack((t, slow, t), axis=1)
    rows = INITIAL[1, :, 0] + DT * updates * SLOPE[1, :, 0]
    return solve_averaged_drive(rows, neural_dt_ms=DT, hemo_dt_ms=5.0,
                                baseline=BASELINE, gain_per_s2_per_observation=SCALE).windowed_bold(20.0)


@pytest.mark.parametrize("cadence", [1, 3])
def test_global_gather_master_clock_reference_and_continuation(cpp, cadence):
    compiled = cpp.CppHybridBackend().compile(_sources(cadence))
    assert compiled._analysis.dt == DT
    assert [s.k for s in compiled._analysis.subnets] == [1, cadence]
    assert [s.dt for s in compiled._analysis.subnets] == [DT, cadence * DT]
    ticks = 203  # 507.5 ms, including a 7.5 ms incomplete output tail
    whole = compiled.run(ticks, chunk_size=3, monitors=[_monitor()], initial_states=_initials())
    t, x = whole[0][0]
    ref_t, _, ref_x = _reference(ticks, cadence)
    np.testing.assert_array_equal(t, ref_t)
    assert x.shape == (25, 1, 3, 1) and x.dtype == np.float32 and t.dtype == np.float64
    np.testing.assert_allclose(x[:, 0, :, 0], ref_x, rtol=1e-7, atol=1e-10)
    controls = [np.zeros_like(ref_x), _reference(ticks, cadence, pre_update=True)[2]]
    if cadence == 3:
        controls.append(_reference(ticks, cadence, always_update=True)[2])
    for wrong in controls:
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(wrong, ref_x, rtol=1e-7, atol=1e-10)
    first, snap = compiled.run(7, monitors=[_monitor()], initial_states=_initials(), return_snapshot=True)
    state = snap["kmon_global_states"][0]
    assert state["bw_drive_count"] == 3 and state["accum_count"] == 2
    rest = cpp.CppHybridBackend().compile(_sources(cadence)).resume(snap, ticks - 7, monitors=[_monitor()])
    np.testing.assert_array_equal(np.concatenate((first[0][0][0], rest[0][0][0])), t)
    np.testing.assert_array_equal(np.concatenate((first[0][0][1], rest[0][0][1])), x)
    fresh = compiled.run(ticks, monitors=[_monitor()], initial_states=_initials())
    np.testing.assert_array_equal(fresh[0][0][1], x)


def test_global_gather_restore_rejects_changed_mapping(cpp):
    be = cpp.CppHybridBackend()
    _, snap = be.compile(_sources()).run(7, monitors=[_monitor()], initial_states=_initials(), return_snapshot=True)
    changed = _network(_subnet("fast", regions=(2, 0), node_indices=(0, 2)),
                       _subnet("slow", cadence=3, regions=(1,), node_indices=(1,)))
    with pytest.raises((ValueError, RuntimeError), match="compatible|mismatch"):
        be.compile(changed).resume(deepcopy(snap), 1, monitors=[_monitor()])


def test_global_gather_rejects_overlapping_mapping(cpp):
    net = _network(_subnet("fast", regions=(2, 0), node_indices=(2, 0)),
                   _subnet("slow", cadence=3, regions=(1,), node_indices=(0,)))
    with pytest.raises((ValueError, RuntimeError), match="disjoint|indices|gather"):
        cpp.CppHybridBackend().compile(net).run(1, monitors=[_monitor()], initial_states=_initials())


@pytest.mark.parametrize("width", [1, 4, 8, 16])
def test_native_global_bw_independent_lanes_and_restore(width):
    from .test_cpp_hybrid_balloon_windkessel_stream import GATHER, _sim, _spec

    def armed(w):
        sim = _sim(width=w)
        sim.set_global_monitors([GATHER], [_spec(istep=8), _spec(slot=1, istep=8)])
        sim.monitor_begin_run()
        return sim

    wide = armed(width)
    # Each lane has a different deterministic drive; flat-buffer native noise
    # inputs are explicitly contiguous, including each width-one control.
    delta = np.zeros((24, 1, 3, 1, width), dtype=np.float32)
    delta[:, 0, :, 0, :] = np.arange(1, width + 1, dtype=np.float32) / 1024
    assert delta.flags.c_contiguous
    wide.run(7, 7, np.ascontiguousarray(delta[:7]), need_tavg=False, need_ctavg=False)
    before = wide.get_global_monitor_states()
    assert before[0]["bw_drive_count"] == 3 and before[0]["accum_count"] == 2
    altered = deepcopy(before)
    altered[0]["bw_stream_regions"][0, 0] += 0.125
    altered[1]["bw_stream_regions"][-1, 3] = -1.0
    with pytest.raises(RuntimeError, match="invalid snapshot"):
        wide.set_global_monitor_states(altered)
    np.testing.assert_equal(wide.get_global_monitor_states(), before)
    wide.set_global_monitor_states(deepcopy(before))
    wide.run(17, 17, np.ascontiguousarray(delta[7:]), need_tavg=False, need_ctavg=False)
    result = wide.take_global_monitor_outputs()[0]
    for lane in range(width):
        single = armed(1)
        lane_delta = np.ascontiguousarray(delta[..., lane:lane + 1])
        assert lane_delta.flags.c_contiguous and lane_delta.dtype == np.float32
        single.run(24, 24, lane_delta, need_tavg=False, need_ctavg=False)
        expected = single.take_global_monitor_outputs()[0]
        np.testing.assert_array_equal(result[0], expected[0])
        np.testing.assert_array_equal(result[1][..., lane], expected[1][..., 0])


def test_global_source_permutation_and_local_matching_drives(cpp):
    be = cpp.CppHybridBackend()
    net = _sources(cadence=1)
    out, local = be.run_network(net, 24, monitors=[_monitor()], initial_states=_initials(),
                                subnet_monitors={"fast": [_monitor()], "slow": [_monitor()]})
    t, global_data = out[0][0]
    for name, indices in (("fast", (2, 0)), ("slow", (1,))):
        lt, ld = local[name][0]
        np.testing.assert_array_equal(t, lt)
        np.testing.assert_array_equal(global_data[:, :, indices, :], ld)
    reverse = _network(_subnet("slow", regions=(1,), node_indices=(1,)),
                       _subnet("fast", regions=(2, 0), node_indices=(2, 0)))
    permuted = be.run_network(reverse, 24, monitors=[_monitor()], initial_states=_initials()[::-1])
    np.testing.assert_array_equal(permuted[0][0][0], t)
    np.testing.assert_array_equal(permuted[0][0][1], global_data)


@pytest.mark.parametrize("width", [1, 4, 8, 16])
def test_global_public_broadcast_lanes(cpp, width):
    be = cpp.CppHybridBackend()
    result = be.compile(_sources(), width=width).run(16, monitors=[_monitor()], initial_states=_initials())
    baseline = be.compile(_sources()).run(16, monitors=[_monitor()], initial_states=_initials())
    for lane in range(width):
        np.testing.assert_array_equal(result[0][0][1][..., lane], baseline[0][0][1][..., 0])


def _native_mixed(width, order=(0, 1), cadence=3):
    from tvb.simulator.backend.cpp_hybrid import _cpp_hybrid
    from .test_cpp_hybrid_balloon_windkessel_stream import _spec

    sim = _cpp_hybrid.Sim(width)
    sim.set_opts(0, DT, 1)
    declarations = []
    for i, source in enumerate(order):
        nodes, k, indices = ((2, 1, [2, 0]), (1, cadence, [1]))[source]
        sim.add_subnet(nodes, 1, 1, 1, 4, 0, 1, k=k, dt=k * DT)
        sim.set_subnet_params(i, np.zeros((nodes, 1, width), np.float32))
        sim.set_voi_specs(i, [0, 0, 0])
        sim.set_subnet_state(i, np.zeros((1, nodes, 1, width), np.float32))
        declarations.append(dict(n_voi=1, n_node=nodes, node_indices=indices,
                                 source_name=("fast", "slow")[source], source_dt=k * DT,
                                 observed_terms=["x"]))
    spec = _spec(istep=8, voi=[0])
    sim.set_global_monitors(declarations, [spec, dict(spec, slot=1)])
    sim.monitor_begin_run()
    return sim


def _mixed_noise(width, ticks=24):
    lane = np.arange(1, width + 1, dtype=np.float32)
    fast = np.empty((ticks, 1, 2, 1, width), np.float32)
    slow = np.empty((ticks, 1, 1, 1, width), np.float32)
    fast[:, 0, 0, 0, :] = lane / 1024
    fast[:, 0, 1, 0, :] = -lane / 2048
    slow[:, 0, 0, 0, :] = (lane + 1) / 4096
    return [fast, slow]


@pytest.mark.parametrize("width", [1, 4, 8, 16])
def test_native_mixed_gather_distinct_source_region_lanes(width):
    noise = _mixed_noise(width)
    assert all(a.flags.c_contiguous and a.dtype == np.float32 for a in noise)
    wide = _native_mixed(width)
    wide.run(24, 24, noise, need_tavg=False, need_ctavg=False)
    actual = wide.take_global_monitor_outputs()[0]
    for lane in range(width):
        reference = _native_mixed(1)
        lane_inputs = [np.ascontiguousarray(a[..., lane:lane + 1]) for a in noise]
        reference.run(24, 24, lane_inputs, need_tavg=False, need_ctavg=False)
        expected = reference.take_global_monitor_outputs()[0]
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1][..., lane], expected[1][..., 0])
    reverse = _native_mixed(width, order=(1, 0))
    reverse.run(24, 24, noise[::-1], need_tavg=False, need_ctavg=False)
    np.testing.assert_equal(reverse.take_global_monitor_outputs()[0], actual)
    perm = np.arange(width)[::-1]
    permuted = _native_mixed(width)
    permuted.run(24, 24, [np.ascontiguousarray(a[..., perm]) for a in noise],
                 need_tavg=False, need_ctavg=False)
    permuted_output = permuted.take_global_monitor_outputs()[0]
    np.testing.assert_array_equal(permuted_output[1], actual[1][..., perm])


def test_native_global_wide_last_lane_close_is_atomic():
    sim, prefix = _native_mixed(4), _native_mixed(4)
    noise = _mixed_noise(4, ticks=12)
    # Fast local node 0 maps to final canonical region 2. A large negative
    # finite drive forces the last lane's first RK4 intermediate out of domain.
    noise[0][8:, 0, 0, 0, -1] = -1e9
    prefix.run(11, 11, [np.ascontiguousarray(a[:11]) for a in noise],
               need_tavg=False, need_ctavg=False)
    with pytest.raises(RuntimeError, match="invalid calibrated drive or RK4 domain failure"):
        sim.run(12, 12, noise, need_tavg=False, need_ctavg=False)
    assert sim.t_abs() == prefix.t_abs() == 11
    np.testing.assert_equal(sim.get_global_monitor_states(), prefix.get_global_monitor_states())
    actual, expected = sim.take_global_monitor_outputs(), prefix.take_global_monitor_outputs()
    assert actual[0][0].size == 1
    np.testing.assert_equal(actual, expected)


def test_native_global_mixed_identity_and_all_set_rejection():
    sim = _native_mixed(4)
    sim.run(7, 7, _mixed_noise(4, 7), need_tavg=False, need_ctavg=False)
    before = sim.get_global_monitor_states()
    for mutate in ("mapping", "cadence", "observations", "sibling"):
        candidate = deepcopy(before)
        candidate[0]["bw_stream_regions"][0, 0] += 0.125
        if mutate == "sibling":
            candidate[1]["bw_stream_regions"][-1, 3] = -1
        else:
            identity = list(candidate[1]["bw_stream_config"])
            if mutate == "mapping":
                sources = list(identity[9])
                row = list(sources[0]); row[4] = [0, 2]; sources[0] = tuple(row)
                identity[9] = sources
            else:
                declarations = list(identity[10]); row = list(declarations[1])
                if mutate == "cadence": row[3] = 2
                else: row[2] = ["other"]
                declarations[1] = tuple(row); identity[10] = declarations
            candidate[1]["bw_stream_config"] = tuple(identity)
        with pytest.raises(RuntimeError, match="incompatible|invalid"):
            sim.set_global_monitor_states(candidate)
        np.testing.assert_equal(sim.get_global_monitor_states(), before)


def test_mixed_global_local_and_average_continue_together(cpp):
    from tvb.simulator import monitors

    be = cpp.CppHybridBackend()
    global_monitors = [_monitor(), monitors.TemporalAverage(period=20.0)]
    local = {"fast": [_monitor()]}
    compiled = be.compile(_sources())
    whole, local_whole = compiled.run(35, monitors=global_monitors,
                                      subnet_monitors=local, initial_states=_initials())
    head, snapshot, local_head = compiled.run(7, monitors=global_monitors,
                                              subnet_monitors=local, initial_states=_initials(),
                                              return_snapshot=True)
    tail, local_tail = be.compile(_sources()).resume(snapshot, 28, monitors=global_monitors,
                                                     subnet_monitors=local)
    for i in range(2):
        for a, b, expected in zip(head[i][0], tail[i][0], whole[i][0]):
            np.testing.assert_array_equal(np.concatenate((a, b)), expected)
    for a, b, expected in zip(local_head["fast"][0], local_tail["fast"][0], local_whole["fast"][0]):
        np.testing.assert_array_equal(np.concatenate((a, b)), expected)


def test_rejected_global_restore_keeps_pending_outputs():
    sim = _native_mixed(4)
    control = _native_mixed(4)
    inputs = _mixed_noise(4, 15)
    for s in (sim, control):
        s.run(15, 15, inputs, need_tavg=False, need_ctavg=False)
    state = sim.get_global_monitor_states()
    invalid = deepcopy(state)
    invalid[0]["bw_stream_regions"][0, 0] += 0.125
    invalid[1]["bw_stream_regions"][-1, 3] = -1
    with pytest.raises(RuntimeError, match="invalid snapshot"):
        sim.set_global_monitor_states(invalid)
    np.testing.assert_equal(sim.get_global_monitor_states(), state)
    output, expected = sim.take_global_monitor_outputs(), control.take_global_monitor_outputs()
    assert output[0][0].size
    np.testing.assert_equal(output, expected)


@pytest.mark.parametrize("dt,ticks,split", [
    (0.1, 450, 313),
    (float(np.float32(0.1)), 450, 313),
    (10.0 / 3.0, 17, 10),
], ids=["decimal-0.1", "binary32-0.1", "third-10"])
@pytest.mark.parametrize("cadence", [1, 3], ids=["co-clock", "mixed-rate"])
def test_global_gather_binary32_rounded_dt_snapshots_and_resume(cpp, dt, ticks, split, cadence):
    def make_net():
        return _network(_subnet("fast", master_dt=dt, regions=(2, 0), node_indices=(2, 0)),
                        _subnet("slow", cadence=cadence, master_dt=dt, regions=(1,), node_indices=(1,)))

    be = cpp.CppHybridBackend()
    compiled = be.compile(make_net())
    whole = compiled.run(ticks, chunk_size=3, monitors=[_monitor()], initial_states=_initials())
    t, x = whole[0][0]
    assert t.size == 2 and x.shape == (2, 1, 3, 1)

    head, snap = compiled.run(split, monitors=[_monitor()], initial_states=_initials(), return_snapshot=True)
    tail = be.compile(make_net()).resume(snap, ticks - split, monitors=[_monitor()])
    np.testing.assert_array_equal(np.concatenate((head[0][0][0], tail[0][0][0])), t)
    np.testing.assert_array_equal(np.concatenate((head[0][0][1], tail[0][0][1])), x)

    # Verify against float64 hemodynamic oracle driven by the float32 neural trajectory.
    neural_dt = 10.0 / round(10.0 / dt)
    state = INITIAL[1, :, 0].astype(np.float32).copy()
    step_fast = np.float32(dt) * SLOPE[1, :, 0].astype(np.float32)
    step_slow = np.float32(cadence * dt) * SLOPE[1, :, 0].astype(np.float32)
    rows = np.empty((ticks, 3), dtype=np.float64)
    for k in range(ticks):
        state[[0, 2]] += step_fast[[0, 2]]
        if (k + 1) % cadence == 0:
            state[1] += step_slow[1]
        rows[k] = state
    ref_t, _, ref_x = solve_averaged_drive(
        rows, neural_dt_ms=neural_dt, hemo_dt_ms=5.0,
        baseline=BASELINE, gain_per_s2_per_observation=SCALE,
    ).windowed_bold(20.0)
    np.testing.assert_array_equal(t, ref_t)
    np.testing.assert_allclose(x[:, 0, :, 0], ref_x, rtol=1e-7, atol=1e-10)
