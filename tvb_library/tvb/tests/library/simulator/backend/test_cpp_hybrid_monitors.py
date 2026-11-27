# -*- coding: utf-8 -*-
"""Hybrid monitor value parity, with native window topology and timing checked separately."""

import numpy as np
import pytest
import scipy.sparse as sp

from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.backend.cpp_hybrid.backend import CppHybridBackend

DT = 0.1
NSTEP = 60


def _network():
    def mk(name, n, offset):
        model = MontbrioPazoRoxin()
        model.configure()
        sn = Subnetwork(name=name, model=model,
                        scheme=HeunDeterministic(dt=DT), nnodes=n,
                        node_indices=np.arange(offset, offset + n))
        sn.configure()
        return sn
    A, B = mk("A", 3, 0), mk("B", 4, 3)
    rng = np.random.RandomState(7)
    W = np.zeros((4, 3), np.float32)
    mask = rng.rand(4, 3) < 0.6
    W[mask] = rng.rand(mask.sum())
    proj = InterProjection(
        source=A, target=B,
        source_cvar=np.array([0], dtype=np.int_),
        target_cvar=np.array([0], dtype=np.int_),
        weights=sp.csr_matrix(W), lengths=sp.csr_matrix(W * 5.0),
        cfun=Linear(), scale=1.0, cv=1.0, dt=DT)
    ns = NetworkSet(subnets=[A, B], projections=[proj])
    ns.configure()
    return ns


def _run_pair(monitors, chunk_size=None):
    nb_out = NbHybridBackend().compile(_network(), eager=True).run(
        NSTEP, chunk_size=chunk_size, monitors=monitors)
    cpp_out = CppHybridBackend().run_network(_network(), NSTEP,
                                             chunk_size=chunk_size,
                                             monitors=monitors)
    return nb_out, cpp_out


def _run_pair_n(monitors, nstep, chunk_size=None):
    """nb_hybrid run at an explicit step count (kernel-engine tests)."""
    return NbHybridBackend().compile(_network(), eager=True).run(
        nstep, chunk_size=chunk_size, monitors=monitors)


def _assert_streams(nb_out, cpp_out, tag):
    assert len(nb_out) == len(cpp_out), tag
    for mi, (per_nb, per_cpp) in enumerate(zip(nb_out, cpp_out)):
        assert len(per_nb) == len(per_cpp), (tag, mi)
        for si, (t_nb, d_nb) in enumerate(per_nb):
            t_cp, d_cp = per_cpp[si]
            np.testing.assert_allclose(
                d_cp, d_nb, rtol=1e-3, atol=1e-4,
                err_msg=f"{tag} monitor#{mi} subnet#{si}")


def _gather_temporal_reference(nb_streams):
    times, data = zip(*nb_streams)
    # The fixture has contiguous source indices. Numba serves separate streams;
    # the native global TemporalAverage now gathers before reducing.
    return [(times[0], np.concatenate(data, axis=2))]


def test_raw():
    from tvb.simulator.monitors import Raw
    nb, cpp = _run_pair([Raw()])
    _assert_streams(nb, cpp, "Raw")


def test_temporal_average():
    from tvb.simulator.monitors import TemporalAverage
    nb, cpp = _run_pair([TemporalAverage(period=0.5)])
    _assert_streams([_gather_temporal_reference(nb[0])], cpp, "TemporalAverage")


def test_subsample():
    from tvb.simulator.monitors import SubSample
    nb, cpp = _run_pair([SubSample(period=1.0)], chunk_size=1)
    _assert_streams(nb, cpp, "SubSample")


def test_global_average():
    from tvb.simulator.monitors import GlobalAverage
    nb, cpp = _run_pair([GlobalAverage(period=0.3)])
    _assert_streams(nb, cpp, "GlobalAverage")


def test_afferent_coupling():
    from tvb.simulator.monitors import AfferentCoupling
    nb, cpp = _run_pair([AfferentCoupling()])
    _assert_streams(nb, cpp, "AfferentCoupling")


def test_afferent_coupling_temporal_average():
    from tvb.simulator.monitors import AfferentCouplingTemporalAverage
    nb, cpp = _run_pair([AfferentCouplingTemporalAverage(period=0.5)])
    _assert_streams(nb, cpp, "AfferentCouplingTemporalAverage")


def test_spatial_average():
    from tvb.simulator.monitors import SpatialAverage
    m = SpatialAverage(period=0.5)
    m.configure()  # default weights (identity / uniform)
    nb, cpp = _run_pair([m])
    _assert_streams(nb, cpp, "SpatialAverage")


def test_raw_voi():
    from tvb.simulator.monitors import RawVoi
    nb, cpp = _run_pair([RawVoi()])
    _assert_streams(nb, cpp, "RawVoi")


def test_bold():
    from tvb.simulator.monitors import Bold
    nb, cpp = _run_pair([Bold(period=1.0)])
    _assert_streams(nb, cpp, "Bold")


def test_projection():
    from tvb.simulator.monitors import Projection
    from tvb.simulator.monitors import Projection as _P
    m = _P(period=0.5)
    ns = _network()
    n_node = sum(sn.nnodes for sn in ns.subnets)
    rng = np.random.RandomState(3)
    m.gain = rng.rand(2, n_node).astype(np.float32)
    nb, cpp = _run_pair([m])
    _assert_streams(nb, cpp, "Projection")


# ---------------------------------------------------------------------------
# Kernel monitor engines (handoff item 8)
#
# CppHybridBackend now runs Bold's HRF convolution and Raw/SubSample
# collection inside the C++ step loop (kernel monitor engines).  The shared
# Python _apply_monitors path remains the parity reference: these tests
# compare the engine output against both (a) the same backend with
# enable_kernel_monitors=False (identical kernel rows, so any difference is
# the moved monitor math) and (b) NbHybridBackend, always at the card's
# rtol 1e-3 / atol 1e-4.
# ---------------------------------------------------------------------------

BOLD_PERIOD = 2000.0      # TR
BOLD_ISTEP = 20000         # steps per TR at DT=0.1
BOLD_INTERIM = 40          # 4 ms / 0.1 ms  (Bold._stock_sample_rate 2**-2)
BOLD_STOCK = 5000          # ceil(2**-2 * hrf_length 20000)


def _cpp_reference():
    """CppHybridBackend with the kernel monitor engines disabled."""
    b = CppHybridBackend()
    b.enable_kernel_monitors = False
    return b


def _run_cpp(backend, pmonitors, nstep, chunk_size=None, repeat=1):
    cn = backend.compile(_network())
    outs = []
    for _ in range(repeat):
        outs.append(cn.run(nstep, chunk_size=chunk_size, monitors=pmonitors))
    return outs


def _assert_close(tag, eng_calls, ref_calls, check_times=True):
    assert len(eng_calls) == len(ref_calls), tag
    for ci, (eng_out, ref_out) in enumerate(zip(eng_calls, ref_calls)):
        assert len(eng_out) == len(ref_out), (tag, ci)
        for mi, (per_eng, per_ref) in enumerate(zip(eng_out, ref_out)):
            assert len(per_eng) == len(per_ref), (tag, ci, mi)
            for si, (t_e, d_e) in enumerate(per_eng):
                t_r, d_r = per_ref[si]
                np.testing.assert_allclose(
                    d_e, d_r, rtol=1e-3, atol=1e-4,
                    err_msg=f"{tag} call#{ci} monitor#{mi} subnet#{si}")
                if check_times:
                    np.testing.assert_allclose(
                        t_e, t_r, rtol=0, atol=0,
                        err_msg=f"{tag} times call#{ci} monitor#{mi} subnet#{si}")


def test_bold_engine_long_full_stock_fill_and_wrap():
    """Engine vs Python reference over a run long enough to fill the HRF
    stock completely and wrap the rolled convolution once (needs
    nstep > BOLD_STOCK * BOLD_INTERIM = 200000)."""
    from tvb.simulator.monitors import Bold
    nstep = 210000
    monitors = [Bold(period=BOLD_PERIOD)]
    eng = _run_cpp(CppHybridBackend(), monitors, nstep)
    ref = _run_cpp(_cpp_reference(), monitors, nstep)
    # 10 samples: steps 20000 .. 200000
    n_out = len(eng[0][0][0][0])
    assert n_out >= 10, f"full-TR samples: {n_out}"
    _assert_close("Bold long fill+wrap", eng, ref)


def test_bold_engine_cross_backend_long():
    """Engine (cpp) vs shared Python reference (nb_hybrid) at 40 s-of-sim
    scale (2 full TRs, stock ~20% filled)."""
    from tvb.simulator.monitors import Bold
    nstep = 40000
    nb = [_run_pair_n([Bold(period=BOLD_PERIOD)], nstep)]
    cpp = _run_cpp(CppHybridBackend(), [Bold(period=BOLD_PERIOD)], nstep)
    _assert_close("Bold engine vs nb long", cpp, nb, check_times=False)


def test_bold_engine_multicall_continuation():
    """Engine state (interim/outer stock, step counter) survives across
    consecutive run() calls exactly like the Python reference runtime."""
    from tvb.simulator.monitors import Bold
    monitors = [Bold(period=BOLD_PERIOD)]
    eng = _run_cpp(CppHybridBackend(), monitors, BOLD_ISTEP, repeat=2)
    ref = _run_cpp(_cpp_reference(), monitors, BOLD_ISTEP, repeat=2)
    # one sample per call (steps 20000 and 40000)
    assert len(eng[0][0][0][0]) == 1
    assert len(eng[1][0][0][0]) == 1
    _assert_close("Bold multicall", eng, ref)


def test_bold_engine_voi_subset():
    """variables_of_interest subset routes the same rows through the engine
    and the reference (covers the engine voi-indexing path)."""
    from tvb.simulator.monitors import Bold
    monitors = [Bold(period=20.0, variables_of_interest=np.array([1],
                                                               dtype=int))]
    eng = _run_cpp(CppHybridBackend(), monitors, 2000)
    ref = _run_cpp(_cpp_reference(), monitors, 2000)
    assert eng[0][0][0][1].shape[1] == 1
    _assert_close("Bold voi subset", eng, ref)


def test_bold_engine_non_volterra_kernel():
    """Non-FirstOrderVolterra HRF (no (dot-1)*k1V0 scaling) matches the
    reference."""
    from tvb.datatypes import equations
    from tvb.simulator.monitors import Bold
    monitors = [Bold(period=BOLD_PERIOD, hrf_kernel=equations.Gamma())]
    eng = _run_cpp(CppHybridBackend(), monitors, 40000)
    ref = _run_cpp(_cpp_reference(), monitors, 40000)
    _assert_close("Bold Gamma kernel", eng, ref)


def test_raw_engine_long():
    """Raw collection in-kernel matches the Python path at reasonable
    length (the engine emits the per-step observed rows)."""
    from tvb.simulator.monitors import Raw
    nstep = 20000
    monitors = [Raw()]
    eng = _run_cpp(CppHybridBackend(), monitors, nstep, chunk_size=1)
    ref = _run_cpp(_cpp_reference(), monitors, nstep, chunk_size=1)
    assert eng[0][0][0][0].shape[0] == nstep
    _assert_close("Raw engine long", eng, ref)


def test_subsample_engine_long():
    """SubSample masking in-kernel matches the Python path."""
    from tvb.simulator.monitors import SubSample
    nstep = 20000
    monitors = [SubSample(period=100.0)]
    eng = _run_cpp(CppHybridBackend(), monitors, nstep, chunk_size=1)
    ref = _run_cpp(_cpp_reference(), monitors, nstep, chunk_size=1)
    # 20 samples: steps 1000 .. 20000 (period 100 ms / dt 0.1 -> istep 1000)
    assert eng[0][0][0][0].shape[0] == nstep // 1000
    _assert_close("SubSample engine long", eng, ref)


def test_kernel_engine_streams_are_used():
    """Prove the engine path (not the Python fallback) serves Bold output:
    _apply_monitors must be handed the kernel streams."""
    import tvb.simulator.backend.cpp_hybrid.backend as cbe
    from tvb.simulator.monitors import Bold
    seen = {}
    orig = cbe._apply_monitors

    def spy(*args, **kwargs):
        seen["kernel_streams"] = kwargs.get("kernel_streams")
        return orig(*args, **kwargs)

    cbe._apply_monitors = spy
    try:
        cpp = CppHybridBackend()
        cpp.run_network(_network(), 20000, monitors=[Bold(period=BOLD_PERIOD)])
    finally:
        cbe._apply_monitors = orig
    ks = seen["kernel_streams"]
    assert ks is not None, "kernel_streams must be provided for Bold"
    per_subnet = ks[0]
    n_samples = sum(len(t) for t, _d in per_subnet)
    assert n_samples >= 1, "engine must emit Bold samples"


def test_chunk_size_gt1_all_kernel_monitors():
    """chunk_size > 1 stays > 1 (no cs=1 fallback, need_tavg=False) for C++ kmon monitors."""
    from tvb.simulator.monitors import (
        AfferentCouplingTemporalAverage,
        Bold,
        TemporalAverage,
    )

    mons = [
        TemporalAverage(period=0.5),
        AfferentCouplingTemporalAverage(period=0.5),
        Bold(period=2.0),
    ]
    nb_out = NbHybridBackend().compile(_network(), eager=True).run(
        NSTEP, chunk_size=5, monitors=mons,
    )

    be = CppHybridBackend()
    seen = {}
    orig_rc = be._run_compiled

    def spy_rc(*args, **kwargs):
        seen["cs"] = kwargs.get("chunk_size")
        seen["need_tavg"] = kwargs.get("need_tavg")
        seen["need_ctavg"] = kwargs.get("need_ctavg")
        return orig_rc(*args, **kwargs)

    be._run_compiled = spy_rc
    cpp_out = be.run_network(_network(), NSTEP, chunk_size=5, monitors=mons)

    assert seen["cs"] == 5
    assert seen["need_tavg"] is False
    assert seen["need_ctavg"] is False
    nb_out = [_gather_temporal_reference(nb_out[0]), *nb_out[1:]]
    _assert_streams(nb_out, cpp_out, "chunk_size=5 all kmon")


def test_voi_subset_selection_all_monitors():
    """variables_of_interest=[1] selects only requested VOIs/cvars across kmon engines."""
    from tvb.simulator.monitors import (
        AfferentCoupling,
        AfferentCouplingTemporalAverage,
        RawVoi,
        SubSample,
        TemporalAverage,
    )

    voi = np.array([1], dtype=int)
    mons_cs1 = [
        RawVoi(variables_of_interest=voi),
        SubSample(period=0.5, variables_of_interest=voi),
        TemporalAverage(period=0.5, variables_of_interest=voi),
        AfferentCoupling(variables_of_interest=voi),
    ]
    pairs = [
        ("cs1", _run_pair(mons_cs1, chunk_size=1)),
        ("ctavg", _run_pair([AfferentCouplingTemporalAverage(period=0.5, variables_of_interest=voi)])),
    ]
    for label, (nb_out, cpp_out) in pairs:
        if label == "cs1":
            nb_out = [*nb_out[:2], _gather_temporal_reference(nb_out[2]), nb_out[3]]
        assert len(nb_out) == len(cpp_out), label
        for mi, (per_nb, per_cpp) in enumerate(zip(nb_out, cpp_out)):
            assert len(per_nb) == len(per_cpp), (label, mi)
            for si, ((_tn, d_nb), (_tc, d_cp)) in enumerate(zip(per_nb, per_cpp)):
                assert d_cp.shape[1] == 1, f"{label} monitor#{mi} did not filter VOIs"
                d_ref = d_nb[:, voi] if d_nb.shape[1] > 1 else d_nb
                np.testing.assert_allclose(
                    d_cp, d_ref, rtol=1e-3, atol=1e-4,
                    err_msg=f"VOI subset {label} monitor#{mi} subnet#{si}",
                )


def test_multisubnet_global_and_spatial_average_monitors_cpp():
    """Eligible native global windows retain values and use interval midpoints."""
    from tvb.simulator.monitors import GlobalAverage, SpatialAverage

    sa = SpatialAverage(period=0.5)
    sa.spatial_mean = np.array([
        [1 / 3, 1 / 3, 1 / 3, 0, 0, 0, 0],
        [0, 0, 0, 0.25, 0.25, 0.25, 0.25],
    ], dtype=np.float64)
    mons = [GlobalAverage(period=0.5), sa]
    nb_out = NbHybridBackend().compile(_network(), eager=True).run(NSTEP, monitors=mons)

    be = CppHybridBackend()
    seen = {}
    orig_rc = be._run_compiled

    def spy_rc(*args, **kwargs):
        seen["need_tavg"] = kwargs.get("need_tavg")
        return orig_rc(*args, **kwargs)

    be._run_compiled = spy_rc
    cpp_out = be.run_network(_network(), NSTEP, monitors=mons)

    assert seen["need_tavg"] is False
    _assert_streams(nb_out, cpp_out, "multi-subnet Global/Spatial")
    for ((times, _),) in cpp_out:
        np.testing.assert_array_equal(times, (np.arange(5, NSTEP + 1, 5) - 2.5) * DT)


def test_bold_and_tavg_mid_period_snapshot_resume_in_cpp():
    """Native resume restores per-source Bold and gathered TemporalAverage independently."""
    from tvb.simulator.monitors import Bold, SpatialAverage, TemporalAverage

    sa = SpatialAverage(period=1.0)
    sa.configure()
    mons = [
        TemporalAverage(period=1.0),  # istep=10, split at 25 -> mid-period!
        Bold(period=2.0),             # istep=20, split at 25 -> mid-period!
        sa,
    ]
    cont = CppHybridBackend().compile(_network()).run(NSTEP, chunk_size=5, monitors=mons)

    sim = CppHybridBackend().compile(_network())
    p1, snap = sim.run(25, chunk_size=5, monitors=mons, return_snapshot=True)
    assert snap.get("kmon_states") is not None
    assert "coclock_window_config" in snap["kmon_global_states"][0]
    assert not snap.get("monitor_states"), "C++ kmon should not populate Python _bold_states"

    p2 = sim.resume(snap, NSTEP - 25, chunk_size=5, monitors=mons)
    assert len(p1) == len(cont) == len(p2)
    for mi in range(len(cont)):
        for si in range(len(cont[mi])):
            t_c, d_c = cont[mi][si]
            t_m = np.concatenate([p1[mi][si][0], p2[mi][si][0]], axis=0)
            d_m = np.concatenate([p1[mi][si][1], p2[mi][si][1]], axis=0)
            np.testing.assert_allclose(t_m, t_c, rtol=0, atol=0, err_msg=f"mon#{mi} sn#{si} t")
            np.testing.assert_allclose(d_m, d_c, rtol=1e-5, atol=1e-6, err_msg=f"mon#{mi} sn#{si} d")


def _unmerged_network():
    def mk(name):
        model = MontbrioPazoRoxin()
        model.configure()
        sn = Subnetwork(name=name, model=model,
                        scheme=HeunDeterministic(dt=DT), nnodes=3)
        sn.configure()
        return sn
    A, B = mk("A"), mk("B")
    ns = NetworkSet(subnets=[A, B], projections=[])
    ns.configure()
    return ns


def test_reused_compiled_sim_reconfigures_modified_legacy_monitors_and_keeps_bold_stock():
    """Modifying istep, voi, gain, or spatial_mean on a reused CompiledCppSim takes effect,
    while consecutive identical Bold runs continue stock state."""
    from tvb.simulator.monitors import Bold, Projection, RawVoi, SpatialAverage, SubSample

    be = CppHybridBackend()

    # 1. istep (SubSample period change on reused sim)
    sim = be.compile(_network())
    sim.run(NSTEP, monitors=[SubSample(period=0.5)])
    got_istep = sim.run(NSTEP, monitors=[SubSample(period=1.0)])
    exp_istep = be.compile(_network()).run(NSTEP, monitors=[SubSample(period=1.0)])
    _assert_streams(exp_istep, got_istep, "reused istep")

    # 2. voi (RawVoi variables_of_interest change on reused sim)
    sim = be.compile(_network())
    sim.run(NSTEP, monitors=[RawVoi(variables_of_interest=np.array([0], dtype=int))])
    got_voi = sim.run(NSTEP, monitors=[RawVoi(variables_of_interest=np.array([1], dtype=int))])
    exp_voi = be.compile(_network()).run(
        NSTEP, monitors=[RawVoi(variables_of_interest=np.array([1], dtype=int))],
    )
    _assert_streams(exp_voi, got_voi, "reused voi")

    # 3. gain (legacy global Projection gain change on reused sim)
    n_node = sum(sn.nnodes for sn in _network().subnets)
    p1 = Projection(period=0.5)
    p1.gain = np.ones((2, n_node), dtype=np.float32)
    p2 = Projection(period=0.5)
    p2.gain = np.arange(1, 2 * n_node + 1, dtype=np.float32).reshape(2, n_node)
    sim = be.compile(_network())
    sim.run(NSTEP, monitors=[p1])
    got_gain = sim.run(NSTEP, monitors=[p2])
    exp_gain = be.compile(_network()).run(NSTEP, monitors=[p2])
    _assert_streams(exp_gain, got_gain, "reused gain")

    # 4. spatial_mean (legacy per-subnet SpatialAverage weight change on reused sim)
    sa1 = SpatialAverage(period=0.5)
    sa1.spatial_mean = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    sa2 = SpatialAverage(period=0.5)
    sa2.spatial_mean = np.array([[0.0, 0.0, 1.0], [0.5, 0.5, 0.0]], dtype=np.float32)
    sim_sm = be.compile(_unmerged_network())
    sim_sm.run(NSTEP, monitors=[sa1])
    got_sm = sim_sm.run(NSTEP, monitors=[sa2])
    exp_sm = be.compile(_unmerged_network()).run(NSTEP, monitors=[sa2])
    _assert_streams(exp_sm, got_sm, "reused spatial_mean")

    # 5. Identical consecutive Bold runs continue stock state
    bold_eng = _run_cpp(be, [Bold(period=BOLD_PERIOD)], BOLD_ISTEP, repeat=2)
    bold_ref = _run_cpp(_cpp_reference(), [Bold(period=BOLD_PERIOD)], BOLD_ISTEP, repeat=2)
    _assert_close("reused identical Bold stock", bold_eng, bold_ref)
