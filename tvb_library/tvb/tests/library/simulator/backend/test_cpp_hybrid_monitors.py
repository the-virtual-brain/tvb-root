# -*- coding: utf-8 -*-
"""Monitor parity tests: each monitor supported by the hybrid path must
produce identical sample streams from CppHybridBackend and NbHybridBackend.
"""

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


def test_raw():
    from tvb.simulator.monitors import Raw
    nb, cpp = _run_pair([Raw()])
    _assert_streams(nb, cpp, "Raw")


def test_temporal_average():
    from tvb.simulator.monitors import TemporalAverage
    nb, cpp = _run_pair([TemporalAverage(period=0.5)])
    _assert_streams(nb, cpp, "TemporalAverage")


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
