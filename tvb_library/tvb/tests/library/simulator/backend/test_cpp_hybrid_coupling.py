# -*- coding: utf-8 -*-
"""Coupling-function parity tests: CppHybridBackend vs NbHybridBackend.

Covers every coupling function/variant nb_hybrid supports, including the
SigmoidalJansenRit legacy form and the PreSigmoidal dynamic form (static and
globalT dynamic threshold).
"""

import numpy as np
import pytest
import scipy.sparse as sp

from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.integrators import HeunDeterministic, EulerDeterministic
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.hybrid.coupling import (
    Linear, Scaling, Sigmoidal, SigmoidalJansenRit, Kuramoto as KuramotoCfun,
    Difference, HyperbolicTangent, PreSigmoidal,
)
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.backend.cpp_hybrid.backend import CppHybridBackend

DT = 0.01
NSTEP = 40


def _subnetwork(
    name,
    n_nodes,
    integrator_cls=HeunDeterministic,
    cr=1.0,
    cv=0.0,
):
    model = MontbrioPazoRoxin(
        cr=np.array([cr], dtype=np.float64),
        cv=np.array([cv], dtype=np.float64),
    )
    model.configure()
    scheme = integrator_cls(dt=DT)
    sn = Subnetwork(name=name, model=model, scheme=scheme, nnodes=n_nodes)
    sn.configure()
    return sn


def _projection(
    src,
    tgt,
    cfun,
    source_cvar=(0,),
    target_cvar=(0,),
    target_scales=None,
    seed=42,
    scale=1.5,
):
    n_src, n_tgt = src.nnodes, tgt.nnodes
    rng = np.random.RandomState(seed)
    W = np.zeros((n_tgt, n_src), np.float32)
    for i in range(n_tgt):
        for j in range(n_src):
            if i != j and rng.rand() < 0.5:
                W[i, j] = rng.rand()
    kwargs = dict(
        source=src, target=tgt,
        source_cvar=np.array(source_cvar, dtype=np.int_),
        target_cvar=np.array(target_cvar, dtype=np.int_),
        weights=sp.csr_matrix(W),
        lengths=sp.csr_matrix(W * 10.0),
        cfun=cfun, scale=scale, cv=1.0, dt=DT,
    )
    if target_scales is not None:
        kwargs["target_scales"] = np.asarray(target_scales, dtype=np.float64)
    return InterProjection(**kwargs)


def _initial_states_pair():
    ic_a = np.array(
        [[[0.15], [0.25], [0.35]], [[-1.1], [-0.7], [-0.4]]],
        dtype=np.float64,
    )
    ic_b = np.array(
        [[[0.10], [0.20], [0.30], [0.40]], [[-1.2], [-0.9], [-0.6], [-0.3]]],
        dtype=np.float64,
    )
    return [ic_a, ic_b]


def _run_pair(cfun, source_cvar=(0,)):
    """Build identical NetworkSets and run both backends; return outputs."""
    A = _subnetwork("A", 3)
    B = _subnetwork("B", 4)
    ns = NetworkSet(subnets=[A, B],
                    projections=[_projection(A, B, cfun, source_cvar)])
    ns.configure()
    nb_out = NbHybridBackend().compile(ns, eager=True).run(NSTEP)

    A2 = _subnetwork("A", 3)
    B2 = _subnetwork("B", 4)
    ns2 = NetworkSet(subnets=[A2, B2],
                     projections=[_projection(A2, B2, cfun, source_cvar)])
    ns2.configure()
    cpp_out = CppHybridBackend().run_network(ns2, NSTEP)
    return nb_out, cpp_out


def _assert_close(nb_out, cpp_out, check_ctavg=False):
    assert len(cpp_out) == len(nb_out)
    for (t_nb, d_nb, c_nb), (t_cp, d_cp, c_cp) in zip(nb_out, cpp_out):
        np.testing.assert_allclose(d_cp, d_nb, rtol=1e-3, atol=1e-4)
        if check_ctavg:
            np.testing.assert_allclose(c_cp, c_nb, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("cfun,src_cvars", [
    (Linear(a=np.array([0.7]), b=np.array([0.2])), (0,)),
    (Scaling(a=np.array([1.3])), (0,)),
    (Sigmoidal(a=np.array([0.9]), sigma=np.array([0.5]),
               midpoint=np.array([2.0]), cmin=np.array([-0.2]),
               cmax=np.array([1.1])), (0,)),
    (Difference(a=np.array([1.1])), (0,)),
    (KuramotoCfun(a=np.array([1.0])), (0,)),
    (HyperbolicTangent(a=np.array([1.0]), b=np.array([0.5]),
                       midpoint=np.array([1.0]), sigma=np.array([2.0])), (0,)),
    (SigmoidalJansenRit(), (0, 1)),               # classic (2 src cvars)
    (SigmoidalJansenRit(use_classic=0), (0,)),    # legacy
    (PreSigmoidal(dynamic=False, theta=np.array([2.0])), (0,)),   # static
    (PreSigmoidal(dynamic=True), (0, 1)),            # dynamic
    (PreSigmoidal(dynamic=True, globalT=True), (0, 1)), # dynamic + global threshold
])
def test_coupling_parity(cfun, src_cvars):
    nb_out, cpp_out = _run_pair(cfun, src_cvars)
    _assert_close(nb_out, cpp_out)


@pytest.mark.parametrize("cr,cv", [(0.0, 0.85), (0.4, 0.85)])
def test_mpr_nonzero_cv_parity(cr, cv):
    """MontbrioPazoRoxin with non-zero cv couples through cvar[1] (Coupling_Term_V)."""
    ics = _initial_states_pair()

    def build(cv_val):
        A = _subnetwork("A", 3)
        B = _subnetwork("B", 4, cr=cr, cv=cv_val)
        proj = _projection(
            A, B, Linear(a=np.array([1.2]), b=np.array([0.0])),
            source_cvar=(0,), target_cvar=(1,),
        )
        ns = NetworkSet(subnets=[A, B], projections=[proj])
        ns.configure()
        return ns

    ns_nb = build(cv)
    nb_out = NbHybridBackend().compile(ns_nb, eager=True).run(
        NSTEP, initial_states=[x.copy() for x in ics],
    )
    ns_cp = build(cv)
    cpp_out = CppHybridBackend().run_network(
        ns_cp, NSTEP, initial_states=[x.copy() for x in ics],
    )
    _assert_close(nb_out, cpp_out, check_ctavg=True)

    # Non-zero cv must drive Coupling_Term_V and change B's trajectory vs cv=0.
    ns_zero = build(0.0)
    zero_out = CppHybridBackend().run_network(
        ns_zero, NSTEP, initial_states=[x.copy() for x in ics],
    )
    assert np.max(np.abs(cpp_out[1][2][:, 1])) > 1e-3
    assert np.max(np.abs(cpp_out[1][1] - zero_out[1][1])) > 1e-4


@pytest.mark.parametrize(
    "source_cvar,target_cvar,target_scales,cfun",
    [
        pytest.param(
            (0,), (0, 1), None,
            Linear(a=np.array([1.1]), b=np.array([0.05])),
            id="1_to_many-default",
        ),
        pytest.param(
            (0,), (0, 1), np.array([0.4, 0.9]),
            Difference(a=np.array([1.2])),
            id="1_to_many-scaled-difference",
        ),
        pytest.param(
            (0, 1), (0, 1), np.array([0.3, 0.7]),
            Linear(a=np.array([1.0]), b=np.array([0.0])),
            id="n_to_n-target_scales",
        ),
    ],
)
def test_multi_target_cvar_parity(source_cvar, target_cvar, target_scales, cfun):
    """Multi-element target_cvar (1_to_many and n_to_n) matches NbHybridBackend."""
    ics = _initial_states_pair()

    def build():
        A = _subnetwork("A", 3)
        B = _subnetwork("B", 4, cr=0.5, cv=0.8)
        proj = _projection(
            A, B, cfun,
            source_cvar=source_cvar,
            target_cvar=target_cvar,
            target_scales=target_scales,
        )
        ns = NetworkSet(subnets=[A, B], projections=[proj])
        ns.configure()
        return ns

    nb_out = NbHybridBackend().compile(build(), eager=True).run(
        NSTEP, initial_states=[x.copy() for x in ics],
    )
    cpp_out = CppHybridBackend().run_network(
        build(), NSTEP, initial_states=[x.copy() for x in ics],
    )
    _assert_close(nb_out, cpp_out, check_ctavg=True)
    # Both target coupling channels must be actively populated.
    ctavg_b = cpp_out[1][2]
    assert np.max(np.abs(ctavg_b[:, 0])) > 1e-3
    assert np.max(np.abs(ctavg_b[:, 1])) > 1e-3


def test_sweep_coupling_scale_updates_all_split_channels():
    """Sweeping coupling_scale on an n_to_n split projection updates every channel."""
    ics = _initial_states_pair()
    scales = np.array([0.0, 0.5, 1.5, 2.5], dtype=np.float32)
    ts = np.array([0.3, 0.7], dtype=np.float64)

    def build(cfun_a=1.0):
        A = _subnetwork("A", 3)
        B = _subnetwork("B", 4, cr=0.5, cv=0.8)
        proj = _projection(
            A, B, Linear(a=np.array([cfun_a]), b=np.array([0.0])),
            source_cvar=(0, 1), target_cvar=(0, 1),
            target_scales=ts, scale=1.5,
        )
        ns = NetworkSet(subnets=[A, B], projections=[proj])
        ns.configure()
        return ns

    cpp_backend = CppHybridBackend()
    cpp_res = cpp_backend.sweep(
        build(), {"coupling_scale": scales}, nstep=NSTEP,
        initial_states=[x.copy() for x in ics],
    )
    nb_res = NbHybridBackend().sweep(
        build(), {"coupling_scale": scales}, nstep=NSTEP,
        backend="cpu", n_workers=1,
        initial_states=[x.copy() for x in ics],
    )

    np.testing.assert_allclose(
        cpp_res.tavg["B"], nb_res.tavg["B"], rtol=1e-3, atol=1e-4,
    )
    np.testing.assert_allclose(
        cpp_res.ctavg["B"], nb_res.ctavg["B"], rtol=1e-3, atol=1e-4,
    )

    # Every sweep point must match an explicit single-point run on both channels.
    for k, val in enumerate(scales):
        single = cpp_backend.run_network(
            build(cfun_a=float(val)), NSTEP,
            initial_states=[x.copy() for x in ics],
        )
        np.testing.assert_allclose(
            cpp_res.tavg["B"][k], single[1][1], rtol=1e-4, atol=1e-5,
        )
        np.testing.assert_allclose(
            cpp_res.ctavg["B"][k], single[1][2], rtol=1e-4, atol=1e-5,
        )

    # Both channels must be zero at scale=0 and non-zero/distinct at scale=2.5.
    np.testing.assert_allclose(cpp_res.ctavg["B"][0], 0.0, atol=1e-7)
    assert np.max(np.abs(cpp_res.ctavg["B"][-1, :, 0])) > 1e-2
    assert np.max(np.abs(cpp_res.ctavg["B"][-1, :, 1])) > 1e-2


@pytest.mark.parametrize(
    "integ_a,integ_b",
    [
        (HeunDeterministic, EulerDeterministic),
        (EulerDeterministic, HeunDeterministic),
    ],
)
def test_mixed_per_subnet_integrators_parity(integ_a, integ_b):
    """Mixed per-subnet integrators (Heun + Euler) match NbHybridBackend."""
    ics = _initial_states_pair()

    def build(cls_a, cls_b):
        A = _subnetwork("A", 3, integrator_cls=cls_a)
        B = _subnetwork("B", 4, integrator_cls=cls_b, cr=0.6, cv=0.4)
        proj = _projection(
            A, B, Linear(a=np.array([1.2]), b=np.array([0.0])),
            source_cvar=(0, 1), target_cvar=(0, 1),
        )
        ns = NetworkSet(subnets=[A, B], projections=[proj])
        ns.configure()
        return ns

    nb_out = NbHybridBackend().compile(build(integ_a, integ_b), eager=True).run(
        NSTEP, initial_states=[x.copy() for x in ics],
    )
    cpp_out = CppHybridBackend().run_network(
        build(integ_a, integ_b), NSTEP, initial_states=[x.copy() for x in ics],
    )
    _assert_close(nb_out, cpp_out, check_ctavg=True)

    # Verify subnet B does not silently inherit subnet A's integrator.
    all_same = CppHybridBackend().run_network(
        build(integ_a, integ_a), NSTEP, initial_states=[x.copy() for x in ics],
    )
    assert np.max(np.abs(cpp_out[1][1] - all_same[1][1])) > 1e-5


def _mm_subnetwork(name, n_nodes, n_modes, cr=1.0, cv=0.0):
    model = MontbrioPazoRoxin(
        cr=np.array([cr], dtype=np.float64),
        cv=np.array([cv], dtype=np.float64),
    )
    model.number_of_modes = n_modes
    model.configure()
    scheme = HeunDeterministic(dt=DT)
    sn = Subnetwork(name=name, model=model, scheme=scheme, nnodes=n_nodes)
    sn.configure()
    return sn


def test_fast_path_inter_mode_map_parity():
    """1-mode InterProjection fast path applies mode_map[0] cleanly."""
    ics = _initial_states_pair()

    def build(mm_val):
        A = _subnetwork("A", 3)
        B = _subnetwork("B", 4)
        proj = _projection(
            A, B, Linear(a=np.array([1.0]), b=np.array([0.0])),
            source_cvar=(0,), target_cvar=(0,),
        )
        # InterProjection.mode_map is NArray(dtype=np.int_); assign via __dict__
        # so fractional mode_map weights pass directly to both backends.
        proj.__dict__["mode_map"] = np.array([[mm_val]], dtype=np.float64)
        ns = NetworkSet(subnets=[A, B], projections=[proj])
        ns.configure()
        return ns

    nb_out = NbHybridBackend().compile(build(2.5), eager=True).run(
        NSTEP, initial_states=[x.copy() for x in ics],
    )
    cpp_out = CppHybridBackend().run_network(
        build(2.5), NSTEP, initial_states=[x.copy() for x in ics],
    )
    _assert_close(nb_out, cpp_out, check_ctavg=True)

    # Must scale ctavg by 2.5 relative to mode_map=[[1.0]].
    unit_out = CppHybridBackend().run_network(
        build(1.0), NSTEP, initial_states=[x.copy() for x in ics],
    )
    assert np.max(np.abs(cpp_out[1][2])) > 1e-3
    np.testing.assert_allclose(
        cpp_out[1][2], 2.5 * unit_out[1][2], rtol=1e-4, atol=1e-5,
    )


@pytest.mark.parametrize(
    "cfun",
    [
        Difference(a=np.array([1.2])),
        KuramotoCfun(a=np.array([1.2])),
    ],
)
def test_multimode_pre_xi_parity(cfun):
    """Multi-mode Difference/Kuramoto index target state xi at mode m, not 0."""
    n_modes = 3
    rng = np.random.RandomState(11)
    ic_a = rng.uniform(0.1, 0.4, (2, 3, n_modes)).astype(np.float64)
    ic_b = rng.uniform(0.1, 0.4, (2, 4, n_modes)).astype(np.float64)
    # Separate modes distinctly so reading mode 0 for modes 1..2 diverges.
    for m in range(n_modes):
        ic_a[:, :, m] += 0.6 * m
        ic_b[:, :, m] -= 0.7 * m
    ics = [ic_a, ic_b]

    def build():
        A = _mm_subnetwork("A", 3, n_modes)
        B = _mm_subnetwork("B", 4, n_modes)
        proj = _projection(
            A, B, cfun,
            source_cvar=(0,), target_cvar=(0,),
        )
        proj.mode_map = np.eye(n_modes, dtype=np.int_)
        ns = NetworkSet(subnets=[A, B], projections=[proj])
        ns.configure()
        return ns

    nb_out = NbHybridBackend().compile(build(), eager=True).run(
        NSTEP, initial_states=[x.copy() for x in ics],
    )
    cpp_out = CppHybridBackend().run_network(
        build(), NSTEP, initial_states=[x.copy() for x in ics],
    )
    _assert_close(nb_out, cpp_out, check_ctavg=True)
    # Coupling across modes must remain distinct.
    assert np.max(np.abs(cpp_out[1][2][..., 1] - cpp_out[1][2][..., 0])) > 1e-2


def test_kuramoto_multi_cvar_inv_n_parity():
    """Multi-cvar Kuramoto (source_cvar=(0, 1)) applies inv_N = 0.5."""
    ics = _initial_states_pair()

    def build():
        A = _subnetwork("A", 3)
        B = _subnetwork("B", 4)
        proj = _projection(
            A, B, KuramotoCfun(a=np.array([1.2])),
            source_cvar=(0, 1), target_cvar=(0,),
        )
        ns = NetworkSet(subnets=[A, B], projections=[proj])
        ns.configure()
        return ns

    nb_out = NbHybridBackend().compile(build(), eager=True).run(
        NSTEP, initial_states=[x.copy() for x in ics],
    )
    cpp_out = CppHybridBackend().run_network(
        build(), NSTEP, initial_states=[x.copy() for x in ics],
    )
    _assert_close(nb_out, cpp_out, check_ctavg=True)


def test_presigmoidal_dynamic_global_t_empty_connectivity():
    """PreSigmoidal(dynamic=True, globalT=True) on 0-nnz connectivity has no NaN/Inf.

    NbHybridBackend currently raises ZeroDivisionError on empty CSR with
    globalT=True, so CppHybridBackend is cross-checked against an equivalent
    uncoupled C++ run (finite output, zero ctavg, identical trajectories).
    """
    ics = _initial_states_pair()

    def build(with_empty_proj=True):
        A = _subnetwork("A", 3)
        B = _subnetwork("B", 4)
        projs = []
        if with_empty_proj:
            empty = sp.csr_matrix((4, 3), dtype=np.float32)
            projs.append(InterProjection(
                source=A, target=B,
                source_cvar=np.array([0, 1], dtype=np.int_),
                target_cvar=np.array([0], dtype=np.int_),
                weights=empty, lengths=empty,
                cfun=PreSigmoidal(dynamic=True, globalT=True),
                scale=1.5, cv=1.0, dt=DT,
            ))
        ns = NetworkSet(subnets=[A, B], projections=projs)
        ns.configure()
        return ns

    cpp_empty = CppHybridBackend().run_network(
        build(True), NSTEP, initial_states=[x.copy() for x in ics],
    )
    cpp_uncoupled = CppHybridBackend().run_network(
        build(False), NSTEP, initial_states=[x.copy() for x in ics],
    )
    for (_, d_cp, c_cp), (_, d_uc, _) in zip(cpp_empty, cpp_uncoupled):
        assert np.all(np.isfinite(d_cp))
        assert np.all(np.isfinite(c_cp))
        np.testing.assert_allclose(d_cp, d_uc, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(cpp_empty[1][2], 0.0, atol=1e-7)


def test_presigmoidal_dynamic_mixed_global_t_sweep_parity():
    """Mixed globalT=0/1 SIMD lanes match independent W=1 runs."""
    ics = _initial_states_pair()

    def build(global_t):
        A = _subnetwork("A", 3)
        B = _subnetwork("B", 4)
        # Use non-saturating G=1.5 so local vs global threshold diverges
        # measurably (default G=60 saturates tanh to 1.0 for both).
        proj = _projection(
            A, B, PreSigmoidal(dynamic=True, globalT=global_t, G=np.array([1.5])),
            source_cvar=(0, 1), target_cvar=(0,),
        )
        ns = NetworkSet(subnets=[A, B], projections=[proj])
        ns.configure()
        return ns

    # Lane 0 has globalT=0, lane 1 has globalT=1 (catches lane-0-only check
    # and confirms per-lane threshold selection).
    sweep_vals = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    desc = [{"type": "cfun", "projection": "A_to_B", "param_idx": 5}]
    cpp_backend = CppHybridBackend()
    sweep_out = cpp_backend.run_sweep(
        build(False), sweep_vals, nstep=NSTEP,
        initial_states=[x.copy() for x in ics],
        sweep_descriptor=desc,
    )
    ref_local = cpp_backend.run_network(
        build(False), NSTEP, initial_states=[x.copy() for x in ics],
    )
    ref_global = cpp_backend.run_network(
        build(True), NSTEP, initial_states=[x.copy() for x in ics],
    )

    np.testing.assert_allclose(
        sweep_out[0][1][1], ref_local[1][1], rtol=1e-4, atol=1e-5,
    )
    np.testing.assert_allclose(
        sweep_out[0][1][2], ref_local[1][2], rtol=1e-4, atol=1e-5,
    )
    np.testing.assert_allclose(
        sweep_out[1][1][1], ref_global[1][1], rtol=1e-4, atol=1e-5,
    )
    np.testing.assert_allclose(
        sweep_out[1][1][2], ref_global[1][2], rtol=1e-4, atol=1e-5,
    )
    assert np.max(np.abs(sweep_out[0][1][2] - sweep_out[1][1][2])) > 1e-4


@pytest.mark.parametrize(
    "cfun",
    [
        Linear(a=np.array([0.8]), b=np.array([0.05])),
        Difference(a=np.array([0.9])),
    ],
)
def test_n_modes_gt_8_parity(cfun):
    """n_modes = 10 (> 8 stack cap) matches NbHybridBackend without dropping modes."""
    n_modes = 10
    rng = np.random.RandomState(23)
    ic_a = rng.uniform(0.1, 0.5, (2, 3, n_modes)).astype(np.float64)
    ic_b = rng.uniform(0.1, 0.5, (2, 4, n_modes)).astype(np.float64)
    # Give modes 8 and 9 distinct amplitude so dropping them is detectable.
    ic_a[:, :, 8] += 0.8
    ic_a[:, :, 9] += 1.4
    ic_b[:, :, 8] -= 0.4
    ic_b[:, :, 9] -= 0.9
    ics = [ic_a, ic_b]

    mm = np.eye(n_modes, dtype=np.int_)
    mm[8, 0] = 1
    mm[9, 9] = 2

    def build():
        A = _mm_subnetwork("A", 3, n_modes)
        B = _mm_subnetwork("B", 4, n_modes)
        proj = _projection(A, B, cfun, source_cvar=(0,), target_cvar=(0,))
        proj.mode_map = mm
        ns = NetworkSet(subnets=[A, B], projections=[proj])
        ns.configure()
        return ns

    nb_out = NbHybridBackend().compile(build(), eager=True).run(
        NSTEP, initial_states=[x.copy() for x in ics],
    )
    cpp_out = CppHybridBackend().run_network(
        build(), NSTEP, initial_states=[x.copy() for x in ics],
    )
    _assert_close(nb_out, cpp_out, check_ctavg=True)
    assert np.max(np.abs(cpp_out[1][2][:, 0, :, 8])) > 1e-3
    assert np.max(np.abs(cpp_out[1][2][:, 0, :, 9])) > 1e-3
