# -*- coding: utf-8 -*-
"""Sweep tests for the C++ backend: SIMD lanes = sweep members.

Verifies that a batched multithreaded sweep reproduces per-point single
simulations, and that the sweep API matches nb_hybrid's semantics.
"""

import numpy as np
import pytest

from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.models.gast_solla_kennedy import GastSollaKennedy
from tvb.simulator.models.oscillator import Generic2dOscillator
from tvb.simulator.integrators import (
    HeunDeterministic,
    EulerDeterministic,
    HeunStochastic,
    EulerStochastic,
)
from tvb.simulator.noise import Additive
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.backend.cpp_hybrid.backend import CppHybridBackend
from .test_cpp_hybrid_parity import _mpr_subnetwork, _projection, DT


def _ns(coupling_scale=1.5):
    A = _mpr_subnetwork("A", 3)
    B = _mpr_subnetwork("B", 4)
    proj = _projection(A, B, 3, 4, scale=coupling_scale)
    ns = NetworkSet(subnets=[A, B], projections=[proj])
    ns.configure()
    return ns


def test_sweep_matches_single_runs():
    nstep = 30
    values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0], np.float32)
    backend = CppHybridBackend()

    result = backend.sweep(_ns(), {"coupling_scale": values}, nstep=nstep)
    assert result.tavg["B"].shape[0] == len(values)
    assert result.backend.startswith("cpp-simd")

    # per-point single runs must match the sweep lanes
    for k, v in enumerate(values):
        ns = _ns(coupling_scale=float(v))
        out = backend.run_network(ns, nstep)
        single_B = out[1][1]  # (n_samp, n_voi, n_nodes, 1)
        swept = result.tavg["B"][k]  # (n_samp, n_voi, n_nodes, 1)
        np.testing.assert_allclose(swept, single_B, rtol=1e-4, atol=1e-5)


def test_run_sweep_legacy_descriptor():
    nstep = 30
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0], np.float32)
    backend = CppHybridBackend()
    out = backend.run_sweep(
        _ns(), values, nstep=nstep,
        sweep_descriptor=[{"type": "cfun", "projection": "A_to_B",
                           "param_idx": 0}],
    )
    assert len(out) == len(values)
    # compare against explicit single runs
    for k, v in enumerate(values):
        ns = _ns(coupling_scale=float(v))
        single = backend.run_network(ns, nstep)[1][1]
        np.testing.assert_allclose(out[k][1][1], single, rtol=1e-4, atol=1e-5)


def test_sweep_model_param():
    nstep = 30
    taus = np.array([0.8, 1.0, 1.2, 1.5], np.float32)
    backend = CppHybridBackend()
    result = backend.sweep(_ns(), {"A.tau": taus}, nstep=nstep)
    for k, tv in enumerate(taus):
        ns = _ns()
        ns.subnets[0].model.tau = np.array([float(tv)])
        ns.subnets[0].model.configure()
        out = backend.run_network(ns, nstep)
        np.testing.assert_allclose(
            result.tavg["A"][k], out[0][1], rtol=1e-4, atol=1e-5)


def test_gast_solla_kennedy_boundaries_run_and_sweep():
    """GastSollaKennedy r >= 0, r_i >= 0 boundaries hold in run_network & sweep."""
    nstep = 30
    x0 = np.array(
        [
            [[0.02], [0.03]],
            [[-65.0], [-65.0]],
            [[0.0], [0.0]],
            [[0.5], [0.5]],
            [[0.02], [0.03]],
            [[-65.0], [-65.0]],
            [[0.0], [0.0]],
            [[22.0], [22.0]],
        ],
        dtype=np.float64,
    )

    def build(I_val=-30.0):
        model = GastSollaKennedy(
            I=np.array([I_val]),
            I_i=np.array([-20.0]),
            Delta=np.array([0.0]),
            Delta_i=np.array([0.0]),
            J_ff=np.array([25.0]),
            J_rf=np.array([100.0]),
            variables_of_interest=("r", "v", "r_i", "v_i"),
        )
        model.configure()
        sn = Subnetwork(
            name="G", model=model, scheme=HeunDeterministic(dt=0.1), nnodes=2,
        )
        sn.configure()
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()
        return ns

    nb = NbHybridBackend()
    cpp = CppHybridBackend()

    # Single run_network parity + non-negativity
    nb_out = nb.compile(build(), eager=True).run(
        nstep, initial_states=[x0.copy()],
    )
    cp_out = cpp.run_network(build(), nstep, initial_states=[x0.copy()])
    np.testing.assert_allclose(cp_out[0][1], nb_out[0][1], rtol=1e-3, atol=1e-5)
    assert np.all(cp_out[0][1][:, 0] >= 0.0)
    assert np.all(cp_out[0][1][:, 2] >= 0.0)
    assert float(cp_out[0][1][:, 0].min()) == 0.0
    assert float(cp_out[0][1][:, 2].min()) == 0.0

    # Batched sweep parity (width=8 and width=1)
    currents = np.array([-40.0, -30.0, -20.0, -5.0], dtype=np.float32)
    nb_sw = nb.sweep(
        build(), {"G.I": currents}, nstep=nstep, backend="cpu",
        initial_states=[x0.copy()],
    )
    for width in (1, 8):
        cp_sw = cpp.sweep(
            build(), {"G.I": currents}, nstep=nstep, width=width,
            initial_states=[x0.copy()],
        )
        np.testing.assert_allclose(
            cp_sw.tavg["G"], nb_sw.tavg["G"], rtol=1e-3, atol=1e-5,
        )
        assert np.all(cp_sw.tavg["G"][:, :, 0] >= 0.0)
        assert np.all(cp_sw.tavg["G"][:, :, 2] >= 0.0)


def test_custom_bounded_model_run_and_sweep():
    """Custom finite upper and lower state_variable_boundaries match NbHybrid."""
    nstep = 25
    x0 = np.array([[[0.22], [-0.15]], [[-0.12], [-0.14]]], dtype=np.float64)

    def build(I_val=2.0):
        model = Generic2dOscillator(
            I=np.array([I_val]),
            d=np.array([0.5]),
            variables_of_interest=("V", "W"),
        )
        model.state_variable_boundaries = {
            "V": np.array([-0.2, 0.25]),
            "W": np.array([-0.15, 0.15]),
        }
        model.configure()
        sn = Subnetwork(
            name="B", model=model, scheme=HeunDeterministic(dt=0.1), nnodes=2,
        )
        sn.configure()
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()
        return ns

    nb = NbHybridBackend()
    cpp = CppHybridBackend()

    nb_out = nb.compile(build(), eager=True).run(
        nstep, initial_states=[x0.copy()],
    )
    cp_out = cpp.run_network(build(), nstep, initial_states=[x0.copy()])
    np.testing.assert_allclose(cp_out[0][1], nb_out[0][1], rtol=1e-4, atol=1e-5)
    assert np.all(cp_out[0][1][:, 0] >= -0.2 - 1e-6)
    assert np.all(cp_out[0][1][:, 0] <= 0.25 + 1e-6)
    assert np.all(cp_out[0][1][:, 1] >= -0.15 - 1e-6)
    assert np.all(cp_out[0][1][:, 1] <= 0.15 + 1e-6)
    # Confirm bounds are actively clamped during the run.
    assert float(cp_out[0][1][:, 0].max()) == pytest.approx(0.25, abs=1e-6)
    assert float(cp_out[0][1][:, 1].min()) == pytest.approx(-0.15, abs=1e-6)

    currents = np.array([-2.0, 0.0, 1.0, 2.0], dtype=np.float32)
    nb_sw = nb.sweep(
        build(), {"B.I": currents}, nstep=nstep, backend="cpu",
        initial_states=[x0.copy()],
    )
    cp_sw = cpp.sweep(
        build(), {"B.I": currents}, nstep=nstep,
        initial_states=[x0.copy()],
    )
    np.testing.assert_allclose(
        cp_sw.tavg["B"], nb_sw.tavg["B"], rtol=1e-4, atol=1e-5,
    )
    assert np.all(cp_sw.tavg["B"][:, :, 0] >= -0.2 - 1e-6)
    assert np.all(cp_sw.tavg["B"][:, :, 0] <= 0.25 + 1e-6)
    assert np.all(cp_sw.tavg["B"][:, :, 1] >= -0.15 - 1e-6)
    assert np.all(cp_sw.tavg["B"][:, :, 1] <= 0.15 + 1e-6)


@pytest.mark.parametrize("integ_cls", [HeunDeterministic, EulerDeterministic])
def test_integrator_clamps_run_and_sweep(integ_cls):
    """Explicit integrator state clamps match NbHybrid in run_network & sweep."""
    nstep = 20
    clamp_val = -1.25
    x0 = np.array(
        [[[0.2], [0.4], [0.6]], [[0.0], [0.5], [-0.5]]], dtype=np.float64,
    )

    def build():
        model = MontbrioPazoRoxin()
        model.configure()
        scheme = integ_cls(dt=0.05)
        scheme.clamped_state_variable_indices = np.array([1], dtype=np.int32)
        scheme.clamped_state_variable_values = np.array([[clamp_val]])
        sn = Subnetwork(name="C", model=model, scheme=scheme, nnodes=3)
        sn.configure()
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()
        return ns

    nb = NbHybridBackend()
    cpp = CppHybridBackend()

    nb_out = nb.compile(build(), eager=True).run(
        nstep, initial_states=[x0.copy()],
    )
    cp_out = cpp.run_network(build(), nstep, initial_states=[x0.copy()])
    np.testing.assert_allclose(cp_out[0][1], nb_out[0][1], rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(cp_out[0][1][:, 1], clamp_val, atol=1e-6)

    taus = np.array([0.8, 1.0, 1.2, 1.5], dtype=np.float32)
    nb_sw = nb.sweep(
        build(), {"C.tau": taus}, nstep=nstep, backend="cpu",
        initial_states=[x0.copy()],
    )
    for width in (1, 8):
        cp_sw = cpp.sweep(
            build(), {"C.tau": taus}, nstep=nstep, width=width,
            initial_states=[x0.copy()],
        )
        np.testing.assert_allclose(
            cp_sw.tavg["C"], nb_sw.tavg["C"], rtol=1e-3, atol=1e-4,
        )
        np.testing.assert_allclose(
            cp_sw.tavg["C"][:, :, 1], clamp_val, atol=1e-6,
        )


def test_sweep_merge_guard_and_node_indices():
    """Verify sweep() merge_tavg behavior across heterogeneous VOIs, dt, and node_indices."""
    nstep = 10
    taus = np.array([0.9, 1.1], dtype=np.float32)
    nb = NbHybridBackend()
    cpp = CppHybridBackend()

    def mk_sn(name, n_nodes, dt=DT, voi=("r", "V"), node_idx=None):
        model = MontbrioPazoRoxin(variables_of_interest=voi)
        model.configure()
        sn = Subnetwork(
            name=name, model=model, scheme=HeunDeterministic(dt=dt), nnodes=n_nodes,
        )
        if node_idx is not None:
            sn.node_indices = np.asarray(node_idx, dtype=np.int64)
        sn.configure()
        return sn

    # 1. Heterogeneous VOI counts across >1 subnets -> merged_tavg is None in both.
    ns_het_voi = NetworkSet(
        subnets=[mk_sn("A", 2, voi=("r", "V")), mk_sn("B", 3, voi=("r",))],
        projections=[],
    )
    ns_het_voi.configure()
    cp_het = cpp.sweep(ns_het_voi, {"A.tau": taus}, nstep=nstep)
    nb_het = nb.sweep(ns_het_voi, {"A.tau": taus}, nstep=nstep, backend="cpu")
    assert cp_het.merged_tavg is None
    assert nb_het.merged_tavg is None
    np.testing.assert_allclose(cp_het.tavg["A"], nb_het.tavg["A"], rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(cp_het.tavg["B"], nb_het.tavg["B"], rtol=1e-3, atol=1e-4)

    # 2. Heterogeneous dt across >1 subnets -> CppHybridBackend sets merged_tavg is None.
    ns_het_dt = NetworkSet(
        subnets=[
            mk_sn("A", 2, dt=0.01, node_idx=[0, 1]),
            mk_sn("B", 2, dt=0.02, node_idx=[2, 3]),
        ],
        projections=[],
    )
    ns_het_dt.configure()
    cp_dt = cpp.sweep(ns_het_dt, {"A.tau": taus}, nstep=nstep)
    assert cp_dt.merged_tavg is None

    # 3. Homogeneous VOIs & equal dt with missing node_indices (none or partial):
    #    both CppHybridBackend and NbHybridBackend fall back to node concatenation.
    for idx_a, idx_b in [(None, None), ([0, 2], None)]:
        ns_no_idx = NetworkSet(
            subnets=[
                mk_sn("A", 2, node_idx=idx_a),
                mk_sn("B", 3, node_idx=idx_b),
            ],
            projections=[],
        )
        ns_no_idx.configure()
        cp_cat = cpp.sweep(ns_no_idx, {"A.tau": taus}, nstep=nstep)
        nb_cat = nb.sweep(ns_no_idx, {"A.tau": taus}, nstep=nstep, backend="cpu")
        assert cp_cat.merged_tavg is not None
        assert cp_cat.merged_tavg.shape == (2, nstep, 2, 5, 1)
        np.testing.assert_allclose(
            cp_cat.merged_tavg, nb_cat.merged_tavg, rtol=1e-3, atol=1e-4,
        )

    # 4. Mergeable subnets with complete node_indices scatter into connectome order.
    ns_merge = NetworkSet(
        subnets=[
            mk_sn("A", 2, node_idx=[0, 2]),
            mk_sn("B", 2, node_idx=[1, 3]),
        ],
        projections=[],
    )
    ns_merge.configure()
    cp_merged = cpp.sweep(ns_merge, {"A.tau": taus}, nstep=nstep)
    nb_merged = nb.sweep(
        ns_merge, {"A.tau": taus}, nstep=nstep, backend="cpu",
        node_indices={"A": [0, 2], "B": [1, 3]},
    )
    assert cp_merged.merged_tavg is not None
    assert cp_merged.merged_tavg.shape == (2, nstep, 2, 4, 1)
    np.testing.assert_allclose(
        cp_merged.merged_tavg, nb_merged.merged_tavg, rtol=1e-3, atol=1e-4,
    )


@pytest.mark.parametrize("stoch_cls", [HeunStochastic, EulerStochastic])
def test_sweep_rejects_stochastic_integrators(stoch_cls):
    """Both sweep() and run_sweep() reject stochastic integrators with NotImplementedError."""
    m_det = MontbrioPazoRoxin()
    m_det.configure()
    m_st = MontbrioPazoRoxin()
    m_st.configure()
    sn_det = Subnetwork(
        name="det", model=m_det, scheme=HeunDeterministic(dt=DT), nnodes=2,
    ).configure()
    sn_st = Subnetwork(
        name="st", model=m_st,
        scheme=stoch_cls(dt=DT, noise=Additive(nsig=np.array([1e-3]))),
        nnodes=2,
    ).configure()
    ns = NetworkSet(subnets=[sn_det, sn_st], projections=[])
    ns.configure()

    backend = CppHybridBackend()
    values = np.array([0.8, 1.2], dtype=np.float32)
    with pytest.raises(NotImplementedError, match="stochastic"):
        backend.sweep(ns, {"st.tau": values}, nstep=10)
    with pytest.raises(NotImplementedError, match="stochastic"):
        backend.run_sweep(
            ns, values, nstep=10,
            sweep_descriptor=[{"type": "model", "subnet": "st", "param": "tau"}],
        )
