# -*- coding: utf-8 -*-
"""Tests for the notebook-friendly sugar + feature layer on top of the
C++ hybrid backend (width-8 SIMD lane sweeps + VBI-subset features)."""

import numpy as np
import pytest
from scipy import sparse

from tvb.simulator.backend.cpp_hybrid import notebook as nb
from tvb.simulator.backend.cpp_hybrid import features as feat
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.integrators import HeunDeterministic


def build_coupled_net(n=6):
    def subnet(name):
        m = MontbrioPazoRoxin(); m.configure()
        return Subnetwork(name=name, model=m,
                          scheme=HeunDeterministic(dt=0.1), nnodes=n)
    A, B = subnet("A"), subnet("B")
    w = sparse.csr_matrix(np.full((n, n), 0.01, np.float64) - np.eye(n) * 0.01)
    proj = InterProjection(
        source=A, target=B, source_cvar=np.array([0], np.int32),
        target_cvar=np.array([0], np.int32), weights=w,
        lengths=sparse.csr_matrix(np.zeros((n, n))), cv=1.0, dt=0.1,
        scale=2.0, cfun=Linear())
    ns = NetworkSet(subnets=[A, B], projections=[proj])
    ns.configure()
    return ns


class TestNotebookSweep:
    def test_width_constraint(self):
        for n in (1, 5, 20, 500, 20000):
            w = nb.default_width(n, None)
            assert w in (1, 8)  # the C++ core only instantiates these

    def test_bad_width_rejected(self):
        # a valid sweep must be supplied: the missing-parameter check fires
        # before the width validation, and an empty sweep would raise the
        # wrong error (masking a width regression)
        with pytest.raises(ValueError, match="width"):
            nb.sweep(build_coupled_net(), 10, **{"A.eta": [0.1, 0.2]},
                     width=4)

    def test_helpful_error_lists_params(self):
        with pytest.raises(ValueError) as e:
            nb.sweep(build_coupled_net(), 10, nonsense=[1.0, 2.0])
        msg = str(e.value)
        assert "A.eta" in msg or "A.tau" in msg

    def test_multidim_sweep_shapes(self):
        ns = build_coupled_net()
        res = nb.sweep(ns, 100, coupling_scale=np.linspace(0.5, 2.0, 6),
                       **{"A.eta": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}, workers=2)
        assert res.param_keys == ["coupling_scale", "A.eta"]
        assert res.sweep_values.shape == (6, 2)
        assert res.tavg["A"].shape[:1] == (6,)

    def test_features_shapes(self):
        ns = build_coupled_net()
        res = nb.sweep(ns, 60, coupling_scale=np.linspace(0.5, 2.0, 4), workers=2)
        f = res.features(["mean", "std"], fs=10.0, svar=0, subnet="A")
        for k, a in f.items():
            assert np.asarray(a).shape == (4, 6)
        assert np.isfinite(f["mean"]).all()
        assert np.isfinite(f["std"]).all()


class TestFeatures:
    def test_univariate_correctness(self):
        rng = np.random.default_rng(3)
        t = rng.standard_normal((400, 2))           # (samples, node)
        for k in ("mean", "max", "std", "rms"):
            got = np.asarray(feat.compute_feature(t, k))
            assert np.allclose(got, {
                "mean": t.mean(0), "max": t.max(0), "std": t.std(0),
                "rms": np.sqrt((t ** 2).mean(0)),
            }[k], rtol=1e-6)

    def test_constant_series_diagnostics(self):
        c = np.full((200, 3), 5.0)
        assert np.allclose(feat.compute_feature(c, "mean"), 5.0)
        assert np.allclose(feat.compute_feature(c, "std"), 0.0)
        assert np.allclose(feat.compute_feature(c, "zero_crossing"), 0.0)

    def test_time_first_multi_leading(self):
        t = np.random.default_rng(2).standard_normal((200, 4, 2))  # s x node x mode
        v = np.asarray(feat.compute_feature(t, "std"))
        assert v.shape == (4, 2)

    def test_feature_table(self):
        t = np.random.default_rng(0).standard_normal((500, 3))
        tab = feat.feature_table(t, fs=10.0)
        assert tab.shape == (3, len(feat.FEATURES) - 4)  # scalar features only
        assert np.isfinite(tab).all()

    def test_all_features_finite(self):
        t = np.random.default_rng(1).standard_normal((256, 2))
        for name in sorted(feat.FEATURES):
            v = np.asarray(feat.compute_feature(t, name, fs=10.0))
            assert np.isfinite(v).all(), name

    def test_feature_table_multimode_rows(self):
        # (samples, 3 regions, 2 modes): scalar features return (3, 2), which
        # is the region-mode row space (NOT components-first) — the table
        # must keep 6 rows so mixed lists concatenate instead of raising or
        # transposing mean into 2 bogus rows
        t = np.random.default_rng(5).standard_normal((256, 3, 2))
        tab = feat.feature_table(t, features=["mean", "moments", "psd_raw"],
                                 fs=10.0)
        assert tab.shape[0] == 6
        assert np.allclose(tab[:, 0], t.mean(axis=0).ravel())
        assert np.isfinite(tab).all()
