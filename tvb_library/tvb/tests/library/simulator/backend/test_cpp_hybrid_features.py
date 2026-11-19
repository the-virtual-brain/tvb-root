# -*- coding: utf-8 -*-
"""Tests for cpp_hybrid data-feature extraction (Python/SciPy + C++ parity)."""

import numpy as np
import pytest
from scipy import signal, stats

from tvb.simulator.backend.cpp_hybrid import _cpp_hybrid
from tvb.simulator.backend.cpp_hybrid import features as feat


_ONLINE_FEATURES = [
    "mean", "var", "std", "min", "max", "rms", "abs_energy", "average_power",
    "auc", "zero_crossing", "burstiness", "moments", "skewness", "kurtosis",
    "centroid",
]


def _ref_entropy(x_1d, bins=20):
    lo = np.min(x_1d)
    hi = np.max(x_1d)
    rng = 1.0 if hi == lo else (hi - lo)
    idx = np.clip(((x_1d - lo) / rng * bins).astype(np.int64), 0, bins - 1)
    _, counts = np.unique(idx, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-15)))


class TestPythonFeatureSemantics:
    def test_all_features_rank1_rank2_rank3(self):
        rng = np.random.default_rng(11)
        for shape in [(200,), (200, 4), (200, 3, 2)]:
            ts = rng.standard_normal(shape) * 2.0 + 0.5
            for name in sorted(feat.FEATURES):
                val = feat.compute_feature(ts, name, fs=20.0, use_cpp=False)
                assert np.isfinite(val).all(), f"{name} on shape {shape}"
                expected_leading = (1,) if len(shape) == 1 else shape[1:]
                if name in ("moments", "spectrum_moments"):
                    assert val.shape == (5,) + expected_leading
                elif name == "spectrum_stats":
                    assert val.shape == (3,) + expected_leading
                elif name == "psd_raw":
                    n_freq = min(256, shape[0]) // 2 + 1
                    assert val.shape == (1 + expected_leading[0],) + expected_leading[1:] + (n_freq,)
                else:
                    assert val.shape == expected_leading

    def test_calc_centroid_rank3_broadcast_fix(self):
        rng = np.random.default_rng(21)
        ts = rng.standard_normal((150, 4, 3))
        got = feat.calc_centroid(ts)
        assert got.shape == (4, 3)
        idx = np.arange(150, dtype=np.float64)
        for r in range(4):
            for m in range(3):
                s = ts[:, r, m]
                den = np.sum(np.abs(s))
                expected = np.sum(idx * s) / den
                assert np.isclose(got[r, m], expected, rtol=1e-12, atol=1e-12)

    def test_exact_formulas_rank2(self):
        rng = np.random.default_rng(42)
        ts = rng.standard_normal((300, 5)) * 1.7 - 0.4
        ts[10, 0] = -0.0
        ts[11, 0] = +0.0

        assert np.allclose(feat.compute_feature(ts, "mean", use_cpp=False), ts.mean(axis=0))
        assert np.allclose(feat.compute_feature(ts, "var", use_cpp=False), ts.var(axis=0, ddof=0))
        assert np.allclose(feat.compute_feature(ts, "std", use_cpp=False), ts.std(axis=0, ddof=0))
        assert np.allclose(feat.compute_feature(ts, "min", use_cpp=False), ts.min(axis=0))
        assert np.allclose(feat.compute_feature(ts, "max", use_cpp=False), ts.max(axis=0))
        assert np.allclose(feat.compute_feature(ts, "rms", use_cpp=False), np.sqrt(np.mean(ts ** 2, axis=0)))
        assert np.allclose(feat.compute_feature(ts, "abs_energy", use_cpp=False), np.sum(ts ** 2, axis=0))
        assert np.allclose(feat.compute_feature(ts, "average_power", use_cpp=False), np.mean(ts ** 2, axis=0))
        assert np.allclose(feat.compute_feature(ts, "auc", use_cpp=False), np.sum(ts, axis=0) / ts.shape[0])
        assert np.allclose(feat.compute_feature(ts, "median", use_cpp=False), np.median(ts, axis=0))
        assert np.allclose(
            feat.compute_feature(ts, "mad", use_cpp=False),
            np.mean(np.abs(ts - ts.mean(axis=0, keepdims=True)), axis=0),
        )
        assert np.allclose(
            feat.compute_feature(ts, "median_abs_dev", use_cpp=False),
            np.median(np.abs(ts - np.median(ts, axis=0, keepdims=True)), axis=0),
        )
        assert np.allclose(
            feat.compute_feature(ts, "iqr", use_cpp=False),
            np.percentile(ts, 75, axis=0) - np.percentile(ts, 25, axis=0),
        )
        assert np.allclose(
            feat.compute_feature(ts, "skewness", use_cpp=False),
            stats.skew(ts, axis=0, bias=True),
        )
        assert np.allclose(
            feat.compute_feature(ts, "kurtosis", use_cpp=False),
            stats.kurtosis(ts, axis=0, fisher=True, bias=True),
        )
        zc_ref = np.sum(np.signbit(ts[:-1]) != np.signbit(ts[1:]), axis=0).astype(np.float64)
        assert np.allclose(feat.compute_feature(ts, "zero_crossing", use_cpp=False), zc_ref)

        ent_ref = np.array([_ref_entropy(ts[:, i], bins=20) for i in range(5)])
        assert np.allclose(feat.compute_feature(ts, "entropy", use_cpp=False), ent_ref)

        # Banded spectrum_auc
        fs = 100.0
        freqs, psd = signal.welch(ts.T, fs=fs, nperseg=256, detrend=False)
        mask = (freqs >= 5.0) & (freqs <= 20.0)
        assert np.allclose(
            feat.compute_feature(ts, "spectrum_auc", fs=fs, lo=5.0, hi=20.0, use_cpp=False),
            np.sum(psd[:, mask], axis=-1),
        )

    def test_constant_zero_nan_empty_edge_cases(self):
        zeros = np.zeros((64, 3))
        assert np.allclose(feat.compute_feature(zeros, "centroid", use_cpp=False), 0.0)
        assert np.allclose(feat.compute_feature(zeros, "burstiness", use_cpp=False), 0.0)
        assert np.isnan(feat.compute_feature(zeros, "skewness", use_cpp=False)).all()
        assert np.isnan(feat.compute_feature(zeros, "kurtosis", use_cpp=False)).all()

        bad = np.ones((32, 2))
        bad[5, 1] = np.nan
        assert np.isnan(feat.compute_feature(bad, "mean", use_cpp=False))

        empty = np.empty((0, 2))
        assert np.isnan(feat.compute_feature(empty, "mean", use_cpp=False))

    def test_single_welch_pass_for_multiple_spectral_features(self, monkeypatch):
        calls = []
        orig_welch = signal.welch

        def counting_welch(*args, **kwargs):
            calls.append(1)
            return orig_welch(*args, **kwargs)

        monkeypatch.setattr(signal, "welch", counting_welch)
        ts = np.random.default_rng(3).standard_normal((300, 4))
        spec_names = ["psd_raw", "spectrum_stats", "spectrum_auc", "spectrum_moments"]

        res = dict(feat.compute_features(ts, spec_names, fs=50.0, use_cpp=False))
        assert len(calls) == 1
        for k in spec_names:
            assert np.allclose(res[k], feat.compute_feature(ts, k, fs=50.0, use_cpp=False))

        calls.clear()
        tab = feat.feature_table(ts, features=spec_names, fs=50.0, use_cpp=False)
        assert len(calls) == 1
        assert tab.shape[0] == 4

    def test_mocked_cpp_dispatch_reshapes_cleanly(self, monkeypatch):
        ts = np.random.default_rng(9).standard_normal((200, 3, 2))
        names = ["mean", "moments", "spectrum_stats", "psd_raw"]
        py_res = dict(feat.compute_features(ts, names, fs=25.0, use_cpp=False))

        called = {}

        def fake_cpp(flat, req_names, fs=1.0, bins=20, order=4, lo=None, hi=None):
            called["shape"] = flat.shape
            called["names"] = list(req_names)
            freqs, psd = signal.welch(flat, fs=fs, nperseg=min(256, flat.shape[1]), detrend=False)
            return {
                "mean": py_res["mean"].ravel(),
                "moments": py_res["moments"].reshape(5, -1),
                "spectrum_stats": py_res["spectrum_stats"].reshape(3, -1),
                "psd_raw": (freqs, psd),
            }

        monkeypatch.setattr(feat, "_get_cpp_compute_features", lambda: fake_cpp)
        cpp_res = dict(feat.compute_features(ts, names, fs=25.0))
        assert called["shape"] == (6, 200)
        assert called["names"] == names
        for k in names:
            assert cpp_res[k].shape == py_res[k].shape
            assert np.allclose(cpp_res[k], py_res[k])


class TestCppFeaturesParity:
    @pytest.mark.parametrize("n_time", [200, 300])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_cpp_batch_features_parity(self, n_time, dtype):
        if not hasattr(_cpp_hybrid, "compute_features"):
            pytest.skip("_cpp_hybrid.compute_features not bound yet (Wave 2)")

        rng = np.random.default_rng(123)
        ts = (rng.standard_normal((n_time, 3, 2)) * 2.5 - 0.3).astype(dtype)
        all_names = sorted(feat.FEATURES.keys())

        cpp_res = dict(feat.compute_features(ts, all_names, fs=40.0, use_cpp=True))
        py_res = dict(feat.compute_features(ts, all_names, fs=40.0, use_cpp=False))

        for k in all_names:
            assert cpp_res[k].shape == py_res[k].shape, k
            assert np.allclose(cpp_res[k], py_res[k], rtol=1e-10, atol=1e-11, equal_nan=True), k

        # Banded spectrum_auc + custom bins/order
        band_cpp = feat.compute_feature(ts, "spectrum_auc", fs=40.0, lo=2.0, hi=12.0, use_cpp=True)
        band_py = feat.compute_feature(ts, "spectrum_auc", fs=40.0, lo=2.0, hi=12.0, use_cpp=False)
        assert np.allclose(band_cpp, band_py, rtol=1e-10, atol=1e-11)

    def test_cpp_online_feature_accum_parity(self):
        if not hasattr(_cpp_hybrid, "OnlineFeatureAccum"):
            pytest.skip("_cpp_hybrid.OnlineFeatureAccum not bound yet (Wave 2)")

        rng = np.random.default_rng(77)
        ts = rng.standard_normal((250, 2, 4, 8), dtype=np.float32) * 3.0 + 0.5
        acc = _cpp_hybrid.OnlineFeatureAccum(8)
        acc.reset(2, 4, 0.1)
        for t in range(ts.shape[0]):
            acc.update(ts[t])
        got_map = acc.finalize(_ONLINE_FEATURES)
        py_res = dict(feat.compute_features(ts, _ONLINE_FEATURES, use_cpp=False))
        for k in _ONLINE_FEATURES:
            assert np.allclose(got_map[k], py_res[k], rtol=1e-10, atol=1e-11, equal_nan=True), k
