# -*- coding: utf-8 -*-
"""Data-feature computation over monitor / time-series output.

Chosen subset of the feature taxonomy in the VBI project
(https://github.com/ins-amu/vbi, Apache-2.0) that is easiest to vendor --
all pure NumPy/SciPy, no Java/JIDT jar and no external ML package.  The
functions operate on a time series produced by a monitor and return one
scalar per (region, mode[, lane]) so results can be indexed into a
feature table aligned with a sweep.

Authorization note
------------------
The *taxonomy* and *semantics* follow VBI's ``vbi/feature_extraction``
(Apache License 2.0).  The implementations below are written for this
backend (they operate directly on the backend's monitor-output layout) and
are intentionally dependency-free.  Skipped from VBI as "not easiest to
vendor": Java/JIDT-backed ``calc_te``/``calc_mi``, ``catch22``, ``hmm_stat``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "FEATURES",
    "compute_feature",
    "compute_features",
    "feature_table",
    "abs_energy", "average_power", "auc", "calc_var", "calc_std",
    "calc_mean", "calc_centroid", "calc_kurtosis", "calc_skewness",
    "calc_max", "calc_min", "calc_median", "mean_abs_dev",
    "median_abs_dev", "rms", "interq_range", "zero_crossing",
    "calc_entropy", "calc_moments", "burstiness",
    "spectrum_stats", "spectrum_auc", "spectrum_moments", "psd_raw",
]


# ------------------------------------------------------------------ #
# layout helpers
# ------------------------------------------------------------------ #
def _as_timeremoved(ts, indices=None):
    """Normalize a time series to (leading..., n_samples) over the LAST axis.

    Monitor output arrives as (n_samples, ...) with TIME first (the TVB
    convention); the VBI taxonomy wants time last.  We move the time axis
    to the end so every feature reduces along ``axis=-1`` and returns one
    scalar per leading element (region, mode, lane, ...).

    Parameters
    ----------
    ts : np.ndarray
        Either ``(n_samples, ...)`` (time first) or ``(..., n_samples)``
        (time last).  Ambiguity is resolved by treating the input as time
        first (TVB monitor convention), which is what this package feeds in.
    indices : list of int, optional
        Subset of regions to retain over the leading region axis.
    """
    ts = np.asarray(ts, dtype=np.float64)
    if ts.ndim == 1:
        # a single time series (time first): give it a singleton region
        # axis so the series stays ONE element after the move below and
        # every feature reduces over the whole series (not per sample)
        ts = ts[:, None]
    if indices is not None:
        # regions live on axis 1 of the time-first layout (axis 0 is time)
        ts = np.take(ts, indices, axis=1)
    # move time axis (0) to the end
    ts = np.moveaxis(ts, 0, -1)
    if np.isnan(ts).any() or np.isinf(ts).any():
        return None
    if ts.shape[-1] == 0:
        return None
    return ts


def _safe(fn):
    """Wrap fn(c, ts) -> scalar so failures yield NaN instead of raising."""
    import functools

    @functools.wraps(fn)
    def _wrapped(ts, indices=None, **kw):
        d = _as_timeremoved(ts, indices)
        if d is None:
            return np.nan
        return fn(d, **kw)
    return _wrapped


def _apply(fn, ts, indices=None):
    """Reduce along the (moved) trailing time axis for every leading element."""
    d = _as_timeremoved(ts, indices)
    if d is None:
        return np.array(np.nan)
    return np.asarray(fn(d, axis=-1), dtype=np.float64)


# ------------------------------------------------------------------ #
# simple statistics  (univariate, per series)
# ------------------------------------------------------------------ #
@_safe
def abs_energy(c):
    return np.nansum(np.abs(c) ** 2, axis=-1)


@_safe
def average_power(c):
    return np.nanmean(np.abs(c) ** 2, axis=-1)


@_safe
def auc(c):
    N = c.shape[-1]
    return np.nansum(c, axis=-1) / (1 if N == 0 else N)


@_safe
def calc_var(c):
    return np.nanvar(c, axis=-1)


@_safe
def calc_std(c):
    return np.nanstd(c, axis=-1)


@_safe
def calc_mean(c):
    return np.nanmean(c, axis=-1)


@_safe
def calc_max(c):
    return np.nanmax(c, axis=-1)


@_safe
def calc_min(c):
    return np.nanmin(c, axis=-1)


@_safe
def calc_median(c):
    return np.nanmedian(c, axis=-1)


@_safe
def calc_skewness(c):
    from scipy import stats
    return stats.skew(c, axis=-1, nan_policy="omit")


@_safe
def calc_kurtosis(c):
    from scipy import stats
    return stats.kurtosis(c, axis=-1, nan_policy="omit")


@_safe
def rms(c):
    return np.sqrt(np.nanmean(c ** 2, axis=-1))


@_safe
def mean_abs_dev(c):
    return np.nanmean(np.abs(c - c.mean(axis=-1, keepdims=True)), axis=-1)


@_safe
def median_abs_dev(c):
    med = np.nanmedian(c, axis=-1, keepdims=True)
    return np.nanmedian(np.abs(c - med), axis=-1)


@_safe
def interq_range(c):
    q75 = np.nanpercentile(c, 75, axis=-1)
    q25 = np.nanpercentile(c, 25, axis=-1)
    return q75 - q25


@_safe
def calc_centroid(c):
    N = c.shape[-1]
    idx = np.arange(N, dtype=np.float64)
    s = np.nansum(np.abs(c), axis=-1)
    return np.nansum(idx * c, axis=-1) / np.where(s == 0, 1, s)


@_safe
def calc_moments(c, order=4):
    """Central moments up to *order* (stacked on a new leading axis)."""
    mom = c.mean(axis=-1, keepdims=True)
    center = c - mom
    out = [np.ones_like(c[..., 0])]
    for k in range(1, int(order) + 1):
        out.append(np.nanmean(center ** k, axis=-1))
    return np.stack(out, axis=0)


@_safe
def zero_crossing(c):
    signs = np.signbit(c[..., :-1]) != np.signbit(c[..., 1:])
    return np.sum(signs, axis=-1).astype(np.float64)


@_safe
def burstiness(c):
    """Burstiness coefficient from the coefficient of variation (k=0 none)."""
    mu = np.nanmean(c, axis=-1)
    sigma = np.nanstd(c, axis=-1)
    den = mu + sigma
    return np.where(den == 0, 0.0, (sigma - mu) / den)


@_safe
def calc_entropy(c, bins=20):
    """Binned (discrete) Shannon entropy per series (any leading dims)."""
    lo = np.nanmin(c, axis=-1, keepdims=True)
    hi = np.nanmax(c, axis=-1, keepdims=True)
    rng = hi - lo
    rng = np.where(rng == 0, 1.0, rng)
    norm = np.clip(((c - lo) / rng * bins).astype(np.int64), 0, bins - 1)
    flat = norm.reshape(-1, norm.shape[-1])
    H = np.array([_entropy_of_bins(flat[i]) for i in range(flat.shape[0])])
    return H.reshape(norm.shape[:-1])


def _entropy_of_bins(idxs):
    vals, counts = np.unique(idxs, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log(p + 1e-15))


# ------------------------------------------------------------------ #
# frequency-domain  (scipy welch)
# ------------------------------------------------------------------ #
def _spec(c, fs):
    from scipy import signal
    freqs, psd = signal.welch(c, fs=fs, nperseg=min(256, c.shape[-1]), detrend=False)
    return freqs, psd  # psd: (leading..., n_freq)


def _spec_apply(fn, ts, fs=1.0, indices=None):
    d = _as_timeremoved(ts, indices)
    if d is None:
        return np.array(np.nan)
    freqs, psd = _spec(d, fs)
    return fn(freqs, psd)


def _psd_raw_from_spec(freqs, psd):
    # prepend the frequency grid as axis-0 row 0, broadcast across psd's
    # other leading (mode, ...) axes so concatenation works for any rank
    # (welch returns (leading..., n_freq))
    frow = np.broadcast_to(
        freqs.reshape((1,) * (psd.ndim - 1) + (-1,)),
        (1,) + psd.shape[1:-1] + psd.shape[-1:])
    return np.concatenate([frow, psd], axis=0)


def _spectrum_stats_from_spec(freqs, psd):
    # weighted mean frequency, dominant freq, total power
    total = np.nansum(psd, axis=-1)
    fmean = np.nansum(freqs * psd, axis=-1) / np.where(total == 0, 1, total)
    fpeak = freqs[np.argmax(psd, axis=-1)]
    return np.stack([fmean, fpeak.astype(np.float64), total], axis=0)


def _spectrum_auc_from_spec(freqs, psd, lo=None, hi=None):
    if lo is not None and hi is not None:
        m = (freqs >= lo) & (freqs <= hi)
        return np.nansum(psd[..., m], axis=-1)
    return np.nansum(psd, axis=-1)


def _spectrum_moments_from_spec(freqs, psd, order=4):
    total = np.nansum(psd, axis=-1, keepdims=True)
    denom = np.where(total == 0, 1.0, total)
    moments = [total[..., 0]]
    for k in range(1, int(order) + 1):
        moments.append(np.nansum((freqs ** k) * psd, axis=-1) / denom[..., 0])
    return np.stack(moments, axis=0)


def psd_raw(ts, fs=1.0, indices=None):
    d = _as_timeremoved(ts, indices)
    if d is None:
        return np.array([np.nan])
    freqs, psd = _spec(d, fs)
    return _psd_raw_from_spec(freqs, psd)


def spectrum_stats(ts, fs=1.0, indices=None):
    return _spec_apply(_spectrum_stats_from_spec, ts, fs, indices)


def spectrum_auc(ts, fs=1.0, indices=None, lo=None, hi=None):
    return _spec_apply(
        lambda freqs, psd: _spectrum_auc_from_spec(freqs, psd, lo=lo, hi=hi),
        ts, fs, indices)


def spectrum_moments(ts, fs=1.0, indices=None, order=4):
    return _spec_apply(
        lambda freqs, psd: _spectrum_moments_from_spec(freqs, psd, order=order),
        ts, fs, indices)


# ------------------------------------------------------------------ #
# registry + batch
# ------------------------------------------------------------------ #
FEATURES = {
    "abs_energy": abs_energy,
    "average_power": average_power,
    "auc": auc,
    "var": calc_var,
    "std": calc_std,
    "mean": calc_mean,
    "max": calc_max,
    "min": calc_min,
    "median": calc_median,
    "skewness": calc_skewness,
    "kurtosis": calc_kurtosis,
    "centroid": calc_centroid,
    "rms": rms,
    "mad": mean_abs_dev,
    "median_abs_dev": median_abs_dev,
    "iqr": interq_range,
    "zero_crossing": zero_crossing,
    "burstiness": burstiness,
    "entropy": calc_entropy,
    "moments": calc_moments,
    "psd_raw": psd_raw,
    "spectrum_stats": spectrum_stats,
    "spectrum_auc": spectrum_auc,
    "spectrum_moments": spectrum_moments,
}

_SPECTRAL_FEATURES = frozenset({
    "psd_raw", "spectrum_stats", "spectrum_auc", "spectrum_moments",
})


def _get_cpp_compute_features():
    try:
        from . import _cpp_hybrid
        return getattr(_cpp_hybrid, "compute_features", None)
    except Exception:
        return None


def _accepted(fn, params):
    import inspect
    try:
        sig = inspect.signature(fn)
        ok = {}
        for k, v in params.items():
            # 'indices' is a universal option: the _safe wrapper and every
            # spectral feature accept it even though the underlying fn does
            # not declare it (functools.wraps shows fn's signature)
            if k == "indices" or (
                    k in sig.parameters
                    and sig.parameters[k].kind is not inspect.Parameter.VAR_POSITIONAL):
                ok[k] = v
        return ok
    except (TypeError, ValueError):
        return params


def _compute_feature_py(ts, feature, **params):
    fn = FEATURES.get(feature, feature)
    if not callable(fn):
        raise KeyError(f"unknown feature {feature!r}; choices: {sorted(FEATURES)}")
    return np.asarray(fn(ts, **_accepted(fn, params)), dtype=np.float64)


def _compute_features_py(ts, features, **params):
    """Pure NumPy/SciPy path with single-Welch caching across spectral features."""
    feat_list = list(features)
    n_spec = sum(1 for k in feat_list if isinstance(k, str) and k in _SPECTRAL_FEATURES)
    if n_spec <= 1:
        return [(k, _compute_feature_py(ts, k, **params)) for k in feat_list]

    indices = params.get("indices", None)
    fs = params.get("fs", 1.0)
    d = _as_timeremoved(ts, indices)
    spec_cache = None if d is None else _spec(d, fs)

    out = []
    for k in feat_list:
        if isinstance(k, str) and k in _SPECTRAL_FEATURES:
            if spec_cache is None:
                val = np.array([np.nan]) if k == "psd_raw" else np.array(np.nan)
            else:
                freqs, psd = spec_cache
                if k == "psd_raw":
                    val = _psd_raw_from_spec(freqs, psd)
                elif k == "spectrum_stats":
                    val = _spectrum_stats_from_spec(freqs, psd)
                elif k == "spectrum_auc":
                    val = _spectrum_auc_from_spec(
                        freqs, psd, lo=params.get("lo"), hi=params.get("hi"))
                else:  # spectrum_moments
                    val = _spectrum_moments_from_spec(
                        freqs, psd, order=params.get("order", 4))
            out.append((k, np.asarray(val, dtype=np.float64)))
        else:
            out.append((k, _compute_feature_py(ts, k, **params)))
    return out


def _try_compute_features_cpp(ts, feat_list, params):
    """Optional C++ fast path delegating to ``_cpp_hybrid.compute_features``.

    Expected binding contract (exposed in Wave 2 via ``cph_features.hpp``):
      ``_cpp_hybrid.compute_features(flat_2d, names, fs=1.0, bins=20, order=4, lo=None, hi=None)``
    where ``flat_2d`` is contiguous ``(n_series, n_time)`` float32/float64 and
    returns a dict (or list of ``(name, value)`` pairs) with:
      - scalar features: ``(n_series,)`` float64
      - ``moments``, ``spectrum_stats``, ``spectrum_moments``: ``(n_comp, n_series)`` float64
      - ``psd_raw``: ``(freqs_1d, psd_2d)`` tuple or ``(1 + n_series, n_freq)`` float64.
    """
    cpp_fn = _get_cpp_compute_features()
    if cpp_fn is None or not feat_list:
        return None
    if any(not isinstance(k, str) or k not in FEATURES for k in feat_list):
        return None
    order = int(params.get("order", 4))
    bins = int(params.get("bins", 20))
    if bins <= 0 or order < 0 or ("moments" in feat_list and order > 4):
        return None

    indices = params.get("indices", None)
    arr = np.asarray(ts)
    if arr.ndim == 1:
        arr = arr[:, None]
    if indices is not None:
        arr = np.take(arr, indices, axis=1)
    if arr.shape[0] == 0:
        return None

    moved = np.moveaxis(arr, 0, -1)
    leading = moved.shape[:-1]
    n_time = moved.shape[-1]
    n_series = int(np.prod(leading))
    if n_series == 0:
        return None

    dtype = np.float32 if arr.dtype == np.float32 else np.float64
    flat = np.ascontiguousarray(moved, dtype=dtype).reshape(n_series, n_time)
    if not np.isfinite(flat).all():
        return None

    fs = float(params.get("fs", 1.0))
    lo = params.get("lo", None)
    hi = params.get("hi", None)
    raw = cpp_fn(flat, feat_list, fs=fs, bins=bins, order=order, lo=lo, hi=hi)
    raw_map = dict(raw) if not isinstance(raw, dict) else raw

    out = []
    for k in feat_list:
        val = raw_map[k]
        if k == "psd_raw":
            if isinstance(val, tuple):
                freqs, psd_2d = np.asarray(val[0], dtype=np.float64), np.asarray(val[1], dtype=np.float64)
            else:
                v_arr = np.asarray(val, dtype=np.float64)
                freqs = v_arr[0]
                psd_2d = v_arr[1:]
            psd_r = psd_2d.reshape(leading + (freqs.shape[0],))
            res = _psd_raw_from_spec(freqs, psd_r)
        elif k in ("moments", "spectrum_stats", "spectrum_moments"):
            v_arr = np.asarray(val, dtype=np.float64)
            res = v_arr.reshape((v_arr.shape[0],) + leading)
        else:
            v_arr = np.asarray(val, dtype=np.float64)
            res = v_arr.reshape(leading)
        out.append((k, np.asarray(res, dtype=np.float64)))
    return out


def compute_feature(ts, feature, **params):
    """Compute one feature over ``ts``, returned as a raw array.

    ``ts`` is time-first ``(n_samples, leading...)`` (monitor layout);
    ``feature`` is a key in :data:`FEATURES` or a callable.  Keyword args
    (``fs``, ``bins``, ...) are forwarded only to features that accept them.
    """
    use_cpp = params.pop("use_cpp", True)
    if isinstance(feature, str) and feature in FEATURES:
        return compute_features(ts, [feature], use_cpp=use_cpp, **params)[0][1]
    return _compute_feature_py(ts, feature, **params)


def compute_features(ts, features, **params):
    """Compute several features over ``ts``.

    Returns a list of ``(feature_name, array)`` where each array has the
    reduced per-(region, mode[, lane]) shape for the moved time axis.
    """
    use_cpp = params.pop("use_cpp", True)
    feat_list = list(features)
    if use_cpp:
        cpp_out = _try_compute_features_cpp(ts, feat_list, params)
        if cpp_out is not None:
            return cpp_out
    return _compute_features_py(ts, feat_list, **params)


def feature_table(ts, features=None, **params):
    """Return a tidy ``(n_regions, n_components)`` float table for a single
    time-first series, best for one network / one monitor channel.  Only
    scalar-per-region features are included by default; ``psd_raw``,
    ``moments`` and ``spectrum_stats`` (which are multi-output) must be
    requested explicitly.

    Layout: one row per region; each feature contributes one column per
    output component (scalar features contribute one column).  Supplied
    options (``fs``, ``bins``, ``order``, ``lo``/``hi``, ...) are forwarded
    to every feature, which accepts the ones it knows.  ``psd_raw``'s
    leading frequency header row is stripped (frequencies are implicit in
    the column order, ``fs`` sets the spacing); its remaining per-region
    spectra become one column per frequency bin.  Feature results are
    never truncated: a feature whose region count disagrees with the table
    raises instead.
    """
    if features is None:
        features = [k for k in FEATURES
                    if k not in ("psd_raw", "moments", "spectrum_stats",
                                 "spectrum_moments")]
    computed = compute_features(ts, features, **params)
    cols = []
    n_regions = None
    for k, v in computed:
        v = np.asarray(v, dtype=np.float64)
        if k == "psd_raw":
            # axis-0 row 0 is the shared frequency grid header; the rest are
            # per-region (per region-mode for higher-rank input) spectra,
            # flattened to rows and transposed into frequency columns
            if v.ndim < 2 or v.shape[0] < 1:
                raise ValueError(f"psd_raw returned {v.shape}; expected "
                                 "(1 + n_regions, ..., n_freq)")
            v = v[1:].reshape(-1, v.shape[-1])   # (regions..., n_freq) columns
        elif k in ("moments", "spectrum_stats", "spectrum_moments"):
            # these stack their components on a NEW axis 0:
            # (n_components, regions...) -> (regions..., n_components) columns
            if v.ndim < 2:
                v = v.reshape(-1, 1)
            else:
                v = v.reshape(v.shape[0], -1).T
        else:
            # scalar per region(-mode) element: the leading shape IS the
            # region-mode row space (time already reduced), so every
            # element becomes a row and the single scalar a column
            v = v.reshape(-1, 1)
        if v.ndim != 2:
            raise ValueError(f"feature {k!r} produced rank-{v.ndim} output")
        if n_regions is None:
            n_regions = v.shape[0]
        elif v.shape[0] != n_regions:
            raise ValueError(
                f"feature {k!r} returned {v.shape[0]} region rows; the "
                f"table has {n_regions} — refusing to truncate")
        cols.append(v)
    if not cols:
        return np.empty((0, 0), dtype=np.float64)
    return np.concatenate(cols, axis=1)
