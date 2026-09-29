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
        # a single time series: assume time
        ts = ts[None, :]
    if indices is not None:
        ts = np.take(ts, indices, axis=0)
    # move time axis (0) to the end
    ts = np.moveaxis(ts, 0, -1)
    if np.isnan(ts).any() or np.isinf(ts).any():
        return None
    if ts.shape[-1] == 0:
        return None
    return ts


def _safe(fn):
    """Wrap fn(c, ts) -> scalar so failures yield NaN instead of raising."""
    def _wrapped(ts, **kw):
        d = _as_timeremoved(ts)
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
    return np.nansum(idx[None, :] * c, axis=-1) / np.where(s == 0, 1, s)


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
    from scipy import signal
    freqs, psd = signal.welch(d, fs=fs,
                              nperseg=min(256, d.shape[-1]), detrend=False)
    return fn(freqs, psd)


def psd_raw(ts, fs=1.0, indices=None):
    d = _as_timeremoved(ts, indices)
    if d is None:
        return np.array([np.nan])
    from scipy import signal
    freqs, psd = signal.welch(d, fs=fs, nperseg=min(256, d.shape[-1]), detrend=False)
    return np.concatenate([freqs[None, ...], psd], axis=0)


def spectrum_stats(ts, fs=1.0, indices=None):
    def _fn(freqs, psd):
        # feigehted mean frequency, dominant freq, bandwidth (FWHM)
        total = np.nansum(psd, axis=-1)
        fmean = np.nansum(freqs[None, :] * psd, axis=-1) / np.where(total == 0, 1, total)
        fpeak = freqs[np.argmax(psd, axis=-1)]
        return np.stack([fmean, fpeak.astype(np.float64), total], axis=0)
    return _spec_apply(_fn, ts, fs, indices)


def spectrum_auc(ts, fs=1.0, indices=None, lo=None, hi=None):
    def _fn(freqs, psd):
        f = freqs
        if lo is not None and hi is not None:
            m = (f >= lo) & (f <= hi)
            return np.nansum(psd[..., m], axis=-1)
        return np.nansum(psd, axis=-1)
    return _spec_apply(_fn, ts, fs, indices)


def spectrum_moments(ts, fs=1.0, indices=None, order=4):
    def _fn(freqs, psd):
        total = np.nansum(psd, axis=-1, keepdims=True)
        total = np.where(total == 0, 1.0, total)
        fn = freqs[None, :]
        moments = [total[..., 0]]
        for k in range(1, int(order) + 1):
            moments.append(np.nansum(fn ** k * psd, axis=-1) / total[..., 0])
        return np.stack(moments, axis=0)
    return _spec_apply(_fn, ts, fs, indices)


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


def _accepted(fn, params):
    import inspect
    try:
        sig = inspect.signature(fn)
        ok = {}
        for k, v in params.items():
            if k in sig.parameters and sig.parameters[k].kind is not inspect.Parameter.VAR_POSITIONAL:
                ok[k] = v
        return ok
    except (TypeError, ValueError):
        return params


def compute_feature(ts, feature, **params):
    """Compute one feature over ``ts``, returned as a raw array.

    ``ts`` is time-first ``(n_samples, leading...)`` (monitor layout);
    ``feature`` is a key in :data:`FEATURES` or a callable.  Keyword args
    (``fs``, ``bins``, ...) are forwarded only to features that accept them.
    """
    fn = FEATURES.get(feature, feature)
    if not callable(fn):
        raise KeyError(f"unknown feature {feature!r}; choices: {sorted(FEATURES)}")
    return np.asarray(fn(ts, **_accepted(fn, params)), dtype=np.float64)


def compute_features(ts, features, **params):
    """Compute several features over ``ts``.

    Returns a list of ``(feature_name, array)`` where each array has the
    reduced per-(region, mode[, lane]) shape for the moved time axis.
    """
    out = []
    for k in features:
        out.append((k, compute_feature(ts, k, **params)))
    return out


def feature_table(ts, features=None, **params):
    """Return a tidy ``(n_regions, n_features)`` float table for a single
    time-first series, best for one network / one monitor channel.  Only
    scalar-per-region features are included by default; ``psd_raw``,
    ``moments`` and ``spectrum_stats`` (which are multi-output) must be
    requested explicitly and are column-joined consistently.
    """
    if features is None:
        features = [k for k in FEATURES
                    if k not in ("psd_raw", "moments", "spectrum_stats",
                                 "spectrum_moments")]
    fs = params.get("fs", 1.0)
    rows = []
    for k in features:
        v = np.asarray(compute_feature(ts, k, fs=fs), dtype=np.float64)
        v = v.ravel()
        rows.append(v)
    n = min(len(r) for r in rows) if rows else 0
    return np.stack([r[:n] for r in rows], axis=1)
