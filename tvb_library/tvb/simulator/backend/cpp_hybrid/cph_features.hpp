#pragma once
// Pure C++17 feature extraction kernels for TVB cpp_hybrid.
//
// Zero nanobind / Python / SciPy / FFTW dependencies; compiles cleanly with
// g++ 13, Clang, and Emscripten em++ (-msimd128).
//
// Two tiers:
//   Tier 1 — Online streaming accumulator (`template <int W> OnlineFeatureAccum`)
//     Consumes observed rows `(n_voi, n_node, W)` per sample step in `double`
//     precision with O(n_voi * n_node * W) memory (zero time-series storage).
//     Computes: mean, var (ddof=0), std (ddof=0), min, max, rms, abs_energy,
//     average_power, auc (matches features.py: sum/N), auc_trapz (dt-scaled
//     trapezoidal integral), zero_crossing, burstiness, moments (orders 0..4),
//     skewness (scipy bias=True), kurtosis (scipy Fisher=True, bias=True),
//     and centroid.
//
//   Tier 2 — Batch post-run kernels over contiguous `(n_series, n_time)` arrays
//     * `compute_order_stats`: median, mad (mean_abs_dev), median_abs_dev,
//       iqr (linear-interpolated Q75 - Q25 matching np.percentile),
//       entropy (20-bin Shannon entropy matching features.py:calc_entropy).
//     * `compute_welch_psd`: self-contained radix-2 Cooley-Tukey FFT (+ Bluestein
//       for non-power-of-two nperseg < 256) implementing Hann-windowed
//       50%-overlap Welch PSD (`nperseg = min(256, n_time)`, `detrend=False`,
//       `scaling='density'`).
//     * `compute_batch_features`: unified batch entry point computing all
//       requested Tier-1, order-stat, and spectral features in minimal passes
//       (a single streaming pass + at most one sort + at most one Welch pass
//       per series).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CPH_M_PI
#define CPH_M_PI 3.14159265358979323846
#endif

namespace cph {

// ------------------------------------------------------------------ //
// Feature metadata & parameters
// ------------------------------------------------------------------ //

struct FeatureParams {
  double fs = 1.0;
  int bins = 20;
  int order = 4;
  bool has_band = false;
  double lo = 0.0;
  double hi = 0.0;
};

struct BatchFeatureOutput {
  std::string name;
  int n_components = 1;
  int n_freq = 0;             // > 0 only for "psd_raw"
  std::vector<double> freqs;  // length n_freq for "psd_raw"
  // Layout:
  //   scalar features:            (n_series,)
  //   moments / spectrum_*:       (n_components, n_series) row-major
  //   psd_raw:                    (n_series, n_freq) row-major (freqs in .freqs)
  std::vector<double> data;
};

inline double quiet_nan() {
  return std::numeric_limits<double>::quiet_NaN();
}

// ------------------------------------------------------------------ //
// Tier 1: Online streaming accumulator (template <int W>)
// ------------------------------------------------------------------ //

struct OnlineFeatureResult {
  int n_voi = 0;
  int n_node = 0;
  int width = 0;
  uint64_t n_samples = 0;
  std::vector<double> mean;
  std::vector<double> var;
  std::vector<double> std_dev;
  std::vector<double> min_v;
  std::vector<double> max_v;
  std::vector<double> rms;
  std::vector<double> abs_energy;
  std::vector<double> average_power;
  std::vector<double> auc;
  std::vector<double> auc_trapz;
  std::vector<double> zero_crossing;
  std::vector<double> burstiness;
  std::vector<double> moments;  // (5, n_voi * n_node * W)
  std::vector<double> skewness;
  std::vector<double> kurtosis;
  std::vector<double> centroid;
};

template <int W>
struct OnlineFeatureAccum {
  static constexpr int WIDTH = W;

  int n_voi = 0;
  int n_node = 0;
  double dt_sample = 1.0;
  uint64_t n = 0;
  bool has_prev = false;

  // Struct-of-Arrays sized (n_voi * n_node * W) with W innermost for SIMD.
  std::vector<double> mean;
  std::vector<double> M2;
  std::vector<double> M3;
  std::vector<double> M4;
  std::vector<double> min_v;
  std::vector<double> max_v;
  std::vector<double> sum_abs;
  std::vector<double> sum_sq;
  std::vector<double> sum_t_x;
  std::vector<double> prev_x;
  std::vector<double> zero_crossings;
  std::vector<double> sum_trapz;
  std::vector<uint8_t> nonfinite;

  void reset(int n_voi_, int n_node_, double dt_sample_ = 1.0) {
    if (n_voi_ < 0 || n_node_ < 0) {
      throw std::invalid_argument("OnlineFeatureAccum::reset: negative shape");
    }
    n_voi = n_voi_;
    n_node = n_node_;
    dt_sample = dt_sample_;
    n = 0;
    has_prev = false;
    const size_t total = static_cast<size_t>(n_voi) *
                         static_cast<size_t>(n_node) * static_cast<size_t>(W);
    mean.assign(total, 0.0);
    M2.assign(total, 0.0);
    M3.assign(total, 0.0);
    M4.assign(total, 0.0);
    min_v.assign(total, 0.0);
    max_v.assign(total, 0.0);
    sum_abs.assign(total, 0.0);
    sum_sq.assign(total, 0.0);
    sum_t_x.assign(total, 0.0);
    prev_x.assign(total, 0.0);
    zero_crossings.assign(total, 0.0);
    sum_trapz.assign(total, 0.0);
    nonfinite.assign(total, 0);
  }

  size_t num_series() const {
    return static_cast<size_t>(n_voi) * static_cast<size_t>(n_node) *
           static_cast<size_t>(W);
  }

  template <typename T>
  void update_typed(const T *obs_row) {
    const size_t n_sn =
        static_cast<size_t>(n_voi) * static_cast<size_t>(n_node);
    if (n_sn == 0) {
      has_prev = true;
      n += 1;
      return;
    }
    if (!has_prev) {
      for (size_t s = 0; s < n_sn; ++s) {
        const size_t base = s * static_cast<size_t>(W);
        const T *xr = obs_row + base;
        double *p_mean = mean.data() + base;
        double *p_min = min_v.data() + base;
        double *p_max = max_v.data() + base;
        double *p_sabs = sum_abs.data() + base;
        double *p_ssq = sum_sq.data() + base;
        double *p_prev = prev_x.data() + base;
        uint8_t *p_nf = nonfinite.data() + base;
#pragma omp simd
        for (int w = 0; w < W; ++w) {
          const double x = static_cast<double>(xr[w]);
          p_nf[w] = static_cast<uint8_t>(!std::isfinite(x));
          p_mean[w] = x;
          p_min[w] = x;
          p_max[w] = x;
          p_sabs[w] = std::fabs(x);
          p_ssq[w] = x * x;
          p_prev[w] = x;
        }
      }
      has_prev = true;
      n = 1;
      return;
    }

    const double n0 = static_cast<double>(n);
    const double n1 = static_cast<double>(n + 1);
    const double inv_n1 = 1.0 / n1;
    const double t_idx = static_cast<double>(n);
    const double c4 = n1 * n1 - 3.0 * n1 + 3.0;
    const double c3 = n1 - 2.0;
    const double half_dt = 0.5 * dt_sample;

    for (size_t s = 0; s < n_sn; ++s) {
      const size_t base = s * static_cast<size_t>(W);
      const T *xr = obs_row + base;
      double *p_mean = mean.data() + base;
      double *p_M2 = M2.data() + base;
      double *p_M3 = M3.data() + base;
      double *p_M4 = M4.data() + base;
      double *p_min = min_v.data() + base;
      double *p_max = max_v.data() + base;
      double *p_sabs = sum_abs.data() + base;
      double *p_ssq = sum_sq.data() + base;
      double *p_stx = sum_t_x.data() + base;
      double *p_prev = prev_x.data() + base;
      double *p_zc = zero_crossings.data() + base;
      double *p_trapz = sum_trapz.data() + base;
      uint8_t *p_nf = nonfinite.data() + base;

#pragma omp simd
      for (int w = 0; w < W; ++w) {
        const double x = static_cast<double>(xr[w]);
        if (!std::isfinite(x)) {
          p_nf[w] = 1;
        }
        // Pébay / Chan-Golub-LeVeque online central moments (orders 1..4)
        const double mu = p_mean[w];
        const double delta = x - mu;
        const double delta_n = delta * inv_n1;
        const double delta_n2 = delta_n * delta_n;
        const double term1 = delta * delta_n * n0;
        p_mean[w] = mu + delta_n;

        const double m2 = p_M2[w];
        const double m3 = p_M3[w];
        p_M4[w] +=
            term1 * delta_n2 * c4 + 6.0 * delta_n2 * m2 - 4.0 * delta_n * m3;
        p_M3[w] = m3 + term1 * delta_n * c3 - 3.0 * delta_n * m2;
        p_M2[w] = m2 + term1;

        p_min[w] = (x < p_min[w]) ? x : p_min[w];
        p_max[w] = (x > p_max[w]) ? x : p_max[w];
        p_sabs[w] += std::fabs(x);
        p_ssq[w] += x * x;
        p_stx[w] += t_idx * x;

        const double px = p_prev[w];
        p_zc[w] +=
            static_cast<double>(std::signbit(px) != std::signbit(x));
        p_trapz[w] += (px + x) * half_dt;
        p_prev[w] = x;
      }
    }
    n += 1;
  }

  void update(const float *obs_row) { update_typed<float>(obs_row); }
  void update(const double *obs_row) { update_typed<double>(obs_row); }

  static bool is_online_feature(const std::string &name) {
    return name == "mean" || name == "var" || name == "std" ||
           name == "min" || name == "max" || name == "rms" ||
           name == "abs_energy" || name == "average_power" || name == "auc" ||
           name == "auc_trapz" || name == "zero_crossing" ||
           name == "burstiness" || name == "moments" ||
           name == "skewness" || name == "kurtosis" || name == "centroid";
  }

  static int feature_components(const std::string &name, int order = 4) {
    if (name == "moments") {
      if (order < 0 || order > 4) {
        throw std::invalid_argument(
            "OnlineFeatureAccum: moments order must be in [0, 4]");
      }
      return order + 1;
    }
    return 1;
  }

  void finalize_into(const std::string &name, double *out,
                     int order = 4) const {
    const size_t total = num_series();
    const double nan_v = quiet_nan();
    if (name == "moments") {
      if (order < 0 || order > 4) {
        throw std::invalid_argument(
            "OnlineFeatureAccum: moments order must be in [0, 4]");
      }
      const int n_comp = order + 1;
      if (n == 0) {
        std::fill(out, out + static_cast<size_t>(n_comp) * total, nan_v);
        return;
      }
      const double inv_n = 1.0 / static_cast<double>(n);
      for (size_t idx = 0; idx < total; ++idx) {
        if (nonfinite[idx]) {
          for (int k = 0; k < n_comp; ++k) out[static_cast<size_t>(k) * total + idx] = nan_v;
          continue;
        }
        out[idx] = 1.0;
        if (order >= 1) out[total + idx] = 0.0;
        if (order >= 2) out[2 * total + idx] = M2[idx] * inv_n;
        if (order >= 3) out[3 * total + idx] = M3[idx] * inv_n;
        if (order >= 4) out[4 * total + idx] = M4[idx] * inv_n;
      }
      return;
    }

    if (n == 0) {
      std::fill(out, out + total, nan_v);
      return;
    }

    const double inv_n = 1.0 / static_cast<double>(n);
    constexpr double kSciPyRes = 1e-15;  // np.finfo(np.float64).resolution

    if (name == "mean" || name == "auc") {
      for (size_t i = 0; i < total; ++i)
        out[i] = nonfinite[i] ? nan_v : mean[i];
    } else if (name == "var") {
      for (size_t i = 0; i < total; ++i)
        out[i] = nonfinite[i] ? nan_v : (M2[i] * inv_n);
    } else if (name == "std") {
      for (size_t i = 0; i < total; ++i) {
        if (nonfinite[i]) {
          out[i] = nan_v;
        } else {
          const double v = M2[i] * inv_n;
          out[i] = std::sqrt(v > 0.0 ? v : 0.0);
        }
      }
    } else if (name == "min") {
      for (size_t i = 0; i < total; ++i)
        out[i] = nonfinite[i] ? nan_v : min_v[i];
    } else if (name == "max") {
      for (size_t i = 0; i < total; ++i)
        out[i] = nonfinite[i] ? nan_v : max_v[i];
    } else if (name == "rms") {
      for (size_t i = 0; i < total; ++i)
        out[i] = nonfinite[i] ? nan_v : std::sqrt(sum_sq[i] * inv_n);
    } else if (name == "abs_energy") {
      for (size_t i = 0; i < total; ++i)
        out[i] = nonfinite[i] ? nan_v : sum_sq[i];
    } else if (name == "average_power") {
      for (size_t i = 0; i < total; ++i)
        out[i] = nonfinite[i] ? nan_v : (sum_sq[i] * inv_n);
    } else if (name == "auc_trapz") {
      for (size_t i = 0; i < total; ++i)
        out[i] = nonfinite[i] ? nan_v : sum_trapz[i];
    } else if (name == "zero_crossing") {
      for (size_t i = 0; i < total; ++i)
        out[i] = nonfinite[i] ? nan_v : zero_crossings[i];
    } else if (name == "burstiness") {
      for (size_t i = 0; i < total; ++i) {
        if (nonfinite[i]) {
          out[i] = nan_v;
          continue;
        }
        const double mu = mean[i];
        const double v = M2[i] * inv_n;
        const double sig = std::sqrt(v > 0.0 ? v : 0.0);
        const double den = mu + sig;
        out[i] = (den == 0.0) ? 0.0 : (sig - mu) / den;
      }
    } else if (name == "skewness") {
      for (size_t i = 0; i < total; ++i) {
        if (nonfinite[i] || n <= 1) {
          out[i] = nan_v;
          continue;
        }
        const double m2 = M2[i] * inv_n;
        const double m3 = M3[i] * inv_n;
        const double thr = kSciPyRes * mean[i];
        out[i] = (m2 <= thr * thr) ? nan_v : (m3 / (m2 * std::sqrt(m2)));
      }
    } else if (name == "kurtosis") {
      for (size_t i = 0; i < total; ++i) {
        if (nonfinite[i] || n <= 1) {
          out[i] = nan_v;
          continue;
        }
        const double m2 = M2[i] * inv_n;
        const double m4 = M4[i] * inv_n;
        const double thr = kSciPyRes * mean[i];
        out[i] = (m2 <= thr * thr) ? nan_v : (m4 / (m2 * m2) - 3.0);
      }
    } else if (name == "centroid") {
      for (size_t i = 0; i < total; ++i) {
        if (nonfinite[i]) {
          out[i] = nan_v;
          continue;
        }
        const double s = sum_abs[i];
        out[i] = (s == 0.0) ? 0.0 : (sum_t_x[i] / s);
      }
    } else {
      throw std::invalid_argument("OnlineFeatureAccum: unknown feature '" +
                                  name + "'");
    }
  }

  OnlineFeatureResult finalize() const {
    OnlineFeatureResult r;
    r.n_voi = n_voi;
    r.n_node = n_node;
    r.width = W;
    r.n_samples = n;
    const size_t total = num_series();
    r.mean.resize(total);
    r.var.resize(total);
    r.std_dev.resize(total);
    r.min_v.resize(total);
    r.max_v.resize(total);
    r.rms.resize(total);
    r.abs_energy.resize(total);
    r.average_power.resize(total);
    r.auc.resize(total);
    r.auc_trapz.resize(total);
    r.zero_crossing.resize(total);
    r.burstiness.resize(total);
    r.moments.resize(5 * total);
    r.skewness.resize(total);
    r.kurtosis.resize(total);
    r.centroid.resize(total);

    finalize_into("mean", r.mean.data());
    finalize_into("var", r.var.data());
    finalize_into("std", r.std_dev.data());
    finalize_into("min", r.min_v.data());
    finalize_into("max", r.max_v.data());
    finalize_into("rms", r.rms.data());
    finalize_into("abs_energy", r.abs_energy.data());
    finalize_into("average_power", r.average_power.data());
    finalize_into("auc", r.auc.data());
    finalize_into("auc_trapz", r.auc_trapz.data());
    finalize_into("zero_crossing", r.zero_crossing.data());
    finalize_into("burstiness", r.burstiness.data());
    finalize_into("moments", r.moments.data(), 4);
    finalize_into("skewness", r.skewness.data());
    finalize_into("kurtosis", r.kurtosis.data());
    finalize_into("centroid", r.centroid.data());
    return r;
  }
};

// ------------------------------------------------------------------ //
// Tier 2a: Order statistics & binned Shannon entropy
// ------------------------------------------------------------------ //

enum OrderStatMask : uint32_t {
  ORDER_MEDIAN = 1u << 0,
  ORDER_MAD = 1u << 1,
  ORDER_MEDIAN_ABS_DEV = 1u << 2,
  ORDER_IQR = 1u << 3,
  ORDER_ENTROPY = 1u << 4,
  ORDER_ALL = 0x1Fu,
};

struct OrderStatsResult {
  std::vector<double> median;
  std::vector<double> mad;
  std::vector<double> median_abs_dev;
  std::vector<double> iqr;
  std::vector<double> entropy;
};

namespace detail {

inline double sorted_median(const double *sorted, int n) {
  if (n <= 0) return quiet_nan();
  if ((n & 1) != 0) {
    return sorted[n / 2];
  }
  return 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
}

inline double sorted_percentile_linear(const double *sorted, int n, double p) {
  if (n <= 0) return quiet_nan();
  if (n == 1) return sorted[0];
  const double idx = (p / 100.0) * static_cast<double>(n - 1);
  const int lo = static_cast<int>(std::floor(idx));
  const int hi = static_cast<int>(std::ceil(idx));
  const double frac = idx - static_cast<double>(lo);
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

template <typename T>
inline bool copy_series_check_finite(const T *row, int n_time, double *dst) {
  bool ok = true;
  for (int t = 0; t < n_time; ++t) {
    const double v = static_cast<double>(row[t]);
    dst[t] = v;
    if (!std::isfinite(v)) ok = false;
  }
  return ok;
}

}  // namespace detail

template <typename T>
OrderStatsResult compute_order_stats(const T *ts, int n_series, int n_time,
                                     int bins = 20,
                                     uint32_t mask = ORDER_ALL) {
  if (n_series < 0 || n_time < 0) {
    throw std::invalid_argument("compute_order_stats: negative dimensions");
  }
  if ((mask & ORDER_ENTROPY) && bins <= 0) {
    throw std::invalid_argument("compute_order_stats: bins must be > 0");
  }

  OrderStatsResult res;
  const size_t ns = static_cast<size_t>(n_series);
  const double nan_v = quiet_nan();
  if (mask & ORDER_MEDIAN) res.median.assign(ns, nan_v);
  if (mask & ORDER_MAD) res.mad.assign(ns, nan_v);
  if (mask & ORDER_MEDIAN_ABS_DEV) res.median_abs_dev.assign(ns, nan_v);
  if (mask & ORDER_IQR) res.iqr.assign(ns, nan_v);
  if (mask & ORDER_ENTROPY) res.entropy.assign(ns, nan_v);

  if (n_series == 0 || n_time == 0) {
    return res;
  }

  const bool need_sort =
      (mask & (ORDER_MEDIAN | ORDER_MEDIAN_ABS_DEV | ORDER_IQR)) != 0;
  std::vector<double> work(static_cast<size_t>(n_time));
  std::vector<double> dev(static_cast<size_t>(n_time));
  std::vector<int> hist(
      (mask & ORDER_ENTROPY) ? static_cast<size_t>(bins) : 0, 0);

  for (int s = 0; s < n_series; ++s) {
    const T *row = ts + static_cast<size_t>(s) * static_cast<size_t>(n_time);
    if (!detail::copy_series_check_finite(row, n_time, work.data())) {
      continue;  // entries remain NaN
    }

    if (mask & ORDER_MAD) {
      double sum = 0.0;
      for (int t = 0; t < n_time; ++t) sum += work[t];
      const double mu = sum / static_cast<double>(n_time);
      double abs_sum = 0.0;
      for (int t = 0; t < n_time; ++t) abs_sum += std::fabs(work[t] - mu);
      res.mad[static_cast<size_t>(s)] = abs_sum / static_cast<double>(n_time);
    }

    if (mask & ORDER_ENTROPY) {
      double lo = work[0];
      double hi = work[0];
      for (int t = 1; t < n_time; ++t) {
        if (work[t] < lo) lo = work[t];
        if (work[t] > hi) hi = work[t];
      }
      double rng = hi - lo;
      if (rng == 0.0) rng = 1.0;
      std::fill(hist.begin(), hist.end(), 0);
      for (int t = 0; t < n_time; ++t) {
        int b = static_cast<int>((work[t] - lo) / rng *
                                 static_cast<double>(bins));
        if (b < 0) b = 0;
        if (b >= bins) b = bins - 1;
        hist[static_cast<size_t>(b)] += 1;
      }
      double H = 0.0;
      const double inv_n = 1.0 / static_cast<double>(n_time);
      for (int b = 0; b < bins; ++b) {
        const int c = hist[static_cast<size_t>(b)];
        if (c > 0) {
          const double p = static_cast<double>(c) * inv_n;
          H -= p * std::log(p + 1e-15);
        }
      }
      res.entropy[static_cast<size_t>(s)] = H;
    }

    if (need_sort) {
      std::sort(work.begin(), work.end());
      const double med = detail::sorted_median(work.data(), n_time);
      if (mask & ORDER_MEDIAN) {
        res.median[static_cast<size_t>(s)] = med;
      }
      if (mask & ORDER_IQR) {
        const double q75 =
            detail::sorted_percentile_linear(work.data(), n_time, 75.0);
        const double q25 =
            detail::sorted_percentile_linear(work.data(), n_time, 25.0);
        res.iqr[static_cast<size_t>(s)] = q75 - q25;
      }
      if (mask & ORDER_MEDIAN_ABS_DEV) {
        for (int t = 0; t < n_time; ++t) {
          dev[static_cast<size_t>(t)] = std::fabs(work[static_cast<size_t>(t)] - med);
        }
        std::sort(dev.begin(), dev.end());
        res.median_abs_dev[static_cast<size_t>(s)] =
            detail::sorted_median(dev.data(), n_time);
      }
    }
  }
  return res;
}

// ------------------------------------------------------------------ //
// Tier 2b: Self-contained Cooley-Tukey radix-2 FFT + Welch PSD
// ------------------------------------------------------------------ //

namespace detail {

// Reusable radix-2 Cooley-Tukey FFT plan with precomputed bit-reversal and
// twiddle factors (power-of-two length N >= 1).
struct Radix2FftPlan {
  int N = 0;
  std::vector<int> bitrev;
  std::vector<double> tw_re;
  std::vector<double> tw_im;

  void init(int n_pow2) {
    if (n_pow2 <= 0 || (n_pow2 & (n_pow2 - 1)) != 0) {
      throw std::invalid_argument("Radix2FftPlan: N must be a power of 2");
    }
    N = n_pow2;
    bitrev.resize(static_cast<size_t>(N));
    int log_n = 0;
    while ((1 << log_n) < N) ++log_n;
    bitrev[0] = 0;
    for (int i = 1; i < N; ++i) {
      bitrev[static_cast<size_t>(i)] =
          (bitrev[static_cast<size_t>(i >> 1)] >> 1) |
          ((i & 1) << (log_n - 1));
    }
    tw_re.resize(static_cast<size_t>(N / 2));
    tw_im.resize(static_cast<size_t>(N / 2));
    for (int k = 0; k < N / 2; ++k) {
      const double ang =
          -2.0 * CPH_M_PI * static_cast<double>(k) / static_cast<double>(N);
      tw_re[static_cast<size_t>(k)] = std::cos(ang);
      tw_im[static_cast<size_t>(k)] = std::sin(ang);
    }
  }

  void transform(double *re, double *im, bool inverse = false) const {
    if (N <= 1) return;
    for (int i = 0; i < N; ++i) {
      const int j = bitrev[static_cast<size_t>(i)];
      if (i < j) {
        std::swap(re[i], re[j]);
        std::swap(im[i], im[j]);
      }
    }
    const double sign = inverse ? -1.0 : 1.0;
    for (int len = 2; len <= N; len <<= 1) {
      const int half = len >> 1;
      const int step = N / len;
      for (int i = 0; i < N; i += len) {
        for (int j = 0; j < half; ++j) {
          const size_t tw_idx = static_cast<size_t>(j * step);
          const double wr = tw_re[tw_idx];
          const double wi = sign * tw_im[tw_idx];
          const int i0 = i + j;
          const int i1 = i0 + half;
          const double ur = re[i0];
          const double ui = im[i0];
          const double vr = re[i1] * wr - im[i1] * wi;
          const double vi = re[i1] * wi + im[i1] * wr;
          re[i0] = ur + vr;
          im[i0] = ui + vi;
          re[i1] = ur - vr;
          im[i1] = ui - vi;
        }
      }
    }
    if (inverse) {
      const double inv_n = 1.0 / static_cast<double>(N);
      for (int i = 0; i < N; ++i) {
        re[i] *= inv_n;
        im[i] *= inv_n;
      }
    }
  }
};

// Arbitrary-length real-to-complex DFT using direct radix-2 FFT when M is a
// power of 2, or Bluestein chirp-z convolution via radix-2 FFT otherwise.
struct ArbitraryRealDftPlan {
  int M = 0;
  int L = 0;
  bool is_pow2 = false;
  Radix2FftPlan fft;
  std::vector<double> chirp_re;
  std::vector<double> chirp_im;
  std::vector<double> B_re;
  std::vector<double> B_im;
  std::vector<double> buf_re;
  std::vector<double> buf_im;

  void init(int len) {
    if (len <= 0) {
      throw std::invalid_argument("ArbitraryRealDftPlan: len must be > 0");
    }
    M = len;
    is_pow2 = (M & (M - 1)) == 0;
    if (is_pow2) {
      L = M;
      fft.init(L);
      buf_re.resize(static_cast<size_t>(L));
      buf_im.resize(static_cast<size_t>(L));
      return;
    }
    L = 1;
    while (L < 2 * M - 1) L <<= 1;
    fft.init(L);
    chirp_re.resize(static_cast<size_t>(M));
    chirp_im.resize(static_cast<size_t>(M));
    const uint64_t two_m = 2ull * static_cast<uint64_t>(M);
    for (int k = 0; k < M; ++k) {
      const uint64_t kk =
          (static_cast<uint64_t>(k) * static_cast<uint64_t>(k)) % two_m;
      const double ang =
          CPH_M_PI * static_cast<double>(kk) / static_cast<double>(M);
      chirp_re[static_cast<size_t>(k)] = std::cos(ang);
      chirp_im[static_cast<size_t>(k)] = -std::sin(ang);
    }
    B_re.assign(static_cast<size_t>(L), 0.0);
    B_im.assign(static_cast<size_t>(L), 0.0);
    B_re[0] = 1.0;
    for (int k = 1; k < M; ++k) {
      const double br = chirp_re[static_cast<size_t>(k)];
      const double bi = -chirp_im[static_cast<size_t>(k)];
      B_re[static_cast<size_t>(k)] = br;
      B_im[static_cast<size_t>(k)] = bi;
      B_re[static_cast<size_t>(L - k)] = br;
      B_im[static_cast<size_t>(L - k)] = bi;
    }
    fft.transform(B_re.data(), B_im.data(), false);
    buf_re.resize(static_cast<size_t>(L));
    buf_im.resize(static_cast<size_t>(L));
  }

  // Computes X[0 .. M/2] for real input `in` of length M.
  void rfft(const double *in, double *out_re, double *out_im) {
    const int n_freq = M / 2 + 1;
    if (is_pow2) {
      for (int i = 0; i < M; ++i) {
        buf_re[static_cast<size_t>(i)] = in[i];
        buf_im[static_cast<size_t>(i)] = 0.0;
      }
      fft.transform(buf_re.data(), buf_im.data(), false);
      for (int k = 0; k < n_freq; ++k) {
        out_re[k] = buf_re[static_cast<size_t>(k)];
        out_im[k] = buf_im[static_cast<size_t>(k)];
      }
      return;
    }
    for (int i = 0; i < M; ++i) {
      buf_re[static_cast<size_t>(i)] = in[i] * chirp_re[static_cast<size_t>(i)];
      buf_im[static_cast<size_t>(i)] = in[i] * chirp_im[static_cast<size_t>(i)];
    }
    std::fill(buf_re.begin() + M, buf_re.end(), 0.0);
    std::fill(buf_im.begin() + M, buf_im.end(), 0.0);
    fft.transform(buf_re.data(), buf_im.data(), false);
    for (int i = 0; i < L; ++i) {
      const double ar = buf_re[static_cast<size_t>(i)];
      const double ai = buf_im[static_cast<size_t>(i)];
      const double br = B_re[static_cast<size_t>(i)];
      const double bi = B_im[static_cast<size_t>(i)];
      buf_re[static_cast<size_t>(i)] = ar * br - ai * bi;
      buf_im[static_cast<size_t>(i)] = ar * bi + ai * br;
    }
    fft.transform(buf_re.data(), buf_im.data(), true);
    for (int k = 0; k < n_freq; ++k) {
      const double cr = buf_re[static_cast<size_t>(k)];
      const double ci = buf_im[static_cast<size_t>(k)];
      const double wr = chirp_re[static_cast<size_t>(k)];
      const double wi = chirp_im[static_cast<size_t>(k)];
      out_re[k] = cr * wr - ci * wi;
      out_im[k] = cr * wi + ci * wr;
    }
  }
};

struct WelchPlan {
  int nperseg = 0;
  int noverlap = 0;
  int step = 0;
  int n_freq = 0;
  double fs = 1.0;
  double scale = 1.0;
  std::vector<double> freqs;
  std::vector<double> win;
  std::vector<double> seg_buf;
  std::vector<double> spec_re;
  std::vector<double> spec_im;
  ArbitraryRealDftPlan dft;

  void init(int n_time, double fs_) {
    if (n_time <= 0) {
      throw std::invalid_argument("WelchPlan: n_time must be > 0");
    }
    if (!(fs_ > 0.0) || !std::isfinite(fs_)) {
      throw std::invalid_argument("WelchPlan: fs must be finite and > 0");
    }
    fs = fs_;
    nperseg = (n_time < 256) ? n_time : 256;
    noverlap = nperseg / 2;
    step = nperseg - noverlap;
    if (step < 1) step = 1;
    n_freq = nperseg / 2 + 1;

    freqs.resize(static_cast<size_t>(n_freq));
    const double df = fs / static_cast<double>(nperseg);
    for (int k = 0; k < n_freq; ++k) {
      freqs[static_cast<size_t>(k)] = static_cast<double>(k) * df;
    }

    win.resize(static_cast<size_t>(nperseg));
    if (nperseg == 1) {
      win[0] = 1.0;
    } else {
      for (int j = 0; j < nperseg; ++j) {
        win[static_cast<size_t>(j)] =
            0.5 - 0.5 * std::cos(2.0 * CPH_M_PI * static_cast<double>(j) /
                                 static_cast<double>(nperseg));
      }
    }
    double sum_w2 = 0.0;
    for (int j = 0; j < nperseg; ++j) {
      sum_w2 += win[static_cast<size_t>(j)] * win[static_cast<size_t>(j)];
    }
    scale = 1.0 / (fs * sum_w2);

    seg_buf.resize(static_cast<size_t>(nperseg));
    spec_re.resize(static_cast<size_t>(n_freq));
    spec_im.resize(static_cast<size_t>(n_freq));
    dft.init(nperseg);
  }

  // Computes Welch PSD into `psd_out[0..n_freq-1]` for one finite series of
  // length `n_time`.
  void compute_series(const double *series, int n_time, double *psd_out) {
    std::fill(psd_out, psd_out + n_freq, 0.0);
    int n_seg = 0;
    for (int start = 0; start + nperseg <= n_time; start += step) {
      for (int j = 0; j < nperseg; ++j) {
        seg_buf[static_cast<size_t>(j)] =
            series[start + j] * win[static_cast<size_t>(j)];
      }
      dft.rfft(seg_buf.data(), spec_re.data(), spec_im.data());

      psd_out[0] +=
          (spec_re[0] * spec_re[0] + spec_im[0] * spec_im[0]) * scale;
      const int last_two_sided =
          ((nperseg & 1) == 0) ? (n_freq - 1) : n_freq;
      for (int k = 1; k < last_two_sided; ++k) {
        const double re = spec_re[static_cast<size_t>(k)];
        const double im = spec_im[static_cast<size_t>(k)];
        psd_out[k] += (re * re + im * im) * (2.0 * scale);
      }
      if ((nperseg & 1) == 0 && n_freq > 1) {
        const int last = n_freq - 1;
        const double re = spec_re[static_cast<size_t>(last)];
        const double im = spec_im[static_cast<size_t>(last)];
        psd_out[last] += (re * re + im * im) * scale;
      }
      ++n_seg;
    }
    if (n_seg > 1) {
      const double inv_seg = 1.0 / static_cast<double>(n_seg);
      for (int k = 0; k < n_freq; ++k) {
        psd_out[k] *= inv_seg;
      }
    }
  }
};

}  // namespace detail

struct WelchPsdResult {
  int n_freq = 0;
  std::vector<double> freqs;  // length n_freq
  std::vector<double> psd;    // (n_series, n_freq) row-major
};

template <typename T>
WelchPsdResult compute_welch_psd(const T *ts, int n_series, int n_time,
                                 double fs = 1.0) {
  if (n_series < 0 || n_time < 0) {
    throw std::invalid_argument("compute_welch_psd: negative dimensions");
  }
  WelchPsdResult res;
  if (n_series == 0 || n_time == 0) {
    return res;
  }
  detail::WelchPlan plan;
  plan.init(n_time, fs);
  res.n_freq = plan.n_freq;
  res.freqs = plan.freqs;
  res.psd.assign(static_cast<size_t>(n_series) *
                     static_cast<size_t>(plan.n_freq),
                 quiet_nan());

  std::vector<double> work(static_cast<size_t>(n_time));
  for (int s = 0; s < n_series; ++s) {
    const T *row = ts + static_cast<size_t>(s) * static_cast<size_t>(n_time);
    if (!detail::copy_series_check_finite(row, n_time, work.data())) {
      continue;  // row stays NaN
    }
    plan.compute_series(
        work.data(), n_time,
        res.psd.data() + static_cast<size_t>(s) * static_cast<size_t>(plan.n_freq));
  }
  return res;
}

// ------------------------------------------------------------------ //
// Unified batch entry point (Tier 1 + Tier 2 in minimal passes)
// ------------------------------------------------------------------ //

namespace detail {

inline bool is_welch_feature(const std::string &name) {
  return name == "psd_raw" || name == "spectrum_stats" ||
         name == "spectrum_auc" || name == "spectrum_moments";
}

inline bool is_order_feature(const std::string &name) {
  return name == "median" || name == "mad" || name == "median_abs_dev" ||
         name == "iqr" || name == "entropy";
}

}  // namespace detail

template <typename T>
std::vector<BatchFeatureOutput> compute_batch_features(
    const T *ts, int n_series, int n_time,
    const std::vector<std::string> &names,
    const FeatureParams &params = FeatureParams{}) {
  if (n_series < 0 || n_time < 0) {
    throw std::invalid_argument("compute_batch_features: negative dimensions");
  }

  bool need_online = false;
  uint32_t order_mask = 0;
  bool need_welch = false;
  for (const auto &name : names) {
    if (OnlineFeatureAccum<1>::is_online_feature(name)) {
      need_online = true;
    } else if (name == "median") {
      order_mask |= ORDER_MEDIAN;
    } else if (name == "mad") {
      order_mask |= ORDER_MAD;
    } else if (name == "median_abs_dev") {
      order_mask |= ORDER_MEDIAN_ABS_DEV;
    } else if (name == "iqr") {
      order_mask |= ORDER_IQR;
    } else if (name == "entropy") {
      order_mask |= ORDER_ENTROPY;
    } else if (detail::is_welch_feature(name)) {
      need_welch = true;
    } else {
      throw std::invalid_argument("compute_batch_features: unknown feature '" +
                                  name + "'");
    }
  }

  const size_t ns = static_cast<size_t>(n_series);
  const double nan_v = quiet_nan();

  // Validate params early
  if (params.order < 0) {
    throw std::invalid_argument("compute_batch_features: order must be >= 0");
  }
  for (const auto &name : names) {
    if (name == "moments" && params.order > 4) {
      throw std::invalid_argument(
          "compute_batch_features: moments order must be <= 4");
    }
  }

  // 1. Tier-1 single pass per series (no time-series copy required beyond scalar loop)
  std::vector<double> s_mean, s_M2, s_M3, s_M4, s_min, s_max, s_abs, s_sq,
      s_tx, s_zc, s_trapz;
  std::vector<uint8_t> s_nf;
  if (need_online && n_series > 0 && n_time > 0) {
    s_mean.assign(ns, 0.0);
    s_M2.assign(ns, 0.0);
    s_M3.assign(ns, 0.0);
    s_M4.assign(ns, 0.0);
    s_min.assign(ns, 0.0);
    s_max.assign(ns, 0.0);
    s_abs.assign(ns, 0.0);
    s_sq.assign(ns, 0.0);
    s_tx.assign(ns, 0.0);
    s_zc.assign(ns, 0.0);
    s_trapz.assign(ns, 0.0);
    s_nf.assign(ns, 0);

    const double dt_samp = (params.fs > 0.0) ? (1.0 / params.fs) : 1.0;
    const double half_dt = 0.5 * dt_samp;

    for (int s = 0; s < n_series; ++s) {
      const size_t si = static_cast<size_t>(s);
      const T *row = ts + si * static_cast<size_t>(n_time);
      const double x0 = static_cast<double>(row[0]);
      uint8_t nf = static_cast<uint8_t>(!std::isfinite(x0));
      double mu = x0;
      double m2 = 0.0, m3 = 0.0, m4 = 0.0;
      double mn = x0, mx = x0;
      double sab = std::fabs(x0);
      double ssq = x0 * x0;
      double stx = 0.0;
      double zc = 0.0;
      double trapz = 0.0;
      double px = x0;

      for (int t = 1; t < n_time; ++t) {
        const double x = static_cast<double>(row[t]);
        if (!std::isfinite(x)) nf = 1;
        const double n0 = static_cast<double>(t);
        const double n1 = static_cast<double>(t + 1);
        const double delta = x - mu;
        const double delta_n = delta / n1;
        const double delta_n2 = delta_n * delta_n;
        const double term1 = delta * delta_n * n0;
        mu += delta_n;
        m4 += term1 * delta_n2 * (n1 * n1 - 3.0 * n1 + 3.0) +
              6.0 * delta_n2 * m2 - 4.0 * delta_n * m3;
        m3 += term1 * delta_n * (n1 - 2.0) - 3.0 * delta_n * m2;
        m2 += term1;
        if (x < mn) mn = x;
        if (x > mx) mx = x;
        sab += std::fabs(x);
        ssq += x * x;
        stx += static_cast<double>(t) * x;
        if (std::signbit(px) != std::signbit(x)) zc += 1.0;
        trapz += (px + x) * half_dt;
        px = x;
      }

      s_nf[si] = nf;
      s_mean[si] = mu;
      s_M2[si] = m2;
      s_M3[si] = m3;
      s_M4[si] = m4;
      s_min[si] = mn;
      s_max[si] = mx;
      s_abs[si] = sab;
      s_sq[si] = ssq;
      s_tx[si] = stx;
      s_zc[si] = zc;
      s_trapz[si] = trapz;
    }
  }

  // 2. Tier-2a order stats & entropy (one pass over requested subset)
  OrderStatsResult ord_res;
  if (order_mask != 0) {
    ord_res = compute_order_stats(ts, n_series, n_time, params.bins, order_mask);
  }

  // 3. Tier-2b single Welch PSD pass for ALL spectral features
  WelchPsdResult welch_res;
  if (need_welch) {
    welch_res = compute_welch_psd(ts, n_series, n_time, params.fs);
  }

  // 4. Materialize requested outputs in caller's exact name order
  std::vector<BatchFeatureOutput> outs;
  outs.reserve(names.size());
  const double inv_n = (n_time > 0) ? (1.0 / static_cast<double>(n_time)) : 0.0;
  constexpr double kSciPyRes = 1e-15;

  for (const auto &name : names) {
    BatchFeatureOutput out;
    out.name = name;

    if (name == "moments") {
      const int n_comp = params.order + 1;
      out.n_components = n_comp;
      out.data.assign(static_cast<size_t>(n_comp) * ns, nan_v);
      if (n_time > 0) {
        for (size_t i = 0; i < ns; ++i) {
          if (s_nf[i]) continue;
          out.data[i] = 1.0;
          if (params.order >= 1) out.data[ns + i] = 0.0;
          if (params.order >= 2) out.data[2 * ns + i] = s_M2[i] * inv_n;
          if (params.order >= 3) out.data[3 * ns + i] = s_M3[i] * inv_n;
          if (params.order >= 4) out.data[4 * ns + i] = s_M4[i] * inv_n;
        }
      }
    } else if (OnlineFeatureAccum<1>::is_online_feature(name)) {
      out.n_components = 1;
      out.data.assign(ns, nan_v);
      if (n_time > 0) {
        if (name == "mean" || name == "auc") {
          for (size_t i = 0; i < ns; ++i)
            if (!s_nf[i]) out.data[i] = s_mean[i];
        } else if (name == "var") {
          for (size_t i = 0; i < ns; ++i)
            if (!s_nf[i]) out.data[i] = s_M2[i] * inv_n;
        } else if (name == "std") {
          for (size_t i = 0; i < ns; ++i) {
            if (!s_nf[i]) {
              const double v = s_M2[i] * inv_n;
              out.data[i] = std::sqrt(v > 0.0 ? v : 0.0);
            }
          }
        } else if (name == "min") {
          for (size_t i = 0; i < ns; ++i)
            if (!s_nf[i]) out.data[i] = s_min[i];
        } else if (name == "max") {
          for (size_t i = 0; i < ns; ++i)
            if (!s_nf[i]) out.data[i] = s_max[i];
        } else if (name == "rms") {
          for (size_t i = 0; i < ns; ++i)
            if (!s_nf[i]) out.data[i] = std::sqrt(s_sq[i] * inv_n);
        } else if (name == "abs_energy") {
          for (size_t i = 0; i < ns; ++i)
            if (!s_nf[i]) out.data[i] = s_sq[i];
        } else if (name == "average_power") {
          for (size_t i = 0; i < ns; ++i)
            if (!s_nf[i]) out.data[i] = s_sq[i] * inv_n;
        } else if (name == "auc_trapz") {
          for (size_t i = 0; i < ns; ++i)
            if (!s_nf[i]) out.data[i] = s_trapz[i];
        } else if (name == "zero_crossing") {
          for (size_t i = 0; i < ns; ++i)
            if (!s_nf[i]) out.data[i] = s_zc[i];
        } else if (name == "burstiness") {
          for (size_t i = 0; i < ns; ++i) {
            if (!s_nf[i]) {
              const double mu = s_mean[i];
              const double v = s_M2[i] * inv_n;
              const double sig = std::sqrt(v > 0.0 ? v : 0.0);
              const double den = mu + sig;
              out.data[i] = (den == 0.0) ? 0.0 : (sig - mu) / den;
            }
          }
        } else if (name == "skewness") {
          if (n_time > 1) {
            for (size_t i = 0; i < ns; ++i) {
              if (!s_nf[i]) {
                const double m2 = s_M2[i] * inv_n;
                const double m3 = s_M3[i] * inv_n;
                const double thr = kSciPyRes * s_mean[i];
                if (m2 > thr * thr) {
                  out.data[i] = m3 / (m2 * std::sqrt(m2));
                }
              }
            }
          }
        } else if (name == "kurtosis") {
          if (n_time > 1) {
            for (size_t i = 0; i < ns; ++i) {
              if (!s_nf[i]) {
                const double m2 = s_M2[i] * inv_n;
                const double m4 = s_M4[i] * inv_n;
                const double thr = kSciPyRes * s_mean[i];
                if (m2 > thr * thr) {
                  out.data[i] = m4 / (m2 * m2) - 3.0;
                }
              }
            }
          }
        } else if (name == "centroid") {
          for (size_t i = 0; i < ns; ++i) {
            if (!s_nf[i]) {
              const double sabs = s_abs[i];
              out.data[i] = (sabs == 0.0) ? 0.0 : (s_tx[i] / sabs);
            }
          }
        }
      }
    } else if (name == "median") {
      out.data = ord_res.median;
    } else if (name == "mad") {
      out.data = ord_res.mad;
    } else if (name == "median_abs_dev") {
      out.data = ord_res.median_abs_dev;
    } else if (name == "iqr") {
      out.data = ord_res.iqr;
    } else if (name == "entropy") {
      out.data = ord_res.entropy;
    } else if (name == "psd_raw") {
      out.n_components = 1;
      out.n_freq = welch_res.n_freq;
      out.freqs = welch_res.freqs;
      out.data = welch_res.psd;
    } else if (name == "spectrum_stats") {
      out.n_components = 3;
      out.data.assign(3 * ns, nan_v);
      const int nf = welch_res.n_freq;
      if (nf > 0) {
        for (size_t s = 0; s < ns; ++s) {
          const double *p = welch_res.psd.data() + s * static_cast<size_t>(nf);
          if (!std::isfinite(p[0])) continue;
          double total = 0.0;
          double fsum = 0.0;
          int peak_k = 0;
          double peak_v = p[0];
          for (int k = 0; k < nf; ++k) {
            const double pk = p[k];
            total += pk;
            fsum += welch_res.freqs[static_cast<size_t>(k)] * pk;
            if (pk > peak_v) {
              peak_v = pk;
              peak_k = k;
            }
          }
          const double denom = (total == 0.0) ? 1.0 : total;
          out.data[s] = fsum / denom;
          out.data[ns + s] = welch_res.freqs[static_cast<size_t>(peak_k)];
          out.data[2 * ns + s] = total;
        }
      }
    } else if (name == "spectrum_auc") {
      out.n_components = 1;
      out.data.assign(ns, nan_v);
      const int nf = welch_res.n_freq;
      if (nf > 0) {
        for (size_t s = 0; s < ns; ++s) {
          const double *p = welch_res.psd.data() + s * static_cast<size_t>(nf);
          if (!std::isfinite(p[0])) continue;
          double total = 0.0;
          if (params.has_band) {
            for (int k = 0; k < nf; ++k) {
              const double fk = welch_res.freqs[static_cast<size_t>(k)];
              if (fk >= params.lo && fk <= params.hi) total += p[k];
            }
          } else {
            for (int k = 0; k < nf; ++k) total += p[k];
          }
          out.data[s] = total;
        }
      }
    } else if (name == "spectrum_moments") {
      const int n_comp = params.order + 1;
      out.n_components = n_comp;
      out.data.assign(static_cast<size_t>(n_comp) * ns, nan_v);
      const int nf = welch_res.n_freq;
      if (nf > 0) {
        std::vector<double> fpow(static_cast<size_t>(nf), 1.0);
        for (size_t s = 0; s < ns; ++s) {
          const double *p = welch_res.psd.data() + s * static_cast<size_t>(nf);
          if (!std::isfinite(p[0])) continue;
          double total = 0.0;
          for (int k = 0; k < nf; ++k) total += p[k];
          out.data[s] = total;
          const double denom = (total == 0.0) ? 1.0 : total;
          std::fill(fpow.begin(), fpow.end(), 1.0);
          for (int m = 1; m <= params.order; ++m) {
            double acc = 0.0;
            for (int k = 0; k < nf; ++k) {
              fpow[static_cast<size_t>(k)] *=
                  welch_res.freqs[static_cast<size_t>(k)];
              acc += fpow[static_cast<size_t>(k)] * p[k];
            }
            out.data[static_cast<size_t>(m) * ns + s] = acc / denom;
          }
        }
      }
    }

    outs.push_back(std::move(out));
  }

  return outs;
}

inline std::vector<BatchFeatureOutput> compute_batch_features(
    const float *ts, int n_series, int n_time, double fs,
    const std::vector<std::string> &names) {
  FeatureParams p;
  p.fs = fs;
  return compute_batch_features<float>(ts, n_series, n_time, names, p);
}

inline std::vector<BatchFeatureOutput> compute_batch_features(
    const double *ts, int n_series, int n_time, double fs,
    const std::vector<std::string> &names) {
  FeatureParams p;
  p.fs = fs;
  return compute_batch_features<double>(ts, n_series, n_time, names, p);
}

}  // namespace cph
