// Vectorized counter-based SIMD PRNG (Philox4x32-10) & noise generator
// for TVB cpp_hybrid.
//
// Pure C++17 header: no nanobind / Python headers; compiles cleanly under
// g++ 13, Clang, and Emscripten em++ 4.0.9 (-msimd128), templated over
// SIMD lane width W in {1, 4, 8, 16}.
//
// ============================================================================
// Wave 2 Integration Contract (_core.cpp & backend.py)
// ============================================================================
//
// Include:
//   #include "cph_rng.hpp"
//
// 1. Low-level Philox4x32-10 primitives (Salmon et al. 2011):
//      cph::philox4x32_10(std::array<uint32_t, 4> ctr,
//                         std::array<uint32_t, 2> key) -> std::array<uint32_t, 4>
//      cph::u32_to_uniform01_f(uint32_t x) -> float  in (0, 1]
//      cph::u32_to_uniform01_d(uint32_t x) -> double in (0, 1]
//      cph::u64_to_uniform01_d(uint64_t x) -> double in (0, 1]
//      cph::box_muller_pair(uint32_t a, uint32_t b, float &z0, float &z1)
//      cph::philox4x32_normals(ctr, key) -> std::array<float, 4>  // 2 N(0,1) pairs
//
// 2. Serializable state struct `cph::rng_state<W>`:
//      std::array<uint64_t, W> lane_keys{};
//      uint64_t step_counter = 0;
//      std::vector<float> ou_state;
//
// 3. Struct `cph::rng_engine<W>`:
//      std::array<uint64_t, W> lane_keys{};  // 64-bit seed/key per SIMD lane
//      uint64_t step_counter = 0;            // monotonic step counter
//      uint32_t subnet_id = 0;               // subnetwork id in counter
//      bool enabled = false;
//      std::vector<float> gfun;              // empty, (n_svar,), (n_svar, n_node, W),
//                                            // or (n_svar, n_node, n_modes, W);
//                                            // stores sqrt(2 * nsig * dt) for Additive
//      bool is_multiplicative = false;       // optional Linear(mul_a, mul_b) * x_state
//      float mul_a = 1.0f;
//      float mul_b = 0.0f;
//      std::vector<float> ou_state;          // (n_svar, n_node, n_modes, W) if ntau > 0
//      float ntau = 0.0f;                    // colored OU correlation time (ms); 0 = white
//      float dt = 0.001f;                    // subnet integration step dt (ms)
//
//    Key methods:
//      void set_uniform_seed(uint64_t seed);
//      void set_lane_seeds(const uint64_t *seeds, size_t count);
//      void fill_step_noise(float *out, int n_svar, int n_node, int n_modes,
//                           uint64_t step_idx, const float *x_state = nullptr);
//      void fill_next_step_noise(float *out, int n_svar, int n_node, int n_modes,
//                                const float *x_state = nullptr);
//      rng_state<W> export_state() const;
//      void import_state(const rng_state<W> &state);
//
//    Determinism & Invariance Guarantee:
//      `fill_step_noise` computes raw N(0, 1) draws as a pure function of
//      `(lane_keys[w], subnet_id, step_idx, flat_elem_idx)` using Philox counter
//      `{lo32(step_idx), hi32(step_idx), elem_idx / 4, subnet_id}` and taking
//      component `elem_idx % 4`. Thus every lane `w` is 100% bit-for-bit
//      invariant to `chunk_size`, thread scheduling, and SIMD width `W`.
//
//    Colored OU Noise (`ntau > 0.0f`):
//      Uses TVB's Fox-Vemuri exact OU discretization (`tvb/simulator/noise.py`):
//        E = exp(-dt / ntau)
//        eta = eta * E + sqrt(1 - E^2) * z
//        noise = gfun * sqrt(dt / ntau) * eta
//      where `gfun = sqrt(2 * nsig * dt)` so `gfun * sqrt(dt / ntau) * eta`
//      equals `sqrt(2 * nsig / ntau) * dt * eta`, matching `Noise.coloured`
//      (for dt << ntau, equivalent to `d_eta = (-eta/ntau)*dt + sqrt(2*nsig/ntau)*dW`).
//      When `ou_state` is empty on first colored step, it is deterministically
//      initialized from N(0, 1) using reserved counter `step_idx = UINT64_MAX`.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cph {

// Salmon et al. (2011) Philox4x32-10 constants.
inline constexpr uint32_t PHILOX_W32_0   = 0x9E3779B9u;
inline constexpr uint32_t PHILOX_W32_1   = 0xBB67AE85u;
inline constexpr uint32_t PHILOX_M4x32_0 = 0xD2511F53u;
inline constexpr uint32_t PHILOX_M4x32_1 = 0xCD9E8D57u;

inline std::array<uint32_t, 4> philox4x32_10(std::array<uint32_t, 4> ctr,
                                             std::array<uint32_t, 2> key) {
  for (int round = 0; round < 10; round++) {
    const uint64_t p0 =
        static_cast<uint64_t>(PHILOX_M4x32_0) * static_cast<uint64_t>(ctr[0]);
    const uint64_t p1 =
        static_cast<uint64_t>(PHILOX_M4x32_1) * static_cast<uint64_t>(ctr[2]);
    const uint32_t hi0 = static_cast<uint32_t>(p0 >> 32);
    const uint32_t lo0 = static_cast<uint32_t>(p0);
    const uint32_t hi1 = static_cast<uint32_t>(p1 >> 32);
    const uint32_t lo1 = static_cast<uint32_t>(p1);

    ctr[0] = hi1 ^ ctr[1] ^ key[0];
    ctr[1] = lo1;
    ctr[2] = hi0 ^ ctr[3] ^ key[1];
    ctr[3] = lo0;

    if (round < 9) {
      key[0] += PHILOX_W32_0;
      key[1] += PHILOX_W32_1;
    }
  }
  return ctr;
}

// Convert uint32_t -> open-interval (0, 1] float32 (uniform spacing 2^-24).
inline float u32_to_uniform01_f(uint32_t x) {
  return static_cast<float>((x >> 8) + 1u) * (1.0f / 16777216.0f);
}

// Convert uint32_t -> open-interval (0, 1] float64 (uniform spacing 2^-32).
inline double u32_to_uniform01_d(uint32_t x) {
  return (static_cast<double>(x) + 1.0) * (1.0 / 4294967296.0);
}

// Convert uint64_t -> open-interval (0, 1] float64 (uniform spacing 2^-53).
inline double u64_to_uniform01_d(uint64_t x) {
  return (static_cast<double>(x >> 11) + 1.0) * (1.0 / 9007199254740992.0);
}

// Box-Muller transform mapping two uniform uint32_t words to a pair of
// independent standard normal N(0, 1) float32 values.
inline void box_muller_pair(uint32_t a, uint32_t b, float &z0, float &z1) {
  const double u1 = u32_to_uniform01_d(a);
  const double u2 = u32_to_uniform01_d(b);
  const double r = std::sqrt(-2.0 * std::log(u1));
  const double theta = 6.283185307179586476925286766559 * u2;
  z0 = static_cast<float>(r * std::cos(theta));
  z1 = static_cast<float>(r * std::sin(theta));
}

// Generate 4 independent N(0, 1) float32 draws from one Philox4x32-10 call.
inline std::array<float, 4> philox4x32_normals(std::array<uint32_t, 4> ctr,
                                               std::array<uint32_t, 2> key) {
  const auto r = philox4x32_10(ctr, key);
  std::array<float, 4> z{};
  box_muller_pair(r[0], r[1], z[0], z[1]);
  box_muller_pair(r[2], r[3], z[2], z[3]);
  return z;
}

template <int W>
struct rng_state {
  static_assert(W == 1 || W == 4 || W == 8 || W == 16,
                "rng_state<W> supports W in {1, 4, 8, 16}");
  std::array<uint64_t, W> lane_keys{};
  uint64_t step_counter = 0;
  std::vector<float> ou_state;
};

template <int W>
struct rng_engine {
  static_assert(W == 1 || W == 4 || W == 8 || W == 16,
                "rng_engine<W> supports W in {1, 4, 8, 16}");

  std::array<uint64_t, W> lane_keys{};
  uint64_t step_counter = 0;
  uint32_t subnet_id = 0;
  bool enabled = false;

  // gfun layout:
  //   empty                         -> scale 1.0f (unit normal / OU)
  //   size == n_svar                -> per-svar gfun[sv]
  //   size == n_svar * n_node * W   -> per-(svar, node, lane)
  //   size == n_svar * n_node * n_modes * W -> full per-element
  std::vector<float> gfun;

  // Optional multiplicative diffusion: b(x) = mul_a * x + mul_b when x_state != nullptr.
  bool is_multiplicative = false;
  float mul_a = 1.0f;
  float mul_b = 0.0f;

  // Colored OU noise state: (n_svar, n_node, n_modes, W) when ntau > 0.
  std::vector<float> ou_state;
  float ntau = 0.0f;
  float dt = 0.001f;

  void set_uniform_seed(uint64_t seed) {
    for (int w = 0; w < W; w++) lane_keys[w] = seed;
  }

  void set_lane_seeds(const uint64_t *seeds, size_t count) {
    if (!seeds || count == 0) return;
    for (int w = 0; w < W; w++) {
      lane_keys[w] = (static_cast<size_t>(w) < count) ? seeds[w] : seeds[count - 1];
    }
  }

  rng_state<W> export_state() const {
    return rng_state<W>{lane_keys, step_counter, ou_state};
  }

  void import_state(const rng_state<W> &st) {
    lane_keys = st.lane_keys;
    step_counter = st.step_counter;
    ou_state = st.ou_state;
  }

  // Fill `out` of shape (n_svar, n_node, n_modes, W) with deterministic noise
  // at master/subnet step `step_idx`.
  // 100% invariant to chunk_size, thread scheduling, and SIMD width W.
  void fill_step_noise(float *out, int n_svar, int n_node, int n_modes,
                       uint64_t step_idx, const float *x_state = nullptr) {
    if (!out || n_svar <= 0 || n_node <= 0 || n_modes <= 0) return;

    const size_t n_elem =
        static_cast<size_t>(n_svar) * n_node * n_modes;

    // 1. Generate raw N(0, 1) draws into `out`.
    fill_raw_normals(out, n_elem, step_idx);

    // 2. Optional colored OU update (Fox-Vemuri exact transition).
    if (ntau > 0.0f && dt > 0.0f) {
      const size_t total = n_elem * static_cast<size_t>(W);
      if (ou_state.size() != total) {
        ou_state.resize(total);
        fill_raw_normals(ou_state.data(), n_elem, ~uint64_t{0});
      }
      const double E =
          std::exp(-static_cast<double>(dt) / static_cast<double>(ntau));
      const double sqrt_1_E2 = std::sqrt(std::max(0.0, 1.0 - E * E));
      const double out_scale =
          std::sqrt(static_cast<double>(dt) / static_cast<double>(ntau));

      for (size_t e = 0; e < n_elem; e++) {
        float *o = out + e * static_cast<size_t>(W);
        float *eta_p = ou_state.data() + e * static_cast<size_t>(W);
        #pragma omp simd
        for (int w = 0; w < W; w++) {
          const double eta = static_cast<double>(eta_p[w]) * E +
                             sqrt_1_E2 * static_cast<double>(o[w]);
          eta_p[w] = static_cast<float>(eta);
          o[w] = static_cast<float>(out_scale * eta);
        }
      }
    }

    // 3. Apply gfun diffusion scale.
    const size_t gs = gfun.size();
    if (gs == static_cast<size_t>(n_svar)) {
      const size_t stride_sv = static_cast<size_t>(n_node) * n_modes;
      for (int sv = 0; sv < n_svar; sv++) {
        const float g = gfun[static_cast<size_t>(sv)];
        float *base = out + static_cast<size_t>(sv) * stride_sv * W;
        const size_t n = stride_sv * static_cast<size_t>(W);
        #pragma omp simd
        for (size_t k = 0; k < n; k++) {
          base[k] *= g;
        }
      }
    } else if (gs == static_cast<size_t>(n_svar) * n_node * static_cast<size_t>(W)) {
      for (int sv = 0; sv < n_svar; sv++) {
        for (int j = 0; j < n_node; j++) {
          const float *g =
              gfun.data() + (static_cast<size_t>(sv) * n_node + j) * W;
          for (int m = 0; m < n_modes; m++) {
            float *dst =
                out + ((static_cast<size_t>(sv) * n_node + j) * n_modes + m) * W;
            #pragma omp simd
            for (int w = 0; w < W; w++) {
              dst[w] *= g[w];
            }
          }
        }
      }
    } else if (gs == n_elem * static_cast<size_t>(W)) {
      const size_t total = n_elem * static_cast<size_t>(W);
      #pragma omp simd
      for (size_t k = 0; k < total; k++) {
        out[k] *= gfun[k];
      }
    }

    // 4. Optional multiplicative state scaling: b(x) = mul_a * x + mul_b.
    if (is_multiplicative && x_state != nullptr) {
      const size_t total = n_elem * static_cast<size_t>(W);
      const float a = mul_a;
      const float b = mul_b;
      #pragma omp simd
      for (size_t k = 0; k < total; k++) {
        out[k] *= (a * x_state[k] + b);
      }
    }

    if (step_idx != ~uint64_t{0}) {
      step_counter = step_idx + 1;
    }
  }

  void fill_next_step_noise(float *out, int n_svar, int n_node, int n_modes,
                            const float *x_state = nullptr) {
    fill_step_noise(out, n_svar, n_node, n_modes, step_counter, x_state);
  }

 private:
  void fill_raw_normals(float *dst, size_t n_elem, uint64_t step_idx) const {
    bool all_same_key = true;
    for (int w = 1; w < W; w++) {
      if (lane_keys[w] != lane_keys[0]) {
        all_same_key = false;
        break;
      }
    }

    const uint32_t c0 = static_cast<uint32_t>(step_idx & 0xFFFFFFFFull);
    const uint32_t c1 = static_cast<uint32_t>(step_idx >> 32);
    const uint32_t c3 = subnet_id;
    const size_t full_blocks = n_elem / 4;
    const size_t tail = n_elem % 4;

    if (all_same_key) {
      const std::array<uint32_t, 2> key = {
          static_cast<uint32_t>(lane_keys[0] & 0xFFFFFFFFull),
          static_cast<uint32_t>(lane_keys[0] >> 32)};
      for (size_t blk = 0; blk < full_blocks; blk++) {
        const std::array<uint32_t, 4> ctr = {
            c0, c1, static_cast<uint32_t>(blk), c3};
        const auto z = philox4x32_normals(ctr, key);
        const size_t base = blk * 4;
        for (int q = 0; q < 4; q++) {
          float *p = dst + (base + static_cast<size_t>(q)) * W;
          const float zq = z[static_cast<size_t>(q)];
          #pragma omp simd
          for (int w = 0; w < W; w++) p[w] = zq;
        }
      }
      if (tail > 0) {
        const std::array<uint32_t, 4> ctr = {
            c0, c1, static_cast<uint32_t>(full_blocks), c3};
        const auto z = philox4x32_normals(ctr, key);
        const size_t base = full_blocks * 4;
        for (size_t q = 0; q < tail; q++) {
          float *p = dst + (base + q) * W;
          const float zq = z[q];
          #pragma omp simd
          for (int w = 0; w < W; w++) p[w] = zq;
        }
      }
      return;
    }

    // Per-lane independent keys: vectorize across W lanes.
    std::array<uint32_t, W> k0{}, k1{};
    for (int w = 0; w < W; w++) {
      k0[w] = static_cast<uint32_t>(lane_keys[w] & 0xFFFFFFFFull);
      k1[w] = static_cast<uint32_t>(lane_keys[w] >> 32);
    }

    for (size_t blk = 0; blk < full_blocks; blk++) {
      const uint32_t c2 = static_cast<uint32_t>(blk);
      const size_t base = blk * 4;
      float *p0 = dst + (base + 0) * W;
      float *p1 = dst + (base + 1) * W;
      float *p2 = dst + (base + 2) * W;
      float *p3 = dst + (base + 3) * W;
      #pragma omp simd
      for (int w = 0; w < W; w++) {
        const auto z = philox4x32_normals({c0, c1, c2, c3}, {k0[w], k1[w]});
        p0[w] = z[0];
        p1[w] = z[1];
        p2[w] = z[2];
        p3[w] = z[3];
      }
    }
    if (tail > 0) {
      const uint32_t c2 = static_cast<uint32_t>(full_blocks);
      const size_t base = full_blocks * 4;
      std::array<float, W> z0{}, z1{}, z2{};
      #pragma omp simd
      for (int w = 0; w < W; w++) {
        const auto z = philox4x32_normals({c0, c1, c2, c3}, {k0[w], k1[w]});
        z0[w] = z[0];
        z1[w] = z[1];
        z2[w] = z[2];
      }
      if (tail >= 1) {
        float *p0 = dst + (base + 0) * W;
        #pragma omp simd
        for (int w = 0; w < W; w++) p0[w] = z0[w];
      }
      if (tail >= 2) {
        float *p1 = dst + (base + 1) * W;
        #pragma omp simd
        for (int w = 0; w < W; w++) p1[w] = z1[w];
      }
      if (tail >= 3) {
        float *p2 = dst + (base + 2) * W;
        #pragma omp simd
        for (int w = 0; w < W; w++) p2[w] = z2[w];
      }
    }
  }
};

}  // namespace cph
