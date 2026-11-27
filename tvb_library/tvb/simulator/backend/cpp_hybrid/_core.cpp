// C++ hybrid simulator core for TVB.
//
// Runtime-tight SIMD kernels with control flow in C++ (no static codegen).
// Layout follows the vendored tvbk approach: the sweep/batch dimension is the
// SIMD lane dimension (W), so model dfuns and coupling math vectorize across
// batch members. State per subnet: (n_svar, n_node, W) float32.
//
// Coupling semantics match nb_hybrid: the weighted delayed sum is computed
// once per step from the source history (slot t-1-delay), the post-cfun is
// applied, and the same coupling is used for both Heun stages.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>  // std::vector<> caster
// gil release via scoped release below

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// ---- build-time generated ("built-in") generic dfuns -----------------------
//
// When the package build ran dfungen (CMakeLists.txt, TVB_CPP_GENERATE_MODELS),
// the generated kernels and their metadata table are compiled into this module
// and CPH_HAVE_BUILTIN_GEN is defined.  What follows redeclares the entry
// points dfungen.emit_sources emits; the declarations must stay identical to
// that emission.  cph_builtin_dfun returns 1 when it handled the model id and
// 0 when the id is not in the built-in table, which is what lets dfun_dispatch
// fall back to the runtime-injected dfun for models the build did not know
// about.  Without the macro nothing is declared and dispatch stays exactly as
// it was: the runtime pointer only.
#ifdef CPH_HAVE_BUILTIN_GEN
#define CPH_GEN_ENTRY_DEFINED 1
extern "C" {
typedef struct cph_gen_entry {
  const char *name;               /* model class name */
  int mid;                        /* model id (>= 100) */
  int n_parm, n_cvar, n_svar;     /* buffer shapes the kernel expects */
  const char *const *parm_names;  /* parr packing order */
  const char *signature;          /* dfun fingerprint (model_dfun_signature) */
} cph_gen_entry;
int cph_builtin_dfun(int model_id, float *dx, const float *x, int node,
                     int mode, int n_node, int n_modes, const float *c,
                     const float *p, int n_parm, int Wn);
int cph_builtin_count(void);
const cph_gen_entry *cph_builtin_entry(int idx);
}
#endif

#include "cph_coupling.hpp"
#include "cph_dfuns.hpp"
#include "cph_features.hpp"
#include "cph_monitors.hpp"
#include "cph_rng.hpp"
#include "cph_stim.hpp"

namespace nb = nanobind;

// nanobind 3.x dropped the legacy `array_t<T>` / `unchecked<N>()` API:
// NumPy arrays are `nb::ndarray<nb::numpy, T>` (dtype enforced at the
// binding boundary) and typed access is `view<nb::ndim<N>>()` with the same
// (k, i, ...) indexing.  Output arrays are backed by a capsule-owned buffer
// so the numpy view handed to Python keeps the memory alive (same lifetime
// semantics as the former binding layer's shape-constructed numpy arrays).
template <typename T>
static nb::ndarray<nb::numpy, T> make_owned_array(
    const std::vector<size_t> &shape) {
  size_t n = 1;
  for (size_t s : shape) n *= s;
  T *buf = new T[n]();
  nb::capsule owner(buf,
                    [](void *p) noexcept { delete[] static_cast<T *>(p); });
  return nb::ndarray<nb::numpy, T>(buf, shape.size(), shape.data(), owner);
}

#include "cph_monitor_bindings.hpp"

namespace cph {

// ---- simulator (one per width instantiation) -------------------------------

template <int W> struct simw {
  static constexpr int W_CONST = W;
  std::vector<subnet<W>> sn;
  std::vector<proj<W>> pr;
  int integ_id = 1;  // 0 = euler, 1 = heun
  float dt = 0.001f;
  int tavg_period = 1;
  std::vector<std::vector<voispec>> voi_specs;       // per subnet
  std::vector<std::vector<float>> obs_row;           // per-subnet (n_voi, n_node, W)
  std::vector<std::vector<kmon<W>>> mons;            // per-subnet monitor engines
  std::vector<GlobalGatherSubnet> global_gather;     // per-subnet global gather spec
  std::vector<kmon<W>> global_mons;                  // global / merged monitor engines
  std::vector<float> global_obs_row;                 // (total_vois, total_nodes, W)
  int global_total_vois = 0;
  int global_total_nodes = 0;
  std::vector<stim_spec<W>> stim_specs;              // C++ stimulus specs
  std::vector<rng_engine<W>> sub_rng;                // per-subnet Philox RNG engines
  std::vector<std::vector<float>> rng_scratch;       // per-subnet step noise buffer
  std::vector<uint8_t> online_enabled;               // per-subnet online feature flag
  std::vector<OnlineFeatureAccum<W>> online_accums;  // per-subnet Tier-1 accumulators
  uint64_t t_abs = 0;

  void validate_local_window_source(size_t si, const kmon<W> &m) const {
    const auto &sub = sn[si];
    if (sub.K != m.cadence_K || !matches_native_clock(m.dt, sub.dt) ||
        !matches_native_clock(m.local_master_dt, dt))
      throw std::runtime_error("Recorder complete window: incompatible source or master clock");
  }

  void configure_local_window(size_t si, const nb::dict &d, kmon<W> &m) const {
    const auto &sub = sn[si];
    const auto &source = voi_specs[si];
    const int slot = nb::cast<int>(d["slot"]);
    const int kind = nb::cast<int>(d["kind"]);
    const int period = nb::cast<int>(d["istep"]);
    const uint32_t cadence = nb::cast<uint32_t>(d["cadence_K"]);
    const double clock = nb::cast<double>(d["dt"]);
    const double master_clock = nb::cast<double>(d["master_dt"]);
    const size_t max_int = (size_t)std::numeric_limits<int>::max();
    const bool bw_local = (kind == KMON_BOLD_BW_LOCAL);
    // Average windows stay width-1 only; the local BW window carries independent
    // per-region, per-lane state and supports every instantiated SIMD width.
    if ((!bw_local && W != 1) || sub.n_modes != 1 || sub.n_node < 1 ||
        source.empty() || source.size() > max_int ||
        source.size() > obs_row[si].max_size() / (size_t)sub.n_node ||
        slot < recorder_slot_base || (!average_window_kind(kind) && !bw_local) ||
        period < 1 || cadence == 0 ||
        dict_get_or<bool>(d, "complete_window", false))
      throw std::runtime_error("Recorder complete window: invalid attachment geometry, kind or period");
    // Match the existing NetworkSet cadence tolerance, and check both stored native clocks.
    if (cadence != sub.K || !matches_native_clock(clock, sub.dt) ||
        !matches_native_clock(master_clock, dt) ||
        !matches_scheduled_cadence(clock, cadence, master_clock))
      throw std::runtime_error("Recorder complete window: incompatible source or master clock");
    for (const auto &v : source) {
      if (v.kind < 0 || v.kind > 1 || v.a < 0 || v.a >= sub.n_svar ||
          v.b < 0 || v.b >= sub.n_svar)
        throw std::runtime_error("Recorder complete window: invalid observed VOI source");
    }
    const auto voi = to_checked_indices(d["voi"], source.size(), "Recorder complete window", "observed VOI");
    const size_t n_voi = voi.empty() ? source.size() : voi.size();
    if (bw_local) {
      for (const char *key : {"bw_stream_params", "bw_drive_baseline", "bw_drive_scale",
                              "source_name", "node_indices"})
        if (!d.contains(key))
          throw std::runtime_error("Recorder complete window: incomplete BW attachment specification");
      if (voi.size() != 1)
        throw std::runtime_error("Recorder complete window: exactly one valid observed VOI is required");
      m.local_window = true;
      m.local_master_dt = master_clock;
      m.local_source_name = nb::cast<std::string>(d["source_name"]);
      m.local_node_indices = nb::cast<std::vector<int64_t>>(d["node_indices"]);
      if (!m.local_node_indices.empty() && m.local_node_indices.size() != (size_t)sub.n_node)
        throw std::runtime_error("Recorder complete window: incompatible node order extent");
      m.configure_bw_local(slot, period, clock, sub.n_node, voi[0],
                           nb::cast<double>(d["bw_drive_baseline"]),
                           nb::cast<double>(d["bw_drive_scale"]),
                           parse_bw_parameters(d, "Recorder complete window"), cadence);
      return;
    }
    const size_t max_space = std::min(m.accum.max_size(), m.d_out.max_size()) / n_voi;
    m.local_window = true;
    m.local_master_dt = master_clock;
    m.local_source_name = nb::cast<std::string>(d["source_name"]);
    m.local_node_indices = nb::cast<std::vector<int64_t>>(d["node_indices"]);
    if (!m.local_node_indices.empty() && m.local_node_indices.size() != (size_t)sub.n_node)
      throw std::runtime_error("Recorder complete window: incompatible node order extent");
    if (kind == KMON_SPATIAL_AVG) {
      using SpatialMean = nb::ndarray<nb::numpy, const float, nb::ndim<2>, nb::c_contig>;
      SpatialMean weights;
      if (!d.contains("spatial_mean") || !nb::try_cast(d["spatial_mean"], weights, false) ||
          weights.shape(0) == 0 || weights.shape(1) != (size_t)sub.n_node ||
          weights.shape(0) > max_space || weights.shape(0) > max_int / weights.shape(1))
        throw std::runtime_error("Recorder complete window: invalid spatial_mean extent or dtype");
      if (std::any_of(weights.data(), weights.data() + weights.size(),
                      [](float w) { return !std::isfinite(w); }))
        throw std::runtime_error("Recorder complete window: nonfinite spatial_mean");
      m.set_spatial_mean(weights.data(), (int)weights.shape(0), sub.n_node);
    } else if ((kind == KMON_GLOBAL_AVG ? 1u : (size_t)sub.n_node) > max_space) {
      throw std::runtime_error("Recorder complete window: output extent out of range");
    }
    m.configure_basic(slot, kind, period, clock, (int)source.size(), sub.n_node, 1, voi, cadence);
  }

  void set_monitors(int si, nb::list atts) {
    if (si < 0 || (size_t)si >= sn.size())
      throw std::runtime_error("bad subnet index in set_monitors");
    if (mons.size() < sn.size()) mons.resize(sn.size());
    auto &cur = mons[si];
    const int n_voi = (int)voi_specs[si].size();
    const int n_node = sn[si].n_node;
    const int n_cvar = sn[si].n_cvar;
    const int n_modes = sn[si].n_modes;
    const double def_dt = (double)dt;
    std::vector<kmon<W>> next;
    next.reserve(atts.size());
    for (auto item : atts) {
      kmon<W> m;
      if (nb::isinstance<nb::dict>(item) &&
          dict_get_or<bool>(nb::cast<nb::dict>(item), "local_window", false)) {
        configure_local_window((size_t)si, nb::cast<nb::dict>(item), m);
      } else {
        parse_kmon_spec<W>(item, n_voi, n_node, n_cvar, n_modes, def_dt, cur, m);
      }
      if (std::any_of(next.begin(), next.end(), [&](const auto &other) {
            return (m.local_window || other.local_window) && other.slot == m.slot;
          }))
        throw std::runtime_error("Recorder complete window: duplicate attachment slot");
      next.push_back(std::move(m));
    }
    cur = std::move(next);
  }

  void set_global_monitors(nb::list gather_specs, nb::list mon_specs) {
    if (gather_specs.size() != sn.size())
      throw std::runtime_error(
          "set_global_monitors: gather_specs length != subnet count");
    bool has_window = false;
    bool has_projection_window = false;
    bool has_bw_stream = false;
    for (auto item : mon_specs) {
      if (!nb::isinstance<nb::dict>(item)) continue;
      const auto d = nb::cast<nb::dict>(item);
      const bool complete = dict_get_or<bool>(d, "complete_window", false);
      const int kind = dict_get_or<int>(d, "kind", 0);
      has_window |= complete;
      has_projection_window |= complete && kind == KMON_PROJECTION;
      has_bw_stream |= kind == KMON_BOLD_BW_STREAM;
    }
    if (has_bw_stream && sn.size() == 1) {
      if (sn[0].K != 1 || sn[0].dt != dt || sn[0].n_modes != 1)
        throw std::runtime_error("global BW stream requires one regional clock and mode");
      validate_single_source_observations("global BW stream");
    }
    if (has_projection_window) {
      if (W != 1 || sn.size() != 1 || sn[0].K != 1 || sn[0].dt != dt || sn[0].n_modes != 1)
        throw std::runtime_error("global complete window projection requires width 1, one regional subnet, clock and mode");
      if (!std::isfinite(dt) || dt <= 0.0f)
        throw std::runtime_error("global complete window projection: invalid stored neural clock");
      validate_single_source_observations(projection_label);
    }
    const bool staged = has_window || has_bw_stream;
    const bool gathered_window = (has_window || has_bw_stream) && sn.size() > 1;
    const bool mixed_rate = gathered_window && std::any_of(sn.begin(), sn.end(),
        [](const auto &sub) { return sub.K > 1; });
    int64_t source_nodes = 0;
    if (gathered_window) {
      if (has_window && W != 1)
        throw std::runtime_error("global gathered complete window requires width 1");
      for (const auto &sub : sn) {
        source_nodes += sub.n_node;
        if (sub.n_node <= 0 || source_nodes > std::numeric_limits<int>::max())
          throw std::runtime_error("global gathered complete window: gather extent out of range");
      }
    }
    std::vector<GlobalGatherSubnet> gather;
    gather.reserve(gather_specs.size());
    int tot_v = 0;
    int tot_n = 0;
    const auto parse_node_indices = [&](nb::handle h) {
      if (gathered_window)
        return to_checked_indices(h, (size_t)source_nodes,
                                  "global gathered complete window", "node gather index");
      return staged
          ? to_checked_indices(h, (size_t)sn[0].n_node,
                               has_bw_stream ? "global BW stream" : has_projection_window
                                   ? projection_label : "global complete window",
                               "node gather index")
          : to_int_vec(h);
    };
    for (auto item : gather_specs) {
      GlobalGatherSubnet gs;
      if (nb::isinstance<nb::dict>(item)) {
        nb::dict d = nb::cast<nb::dict>(item);
        gs.n_voi = nb::cast<int>(d["n_voi"]);
        gs.n_node = nb::cast<int>(d["n_node"]);
        gs.voi_offset = dict_get_or<int>(d, "voi_offset", 0);
        gs.node_offset = dict_get_or<int>(d, "node_offset", 0);
        if (d.contains("node_indices") && !d["node_indices"].is_none())
          gs.node_indices = parse_node_indices(d["node_indices"]);
        if (gathered_window) {
          for (const char *key : {"source_name", "source_dt", "observed_terms"})
            if (!d.contains(key))
              throw std::runtime_error("global gathered complete window: incomplete source declaration");
          gs.source_name = nb::cast<std::string>(d["source_name"]);
          gs.source_dt = nb::cast<double>(d["source_dt"]);
          gs.observed_terms = nb::cast<std::vector<std::string>>(d["observed_terms"]);
        }
      } else {
        if (gathered_window)
          throw std::runtime_error("global gathered complete window requires explicit source declarations");
        nb::tuple t = nb::cast<nb::tuple>(item);
        gs.n_voi = nb::cast<int>(t[0]);
        gs.n_node = nb::cast<int>(t[1]);
        gs.voi_offset = nb::cast<int>(t[2]);
        gs.node_offset = nb::cast<int>(t[3]);
        if (t.size() > 4 && !t[4].is_none())
          gs.node_indices = parse_node_indices(t[4]);
      }
      if (!staged) {
        if (gs.voi_offset + gs.n_voi > tot_v) tot_v = gs.voi_offset + gs.n_voi;
        if (gs.node_indices.empty()) {
          if (gs.node_offset + gs.n_node > tot_n)
            tot_n = gs.node_offset + gs.n_node;
        } else {
          for (int gn : gs.node_indices)
            if (gn + 1 > tot_n) tot_n = gn + 1;
        }
      }
      gather.push_back(std::move(gs));
    }
    double master_clock = (double)dt;
    if (gathered_window) {
      // The declared master clock is the smallest source clock, not source 0.
      master_clock = std::numeric_limits<double>::infinity();
      for (size_t i = 0; i < gather.size(); ++i) {
        if (!matches_native_clock(gather[i].source_dt, sn[i].dt))
          throw std::runtime_error("global gathered complete window: incompatible declared source clock");
        master_clock = std::min(master_clock, gather[i].source_dt);
      }
      if (!matches_native_clock(master_clock, dt))
        throw std::runtime_error("global gathered complete window: incompatible declared master clock");
    }
    if (staged) {
      const auto [n_voi, n_node] = validate_window_gather(gather, gathered_window, mixed_rate, master_clock);
      tot_v = n_voi;
      tot_n = n_node;
    }
    if (has_projection_window) {
      const auto &g = gather.front();
      for (int j = 0; j < g.n_node; ++j)
        if (g.node_indices.empty() ? g.node_offset != 0 : g.node_indices[j] != j)
          throw std::runtime_error("global complete window projection requires local node order");
    }

    std::vector<kmon<W>> next;
    next.reserve(mon_specs.size());
    const double def_dt = (double)dt;
    for (auto item : mon_specs) {
      kmon<W> m;
      if (gathered_window && nb::isinstance<nb::dict>(item) &&
          dict_get_or<bool>(nb::cast<nb::dict>(item), "complete_window", false)) {
        const auto spec = nb::cast<nb::dict>(item);
        if (dict_get_or<bool>(spec, "local_window", false))
          throw std::runtime_error("global gathered complete window requires a global attachment");
        configure_gathered_window(spec, tot_v, tot_n, master_clock, m);
      } else {
        if (mixed_rate && !(nb::isinstance<nb::dict>(item) &&
            dict_get_or<int>(nb::cast<nb::dict>(item), "kind", 0) == KMON_BOLD_BW_STREAM))
          throw std::runtime_error("global gathered complete window: mixed-rate gather requires complete-window monitors");
        parse_kmon_spec<W>(item, tot_v, tot_n, tot_v, 1, def_dt, global_mons, m, true);
      }
      if (m.bw_stream()) m.bw.source_voi = voi_specs[0][(size_t)m.voi[0]];
      if (m.staged_global()) m.begin_run(t_abs);
      next.push_back(std::move(m));
    }
    if (staged) {
      for (const auto &m : next) {
        if (!m.staged_global()) continue;
        if (m.slot < 0 || std::count_if(next.begin(), next.end(), [&](const auto &other) {
              return other.slot == m.slot;
            }) != 1)
          throw std::runtime_error(m.bw_stream() ? "global BW stream: monitor slot must be unique"
                                                 : "global complete window: monitor slot must be unique");
      }
    }
    global_obs_row.assign(
        (size_t)std::max(0, tot_v) * (size_t)std::max(0, tot_n) * (size_t)W,
        0.0f);
    global_total_vois = tot_v;
    global_total_nodes = tot_n;
    global_gather = std::move(gather);
    global_mons = std::move(next);
  }

  std::pair<int, int> validate_window_gather(
      const std::vector<GlobalGatherSubnet> &gather, bool gathered_window,
      bool mixed_rate, double master_clock) const {
    const std::string label = gathered_window ? gathered_window_label : "global complete window: ";
    const size_t n_voi = voi_specs.empty() ? 0 : voi_specs.front().size();
    constexpr int max_extent = std::numeric_limits<int>::max();
    if (gather.empty() || n_voi == 0 || n_voi > (size_t)max_extent)
      throw std::runtime_error(label + "incompatible gather shape");
    // Derive the allocation extents from the sources, never from offsets.
    int64_t n_node = 0;
    for (size_t i = 0; i < gather.size(); ++i) {
      const auto &g = gather[i];
      const auto &sub = sn[i];
      if (sub.n_modes != 1 || (!gathered_window && (sub.K != 1 || sub.dt != dt)))
        throw std::runtime_error(gathered_window ? "global gathered complete window requires one source mode"
                                                : "global complete window requires one regional clock and mode");
      if (g.n_node != sub.n_node || sub.n_node <= 0 || g.n_voi != (int)n_voi ||
          voi_specs[i].size() != n_voi || g.voi_offset != 0 ||
          (!g.node_indices.empty() && g.node_indices.size() != (size_t)sub.n_node))
        throw std::runtime_error(label + "incompatible gather shape");
      if (gathered_window) {
        const auto &first = gather.front();
        if (g.node_indices.empty() || g.node_offset != 0)
          throw std::runtime_error("global gathered complete window requires explicit node_indices");
        if (!mixed_rate) {
          if (sub.K != 1 || sub.dt != dt || g.source_dt != master_clock)
            throw std::runtime_error("global gathered complete window requires a common declared clock");
        } else {
          if (!matches_scheduled_cadence(g.source_dt, sub.K, master_clock))
            throw std::runtime_error("global gathered complete window: incompatible declared source clock or cadence");
        }
        if (g.observed_terms.size() != n_voi || g.observed_terms != first.observed_terms)
          throw std::runtime_error("global gathered complete window requires homogeneous observed VOIs");
        for (size_t v = 0; v < n_voi; ++v) {
          const auto &row = voi_specs[i][v];
          const auto &ref = voi_specs[0][v];
          if (row.kind < 0 || row.kind > 1 || row.a < 0 || row.a >= sub.n_svar ||
              row.b < 0 || row.b >= sub.n_svar || row.kind != ref.kind ||
              row.a != ref.a || row.b != ref.b)
            throw std::runtime_error("global gathered complete window: incompatible observed VOI source");
        }
      }
      n_node += sub.n_node;
      if (n_node > max_extent)
        throw std::runtime_error(label + "gather extent out of range");
    }
    const size_t max_elems = std::min(global_obs_row.max_size(),
                                     std::vector<double>().max_size());
    if (n_voi > max_elems / (size_t)W / (size_t)n_node)
      throw std::runtime_error(label + "gather extent out of range");
    std::vector<bool> seen((size_t)n_node, false);
    for (const auto &g : gather) {
      for (int j = 0; j < g.n_node; ++j) {
        const int64_t node = g.node_indices.empty() ? (int64_t)g.node_offset + j
                                                    : g.node_indices[j];
        if (node < 0 || node >= n_node || seen[(size_t)node])
          throw std::runtime_error(gathered_window ? "global gathered complete window requires a complete, disjoint node gather"
                                           : "global complete window requires a complete, disjoint node gather");
        seen[(size_t)node] = true;
      }
    }
    return {(int)n_voi, (int)n_node};
  }

  bool projection_window_armed() const {
    return std::any_of(global_mons.begin(), global_mons.end(),
                       [](const auto &m) { return m.projection_window(); });
  }

  bool gathered_window_armed() const {
    return std::any_of(global_mons.begin(), global_mons.end(),
                       [](const auto &m) { return m.gathered_window; });
  }

  bool bw_stream_armed() const {
    return std::any_of(global_mons.begin(), global_mons.end(),
                       [](const auto &m) { return m.bw_stream(); });
  }

  bool regional_average_window_armed() const {
    return std::any_of(global_mons.begin(), global_mons.end(), [](const auto &m) {
      return m.complete_window && !m.gathered_window && average_window_kind(m.kind);
    });
  }

  const char *staged_global_armed_label() const {
    if (bw_stream_armed()) return "global BW stream";
    if (projection_window_armed()) return "global complete window projection";
    if (gathered_window_armed()) return "global gathered complete window";
    if (regional_average_window_armed()) return "global complete window";
    return nullptr;
  }

  void require_no_staged_global_armed(const char *action) const {
    if (const char *label = staged_global_armed_label())
      throw std::runtime_error(std::string(label) + ": detach before " + action);
  }

  void set_observed_rows(int si, std::vector<voispec> rows) {
    auto &current = voi_specs[si];
    if (!std::equal(current.begin(), current.end(), rows.begin(), rows.end(),
                    [](const auto &a, const auto &b) {
                      return a.kind == b.kind && a.a == b.a && a.b == b.b;
                    })) {
      require_no_staged_global_armed("changing observed rows");
      if (std::any_of(mons[si].begin(), mons[si].end(), [](const auto &m) { return m.local_window; }))
        throw std::runtime_error("Recorder complete window: detach before changing observed rows");
    }
    current = std::move(rows);
  }

  void validate_single_source_observations(const char *label) const {
    for (const auto &v : voi_specs[0]) {
      if (v.kind < 0 || v.kind > 1 || v.a < 0 || v.a >= sn[0].n_svar ||
          v.b < 0 || v.b >= sn[0].n_svar)
        throw std::runtime_error(std::string(label) + ": invalid observed VOI source");
    }
  }

  void validate_bw_live_config(const kmon<W> &m) const {
    if (sn.empty() || dt != (float)m.dt || voi_specs[0].size() != (size_t)global_total_vois)
      throw std::runtime_error("global BW stream: changed simulator geometry or clock; reattach monitor");
    const bool multi = sn.size() > 1;
    const bool mixed = multi && std::any_of(sn.begin(), sn.end(), [](const auto &sub) { return sub.K > 1; });
    double master_clock = m.dt;
    if (multi) {
      if (global_gather.size() != sn.size())
        throw std::runtime_error("global BW stream: changed simulator geometry or clock; reattach monitor");
      master_clock = std::numeric_limits<double>::infinity();
      for (size_t i = 0; i < global_gather.size(); ++i) {
        if (!matches_native_clock(global_gather[i].source_dt, sn[i].dt))
          throw std::runtime_error("global BW stream: changed simulator geometry or clock; reattach monitor");
        master_clock = std::min(master_clock, global_gather[i].source_dt);
      }
      if (!matches_native_clock(master_clock, dt))
        throw std::runtime_error("global BW stream: changed simulator geometry or clock; reattach monitor");
    }
    const auto shape = validate_window_gather(global_gather, multi, mixed, master_clock);
    if (shape.second != m.n_out_space || shape.first != global_total_vois)
      throw std::runtime_error("global BW stream: changed simulator geometry; reattach monitor");
    const auto &v = voi_specs[0][(size_t)m.voi[0]];
    if (v.kind != m.bw.source_voi.kind || v.a != m.bw.source_voi.a || v.b != m.bw.source_voi.b)
      throw std::runtime_error("global BW stream: changed selected observation; reattach monitor");
    validate_single_source_observations("global BW stream");
  }

  nb::list window_source_config() const {
    nb::list sources;
    for (size_t i = 0; i < global_gather.size(); ++i) {
      const auto &g = global_gather[i];
      nb::list source_voi;
      for (const auto &v : voi_specs[i])
        source_voi.append(nb::make_tuple(v.kind, v.a, v.b));
      sources.append(nb::make_tuple(g.n_voi, g.n_node, g.voi_offset, g.node_offset,
                                    g.node_indices, source_voi));
    }
    return sources;
  }

  nb::tuple window_config(const kmon<W> &m, const nb::list &sources) const {
    // The projection's live source definitions and clock cannot change while armed.
    if (m.projection_window())
      return nb::make_tuple(m.istep, m.dt, m.voi, sources, m.proj.n_sensors, m.proj.n_nodes,
                            m.proj.gain, m.proj.ref_vec);
    // Exact effective configuration, including weights and input row identity.
    return nb::make_tuple(m.istep, m.dt, m.voi, sources, m.spatial.compact,
                          m.spatial.area_idx, m.spatial.scale, m.spatial.row_ptr,
                          m.spatial.col_idx, m.spatial.val);
  }

  nb::tuple gathered_window_config(const kmon<W> &m) const {
    nb::list declared_sources;
    for (size_t i = 0; i < global_gather.size(); ++i) {
      const auto &g = global_gather[i];
      declared_sources.append(nb::make_tuple(g.source_name, g.source_dt, g.observed_terms,
                                           sn[i].K, sn[i].n_svar, sn[i].n_modes));
    }
    return nb::make_tuple(1, m.slot, m.kind, m.istep, m.dt, m.voi, declared_sources,
                          window_source_config(), m.spatial.row_ptr, m.spatial.col_idx, m.spatial.val);
  }

  nb::tuple bw_stream_config(const kmon<W> &m) const {
    nb::list params;
    for (const auto &field : bw_parameter_fields)
      params.append(m.bw.params.*field.second);
    if (sn.size() == 1 && W == 1)
      return nb::make_tuple(m.istep, m.dt, m.bw.drive_steps, kmon<W>::BwWindowState::drive_ms,
                            kmon<W>::BwWindowState::substeps,
                            m.voi, window_source_config(), m.bw.drive_baseline, m.bw.drive_scale,
                            params);
    nb::list declarations;
    for (size_t i = 0; i < global_gather.size(); ++i) {
      const auto &g = global_gather[i];
      declarations.append(nb::make_tuple(g.source_name, g.source_dt, g.observed_terms,
                                         sn[i].K, sn[i].dt, sn[i].n_svar, sn[i].n_modes));
    }
    return nb::make_tuple(2, W, m.slot, m.istep, m.dt, m.bw.drive_steps, kmon<W>::BwWindowState::drive_ms,
                          kmon<W>::BwWindowState::substeps, m.voi, window_source_config(), declarations,
                          m.bw.drive_baseline, m.bw.drive_scale, params);
  }

  nb::list take_global_monitor_outputs() {
    nb::list out;
    for (auto &gm : global_mons) {
      const size_t n = gm.t_out.size();
      nb::ndarray<nb::numpy, double> mt = make_owned_array<double>({n});
      if (n > 0)
        std::copy(gm.t_out.begin(), gm.t_out.end(),
                  static_cast<double *>(mt.data()));
      nb::ndarray<nb::numpy, float> md = make_owned_array<float>(
          {n, (size_t)gm.n_out_v, (size_t)gm.n_out_space,
           (size_t)gm.n_out_modes * (size_t)W});
      if (!gm.d_out.empty())
        std::copy(gm.d_out.begin(), gm.d_out.end(),
                  static_cast<float *>(md.data()));
      gm.begin_call();
      out.append(nb::make_tuple(mt, md));
    }
    return out;
  }

  nb::tuple local_window_config(size_t si, const kmon<W> &m) const {
    nb::list observations;
    for (const auto &v : voi_specs[si])
      observations.append(nb::make_tuple(v.kind, v.a, v.b));
    const auto &sub = sn[si];
    if (m.bw_local()) {
      // Version 2 appends the held-drive window geometry, the calibration and
      // the whole explicit hemodynamic set to the shared local-window identity.
      nb::list params;
      for (const auto &field : bw_parameter_fields)
        params.append(m.bw.params.*field.second);
      return nb::make_tuple(2, si, m.slot, m.kind, m.local_source_name, m.dt, m.local_master_dt,
                            m.cadence_K, m.istep, m.voi, observations,
                            sub.n_svar, sub.n_node, sub.n_modes, m.local_node_indices,
                            m.spatial.row_ptr, m.spatial.col_idx, m.spatial.val,
                            m.bw.drive_steps, m.bw.output_endpoints, m.bw.drive_baseline,
                            m.bw.drive_scale, params);
    }
    return nb::make_tuple(1, si, m.slot, m.kind, m.local_source_name, m.dt, m.local_master_dt,
                          m.cadence_K, m.istep, m.voi, observations,
                          sub.n_svar, sub.n_node, sub.n_modes, m.local_node_indices,
                          m.spatial.row_ptr, m.spatial.col_idx, m.spatial.val);
  }

  // Export the two open local BW windows per region and lane. The keys are
  // specific to this kind; the legacy scalar fields stay empty in the dict.
  void append_bw_local_state(nb::dict &d, const KMonState &st, const kmon<W> &m) const {
    const size_t n_node = (size_t)m.n_out_space;
    auto regions = make_owned_array<double>({n_node, (size_t)W, (size_t)6});
    if (!st.bw_regions.empty())
      pack_bw_regions(st.bw_regions, regions.data());
    d["bw_local_regions"] = regions;
    d["bw_drive_count"] = st.bw_drive_count;
  }

  void parse_bw_local_state(KMonState &st, const nb::dict &d, const kmon<W> &m,
                            const char *label) const {
    if (!d.contains("bw_local_regions") || !d.contains("bw_drive_count") ||
        !PyLong_CheckExact(d["bw_drive_count"].ptr()))
      throw std::runtime_error(std::string(label) + "invalid snapshot regional state");
    using BwLocal = nb::ndarray<nb::numpy, const double, nb::ndim<3>, nb::c_contig>;
    BwLocal regions;
    if (!nb::try_cast(d["bw_local_regions"], regions, false) ||
        regions.shape(0) != (size_t)m.n_out_space ||
        regions.shape(1) != (size_t)W || regions.shape(2) != 6)
      throw std::runtime_error(std::string(label) + "invalid snapshot regional extent or dtype");
    st.bw_drive_count = nb::cast<uint32_t>(d["bw_drive_count"]);
    unpack_bw_regions(regions.data(), (size_t)m.n_out_space * (size_t)W, st.bw_regions);
  }

  nb::list get_monitor_states() const {
    nb::list per_sn;
    for (size_t si = 0; si < sn.size(); si++) {
      nb::list ms;
      if (si < mons.size()) {
        for (const auto &m : mons[si]) {
          if (m.local_window) validate_local_window_source(si, m);
          KMonState st = m.export_state();
          nb::dict d = kmon_state_to_dict(st);
          if (m.local_window) {
            m.validate_local_window_state(st, t_abs);
            d["local_window_config"] = local_window_config(si, m);
            d["local_window_phase"] = t_abs % m.cadence_K;
            if (m.bw_local()) append_bw_local_state(d, st, m);
          }
          ms.append(d);
        }
      }
      per_sn.append(ms);
    }
    return per_sn;
  }

  void set_monitor_states(nb::list states) {
    const bool strict = std::any_of(mons.begin(), mons.end(), [](const auto &monitors) {
      return std::any_of(monitors.begin(), monitors.end(), [](const auto &m) { return m.local_window; });
    });
    std::vector<std::pair<kmon<W> *, KMonState>> staged;
    for (size_t si = 0; si < states.size(); ++si) {
      nb::list ms = nb::cast<nb::list>(states[si]);
      for (auto item : ms) {
        nb::dict d = nb::cast<nb::dict>(item);
        const bool window = d.contains("local_window_config");
        if (strict) {
          for (const char *key : {"slot", "kind", "width", "n_out_v", "n_out_space", "n_out_modes"})
            if (!d.contains(key) || !PyLong_CheckExact(d[key].ptr()))
              throw std::runtime_error("Recorder complete window: invalid snapshot identity or geometry");
        }
        const int slot = dict_get_or<int>(d, "slot", -1);
        const int kind = dict_get_or<int>(d, "kind", 0);
        if (!window) {
          // The reserved Recorder slots distinguish historical local windows from global
          // monitors routed through the legacy per-subnet collector.
          if (d.contains("local_window_phase") || (slot >= recorder_slot_base &&
              (average_window_kind(kind) || kind == KMON_BOLD_BW_LOCAL)))
            throw std::runtime_error("Recorder complete window: historical snapshot monitor");
          if (!strict) continue;
        }
        if (si >= mons.size())
          throw std::runtime_error("Recorder complete window: unmatched snapshot subnet");
        const auto matches = [&](const auto &m) { return m.slot == slot && m.kind == kind; };
        auto it = std::find_if(mons[si].begin(), mons[si].end(), matches);
        if (it == mons[si].end() || it->local_window != window ||
            std::find_if(std::next(it), mons[si].end(), matches) != mons[si].end())
          throw std::runtime_error("Recorder complete window: unmatched or ambiguous snapshot monitor");
        if (std::any_of(staged.begin(), staged.end(), [&](const auto &entry) { return entry.first == &*it; }))
          throw std::runtime_error("Recorder complete window: duplicate snapshot monitor");
        if (window) {
          validate_local_window_source(si, *it);
          if (!equal_checked_window_config(d["local_window_config"].ptr(), local_window_config(si, *it).ptr()))
            throw std::runtime_error("Recorder complete window: incompatible snapshot configuration");
          if (!d.contains("local_window_phase") || !PyLong_CheckExact(d["local_window_phase"].ptr()) ||
              nb::cast<uint32_t>(d["local_window_phase"]) != t_abs % it->cadence_K)
            throw std::runtime_error("Recorder complete window: invalid snapshot master cadence phase");
          KMonState st;
          if (it->bw_local()) {
            // Keep the strict local Recorder state checks (exact integer
            // identity fields, one-dimensional float64 accumulator) and parse
            // the BW windows afterwards; the accumulator extent is zero here.
            st = dict_to_checked_window_state(d, 0, local_window_label);
            parse_bw_local_state(st, d, *it, local_window_label);
          } else {
            st = dict_to_checked_window_state(d, it->accum.size(), local_window_label);
          }
          it->validate_local_window_state(st, t_abs);
          staged.emplace_back(&*it, std::move(st));
        } else {
          KMonState st = dict_to_legacy_sibling_state(d, local_window_label);
          it->validate_legacy_sibling_state(st, local_window_label);
          staged.emplace_back(&*it, std::move(st));
        }
      }
    }
    if (strict) {
      // Every attached local monitor must match once before any state or output is changed.
      for (const auto &monitors : mons) {
        for (const auto &m : monitors) {
          if (std::none_of(staged.begin(), staged.end(), [&](const auto &entry) { return entry.first == &m; }))
            throw std::runtime_error("Recorder complete window: missing compatible snapshot monitor");
        }
      }
      for (auto &[monitor, state] : staged) monitor->commit_validated_state(state);
      return;
    }

    // Keep historical matching and position fallback for legacy-only restores.
    const size_t n = std::min(states.size(), mons.size());
    for (size_t si = 0; si < n; si++) {
      nb::list ms = nb::cast<nb::list>(states[si]);
      for (size_t mi = 0; mi < ms.size(); mi++) {
        nb::dict d = nb::cast<nb::dict>(ms[mi]);
        if (d.contains("local_window_config")) continue;
        KMonState st = dict_to_kmon_state(d);
        auto it = std::find_if(mons[si].begin(), mons[si].end(),
                               [&](const kmon<W> &m) {
                                 return !m.local_window && m.slot == st.slot && m.kind == st.kind;
                               });
        if (it != mons[si].end()) {
          it->import_state(st);
        } else if (mi < mons[si].size() && !mons[si][mi].local_window) {
          mons[si][mi].import_state(st);
        }
      }
    }
  }

  nb::list get_global_monitor_states() const {
    nb::list out;
    for (const auto &gm : global_mons) {
      const auto st = gm.export_state();
      nb::dict state = kmon_state_to_dict(st);
      if (gm.gathered_window) {
        gm.validate_window_state(st, t_abs);
        // Keep the established key and schema: the checked identity already includes each source's K.
        state["coclock_window_config"] = gathered_window_config(gm);
      } else if (gm.complete_window) {
        state["window_config"] = window_config(gm, window_source_config());
      } else if (gm.bw_stream()) {
        validate_bw_live_config(gm);
        state["bw_stream_config"] = bw_stream_config(gm);
        export_bw_stream_state(state, st);
      }
      out.append(state);
    }
    return out;
  }

  void set_global_monitor_states(nb::list states) {
    std::vector<bool> restored(global_mons.size(), false);
    std::vector<std::pair<size_t, KMonState>> staged_states;
    bool has_bw_stream = false;
    const bool has_projection_window = projection_window_armed();
    for (auto item : states) {
      nb::dict d = nb::cast<nb::dict>(item);
      const bool bw = d.contains("bw_stream_config");
      const bool gathered_window = d.contains("coclock_window_config");
      const bool regional = d.contains("window_config");
      if (!bw && !gathered_window && !regional) {
        if (has_projection_window && dict_get_or<int>(d, "kind", 0) == KMON_PROJECTION) {
          const int slot = dict_get_or<int>(d, "slot", -1);
          if (std::any_of(global_mons.begin(), global_mons.end(), [&](const auto &m) {
                return m.projection_window() && m.slot == slot;
              }))
            throw std::runtime_error("global complete window projection: historical snapshot monitor");
        }
        continue;
      }
      const char *label = bw ? "global BW stream: " : gathered_window ? gathered_window_label
                                                             : "global complete window: ";
      if ((int)bw + (int)gathered_window + (int)regional != 1)
        throw std::runtime_error(std::string(label) + "ambiguous snapshot identity");
      if (!d.contains("slot") || !PyLong_CheckExact(d["slot"].ptr()) ||
          !d.contains("kind") || !PyLong_CheckExact(d["kind"].ptr()))
        throw std::runtime_error(std::string(label) + "invalid snapshot identity");
      const int slot = nb::cast<int>(d["slot"]);
      const int kind = nb::cast<int>(d["kind"]);
      auto it = std::find_if(global_mons.begin(), global_mons.end(),
                             [&](const auto &m) {
                               return (bw ? m.bw_stream() : gathered_window ? m.gathered_window
                                             : (m.complete_window && !m.gathered_window)) &&
                                      m.slot == slot && m.kind == kind;
                             });
      if (it == global_mons.end())
        throw std::runtime_error(std::string(label) + "unmatched snapshot monitor");
      const size_t i = (size_t)(it - global_mons.begin());
      if (bw) validate_bw_live_config(*it);
      const nb::tuple config = bw ? bw_stream_config(*it)
                                  : gathered_window ? gathered_window_config(*it)
                                            : window_config(*it, window_source_config());
      const char *config_key = bw ? "bw_stream_config"
                                  : gathered_window ? "coclock_window_config" : "window_config";
      const bool equal = equal_checked_window_config(d[config_key].ptr(), config.ptr());
      if (restored[i] || !equal)
        throw std::runtime_error(std::string(label) + "incompatible snapshot configuration");
      KMonState st = dict_to_checked_window_state(d, bw ? 0 : it->accum.size(), label);
      if (bw) {
        parse_bw_stream_state(st, d, (size_t)it->n_out_space * (size_t)W);
        it->validate_bw_regions(st, t_abs, "global BW stream");
        has_bw_stream = true;
      } else {
        it->validate_window_state(st, t_abs);
      }
      restored[i] = true;
      staged_states.emplace_back(i, std::move(st));
    }
    for (size_t i = 0; i < global_mons.size(); ++i) {
      if (!global_mons[i].staged_global() || restored[i]) continue;
      throw std::runtime_error(global_mons[i].bw_stream()
          ? "global BW stream: missing compatible snapshot monitor"
          : global_mons[i].gathered_window
              ? "global gathered complete window: missing compatible snapshot monitor"
              : "global complete window: missing compatible snapshot monitor");
    }

    // With any staged attachment, validate the entire set before committing any state.
    // Keep strict matching, validation, and output clearing out of the historical restore path.
    const bool strict_siblings = std::any_of(global_mons.begin(), global_mons.end(),
                                             [](const auto &m) { return m.staged_global(); });
    const char *sibling_label = has_bw_stream ? "global BW stream: "
                               : gathered_window_armed() ? gathered_window_label
                               : has_projection_window ? "global complete window projection: "
                                                       : "global complete window: ";
    for (size_t mi = 0; mi < states.size(); mi++) {
      nb::dict d = nb::cast<nb::dict>(states[mi]);
      if (d.contains("window_config") || d.contains("bw_stream_config") ||
          d.contains("coclock_window_config")) continue;
      if (strict_siblings) {
        for (const char *key : {"slot", "kind", "width", "n_out_v", "n_out_space", "n_out_modes"})
          if (!d.contains(key) || d[key].is_none())
            throw std::runtime_error(std::string(sibling_label) + "incomplete sibling snapshot identity or geometry");
      }
      KMonState st = strict_siblings ? dict_to_legacy_sibling_state(d, sibling_label)
                                     : dict_to_kmon_state(d);
      const auto matches = [&](const auto &m) {
        return !m.staged_global() && m.slot == st.slot && m.kind == st.kind;
      };
      auto it = std::find_if(global_mons.begin(), global_mons.end(), matches);
      if (strict_siblings) {
        if (it == global_mons.end() || std::find_if(std::next(it), global_mons.end(), matches) != global_mons.end())
          throw std::runtime_error(std::string(sibling_label) + "unmatched or ambiguous sibling snapshot monitor");
        const size_t i = (size_t)(it - global_mons.begin());
        if (restored[i])
          throw std::runtime_error(std::string(sibling_label) + "duplicate sibling snapshot monitor");
        it->validate_legacy_sibling_state(st, sibling_label);
        restored[i] = true;
        staged_states.emplace_back(i, std::move(st));
      } else {
        if (it == global_mons.end() && mi < global_mons.size() && !global_mons[mi].staged_global())
          it = global_mons.begin() + mi;
        if (it != global_mons.end()) it->import_state(st);
      }
    }
    if (strict_siblings && std::find(restored.begin(), restored.end(), false) != restored.end())
      throw std::runtime_error(std::string(sibling_label) + "missing sibling snapshot monitor");
    for (auto &[i, st] : staged_states)
      global_mons[i].commit_validated_state(st);
  }

  void set_stim_specs(int si, nb::list specs) {
    if (si < 0 || (size_t)si >= sn.size())
      throw std::runtime_error("bad subnet index in set_stim_specs");
    stim_specs.erase(
        std::remove_if(stim_specs.begin(), stim_specs.end(),
                       [si](const stim_spec<W> &s) {
                         return s.subnet_idx == si;
                       }),
        stim_specs.end());
    for (auto item : specs) {
      nb::dict d = nb::cast<nb::dict>(item);
      stim_spec<W> sp;
      sp.subnet_idx = si;
      sp.n_node = dict_get_or<int>(d, "n_node", sn[si].n_node);
      if (d.contains("target_cvars") && !d["target_cvars"].is_none())
        sp.target_cvars = to_int_vec(d["target_cvars"]);
      if (d.contains("spatial") && !d["spatial"].is_none()) {
        sp.spatial = to_float_vec(d["spatial"]);
        if (d.contains("lane_varying") && !d["lane_varying"].is_none()) {
          sp.lane_varying = nb::cast<bool>(d["lane_varying"]);
        } else {
          sp.lane_varying =
              (W > 1 && sp.spatial.size() == (size_t)sp.n_node * (size_t)W);
        }
      }
      sp.eq_kind = dict_get_or<int>(d, "eq_kind", STIM_DISCRETE_1D);
      sp.onset_t0 = dict_get_or<double>(d, "onset_t0", 0.0);
      sp.dt0 = dict_get_or<float>(d, "dt0", dt);
      if (d.contains("eq_params") && !d["eq_params"].is_none()) {
        std::vector<double> ep = to_double_vec(d["eq_params"]);
        for (size_t k = 0; k < ep.size() && k < sp.eq_params.size(); k++)
          sp.eq_params[k] = ep[k];
      }
      if (d.contains("temporal_1d") && !d["temporal_1d"].is_none())
        sp.temporal_1d = to_float_vec(d["temporal_1d"]);
      sp.step_offset = dict_get_or<int64_t>(d, "step_offset", 0);
      sp.scale = dict_get_or<float>(d, "scale", 1.0f);
      if (d.contains("lane_scale") && !d["lane_scale"].is_none())
        sp.lane_scale = to_float_vec(d["lane_scale"]);
      stim_specs.push_back(std::move(sp));
    }
  }

  void clear_stim_specs() { stim_specs.clear(); }

  void set_subnet_rng(int si, nb::object lane_keys_obj, nb::object gfun_obj,
                      double ntau = 0.0, bool is_multiplicative = false,
                      nb::object mul_a_obj = nb::none(),
                      nb::object mul_b_obj = nb::none()) {
    if (si < 0 || (size_t)si >= sn.size())
      throw std::runtime_error("bad subnet index in set_subnet_rng");
    if (sub_rng.size() < sn.size()) sub_rng.resize(sn.size());
    if (rng_scratch.size() < sn.size()) rng_scratch.resize(sn.size());
    auto &eng = sub_rng[si];
    eng.subnet_id = (uint32_t)si;
    eng.dt = sn[si].dt;
    eng.ntau = (float)ntau;
    eng.step_counter = 0;
    eng.ou_state.clear();
    if (lane_keys_obj.is_none()) {
      eng.enabled = false;
      return;
    }
    if (nb::isinstance<nb::int_>(lane_keys_obj)) {
      eng.set_uniform_seed(nb::cast<uint64_t>(lane_keys_obj));
    } else {
      std::vector<uint64_t> seeds = to_u64_vec(lane_keys_obj);
      if (seeds.empty()) {
        eng.enabled = false;
        return;
      }
      eng.set_lane_seeds(seeds.data(), seeds.size());
    }
    eng.gfun = to_float_vec(gfun_obj);
    eng.is_multiplicative = is_multiplicative;
    eng.mul_a = mul_a_obj.is_none() ? 1.0f : nb::cast<float>(mul_a_obj);
    eng.mul_b = mul_b_obj.is_none() ? 0.0f : nb::cast<float>(mul_b_obj);
    eng.enabled = true;
  }

  nb::list get_rng_states() const {
    nb::list out;
    for (size_t si = 0; si < sn.size(); si++) {
      nb::dict d;
      if (si < sub_rng.size()) {
        const auto &eng = sub_rng[si];
        auto st = eng.export_state();
        d["enabled"] = eng.enabled;
        auto lk = make_owned_array<uint64_t>({(size_t)W});
        for (int w = 0; w < W; w++) lk.data()[w] = st.lane_keys[(size_t)w];
        d["lane_keys"] = lk;
        d["step_counter"] = st.step_counter;
        auto ou = make_owned_array<float>({st.ou_state.size()});
        if (!st.ou_state.empty())
          std::memcpy(ou.data(), st.ou_state.data(),
                      st.ou_state.size() * sizeof(float));
        d["ou_state"] = ou;
      } else {
        d["enabled"] = false;
      }
      out.append(d);
    }
    return out;
  }

  void set_rng_states(nb::list states) {
    if (sub_rng.size() < sn.size()) sub_rng.resize(sn.size());
    const size_t n = std::min(states.size(), sub_rng.size());
    for (size_t si = 0; si < n; si++) {
      nb::dict d = nb::cast<nb::dict>(states[si]);
      sub_rng[si].enabled = dict_get_or<bool>(d, "enabled", false);
      if (d.contains("lane_keys") && !d["lane_keys"].is_none()) {
        rng_state<W> st;
        std::vector<uint64_t> lk = to_u64_vec(d["lane_keys"]);
        for (int w = 0; w < W && (size_t)w < lk.size(); w++)
          st.lane_keys[(size_t)w] = lk[(size_t)w];
        st.step_counter = dict_get_or<uint64_t>(d, "step_counter", 0);
        if (d.contains("ou_state") && !d["ou_state"].is_none())
          st.ou_state = to_float_vec(d["ou_state"]);
        sub_rng[si].import_state(st);
      }
    }
  }

  void enable_online_features(int si, double dt_sample = 1.0) {
    if (si < 0 || (size_t)si >= sn.size())
      throw std::runtime_error("bad subnet index in enable_online_features");
    if (online_enabled.size() < sn.size()) online_enabled.resize(sn.size(), 0);
    if (online_accums.size() < sn.size()) online_accums.resize(sn.size());
    online_accums[si].reset((int)voi_specs[si].size(), sn[si].n_node,
                            dt_sample);
    online_enabled[si] = 1;
  }

  void disable_online_features() {
    std::fill(online_enabled.begin(), online_enabled.end(), 0);
  }

  nb::dict take_online_features(int si, const std::vector<std::string> &names,
                                int order = 4) const {
    if (si < 0 || (size_t)si >= online_accums.size())
      throw std::runtime_error("bad subnet index in take_online_features");
    const auto &acc = online_accums[si];
    nb::dict out;
    for (const auto &name : names) {
      const int n_comp = OnlineFeatureAccum<W>::feature_components(name, order);
      std::vector<size_t> shape;
      if (name == "moments") {
        shape = {(size_t)n_comp, (size_t)acc.n_voi, (size_t)acc.n_node,
                 (size_t)W};
      } else {
        shape = {(size_t)acc.n_voi, (size_t)acc.n_node, (size_t)W};
      }
      auto arr = make_owned_array<double>(shape);
      acc.finalize_into(name, arr.data(), order);
      out[nb::str(name.c_str())] = arr;
    }
    return out;
  }

  void monitor_begin_run() {
    if (t_abs != 0 && gathered_window_armed())
      throw std::runtime_error("global gathered complete window: fresh origin must be zero");
    for (auto &mons_s : mons)
      for (auto &m : mons_s) m.begin_run(t_abs);
    for (auto &gm : global_mons) gm.begin_run(t_abs);
  }

  void monitor_begin_call() {
    for (auto &mons_s : mons)
      for (auto &m : mons_s) m.begin_call();
    for (auto &gm : global_mons) gm.begin_call();
  }

  // Run nstep steps. Output per chunk of `chunk_size` steps:
  // tavg (n_chunks, n_voi, n_node, W) or None when !need_tavg,
  // ctavg (n_chunks, n_cvar, n_node, n_modes * W) or None when !need_ctavg,
  // followed by per-subnet monitor engine streams (times, data).
  nb::list run(int nstep, int chunk_size, nb::object noise_obj,
               nb::object stim_obj, bool need_tavg = true,
               bool need_ctavg = true) {
    for (const auto &gm : global_mons) {
      if (!gm.staged_global()) continue;
      if (gm.bw_stream()) validate_bw_live_config(gm);
      if (gm.gathered_window && (gm.run_start != 0 || gm.accum_count != t_abs % (uint64_t)gm.istep || nstep < 0))
        throw std::runtime_error("global gathered complete window: reset or restore at the simulator clock before run");
      if (gm.m_step != t_abs)
        throw std::runtime_error(gm.bw_stream()
            ? "global BW stream: reset or restore at the simulator clock before run"
            : "global complete window: reset or restore at the simulator clock before run");
      if (nstep > 0 && (uint64_t)nstep > std::numeric_limits<uint64_t>::max() - t_abs)
        throw std::runtime_error(gm.bw_stream() ? "global BW stream: run exceeds remaining tick range"
                                              : "global complete window: run exceeds remaining tick range");
    }
    for (size_t si = 0; si < mons.size(); ++si) {
      for (const auto &m : mons[si]) {
        if (!m.local_window) continue;
        validate_local_window_source(si, m);
        const uint64_t expected_count = m.bw_local()
            ? m.m_step % (uint64_t)m.istep / (uint64_t)m.bw.drive_steps *
                  (uint64_t)kmon<W>::BwWindowState::substeps
            : m.m_step % (uint64_t)m.istep;
        if (m.run_start != 0 || m.m_step != t_abs / m.cadence_K ||
            m.accum_count != expected_count ||
            (m.bw_local() && m.bw.drive_count != m.m_step % (uint64_t)m.bw.drive_steps))
          throw std::runtime_error("Recorder complete window: reset or restore at the simulator clock before run");
        if (nstep < 0 || (uint64_t)nstep > std::numeric_limits<uint64_t>::max() - t_abs)
          throw std::runtime_error("Recorder complete window: run exceeds tick range");
      }
    }
    if (nstep < 0)
      throw std::invalid_argument("nstep must be non-negative");
    const int cs = chunk_size > 0 ? chunk_size : tavg_period;
    const int rem = nstep % cs;
    const int n_chunks = nstep / cs + (rem > 0);
    const size_t n_sn = sn.size();
    monitor_begin_call();
    for (size_t s = 0; s < mons.size(); s++) {
      for (auto &mon : mons[s]) {
        if (mon.local_window) mon.reserve_local_window(t_abs, t_abs + (uint64_t)nstep);
        else mon.reserve_for_steps(nstep);
      }
    }
    for (auto &gm : global_mons) gm.reserve_for_steps(nstep);

    // Ensure per-subnet scratch buffers are sized
    if (obs_row.size() < n_sn) obs_row.resize(n_sn);
    if (rng_scratch.size() < n_sn) rng_scratch.resize(n_sn);
    for (size_t s = 0; s < n_sn; s++) {
      const size_t need_obs =
          (size_t)voi_specs[s].size() * (size_t)sn[s].n_node * (size_t)W;
      if (obs_row[s].size() < need_obs) obs_row[s].resize(need_obs);
    }
    std::vector<const float *> global_src_rows(n_sn, nullptr);
    if (!global_mons.empty()) {
      if (global_gather.size() != n_sn)
        throw std::runtime_error(
            "simw::run: global_gather size != subnet count");
      const size_t gsz =
          (size_t)std::max(0, global_total_vois) *
          (size_t)std::max(0, global_total_nodes) * (size_t)W;
      if (global_obs_row.size() < gsz) global_obs_row.resize(gsz);
    }

    std::vector<nb::ndarray<nb::numpy, float>> keep_alive;
    auto resolve_per_sn = [&](nb::object obj) -> std::vector<const float *> {
      std::vector<const float *> ps(n_sn, nullptr);
      if (obj.is_none()) return ps;
      if (nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj)) {
        auto seq = nb::cast<nb::sequence>(obj);
        size_t idx = 0;
        for (auto item : seq) {
          if (idx >= n_sn)
            throw std::runtime_error(
                "noise/stim sequence longer than number of subnetworks");
          if (!item.is_none()) {
            keep_alive.push_back(
                nb::cast<nb::ndarray<nb::numpy, float>>(item));
            ps[idx] = keep_alive.back().data();
          }
          idx++;
        }
      } else {
        keep_alive.push_back(nb::cast<nb::ndarray<nb::numpy, float>>(obj));
        const float *d = keep_alive.back().data();
        for (size_t s = 0; s < n_sn; s++) ps[s] = d;
      }
      return ps;
    };
    const std::vector<const float *> noise_ps = resolve_per_sn(noise_obj);
    const std::vector<const float *> stim_ps = resolve_per_sn(stim_obj);

    std::vector<std::vector<float>> tacc(n_sn), cacc(n_sn);
    std::vector<std::vector<int>> counts(n_sn);
    for (size_t s = 0; s < n_sn; s++) {
      const int n_voi = (int)voi_specs[s].size();
      if (need_tavg) {
        tacc[s].assign((size_t)n_chunks * n_voi * sn[s].n_node * W, 0.f);
      }
      if (need_ctavg) {
        cacc[s].assign((size_t)n_chunks * sn[s].n_cvar * sn[s].n_node *
                           sn[s].n_modes * W,
                       0.f);
      }
      counts[s].assign(n_chunks, 0);
    }

    std::exception_ptr pend = nullptr;
    {
      nb::gil_scoped_release nogil;
      try {
        for (int step = 0; step < nstep; step++) {
          const size_t chunk = (size_t)(step / cs);
          for (auto &sub : sn) std::fill(sub.c.begin(), sub.c.end(), 0.f);
          for (auto &p : pr) p.apply(sn[p.src_sn], sn[p.tgt_sn], t_abs);

          for (size_t s = 0; s < n_sn; s++) {
            auto &sub = sn[s];
            counts[s][chunk]++;

            if (const float *stim = stim_ps[s]) {
              apply_legacy_dense_stim<W>(sub.c.data(), stim, step, sub.n_cvar,
                                         sub.n_node, sub.n_modes);
            }
            if (!stim_specs.empty()) {
              apply_subnet_stims<W>(stim_specs, (int)s, sub.c.data(),
                                    sub.n_modes, (int64_t)t_abs + 1,
                                    (double)(t_abs + 1u) * (double)dt);
            }

            if (need_ctavg) {
              const size_t n_cv =
                  (size_t)sub.n_cvar * sub.n_node * sub.n_modes * W;
              const float *cv = sub.c.data();
              float *dst = cacc[s].data() + chunk * n_cv;
#pragma omp simd
              for (size_t k = 0; k < n_cv; k++) dst[k] += cv[k];
            }

            if (((t_abs + 1) % sub.K) == 0) {
              const float *nz = nullptr;
              if (const float *nptr = noise_ps[s]) {
                nz = nptr + ((size_t)step) * (size_t)sub.n_svar * sub.n_node *
                                sub.n_modes * W;
              } else if (s < sub_rng.size() && sub_rng[s].enabled) {
                const size_t need_nz =
                    (size_t)sub.n_svar * sub.n_node * sub.n_modes * W;
                if (rng_scratch[s].size() < need_nz)
                  rng_scratch[s].resize(need_nz);
                sub_rng[s].fill_next_step_noise(rng_scratch[s].data(),
                                                sub.n_svar, sub.n_node,
                                                sub.n_modes, sub.x.data());
                nz = rng_scratch[s].data();
              }
              const int sub_integ = (sub.integ_id >= 0) ? sub.integ_id : integ_id;
              if (sub_integ == 1)
                sub.step_heun(sub.dt, nz);
              else
                sub.step_euler(sub.dt, nz);
              sub.push_state((int)((((t_abs + 1) / sub.K) - 1) & (sub.H - 1)));
            }

            const int n_voi = (int)voi_specs[s].size();
            const bool has_sn_mons = (s < mons.size() && !mons[s].empty());
            const bool has_online =
                (s < online_enabled.size() && online_enabled[s] != 0);
            const bool need_obs =
                need_tavg || has_sn_mons || !global_mons.empty() || has_online;
            if (need_obs && n_voi > 0) {
              observe_voi_row<W>(sub.x.data(), sub.n_node, sub.n_modes,
                                 voi_specs[s], obs_row[s].data());
            }
            if (need_tavg && n_voi > 0) {
              const size_t ntv = (size_t)n_voi * sub.n_node * W;
              float *dst = tacc[s].data() + chunk * ntv;
              const float *src_obs = obs_row[s].data();
#pragma omp simd
              for (size_t k = 0; k < ntv; k++) dst[k] += src_obs[k];
            }
            if (has_sn_mons) {
              for (auto &mon : mons[s]) {
                if (mon.kind == KMON_CTAVG) {
                  mon.step_one(sub.c.data(), sub.n_cvar, sub.n_node,
                               sub.n_modes, t_abs);
                } else {
                  mon.step_one(obs_row[s].data(), n_voi, sub.n_node, 1, t_abs);
                }
              }
            }
            if (has_online) {
              online_accums[s].update(obs_row[s].data());
            }
          }

          if (!global_mons.empty()) {
            // All rows reflect current state, including initial/held state on ineligible source ticks.
            for (size_t s = 0; s < n_sn; s++)
              global_src_rows[s] = obs_row[s].data();
            gather_global_voi_row<W>(global_src_rows, global_gather,
                                     global_total_vois, global_total_nodes,
                                     global_obs_row.data());
            for (auto &gm : global_mons) {
              gm.step_one(global_obs_row.data(), global_total_vois,
                          global_total_nodes, 1, t_abs);
            }
          }
          t_abs++;
        }
      } catch (...) {
        pend = std::current_exception();
      }
    }
    if (pend) std::rethrow_exception(pend);

    if (nstep > 0 && rem != 0) {
      const size_t chunk = (size_t)(nstep / cs);
      for (size_t s = 0; s < n_sn; s++) {
        counts[s][chunk] = rem;
      }
    }

    nb::list outs;
    for (size_t s = 0; s < n_sn; s++) {
      const int n_voi = (int)voi_specs[s].size();
      if (need_tavg) {
        nb::ndarray<nb::numpy, float> tarr = make_owned_array<float>(
            {(size_t)n_chunks, (size_t)n_voi, (size_t)sn[s].n_node, (size_t)W});
        float *td = tarr.data();
        const size_t ntv = (size_t)n_voi * sn[s].n_node * W;
        for (size_t chunk = 0; chunk < (size_t)n_chunks; chunk++) {
          const float tinv =
              1.f / (float)(counts[s][chunk] ? counts[s][chunk] : 1);
#pragma omp simd
          for (size_t k = 0; k < ntv; k++)
            td[chunk * ntv + k] = tacc[s][chunk * ntv + k] * tinv;
        }
        outs.append(tarr);
      } else {
        outs.append(nb::none());
      }

      if (need_ctavg) {
        nb::ndarray<nb::numpy, float> carr = make_owned_array<float>(
            {(size_t)n_chunks, (size_t)sn[s].n_cvar, (size_t)sn[s].n_node,
             (size_t)sn[s].n_modes * W});
        float *cd = carr.data();
        const size_t ncv =
            (size_t)sn[s].n_cvar * sn[s].n_node * sn[s].n_modes * W;
        for (size_t chunk = 0; chunk < (size_t)n_chunks; chunk++) {
          const float cinv =
              1.f / (float)(counts[s][chunk] ? counts[s][chunk] : 1);
#pragma omp simd
          for (size_t k = 0; k < ncv; k++)
            cd[chunk * ncv + k] = cacc[s][chunk * ncv + k] * cinv;
        }
        outs.append(carr);
      } else {
        outs.append(nb::none());
      }

      if (mons.size() == n_sn) {
        for (auto &mon : mons[s]) {
          const size_t n = mon.t_out.size();
          nb::ndarray<nb::numpy, double> mt = make_owned_array<double>({n});
          if (n > 0)
            std::copy(mon.t_out.begin(), mon.t_out.end(),
                      static_cast<double *>(mt.data()));
          nb::ndarray<nb::numpy, float> md = make_owned_array<float>(
              {n, (size_t)mon.n_out_v, (size_t)mon.n_out_space,
               (size_t)mon.n_out_modes * (size_t)W});
          if (!mon.d_out.empty())
            std::copy(mon.d_out.begin(), mon.d_out.end(),
                      static_cast<float *>(md.data()));
          outs.append(mt);
          outs.append(md);
        }
      }
    }
    return outs;
  }
};

// ---- width-erasing wrapper (W in {1, 4, 8, 16}) -----------------------------

struct sim {
  int width = 8;
  std::unique_ptr<simw<1>> s1;
  std::unique_ptr<simw<4>> s4;
  std::unique_ptr<simw<8>> s8;
  std::unique_ptr<simw<16>> s16;

  explicit sim(int width_ = 8) : width(width_) {
    if (width_ != 1 && width_ != 4 && width_ != 8 && width_ != 16) {
      throw std::invalid_argument(
          "Sim width must be one of {1, 4, 8, 16}, got " +
          std::to_string(width_));
    }
  }

  void ensure_sim() {
    switch (width) {
      case 1:  if (!s1)  s1  = std::make_unique<simw<1>>();  break;
      case 4:  if (!s4)  s4  = std::make_unique<simw<4>>();  break;
      case 8:  if (!s8)  s8  = std::make_unique<simw<8>>();  break;
      case 16: if (!s16) s16 = std::make_unique<simw<16>>(); break;
      default: throw std::invalid_argument("unsupported Sim width");
    }
  }

  template <typename F> void dispatch(F &&f) {
    switch (width) {
      case 1:  if (!s1)  throw std::runtime_error("sim not configured"); f(*s1);  break;
      case 4:  if (!s4)  throw std::runtime_error("sim not configured"); f(*s4);  break;
      case 8:  if (!s8)  throw std::runtime_error("sim not configured"); f(*s8);  break;
      case 16: if (!s16) throw std::runtime_error("sim not configured"); f(*s16); break;
      default: throw std::invalid_argument("unsupported Sim width");
    }
  }

  template <typename F> void dispatch(F &&f) const {
    switch (width) {
      case 1:  if (!s1)  throw std::runtime_error("sim not configured"); f(*s1);  break;
      case 4:  if (!s4)  throw std::runtime_error("sim not configured"); f(*s4);  break;
      case 8:  if (!s8)  throw std::runtime_error("sim not configured"); f(*s8);  break;
      case 16: if (!s16) throw std::runtime_error("sim not configured"); f(*s16); break;
      default: throw std::invalid_argument("unsupported Sim width");
    }
  }

  void add_subnet(int n_node, int n_svar, int n_parm, int n_cvar,
                  int model_id, int horizon, int n_modes, uint32_t k = 1,
                  float dt = 0.001f, int integ_id = -1, uint64_t generic_dfun = 0) {
    uint32_t H = 1;
    while (H < (uint32_t)(horizon + 2)) H <<= 1;
    ensure_sim();
    dispatch([&](auto &sw) {
      sw.require_no_staged_global_armed("adding a subnet");
      constexpr int W = std::remove_reference_t<decltype(sw)>::W_CONST;
      subnet<W> s;
      s.alloc(n_node, n_svar, n_parm, n_cvar, model_id, H, n_modes);
      s.generic_fn = reinterpret_cast<cph_generic_fn_t>(generic_dfun);
      s.K = k;
      s.dt = dt;
      s.integ_id = integ_id;
      sw.sn.push_back(std::move(s));
      sw.voi_specs.push_back({});
      sw.obs_row.push_back({});
      sw.mons.push_back({});
      sw.sub_rng.push_back(rng_engine<W>{});
      sw.rng_scratch.push_back({});
      sw.online_enabled.push_back(0);
      sw.online_accums.push_back(OnlineFeatureAccum<W>{});
    });
  }

  void add_projection(int src_sn, int tgt_sn, nb::ndarray<nb::numpy, float> w,
                      nb::ndarray<nb::numpy, uint32_t> idx,
                      nb::ndarray<nb::numpy, uint32_t> ptr,
                      nb::ndarray<nb::numpy, uint32_t> del, int cfun_id,
                      nb::ndarray<nb::numpy, int> src_cvars,
                      int tgt_cvar, int n_cfun_parm, float scale,
                      float ts, int tgt_state_cvar,
                      nb::object mode_map_obj) {
    ensure_sim();
    dispatch([&](auto &sw) {
      constexpr int W = std::remove_reference_t<decltype(sw)>::W_CONST;
      proj<W> p;
      p.src_sn = src_sn;
      p.tgt_sn = tgt_sn;
      _fill_proj(p, w, idx, ptr, del, cfun_id, src_cvars, tgt_cvar,
                 n_cfun_parm, scale, ts, tgt_state_cvar, mode_map_obj);
      sw.pr.push_back(std::move(p));
    });
  }

  template <int W>
  void _fill_proj(proj<W> &p, nb::ndarray<nb::numpy, float> w,
                  nb::ndarray<nb::numpy, uint32_t> idx,
                  nb::ndarray<nb::numpy, uint32_t> ptr,
                  nb::ndarray<nb::numpy, uint32_t> del, int cfun_id,
                  nb::ndarray<nb::numpy, int> src_cvars,
                  int tgt_cvar, int n_cfun_parm, float scale,
                  float ts, int tgt_state_cvar,
                  nb::object mode_map_obj) {
    p.ts = ts;
    p.tgt_state_cvar = tgt_state_cvar;
    p.has_pre = (cfun_id >= 3 && cfun_id <= 9);
    if (mode_map_obj.is_none()) {
      p.mode_map.assign(1, 1.0f);
    } else {
      nb::ndarray<nb::numpy, float> mm =
          nb::cast<nb::ndarray<nb::numpy, float>>(mode_map_obj);
      auto M = mm.view<nb::ndim<2>>();
      p.mode_map.resize((size_t)M.shape(0) * M.shape(1));
      for (size_t r = 0; r < M.shape(0); r++)
        for (size_t cq = 0; cq < M.shape(1); cq++)
          p.mode_map[r * M.shape(1) + cq] = M(r, cq);
    }
    p.scale = scale;
    p.cfun_id = cfun_id;
    auto S = src_cvars.view<nb::ndim<1>>();
    for (size_t k = 0; k < S.shape(0); k++) p.src_cvars.push_back(S(k));
    p.tgt_cvar = tgt_cvar;
    p.n_cfun_parm = n_cfun_parm;
    auto W_ = w.view<nb::ndim<1>>();
    auto I_ = idx.view<nb::ndim<1>>();
    auto T_ = ptr.view<nb::ndim<1>>();
    auto D_ = del.view<nb::ndim<1>>();
    p.w.resize(W_.shape(0));
    for (size_t k = 0; k < W_.shape(0); k++) p.w[k] = W_(k);
    p.idx.resize(I_.shape(0));
    for (size_t k = 0; k < I_.shape(0); k++) p.idx[k] = I_(k);
    p.ptr.resize(T_.shape(0));
    for (size_t k = 0; k < T_.shape(0); k++) p.ptr[k] = T_(k);
    p.del.resize(D_.shape(0));
    for (size_t k = 0; k < D_.shape(0); k++) p.del[k] = D_(k);
    p.cfp.assign((size_t)n_cfun_parm * W, 0.f);
  }

  void set_subnet_params(int si, nb::ndarray<nb::numpy, float> v) {
    dispatch([&](auto &s) {
      auto &sn = s.sn[si];
      auto V = v.view<nb::ndim<3>>();
      if ((int)V.shape(0) != sn.n_node || (int)V.shape(1) != sn.n_parm ||
          (int)V.shape(2) != (int)std::remove_reference_t<decltype(s)>::W_CONST)
        throw std::runtime_error("bad params shape");
      const int W = std::remove_reference_t<decltype(s)>::W_CONST;
      for (int j = 0; j < sn.n_node; j++)
        for (int k = 0; k < sn.n_parm; k++)
          for (int i = 0; i < W; i++)
            sn.p[((size_t)j * sn.n_parm + k) * W + i] = V(j, k, i);
    });
  }

  void set_subnet_state(int si, nb::ndarray<nb::numpy, float> v) {
    dispatch([&](auto &s) {
      auto &sn = s.sn[si];
      const int W = std::remove_reference_t<decltype(s)>::W_CONST;
      auto V = v.view<nb::ndim<4>>();
      if ((int)V.shape(0) != sn.n_svar || (int)V.shape(1) != sn.n_node ||
          (int)V.shape(2) != sn.n_modes || (int)V.shape(3) != W)
        throw std::runtime_error("bad state shape");
      std::memcpy(sn.x.data(), v.data(), sn.x.size() * sizeof(float));
      const size_t rows = (size_t)sn.n_svar * sn.n_node * sn.n_modes;
      for (size_t row = 0; row < rows; row++) {
        const size_t svar = row / ((size_t)sn.n_node * sn.n_modes);
        const size_t rem = row % ((size_t)sn.n_node * sn.n_modes);
        for (uint32_t hh = 0; hh < sn.H; hh++) {
          float *dst = sn.buf.data() +
              ((svar * (size_t)sn.n_node * sn.n_modes + rem) * sn.H + hh) * W;
          const float *srcv = sn.x.data() + row * W;
#pragma omp simd
          for (int i = 0; i < W; i++) dst[i] = srcv[i];
        }
      }
      s.t_abs = 0;
    });
  }

  uint64_t t_abs() const {
    if (s1) return s1->t_abs;
    if (s4) return s4->t_abs;
    if (s8) return s8->t_abs;
    if (s16) return s16->t_abs;
    return 0;
  }

  void set_t_abs(uint64_t t) {
    dispatch([&](auto &s) { s.t_abs = t; });
  }

  nb::ndarray<nb::numpy, float> get_subnet_state(int si) {
    nb::ndarray<nb::numpy, float> out;
    dispatch([&](auto &s) {
      auto &sn = s.sn[si];
      const int W = std::remove_reference_t<decltype(s)>::W_CONST;
      out = make_owned_array<float>({(size_t)sn.n_svar, (size_t)sn.n_node,
                                     (size_t)sn.n_modes, (size_t)W});
      std::memcpy(out.data(), sn.x.data(), sn.x.size() * sizeof(float));
    });
    return out;
  }

  nb::ndarray<nb::numpy, float> get_subnet_buf(int si) {
    nb::ndarray<nb::numpy, float> out;
    dispatch([&](auto &s) {
      if (si < 0 || (size_t)si >= s.sn.size())
        throw std::runtime_error("bad subnet index");
      auto &sn = s.sn[si];
      const int W = std::remove_reference_t<decltype(s)>::W_CONST;
      out = make_owned_array<float>({(size_t)sn.n_svar, (size_t)sn.n_node,
                                     (size_t)sn.n_modes, (size_t)sn.H,
                                     (size_t)W});
      std::memcpy(out.data(), sn.buf.data(), sn.buf.size() * sizeof(float));
    });
    return out;
  }

  void set_subnet_buf(int si, nb::ndarray<nb::numpy, float> v) {
    dispatch([&](auto &s) {
      if (si < 0 || (size_t)si >= s.sn.size())
        throw std::runtime_error("bad subnet index");
      auto &sn = s.sn[si];
      const int W = std::remove_reference_t<decltype(s)>::W_CONST;
      auto V = v.view<nb::ndim<5>>();
      if ((int)V.shape(0) != sn.n_svar || (int)V.shape(1) != sn.n_node ||
          (int)V.shape(2) != sn.n_modes || (uint32_t)V.shape(3) != sn.H ||
          (int)V.shape(4) != W)
        throw std::runtime_error("bad buffer shape");
      size_t pos = 0;
      for (int sv = 0; sv < sn.n_svar; sv++)
        for (int j = 0; j < sn.n_node; j++)
          for (int m = 0; m < sn.n_modes; m++)
            for (uint32_t h = 0; h < sn.H; h++)
              for (int i = 0; i < W; i++)
                sn.buf[pos++] = V(sv, j, m, h, i);
    });
  }

  void set_subnet_bounds(int si, nb::ndarray<nb::numpy, float> lo,
                         nb::ndarray<nb::numpy, float> hi) {
    dispatch([&](auto &s) {
      if (si < 0 || (size_t)si >= s.sn.size())
        throw std::runtime_error("bad subnet index");
      auto &sn = s.sn[si];
      auto L = lo.view<nb::ndim<1>>();
      auto H = hi.view<nb::ndim<1>>();
      if ((int)L.shape(0) != sn.n_svar || (int)H.shape(0) != sn.n_svar)
        throw std::runtime_error("bad subnet bounds shape");
      sn.sv_lo.resize(sn.n_svar);
      sn.sv_hi.resize(sn.n_svar);
      for (int k = 0; k < sn.n_svar; k++) {
        sn.sv_lo[k] = L(k);
        sn.sv_hi[k] = H(k);
      }
    });
  }

  void set_subnet_clamps(int si, nb::ndarray<nb::numpy, int> indices,
                         nb::ndarray<nb::numpy, float> values) {
    dispatch([&](auto &s) {
      if (si < 0 || (size_t)si >= s.sn.size())
        throw std::runtime_error("bad subnet index");
      auto &sn = s.sn[si];
      auto I = indices.view<nb::ndim<1>>();
      auto V = values.view<nb::ndim<3>>();
      const size_t nc = I.shape(0);
      if (V.shape(0) != nc || (int)V.shape(1) != sn.n_node ||
          (int)V.shape(2) != sn.n_modes)
        throw std::runtime_error("bad clamp shape");
      sn.clamp_idx.resize(nc);
      for (size_t c = 0; c < nc; c++) {
        const int sv = I(c);
        if (sv < 0 || sv >= sn.n_svar)
          throw std::runtime_error("clamp state variable index out of range");
        sn.clamp_idx[c] = sv;
      }
      sn.clamp_val.resize(nc * (size_t)sn.n_node * (size_t)sn.n_modes);
      for (size_t c = 0; c < nc; c++)
        for (int j = 0; j < sn.n_node; j++)
          for (int m = 0; m < sn.n_modes; m++)
            sn.clamp_val[(c * (size_t)sn.n_node + (size_t)j) *
                             (size_t)sn.n_modes +
                         (size_t)m] = V(c, j, m);
    });
  }

  void set_cfun_params(int pi, nb::ndarray<nb::numpy, float> v) {
    dispatch([&](auto &s) {
      auto &p = s.pr[pi];
      const int W = std::remove_reference_t<decltype(s)>::W_CONST;
      auto V = v.view<nb::ndim<2>>();
      if ((int)V.shape(1) != W) throw std::runtime_error("cfun lane dim");
      for (int k = 0; k < (int)V.shape(0); k++)
        for (int i = 0; i < W; i++) p.cfp[k * W + i] = V(k, i);
    });
  }

  void set_voi(int si, std::vector<int> v) {
    dispatch([&](auto &s) {
      std::vector<voispec> rows;
      for (int idx : v) rows.push_back({0, idx, 0});
      s.set_observed_rows(si, std::move(rows));
    });
  }

  void set_voi_specs(int si, std::vector<int> flat) {
    dispatch([&](auto &s) {
      std::vector<voispec> rows;
      for (size_t k = 0; k + 2 < flat.size(); k += 3)
        rows.push_back({flat[k], flat[k + 1], flat[k + 2]});
      s.set_observed_rows(si, std::move(rows));
    });
  }

  void set_opts(int integ_id, float dt, int tavg_period) {
    ensure_sim();
    dispatch([&](auto &s) {
      if (s.dt != dt) s.require_no_staged_global_armed("changing the clock");
      s.integ_id = integ_id;
      s.dt = dt;
      s.tavg_period = tavg_period;
    });
  }

  void set_monitors(int si, nb::list atts) {
    dispatch([&](auto &s) { s.set_monitors(si, atts); });
  }

  void set_global_monitors(nb::list gather_specs, nb::list mon_specs) {
    dispatch([&](auto &s) { s.set_global_monitors(gather_specs, mon_specs); });
  }

  nb::list take_global_monitor_outputs() {
    nb::list out;
    dispatch([&](auto &s) { out = s.take_global_monitor_outputs(); });
    return out;
  }

  nb::list get_monitor_states() const {
    nb::list out;
    dispatch([&](const auto &s) { out = s.get_monitor_states(); });
    return out;
  }

  void set_monitor_states(nb::list states) {
    dispatch([&](auto &s) { s.set_monitor_states(states); });
  }

  nb::list get_global_monitor_states() const {
    nb::list out;
    dispatch([&](const auto &s) { out = s.get_global_monitor_states(); });
    return out;
  }

  void set_global_monitor_states(nb::list states) {
    dispatch([&](auto &s) { s.set_global_monitor_states(states); });
  }

  void set_stim_specs(int si, nb::list specs) {
    dispatch([&](auto &s) { s.set_stim_specs(si, specs); });
  }

  void clear_stim_specs() {
    dispatch([&](auto &s) { s.clear_stim_specs(); });
  }

  void set_subnet_rng(int si, nb::object lane_keys, nb::object gfun,
                      double ntau = 0.0, bool is_multiplicative = false,
                      nb::object mul_a = nb::none(),
                      nb::object mul_b = nb::none()) {
    dispatch([&](auto &s) {
      s.set_subnet_rng(si, lane_keys, gfun, ntau, is_multiplicative, mul_a,
                       mul_b);
    });
  }

  nb::list get_rng_states() const {
    nb::list out;
    dispatch([&](const auto &s) { out = s.get_rng_states(); });
    return out;
  }

  void set_rng_states(nb::list states) {
    dispatch([&](auto &s) { s.set_rng_states(states); });
  }

  void enable_online_features(int si, double dt_sample = 1.0) {
    dispatch([&](auto &s) { s.enable_online_features(si, dt_sample); });
  }

  void disable_online_features() {
    dispatch([&](auto &s) { s.disable_online_features(); });
  }

  nb::dict take_online_features(int si, const std::vector<std::string> &names,
                                int order = 4) const {
    nb::dict out;
    dispatch([&](const auto &s) {
      out = s.take_online_features(si, names, order);
    });
    return out;
  }

  void monitor_begin_run() {
    dispatch([&](auto &s) { s.monitor_begin_run(); });
  }

  nb::list run(int nstep, int chunk_size, nb::object noise, nb::object stim,
               bool need_tavg = true, bool need_ctavg = true) {
    nb::list outs;
    dispatch([&](auto &s) {
      outs = s.run(nstep, chunk_size, noise, stim, need_tavg, need_ctavg);
    });
    return outs;
  }
};

// ---- Feature extraction bindings (cph_features.hpp) ------------------------

template <typename T>
static nb::dict compute_features_impl(
    nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> ts,
    const std::vector<std::string> &names, double fs, int bins, int order,
    nb::object lo, nb::object hi) {
  const int n_series = static_cast<int>(ts.shape(0));
  const int n_time = static_cast<int>(ts.shape(1));
  FeatureParams p;
  p.fs = fs;
  p.bins = bins;
  p.order = order;
  if (!lo.is_none() && !hi.is_none()) {
    p.has_band = true;
    p.lo = nb::cast<double>(lo);
    p.hi = nb::cast<double>(hi);
  }
  const T *ptr = ts.data();
  std::vector<BatchFeatureOutput> res;
  {
    nb::gil_scoped_release nogil;
    res = compute_batch_features<T>(ptr, n_series, n_time, names, p);
  }

  nb::dict out;
  for (const auto &bo : res) {
    if (bo.name == "psd_raw") {
      auto freqs_arr = make_owned_array<double>({(size_t)bo.n_freq});
      if (!bo.freqs.empty())
        std::memcpy(freqs_arr.data(), bo.freqs.data(),
                    bo.freqs.size() * sizeof(double));
      auto psd_arr =
          make_owned_array<double>({(size_t)n_series, (size_t)bo.n_freq});
      if (!bo.data.empty())
        std::memcpy(psd_arr.data(), bo.data.data(),
                    bo.data.size() * sizeof(double));
      out[nb::str(bo.name.c_str())] = nb::make_tuple(freqs_arr, psd_arr);
    } else if (bo.name == "moments" || bo.name == "spectrum_stats" ||
               bo.name == "spectrum_moments") {
      auto arr = make_owned_array<double>(
          {(size_t)bo.n_components, (size_t)n_series});
      if (!bo.data.empty())
        std::memcpy(arr.data(), bo.data.data(),
                    bo.data.size() * sizeof(double));
      out[nb::str(bo.name.c_str())] = arr;
    } else {
      auto arr = make_owned_array<double>({(size_t)n_series});
      if (!bo.data.empty())
        std::memcpy(arr.data(), bo.data.data(),
                    bo.data.size() * sizeof(double));
      out[nb::str(bo.name.c_str())] = arr;
    }
  }
  return out;
}

struct py_online_feature_accum {
  int width = 8;
  std::unique_ptr<OnlineFeatureAccum<1>> a1;
  std::unique_ptr<OnlineFeatureAccum<4>> a4;
  std::unique_ptr<OnlineFeatureAccum<8>> a8;
  std::unique_ptr<OnlineFeatureAccum<16>> a16;

  explicit py_online_feature_accum(int width_ = 8) : width(width_) {
    switch (width_) {
      case 1:  a1  = std::make_unique<OnlineFeatureAccum<1>>();  break;
      case 4:  a4  = std::make_unique<OnlineFeatureAccum<4>>();  break;
      case 8:  a8  = std::make_unique<OnlineFeatureAccum<8>>();  break;
      case 16: a16 = std::make_unique<OnlineFeatureAccum<16>>(); break;
      default:
        throw std::invalid_argument(
            "OnlineFeatureAccum width must be one of {1, 4, 8, 16}");
    }
  }

  template <typename F> void dispatch(F &&f) {
    switch (width) {
      case 1:  f(*a1);  break;
      case 4:  f(*a4);  break;
      case 8:  f(*a8);  break;
      case 16: f(*a16); break;
    }
  }

  template <typename F> void dispatch(F &&f) const {
    switch (width) {
      case 1:  f(*a1);  break;
      case 4:  f(*a4);  break;
      case 8:  f(*a8);  break;
      case 16: f(*a16); break;
    }
  }

  void reset(int n_voi, int n_node, double dt_sample = 1.0) {
    dispatch([&](auto &a) { a.reset(n_voi, n_node, dt_sample); });
  }

  template <typename T>
  void update_typed(nb::ndarray<nb::numpy, const T, nb::c_contig> obs_row) {
    dispatch([&](auto &a) {
      const size_t expected = a.num_series();
      if (obs_row.size() != expected)
        throw std::invalid_argument("OnlineFeatureAccum::update: size mismatch");
      a.update(obs_row.data());
    });
  }

  nb::dict finalize(const std::vector<std::string> &names,
                    int order = 4) const {
    nb::dict out;
    dispatch([&](const auto &a) {
      constexpr int W = std::remove_reference_t<decltype(a)>::WIDTH;
      for (const auto &name : names) {
        const int n_comp = OnlineFeatureAccum<W>::feature_components(name, order);
        std::vector<size_t> shape;
        if (name == "moments") {
          shape = {(size_t)n_comp, (size_t)a.n_voi, (size_t)a.n_node, (size_t)W};
        } else {
          shape = {(size_t)a.n_voi, (size_t)a.n_node, (size_t)W};
        }
        auto arr = make_owned_array<double>(shape);
        a.finalize_into(name, arr.data(), order);
        out[nb::str(name.c_str())] = arr;
      }
    });
    return out;
  }
};

// Model metadata for the dfuns compiled into this module at build time:
// class name -> {mid, n_parm, n_cvar, n_svar, parm_names}, i.e. the same
// shape dfungen's meta dict has.  Read straight from the generated table, so
// Python and the kernels cannot disagree about ids or parameter order.
// Empty when the extension was built without the generated TU.
static nb::dict generic_model_table() {
  nb::dict out;
#ifdef CPH_HAVE_BUILTIN_GEN
  const int n = cph_builtin_count();
  for (int i = 0; i < n; i++) {
    const cph_gen_entry *e = cph_builtin_entry(i);
    if (e == nullptr || e->name == nullptr) continue;
    nb::list parm;
    for (int k = 0; k < e->n_parm; k++)
      parm.append(nb::str(e->parm_names[k] ? e->parm_names[k] : ""));
    nb::dict rec;
    rec["mid"] = e->mid;
    rec["n_parm"] = e->n_parm;
    rec["n_cvar"] = e->n_cvar;
    rec["n_svar"] = e->n_svar;
    rec["parm_names"] = parm;
    rec["signature"] = nb::str(e->signature ? e->signature : "");
    out[nb::str(e->name)] = rec;
  }
#endif
  return out;
}

// debug helper: single MPR dfun evaluation (1 node, 1 lane)
std::vector<float> dbg_mpr_dfun(float r, float V, float c,
                                nb::ndarray<nb::numpy, float> parr) {
  float x[2] = {r, V}, dx[2] = {0.f, 0.f};
  auto P = parr.view<nb::ndim<1>>();
  float p[6];
  for (int k = 0; k < 6; k++) p[k] = P(k);
  float cc[1] = {c};
  dfun_mpr<1>(dx, x, 0, 0, 1, 1, cc, p, 6, 1);
  return {dx[0], dx[1]};
}

// debug helper: full heun step replicating subnet::step_heun (1 node)
std::vector<float> dbg_mpr_heun(float r, float V, float c,
                                nb::ndarray<nb::numpy, float> parr,
                                float dt) {
  float x[2] = {r, V}, dx[2] = {0.f, 0.f}, xi[2], dxi[2] = {0.f, 0.f};
  auto P = parr.view<nb::ndim<1>>();
  float p[6];
  for (int k = 0; k < 6; k++) p[k] = P(k);
  float cc[1] = {c};
  dfun_mpr<1>(dx, x, 0, 0, 1, 1, cc, p, 6, 1);
  for (int k = 0; k < 2; k++) xi[k] = x[k] + dt * dx[k];
  xi[0] = xi[0] * (xi[0] > 0.f);
  dfun_mpr<1>(dxi, xi, 0, 0, 1, 1, cc, p, 6, 1);
  for (int k = 0; k < 2; k++) x[k] += dt * 0.5f * (dx[k] + dxi[k]);
  x[0] = x[0] * (x[0] > 0.f);
  return {x[0], x[1], dx[0], dx[1], dxi[0], dxi[1]};
}

}  // namespace cph

NB_MODULE(_cpp_hybrid, m) {
  m.doc() = "C++ hybrid simulator core (runtime SIMD kernels, no codegen)";
  m.def("enable_generic_dfuns", [](nb::int_ fn) {
    cph::cph_generic_fn = reinterpret_cast<cph::cph_generic_fn_t>(
        static_cast<uint64_t>(fn));
  }, nb::arg("fn"));
  m.def("dbg_mpr_dfun", &cph::dbg_mpr_dfun);
  m.def("dbg_mpr_heun", &cph::dbg_mpr_heun);
  m.def("generic_model_table", &cph::generic_model_table);
  m.def("balloon_windkessel_nominal_params",
        &cph::balloon_windkessel_nominal_params);
  m.def("compute_features", &cph::compute_features_impl<float>,
        nb::arg("ts"), nb::arg("names"), nb::arg("fs") = 1.0,
        nb::arg("bins") = 20, nb::arg("order") = 4,
        nb::arg("lo") = nb::none(), nb::arg("hi") = nb::none());
  m.def("compute_features", &cph::compute_features_impl<double>,
        nb::arg("ts"), nb::arg("names"), nb::arg("fs") = 1.0,
        nb::arg("bins") = 20, nb::arg("order") = 4,
        nb::arg("lo") = nb::none(), nb::arg("hi") = nb::none());

  nb::class_<cph::py_online_feature_accum>(m, "OnlineFeatureAccum")
      .def(nb::init<int>(), nb::arg("width") = 8)
      .def("reset", &cph::py_online_feature_accum::reset,
           nb::arg("n_voi"), nb::arg("n_node"), nb::arg("dt_sample") = 1.0)
      .def("update", &cph::py_online_feature_accum::update_typed<float>,
           nb::arg("obs_row"))
      .def("update", &cph::py_online_feature_accum::update_typed<double>,
           nb::arg("obs_row"))
      .def("finalize", &cph::py_online_feature_accum::finalize,
           nb::arg("names"), nb::arg("order") = 4);

  nb::class_<cph::sim>(m, "Sim")
      .def(nb::init<int>(), nb::arg("width") = 8)
      .def("add_subnet", &cph::sim::add_subnet, nb::arg("n_node"),
           nb::arg("n_svar"), nb::arg("n_parm"), nb::arg("n_cvar"),
           nb::arg("model_id"), nb::arg("horizon"), nb::arg("n_modes"),
           nb::arg("k") = 1, nb::arg("dt") = 0.001f, nb::arg("integ_id") = -1,
           nb::arg("generic_dfun") = 0)
      .def("add_projection", &cph::sim::add_projection, nb::arg("src_sn"),
           nb::arg("tgt_sn"), nb::arg("w"), nb::arg("idx"), nb::arg("ptr"),
           nb::arg("del"), nb::arg("cfun_id"), nb::arg("src_cvars"),
           nb::arg("tgt_cvar"), nb::arg("n_cfun_parm"), nb::arg("scale"),
           nb::arg("ts") = 1.0f, nb::arg("tgt_state_cvar") = 0,
           nb::arg("mode_map") = nb::none())
      .def("set_subnet_params", &cph::sim::set_subnet_params)
      .def("set_subnet_state", &cph::sim::set_subnet_state)
      .def("set_subnet_bounds", &cph::sim::set_subnet_bounds, nb::arg("si"),
           nb::arg("lo"), nb::arg("hi"))
      .def("set_subnet_clamps", &cph::sim::set_subnet_clamps, nb::arg("si"),
           nb::arg("indices"), nb::arg("values"))
      .def("set_cfun_params", &cph::sim::set_cfun_params)
      .def("set_voi", &cph::sim::set_voi)
      .def("set_voi_specs", &cph::sim::set_voi_specs)
      .def("set_opts", &cph::sim::set_opts)
      .def("set_monitors", &cph::sim::set_monitors, nb::arg("si"),
           nb::arg("atts"))
      .def("set_global_monitors", &cph::sim::set_global_monitors,
           nb::arg("gather_specs"), nb::arg("mon_specs"))
      .def("take_global_monitor_outputs",
           &cph::sim::take_global_monitor_outputs)
      .def("get_monitor_states", &cph::sim::get_monitor_states)
      .def("set_monitor_states", &cph::sim::set_monitor_states,
           nb::arg("states"))
      .def("get_global_monitor_states",
           &cph::sim::get_global_monitor_states)
      .def("set_global_monitor_states",
           &cph::sim::set_global_monitor_states, nb::arg("states"))
      .def("monitor_begin_run", &cph::sim::monitor_begin_run)
      .def("set_stim_specs", &cph::sim::set_stim_specs, nb::arg("si"),
           nb::arg("specs"))
      .def("clear_stim_specs", &cph::sim::clear_stim_specs)
      .def("set_subnet_rng", &cph::sim::set_subnet_rng, nb::arg("si"),
           nb::arg("lane_keys"), nb::arg("gfun"), nb::arg("ntau") = 0.0,
           nb::arg("is_multiplicative") = false,
           nb::arg("mul_a") = nb::none(), nb::arg("mul_b") = nb::none())
      .def("get_rng_states", &cph::sim::get_rng_states)
      .def("set_rng_states", &cph::sim::set_rng_states, nb::arg("states"))
      .def("enable_online_features", &cph::sim::enable_online_features,
           nb::arg("si"), nb::arg("dt_sample") = 1.0)
      .def("disable_online_features", &cph::sim::disable_online_features)
      .def("take_online_features", &cph::sim::take_online_features,
           nb::arg("si"), nb::arg("names"), nb::arg("order") = 4)
      .def("get_subnet_state", &cph::sim::get_subnet_state)
      .def("get_subnet_buf", &cph::sim::get_subnet_buf, nb::arg("si"))
      .def("set_subnet_buf", &cph::sim::set_subnet_buf, nb::arg("si"),
           nb::arg("v"))
      .def("t_abs", &cph::sim::t_abs)
      .def("set_t_abs", &cph::sim::set_t_abs, nb::arg("t"))
      .def("run", &cph::sim::run, nb::arg("nstep"), nb::arg("chunk_size") = 0,
           nb::arg("noise") = nb::none(), nb::arg("stim") = nb::none(),
           nb::arg("need_tavg") = true, nb::arg("need_ctavg") = true);
}
