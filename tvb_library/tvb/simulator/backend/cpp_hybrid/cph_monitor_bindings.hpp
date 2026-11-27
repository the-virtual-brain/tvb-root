#pragma once
// Nanobind marshaling helpers and monitor spec/state converters for tvb cpp_hybrid.
//
// Compiled into the single _core.cpp LTO translation unit after `make_owned_array`
// and `namespace nb = nanobind;`.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cph_monitors.hpp"

namespace cph {

namespace nb = nanobind;

// ---- Python / nanobind marshaling helpers ----------------------------------

template <typename T>
static T dict_get_or(const nb::dict &d, const char *key, T defval) {
  if (!d.contains(key)) return defval;
  nb::object val = d[key];
  if (val.is_none()) return defval;
  return nb::cast<T>(val);
}

static std::vector<int> to_int_vec(nb::handle h) {
  std::vector<int> out;
  if (h.is_none()) return out;
  try {
    auto arr = nb::cast<nb::ndarray<nb::numpy, const int32_t, nb::c_contig>>(h);
    out.assign(arr.data(), arr.data() + arr.size());
    return out;
  } catch (const std::exception &) {
  }
  try {
    auto arr = nb::cast<nb::ndarray<nb::numpy, const int64_t, nb::c_contig>>(h);
    out.resize(arr.size());
    const int64_t *p = arr.data();
    for (size_t i = 0; i < arr.size(); i++) out[i] = static_cast<int>(p[i]);
    return out;
  } catch (const std::exception &) {
  }
  auto seq = nb::cast<nb::sequence>(h);
  for (auto item : seq) out.push_back(nb::cast<int>(item));
  return out;
}

// Legacy ndarray conversions may narrow before the caller can check the index.
static std::vector<int> to_checked_indices(nb::handle h, size_t upper_bound,
                                           const char *label, const char *what) {
  const auto invalid = [label, what]() {
    return std::runtime_error(std::string(label) + ": invalid " + what);
  };
  const auto numpy = nb::module_::import_("numpy");
  if (nb::isinstance(h, numpy.attr("ndarray"))) {
    const auto kind = nb::cast<std::string>(h.attr("dtype").attr("kind"));
    if ((kind != "i" && kind != "u") || nb::cast<int>(h.attr("ndim")) != 1)
      throw invalid();
  } else if (!nb::isinstance<nb::sequence>(h)) {
    throw invalid();
  }
  const auto numpy_bool = numpy.attr("bool_");
  const auto seq = nb::borrow<nb::sequence>(h);
  if (nb::len(seq) > (size_t)std::numeric_limits<int>::max()) throw invalid();
  std::vector<int> out;
  for (auto item : seq) {
    if (PyBool_Check(item.ptr()) || nb::isinstance(item, numpy_bool) || !PyIndex_Check(item.ptr()))
      throw invalid();
    auto index = nb::steal<nb::object>(PyNumber_Index(item.ptr()));
    if (!index.is_valid()) throw nb::python_error();
    int overflow = 0;
    const long value = PyLong_AsLongAndOverflow(index.ptr(), &overflow);
    if (overflow || value < 0 || value > std::numeric_limits<int>::max() ||
        (size_t)value >= upper_bound)
      throw invalid();
    out.push_back(static_cast<int>(value));
  }
  return out;
}

static std::vector<uint64_t> to_u64_vec(nb::handle h) {
  std::vector<uint64_t> out;
  if (h.is_none()) return out;
  try {
    auto arr = nb::cast<nb::ndarray<nb::numpy, const uint64_t, nb::c_contig>>(h);
    out.assign(arr.data(), arr.data() + arr.size());
    return out;
  } catch (const std::exception &) {
  }
  try {
    auto arr = nb::cast<nb::ndarray<nb::numpy, const int64_t, nb::c_contig>>(h);
    out.resize(arr.size());
    const int64_t *p = arr.data();
    for (size_t i = 0; i < arr.size(); i++)
      out[i] = static_cast<uint64_t>(p[i]);
    return out;
  } catch (const std::exception &) {
  }
  try {
    auto arr = nb::cast<nb::ndarray<nb::numpy, const uint32_t, nb::c_contig>>(h);
    out.resize(arr.size());
    const uint32_t *p = arr.data();
    for (size_t i = 0; i < arr.size(); i++)
      out[i] = static_cast<uint64_t>(p[i]);
    return out;
  } catch (const std::exception &) {
  }
  auto seq = nb::cast<nb::sequence>(h);
  for (auto item : seq) out.push_back(nb::cast<uint64_t>(item));
  return out;
}

static std::vector<float> to_float_vec(nb::handle h) {
  std::vector<float> out;
  if (h.is_none()) return out;
  try {
    auto arr = nb::cast<nb::ndarray<nb::numpy, const float, nb::c_contig>>(h);
    out.assign(arr.data(), arr.data() + arr.size());
    return out;
  } catch (const std::exception &) {
  }
  try {
    auto arr = nb::cast<nb::ndarray<nb::numpy, const double, nb::c_contig>>(h);
    out.resize(arr.size());
    const double *p = arr.data();
    for (size_t i = 0; i < arr.size(); i++) out[i] = static_cast<float>(p[i]);
    return out;
  } catch (const std::exception &) {
  }
  auto seq = nb::cast<nb::sequence>(h);
  for (auto item : seq) out.push_back(nb::cast<float>(item));
  return out;
}

static std::vector<double> to_double_vec(nb::handle h) {
  std::vector<double> out;
  if (h.is_none()) return out;
  try {
    auto arr = nb::cast<nb::ndarray<nb::numpy, const double, nb::c_contig>>(h);
    out.assign(arr.data(), arr.data() + arr.size());
    return out;
  } catch (const std::exception &) {
  }
  try {
    auto arr = nb::cast<nb::ndarray<nb::numpy, const float, nb::c_contig>>(h);
    out.resize(arr.size());
    const float *p = arr.data();
    for (size_t i = 0; i < arr.size(); i++) out[i] = static_cast<double>(p[i]);
    return out;
  } catch (const std::exception &) {
  }
  auto seq = nb::cast<nb::sequence>(h);
  for (auto item : seq) out.push_back(nb::cast<double>(item));
  return out;
}

static void extract_2d_shape(nb::handle h, int &rows, int &cols) {
  try {
    auto arr = nb::cast<nb::ndarray<nb::numpy, const float, nb::c_contig>>(h);
    if (arr.ndim() != 2) throw std::runtime_error("expected 2D float32 array");
    rows = static_cast<int>(arr.shape(0));
    cols = static_cast<int>(arr.shape(1));
    return;
  } catch (const std::exception &) {
  }
  auto arr = nb::cast<nb::ndarray<nb::numpy, const double, nb::c_contig>>(h);
  if (arr.ndim() != 2) throw std::runtime_error("expected 2D float64 array");
  rows = static_cast<int>(arr.shape(0));
  cols = static_cast<int>(arr.shape(1));
}

static nb::dict kmon_state_to_dict(const KMonState &st) {
  nb::dict d;
  d["slot"] = st.slot;
  d["kind"] = st.kind;
  d["width"] = st.width;
  d["n_out_v"] = st.n_out_v;
  d["n_out_space"] = st.n_out_space;
  d["n_out_modes"] = st.n_out_modes;
  d["m_step"] = st.m_step;
  d["run_start"] = st.run_start;

  auto acc = make_owned_array<double>({st.accum.size()});
  if (!st.accum.empty())
    std::memcpy(acc.data(), st.accum.data(), st.accum.size() * sizeof(double));
  d["accum"] = acc;
  d["accum_count"] = st.accum_count;

  auto itm = make_owned_array<float>({st.interim.size()});
  if (!st.interim.empty())
    std::memcpy(itm.data(), st.interim.data(),
                st.interim.size() * sizeof(float));
  d["interim"] = itm;
  d["interim_count"] = st.interim_count;

  auto stk = make_owned_array<float>({st.stock.size()});
  if (!st.stock.empty())
    std::memcpy(stk.data(), st.stock.data(), st.stock.size() * sizeof(float));
  d["stock"] = stk;
  d["stock_pos"] = st.stock_pos;

  auto bws = make_owned_array<float>({st.bw_state.size()});
  if (!st.bw_state.empty())
    std::memcpy(bws.data(), st.bw_state.data(),
                st.bw_state.size() * sizeof(float));
  d["bw_state"] = bws;
  return d;
}

static KMonState dict_to_kmon_state(const nb::dict &d) {
  KMonState st;
  st.slot = dict_get_or<int>(d, "slot", -1);
  st.kind = dict_get_or<int>(d, "kind", 0);
  st.width = dict_get_or<int>(d, "width", 1);
  st.n_out_v = dict_get_or<int>(d, "n_out_v", 0);
  st.n_out_space = dict_get_or<int>(d, "n_out_space", 0);
  st.n_out_modes = dict_get_or<int>(d, "n_out_modes", 1);
  st.m_step = dict_get_or<uint64_t>(d, "m_step", 0);
  st.run_start = dict_get_or<uint64_t>(d, "run_start", 0);
  if (d.contains("accum")) st.accum = to_double_vec(d["accum"]);
  st.accum_count = dict_get_or<uint32_t>(d, "accum_count", 0);
  if (d.contains("interim")) st.interim = to_float_vec(d["interim"]);
  st.interim_count = dict_get_or<uint32_t>(d, "interim_count", 0);
  if (d.contains("stock")) st.stock = to_float_vec(d["stock"]);
  st.stock_pos = dict_get_or<uint32_t>(d, "stock_pos", 0);
  if (d.contains("bw_state")) st.bw_state = to_float_vec(d["bw_state"]);
  return st;
}

static KMonState dict_to_legacy_sibling_state(const nb::dict &d, const char *label) {
  for (const char *key : {"m_step", "run_start", "accum_count", "interim_count", "stock_pos"}) {
    if (!d.contains(key) || !PyLong_CheckExact(d[key].ptr()))
      throw std::runtime_error(std::string(label) + "invalid sibling snapshot counter: " + key);
  }
  return dict_to_kmon_state(d);
}

static KMonState dict_to_staged_state(const nb::dict &d, bool bw,
                                      const std::string &label) {
  for (const char *key : {"slot", "kind", "width", "n_out_v", "n_out_space",
                          "n_out_modes", "accum", "accum_count", "m_step", "run_start"}) {
    if (!d.contains(key))
      throw std::runtime_error(label + "incomplete snapshot state");
  }
  // Staged snapshots may carry the empty legacy fields emitted by export.
  for (const char *key : {"interim", "stock", "bw_state"}) {
    if (d.contains(key) && nb::len(nb::handle(d[key].ptr())) != 0)
      throw std::runtime_error(label + "unexpected legacy snapshot state");
  }
  for (const char *key : {"interim_count", "stock_pos"}) {
    if (d.contains(key) && (!PyLong_CheckExact(d[key].ptr()) || nb::cast<uint32_t>(d[key]) != 0))
      throw std::runtime_error(label + "unexpected legacy snapshot state");
  }
  KMonState st;
  st.slot = nb::cast<int>(d["slot"]);
  st.kind = nb::cast<int>(d["kind"]);
  st.width = nb::cast<int>(d["width"]);
  st.n_out_v = nb::cast<int>(d["n_out_v"]);
  st.n_out_space = nb::cast<int>(d["n_out_space"]);
  st.n_out_modes = nb::cast<int>(d["n_out_modes"]);
  if (bw) {
    if (nb::len(nb::handle(d["accum"].ptr())) != 0)
      throw std::runtime_error(label + "unexpected legacy snapshot accumulator");
  } else {
    st.accum = to_double_vec(d["accum"]);
  }
  st.accum_count = nb::cast<uint32_t>(d["accum_count"]);
  st.m_step = nb::cast<uint64_t>(d["m_step"]);
  st.run_start = nb::cast<uint64_t>(d["run_start"]);
  return st;
}

using BwParameters = balloon_windkessel::Parameters;
static constexpr std::pair<const char *, double BwParameters::*> bw_parameter_fields[] = {
    {"signal_decay_s", &BwParameters::signal_decay_s},
    {"flow_feedback_s2", &BwParameters::flow_feedback_s2},
    {"transit_time_s", &BwParameters::transit_time_s},
    {"grubb_exponent", &BwParameters::grubb_exponent},
    {"resting_extraction", &BwParameters::resting_extraction},
    {"intravascular_ratio", &BwParameters::intravascular_ratio},
    {"frequency_offset_hz", &BwParameters::frequency_offset_hz},
    {"relaxation_rate_per_s", &BwParameters::relaxation_rate_per_s},
    {"echo_time_s", &BwParameters::echo_time_s},
    {"resting_blood_volume", &BwParameters::resting_blood_volume},
};

// Nominal 10-parameter Balloon-Windkessel set exposed to Python from C++.
static nb::dict balloon_windkessel_nominal_params() {
  nb::dict d;
  for (const auto &[name, member] : bw_parameter_fields)
    d[name] = balloon_windkessel::kNominalParameters.*member;
  return d;
}

// One explicit, complete scalar hemodynamic parameter set (no defaults).
static BwParameters parse_bw_parameters(const nb::dict &d, const char *label) {
  if (!d.contains("bw_stream_params"))
    throw std::runtime_error(std::string(label) + ": incomplete parameter set");
  const auto parameters = nb::cast<nb::dict>(d["bw_stream_params"]);
  BwParameters p;
  if (parameters.size() != std::size(bw_parameter_fields))
    throw std::runtime_error(std::string(label) + ": expected one explicit scalar parameter set");
  for (const auto &[name, member] : bw_parameter_fields) {
    if (!parameters.contains(name))
      throw std::runtime_error(std::string(label) + ": incomplete parameter set");
    p.*member = nb::cast<double>(parameters[name]);
  }
  return p;
}

static void pack_bw_regions(const std::vector<BwStreamRegion> &src, double *dst) {
  for (const auto &r : src) {
    const double values[] = {r.drive_sum, r.bold_sum, r.ode.signal,
                             r.ode.inflow, r.ode.volume, r.ode.deoxy};
    dst = std::copy(std::begin(values), std::end(values), dst);
  }
}

static void unpack_bw_regions(const double *src, size_t n_lanes,
                              std::vector<BwStreamRegion> &dst) {
  dst.resize(n_lanes);
  for (auto &r : dst) {
    r = {src[0], src[1], {src[2], src[3], src[4], src[5]}};
    src += 6;
  }
}

static void export_bw_stream_state(nb::dict &state, const KMonState &st) {
  auto regions = make_owned_array<double>({st.bw_regions.size(), 6});
  pack_bw_regions(st.bw_regions, regions.data());
  state["bw_stream_regions"] = regions;
  state["bw_drive_count"] = st.bw_drive_count;
}

static void parse_bw_stream_state(KMonState &st, const nb::dict &d, size_t n_node) {
  if (!d.contains("bw_stream_regions") || !d.contains("bw_drive_count") ||
      !PyLong_CheckExact(d["bw_drive_count"].ptr()))
    throw std::runtime_error("global BW stream: incomplete or invalid snapshot state");
  auto regions = nb::cast<nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>>(
      d["bw_stream_regions"], false);
  if (regions.shape(0) != n_node || regions.shape(1) != 6)
    throw std::runtime_error("global BW stream: invalid snapshot region shape");
  st.bw_drive_count = nb::cast<uint32_t>(d["bw_drive_count"]);
  unpack_bw_regions(regions.data(), n_node, st.bw_regions);
}

static constexpr const char *projection_label = "global complete window projection";

static auto projection_array(nb::handle h, size_t ndim, const char *name) {
  const auto numpy = nb::module_::import_("numpy");
  const auto invalid = [name]() {
    return std::runtime_error(std::string(projection_label) + ": invalid " + name);
  };
  if (!nb::isinstance(h, numpy.attr("ndarray"))) throw invalid();
  const auto kind = nb::cast<std::string>(h.attr("dtype").attr("kind"));
  auto a = nb::cast<nb::ndarray<nb::numpy>>(h, false);
  if ((kind != "f" && kind != "i" && kind != "u") || a.ndim() != ndim)
    throw invalid();
  return a;
}

template <int W>
static void configure_window_projection(kmon<W> &m, const nb::dict &d, int n_nodes,
                                        size_t n_voi) {
  if (!d.contains("gain") || d["gain"].is_none())
    throw std::runtime_error("global complete window projection: preconfigured gain required");
  if (d.contains("obsnoise") && !d["obsnoise"].is_none())
    throw std::runtime_error("global complete window projection: observation noise is unsupported");
  const auto shape = projection_array(d["gain"], 2, "gain");
  const size_t n_sensors = shape.shape(0);
  if (n_sensors == 0 || n_sensors > (size_t)std::numeric_limits<int>::max() ||
      shape.shape(1) != (size_t)n_nodes || n_sensors > m.proj.gain.max_size() / (size_t)n_nodes ||
      n_sensors > std::min(m.accum.max_size(), m.d_out.max_size()) / n_voi)
    throw std::runtime_error("global complete window projection: invalid gain or output extent");
  using FloatArray = nb::ndarray<nb::numpy, const float, nb::c_contig>;
  FloatArray ref;
  if (d.contains("ref_vec") && !d["ref_vec"].is_none()) {
    const auto ref_shape = projection_array(d["ref_vec"], 1, "ref_vec");
    if (ref_shape.shape(0) != n_sensors)
      throw std::runtime_error("global complete window projection: ref_vec length != sensor count");
    ref = nb::cast<FloatArray>(d["ref_vec"]);
    // Reference weights are prepared by the caller, including any masked
    // contribution. Do not rescale the effective operator here.
    if (std::any_of(ref.data(), ref.data() + n_sensors,
                    [](float x) { return !std::isfinite(x); }))
      throw std::runtime_error("global complete window projection: nonfinite ref_vec");
  }
  const auto gain = nb::cast<FloatArray>(d["gain"]);
  if (std::any_of(gain.data(), gain.data() + n_sensors * (size_t)n_nodes,
                  [](float x) { return !std::isfinite(x); }))
    throw std::runtime_error("global complete window projection: nonfinite gain");
  m.set_projection(gain.data(), (int)n_sensors, n_nodes, ref.is_valid() ? ref.data() : nullptr,
                   ref.is_valid() ? (int)n_sensors : 0);
  if (!m.proj.has_ref_vec) m.proj.ref_vec.clear();  // Canonical identity for no reference.
}

static constexpr const char *local_window_label = "Recorder complete window: ";
static constexpr int recorder_slot_base = 10000;  // Python _RECORDER_SLOT_BASE

static bool matches_native_clock(double clock, float native) {
  return std::isfinite(clock) && clock > 0.0 && clock <= std::numeric_limits<float>::max() &&
         native > 0.0f && (float)clock == native;
}

static bool matches_scheduled_cadence(double source_dt, uint32_t cadence,
                                      double master_clock) {
  if (cadence == 0 || !std::isfinite(source_dt) || source_dt <= 0.0 ||
      !std::isfinite(master_clock) || master_clock <= 0.0)
    return false;
  const double scheduled_dt = (double)cadence * master_clock;
  const double error_ticks = std::abs(source_dt / master_clock - (double)cadence);
  // Match NetworkSet's nearest-integer cadence (ties to even) and clock tolerance.
  return error_ticks <= 0.5 && (error_ticks < 0.5 || cadence % 2 == 0) &&
         std::abs(source_dt - scheduled_dt) <= 1e-12 + 1e-9 * scheduled_dt;
}

static bool checked_python_equal(PyObject *actual, PyObject *expected) {
  const int equal = PyObject_RichCompareBool(actual, expected, Py_EQ);
  if (equal < 0) throw nb::python_error();
  return equal != 0;
}

// Walk the exported schema and compare only exact built-in scalar leaves.
static bool equal_checked_window_config(PyObject *actual, PyObject *expected) {
  if (PyTuple_CheckExact(expected)) {
    if (!PyTuple_CheckExact(actual) || PyTuple_Size(actual) != PyTuple_Size(expected))
      return false;
    for (Py_ssize_t i = 0, n = PyTuple_Size(expected); i < n; ++i)
      if (!equal_checked_window_config(PyTuple_GetItem(actual, i), PyTuple_GetItem(expected, i)))
        return false;
    return true;
  }
  if (PyList_CheckExact(expected)) {
    if (!PyList_CheckExact(actual) || PyList_Size(actual) != PyList_Size(expected))
      return false;
    for (Py_ssize_t i = 0, n = PyList_Size(expected); i < n; ++i)
      if (!equal_checked_window_config(PyList_GetItem(actual, i), PyList_GetItem(expected, i)))
        return false;
    return true;
  }
  if (!((PyBool_Check(expected) && PyBool_Check(actual)) ||
        (PyLong_CheckExact(expected) && PyLong_CheckExact(actual)) ||
        (PyFloat_CheckExact(expected) && PyFloat_CheckExact(actual)) ||
        (PyUnicode_CheckExact(expected) && PyUnicode_CheckExact(actual))))
    return false;
  return checked_python_equal(actual, expected);
}

static KMonState dict_to_checked_window_state(const nb::dict &d, size_t extent,
                                               const char *label) {
  for (const char *key : {"slot", "kind", "width", "n_out_v", "n_out_space", "n_out_modes",
                          "m_step", "run_start", "accum_count"}) {
    if (!d.contains(key) || !PyLong_CheckExact(d[key].ptr()))
      throw std::runtime_error(std::string(label) + "invalid snapshot integer field");
  }
  using Accumulator = nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig>;
  Accumulator acc;
  if (!d.contains("accum") || !nb::try_cast(d["accum"], acc, false) || acc.size() != extent)
    throw std::runtime_error(std::string(label) + "invalid snapshot accumulator extent or dtype");
  return dict_to_staged_state(d, false, label);
}

static constexpr const char *gathered_window_label = "global gathered complete window: ";

template <int W>
static void configure_gathered_window(const nb::dict &d, int n_voi, int n_node,
                                      double clock, kmon<W> &m) {
  const int kind = nb::cast<int>(d["kind"]);
  const int period = dict_get_or<int>(d, "istep", 0);
  if (!average_window_kind(kind) || period < 1 || !d.contains("dt") ||
      nb::cast<double>(d["dt"]) != clock || dict_get_or<uint32_t>(d, "cadence_K", 1u) != 1)
    throw std::runtime_error("global gathered complete window: invalid period, kind or clock");
  const auto voi = d.contains("voi") ? to_checked_indices(d["voi"], (size_t)n_voi,
                                     "global gathered complete window", "observed VOI")
                                    : std::vector<int>{};
  const size_t max_space = std::min(m.accum.max_size(), m.d_out.max_size()) /
                           (voi.empty() ? (size_t)n_voi : voi.size());
  if (kind == KMON_SPATIAL_AVG) {
    using SpatialMean = nb::ndarray<nb::numpy, const float, nb::ndim<2>, nb::c_contig>;
    SpatialMean weights;
    if (!d.contains("spatial_mean") || !nb::try_cast(d["spatial_mean"], weights, false) ||
        weights.shape(0) == 0 || weights.shape(1) != (size_t)n_node ||
        weights.shape(0) > max_space || weights.shape(0) > std::numeric_limits<int>::max() / (size_t)n_node)
      throw std::runtime_error("global gathered complete window: invalid spatial_mean extent or dtype");
    if (std::any_of(weights.data(), weights.data() + weights.size(),
                    [](float w) { return !std::isfinite(w); }))
      throw std::runtime_error("global gathered complete window: nonfinite spatial_mean");
    m.set_spatial_mean(weights.data(), (int)weights.shape(0), n_node);
  } else if ((kind == KMON_GLOBAL_AVG ? 1u : (size_t)n_node) > max_space) {
    throw std::runtime_error("global gathered complete window: output extent out of range");
  }
  m.complete_window = true;
  m.gathered_window = true;
  m.configure_basic(nb::cast<int>(d["slot"]), kind, period, clock, n_voi, n_node, 1, voi, 1u);
}

template <int W>
static void parse_kmon_spec(nb::handle item, int n_src_v, int n_space,
                            int n_cvar, int n_modes, double default_dt,
                            const std::vector<kmon<W>> &prev,
                            kmon<W> &out, bool global = false) {
  int slot = -1;
  int kind = 0;
  if (nb::isinstance<nb::dict>(item)) {
    nb::dict d = nb::cast<nb::dict>(item);
    slot = nb::cast<int>(d["slot"]);
    kind = nb::cast<int>(d["kind"]);
    if (dict_get_or<bool>(d, "local_window", false))
      throw std::runtime_error("Recorder complete window requires a local attachment");
    if (kind == KMON_BOLD_BW_LOCAL)
      throw std::runtime_error("Recorder complete window requires a local attachment");
    const bool complete_window = dict_get_or<bool>(d, "complete_window", false);
    if (kind == KMON_BOLD_BW_STREAM) {
      if (!global || complete_window || dict_get_or<uint32_t>(d, "cadence_K", 1u) != 1)
        throw std::runtime_error("global BW stream requires a global attachment");
      for (const char *key : {"istep", "dt", "voi", "bw_drive_baseline", "bw_drive_scale",
                              "bw_stream_params"})
        if (!d.contains(key))
          throw std::runtime_error("global BW stream: incomplete attachment specification");
      const double clock = nb::cast<double>(d["dt"]);
      if (!std::isfinite(clock) || clock <= 0 || clock > std::numeric_limits<float>::max() ||
          (float)clock != default_dt)
        throw std::runtime_error("global BW stream: incompatible neural clock");
      const auto voi = to_checked_indices(d["voi"], (size_t)n_src_v, "global BW stream",
                                          "observed VOI");
      if (voi.size() != 1)
        throw std::runtime_error("global BW stream: exactly one valid observed VOI is required");
      BwParameters p = parse_bw_parameters(d, "global BW stream");
      out.configure_bw_stream(slot, nb::cast<int>(d["istep"]), default_dt, n_space, voi[0],
                              nb::cast<double>(d["bw_drive_baseline"]),
                              nb::cast<double>(d["bw_drive_scale"]), p);
      return;
    }
    if (complete_window &&
        (!global || W != 1 || (kind != KMON_TAVG && kind != KMON_GLOBAL_AVG &&
                               kind != KMON_SPATIAL_AVG && kind != KMON_PROJECTION)))
      throw std::runtime_error("complete_window requires a supported global monitor at width 1");
    const int istep = dict_get_or<int>(d, "istep", 1);
    const uint32_t cadence_K = dict_get_or<uint32_t>(d, "cadence_K", 1u);
    const bool midpoint_time = dict_get_or<bool>(d, "midpoint_time", false);
    double m_dt = default_dt;
    if (d.contains("dt") && !d["dt"].is_none())
      m_dt = nb::cast<double>(d["dt"]);
    else if (d.contains("dt0") && !d["dt0"].is_none())
      m_dt = nb::cast<double>(d["dt0"]);
    const bool window_projection = complete_window && kind == KMON_PROJECTION;
    std::vector<int> voi;
    if (d.contains("voi") && !d["voi"].is_none())
      voi = complete_window
          ? to_checked_indices(d["voi"], (size_t)n_src_v,
                               window_projection ? projection_label : "global complete window", "observed VOI")
          : to_int_vec(d["voi"]);

    if (complete_window &&
        (istep < 1 || cadence_K != 1 || !std::isfinite(m_dt) || m_dt <= 0.0 ||
         (window_projection && m_dt > std::numeric_limits<float>::max()) ||
         (float)m_dt != (float)default_dt || n_src_v < 1 || n_space < 1 ||
         std::any_of(voi.begin(), voi.end(),
                     [n_src_v](int v) { return v < 0 || v >= n_src_v; })))
      throw std::runtime_error("global complete window: invalid period, clock, or VOI");
    out.complete_window = complete_window;

    if (kind == KMON_BOLD) {
      const int ii = dict_get_or<int>(d, "interim_istep", 1);
      const int S = dict_get_or<int>(d, "stock_steps", 1);
      const double k1v0 = dict_get_or<double>(d, "k1v0", 1.0);
      const bool fo = dict_get_or<bool>(d, "first_order", false);
      std::vector<double> hrf;
      if (d.contains("hrf") && !d["hrf"].is_none())
        hrf = to_double_vec(d["hrf"]);
      out.configure_bold(slot, istep, ii, S, k1v0, fo, m_dt, n_src_v, n_space,
                         voi, hrf, cadence_K);
    } else if (kind == KMON_BOLD_BW) {
      const float bw_dt = dict_get_or<float>(d, "bw_dt", 0.0f);
      std::vector<float> bwp;
      if (d.contains("bw_params") && !d["bw_params"].is_none())
        bwp = to_float_vec(d["bw_params"]);
      const float *bwp_ptr = (bwp.size() >= 10) ? bwp.data() : nullptr;
      out.configure_bold_bw(slot, istep, m_dt, n_src_v, n_space, voi, bw_dt,
                            bwp_ptr, cadence_K);
    } else if (kind == KMON_SPATIAL_AVG) {
      if (d.contains("spatial_mean") && !d["spatial_mean"].is_none()) {
        int n_areas = 0, n_sn = 0;
        extract_2d_shape(d["spatial_mean"], n_areas, n_sn);
        std::vector<float> sm = to_float_vec(d["spatial_mean"]);
        if (complete_window &&
            (n_areas < 1 || n_sn != n_space ||
             std::any_of(sm.begin(), sm.end(), [](float x) { return !std::isfinite(x); })))
          throw std::runtime_error("global complete window: invalid spatial_mean");
        out.set_spatial_mean(sm.data(), n_areas, n_sn);
      } else if (d.contains("spatial_area_idx") &&
                 !d["spatial_area_idx"].is_none()) {
        std::vector<int> aidx = to_int_vec(d["spatial_area_idx"]);
        std::vector<float> ascale = to_float_vec(d["spatial_scale"]);
        const int n_areas = dict_get_or<int>(d, "n_areas", 0);
        if (complete_window &&
            (n_areas < 1 || aidx.size() != (size_t)n_space ||
             ascale.size() != aidx.size() ||
             std::any_of(aidx.begin(), aidx.end(),
                         [n_areas](int a) { return a < 0 || a >= n_areas; }) ||
             std::any_of(ascale.begin(), ascale.end(),
                         [](float x) { return !std::isfinite(x); })))
          throw std::runtime_error("global complete window: invalid compact spatial weights");
        out.set_spatial_compact(aidx.data(), ascale.data(), (int)aidx.size(),
                                n_areas);
      } else if (complete_window) {
        throw std::runtime_error("global complete window: preconfigured spatial weights required");
      }
      out.configure_basic(slot, kind, istep, m_dt, n_src_v, n_space, 1, voi,
                          cadence_K, midpoint_time);
    } else if (kind == KMON_PROJECTION) {
      if (window_projection) {
        configure_window_projection(out, d, n_space, voi.empty() ? (size_t)n_src_v : voi.size());
      } else if (d.contains("gain") && !d["gain"].is_none()) {
        int n_sensors = 0, n_pn = 0;
        extract_2d_shape(d["gain"], n_sensors, n_pn);
        std::vector<float> g = to_float_vec(d["gain"]);
        std::vector<float> rv;
        if (d.contains("ref_vec") && !d["ref_vec"].is_none())
          rv = to_float_vec(d["ref_vec"]);
        out.set_projection(g.data(), n_sensors, n_pn,
                           rv.empty() ? nullptr : rv.data(), (int)rv.size());
      }
      out.configure_basic(slot, kind, istep, m_dt, n_src_v, n_space, 1, voi,
                          cadence_K, midpoint_time);
    } else if (kind == KMON_CTAVG) {
      out.configure_basic(slot, kind, istep, m_dt, n_cvar, n_space, n_modes,
                          voi, cadence_K, midpoint_time);
    } else {
      out.configure_basic(slot, kind, istep, m_dt, n_src_v, n_space, 1, voi,
                          cadence_K, midpoint_time);
    }
    auto it = std::find_if(prev.begin(), prev.end(), [&](const kmon<W> &e) {
      return !e.staged_global() && !e.local_window && out.same_legacy_config(e);
    });
    if (!complete_window && it != prev.end()) {
      out = *it;
      return;
    }
    return;
  }

  // Legacy 10-tuple: (slot, kind, istep, interim_istep, stock_steps, k1v0,
  //                   first_order, voi, hrf, dt)
  nb::tuple a = nb::cast<nb::tuple>(item);
  slot = nb::cast<int>(a[0]);
  kind = nb::cast<int>(a[1]);
  if (kind == KMON_BOLD_BW_STREAM)
    throw std::runtime_error("global BW stream requires an explicit native dictionary");
  if (kind == KMON_BOLD_BW_LOCAL)
    throw std::runtime_error("Recorder complete window requires a local attachment");
  const int istep = nb::cast<int>(a[2]);
  const int ii = nb::cast<int>(a[3]);
  const int S = nb::cast<int>(a[4]);
  const double k1v0 = nb::cast<double>(a[5]);
  const bool fo = nb::cast<bool>(a[6]);
  std::vector<int> voi = to_int_vec(a[7]);
  std::vector<double> hrf = to_double_vec(a[8]);
  const double m_dt = nb::cast<double>(a[9]);
  if (kind == KMON_BOLD) {
    out.configure_bold(slot, istep, ii, S, k1v0, fo, m_dt, n_src_v, n_space,
                       voi, hrf, 1u);
  } else if (kind == KMON_BOLD_BW) {
    out.configure_bold_bw(slot, istep, m_dt, n_src_v, n_space, voi, 0.0f,
                          nullptr, 1u);
  } else if (kind == KMON_CTAVG) {
    out.configure_basic(slot, kind, istep, m_dt, n_cvar, n_space, n_modes, voi,
                        1u, false);
  } else {
    out.configure_basic(slot, kind, istep, m_dt, n_src_v, n_space, 1, voi, 1u,
                        false);
  }
  auto it = std::find_if(prev.begin(), prev.end(), [&](const kmon<W> &e) {
    return !e.staged_global() && !e.local_window && out.same_legacy_config(e);
  });
  if (it != prev.end()) {
    out = *it;
    return;
  }
}

}  // namespace cph
