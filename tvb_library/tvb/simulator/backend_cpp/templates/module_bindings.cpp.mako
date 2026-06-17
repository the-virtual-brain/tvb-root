#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "${generated_cpp_filename}"

namespace py = pybind11;
using namespace tvb::hybrid::runtime;

PYBIND11_MODULE(${module_name}, m) {
  m.doc() = "Auto-generated pybind11 module for TVB hybrid C++ backend";

  m.def("describe_metadata", []() {
    const auto meta = tvb::hybrid::generated::describe();
    py::dict out;
    out["module_name"]          = meta.module_name;
    out["cache_key"]            = meta.cache_key;
    out["backend_version"]      = meta.backend_version;
    out["user_source_hint"]     = meta.user_source_hint;
    out["dt"]                   = meta.dt;
    out["num_subnetworks"]      = py::int_(meta.num_subnetworks);
    out["num_inter_projections"] = py::int_(meta.num_inter_projections);
    out["num_intra_projections"] = py::int_(meta.num_intra_projections);
    out["num_monitors"]         = py::int_(meta.num_monitors);
    return out;
  });

  m.def("debug_probe_history", []() {
    HistoryBuffer history(3, 1, 1, 1);
    StateBuffer s0(1, 1, 1); StateBuffer s1(1, 1, 1);
    StateBuffer s2(1, 1, 1); StateBuffer s3(1, 1, 1);
    s0(0,0,0)=10.0; s1(0,0,0)=20.0; s2(0,0,0)=30.0; s3(0,0,0)=40.0;
    history.push(s0); history.push(s1); history.push(s2); history.push(s3);
    py::dict out;
    out["capacity"] = py::int_(history.capacity());
    out["size"]     = py::int_(history.size());
    out["delay_0"]  = history.read_value(0, 0, 0, 0);
    out["delay_1"]  = history.read_value(1, 0, 0, 0);
    out["delay_2"]  = history.read_value(2, 0, 0, 0);
    return out;
  });

  // run_simulation
  //
  // Parameters
  // ----------
  // initial_states        : list of ${num_subnetworks} numpy arrays, one per subnet,
  //                         each with shape (n_state_vars, n_nodes, n_modes), dtype float64.
  // nstep                 : number of integration steps.
  // chunk_size            : monitor averaging window (steps per output sample).
  //
  // Intra-projection arrays (parallel lists, one entry per intra-projection):
  //   intra_proj_weights_data    / intra_proj_weights_indices / intra_proj_weights_indptr
  //   intra_proj_idelays
  //   intra_proj_source_svars    : source state-variable index (int)
  //   intra_proj_target_cvars    : target coupling-slot index (int)
  //   intra_proj_scales          : global projection scale (float)
  //
  // Inter-projection arrays (same structure, source and target are different subnets):
  //   inter_proj_weights_data    / inter_proj_weights_indices / inter_proj_weights_indptr
  //   inter_proj_idelays
  //   inter_proj_source_svars
  //   inter_proj_target_cvars
  //   inter_proj_scales          : effective scale = proj.scale * mode_map[0,0]
  //
  // Returns
  // -------
  // list of (times, data) tuples, one per subnet.
  //   times : shape (num_chunks,)
  //   data  : shape (num_chunks, n_voi, n_nodes, n_modes)

  m.def(
      "run_simulation",
      [](py::list  initial_states,
         std::size_t nstep,
         std::size_t chunk_size,
         // --- intra-projection arrays ---
         py::list intra_proj_weights_data,
         py::list intra_proj_weights_indices,
         py::list intra_proj_weights_indptr,
         py::list intra_proj_idelays,
         py::list intra_proj_source_svars,
         py::list intra_proj_target_cvars,
         py::list intra_proj_scales,
         py::list intra_proj_cfun_types,
         py::list intra_proj_cfun_params,
         // --- inter-projection arrays ---
         py::list inter_proj_weights_data,
         py::list inter_proj_weights_indices,
         py::list inter_proj_weights_indptr,
         py::list inter_proj_idelays,
         py::list inter_proj_source_svars,
         py::list inter_proj_target_cvars,
         py::list inter_proj_scales,
         py::list inter_proj_cfun_types,
         py::list inter_proj_cfun_params,
         // --- per-subnet noise arrays (empty array for deterministic subnets) ---
         py::list noise_data_list,
         // --- per-subnet stimulus arrays (empty array for no-stimulus subnets) ---
         // layout: (n_cvar, n_nodes, nstep) float64, target_cvar applied Python-side
         py::list stim_data_list) {

        // ---- initial states ----
        const std::size_t n_subnets = initial_states.size();
        if (n_subnets != ${num_subnetworks}) {
          throw std::runtime_error(
              "run_simulation: expected ${num_subnetworks} initial_states, got " +
              std::to_string(n_subnets));
        }
        std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>> state_arrs(n_subnets);
        std::vector<std::vector<double>> flat_states(n_subnets);
        for (std::size_t i = 0; i < n_subnets; ++i) {
          state_arrs[i] = initial_states[i].cast<
              py::array_t<double, py::array::c_style | py::array::forcecast>>();
          if (state_arrs[i].ndim() != 3) {
            throw std::runtime_error(
                "initial_states[" + std::to_string(i) + "] must be 3-dimensional");
          }
          const double* src = state_arrs[i].data();
          flat_states[i].assign(src, src + static_cast<std::size_t>(state_arrs[i].size()));
        }

        // ---- helper: build ProjectionArrays vector from parallel python lists ----
        auto build_projections = [](
            py::list& wd, py::list& wi, py::list& wp, py::list& dl,
            py::list& ss, py::list& tc, py::list& sc_,
            py::list& ct, py::list& cp,
            std::vector<py::array_t<double,  py::array::c_style | py::array::forcecast>>& p_data,
            std::vector<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>& p_idx,
            std::vector<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>& p_ptr,
            std::vector<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>& p_del,
            std::vector<py::array_t<float,   py::array::c_style | py::array::forcecast>>& p_cfun)
            -> std::vector<ProjectionArrays>
        {
          const std::size_t n = wd.size();
          p_data.resize(n); p_idx.resize(n); p_ptr.resize(n); p_del.resize(n);
          p_cfun.resize(n);
          for (std::size_t i = 0; i < n; ++i) {
            p_data[i] = wd[i].cast<py::array_t<double,  py::array::c_style | py::array::forcecast>>();
            p_idx[i]  = wi[i].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
            p_ptr[i]  = wp[i].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
            p_del[i]  = dl[i].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
          }
          // cfun arrays only populated when lists are non-empty (backward compat)
          const bool has_cfun = (ct.size() == n) && (cp.size() == n);
          if (has_cfun) {
            for (std::size_t i = 0; i < n; ++i) {
              p_cfun[i] = cp[i].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
            }
          }
          std::vector<ProjectionArrays> projections(n);
          for (std::size_t i = 0; i < n; ++i) {
            projections[i].weights_data    = p_data[i].data();
            projections[i].weights_indices = p_idx[i].data();
            projections[i].weights_indptr  = p_ptr[i].data();
            projections[i].idelays         = p_del[i].data();
            projections[i].n_target_nodes  = static_cast<std::size_t>(p_ptr[i].size() - 1);
            projections[i].source_svar     = ss[i].cast<std::size_t>();
            projections[i].target_cvar_slot = tc[i].cast<std::size_t>();
            projections[i].scale           = sc_[i].cast<double>();
            if (has_cfun) {
              projections[i].cfun_type = static_cast<uint8_t>(ct[i].cast<int>());
              const float* cfun_data = p_cfun[i].data();
              for (int k = 0; k < 8; ++k) {
                projections[i].cfun_params[k] = cfun_data[k];
              }
            } else {
              projections[i].cfun_type = 0;  // kCfunNone
              for (int k = 0; k < 8; ++k) {
                projections[i].cfun_params[k] = (k == 0) ? 1.0f : 0.0f;
              }
            }
          }
          return projections;
        };

        // Keep arrays alive for the duration of run_simulation.
        std::vector<py::array_t<double,  py::array::c_style | py::array::forcecast>>
            intra_p_data, inter_p_data;
        std::vector<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>
            intra_p_idx, intra_p_ptr, intra_p_del,
            inter_p_idx,  inter_p_ptr,  inter_p_del;
        std::vector<py::array_t<float,   py::array::c_style | py::array::forcecast>>
            intra_p_cfun, inter_p_cfun;

        const auto intra_projections = build_projections(
            intra_proj_weights_data, intra_proj_weights_indices,
            intra_proj_weights_indptr, intra_proj_idelays,
            intra_proj_source_svars, intra_proj_target_cvars, intra_proj_scales,
            intra_proj_cfun_types, intra_proj_cfun_params,
            intra_p_data, intra_p_idx, intra_p_ptr, intra_p_del, intra_p_cfun);

        const auto inter_projections = build_projections(
            inter_proj_weights_data, inter_proj_weights_indices,
            inter_proj_weights_indptr, inter_proj_idelays,
            inter_proj_source_svars, inter_proj_target_cvars, inter_proj_scales,
            inter_proj_cfun_types, inter_proj_cfun_params,
            inter_p_data, inter_p_idx, inter_p_ptr, inter_p_del, inter_p_cfun);

        // ---- noise arrays (one float64 array per subnet; empty for deterministic) ----
        const std::size_t n_sn = ${num_subnetworks};
        std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>> noise_arrs(n_sn);
        std::vector<const double*> noise_ptrs(n_sn, nullptr);
        if (static_cast<std::size_t>(noise_data_list.size()) == n_sn) {
          for (std::size_t i = 0; i < n_sn; ++i) {
            noise_arrs[i] = noise_data_list[i].cast<
                py::array_t<double, py::array::c_style | py::array::forcecast>>();
            if (noise_arrs[i].size() > 0) {
              noise_ptrs[i] = noise_arrs[i].data();
            }
          }
        }

        // ---- stimulus arrays ----
        std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>> stim_arrs(n_sn);
        std::vector<const double*> stim_ptrs(n_sn, nullptr);
        if (static_cast<std::size_t>(stim_data_list.size()) == n_sn) {
          for (std::size_t i = 0; i < n_sn; ++i) {
            stim_arrs[i] = stim_data_list[i].cast<
                py::array_t<double, py::array::c_style | py::array::forcecast>>();
            if (stim_arrs[i].size() > 0) {
              stim_ptrs[i] = stim_arrs[i].data();
            }
          }
        }

        // ---- run ----
        // Release the GIL so Python threads can run other sweep points concurrently.
        // All numpy→C++ copies are done above; result→numpy conversion happens below.
        const auto results = [&]() {
            py::gil_scoped_release release;
            return tvb::hybrid::generated::run_simulation(
                flat_states, intra_projections, inter_projections, nstep, chunk_size,
                noise_ptrs, stim_ptrs);
        }();

        // ---- pack output: list of (times, data, ctavg) per subnet ----
        // ctavg shape: (num_chunks, n_coupling_vars, n_nodes, 1)
        // Backend selects which of data or ctavg to expose based on monitor type.
        py::list out;
        for (const auto& result : results) {
          // times: (num_chunks,)
          py::array_t<double> times(
              std::vector<py::ssize_t>{static_cast<py::ssize_t>(result.times.size())});
          {
            auto m = times.mutable_unchecked<1>();
            for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(result.times.size()); ++i) {
              m(i) = result.times[static_cast<std::size_t>(i)];
            }
          }

          // data: (num_chunks, n_voi, n_nodes, n_modes)
          py::array_t<double> data(std::vector<py::ssize_t>{
              static_cast<py::ssize_t>(result.num_chunks),
              static_cast<py::ssize_t>(result.num_voi),
              static_cast<py::ssize_t>(result.num_nodes),
              static_cast<py::ssize_t>(result.num_modes)});
          {
            auto m = data.mutable_unchecked<4>();
            std::size_t idx = 0;
            for (std::size_t chunk = 0; chunk < result.num_chunks; ++chunk) {
              for (std::size_t ivoi = 0; ivoi < result.num_voi; ++ivoi) {
                for (std::size_t node = 0; node < result.num_nodes; ++node) {
                  for (std::size_t mode = 0; mode < result.num_modes; ++mode) {
                    m(static_cast<py::ssize_t>(chunk),
                      static_cast<py::ssize_t>(ivoi),
                      static_cast<py::ssize_t>(node),
                      static_cast<py::ssize_t>(mode)) = result.data[idx++];
                  }
                }
              }
            }
          }

          // ctavg: (num_chunks, n_coupling_vars, n_nodes, 1)
          py::array_t<double> ctavg(std::vector<py::ssize_t>{
              static_cast<py::ssize_t>(result.num_chunks),
              static_cast<py::ssize_t>(result.num_coupling_vars),
              static_cast<py::ssize_t>(result.num_nodes),
              1});
          {
            auto m = ctavg.mutable_unchecked<4>();
            std::size_t idx = 0;
            for (std::size_t chunk = 0; chunk < result.num_chunks; ++chunk) {
              for (std::size_t cv = 0; cv < result.num_coupling_vars; ++cv) {
                for (std::size_t node = 0; node < result.num_nodes; ++node) {
                  m(static_cast<py::ssize_t>(chunk),
                    static_cast<py::ssize_t>(cv),
                    static_cast<py::ssize_t>(node),
                    0) = result.ctavg_data[idx++];
                }
              }
            }
          }

          out.append(py::make_tuple(times, data, ctavg));
        }
        return out;
      },
      py::arg("initial_states"),
      py::arg("nstep"),
      py::arg("chunk_size")                   = 1,
      py::arg("intra_proj_weights_data")       = py::list(),
      py::arg("intra_proj_weights_indices")    = py::list(),
      py::arg("intra_proj_weights_indptr")     = py::list(),
      py::arg("intra_proj_idelays")            = py::list(),
      py::arg("intra_proj_source_svars")       = py::list(),
      py::arg("intra_proj_target_cvars")       = py::list(),
      py::arg("intra_proj_scales")             = py::list(),
      py::arg("intra_proj_cfun_types")         = py::list(),
      py::arg("intra_proj_cfun_params")        = py::list(),
      py::arg("inter_proj_weights_data")       = py::list(),
      py::arg("inter_proj_weights_indices")    = py::list(),
      py::arg("inter_proj_weights_indptr")     = py::list(),
      py::arg("inter_proj_idelays")            = py::list(),
      py::arg("inter_proj_source_svars")       = py::list(),
      py::arg("inter_proj_target_cvars")       = py::list(),
      py::arg("inter_proj_scales")             = py::list(),
      py::arg("inter_proj_cfun_types")         = py::list(),
      py::arg("inter_proj_cfun_params")        = py::list(),
      py::arg("noise_data_list")               = py::list(),
      py::arg("stim_data_list")                = py::list());
}
