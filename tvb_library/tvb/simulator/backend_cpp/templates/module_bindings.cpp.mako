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
    out["module_name"] = meta.module_name;
    out["cache_key"] = meta.cache_key;
    out["backend_version"] = meta.backend_version;
    out["user_source_hint"] = meta.user_source_hint;
    out["dt"] = meta.dt;
    out["num_subnetworks"] = py::int_(meta.num_subnetworks);
    out["num_inter_projections"] = py::int_(meta.num_inter_projections);
    out["num_intra_projections"] = py::int_(meta.num_intra_projections);
    out["num_monitors"] = py::int_(meta.num_monitors);
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
    out["size"] = py::int_(history.size());
    out["delay_0"] = history.read_value(0, 0, 0, 0);
    out["delay_1"] = history.read_value(1, 0, 0, 0);
    out["delay_2"] = history.read_value(2, 0, 0, 0);
    return out;
  });

  m.def(
      "run_simulation",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> initial_state,
         std::size_t nstep,
         std::size_t chunk_size,
         // Projection arrays — parallel lists, one entry per projection.
         // Each list element is a numpy array; lists are empty when no projections.
         py::list proj_weights_data,
         py::list proj_weights_indices,
         py::list proj_weights_indptr,
         py::list proj_idelays,
         py::list proj_source_svars,
         py::list proj_target_cvars,
         py::list proj_scales) {
        using Generated = tvb::hybrid::generated::GeneratedModel;

        if (initial_state.ndim() != 3) {
          throw std::runtime_error(
              "initial_state must have shape (n_state_vars, n_nodes, n_modes)");
        }
        {
          const auto shape = initial_state.shape();
          if (shape[0] != static_cast<py::ssize_t>(Generated::kNumStateVars) ||
              shape[1] != static_cast<py::ssize_t>(Generated::kNumNodes) ||
              shape[2] != static_cast<py::ssize_t>(Generated::kNumModes)) {
            throw std::runtime_error("initial_state shape does not match generated spec.");
          }
        }

        const double* src = static_cast<const double*>(initial_state.data());
        std::vector<double> flat(src, src + initial_state.size());

        // Build ProjectionArrays from the parallel python lists.
        // Keep references alive for the duration of run_simulation.
        const std::size_t n_proj = proj_weights_data.size();
        std::vector<py::array_t<double,  py::array::c_style | py::array::forcecast>> p_data(n_proj);
        std::vector<py::array_t<int32_t, py::array::c_style | py::array::forcecast>> p_idx(n_proj);
        std::vector<py::array_t<int32_t, py::array::c_style | py::array::forcecast>> p_ptr(n_proj);
        std::vector<py::array_t<int32_t, py::array::c_style | py::array::forcecast>> p_del(n_proj);
        for (std::size_t i = 0; i < n_proj; ++i) {
          p_data[i] = proj_weights_data[i].cast<py::array_t<double,  py::array::c_style | py::array::forcecast>>();
          p_idx[i]  = proj_weights_indices[i].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
          p_ptr[i]  = proj_weights_indptr[i].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
          p_del[i]  = proj_idelays[i].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
        }
        std::vector<ProjectionArrays> projections(n_proj);
        for (std::size_t i = 0; i < n_proj; ++i) {
          projections[i].weights_data    = p_data[i].data();
          projections[i].weights_indices = p_idx[i].data();
          projections[i].weights_indptr  = p_ptr[i].data();
          projections[i].idelays         = p_del[i].data();
          projections[i].n_target_nodes  = static_cast<std::size_t>(p_ptr[i].size() - 1);
          projections[i].source_svar     = proj_source_svars[i].cast<std::size_t>();
          projections[i].target_cvar_slot = proj_target_cvars[i].cast<std::size_t>();
          projections[i].scale           = proj_scales[i].cast<double>();
        }

        const auto result = tvb::hybrid::generated::run_simulation(flat, projections, nstep, chunk_size);

        std::vector<py::ssize_t> times_shape = {
            static_cast<py::ssize_t>(result.times.size())};
        py::array_t<double> times(times_shape);
        auto times_mut = times.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(result.times.size()); ++i) {
          times_mut(i) = result.times[static_cast<std::size_t>(i)];
        }

        std::vector<py::ssize_t> data_shape = {
            static_cast<py::ssize_t>(result.num_chunks),
            static_cast<py::ssize_t>(result.num_voi),
            static_cast<py::ssize_t>(result.num_nodes),
            static_cast<py::ssize_t>(result.num_modes)};
        py::array_t<double> data(data_shape);
        auto data_mut = data.mutable_unchecked<4>();
        std::size_t idx = 0;
        for (std::size_t chunk = 0; chunk < result.num_chunks; ++chunk) {
          for (std::size_t ivoi = 0; ivoi < result.num_voi; ++ivoi) {
            for (std::size_t node = 0; node < result.num_nodes; ++node) {
              for (std::size_t mode = 0; mode < result.num_modes; ++mode) {
                data_mut(chunk, ivoi, node, mode) = result.data[idx++];
              }
            }
          }
        }

        return py::make_tuple(times, data);
      },
      py::arg("initial_state"),
      py::arg("nstep"),
      py::arg("chunk_size") = 1,
      py::arg("proj_weights_data")    = py::list(),
      py::arg("proj_weights_indices") = py::list(),
      py::arg("proj_weights_indptr")  = py::list(),
      py::arg("proj_idelays")         = py::list(),
      py::arg("proj_source_svars")    = py::list(),
      py::arg("proj_target_cvars")    = py::list(),
      py::arg("proj_scales")          = py::list());
}
