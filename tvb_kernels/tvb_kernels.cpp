#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

extern "C" {
#include "tvbk.h"
}

namespace nb = nanobind;
using namespace nb::literals;

static void conn_init_from_arrays(
    tvbk_conn *t,
    /* add more type info for validation */
    nb::ndarray<float, nb::shape<-1>> weights,
    nb::ndarray<int32_t, nb::shape<-1>> indices,
    nb::ndarray<int32_t, nb::shape<-1>> indptr,
    nb::ndarray<int32_t, nb::shape<-1>> idelays,
    nb::ndarray<float, nb::shape<-1, -1, -1>, nb::c_contig> buf,
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> cx) {
  new (t) tvbk_conn{/* TODO converting from unsigned long to int here */
                    .num_node = static_cast<int>(buf.shape(1)),
                    .num_nonzero = static_cast<int>(weights.shape(0)),
                    .num_cvar = static_cast<int>(buf.shape(0)),
                    .horizon = static_cast<int>(buf.shape(2)),
                    .horizon_minus_1 = static_cast<int>(buf.shape(2) - 1)};
  if (!((t->horizon & t->horizon_minus_1) == 0))
    throw nb::value_error("horizon (buf.shape[2]) must be power of 2");
  /* TODO more shape validation */
  t->weights = weights.data();
  t->indices = indices.data();
  t->indptr = indptr.data();
  t->idelays = idelays.data();
  t->buf = buf.data();
  /* TODO cx1, cx2 should be owned by model instance */
  t->cx1 = cx.data();
  t->cx2 = cx.data() + t->num_node;
}

NB_MODULE(tvb_kernels, m) {
  nb::class_<tvbk_conn>(m, "Conn")
      .def("__init__", &conn_init_from_arrays, "weights"_a, "indices"_a,
           "indptr"_a, "idelays"_a, "buf"_a, "cx"_a);
    /* TODO add accessors for arrays? */
    /* TODO take ownership of arrays? */

  m.def("cx_nop", &tvbk_cx_nop);
  m.def("cx_j", &tvbk_cx_j);
  m.def("cx_i", &tvbk_cx_i);
}
