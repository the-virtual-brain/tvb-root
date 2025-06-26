#include <emscripten/bind.h>

using namespace emscripten;

extern "C" {
#include "tvbk.h"
}

class Conn {
  const tvbk_conn conn;

public:
  Conn() : conn({0}) {}
  void cx_nop(uint32_t t) { tvbk_cx_nop(&conn, t); }
};

EMSCRIPTEN_BINDINGS(tvb_kernels) {
  class_<Conn>("Conn").constructor().function("cx_nop", &Conn::cx_nop);
}
