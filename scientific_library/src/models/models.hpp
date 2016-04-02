#pragma once

#include "pybind11/pybind11.h"

#include "gen2d.hpp"
#include "hmje.hpp"
#include "jr.hpp"

PYBIND11_PLUGIN(_models)
{
  pybind11::module mod("_models", "Computational kernels for neural mass models");
  mod.def("gen2d_dfun", &gen2d::dfun, "Compute differential equation RHS of Gen2D model");
  return mod.ptr();
}

// vim: sw=2 et ai
