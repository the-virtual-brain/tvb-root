# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

import numpy
from numba import cuda, float32, guvectorize, float64


def make_bistable():
    "Construct CUDA device function for a bistable system."

    @cuda.jit(device=True)
    def f(dX, X, I):
        t = cuda.threadIdx.x
        x = X[0, t]
        dX[0, t] = (x - x*x*x - float32(1.0) + I) / float32(50.0)

    return f


def make_jr():
    "Construct CUDA device function for the Jansen-Rit model."

    # parameters
    A   ,    B,   a,    b,   v0, nu_max,    r,     J, a_1, a_2,  a_3,  a_4, p_min, p_max,   mu = list(map(float32, [
    3.25, 22.0, 0.1, 0.05, 5.52, 0.0025, 0.56, 135.0, 1.0, 0.8, 0.25, 0.25,  0.12,  0.32, 0.22]))

    # jit maps this to CUDA exp
    from math import exp

    @cuda.jit(device=True)
    def f(dX, X, I):
        one, two = float32(1.0), float32(2.0)
        t = cuda.threadIdx.x
        dX[0, t] = X[3, t]
        dX[1, t] = X[4, t]
        dX[2, t] = X[5, t]
        dX[3, t] = A * a * two * nu_max / (one + exp(r * (v0 - (X[1, t] - X[2, t])))) - float32(2.0) * a * X[3, t] - a * a * X[0, t]
        dX[4, t] = A * a * (mu + a_2 * J * two * nu_max / (one + exp(r * (v0 - (a_1 * J * X[0, t])))) + I) - two * a * X[4, t] - a * a * X[1, t]
        dX[5, t] = B * b * (a_4 * J * two * nu_max / (one + exp(r * (v0 - (a_3 * J * X[0, t]))))) - two * b * X[5, t] - b * b * X[2, t]

    return f