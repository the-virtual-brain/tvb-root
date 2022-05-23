# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

from numba import cuda, float32

def make_loop(cfun, model, n_svar):
    "Construct CUDA device function for integration loop."

    @cuda.jit
    def loop(n_step, W, X, G):
        # TODO only for 1D grid/block dims
        t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        ti = cuda.threadIdx.x
        x = cuda.shared.array((n_svar, 64), float32)
        g = G[t]
        for j in range(n_step):
            for i in range(W.shape[0]):
                x[:, ti] = X[i, :, t]
                model(x, g * cfun(W, X, i, t))
                X[i, :, t] = x[:, ti]

    # TODO hack
    loop.n_svar = n_svar
    return loop
