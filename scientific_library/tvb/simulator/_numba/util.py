# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

import os
import numba
import numba.cuda

# http://numba.pydata.org/numba-doc/dev/cuda/simulator.html
try:
    CUDA_SIM = int(os.environ['NUMBA_ENABLE_CUDASIM']) == 1
except:
    CUDA_SIM = False


_cu_expr_type_map = {
    int: numba.int32,
    float: numba.float32
}


def cu_expr(expr, parameters, constants, return_fn=False):
    "Generate CUDA device function for given expression, with parameters and constants."
    ns = {}
    template = "from math import *\ndef fn(%s):\n    return %s"
    for name, value in constants.items():
        value_type = type(value)
        if value_type not in _cu_expr_type_map:
            msg = "unhandled constant type: %r" % (value_type, )
            raise TypeError(msg)
        ns[name] = _cu_expr_type_map[value_type](value)
    template %= ', '.join(parameters), expr
    exec(template, ns)
    fn = ns['fn']
    cu_fn = numba.cuda.jit(device=True)(fn)
    if return_fn:
        return cu_fn, fn
    return cu_fn


@numba.jit(
    numba.void(numba.double[:, :, :], numba.long_[:], numba.double[:, :, :]),
    nopython=True,
    cache=True,
)
def add_at_313(state_reg, rmap, state):
    """
    numba accelerated version of numpy.add.at
    this is specialized for these shapes:
    state_reg.ndim == 3
    rmap.ndim == 1
    state.ndim == 3
    """
    nreg = state_reg.shape[0]
    nsv = state_reg.shape[1]
    nmode = state_reg.shape[2]
    nvert = state.shape[0]

    assert rmap.shape[0] == nvert
    assert state.shape[1] == nsv
    assert state.shape[2] == nmode

    # this is costly, and dropped by the unsafe version
    for vi in range(nvert):
        reg_id = rmap[vi]
        assert 0 <= reg_id < nreg

    for vi in range(nvert):
        reg_id = rmap[vi]

        for svi in range(nsv):
            for mi in range(nmode):
                state_reg[reg_id, svi, mi] += state[vi, svi, mi]

@numba.jit(
    numba.void(numba.double[:, :, :], numba.long_[:], numba.double[:, :, :]),
    nopython=True,
    cache=True,
)
def add_at_313_unsafe(state_reg, rmap, state):
    """
    numba accelerated version of numpy.add.at
    this is specialized for these shapes:
    state_reg.ndim == 3
    rmap.ndim == 1
    state.ndim == 3

    Unsafe version of add_at_313.
    You *MUST* GUARANTEE the following or we SEGFAULT !

    assert 0 <= rmap[i] < nreg

    """
    nreg = state_reg.shape[0]
    nsv = state_reg.shape[1]
    nmode = state_reg.shape[2]
    nvert = state.shape[0]

    assert rmap.shape[0] == nvert
    assert state.shape[1] == nsv
    assert state.shape[2] == nmode

    for vi in range(nvert):
        reg_id = rmap[vi]

        for svi in range(nsv):
            for mi in range(nmode):
                state_reg[reg_id, svi, mi] += state[vi, svi, mi]

