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