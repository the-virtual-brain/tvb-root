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

"""
A Numba backend which handles inner and outer batching via
unrolling and numba.prange respectively.  This backend is
ideal for CPU parameter sweeps: when benchmarked across a
range of machines, this approach achieves a significant
portion or exceeds the CPU's state memory bandwidth; on an
i7-8665U, it is nearly 50x faster throughput than single
threaded Numba backend.

... moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""


from .nb import NbBackend
import numba, mako.template, time, tqdm, numpy as np

class NbbuBackend(NbBackend):

    def prep_poc_bench(self, conn, dt=0.1, nt=1000, cv=10.0, nl=1, k=1, pars=(1.0,1.0,-5.0,15.0,0.0)):
        # prep connectome
        weights = conn.weights.astype('f')
        idelays = (conn.tract_lengths / cv / dt).astype('i')
        nh = idelays.max() + 1
        nn = weights.shape[0]
        # prep template
        template = '<%include file="nbbu-poc.py.mako"/>'
        params = {'nl': nl, 'nt': nt, 'nh': nh}
        delays = self.build_py_func(template, params, name='delays')
        # prep arrays for one chunk of sim
        g = np.linspace(0,1,k*nl).reshape((k,nl))
        r, V = np.random.randn(2,k,nn,nh+nt+1, nl).astype('f')/10.0
        V -= 2.0
        return lambda : delays(0.01, r, V, weights, idelays, g, *pars)
