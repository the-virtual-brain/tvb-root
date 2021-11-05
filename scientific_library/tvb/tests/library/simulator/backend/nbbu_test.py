# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
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
Tests for the Numba batch-unrolling backend.

"""

from .backendtestbase import BaseTestSim
from tvb.simulator.backend.nbbu import NbbuBackend


# This does a benchmark, not a proper test yet.

if __name__ == '__main__':
    import numba, mako.template, time, tqdm, numpy as np
    numba.config.threading_layer='tbb'
    from tvb.datatypes.connectivity import Connectivity
    conn = Connectivity.from_file()
    weights = conn.weights.astype('f')
    dt = 0.1
    idelays = (conn.tract_lengths / 10 / dt).astype('i')

    backend = NbbuBackend()
    template = '<%include file="nbbu-poc.py.mako"/>'

    nt = 1000
    for nl in [1,16,32]:#,64]: #[16,32]:
        for k in [1,4,8]:#,64,96,128]:
            ns = {'numba': numba}
            delays = backend.build_py_func(
                template, dict(nl=nl, nt=nt), name='delays')
            g = np.linspace(0,1,k*nl).reshape((k,nl))
            r, V = np.random.randn(*(2,k,76,2000, nl)).astype('f')/10.0
            V -= 2.0
            pars = 1.0,1.0,-5.0,15.0,0.0
            print(f'compile {nl}, ', end='', flush=True)
            delays(0.01, r, V, weights, idelays, g, *pars)
            print(f'run {k}: ', end='', flush=True)
            rep = 50
            tic = time.time()
            for i in range(rep):
                delays(0.01, r, V, weights, idelays, g, *pars)
                print('.',end='',flush=True)
            # t = timeit.timeit('delays(out, rnl, weights, idelays, g)', number=rep, globals=globals())
            t = time.time() - tic
            us = (t / rep) * 1e6
            it = nt * nl * k
            print(f'\t{us/it:0.2f} us/it {it/us:0.2f} M it/s {(2*r.nbytes)>>20} MB')
