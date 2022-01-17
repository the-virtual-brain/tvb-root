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

import numpy as np
import pytest
from .backendtestbase import BaseTestSim
from tvb.simulator.backend.nbbu import NbbuBackend
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.integrators import EulerStochastic
from tvb.simulator.noise import Additive


@pytest.mark.parametrize('cv', [3.0, np.inf])
@pytest.mark.parametrize('k', [1, 8])
@pytest.mark.parametrize('nl', [1, 8])
def test_bench_poc(benchmark, nl, k, cv):
    conn = Connectivity.from_file()
    backend = NbbuBackend()
    g, r, V, kernel = backend.prep_poc_bench(conn, nl=nl, nt=100, k=k, cv=cv)
    benchmark(kernel)


class TestNbbuSim(BaseTestSim):

    def test_poc(self):
        dt = 0.01
        integrator = EulerStochastic(dt=dt, noise=Additive(nsig=np.r_[dt]))
        integrator.noise.dt = integrator.dt
        sim = self._create_sim(
            integrator,
            inhom_mmpr=False,
            delays=True,
            run_sim=False
        )
        sim.coupling.a = np.r_[0.0]
        conn = sim.connectivity
        nn = conn.weights.shape[0]
        backend = NbbuBackend()
        nl, k, cv, nt = 4, 4, sim.conduction_speed, 10
        g, r, V, kernel = backend.prep_poc_bench(conn, nl=nl, nt=10, k=k, cv=cv)
        kernel()
        np.random.seed(42)
        state = np.stack([r,V])  # (2, k, nnode, horizon + nstep + 1, nl)
        nht = conn.idelays.max()+2+nt
        assert state.shape == (2, k, nn, nht, nl)
        yh = np.transpose(state[:,0,:,conn.idelays.max()+2:,0], (2, 0, 1))
        (_, y), = sim.run()
        y = y[...,0]
        assert y.shape == yh.shape
        np.testing.assert_allclose(yh, y)
