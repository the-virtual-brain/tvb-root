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

conn = Connectivity.from_file()

@pytest.mark.parametrize('cv', [3.0, np.inf])
@pytest.mark.parametrize('k', [1, 2])
@pytest.mark.parametrize('nl', [1, 4])
def test_bench_poc(benchmark, nl, k, cv):
    backend = NbbuBackend()
    g, r, V, kernel = backend.prep_poc_bench(conn, nl=nl, nt=100, k=k, cv=cv)
    bkernel = lambda : kernel(r,V,g)
    benchmark(bkernel)


class TestNbbuSim(BaseTestSim):

    def make_sim(self, dt, g_i): 
        integrator = EulerStochastic(dt=dt, noise=Additive(nsig=np.r_[dt]))
        integrator.noise.dt = integrator.dt
        sim = self._create_sim(
            integrator,
            inhom_mmpr=False,
            delays=True,
            run_sim=False,
            conn=conn,
        )
        sim.simulation_length = 1.0
        sim.coupling.a = np.r_[g_i]
        sim.configure()
        return sim
    
    def test_grid(self):
        # setup template sim
        dt = 0.01
        sim = self.make_sim(dt, 0.05)
        conn = sim.connectivity
        nn = conn.weights.shape[0]
        # backend & kernel
        backend = NbbuBackend()
        nl, k, cv, nt = 4, 4, sim.conduction_speed, int(sim.simulation_length/dt)
        np.random.seed(42)
        g, r, V, kernel = backend.prep_poc_bench(conn, nl=nl, nt=nt, k=k, cv=cv, dt=dt)
        g *= 0.05
        # convert history
        nh = 5117
        assert nh == conn.idelays.max() + 1
        for ii in range(16):
            i, j = ii//4, ii%4
            r[i, :, :nh-1, j] = sim.history.buffer[1:, 0, :, 0].T[:,::-1]
            r[i, :, nh, j] = sim.history.buffer[0, 0, :, 0]
            V[i, :, :nh-1, j] = sim.history.buffer[1:, 1, :, 0].T[:,::-1]
            V[i, :, nh, j] = sim.history.buffer[0, 1, :, 0]
        # run both 
        kernel(r,V,g)
        np.random.seed(42)
        state = np.stack([r,V])  # (2, k, nnode, horizon + nstep + 1, nl)
        nht = conn.idelays.max()+1+nt
        assert state.shape == (2, k, nn, nht, nl)
        yh = np.transpose(state[:,3,:,conn.idelays.max()+1:,3], (2, 0, 1))
        (_, y), = sim.run()
        y = y[...,0]
        assert y.shape == yh.shape
        # run tvb on all g values and check median error
        for i, g_ in enumerate(g.flat[:]):
            gsim = self.make_sim(dt, g_)
            (_, y), = gsim.run()
            y = y[...,0]
            yh = np.transpose(state[:,i//4,:,conn.idelays.max()+1:,i%4], (2, 0, 1))
            assert y.shape == yh.shape
            #subplot(4,4,i+1)
            #plot(yh[:,0,0])
            #plot(y[:,0,0])
            me = np.median(np.sum(y[:,0,:] - yh[:,0,:],axis=1)**2)
            print(f'{g_:0.3f} {me:0.1f}')        
            assert me < 2.0
