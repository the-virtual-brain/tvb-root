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
Tests for the CUDA backend.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import unittest
import numpy as np

from tvb.simulator.backend.cu import CuBackend, pycuda_available
if pycuda_available:  # quickfix
    from tvb.simulator.backend.cu import Out, In, InOut
from tvb.simulator.coupling import Sigmoidal, Linear
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin

from .backendtestbase import BaseTestSim, BaseTestCoupling, BaseTestDfun


class TestCUSim(BaseTestSim):

    @unittest.skipUnless(pycuda_available, 'requires working PyCUDA')
    def test_mpr(self):
        "Test generated CUDA kernel directly from Simulator instance."
        sim, state, t, y = self._create_sim(inhom_mmpr=True)
        template = '<%include file="cu-sim-ode.cu.mako"/>'
        kernel = CuBackend().build_func(template, dict(sim=sim, pi=np.pi))
        dX = state.copy()
        weights = sim.connectivity.weights.T.copy().astype('f')
        parmat = sim.model.spatial_parameter_matrix.astype('f')
        yh = np.empty((len(t),)+state.shape, 'f')
        kernel(
            In(state), In(weights), Out(yh), In(parmat),
            grid=(1,1), block=(128,1,1))
        self._check_match(y, yh[:,:,0])


class TestCUCoupling(BaseTestCoupling):

    def _test_cfun(self, cfun):
        "Test CUDA cfun template."
        class sim:  # dummy
            model = MontbrioPazoRoxin()
            coupling = cfun
        template = '''
<%include file="cu-coupling.cu.mako"/>
__global__ void kernel(float *state, float *weights, float *cX) {
    coupling(threadIdx.x, ${n_node}, cX, weights, state);
}
'''
        content = dict(n_node=128, sim=sim)
        kernel = CuBackend().build_func(template, content)
        state = np.random.rand(2, content['n_node']).astype('f')
        weights = np.random.randn(state.shape[1], state.shape[1]).astype('f')
        cX = np.empty_like(state)
        kernel(In(state), In(weights), Out(cX), 
            grid=(1,1), block=(content['n_node'],1,1))
        expected = self._eval_cfun_no_delay(sim.coupling, weights, state)
        np.testing.assert_allclose(cX, expected, 1e-5, 1e-6)

    @unittest.skipUnless(pycuda_available, 'requires working PyCUDA')
    def test_linear(self):
        self._test_cfun(Linear())

    @unittest.skipUnless(pycuda_available, 'requires working PyCUDA')
    def test_sigmoidal(self):
        self._test_cfun(Sigmoidal())


class TestCUDfun(BaseTestDfun):

    def _test_model(self, model_):
        "Test CUDA model dfuns."
        class sim:  # dummy
            model = model_
        template = '''

#define M_PI_F 3.14159265358979f

<%include file="cu-dfuns.cu.mako"/>
__global__ void kernel(float *dX, float *state, float *cX, float *parmat) {
    dfuns(threadIdx.x, ${n_node}, dX, state, cX, parmat);
}
'''
        content = dict(n_node=128, sim=sim)
        kernel = CuBackend().build_func(template, content, print_source=True)
        dX, state, cX = np.random.rand(3, 2, content['n_node']).astype('f')
        parmat = sim.model.spatial_parameter_matrix.astype('f')
        if parmat.size == 0:
            parmat = np.zeros((1,),'f') # dummy
        kernel(Out(dX), In(state), In(cX), In(parmat), 
            grid=(1,1), block=(content['n_node'],1,1))
        expected = sim.model.dfun(state, cX)
        np.testing.assert_allclose(dX, expected, 1e-3, 1e-5)

    @unittest.skipUnless(pycuda_available, 'requires working PyCUDA')
    def test_mpr_symmetric(self):
        self._test_model(self._prep_model())

    @unittest.skipUnless(pycuda_available, 'requires working PyCUDA')
    def test_mpr_spatial1(self):
        "Test MPR w/ 1 spatial parameter."
        self._test_model(self._prep_model(1))

    @unittest.skipUnless(pycuda_available, 'requires working PyCUDA')
    def test_mpr_spatial2(self):
        "Test MPR w/ 2 spatial parameters."
        self._test_model(self._prep_model(2))
