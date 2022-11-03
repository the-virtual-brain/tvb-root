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
Tests for the theano backend.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import numpy as np
import theano
import theano.tensor as tt
from theano.tensor.random.utils import RandomStream

from tvb.simulator.backend.theano import TheanoBackend
# from tvb.simulator.coupling import Sigmoidal, Linear
# from tvb.simulator.integrators import (EulerDeterministic, EulerStochastic,
#     HeunDeterministic, HeunStochastic, IntegratorStochastic,
#     RungeKutta4thOrderDeterministic, Identity, IdentityStochastic,
#     VODEStochastic)
# from tvb.simulator.noise import Additive, Multiplicative
# from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.models.oscillator import Generic2dOscillator

from .backendtestbase import BaseTestDfun


class TestTheanoDfun(BaseTestDfun):

    def _test_dfun(self, model_, **model_params):
        """Test a Python cfun template."""

        class sim:  # dummy sim
            model = model_

        template = '''
        import numpy as np
        import theano
        import theano.tensor as tt
        <%include file="theano-dfuns.py.mako"/>
        '''
        kernel = TheanoBackend().build_py_func(template, dict(sim=sim, theano=theano, params=model_params),
                                               name='dfuns', print_source=True)

        cX_numpy = np.random.rand(2, 128, 1)
        cX = tt.as_tensor_variable(cX_numpy, name="cX")

        dX = tt.zeros(shape=(2, 128, 1))

        state_numpy = np.random.rand(2, 128, 1)
        state = tt.as_tensor_variable(state_numpy, name="state")

        parmat = sim.model.spatial_parameter_matrix
        dX = kernel(dX, state, cX, parmat)
        np.testing.assert_allclose(dX.eval(),
                                   sim.model.dfun(state_numpy, cX_numpy))

    def test_oscillator(self):
        """Test Generic2dOscillator model"""
        oscillator_model = Generic2dOscillator()
        self._test_dfun(oscillator_model)
