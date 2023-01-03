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

"""
Test for tvb.simulator.coupling module

.. moduleauthor:: Paula Sanz Leon <paula@tvb.invalid>

"""

import copy
import numpy
import pytest

from tvb.simulator.simulator import Simulator
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.simulator import coupling, models, simulator
from tvb.datatypes import cortex, connectivity
from tvb.simulator.history import SparseHistory


class TestCoupling(BaseTestCase):
    """
    Define test cases for coupling:
        - initialise each class
        - check functionality
        
    """
    weights = numpy.array([[0, 1], [1, 0]])

    state_1sv = numpy.array([[[1], [2]]])  # (state_variables, nodes, modes)
    state_2sv = numpy.array([[[1], [2]], [[1], [2]]])

    history_1sv = SparseHistory(weights, weights * 0, numpy.r_[0], 1)
    history_2sv = SparseHistory(weights, weights * 0, numpy.r_[0, 1], 1)

    history_1sv.update(0, state_1sv)
    history_2sv.update(0, state_2sv)

    def _apply_coupling(self, k):
        k.configure()
        return k(0, self.history_1sv)

    def _apply_coupling_2sv(self, k):
        k.configure()
        return k(0, self.history_2sv)

    def test_difference_coupling(self):
        k = coupling.Difference()
        assert k.a, 0.1

        result = self._apply_coupling(k)
        assert result.shape, (1, 2 == 1)  # One state variable, two nodes, one mode

        result = result[0, :, 0].tolist()
        expected_result = [
            k.a * (self.weights[0, 0] * (self.state_1sv[0, 0, 0] - self.state_1sv[0, 0, 0])
                   + self.weights[0, 1] * (self.state_1sv[0, 1, 0] - self.state_1sv[0, 0, 0])),
            k.a * (self.weights[1, 0] * (self.state_1sv[0, 0, 0] - self.state_1sv[0, 1, 0])
                   + self.weights[1, 1] * (self.state_1sv[0, 1, 0] - self.state_1sv[0, 1, 0]))]

        assert self.almost_equal(result, expected_result)

    def test_hyperbolic_coupling(self):
        k = coupling.HyperbolicTangent()
        assert k.a == 1
        assert k.b == 1
        assert k.midpoint == 0
        assert k.sigma == 1
        self._apply_coupling(k)

    def test_kuramoto_coupling(self):
        k = coupling.Kuramoto()
        assert k.a == 1
        self._apply_coupling(k)

    def test_linear_coupling(self):
        k = coupling.Linear()
        assert k.a == 0.00390625
        assert k.b == 0.0
        self._apply_coupling(k)

    def test_pre_sigmoidal_coupling(self):
        k = coupling.PreSigmoidal()
        assert k.H == 0.5
        assert k.Q == 1.
        assert k.G == 60.
        assert k.P == 1.
        assert k.theta == 0.5
        assert k.dynamic is True
        assert k.globalT is False
        self._apply_coupling_2sv(k)

    def test_scaling_coupling(self):
        k = coupling.Scaling()
        # Check scaling -factor
        assert k.a == 0.00390625
        self._apply_coupling(k)

    def test_sigmoidal_coupling(self):
        k = coupling.Sigmoidal()
        assert k.cmin == -1.0
        assert k.cmax == 1.0
        assert k.midpoint == 0.0
        assert k.sigma == 230.
        assert k.a == 1.0
        self._apply_coupling(k)

    def test_sigmoidal_jr_coupling(self):
        k = coupling.SigmoidalJansenRit()
        assert k.cmin == 0.0
        assert k.cmax == 2.0 * 0.0025
        assert k.midpoint == 6.0
        assert k.r == 0.56
        assert k.a == 1.0
        self._apply_coupling_2sv(k)


class TestCouplingShape(BaseTestCase):
    @pytest.mark.slow
    def test_shape(self):

        # try to avoid introspector picking up this model
        Gen2D = copy.deepcopy(models.Generic2dOscillator)

        class CouplingShapeTestModel(Gen2D):
            def __init__(self, test_case=None, n_node=None, **kwds):
                super(CouplingShapeTestModel, self).__init__(**kwds)
                self.cvar = numpy.r_[0, 1]
                self.n_node = n_node
                self.test_case = test_case

            def dfun(self, state, coupling, local_coupling):
                if self.test_case is not None:
                    self.test_case.assert_equal(
                        (2, self.n_node, 1),
                        coupling.shape
                    )
                    return state

        conn = connectivity.Connectivity.from_file()
        surf = cortex.Cortex.from_file()
        surf.region_mapping_data.connectivity = conn
        sim = Simulator(
            model=CouplingShapeTestModel(self, surf.vertices.shape[0]),
            connectivity=conn,
            surface=surf)

        sim.configure()

        for _ in sim(simulation_length=sim.integrator.dt * 2):
            pass
