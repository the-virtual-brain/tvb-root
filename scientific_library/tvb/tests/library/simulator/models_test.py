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

"""
Test for tvb.simulator.models module

.. moduleauthor:: Paula Sanz Leon <sanzleon.paula@gmail.com>

"""

from tvb.tests.library.base_testcase import BaseTestCase
from tvb.basic.neotraits.api import Final, List
from tvb.simulator import models
from tvb.simulator.models.base import Model
import numpy


class TestBoundsModel(Model):
    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"x1": numpy.array([0.0, 1.0]),
                 "x2": numpy.array([None, 1.0]),
                 "x3": numpy.array([0.0, None]),
                 "x4": numpy.array([-numpy.inf, numpy.inf])
                 },
        doc="""The values for each state-variable should be set to encompass
            the boundaries of the dynamic range of that state-variable. Set None for one-sided boundaries""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('x1', 'x2', 'x3', 'x4', 'x5'),
        default=('x1', 'x2', 'x3', 'x4', 'x5'),
        doc="""default state variables to be monitored""")

    state_variables = ['x1', 'x2', 'x3', 'x4', 'x5']

    state_variable_range = Final(
        default={
            "x1": numpy.array([-1.0, 2.0]),
            "x2": numpy.array([-1.0, 2.0]),
            "x3": numpy.array([-1.0, 2.0]),
            "x4": numpy.array([-1.0, 2.0]),
            "x5": numpy.array([-1.0, 2.0])
        })
    _nvar = 5
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, state, node_coupling, local_coupling=0.0):
        return 0.0 * state


class TestModels(BaseTestCase):
    """
    Define test cases for models:
        - initialise each class
        - check that initial conditions are always in range

    """

    @staticmethod
    def _validate_initialization(model, expected_sv, expected_modes=1):

        model.configure()
        dt = 2 ** -4
        history_shape = (1, model._nvar, 1, model.number_of_modes)
        model_ic = model.initial(dt, history_shape)
        assert expected_sv == model._nvar
        assert expected_modes == model.number_of_modes

        svr = model.state_variable_range
        sv = model.state_variables
        for i, (lo, hi) in enumerate([svr[sv[i]] for i in range(model._nvar)]):
            for val in model_ic[:, i, :].flatten():
                assert (lo < val < hi)

        state = numpy.zeros((expected_sv, 10, model.number_of_modes))
        obser = model.observe(state)
        assert (len(model.variables_of_interest), 10, model.number_of_modes) == obser.shape
        return state, obser

    def test_sv_boundaries_setup(self):
        model = TestBoundsModel()
        model.configure()
        min_float = numpy.finfo("double").min
        max_float = numpy.finfo("double").max
        min_positive = 1.0 / numpy.finfo("single").max
        state_variable_boundaries = \
            {"x1": numpy.array([0.0, 1.0]),
             "x2": numpy.array([min_float, 1.0]),
             "x3": numpy.array([0.0, max_float]),
             "x4": numpy.array([min_float, max_float])}
        for sv, sv_bounds in state_variable_boundaries.items():
            assert numpy.allclose(sv_bounds, model.state_variable_boundaries[sv], min_positive)

    def test_stvar_init(self):
        model = TestBoundsModel()
        model.configure()
        numpy.testing.assert_array_equal(model.stvar, model.cvar)

        model = TestBoundsModel()
        model.stvar=numpy.r_[1,3]
        model.configure()
        numpy.testing.assert_array_equal(model.stvar, numpy.r_[1,3])

        

    def test_wilson_cowan(self):
        """
        Default parameters are taken from figure 4 of [WC_1972]_, pag. 10
        """
        model = models.WilsonCowan()
        self._validate_initialization(model, 2)

    def test_g2d(self):
        """
        Default parameters:

        +---------------------------+
        |  SanzLeonetAl  2013       |
        +--------------+------------+
        |Parameter     |  Value     |
        +==============+============+
        | a            |    - 0.5   |
        +--------------+------------+
        | b            |    -10.0   |
        +--------------+------------+
        | c            |      0.0   |
        +--------------+------------+
        | d            |      0.02  |
        +--------------+------------+
        | I            |      0.0   |
        +--------------+------------+
        |* excitable regime if      |
        |* intrinsic frequency is   |
        |  approx 10 Hz             |
        +---------------------------+


        """
        model = models.Generic2dOscillator()
        state, obser = self._validate_initialization(model, 2)
        numpy.testing.assert_allclose(obser[0], state[0])

    def test_g2d_voi(self):
        model = models.Generic2dOscillator(
            variables_of_interest=('W', 'V - W')
        )
        (V, W), (voi_W, voi_WmV) = self._validate_initialization(model, 2)
        numpy.testing.assert_allclose(voi_W, W)
        numpy.testing.assert_allclose(voi_WmV, W - V)

    def test_jansen_rit(self):
        """
        """
        model = models.JansenRit()
        self._validate_initialization(model, 6)

    def test_sj2d(self):
        """
        """
        model = models.ReducedSetFitzHughNagumo()
        self._validate_initialization(model, 4, 3)

    def test_sj3d(self):
        """
        """
        model = models.ReducedSetHindmarshRose()
        self._validate_initialization(model, 6, 3)

    def test_reduced_wong_wang(self):
        """
        """
        model = models.ReducedWongWang()
        self._validate_initialization(model, 1)

    def test_zetterberg_jansen(self):
        """
        """
        model = models.ZetterbergJansen()
        self._validate_initialization(model, 12)

    def test_epileptor(self):
        """
        """
        model = models.Epileptor()
        self._validate_initialization(model, 6)

    def test_hopfield(self):
        """
        """
        model = models.Hopfield()
        self._validate_initialization(model, 2)

    def test_kuramoto(self):
        """
        """
        model = models.Kuramoto()
        self._validate_initialization(model, 1)

    def test_larter(self):
        """
        """
        model = models.LarterBreakspear()
        self._validate_initialization(model, 3)

    def test_linear(self):
        model = models.Linear()
        self._validate_initialization(model, 1)

    def test_ww(self):
        model = models.ReducedWongWang()
        self._validate_initialization(model, 1)

        model = models.ReducedWongWangExcInh()
        self._validate_initialization(model, 2)
