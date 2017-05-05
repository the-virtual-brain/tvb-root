# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

if __name__ == "__main__":
    from tvb.tests.library import setup_test_console_env
    setup_test_console_env()

import unittest
from tvb.simulator import models
from tvb.tests.library.base_testcase import BaseTestCase
import numpy


class ModelsTest(BaseTestCase):
    """
    Define test cases for models:
        - initialise each class
        TODO - check default parameters (should correspond to those used in the original work with the aim to
         reproduce at least one figure)
        - check that initial conditions are always in range

    """


    def _validate_initialization(self, model, expected_sv, expected_models=1):

        model.configure()
        dt = 2 ** -4
        history_shape = (1, model._nvar, 1, model.number_of_modes)
        model_ic = model.initial(dt, history_shape)
        self.assertEqual(expected_sv, model._nvar)
        self.assertEqual(expected_models, model.number_of_modes)

        svr = model.state_variable_range
        sv = model.state_variables
        for i, (lo, hi) in enumerate([svr[sv[i]] for i in range(model._nvar)]):
            for val in model_ic[:, i, :].flatten():
                self.assertTrue(lo < val < hi)

        state = numpy.zeros((expected_sv, 10, model.number_of_modes))
        obser = model.observe(state)
        self.assertEqual((len(model.variables_of_interest), 10, model.number_of_modes), obser.shape)
        return state, obser

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
            variables_of_interest = ['W', 'W - V']
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


def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(ModelsTest))
    return test_suite



if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)
