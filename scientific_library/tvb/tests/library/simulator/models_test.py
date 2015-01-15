# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
import numpy
from numpy.testing import assert_array_almost_equal

from tvb.tests.library.base_testcase import BaseTestCase
from tvb.simulator import models


# NOTE: Comparing arrays of floating point numbers is, to say the least,
#        akward and generally ill posed. Numpy provides a testing suite to compare
#        arrays.
# TODO: use numpy.testing.ssert_array_almost_equal_nulp or assert_array_max_ulp
# which are built for floating point comparisons.

dt = 2**-4

class ModelsTest(BaseTestCase):
    """
    Define test cases for models:
        - initialise each class
        - check default parameters
         (should correspond to those used in the original work with the aim to
         reproduce at least one figure)
        - check that initial conditions are always the same. Fails if seed != 42
          or sv ranges are changed or intial method is overwritten... mmm


    """

    def test_wilson_cowan(self):
        """
        Default parameters are taken from figure 4 of [WC_1972]_, pag. 10
        """
        model = models.WilsonCowan()
        history_shape = (1, model._nvar, 1, model.number_of_modes)
        model_ic = model.initial(dt, history_shape)
        self.assertEqual(model._nvar, 2)
        assert_array_almost_equal(model_ic, numpy.array([[[[ 0.49023095]],
                                                          [[ 0.49023095]]]]))

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
        history_shape = (1, model._nvar, 1, model.number_of_modes)
        model_ic = model.initial(dt, history_shape)
        self.assertEqual(model._nvar, 2)
        assert_array_almost_equal(model_ic, numpy.array([[[[ 0.97607082]],
                                                          [[-0.03384097]]]]))


    def test_jansen_rit(self):
        """
        """
        model = models.JansenRit()
        history_shape = (1, model._nvar, 1, model.number_of_modes)
        model_ic = model.initial(dt, history_shape)
        self.assertEqual(model._nvar, 6)
        assert_array_almost_equal(model_ic, numpy.array([[[[-0.01381552]],
                                                           [[-0.30892439]],
                                                           [[-0.09769047]],
                                                           [[-0.03384097]],
                                                           [[-0.06178488]],
                                                           [[-0.30892439]]]]))


    def test_sj2d(self):
        """
        """
        model = models.ReducedSetFitzHughNagumo()
        history_shape = (1, model._nvar, 1, model.number_of_modes)
        model_ic = model.initial(dt, history_shape)
        self.assertEqual(model._nvar, 4)
        self.assertEqual(model.number_of_modes, 3)
        assert_array_almost_equal(model_ic, numpy.array([[[[-0.02763104,  0.14269546,  0.0534609 ]],
                                                          [[-0.02392918,  0.12357789,  0.0462985 ]],
                                                          [[-0.02763104,  0.14269546,  0.0534609 ]],
                                                          [[-0.02392918,  0.12357789,  0.0462985 ]]]]))


    def test_sj3d(self):
        """
        """
        model = models.ReducedSetHindmarshRose()
        history_shape = (1, model._nvar, 1, model.number_of_modes)
        model_ic = model.initial(dt, history_shape)
        self.assertEqual(model._nvar, 6)
        self.assertEqual(model.number_of_modes, 3)
        assert_array_almost_equal(model_ic, numpy.array([[[[-0.02763104,  0.14269546,  0.0534609 ]],
                                                          [[-2.56553276, -2.16156801, -2.37320634]],
                                                          [[ 5.97236896,  6.14269546,  6.0534609 ]],
                                                          [[-0.02763104,  0.14269546,  0.0534609 ]],
                                                          [[-0.06178488,  0.31907674,  0.1195422 ]],
                                                          [[ 5.97236896,  6.14269546,  6.0534609 ]]]]))

    def test_reduced_wong_wang(self):
        """
        """
        model = models.ReducedWongWang()
        history_shape = (1, model._nvar, 1, model.number_of_modes)
        model_ic = model.initial(dt, history_shape)
        self.assertEqual(model._nvar, 1)
        assert_array_almost_equal(model_ic, numpy.array([[[[ 0.49023095]]]]))


    def test_zetterberg_jansen(self):
        """
        """
        model = models.ZetterbergJansen()
        history_shape = (1, model._nvar, 1, model.number_of_modes)
        model_ic = model.initial(dt, history_shape)
        self.assertEqual(model._nvar, 12)
        assert_array_almost_equal(model_ic, numpy.array([[[[ -0.13815519]],
                                                          [[ -0.30892439]],
                                                          [[-25.1196459 ]],
                                                          [[-47.10057849]],
                                                          [[-47.10057849]],
                                                          [[-47.10057849]],
                                                          [[-40.10701455]],
                                                          [[-40.10701455]],
                                                          [[-40.10701455]],
                                                          [[ -0.30892439]],
                                                          [[-40.10701455]],
                                                          [[-40.10701455]]]]))



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
