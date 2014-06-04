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
from numpy.testing import assert_array_almost_equal as ArrayAlmostEqual

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
    Define test cases for coupling:
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
        # self.assertTrue(ArrayAlmostEqual(model_ic, numpy.array([[[[ 0.4527827 ]],
        #                                                          [[ 0.30644616]]]])) 
        #                                                          is None)
        
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
        #self.assertTrue(ArrayAlmostEqual(model_ic, numpy.array([[[[ 1.70245989]], [[ 0.2765286 ]]]])) is None)
        

    def test_jansen_rit(self):
        """
        """
        model = models.JansenRit()
        history_shape = (1, model._nvar, 1, model.number_of_modes)
        model_ic = model.initial(dt, history_shape)
        self.assertEqual(model._nvar, 6)
        # self.assertTrue(ArrayAlmostEqual(model_ic, numpy.array([[[[ 0.40556541]],[[ 2.52434922]],
        #                                                          [[ 3.73943152]],[[ 3.04605971]],
        #                                                          [[ 0.85500724]],[[ 4.27473643]]]])) 
        #                                                          is None)
        
        
    def test_sj2d(self):
        """
        """
        model = models.ReducedSetFitzHughNagumo()
        history_shape = (1, model._nvar, 1, model.number_of_modes)
        model_ic = model.initial(dt, history_shape)
        self.assertEqual(model._nvar, 4)
        self.assertEqual(model.number_of_modes, 3)
        # self.assertTrue(ArrayAlmostEqual(model_ic, numpy.array([[[[ 0.81113082, 0.22578466, 1.05767095]],
        #                                                          [[ 2.15388948, 0.33114288, 0.33111966]],
        #                                                          [[ 2.57884373, 1.25321566, 0.76664846]],
        #                                                          [[ 0.76729577, 0.65537159, 0.65864133]]]])) 
        #                                                          is None)
        
        
    def test_sj3d(self):
        """
        """
        model = models.ReducedSetHindmarshRose()
        history_shape = (1, model._nvar, 1, model.number_of_modes)
        model_ic = model.initial(dt, history_shape)
        self.assertEqual(model._nvar, 6)
        self.assertEqual(model.number_of_modes, 3)
        # self.assertTrue(ArrayAlmostEqual(model_ic, numpy.array([[[[ 0.81113082, 0.22578466, 1.05767095]],
        #                                                          [[ 3.39866927, -1.59312788, -1.59319147]],
        #                                                          [[ 8.57884373, 7.25321566, 6.76664846]],
        #                                                          [[ 0.88599684, 0.75675792, 0.7605335 ]],
        #                                                          [[ 0.88352129, 6.98631166, 6.29850938]],
        #                                                          [[ 6.91821169, 7.65394629, 6.51316375]]]])) 
        #                                                          is None)
    
    def test_reduced_wong_wang(self):
        """
        """
        model = models.ReducedWongWang()
        history_shape = (1, model._nvar, 1, model.number_of_modes)
        model_ic = model.initial(dt, history_shape)
        self.assertEqual(model._nvar, 1)
        #self.assertTrue(ArrayAlmostEqual(model_ic, numpy.array([[[[ 0.78677805]]]])) is None)
                                        

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