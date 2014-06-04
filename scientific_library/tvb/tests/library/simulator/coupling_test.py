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
Test for tvb.simulator.coupling module

.. moduleauthor:: Paula Sanz Leon <sanzleon.paula@gmail.com>

"""

if __name__ == "__main__":
    from tvb.tests.library import setup_test_console_env

    setup_test_console_env()

import unittest

from tvb.tests.library.base_testcase import BaseTestCase
from tvb.simulator import coupling



class CouplingTest(BaseTestCase):
    """
    Define test cases for coupling:
        - initialise each class
        - check functionality
        
    """


    def test_linear_coupling(self):
        k = coupling.Linear()
        self.assertEqual(k.a, 0.00390625)
        self.assertEqual(k.b, 0.0)


    def test_scaling_coupling(self):
        k = coupling.Scaling()
        # Check scaling -factor
        self.assertEqual(k.a, 0.00390625)


    def test_sigmoidal_coupling(self):
        k = coupling.Sigmoidal()
        self.assertEqual(k.cmin, -1.0)
        self.assertEqual(k.cmax, 1.0)
        self.assertEqual(k.midpoint, 0.0)
        self.assertEqual(k.sigma, 230.)
        self.assertEqual(k.a, 1.0)



def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(CouplingTest))
    return test_suite



if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE) 