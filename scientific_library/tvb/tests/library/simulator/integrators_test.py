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
# TODO: evaluate equations?

.. moduleauthor:: Paula Sanz Leon <sanzleon.paula@gmail.com>
.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

"""

if __name__ == "__main__":
    from tvb.tests.library import setup_test_console_env
    setup_test_console_env()
    
import unittest

from tvb.tests.library.base_testcase import BaseTestCase
from tvb.simulator import integrators
from tvb.simulator import noise

# For the moment all integrators inherit dt from the base class
dt = 0.01220703125

class IntegratorsTest(BaseTestCase):
    """
    Define test cases for coupling:
        - initialise each class
        - check default parameters
        - change parameters 
        
    """
    
    def test_integrator_base_class(self):
        integrator = integrators.Integrator()
        self.assertEqual(integrator.dt, dt)
        
        
    def test_heun(self):
        heun_det = integrators.HeunDeterministic()
        heun_sto = integrators.HeunStochastic()
        self.assertEqual(heun_det.dt, dt)
        self.assertEqual(heun_sto.dt, dt)
        self.assertTrue(isinstance(heun_sto.noise, noise.Additive))
        self.assertEqual(heun_sto.noise.nsig, 1.0)
        self.assertEqual(heun_sto.noise.ntau, 0.0)
        
    def test_euler(self):
        euler_det = integrators.EulerDeterministic()
        euler_sto = integrators.EulerStochastic()
        self.assertEqual(euler_det.dt, dt)
        self.assertEqual(euler_sto.dt, dt)
        self.assertTrue(isinstance(euler_sto.noise, noise.Additive))
        self.assertEqual(euler_sto.noise.nsig, 1.0)
        self.assertEqual(euler_sto.noise.ntau, 0.0)
 

    def test_rk4(self):
        rk4 = integrators.RungeKutta4thOrderDeterministic()
        self.assertEqual(rk4.dt, dt)

    def test_identity_scheme(self):
        "Verify identity scheme works"
        x, c, lc, s = 1, 2, 3, 4
        def dfun(x, c, lc):
            return x + c - lc
        integ = integrators.Identity()
        xp1 = integ.scheme(x, dfun, c, lc, s)
        self.assertEqual(xp1, 4)


def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(IntegratorsTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)
