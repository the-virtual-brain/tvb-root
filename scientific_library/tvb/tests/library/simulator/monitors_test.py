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
Check monitors.
# TODO: +test - configure a simulator and check that the period is indeed sim.int.dt
        +test - check if data reduction works in avg monitors 

.. moduleauthor:: Paula Sanz Leon <sanzleon.paula@gmail.com>
"""

# NOTE: Just checking if they are correctly initialised, though we need to create
# tests that actually check their functionality.

if __name__ == "__main__":
    from tvb.tests.library import setup_test_console_env
    setup_test_console_env()
    
import unittest

from tvb.tests.library.base_testcase import BaseTestCase
from tvb.simulator import monitors
from tvb.datatypes import sensors


default_period = 0.9765625  # 1024Hz


class MonitorsTest(BaseTestCase):
    """
    Define test cases for monitors:
        - initialise each class
        - check default parameters (period)
        - 
    """
    
    def test_monitor_raw(self):
        monitors.Raw()
    
    
    def test_monitor_tavg(self):
        monitor = monitors.TemporalAverage()
        self.assertEqual(monitor.period, default_period)
        
        
    def test_monitor_gavg(self):
        monitor = monitors.GlobalAverage()
        self.assertEqual(monitor.period, default_period)
        
        
    def test_monitor_savg(self):
        monitor = monitors.SpatialAverage()
        self.assertEqual(monitor.period, default_period)
        
        
    def test_monitor_subsample(self):
        monitor = monitors.SubSample()
        self.assertEqual(monitor.period, default_period)
    

    def test_monitor_eeg(self):
        monitor = monitors.EEG()
        self.assertEqual(monitor.period, default_period)


    def test_monitor_smeg(self):
        """
        This has to be verified.
        """
        monitor = monitors.SphericalMEG()
        self.assertEqual(monitor.period, default_period)
     
     
    def test_monitor_seeg(self):
        """
        This has to be verified.
        """
        monitor = monitors.SphericalEEG()
        self.assertEqual(monitor.period, default_period)
        
        
    def test_monitor_stereoeeg(self):
        """
        This has to be verified.
        """
        monitor = monitors.SEEG()
        monitor.sensors = sensors.SensorsInternal(load_default=True)
        self.assertEqual(monitor.period, default_period)


    def test_monitor_bold(self):
        """
        This has to be verified.
        """
        monitor = monitors.Bold()
        self.assertEqual(monitor.period, 2000.0)
        
        
class MonitorsConfigurationTest(BaseTestCase):
    """
    Configure Monitors
    
    """
    def test_monitor_bold(self):
        """
        This has to be verified.
        """
        monitor = monitors.Bold()
        self.assertEqual(monitor.period, 2000.0)
        
        
def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(MonitorsTest))
    test_suite.addTest(unittest.makeSuite(MonitorsConfigurationTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE) 