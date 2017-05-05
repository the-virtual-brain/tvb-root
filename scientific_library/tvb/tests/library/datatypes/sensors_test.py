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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

if __name__ == "__main__":
    from tvb.tests.library import setup_test_console_env
    setup_test_console_env()

import numpy
import unittest
from tvb.datatypes import sensors
from tvb.datatypes.surfaces import SkinAir
from tvb.datatypes.sensors import INTERNAL_POLYMORPHIC_IDENTITY, MEG_POLYMORPHIC_IDENTITY, EEG_POLYMORPHIC_IDENTITY
from tvb.tests.library.base_testcase import BaseTestCase


class SensorsTest(BaseTestCase):
    """
    Tests the defaults for `tvb.datatypes.sensors` module.
    """

    def test_sensors(self):

        dt = sensors.Sensors(load_default=True)
        dt.configure()

        summary_info = dt.summary_info
        self.assertEqual(summary_info['Sensor type'], '')
        self.assertEqual(summary_info['Number of Sensors'], 65)
        self.assertFalse(dt.has_orientation)
        self.assertEqual(dt.labels.shape, (65,))
        self.assertEqual(dt.locations.shape, (65, 3))
        self.assertEqual(dt.number_of_sensors, 65)
        self.assertEqual(dt.orientations.shape, (0,))
        self.assertEqual(dt.sensors_type, '')

        ## Mapping 62 sensors on a Skin surface should work
        surf = SkinAir(load_default=True)
        surf.configure()
        mapping = dt.sensors_to_surface(surf)
        self.assertEqual(mapping.shape, (65, 3))

        ## Mapping on a surface with holes should fail
        dummy_surf = SkinAir()
        dummy_surf.vertices = numpy.array(range(30)).reshape(10, 3).astype('f')
        dummy_surf.triangles = numpy.array(range(9)).reshape(3, 3)
        dummy_surf.configure()
        try:
            dt.sensors_to_surface(dummy_surf)
            self.fail("Should have failed for this simple surface!")
        except Exception:
            pass


    def test_sensorseeg(self):
        dt = sensors.SensorsEEG(load_default=True)
        dt.configure()
        self.assertTrue(isinstance(dt, sensors.SensorsEEG))
        self.assertFalse(dt.has_orientation)
        self.assertEqual(dt.labels.shape, (65,))
        self.assertEqual(dt.locations.shape, (65, 3))
        self.assertEqual(dt.number_of_sensors, 65)
        self.assertEqual(dt.orientations.shape, (0,))
        self.assertEqual(dt.sensors_type, EEG_POLYMORPHIC_IDENTITY)


    def test_sensorsmeg(self):
        dt = sensors.SensorsMEG(load_default=True)
        dt.configure()
        self.assertTrue(isinstance(dt, sensors.SensorsMEG))
        self.assertTrue(dt.has_orientation)
        self.assertEqual(dt.labels.shape, (151,))
        self.assertEqual(dt.locations.shape, (151, 3))
        self.assertEqual(dt.number_of_sensors, 151)
        self.assertEqual(dt.orientations.shape, (151, 3))
        self.assertEqual(dt.sensors_type, MEG_POLYMORPHIC_IDENTITY)


    def test_sensorsinternal(self):
        dt = sensors.SensorsInternal(load_default=True)
        dt.configure()
        self.assertTrue(isinstance(dt, sensors.SensorsInternal))
        self.assertFalse(dt.has_orientation)
        self.assertEqual(dt.labels.shape, (103,))
        self.assertEqual(dt.locations.shape, (103, 3))
        self.assertEqual(dt.number_of_sensors, 103)
        self.assertEqual(dt.orientations.shape, (0,))
        self.assertEqual(dt.sensors_type, INTERNAL_POLYMORPHIC_IDENTITY)



def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(SensorsTest))
    return test_suite



if __name__ == "__main__":
    # So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)
