# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import numpy
from tvb.datatypes.sensors import SensorTypesEnum

from tvb.tests.library.base_testcase import BaseTestCase
from tvb.datatypes import sensors
from tvb.datatypes.surfaces import SkinAir


class TestSensors(BaseTestCase):
    """
    Tests the defaults for `tvb.datatypes.sensors` module.
    """

    def test_sensors(self):

        dt = sensors.Sensors.from_file()
        dt.configure()

        summary_info = dt.summary_info()
        assert summary_info['Sensor type'] is None
        assert summary_info['Number of Sensors'] == 65
        assert not dt.has_orientation
        assert dt.labels.shape == (65,)
        assert dt.locations.shape == (65, 3)
        assert dt.number_of_sensors == 65
        assert dt.orientations is None
        assert dt.sensors_type is None

        ## Mapping 62 sensors on a Skin surface should work
        surf = SkinAir.from_file()
        surf.configure()
        mapping = dt.sensors_to_surface(surf)
        assert mapping.shape == (65, 3)

        ## Mapping on a surface with holes should fail
        dummy_surf = SkinAir()
        dummy_surf.vertices = numpy.array(list(range(30))).reshape(10, 3).astype('f')
        dummy_surf.triangles = numpy.array(list(range(9))).reshape(3, 3)
        dummy_surf.triangle_normals = numpy.array(list(range(9))).reshape(3, 3)
        dummy_surf.vertex_normals = numpy.array(list(range(30))).reshape(10, 3).astype('f')
        dummy_surf.configure()

        try:
            dt.sensors_to_surface(dummy_surf)
            pytest.fail("Should have failed for this simple surface!")
        except Exception:
            pass

    def test_sensorseeg(self):
        dt = sensors.SensorsEEG.from_file()
        dt.configure()
        assert isinstance(dt, sensors.SensorsEEG)
        assert not dt.has_orientation
        assert dt.labels.shape == (65,)
        assert dt.locations.shape == (65, 3)
        assert dt.number_of_sensors == 65
        assert dt.orientations is None
        assert dt.sensors_type == SensorTypesEnum.TYPE_EEG.value

    def test_sensorsmeg(self):
        dt = sensors.SensorsMEG.from_file()
        dt.configure()
        assert isinstance(dt, sensors.SensorsMEG)
        assert dt.has_orientation
        assert dt.labels.shape == (151,)
        assert dt.locations.shape == (151, 3)
        assert dt.number_of_sensors == 151
        assert dt.orientations.shape == (151, 3)
        assert dt.sensors_type == SensorTypesEnum.TYPE_MEG.value

    def test_sensorsinternal(self):
        dt = sensors.SensorsInternal.from_file()
        dt.configure()
        assert isinstance(dt, sensors.SensorsInternal)
        assert not dt.has_orientation
        assert dt.labels.shape == (103,)
        assert dt.locations.shape == (103, 3)
        assert dt.number_of_sensors == 103
        assert dt.orientations is None
        assert dt.sensors_type == SensorTypesEnum.TYPE_INTERNAL.value
