# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""

import os

import tvb_data.sensors as demo_data
from tvb.adapters.uploaders.sensors_importer import SensorsImporter, SensorsImporterModel
from tvb.core.neocom import h5
from tvb.core.services.exceptions import OperationException
from tvb.datatypes.sensors import SensorTypesEnum
from tvb.storage.storage_interface import StorageInterface
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.tests.framework.core.factory import TestFactory


class TestSensorsImporter(BaseTestCase):
    """
    Unit-tests for Sensors importer.
    """
    EEG_FILE = os.path.join(os.path.dirname(demo_data.__file__), 'eeg_unitvector_62.txt.bz2')
    MEG_FILE = os.path.join(os.path.dirname(demo_data.__file__), 'meg_151.txt.bz2')

    def setup_method(self):
        """
        Sets up the environment for running the tests;
        creates a test user, a test project and a `Sensors_Importer`
        """
        self.test_user = TestFactory.create_user('Sensors_User')
        self.test_project = TestFactory.create_project(self.test_user, "Sensors_Project")
        self.importer = SensorsImporter()

    def teardown_method(self):
        """
        Clean-up tests data
        """
        self.clean_database()

    def test_import_eeg_sensors(self):
        """
        This method tests import of a file containing EEG sensors.
        """
        eeg_sensors_index = TestFactory.import_sensors(self.test_user, self.test_project, self.EEG_FILE,
                                                       SensorTypesEnum.TYPE_EEG, False)

        expected_size = 62
        assert expected_size == eeg_sensors_index.number_of_sensors

        eeg_sensors = h5.load_from_index(eeg_sensors_index)

        assert expected_size == len(eeg_sensors.labels)
        assert expected_size == len(eeg_sensors.locations)
        assert (expected_size, 3) == eeg_sensors.locations.shape

    def test_import_meg_sensors(self):
        """
        This method tests import of a file containing MEG sensors.
        """
        meg_sensors_index = TestFactory.import_sensors(self.test_user, self.test_project, self.MEG_FILE,
                                                       SensorTypesEnum.TYPE_MEG, False)

        expected_size = 151
        assert expected_size == meg_sensors_index.number_of_sensors

        meg_sensors = h5.load_from_index(meg_sensors_index)

        assert expected_size == len(meg_sensors.labels)
        assert expected_size == len(meg_sensors.locations)
        assert (expected_size, 3) == meg_sensors.locations.shape
        assert meg_sensors.has_orientation
        assert expected_size == len(meg_sensors.orientations)
        assert (expected_size, 3) == meg_sensors.orientations.shape

    def test_import_meg_without_orientation(self):
        """
        This method tests import of a file without orientation.
        """
        try:
            TestFactory.import_sensors(self.test_user, self.test_project, self.EEG_FILE,
                                       SensorTypesEnum.TYPE_MEG, False)
            raise AssertionError("Import should fail in case of a MEG import without orientation.")
        except OperationException:
            # Expected exception
            pass

    def test_import_internal_sensors(self):
        """
        This method tests import of a file containing internal sensors.
        """
        internal_sensors_index = TestFactory.import_sensors(self.test_user, self.test_project, self.EEG_FILE,
                                                            SensorTypesEnum.TYPE_INTERNAL, False)

        expected_size = 62
        assert expected_size == internal_sensors_index.number_of_sensors

        internal_sensors = h5.load_from_index(internal_sensors_index)

        assert expected_size == len(internal_sensors.labels)
        assert expected_size == len(internal_sensors.locations)
        assert (expected_size, 3) == internal_sensors.locations.shape
