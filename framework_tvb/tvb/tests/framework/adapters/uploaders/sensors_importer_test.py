# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""

import os
from tvb.core.neocom import h5
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.services.exceptions import OperationException
from tvb.adapters.uploaders.sensors_importer import SensorsImporter, SensorsImporterModel
from tvb.tests.framework.core.factory import TestFactory
import tvb_data.sensors as demo_data


class TestSensorsImporter(TransactionalTestCase):
    """
    Unit-tests for Sensors importer.
    """
    EEG_FILE = os.path.join(os.path.dirname(demo_data.__file__), 'eeg_unitvector_62.txt.bz2')
    MEG_FILE = os.path.join(os.path.dirname(demo_data.__file__), 'meg_151.txt.bz2')

    def transactional_setup_method(self):
        """
        Sets up the environment for running the tests;
        creates a test user, a test project and a `Sensors_Importer`
        """
        self.test_user = TestFactory.create_user('Sensors_User')
        self.test_project = TestFactory.create_project(self.test_user, "Sensors_Project")
        self.importer = SensorsImporter()

    def transactional_teardown_method(self):
        """
        Clean-up tests data
        """
        FilesHelper().remove_project_structure(self.test_project.name)

    def test_import_eeg_sensors(self):
        """
        This method tests import of a file containing EEG sensors.
        """
        eeg_sensors_index = TestFactory.import_sensors(self.test_user, self.test_project, self.EEG_FILE,
                                                       SensorsImporterModel.OPTIONS['EEG Sensors'])

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
                                                       SensorsImporterModel.OPTIONS['MEG Sensors'])

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
                                       SensorsImporterModel.OPTIONS['MEG Sensors'])
            raise AssertionError("Import should fail in case of a MEG import without orientation.")
        except OperationException:
            # Expected exception
            pass

    def test_import_internal_sensors(self):
        """
        This method tests import of a file containing internal sensors.
        """
        internal_sensors_index = TestFactory.import_sensors(self.test_user, self.test_project, self.EEG_FILE,
                                                            SensorsImporterModel.OPTIONS['Internal Sensors'])

        expected_size = 62
        assert expected_size == internal_sensors_index.number_of_sensors

        internal_sensors = h5.load_from_index(internal_sensors_index)

        assert expected_size == len(internal_sensors.labels)
        assert expected_size == len(internal_sensors.locations)
        assert (expected_size, 3) == internal_sensors.locations.shape


