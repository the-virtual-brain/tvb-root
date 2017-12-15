# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""

import os
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.services.exceptions import OperationException
from tvb.core.services.flow_service import FlowService
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.datatypes.sensors import SensorsEEG, SensorsMEG, SensorsInternal
from tvb.adapters.uploaders.sensors_importer import Sensors_Importer
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.datatypes.datatypes_factory import DatatypesFactory
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
        self.datatypeFactory = DatatypesFactory()
        self.test_project = self.datatypeFactory.get_project()
        self.test_user = self.datatypeFactory.get_user()
        self.importer = Sensors_Importer()

    def transactional_teardown_method(self):
        """
        Clean-up tests data
        """
        FilesHelper().remove_project_structure(self.test_project.name)

    def _import(self, import_file_path, sensors_type, expected_data):
        """
        This method is used for importing sensors
        :param import_file_path: absolute path of the file to be imported
        """

        ### Retrieve Adapter instance 
        importer = TestFactory.create_adapter('tvb.adapters.uploaders.sensors_importer', 'Sensors_Importer')

        args = {'sensors_file': import_file_path, 'sensors_type': sensors_type}

        ### Launch import Operation
        FlowService().fire_operation(importer, self.test_user, self.test_project.id, **args)

        data_types = FlowService().get_available_datatypes(self.test_project.id,
                                                           expected_data.module + "." + expected_data.type)[0]
        assert 1 == len(data_types), "Project should contain only one data type = Sensors."

        time_series = ABCAdapter.load_entity_by_gid(data_types[0][2])
        assert time_series is not None, "Sensors instance should not be none"

        return time_series

    def test_import_eeg_sensors(self):
        """
        This method tests import of a file containing EEG sensors.
        """
        eeg_sensors = self._import(self.EEG_FILE, self.importer.EEG_SENSORS, SensorsEEG())

        expected_size = 62
        assert eeg_sensors.labels is not None
        assert expected_size == len(eeg_sensors.labels)
        assert expected_size == len(eeg_sensors.locations)
        assert (expected_size, 3) == eeg_sensors.locations.shape
        assert expected_size == eeg_sensors.number_of_sensors

    def test_import_meg_sensors(self):
        """
        This method tests import of a file containing MEG sensors.
        """
        meg_sensors = self._import(self.MEG_FILE, self.importer.MEG_SENSORS, SensorsMEG())

        expected_size = 151
        assert meg_sensors.labels is not None
        assert expected_size == len(meg_sensors.labels)
        assert expected_size == len(meg_sensors.locations)
        assert (expected_size, 3) == meg_sensors.locations.shape
        assert expected_size == meg_sensors.number_of_sensors
        assert meg_sensors.has_orientation
        assert expected_size == len(meg_sensors.orientations)
        assert (expected_size, 3) == meg_sensors.orientations.shape

    def test_import_meg_without_orientation(self):
        """
        This method tests import of a file without orientation.
        """
        try:
            self._import(self.EEG_FILE, self.importer.MEG_SENSORS, SensorsMEG())
            raise AssertionError("Import should fail in case of a MEG import without orientation.")
        except OperationException:
            # Expected exception
            pass

    def test_import_internal_sensors(self):
        """
        This method tests import of a file containing internal sensors.
        """
        internal_sensors = self._import(self.EEG_FILE, self.importer.INTERNAL_SENSORS, SensorsInternal())

        expected_size = 62
        assert internal_sensors.labels is not None
        assert expected_size == len(internal_sensors.labels)
        assert expected_size == len(internal_sensors.locations)
        assert (expected_size, 3) == internal_sensors.locations.shape
        assert expected_size == internal_sensors.number_of_sensors
