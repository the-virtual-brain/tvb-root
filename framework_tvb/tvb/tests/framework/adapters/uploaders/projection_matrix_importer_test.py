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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

"""

import os
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
import tvb_data.sensors
import tvb_data.surfaceData
import tvb_data.projectionMatrix as dataset
from tvb.adapters.uploaders.sensors_importer import Sensors_Importer
from tvb.tests.framework.core.factory import TestFactory
from tvb.core.services.flow_service import FlowService
from tvb.core.services.exceptions import OperationException
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.datatypes.projections import ProjectionSurfaceEEG
from tvb.datatypes.sensors import SensorsEEG
from tvb.datatypes.surfaces import CorticalSurface, CORTICAL


class TestProjectionMatrix(TransactionalTestCase):
    """
    Unit-tests for CFF-importer.
    """
    
    def transactional_setup_method(self):
        """
        Reset the database before each test.
        """
        self.test_user = TestFactory.create_user("UserPM")
        self.test_project = TestFactory.create_project(self.test_user)

        zip_path = os.path.join(os.path.dirname(tvb_data.sensors.__file__), 'eeg_brainstorm_65.txt')
        TestFactory.import_sensors(self.test_user, self.test_project, zip_path, Sensors_Importer.EEG_SENSORS)

        zip_path = os.path.join(os.path.dirname(tvb_data.surfaceData.__file__), 'cortex_16384.zip')
        TestFactory.import_surface_zip(self.test_user, self.test_project, zip_path, CORTICAL, True)

        self.surface = TestFactory.get_entity(self.test_project, CorticalSurface())
        assert self.surface is not None
        self.sensors = TestFactory.get_entity(self.test_project, SensorsEEG())
        assert self.sensors is not None

        self.importer = TestFactory.create_adapter('tvb.adapters.uploaders.projection_matrix_importer',
                                                   'ProjectionMatrixSurfaceEEGImporter')


    def transactional_teardown_method(self):
        """
        Clean-up tests data
        """
        FilesHelper().remove_project_structure(self.test_project.name)


    def test_wrong_shape(self):
        """
        Verifies that importing a different shape throws exception
        """
        file_path = os.path.join(os.path.abspath(os.path.dirname(dataset.__file__)),
                                 'projection_eeg_62_surface_16k.mat')
        args = {'projection_file': file_path,
                'dataset_name': 'ProjectionMatrix',
                'sensors': self.sensors.gid,
                'surface': self.surface.gid,
                DataTypeMetaData.KEY_SUBJECT: DataTypeMetaData.DEFAULT_SUBJECT}

        try:
            FlowService().fire_operation(self.importer, self.test_user, self.test_project.id, **args)
            raise AssertionError("This was expected not to run! 62 rows in proj matrix, but 65 sensors")
        except OperationException:
            pass


    def test_happy_flow_surface_import(self):
        """
        Verifies the happy flow for importing a surface.
        """
        dt_count_before = TestFactory.get_entity_count(self.test_project, ProjectionSurfaceEEG())
        file_path = os.path.join(os.path.abspath(os.path.dirname(dataset.__file__)),
                                 'projection_eeg_65_surface_16k.npy')
        args = {'projection_file': file_path,
                'dataset_name': 'ProjectionMatrix',
                'sensors': self.sensors.gid,
                'surface': self.surface.gid,
                DataTypeMetaData.KEY_SUBJECT: DataTypeMetaData.DEFAULT_SUBJECT}

        FlowService().fire_operation(self.importer, self.test_user, self.test_project.id, **args)
        dt_count_after = TestFactory.get_entity_count(self.test_project, ProjectionSurfaceEEG())

        assert dt_count_before + 1 == dt_count_after
    
