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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

"""

import os

import tvb_data.projectionMatrix as dataset
import tvb_data.sensors
import tvb_data.surfaceData
from tvb.adapters.datatypes.db.projections import ProjectionMatrixIndex
from tvb.core.services.exceptions import OperationException
from tvb.datatypes.sensors import SensorTypesEnum
from tvb.datatypes.surfaces import SurfaceTypesEnum
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.tests.framework.core.factory import TestFactory


class TestProjectionMatrix(BaseTestCase):
    """
    Unit-tests for CFF-importer.
    """

    def setup_method(self):
        """
        Reset the database before each test.
        """
        self.test_user = TestFactory.create_user("UserPM")
        self.test_project = TestFactory.create_project(self.test_user)

        zip_path = os.path.join(os.path.dirname(tvb_data.sensors.__file__), 'eeg_brainstorm_65.txt')
        self.sensors = TestFactory.import_sensors(self.test_user, self.test_project, zip_path,
                                                  SensorTypesEnum.TYPE_EEG)

        zip_path = os.path.join(os.path.dirname(tvb_data.surfaceData.__file__), 'cortex_16384.zip')
        self.surface = TestFactory.import_surface_zip(self.test_user, self.test_project, zip_path,
                                                      SurfaceTypesEnum.CORTICAL_SURFACE, True)

    def teardown_method(self):
        """
        Clean-up tests data
        """
        self.clean_database()

    def test_wrong_shape(self):
        """
        Verifies that importing a different shape throws exception
        """
        file_path = os.path.join(os.path.abspath(os.path.dirname(dataset.__file__)),
                                 'projection_eeg_62_surface_16k.mat')

        try:
            TestFactory.import_projection_matrix(self.test_user, self.test_project, file_path, self.sensors.gid,
                                                 self.surface.gid, False)
            raise AssertionError("This was expected not to run! 62 rows in proj matrix, but 65 sensors")
        except OperationException:
            pass

    def test_happy_flow_surface_import(self):
        """
        Verifies the happy flow for importing a surface.
        """
        dt_count_before = TestFactory.get_entity_count(self.test_project, ProjectionMatrixIndex)
        file_path = os.path.join(os.path.abspath(os.path.dirname(dataset.__file__)),
                                 'projection_eeg_65_surface_16k.npy')

        TestFactory.import_projection_matrix(self.test_user, self.test_project, file_path, self.sensors.gid,
                                             self.surface.gid, False)

        dt_count_after = TestFactory.get_entity_count(self.test_project, ProjectionMatrixIndex)
        assert dt_count_before + 1 == dt_count_after
