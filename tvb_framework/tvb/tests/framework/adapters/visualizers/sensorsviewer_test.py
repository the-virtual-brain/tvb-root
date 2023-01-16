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
import tvb_data.obj
import tvb_data.sensors
from tvb.adapters.datatypes.db.sensors import SensorsIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.adapters.uploaders.sensors_importer import SensorsImporterModel
from tvb.adapters.visualizers.sensors import SensorsViewer
from tvb.core.entities.filters.chain import FilterChain
from tvb.datatypes.sensors import SensorTypesEnum
from tvb.datatypes.surfaces import SurfaceTypesEnum
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory


class TestSensorViewers(TransactionalTestCase):
    """
    Unit-tests for Sensors viewers.
    """

    EXPECTED_KEYS_INTERNAL = {'urlMeasurePoints': None, 'urlMeasurePointsLabels': None, 'noOfMeasurePoints': 103,
                              'minMeasure': 0, 'maxMeasure': 103, 'urlMeasure': None, 'shellObject': None}

    EXPECTED_KEYS_EEG = EXPECTED_KEYS_INTERNAL.copy()
    EXPECTED_KEYS_EEG.update({'urlVertices': None, 'urlTriangles': None, 'urlLines': None, 'urlNormals': None,
                              'noOfMeasurePoints': 62, 'maxMeasure': 62})

    EXPECTED_KEYS_MEG = EXPECTED_KEYS_EEG.copy()
    EXPECTED_KEYS_MEG.update({'noOfMeasurePoints': 151, 'maxMeasure': 151})

    def transactional_setup_method(self):
        """
        Sets up the environment for running the tests;
        creates a test user, a test project, a connectivity and a surface;
        imports a CFF data-set
        """
        self.test_user = TestFactory.create_user('Sensors_Viewer_User')
        self.test_project = TestFactory.create_project(self.test_user, 'Sensors_Viewer_Project')

    def test_launch_eeg(self):
        """
        Check that all required keys are present in output from EegSensorViewer launch.
        """
        # Import Sensors
        zip_path = os.path.join(os.path.dirname(tvb_data.sensors.__file__), 'eeg_unitvector_62.txt.bz2')
        TestFactory.import_sensors(self.test_user, self.test_project, zip_path,
                                   SensorTypesEnum.TYPE_EEG)
        field = FilterChain.datatype + '.sensors_type'
        filters = FilterChain('', [field], [SensorTypesEnum.TYPE_EEG.value], ['=='])
        sensors_index = TestFactory.get_entity(self.test_project, SensorsIndex, filters)

        # Import EEGCap
        cap_path = os.path.join(os.path.dirname(tvb_data.obj.__file__), 'eeg_cap.obj')
        TestFactory.import_surface_obj(self.test_user, self.test_project, cap_path, SurfaceTypesEnum.EEG_CAP_SURFACE)
        field = FilterChain.datatype + '.surface_type'
        filters = FilterChain('', [field], [SurfaceTypesEnum.EEG_CAP_SURFACE.value], ['=='])
        eeg_cap_surface_index = TestFactory.get_entity(self.test_project, SurfaceIndex, filters)

        viewer = SensorsViewer()
        view_model = viewer.get_view_model_class()()
        view_model.sensors = sensors_index.gid
        viewer.current_project_id = self.test_project.id

        # Launch without EEG Cap
        result = viewer.launch(view_model)
        self.assert_compliant_dictionary(self.EXPECTED_KEYS_EEG, result)

        # Launch with EEG Cap selected
        view_model.shell_surface = eeg_cap_surface_index.gid
        result = viewer.launch(view_model)
        self.assert_compliant_dictionary(self.EXPECTED_KEYS_EEG, result)
        for key in ['urlVertices', 'urlTriangles', 'urlLines', 'urlNormals']:
            assert result[key] is not None, "Value at key %s should not be None" % key

    def test_launch_meg(self):
        """
        Check that all required keys are present in output from MEGSensorViewer launch.
        """

        zip_path = os.path.join(os.path.dirname(tvb_data.sensors.__file__), 'meg_151.txt.bz2')
        TestFactory.import_sensors(self.test_user, self.test_project, zip_path,
                                   SensorTypesEnum.TYPE_MEG)

        field = FilterChain.datatype + '.sensors_type'
        filters = FilterChain('', [field], [SensorTypesEnum.TYPE_MEG.value], ['=='])
        sensors_index = TestFactory.get_entity(self.test_project, SensorsIndex, filters)

        viewer = SensorsViewer()
        viewer.current_project_id = self.test_project.id
        view_model = viewer.get_view_model_class()()
        view_model.sensors = sensors_index.gid

        result = viewer.launch(view_model)
        self.assert_compliant_dictionary(self.EXPECTED_KEYS_MEG, result)

    def test_launch_internal(self):
        """
        Check that all required keys are present in output from InternalSensorViewer launch.
        """
        zip_path = os.path.join(os.path.dirname(tvb_data.sensors.__file__), 'seeg_39.txt.bz2')
        sensors_index = TestFactory.import_sensors(self.test_user, self.test_project, zip_path,
                                                   SensorTypesEnum.TYPE_INTERNAL)
        viewer = SensorsViewer()
        viewer.current_project_id = self.test_project.id
        view_model = viewer.get_view_model_class()()
        view_model.sensors = sensors_index.gid

        result = viewer.launch(view_model)
        self.assert_compliant_dictionary(self.EXPECTED_KEYS_INTERNAL, result)
