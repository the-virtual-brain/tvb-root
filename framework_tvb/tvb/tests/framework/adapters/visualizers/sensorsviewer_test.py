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
import tvb_data.obj
import tvb_data.sensors
from tvb.adapters.uploaders.sensors_importer import Sensors_Importer
from tvb.adapters.visualizers.sensors import SensorsViewer
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.datatypes.sensors import SensorsEEG, SensorsMEG, SensorsInternal
from tvb.datatypes.surfaces import EEGCap, EEG_CAP, FACE
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.datatypes.datatypes_factory import DatatypesFactory


class TestSensorViewers(TransactionalTestCase):
    """
    Unit-tests for Sensors viewers.
    """

    EXPECTED_KEYS_INTERNAL = {'urlMeasurePoints': None, 'urlMeasurePointsLabels': None, 'noOfMeasurePoints': 103,
                              'minMeasure': 0, 'maxMeasure': 103, 'urlMeasure': None, 'shelfObject': None}

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
        self.factory = DatatypesFactory()
        self.test_project = self.factory.get_project()
        self.test_user = self.factory.get_user()

        ## Import Shelf Face Object
        face_path = os.path.join(os.path.dirname(tvb_data.obj.__file__), 'face_surface.obj')
        TestFactory.import_surface_obj(self.test_user, self.test_project, face_path, FACE)

    def transactional_teardown_method(self):
        """
        Clean-up tests data
        """
        FilesHelper().remove_project_structure(self.test_project.name)

    def test_launch_eeg(self):
        """
        Check that all required keys are present in output from EegSensorViewer launch.
        """
        ## Import Sensors
        zip_path = os.path.join(os.path.dirname(tvb_data.sensors.__file__), 'eeg_unitvector_62.txt.bz2')
        TestFactory.import_sensors(self.test_user, self.test_project, zip_path, Sensors_Importer.EEG_SENSORS)
        sensors = TestFactory.get_entity(self.test_project, SensorsEEG())

        ## Import EEGCap
        cap_path = os.path.join(os.path.dirname(tvb_data.obj.__file__), 'eeg_cap.obj')
        TestFactory.import_surface_obj(self.test_user, self.test_project, cap_path, EEG_CAP)
        eeg_cap_surface = TestFactory.get_entity(self.test_project, EEGCap())

        viewer = SensorsViewer()
        viewer.current_project_id = self.test_project.id

        ## Launch with EEG Cap selected
        result = viewer.launch(sensors, eeg_cap_surface)
        self.assert_compliant_dictionary(self.EXPECTED_KEYS_EEG, result)
        for key in ['urlVertices', 'urlTriangles', 'urlLines', 'urlNormals']:
            assert result[key] is not None, "Value at key %s should not be None" % key

        ## Launch without EEG Cap
        result = viewer.launch(sensors)
        self.assert_compliant_dictionary(self.EXPECTED_KEYS_EEG, result)
        for key in ['urlVertices', 'urlTriangles', 'urlLines', 'urlNormals']:
            assert not result[key] or result[key] == "[]", "Value at key %s should be None or empty, " \
                                                           "but is %s" % (key, result[key])

    def test_launch_meg(self):
        """
        Check that all required keys are present in output from MEGSensorViewer launch.
        """

        zip_path = os.path.join(os.path.dirname(tvb_data.sensors.__file__), 'meg_151.txt.bz2')
        TestFactory.import_sensors(self.test_user, self.test_project, zip_path, Sensors_Importer.MEG_SENSORS)
        sensors = TestFactory.get_entity(self.test_project, SensorsMEG())

        viewer = SensorsViewer()
        viewer.current_project_id = self.test_project.id

        result = viewer.launch(sensors)
        self.assert_compliant_dictionary(self.EXPECTED_KEYS_MEG, result)

    def test_launch_internal(self):
        """
        Check that all required keys are present in output from InternalSensorViewer launch.
        """
        zip_path = os.path.join(os.path.dirname(tvb_data.sensors.__file__), 'seeg_39.txt.bz2')
        TestFactory.import_sensors(self.test_user, self.test_project, zip_path, Sensors_Importer.INTERNAL_SENSORS)
        sensors = TestFactory.get_entity(self.test_project, SensorsInternal())

        viewer = SensorsViewer()
        viewer.current_project_id = self.test_project.id

        result = viewer.launch(sensors)
        self.assert_compliant_dictionary(self.EXPECTED_KEYS_INTERNAL, result)
