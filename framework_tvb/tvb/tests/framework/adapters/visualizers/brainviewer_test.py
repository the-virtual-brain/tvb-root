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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import os
import tvb_data.surfaceData
import tvb_data.regionMapping
from tvb.core.neocom import h5
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.datatypes.surfaces import CORTICAL
from tvb.adapters.visualizers.brain import BrainViewer, DualBrainViewer, ConnectivityIndex
from tvb.tests.framework.core.factory import TestFactory


class TestBrainViewer(TransactionalTestCase):
    """
    Unit-tests for BrainViewer.
    """

    EXPECTED_KEYS = ['urlVertices', 'urlNormals', 'urlTriangles', 'urlLines', 'urlRegionMap',
                     'base_adapter_url', 'isOneToOneMapping', 'minActivity', 'maxActivity',
                     'noOfMeasurePoints', 'isAdapter']
    EXPECTED_EXTRA_KEYS = ['urlMeasurePointsLabels', 'urlMeasurePoints', 'pageSize', 'shelfObject',
                           'extended_view', 'legendLabels', 'labelsStateVar', 'labelsModes', 'title']

    face = os.path.join(os.path.dirname(tvb_data.surfaceData.__file__), 'cortex_16384.zip')
    region_mapping_path = os.path.join(os.path.dirname(tvb_data.regionMapping.__file__), 'regionMapping_16k_76.txt')

    def transactional_setup_method(self):
        """
        Sets up the environment for running the tests;
        creates a test user, a test project, a connectivity, a cortical surface and a face surface;
        imports a CFF data-set
        """
        self.test_user = TestFactory.create_user('Brain_Viewer_User')
        self.test_project = TestFactory.create_project(self.test_user, 'Brain_Viewer_Project')

        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_96.zip')
        TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path, "John")
        connectivity_idx = TestFactory.get_entity(self.test_project, ConnectivityIndex)
        assert connectivity_idx is not None

        self.face_surface = TestFactory.import_surface_zip(self.test_user, self.test_project, self.face, CORTICAL)

        region_mapping = TestFactory.import_region_mapping(self.test_user, self.test_project,
                                                           self.region_mapping_path, self.face_surface.gid,
                                                           connectivity_idx.gid)
        self.connectivity = h5.load_from_index(connectivity_idx)
        self.region_mapping = h5.load_from_index(region_mapping)

    def transactional_teardown_method(self):
        """
        Clean-up tests data
        """
        FilesHelper().remove_project_structure(self.test_project.name)

    def test_launch(self, time_series_region_index_factory):
        """
        Check that all required keys are present in output from BrainViewer launch.
        """
        time_series_index = time_series_region_index_factory(self.connectivity, self.region_mapping,
                                                             self.test_user, self.test_project)
        viewer = BrainViewer()
        viewer.current_project_id = self.test_project.id
        view_model = viewer.get_view_model_class()()
        view_model.time_series = time_series_index.gid
        view_model.shell_surface = self.face_surface.gid
        result = viewer.launch(view_model)

        for key in TestBrainViewer.EXPECTED_KEYS + TestBrainViewer.EXPECTED_EXTRA_KEYS:
            assert key in result and result[key] is not None
        assert not result['extended_view']

    def test_get_required_memory(self, time_series_region_index_factory):
        """
        Brainviewer should know required memory so expect positive number and not -1.
        """
        time_series_index = time_series_region_index_factory(self.connectivity, self.region_mapping,
                                                             self.test_user, self.test_project)
        viewer = BrainViewer()
        viewer.current_project_id = self.test_project.id
        view_model = viewer.get_view_model_class()()
        view_model.time_series = time_series_index.gid
        assert viewer.get_required_memory_size(view_model) > 0

    def test_generate_preview(self, time_series_region_index_factory):
        """
        Check that all required keys are present in preview generate by BrainViewer.
        """
        time_series_index = time_series_region_index_factory(self.connectivity, self.region_mapping,
                                                             self.test_user, self.test_project)
        viewer = BrainViewer()
        viewer.current_project_id = self.test_project.id
        view_model = viewer.get_view_model_class()()
        view_model.time_series = time_series_index.gid
        result = viewer.generate_preview(view_model, figure_size=(500, 200))
        for key in TestBrainViewer.EXPECTED_KEYS:
            assert key in result and result[key] is not None, key

    def test_launch_eeg(self, time_series_region_index_factory):
        """
        Tests successful launch of a BrainEEG and that all required keys are present in returned template dictionary
        """
        time_series_index = time_series_region_index_factory(self.connectivity, self.region_mapping,
                                                             self.test_user, self.test_project)
        viewer = DualBrainViewer()
        viewer.current_project_id = self.test_project.id
        view_model = viewer.get_view_model_class()()
        view_model.time_series = time_series_index.gid
        view_model.shell_surface = self.face_surface.gid
        result = viewer.launch(view_model)
        for key in TestBrainViewer.EXPECTED_KEYS + TestBrainViewer.EXPECTED_EXTRA_KEYS:
            assert key in result and result[key] is not None
        assert result['extended_view']
