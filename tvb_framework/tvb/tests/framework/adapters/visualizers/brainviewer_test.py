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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import os
import tvb_data.surfaceData
import tvb_data.regionMapping

from tvb.core.neocom import h5
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.datatypes.surfaces import SurfaceTypesEnum
from tvb.adapters.visualizers.brain import BrainViewer, DualBrainViewer, ConnectivityIndex
from tvb.tests.framework.core.factory import TestFactory


class TestBrainViewer(TransactionalTestCase):
    """
    Unit-tests for BrainViewer.
    """

    EXPECTED_KEYS = ['urlVertices', 'urlNormals', 'urlTriangles', 'urlLines', 'urlRegionMap',
                     'base_adapter_url', 'isOneToOneMapping', 'minActivity', 'maxActivity',
                     'noOfMeasurePoints', 'isAdapter']
    EXPECTED_EXTRA_KEYS = ['urlMeasurePointsLabels', 'urlMeasurePoints', 'pageSize', 'shellObject',
                           'extended_view', 'legendLabels', 'labelsStateVar', 'labelsModes', 'title']

    cortex = os.path.join(os.path.dirname(tvb_data.surfaceData.__file__), 'cortex_16384.zip')
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

        self.face_surface = TestFactory.import_surface_zip(self.test_user, self.test_project, self.cortex,
                                                           SurfaceTypesEnum.CORTICAL_SURFACE)

        region_mapping = TestFactory.import_region_mapping(self.test_user, self.test_project,
                                                           self.region_mapping_path, self.face_surface.gid,
                                                           connectivity_idx.gid)
        self.connectivity = h5.load_from_index(connectivity_idx)
        self.region_mapping = h5.load_from_index(region_mapping)

    def test_launch(self, time_series_region_index_factory):
        """
        Check that all required keys are present in output from BrainViewer launch.
        """
        time_series_index = time_series_region_index_factory(self.connectivity, self.region_mapping,
                                                             test_user=self.test_user, test_project=self.test_project)
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
                                                             test_user=self.test_user, test_project=self.test_project)
        viewer = BrainViewer()
        viewer.current_project_id = self.test_project.id
        view_model = viewer.get_view_model_class()()
        view_model.time_series = time_series_index.gid
        assert viewer.get_required_memory_size(view_model) > 0

    def test_launch_eeg(self, time_series_region_index_factory):
        """
        Tests successful launch of a BrainEEG and that all required keys are present in returned template dictionary
        """
        time_series_index = time_series_region_index_factory(self.connectivity, self.region_mapping,
                                                             test_user=self.test_user, test_project=self.test_project)
        viewer = DualBrainViewer()
        viewer.current_project_id = self.test_project.id
        view_model = viewer.get_view_model_class()()
        view_model.time_series = time_series_index.gid
        view_model.shell_surface = self.face_surface.gid
        result = viewer.launch(view_model)
        for key in TestBrainViewer.EXPECTED_KEYS + TestBrainViewer.EXPECTED_EXTRA_KEYS:
            assert key in result and result[key] is not None
        assert result['extended_view']
