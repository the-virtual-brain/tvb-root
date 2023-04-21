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
import tvb_data.surfaceData
import tvb_data.regionMapping as demo_data
from uuid import UUID

from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.visualizers.surface_view import SurfaceViewer, RegionMappingViewer
from tvb.datatypes.surfaces import SurfaceTypesEnum
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory


class TestSurfaceViewers(TransactionalTestCase):
    """
    Unit-tests for Surface & RegionMapping viewers.
    """

    EXPECTED_KEYS = {'urlVertices': None, 'urlTriangles': None, 'urlLines': None, 'urlNormals': None,
                     'urlRegionMap': None, 'biHemispheric': False, 'hemisphereChunkMask': None,
                     'noOfMeasurePoints': 76, 'urlMeasurePoints': None, 'boundaryURL': None, 'minMeasure': 0,
                     'maxMeasure': 76, 'clientMeasureUrl': None}

    def transactional_setup_method(self):
        """
        Sets up the environment for running the tests;
        creates a test user, a test project, a connectivity and a surface;
        imports a CFF data-set
        """
        test_user = TestFactory.create_user('Surface_Viewer_User')
        self.test_project = TestFactory.create_project(test_user, 'Surface_Viewer_Project')

        surf_skull = os.path.join(os.path.dirname(tvb_data.surfaceData.__file__), 'cortex_16384.zip')
        self.surface = TestFactory.import_surface_zip(test_user, self.test_project, surf_skull,
                                                      SurfaceTypesEnum.CORTICAL_SURFACE)
        assert self.surface is not None

        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_76.zip')
        TestFactory.import_zip_connectivity(test_user, self.test_project, zip_path, "John")
        connectivity_index = TestFactory.get_entity(self.test_project, ConnectivityIndex)
        assert connectivity_index is not None

        TXT_FILE = os.path.join(os.path.dirname(demo_data.__file__), 'regionMapping_16k_76.txt')
        self.region_mapping = TestFactory.import_region_mapping(test_user, self.test_project, TXT_FILE,
                                                                self.surface.gid, connectivity_index.gid)
        assert self.region_mapping is not None

    def test_launch_surface(self):
        """
        Check that all required keys are present in output from SurfaceViewer launch.
        """
        viewer = SurfaceViewer()
        viewer.current_project_id = self.test_project.id
        view_model = viewer.get_view_model_class()()
        view_model.surface = UUID(self.surface.gid)
        view_model.region_map = UUID(self.region_mapping.gid)
        result = viewer.launch(view_model)

        self.assert_compliant_dictionary(self.EXPECTED_KEYS, result)

    def test_launch_region(self):
        """
        Check that all required keys are present in output from RegionMappingViewer launch.
        """
        viewer = RegionMappingViewer()
        viewer.current_project_id = self.test_project.id
        view_model = viewer.get_view_model_class()()
        view_model.region_map = UUID(self.region_mapping.gid)
        result = viewer.launch(view_model)

        self.assert_compliant_dictionary(self.EXPECTED_KEYS, result)
