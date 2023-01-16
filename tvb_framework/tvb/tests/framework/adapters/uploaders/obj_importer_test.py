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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import os
import tvb_data.obj

from tvb.core.neocom import h5
from tvb.datatypes.surfaces import SurfaceTypesEnum
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.tests.framework.core.factory import TestFactory


class TestObjSurfaceImporter(BaseTestCase):
    """
    Unit-tests for Obj Surface importer.
    """

    torus = os.path.join(os.path.dirname(tvb_data.obj.__file__), 'test_torus.obj')
    face = os.path.join(os.path.dirname(tvb_data.obj.__file__), 'face_surface.obj')

    def setup_method(self):
        self.test_user = TestFactory.create_user('Obj_Importer_User')
        self.test_project = TestFactory.create_project(self.test_user, "Obj_Importer_Project")

    def teardown_method(self):
        self.clean_database()

    def test_import_quads_no_normals(self):
        """
        Test that import works with a file which contains quads and no normals
        """
        surface_index = TestFactory.import_surface_obj(self.test_user, self.test_project, self.face,
                                                       SurfaceTypesEnum.FACE_SURFACE, False)

        surface = h5.load_from_index(surface_index)
        assert 8614 == len(surface.vertex_normals)
        assert 8614 == len(surface.vertices)
        assert 17224 == len(surface.triangles)

    def test_import_simple_with_normals(self):
        """
        Test that import works with an OBJ file which includes normals
        """
        surface_index = TestFactory.import_surface_obj(self.test_user, self.test_project, self.torus,
                                                       SurfaceTypesEnum.FACE_SURFACE, False)
        assert 441 == surface_index.number_of_vertices
        assert 800 == surface_index.number_of_triangles

        surface = h5.load_from_index(surface_index)
        assert 441 == len(surface.vertex_normals)
