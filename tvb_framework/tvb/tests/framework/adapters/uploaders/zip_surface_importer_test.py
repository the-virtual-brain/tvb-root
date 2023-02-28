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

import tvb_data.surfaceData
from tvb.datatypes.surfaces import SurfaceTypesEnum
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.tests.framework.core.factory import TestFactory


class TestZIPSurfaceImporter(BaseTestCase):
    """
    Unit-tests for Zip Surface importer.
    """

    surf_skull = os.path.join(os.path.dirname(tvb_data.surfaceData.__file__), 'outer_skull_4096.zip')

    def setup_method(self):
        self.test_user = TestFactory.create_user('Zip_Surface_User')
        self.test_project = TestFactory.create_project(self.test_user, 'Zip_Surface_Project')

    def teardown_method(self):
        self.clean_database()

    def test_import_surf_zip(self):
        surface = TestFactory.import_surface_zip(self.test_user, self.test_project, self.surf_skull,
                                                 SurfaceTypesEnum.CORTICAL_SURFACE,  same_process=False)
        assert 4096 == surface.number_of_vertices
        assert 8188 == surface.number_of_triangles
        assert surface.valid_for_simulations
