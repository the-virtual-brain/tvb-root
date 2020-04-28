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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import os
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.datatypes.surfaces import CORTICAL
import tvb_data.surfaceData


class TestZIPSurfaceImporter(TransactionalTestCase):
    """
    Unit-tests for Zip Surface importer.
    """

    surf_skull = os.path.join(os.path.dirname(tvb_data.surfaceData.__file__), 'outer_skull_4096.zip')

    def transactional_setup_method(self):
        self.test_user = TestFactory.create_user('Zip_Surface_User')
        self.test_project = TestFactory.create_project(self.test_user, 'Zip_Surface_Project')

    def transactional_teardown_method(self):
        FilesHelper().remove_project_structure(self.test_project.name)

    def test_import_surf_zip(self):
        surface = TestFactory.import_surface_zip(self.test_user, self.test_project, self.surf_skull, CORTICAL)
        assert 4096 == surface.number_of_vertices
        assert 8188 == surface.number_of_triangles
        assert surface.valid_for_simulations

