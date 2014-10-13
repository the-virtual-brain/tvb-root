# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
import unittest
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.datatypes.datatypes_factory import DatatypesFactory
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.storage import dao
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.services.flow_service import FlowService
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.datatypes.surfaces import SkullSkin
from tvb.datatypes.surfaces_data import OUTER_SKULL
import tvb_data.surfaceData



class ZIPSurfaceImporterTest(TransactionalTestCase):
    """
    Unit-tests for Zip Surface importer.
    """

    surf_skull = os.path.join(os.path.dirname(tvb_data.surfaceData.__file__), 'outer_skull_4096.zip')


    def setUp(self):
        self.datatypeFactory = DatatypesFactory()
        self.test_project = self.datatypeFactory.get_project()
        self.test_user = self.datatypeFactory.get_user()


    def tearDown(self):
        FilesHelper().remove_project_structure(self.test_project.name)


    def _importSurface(self, import_file_path=None):
        ### Retrieve Adapter instance
        group = dao.find_group('tvb.adapters.uploaders.zip_surface_importer', 'ZIPSurfaceImporter')
        importer = ABCAdapter.build_adapter(group)
        args = {
            'uploaded': import_file_path, 'surface_type': OUTER_SKULL,
            'zero_based_triangles': True,
            DataTypeMetaData.KEY_SUBJECT: "John"
        }

        ### Launch import Operation
        FlowService().fire_operation(importer, self.test_user, self.test_project.id, **args)

        data_types = FlowService().get_available_datatypes(self.test_project.id, SkullSkin)[0]
        self.assertEqual(1, len(data_types), "Project should contain only one data type.")

        surface = ABCAdapter.load_entity_by_gid(data_types[0][2])
        self.assertTrue(surface is not None, "Surface should not be None")
        return surface


    def test_import_surf_zip(self):
        surface = self._importSurface(self.surf_skull)
        self.assertEqual(4096, len(surface.vertices))
        self.assertEqual(4096, surface.number_of_vertices)
        self.assertEqual(8188, len(surface.triangles))
        self.assertEqual(8188, surface.number_of_triangles)
        self.assertEqual('', surface.user_tag_3)
        self.assertTrue(surface.valid_for_simulations)


def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(ZIPSurfaceImporterTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)

