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
.. moduleauthor:: Gabriel Florea <gabriel.florea@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

"""
from os import path

import tvb_data
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.tests.framework.core.factory import TestFactory


class TestConnectivityZipImporter(BaseTestCase):
    """
    Unit-tests for CFF-importer.
    """

    def setup_method(self):
        """
        Reset the database before each test.
        """
        self.test_user = TestFactory.create_user('CFF_User')
        self.test_project = TestFactory.create_project(self.test_user, "CFF_Project")

    def teardown_method(self):
        """
        Clean-up tests data
        """
        self.clean_database()
        FilesHelper().remove_project_structure(self.test_project.name)

    def test_happy_flow_import(self):
        """
        Test that importing a CFF generates at least one DataType in DB.
        """
        zip_path = path.join(path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_96.zip')
        dt_count_before = TestFactory.get_entity_count(self.test_project, ConnectivityIndex)
        TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path, "John", False)
        dt_count_after = TestFactory.get_entity_count(self.test_project, ConnectivityIndex)
        assert dt_count_before + 1 == dt_count_after
