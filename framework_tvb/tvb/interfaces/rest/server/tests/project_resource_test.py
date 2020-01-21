# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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

import os
import pytest
import tvb_data
from tvb.interfaces.rest.commons.exceptions import InvalidIdentifierException
from tvb.interfaces.rest.server.resources.project.project_resource import GetDataInProjectResource, \
    GetOperationsInProjectResource
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory


class TestProjectResource(TransactionalTestCase):

    def transactional_setup_method(self):
        self.data_resource = GetDataInProjectResource()
        self.operations_resource = GetOperationsInProjectResource()
        self.test_user = TestFactory.create_user('Rest_User')
        self.test_project_without_data = TestFactory.create_project(self.test_user, 'Rest_Project')
        self.test_project_with_data = TestFactory.create_project(self.test_user, 'Rest_Project2')
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_96.zip')
        TestFactory.import_zip_connectivity(self.test_user, self.test_project_with_data, zip_path)

    def test_server_get_data_in_project_inexistent_gid(self):
        project_gid = "inexistent-gid"
        with pytest.raises(InvalidIdentifierException): self.data_resource.get(project_gid)

    def test_server_get_data_in_project_empty(self):
        project_gid = self.test_project_without_data.gid
        result = self.data_resource.get(project_gid)
        assert type(result) is list
        assert len(result) == 0

    def test_get_data_in_project(self):
        project_gid = self.test_project_with_data.gid

        result = self.data_resource.get(project_gid)
        assert type(result) is list
        assert len(result) > 0

    def test_server_get_operations_in_project_inexistent_gid(self):
        project_gid = "inexistent-gid"
        with pytest.raises(InvalidIdentifierException): self.operations_resource.get(project_gid)

    def test_server_get_operations_in_project_empty(self):
        project_gid = self.test_project_without_data.gid
        result = self.operations_resource.get(project_gid)
        assert type(result) is list
        assert len(result) == 0

    def test_get_operations_in_project(self):
        project_gid = self.test_project_with_data.gid
        result = self.operations_resource.get(project_gid)
        assert type(result) is list
        assert len(result) > 0
