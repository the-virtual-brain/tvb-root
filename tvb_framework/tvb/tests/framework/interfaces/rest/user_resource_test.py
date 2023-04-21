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

from tvb.interfaces.rest.server.resources.user.user_resource import GetProjectsListResource
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.interfaces.rest.base_resource_test import RestResourceTest


class TestUserResource(RestResourceTest):

    def transactional_setup_method(self):
        self.username = 'Rest_User'
        self.test_user = TestFactory.create_user(self.username)
        self.test_project = TestFactory.create_project(self.test_user, 'Rest_Project', users=[self.test_user.id])
        self.projects_list_resource = GetProjectsListResource()

    def test_get_projects(self, mocker):
        self._mock_user(mocker)

        result = self.projects_list_resource.get()
        assert type(result) is list
        assert len(result) == 1
