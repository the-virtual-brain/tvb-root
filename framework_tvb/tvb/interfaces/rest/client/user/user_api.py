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

from tvb.interfaces.rest.client.client_decorators import handle_response
from tvb.interfaces.rest.client.main_api import MainApi
from tvb.interfaces.rest.commons.dtos import ProjectDto
from tvb.interfaces.rest.commons.strings import RestLink, FormKeyInput


class UserApi(MainApi):
    @handle_response
    def login(self, username, password):
        response = self.secured_request().post(self.build_request_url(RestLink.LOGIN.compute_url(True)), json={
            FormKeyInput.USERS_USERNAME.value: username,
            FormKeyInput.USERS_PASSWORD.value: password
        })
        return response

    @handle_response
    def logout(self):
        response = self.secured_request().delete(self.build_request_url(RestLink.LOGIN.compute_url(True)), json={
            FormKeyInput.KEYCLOAK_REFRESH_TOKEN.value: self.refresh_token,
        })
        return response

    @handle_response
    def get_projects_list(self):
        response = self.secured_request().get(self.build_request_url(RestLink.PROJECTS.compute_url(True)))
        return response, ProjectDto

    @handle_response
    def create_project(self, project_name, project_description):
        return self.secured_request().post(self.build_request_url(RestLink.PROJECTS.compute_url(True)), json={
            FormKeyInput.CREATE_PROJECT_NAME.value: project_name,
            FormKeyInput.CREATE_PROJECT_DESCRIPTION.value: project_description
        })
