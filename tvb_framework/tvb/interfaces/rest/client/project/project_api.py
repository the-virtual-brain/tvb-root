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

from tvb.interfaces.rest.client.client_decorators import handle_response
from tvb.interfaces.rest.client.main_api import MainApi
from tvb.interfaces.rest.commons.dtos import DataTypeDto
from tvb.interfaces.rest.commons.exceptions import ClientException
from tvb.interfaces.rest.commons.strings import Strings, RestLink, LinkPlaceholder, FormKeyInput


class ProjectApi(MainApi):
    @handle_response
    def get_data_in_project(self, project_gid):
        response = self.secured_request().get(self.build_request_url(RestLink.DATA_IN_PROJECT.compute_url(True, {
            LinkPlaceholder.PROJECT_GID.value: project_gid
        })))
        return response, DataTypeDto

    @handle_response
    def get_operations_in_project(self, project_gid, page_number=1):
        try:
            page_number = int(page_number)
        except ValueError:
            raise ClientException(message="Invalid page number")
        response = self.secured_request().get(self.build_request_url(RestLink.OPERATIONS_IN_PROJECT.compute_url(True, {
            LinkPlaceholder.PROJECT_GID.value: project_gid
        })), params={Strings.PAGE_NUMBER.value: page_number})
        return response

    @handle_response
    def add_members_to_project(self, project_gid, new_members_gid):
        if type(new_members_gid) is not list:
            raise ClientException("New members gid parameter must be a list.")
        return self.secured_request().put(self.build_request_url(RestLink.PROJECT_MEMBERS.compute_url(True, {
            LinkPlaceholder.PROJECT_GID.value: project_gid
        })), json={FormKeyInput.NEW_MEMBERS_GID.value: new_members_gid})
