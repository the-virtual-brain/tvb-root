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

import flask
from tvb.interfaces.rest.commons.exceptions import InvalidInputException
from tvb.interfaces.rest.commons.strings import FormKeyInput
from tvb.interfaces.rest.server.access_permissions.permissions import ProjectAccessPermission
from tvb.interfaces.rest.server.decorators.rest_decorators import check_permission
from tvb.interfaces.rest.server.facades.project_facade import ProjectFacade
from tvb.interfaces.rest.server.request_helper import get_current_user
from tvb.interfaces.rest.server.resources.rest_resource import RestResource

INVALID_PROJECT_GID_MESSAGE = 'No project found for GID: %s'


class GetDataInProjectResource(RestResource):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_facade = ProjectFacade()

    @check_permission(ProjectAccessPermission, 'project_gid')
    def get(self, project_gid):
        """
        :return a list of DataType instances (subclasses) associated with the current project
        """
        return self.project_facade.get_datatypes_in_project(project_gid)


class GetOperationsInProjectResource(RestResource):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_facade = ProjectFacade()

    @check_permission(ProjectAccessPermission, 'project_gid')
    def get(self, project_gid):
        """
        :return a list of project's Operation entities
        """
        page_number = self.extract_page_number()
        operation_dto_list, pages = self.project_facade.get_project_operations(project_gid, page_number)
        return {"operations": operation_dto_list, "pages": pages}


class ProjectMembersResource(RestResource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_facade = ProjectFacade()

    def put(self, project_gid):
        """
        Add members to the given project
        :param project_gid: project gid
        :param
        """
        input_data = flask.request.json
        new_members_gid = input_data[
            FormKeyInput.NEW_MEMBERS_GID.value] if FormKeyInput.NEW_MEMBERS_GID.value in input_data else []
        if len(new_members_gid) == 0:
            raise InvalidInputException("Empty users list.")

        self.project_facade.add_members_to_project(get_current_user().id, project_gid, new_members_gid)
