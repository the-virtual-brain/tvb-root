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

import flask
from tvb.core.entities.storage import CaseDAO
from tvb.core.services.exceptions import ProjectServiceException
from tvb.core.services.project_service import ProjectService
from tvb.core.services.user_service import UserService
from tvb.interfaces.rest.commons.dtos import OperationDto, DataTypeDto
from tvb.interfaces.rest.commons.exceptions import InvalidIdentifierException, InvalidInputException, \
    AuthorizationRequestException
from tvb.interfaces.rest.commons.strings import Strings, FormKeyInput
from tvb.interfaces.rest.server.access_permissions.permissions import ProjectAccessPermission
from tvb.interfaces.rest.server.decorators.rest_decorators import check_permission
from tvb.interfaces.rest.server.request_helper import get_current_user
from tvb.interfaces.rest.server.resources.rest_resource import RestResource

INVALID_PROJECT_GID_MESSAGE = 'No project found for GID: %s'


class GetDataInProjectResource(RestResource):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_service = ProjectService()

    @check_permission(ProjectAccessPermission, 'project_gid')
    def get(self, project_gid):
        """
        :return a list of DataType instances (subclasses) associated with the current project
        """
        try:
            project = self.project_service.find_project_lazy_by_gid(project_gid)
        except ProjectServiceException:
            raise InvalidIdentifierException(INVALID_PROJECT_GID_MESSAGE % project_gid)

        datatypes = self.project_service.get_datatypes_in_project(project.id)
        return [DataTypeDto(datatype) for datatype in datatypes]


class GetOperationsInProjectResource(RestResource):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_service = ProjectService()

    @check_permission(ProjectAccessPermission, 'project_gid')
    def get(self, project_gid):
        """
        :return a list of project's Operation entities
        """
        page_number = flask.request.args.get(Strings.PAGE_NUMBER.value)
        if page_number is None:
            page_number = 1
        try:
            page_number = int(page_number)
        except ValueError:
            raise InvalidInputException(message="Invalid page number")

        try:
            project = self.project_service.find_project_lazy_by_gid(project_gid)
        except ProjectServiceException:
            raise InvalidIdentifierException(INVALID_PROJECT_GID_MESSAGE % project_gid)

        _, _, operations, pages = self.project_service.retrieve_project_full(project.id, current_page=int(page_number))
        return {"operations": [OperationDto(operation) for operation in operations], "pages": pages}


class ProjectMembersResource(RestResource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_service = ProjectService()
        self.user_service = UserService()
        self.project_dao = CaseDAO()

    def put(self, project_gid):
        """
        Add members to the given project
        :param project_gid: project gid
        :param
        """
        try:
            project = self.project_service.find_project_lazy_by_gid(project_gid)
        except Exception:
            raise InvalidIdentifierException("Invalid project identifier.")

        if get_current_user().id != project.fk_admin:
            raise AuthorizationRequestException("Your are not allowed to edit given project")

        input_data = flask.request.json
        new_members_gid = input_data[
            FormKeyInput.NEW_MEMBERS_GID.value] if FormKeyInput.NEW_MEMBERS_GID.value in input_data else []
        new_members_id = []
        for gid in new_members_gid:
            user = self.user_service.get_user_by_gid(gid)
            if user is None:
                raise InvalidInputException("Invalid user gid {}".format(gid))
            new_members_id.append(user.id)
        self.project_dao.add_members_to_project(project.id, new_members_id)
