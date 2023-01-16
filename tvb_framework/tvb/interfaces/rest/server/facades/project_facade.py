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
from tvb.core.entities.storage import CaseDAO
from tvb.core.services.exceptions import ProjectServiceException
from tvb.core.services.project_service import ProjectService
from tvb.core.services.user_service import UserService
from tvb.interfaces.rest.commons.dtos import ProjectDto, DataTypeDto, OperationDto
from tvb.interfaces.rest.commons.exceptions import InvalidIdentifierException, AuthorizationRequestException, \
    InvalidInputException


class ProjectFacade:
    def __init__(self):
        self.project_service = ProjectService()
        self.user_service = UserService()
        self.project_dao = CaseDAO()

    @staticmethod
    def retrieve_logged_user_projects(logged_user_id):
        projects = ProjectService.retrieve_all_user_projects(user_id=logged_user_id)
        return [ProjectDto(project) for project in projects]

    def create_project(self, logged_user, project_name, project_description):
        self.project_service.store_project(logged_user, True, None, name=project_name,
                                           description=project_description)

    def get_datatypes_in_project(self, project_gid):
        try:
            project = self.project_service.find_project_lazy_by_gid(project_gid)
        except ProjectServiceException:
            raise InvalidIdentifierException()

        datatypes = self.project_service.get_datatypes_in_project(project.id)
        return [DataTypeDto(datatype) for datatype in datatypes]

    def get_project_operations(self, project_gid, page_number):
        try:
            project = self.project_service.find_project_lazy_by_gid(project_gid)
        except ProjectServiceException:
            raise InvalidIdentifierException()

        _, _, operations, pages = self.project_service.retrieve_project_full(project.id, current_page=int(page_number))
        return [OperationDto(operation) for operation in operations], pages

    def add_members_to_project(self, current_user_id, project_gid, new_members_gid):
        try:
            project = self.project_service.find_project_lazy_by_gid(project_gid)
        except Exception:
            raise InvalidIdentifierException("Invalid project identifier.")

        if current_user_id != project.fk_admin:
            raise AuthorizationRequestException("Your are not allowed to edit given project")

        new_members_id = []
        for gid in new_members_gid:
            user = self.user_service.get_user_by_gid(gid)
            if user is None:
                raise InvalidInputException("Invalid user gid {}".format(gid))
            new_members_id.append(user.id)
        self.project_dao.add_members_to_project(project.id, new_members_id)
