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
from abc import abstractmethod

from sqlalchemy.orm.exc import NoResultFound
from tvb.core.entities.storage import CaseDAO, DatatypeDAO
from tvb.core.services.exceptions import ProjectServiceException
from tvb.core.services.project_service import ProjectService
from tvb.interfaces.rest.commons.exceptions import InvalidIdentifierException
from tvb.interfaces.rest.server.request_helper import get_current_user


class ResourceAccessPermission:
    def __init__(self, resource_identifier, required_role=None):
        self.resource_identifier = resource_identifier
        self.required_role = required_role

    def has_access(self):
        current_user = get_current_user()
        if self.required_role is not None and current_user.role != self.required_role:
            return False
        return self._check_permission(current_user.id)

    @abstractmethod
    def _check_permission(self, logged_user_id):
        """
        :return: a list of users id who can access the requested resource
        """
        raise RuntimeError("Not implemented.")


class ProjectAccessPermission(ResourceAccessPermission):
    def __init__(self, project_gid):
        super(ProjectAccessPermission, self).__init__(project_gid)
        self.project_dao = CaseDAO()

    def _check_permission(self, logged_user_id):
        try:
            project = self.project_dao.get_project_lazy_by_gid(self.resource_identifier)
        except (ProjectServiceException, NoResultFound):
            raise InvalidIdentifierException()
        return self.check_project_permission(logged_user_id, project.id)

    def check_project_permission(self, logged_user_id, project_id):
        project_members = self.project_dao.get_members_of_project(project_id)
        return logged_user_id in [project_member.id for project_member in project_members]


class OperationAccessPermission(ProjectAccessPermission):
    def __init__(self, operation_gid):
        super(OperationAccessPermission, self).__init__(operation_gid)

    def _check_permission(self, logged_user_id):
        operation = ProjectService.load_operation_by_gid(self.resource_identifier)
        if operation is None:
            raise InvalidIdentifierException()
        return self.check_project_permission(logged_user_id, operation.fk_launched_in)


class DataTypeAccessPermission(ProjectAccessPermission):
    def __init__(self, datatype_gid):
        super(DataTypeAccessPermission, self).__init__(datatype_gid)
        self.datatype_dao = DatatypeDAO()

    def _check_permission(self, logged_user_id):
        datatype = self.datatype_dao.get_datatype_by_gid(self.resource_identifier)
        if datatype is None:
            raise InvalidIdentifierException()
        if self.check_project_permission(logged_user_id, datatype.parent_operation.fk_launched_in):
            return True
        links = self.datatype_dao.get_links_for_datatype(datatype.id)
        if links is not None:
            for link in links:
                if self.check_project_permission(logged_user_id, link.fk_to_project):
                    return True
        return False
