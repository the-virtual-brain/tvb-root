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
from abc import abstractmethod

from tvb.core.entities.storage import CaseDAO, DatatypeDAO
from tvb.core.services.exceptions import ProjectServiceException
from tvb.core.services.project_service import ProjectService
from tvb.interfaces.rest.server.request_helper import get_current_user


class ResourceAccessPermission:
    def __init__(self, resource_identifier):
        self.resource_identifier = resource_identifier

    def has_access(self):
        current_user = get_current_user()
        return current_user.id in self.get_resource_owners()

    @abstractmethod
    def get_resource_owners(self):
        """
        :return: a list of users id who can access the requested resource
        """
        raise RuntimeError("Not implemented.")


class OperationAccessPermission(ResourceAccessPermission):
    def __init__(self, operation_gid):
        super(OperationAccessPermission, self).__init__(operation_gid)

    def get_resource_owners(self):
        operation = ProjectService.load_operation_by_gid(self.resource_identifier)
        return [operation.fk_launched_by] if operation is not None else []


class ProjectAccessPermission(ResourceAccessPermission):
    def __init__(self, project_gid):
        super(ProjectAccessPermission, self).__init__(project_gid)
        self.project_dao = CaseDAO()

    def get_resource_owners(self):
        try:
            project = self.project_dao.get_project_lazy_by_gid(self.resource_identifier)
        except ProjectServiceException:
            return []
        project_members = self.project_dao.get_members_of_project(project.id)
        return [project_member.id for project_member in project_members]


class DataTypeAccessPermission(ResourceAccessPermission):
    def __init__(self, datatype_gid):
        super(DataTypeAccessPermission, self).__init__(datatype_gid)
        self.datatype_dao = DatatypeDAO()

    def get_resource_owners(self):
        datatype = self.datatype_dao.get_datatype_by_gid(self.resource_identifier)
        return [datatype.parent_operation.user.id] if datatype is not None else []
