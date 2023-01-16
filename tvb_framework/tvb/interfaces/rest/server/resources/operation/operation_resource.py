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

from tvb.basic.logger.builder import get_logger
from tvb.interfaces.rest.commons.status_codes import HTTP_STATUS_CREATED
from tvb.interfaces.rest.commons.strings import RequestFileKey
from tvb.interfaces.rest.server.access_permissions.permissions import OperationAccessPermission, \
    ProjectAccessPermission
from tvb.interfaces.rest.server.decorators.rest_decorators import check_permission
from tvb.interfaces.rest.server.facades.operation_facade import OperationFacade
from tvb.interfaces.rest.server.request_helper import get_current_user
from tvb.interfaces.rest.server.resources.rest_resource import RestResource

INVALID_OPERATION_GID_MESSAGE = "No operation found for GID: %s"


class GetOperationStatusResource(RestResource):
    @check_permission(OperationAccessPermission, 'operation_gid')
    def get(self, operation_gid):
        """
        :return status of an operation
        """
        return OperationFacade.get_operation_status(operation_gid)


class GetOperationResultsResource(RestResource):

    @check_permission(OperationAccessPermission, 'operation_gid')
    def get(self, operation_gid):
        """
        :return list of DataType instances (subclasses), representing the results of that operation if it has finished and
        None, if the operation is still running, has failed or simply has no results.
        """
        return OperationFacade.get_operations_results(operation_gid)


class LaunchOperationResource(RestResource):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__module__)
        self.operation_facade = OperationFacade()

    @check_permission(ProjectAccessPermission, 'project_gid')
    def post(self, project_gid, algorithm_module, algorithm_classname):
        """
        :generic method of launching Analyzers
        """
        model_file = self.extract_file_from_request(request_file_key=RequestFileKey.LAUNCH_ANALYZERS_MODEL_FILE.value)
        current_user = get_current_user()
        operation_gid = self.operation_facade.launch_operation(current_user.id, model_file, project_gid,
                                                               algorithm_module,
                                                               algorithm_classname, self.extract_file_from_request)

        return operation_gid, HTTP_STATUS_CREATED
