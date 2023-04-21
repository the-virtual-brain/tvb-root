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

import os

from tvb.basic.logger.builder import get_logger
from tvb.interfaces.rest.commons.exceptions import InvalidInputException
from tvb.interfaces.rest.commons.files_helper import save_temporary_file
from tvb.interfaces.rest.commons.status_codes import HTTP_STATUS_CREATED
from tvb.interfaces.rest.commons.strings import RequestFileKey
from tvb.interfaces.rest.server.access_permissions.permissions import ProjectAccessPermission
from tvb.interfaces.rest.server.decorators.rest_decorators import check_permission
from tvb.interfaces.rest.server.facades.simulation_facade import SimulationFacade
from tvb.interfaces.rest.server.request_helper import get_current_user
from tvb.interfaces.rest.server.resources.rest_resource import RestResource
from tvb.storage.storage_interface import StorageInterface


class FireSimulationResource(RestResource):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__module__)
        self.simulation_facade = SimulationFacade()

    @check_permission(ProjectAccessPermission, 'project_gid')
    def post(self, project_gid):
        """
        :start a simulation using a project id and a zip archive with the simulator data serialized
        """
        file = self.extract_file_from_request(request_file_key=RequestFileKey.SIMULATION_FILE_KEY.value,
                                              file_extension=StorageInterface.TVB_ZIP_FILE_EXTENSION)
        zip_path = save_temporary_file(file)
        result = StorageInterface().unpack_zip(zip_path, os.path.dirname(zip_path))

        if len(result) == 0:
            self.logger.error("Empty zip archive. {}".format(zip_path))
            raise InvalidInputException("Empty zip archive")

        zip_directory = os.path.dirname(result[0])

        simulation_gid = self.simulation_facade.launch_simulation(get_current_user().id, zip_directory, project_gid)
        return simulation_gid, HTTP_STATUS_CREATED
