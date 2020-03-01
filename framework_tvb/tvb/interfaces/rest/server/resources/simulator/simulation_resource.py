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

import os

from tvb.adapters.simulator.simulator_adapter import SimulatorAdapter
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.neocom._h5loader import DirLoader
from tvb.core.services.exceptions import ProjectServiceException
from tvb.core.services.flow_service import FlowService
from tvb.core.services.project_service import ProjectService
from tvb.core.services.simulator_service import SimulatorService
from tvb.interfaces.rest.commons.exceptions import InvalidIdentifierException, InvalidInputException, ServiceException
from tvb.interfaces.rest.commons.status_codes import HTTP_STATUS_CREATED
from tvb.interfaces.rest.commons.strings import RequestFileKey
from tvb.interfaces.rest.server.resources.project.project_resource import INVALID_PROJECT_GID_MESSAGE
from tvb.interfaces.rest.server.resources.rest_resource import RestResource
from tvb.interfaces.rest.server.security.authorization import get_current_user
from tvb.simulator.simulator import Simulator


class FireSimulationResource(RestResource):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__module__)
        self.simulator_service = SimulatorService()
        self.project_service = ProjectService()

    def post(self, project_gid):
        """
        :start a simulation using a project id and a zip archive with the simulator data serialized
        """
        file = self.extract_file_from_request(request_file_key=RequestFileKey.SIMULATION_FILE_KEY.value,
                                              file_extension=FilesHelper.TVB_ZIP_FILE_EXTENSION)
        destination_folder = RestResource.get_destination_folder()
        zip_path = RestResource.save_temporary_file(file, destination_folder)

        try:
            project = self.project_service.find_project_lazy_by_gid(project_gid)
        except ProjectServiceException:
            raise InvalidIdentifierException(INVALID_PROJECT_GID_MESSAGE % project_gid)

        result = FilesHelper().unpack_zip(zip_path, os.path.dirname(zip_path))
        if len(result) == 0:
            raise InvalidInputException("Empty zip archive")

        folder_path = os.path.dirname(result[0])
        simulator_algorithm = FlowService().get_algorithm_by_module_and_class(SimulatorAdapter.__module__,
                                                                              SimulatorAdapter.__name__)
        try:
            simulator_h5_name = DirLoader(folder_path, None).find_file_for_has_traits_type(Simulator)
            simulator_file = os.path.join(folder_path, simulator_h5_name)
        except IOError:
            raise InvalidInputException('No Simulator h5 file found in the archive')

        try:
            current_user = get_current_user()
            operation = self.simulator_service.prepare_simulation_on_server(user_id=current_user.id,
                                                                            project=project,
                                                                            algorithm=simulator_algorithm,
                                                                            zip_folder_path=folder_path,
                                                                            simulator_file=simulator_file)
        except Exception as excep:
            self.logger.error(excep, exc_info=True)
            raise ServiceException(str(excep))

        return operation.gid, HTTP_STATUS_CREATED
