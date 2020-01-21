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

from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.services.exceptions import ProjectServiceException
from tvb.core.services.project_service import ProjectService
from tvb.core.services.simulator_service import SimulatorService
from tvb.interfaces.rest.commons.exceptions import InvalidIdentifierException
from tvb.interfaces.rest.server.resources.project.project_resource import INVALID_PROJECT_GID_MESSAGE
from tvb.interfaces.rest.server.resources.rest_resource import RestResource
from tvb.interfaces.rest.server.resources.util import save_temporary_file


class FireSimulationResource(RestResource):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulator_service = SimulatorService()
        self.project_service = ProjectService()

    def post(self, project_gid):
        """
        :start a simulation using a project id and a zip archive with the simulator data serialized
        """
        file = self.extract_file_from_request(FilesHelper.TVB_ZIP_FILE_EXTENSION)
        zip_path = save_temporary_file(file)

        try:
            project = self.project_service.find_project_lazy_by_gid(project_gid)
        except ProjectServiceException:
            raise InvalidIdentifierException(INVALID_PROJECT_GID_MESSAGE % project_gid)

        FilesHelper().unpack_zip(zip_path, os.path.dirname(zip_path))
        user_id = project.fk_admin

        operation = self.simulator_service.prepare_simulation_on_server(burst_config=None, user_id=user_id,
                                                                        project=project,
                                                                        zip_folder_path=zip_path[:-4])

        return operation.gid, 201
