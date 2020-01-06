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
import tempfile

from flask import request
from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.services.project_service import ProjectService
from tvb.core.services.simulator_service import SimulatorService
from tvb.interfaces.rest.server.resources.exceptions import BadRequestException
from tvb.interfaces.rest.server.resources.rest_resource import RestResource
from werkzeug.utils import secure_filename


class FireSimulationResource(RestResource):
    """
    Start a simulation using a project id and a zip archive with the simulator data serialized
    """

    def __init__(self):
        self.simulator_service = SimulatorService()
        self.project_service = ProjectService()

    def post(self, project_gid):
        # check if the post request has the file part
        if 'file' not in request.files:
            raise BadRequestException('No file part in the request!')
        file = request.files['file']
        if not file.filename.endswith(FilesHelper.TVB_ZIP_FILE_EXTENSION):
            raise BadRequestException('Only ZIP files are allowed!')

        filename = secure_filename(file.filename)
        temp_name = tempfile.mkdtemp(dir=TvbProfile.current.TVB_TEMP_FOLDER)
        destination_folder = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, temp_name)
        zip_path = os.path.join(destination_folder, filename)
        file.save(zip_path)
        FilesHelper().unpack_zip(zip_path, destination_folder)
        project = self.project_service.find_project_lazy_by_gid(project_gid)
        user_id = project.fk_admin

        self.simulator_service.prepare_simulation_on_server(burst_config=None, user_id=user_id, project=project,
                                                            zip_folder_path=zip_path[:-4])

        return {'message': 'Simulation started'}, 201
