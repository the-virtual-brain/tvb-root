import os
import tempfile

from flask import request
from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.services.project_service import ProjectService
from tvb.core.services.simulator_service import SimulatorService
from tvb.interfaces.rest.server.resources.exceptions import BaseRestException
from tvb.interfaces.rest.server.resources.rest_resource import RestResource
from werkzeug.utils import secure_filename


class FireSimulationResource(RestResource):
    """
    Start a simulation using a project id and a zip archive with the simulator data serialized
    """

    def __init__(self):
        self.simulator_service = SimulatorService()
        self.project_service = ProjectService()

    def post(self, project_id):
        # check if the post request has the file part
        if 'file' not in request.files:
            raise BaseRestException('No file part in the request!', 400)
        file = request.files['file']
        if not file.filename.endswith(FilesHelper.TVB_ZIP_FILE_EXTENSION):
            raise BaseRestException('Only ZIP files are allowed!', 400)

        filename = secure_filename(file.filename)
        temp_name = tempfile.mkdtemp(dir=TvbProfile.current.TVB_TEMP_FOLDER)
        destination_folder = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, temp_name)
        zip_path = os.path.join(destination_folder, filename)
        file.save(zip_path)
        FilesHelper().unpack_zip(zip_path, destination_folder)
        project = self.project_service.find_project(project_id)
        user_id = project.fk_admin

        self.simulator_service.prepare_simulation_on_server(burst_config=None, user_id=user_id, project=project,
                                                            zip_folder_path=zip_path[:-4])

        return {'message': 'Simulation started'}, 201
