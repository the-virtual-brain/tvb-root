import os
import tempfile

from flask import request
from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.services.exceptions import BurstServiceException
from tvb.core.services.project_service import ProjectService
from tvb.core.services.simulator_service import SimulatorService
from tvb.interfaces.rest.server.resources.exceptions import BaseRestException
from tvb.interfaces.rest.server.resources.rest_resource import RestResource
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = TvbProfile.current.TVB_TEMP_FOLDER


def _allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'zip'


class FireSimulationResource(RestResource):

    def __init__(self):
        self.simulator_service = SimulatorService()
        self.project_service = ProjectService()

    def post(self, project_id):
        # check if the post request has the file part
        if 'file' not in request.files:
            raise BaseRestException('No file part in the request!', 400)
        file = request.files['file']
        if file.filename == '':
            raise BaseRestException('No file selected for uploading!', 400)
        if not (file and _allowed_file(file.filename)):
            raise BaseRestException('Only ZIP files are allowed!', 400)

        filename = secure_filename(file.filename)
        temp_name = tempfile.mkdtemp(dir=UPLOAD_FOLDER)
        destination_folder = os.path.join(UPLOAD_FOLDER, temp_name)
        zip_path = os.path.join(destination_folder, filename)
        file.save(zip_path)
        FilesHelper().unpack_zip(zip_path, destination_folder)
        project = self.project_service.find_project(project_id)
        user_id = project.fk_admin

        try:
            self.simulator_service.prepare_simulation_on_server(burst_config=None, user_id=user_id, project=project,
                                                                zip_folder_path=zip_path[:-4])
        except BurstServiceException as e:
            self.logger.exception('Could not launch burst!')
            raise BaseRestException(e.message, 500)

        return {'message': 'File succesfully uploaded!'}, 201
