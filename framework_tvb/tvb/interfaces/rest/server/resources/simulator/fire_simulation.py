import os
import tempfile
from flask import request, jsonify
from flask_restful import Resource
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.services.exceptions import BurstServiceException
from tvb.core.services.project_service import ProjectService
from tvb.core.services.simulator_service import SimulatorService
from werkzeug.utils import secure_filename
from tvb.basic.profile import TvbProfile

UPLOAD_FOLDER = TvbProfile.current.TVB_TEMP_FOLDER


class FireSimulationResource(Resource):

    def __init__(self):
        self.simulator_service = SimulatorService()
        self.project_service = ProjectService()

    def post(self, project_id):
        # check if the post request has the file part
        if 'file' not in request.files:
            resp = jsonify({'message': 'No file part in the request!'})
            resp.status_code = 400
            return resp
        file = request.files['file']
        if file.filename == '':
            resp = jsonify({'message': 'No file selected for uploading!'})
            resp.status_code = 400
            return resp
        if not (file and self.allowed_file(file.filename)):
            resp = jsonify({'message': 'Only ZIP files are allowed!'})
            resp.status_code = 400
            return resp

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
            resp = jsonify({'message': 'Some unexpected error happened!'})
            resp.status_code = 500
            return resp

        resp = jsonify({'message': 'File succesfully uploaded!'})
        resp.status_code = 201
        return resp

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'zip'
