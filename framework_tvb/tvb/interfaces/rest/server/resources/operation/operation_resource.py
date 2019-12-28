import os
import tempfile
from flask import request
from tvb.basic.profile import TvbProfile
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.neotraits.h5 import ViewModelH5
from tvb.core.services.flow_service import FlowService
from tvb.core.services.project_service import ProjectService
from tvb.core.services.user_service import UserService
from tvb.interfaces.rest.server.dto.dtos import DataTypeDto
from tvb.interfaces.rest.server.resources.exceptions import BadRequestException
from tvb.interfaces.rest.server.resources.rest_resource import RestResource
from werkzeug.utils import secure_filename


class GetOperationStatusResource(RestResource):
    """
    :return status of an operation
    """

    def get(self, operation_gid):
        operation = ProjectService.load_operation_by_gid(operation_gid)
        return {"status": operation.status}


class GetOperationResultsResource(RestResource):
    """
    :return list of DataType instances (subclasses), representing the results of that operation if it has finished and
    None, if the operation is still running, has failed or simply has no results.
    """

    def get(self, operation_gid):
        operation = ProjectService.load_operation_lazy_by_gid(operation_gid)
        data_types = ProjectService.get_results_for_operation(operation.id)

        if data_types is None:
            return []
        return [DataTypeDto(datatype) for datatype in data_types]


class LaunchOperationResource(RestResource):
    """ A generic method of launching Analyzers  """

    def __init__(self):
        self.flow_service = FlowService()
        self.project_service = ProjectService()
        self.user_service = UserService()

    def post(self, project_gid, algorithm_module, algorithm_classname):
        # Check if there is any h5 file in the form data
        if 'file' not in request.files:
            raise BadRequestException('No file part in the request!')
        file = request.files['file']
        if not file.filename.endswith(FilesHelper.TVB_STORAGE_FILE_EXTENSION):
            raise BadRequestException('Only h5 files are allowed!')

        # Save current view_model h5 file in a temporary directory
        filename = secure_filename(file.filename)
        temp_name = tempfile.mkdtemp(dir=TvbProfile.current.TVB_TEMP_FOLDER)
        destination_folder = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, temp_name)
        h5_path = os.path.join(destination_folder, filename)
        file.save(h5_path)

        # Prepare and fire operation
        algorithm = self.flow_service.get_algorithm_by_module_and_class(algorithm_module, algorithm_classname)
        project = self.project_service.find_project_lazy_by_gid(project_gid)
        adapter_instance = ABCAdapter.build_adapter(algorithm)
        view_model = adapter_instance.get_view_model_class()()
        view_model_h5 = ViewModelH5(h5_path, view_model)
        view_model_h5.load_into(view_model)
        # TODO: use logged user
        self.flow_service.fire_operation(adapter_instance, self.user_service.get_user_by_id(1), project.id,
                                         view_model=view_model)
