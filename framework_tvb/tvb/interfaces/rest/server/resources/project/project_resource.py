from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.storage import dao
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.services.flow_service import FlowService
from tvb.core.services.project_service import ProjectService
from tvb.core.services.user_service import UserService

from tvb.interfaces.rest.server.dto.dtos import ProjectDto, OperationDto, AlgorithmDto
from tvb.interfaces.rest.server.resources.rest_resource import RestResource


class GetProjectsListResource(RestResource):

    def __init__(self):
        self.project_service = ProjectService()

    def get(self, user_id):
        projects, _ = self.project_service.retrieve_projects_for_user(user_id=user_id)
        return [ProjectDto(project) for project in projects]


class GetDataInProjectResource(RestResource):

    def __init__(self):
        self.project_service = ProjectService()

    def get(self, project_id):
        project = self.project_service.find_project(project_id)
        datatypes = self.project_service.get_project_structure(project, None, DataTypeMetaData.KEY_STATE,
                                                               DataTypeMetaData.KEY_SUBJECT, None)
        return datatypes


class GetOperationsInProjectResource(RestResource):

    def __init__(self):
        self.project_service = ProjectService()

    def get(self, project_id):
        _, _, operations, _ = self.project_service.retrieve_project_full(project_id)
        return [OperationDto(operation) for operation in operations]


class GetOperationsForDatatypeResource(RestResource):

    def __init__(self):
        self.flow_service = FlowService()

    def get(self, guid):
        categories = dao.get_launchable_categories()
        filtered_adapters = self.flow_service.get_filtered_adapters(guid, categories)
        return [AlgorithmDto(algorithm) for algorithm in filtered_adapters]


class LaunchOperationResource(RestResource):
    def __init__(self):
        self.flow_service = FlowService()
        self.project_service = ProjectService()
        self.user_service = UserService()

    def post(self, project_id, algorithm_id):
        algorithm = self.flow_service.get_algorithm_by_identifier(algorithm_id)
        adapter_instance = ABCAdapter.build_adapter(algorithm)

        form = adapter_instance.get_form()(project_id=project_id)
        # TODO: REVIEW THIS/FILL MODEL
        view_model = form.get_view_model()()
        self.flow_service.fire_operation(adapter_instance, self.user_service.get_user_by_id(1), project_id,
                                         view_model=view_model)
