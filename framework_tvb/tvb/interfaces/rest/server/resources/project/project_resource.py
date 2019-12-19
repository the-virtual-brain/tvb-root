from tvb.core.services.project_service import ProjectService
from tvb.interfaces.rest.server.dto.dtos import  OperationDto, DataTypeDto
from tvb.interfaces.rest.server.resources.rest_resource import RestResource


class GetDataInProjectResource(RestResource):
    """
    "return a list of DataType instances (subclasses) associated with the current project
    """

    def __init__(self):
        self.project_service = ProjectService()

    def get(self, project_id):
        project = self.project_service.find_project(project_id)
        datatypes = self.project_service.get_datatypes_in_project(project_id)
        return [DataTypeDto(datatype) for datatype in datatypes]


class GetOperationsInProjectResource(RestResource):
    """
    :return a list of project's Operation entities
    """

    def __init__(self):
        self.project_service = ProjectService()

    def get(self, project_id):
        _, _, operations, _ = self.project_service.retrieve_project_full(project_id)
        return [OperationDto(operation) for operation in operations]
