from tvb.core.entities.storage import dao

from tvb.interfaces.rest.server.dto.dtos import DataTypeDto
from tvb.interfaces.rest.server.resources.rest_resource import RestResource


class GetOperationStatusResource(RestResource):
    def get(self, operation_id):
        operation = dao.get_operation_by_id(operation_id)
        return {"status": operation.status}


class GetOperationResultsResource(RestResource):
    def get(self, operation_id):
        data_types = dao.get_results_for_operation(operation_id)

        if data_types is None:
            return []
        return [DataTypeDto(datatype) for datatype in data_types]
