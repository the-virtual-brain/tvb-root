from flask_restful import Resource
from tvb.core.entities.storage import dao


class GetOperationStatusResource(Resource):

    def get(self, operation_id):
        operation = dao.get_operation_by_id(operation_id)
        return operation.status