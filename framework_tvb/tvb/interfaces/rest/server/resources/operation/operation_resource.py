import json

from flask import jsonify
from flask_restful import Resource
from tvb.core.entities.storage import dao
from tvb.interfaces.rest.server.encoders.json_encoders import AlgorithmEncoder


class GetOperationStatusResource(Resource):

    def get(self, operation_id):
        operation = dao.get_operation_by_id(operation_id)
        return operation.status


class GetOperationResultsResource(Resource):

    def get(self, operation_id):
        operation_results = dao.get_results_for_operation(operation_id)

        if operation_results is None:
            return None
        results = dict()

        for i in range(len(operation_results)):
            results[str(i)] = json.dumps(operation_results[i].__dict__, cls=AlgorithmEncoder)

        return results
