from flask_restful import Resource

from tvb.interfaces.rest.server.decorators.rest_decorators import rest_jsonify


class RestResource(Resource):
    method_decorators = [rest_jsonify]