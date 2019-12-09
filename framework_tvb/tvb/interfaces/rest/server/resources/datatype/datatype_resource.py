from flask_restful import Resource
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.neocom import h5
import json
from tvb.interfaces.rest.server.encoder.json_encoder import CustomEncoder


class GetDatatypeResource(Resource):

    def get(self, guid):
        index = ABCAdapter.load_entity_by_gid(guid)
        h5_file = h5.load_from_index(index)
        return json.dumps(h5_file.__dict__, cls=CustomEncoder)

