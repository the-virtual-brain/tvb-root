from flask_restful import Resource
from flask import send_file
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.neocom.h5 import h5_file_for_index


class RetrieveDatatypeResource(Resource):

    def get(self, guid):
        index = ABCAdapter.load_entity_by_gid(guid)
        h5_file = h5_file_for_index(index)
        return send_file(h5_file.path)
