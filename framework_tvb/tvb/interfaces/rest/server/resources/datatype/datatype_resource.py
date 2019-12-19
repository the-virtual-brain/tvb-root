from flask_restful import Resource
from flask import send_file
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.storage import dao
from tvb.core.neocom.h5 import h5_file_for_index
from tvb.core.services.flow_service import FlowService
from tvb.interfaces.rest.server.dto.dtos import AlgorithmDto
from tvb.interfaces.rest.server.resources.rest_resource import RestResource


class RetrieveDatatypeResource(Resource):
    """
    Given a guid, this function will download the H5 full data
    """


def get(self, guid):
    index = ABCAdapter.load_entity_by_gid(guid)
    h5_file = h5_file_for_index(index)
    return send_file(h5_file.path)


class GetOperationsForDatatypeResource(RestResource):
    """
    :return the available operations for that datatype, as a list of Algorithm instances
    """

    def __init__(self):
        self.flow_service = FlowService()

    def get(self, datatype_gid):
        categories = dao.get_launchable_categories()
        filtered_adapters = self.flow_service.get_filtered_adapters(datatype_gid, categories)
        return [AlgorithmDto(algorithm) for algorithm in filtered_adapters]