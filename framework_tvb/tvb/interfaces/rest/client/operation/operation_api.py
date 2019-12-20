import requests
from tvb.interfaces.rest.client.main_api import MainApi


class OperationApi(MainApi):

    def get_operation_status(self, operation_gid):
        response = requests.get(self.server_url + "/operations/" + operation_gid + "/status")
        return response.content

    def get_operations_results(self, operation_gid):
        response = requests.get(self.server_url + "/operations/" + operation_gid + "/results")
        return response.content

    #TODO: ADD CLIENT SIDE OF LAUNCH_OPERATION