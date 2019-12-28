import requests
from tvb.interfaces.rest.client.main_api import MainApi


class ProjectApi(MainApi):

    def get_data_in_project(self, project_gid):
        response = requests.get(self.server_url + '/projects/' + project_gid + "/data")
        return response.content

    def get_operations_in_project(self, project_gid):
        response = requests.get(self.server_url + '/projects/' + project_gid + "/operations")
        return response.content

