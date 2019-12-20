import requests
from tvb.interfaces.rest.client.main_api import MainApi


class UserApi(MainApi):

    def get_users(self):
        response = requests.get(self.server_url + "/users")
        return response.content

    def get_projects_list(self, username):
        response = requests.get(self.server_url + "/users/" + username + "/projects")
        return response.content
