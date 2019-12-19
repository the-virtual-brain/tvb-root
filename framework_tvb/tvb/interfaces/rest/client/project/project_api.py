import requests

BASE_PATH = "http://127.0.0.1:9090/api/"


class ProjectApi:

    def get_projects_list(self, user_id):
        response = requests.get(BASE_PATH + 'projects/' + user_id)
        return response.content

    def get_data_in_project(self, project_id):
        response = requests.get(BASE_PATH + 'datatypes/project/' + project_id)
        return response.content

    def get_operations_in_project(self, project_id):
        response = requests.get(BASE_PATH + 'operations/' + project_id)
        return response.content

