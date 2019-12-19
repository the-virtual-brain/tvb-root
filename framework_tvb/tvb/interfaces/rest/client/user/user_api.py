import requests

BASE_PATH = "http://127.0.0.1:9090/api/"


class UserApi:

    def get_users(self):
        response = requests.get(BASE_PATH + 'users')
        return response.content

