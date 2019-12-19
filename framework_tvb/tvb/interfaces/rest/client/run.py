import requests


# while 1:
#     operation = input('Enter operation:')
#     response = requests.get(url+operation)
#     print(response.text)
#
#     # here we are going to test the launch of simulation
#     if(response == url + 'test_simulation'):
#         response = requests.get(url + operation)
from tvb.interfaces.rest.client.datatype.datatype_api import DataTypeApi

url = "http://127.0.0.1:9090/test_simulation/1"

datatype = DataTypeApi()
datatype.retrieve_datatype("e300cde19d754be7ab3f7123d52f3131", 'C:/Users/Robert.Vincze/TVB/PROJECTS/HappyProject/TEMP')
