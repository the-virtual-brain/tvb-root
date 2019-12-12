import requests


# while 1:
#     operation = input('Enter operation:')
#     response = requests.get(url+operation)
#     print(response.text)
#
#     # here we are going to test the launch of simulation
#     if(response == url + 'test_simulation'):
#         response = requests.get(url + operation)
url = "http://127.0.0.1:9090/test_simulation/1"
