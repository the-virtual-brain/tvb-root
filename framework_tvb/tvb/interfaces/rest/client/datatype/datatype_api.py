import requests


BASE_PATH = "http://127.0.0.1:9090/api/"


class DataTypeApi:

    def retrieve_datatype(self, gid, folder):
        response = requests.get(BASE_PATH + "datatypes/" + gid)
        content_disposition = response.headers['Content-Disposition']
        start_index = content_disposition.index("filename=") + 9
        end_index = len(content_disposition)
        file_name = content_disposition[start_index:end_index]

        file_path = folder + '/' + file_name

        if response.status_code == 200:
            with open(file_path, 'wb') as local_file:
                for chunk in response.iter_content(chunk_size=128):
                    local_file.write(chunk)
