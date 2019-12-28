import requests
from tvb.interfaces.rest.client.main_api import MainApi


class DataTypeApi(MainApi):

    def retrieve_datatype(self, datatype_gid, download_folder):
        response = requests.get(self.server_url + "/datatypes/" + datatype_gid)
        content_disposition = response.headers['Content-Disposition']
        start_index = content_disposition.index("filename=") + 9
        end_index = len(content_disposition)
        file_name = content_disposition[start_index:end_index]

        file_path = download_folder + '/' + file_name

        if response.status_code == 200:
            with open(file_path, 'wb') as local_file:
                for chunk in response.iter_content(chunk_size=128):
                    local_file.write(chunk)

            return True
        return False

    def get_operations_for_datatype(self, datatype_gid):
        response = requests.get(self.server_url + "/datatypes/" + datatype_gid + "/operations")
        return response.content
