# code from https://github.com/space-physics/pyzenodo3 and https://github.com/space-physics/pyzenodo3/pull/9


import requests
import re
import pooch
from pathlib import Path
import json

BASE_URL = "https://zenodo.org/api/"


class Record:
    def __init__(self, data, base_url: str = BASE_URL) -> None:
        self.base_url = base_url
        self.data = data
        self.file_loc = {}

    def describe(self):
        return self.data['metadata']['description']


    def __str__(self):
        return json.dumps(self.data, indent=2) 

    def download(self):

        if 'files' not in self.data:
            raise AttributeError("No files to download! Please check if the record id entered is correct! or the data is publically accessible")

        

        for file in self.data["files"]:
            url = file['links']['self']
            known_hash = file['checksum']
            file_name = file['key']
            
            file_path = pooch.retrieve(url= url, known_hash= known_hash, progressbar=True)
            
            self.file_loc['file_name'] = file_path


            print(f"file {file_name} is downloaded at {file_path}")
        

    def get_latest_version(self):
        
        return Zenodo().get_record(self.data['links']['latest'].split("/")[-1])


    def __eq__(self, record_b):
        return (self.data == record_b.data)



class Zenodo:
    def __init__(self, api_key: str = "", base_url: str = BASE_URL) -> None:
        """
        This class handles all the interactions of the user to the zenodo platform. 

        
        """
        self.base_url = base_url
        self._api_key = api_key
        self.re_github_repo = re.compile(r".*github.com/(.*?/.*?)[/$]")
    

    def get_record(self, recid: str) -> Record:
        """
        recid: unique id of the data repository
        """
        url = self.base_url + "records/" + recid
        
        return Record(requests.get(url).json())


    def _get_records(self, params: dict[str, str]) -> list[Record]:
        url = self.base_url + "records?" + urlencode(params)

        return [Record(hit) for hit in requests.get(url).json()["hits"]["hits"]]




    def get_versions_info(self, recid):

        recid = self.get_record(recid).data['metadata']['relations']['version'][0]['parent']['pid_value']

        print(recid)

        versions = {}

        url = f"{self.base_url}records?q=conceptrecid:{recid}&all_versions=true" 


        for hit in requests.get(url).json()['hits']['hits']:

            version = hit['metadata']['version']
            recid = hit['doi'].split(".")[-1]

            versions[version] = recid
        

        return versions




