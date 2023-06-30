## -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
.. moduleauthor:: Abhijit Deo <f20190041@goa.bits-pilani.ac.in>
"""

# code from https://github.com/space-physics/pyzenodo3 and https://github.com/space-physics/pyzenodo3/pull/9


import requests
import re
import pooch
from typing import List
from pathlib import Path
import json

BASE_URL = "https://zenodo.org/api/"


class Record:
    def __init__(self, data, base_url: str = BASE_URL) -> None:
        """
        Record represents the repsonse from the Zenodo. 
        """

        self.base_url = base_url
        self.data = data
        self.file_loc = {}

   

    def download(self, path: str = None) -> None:

        if 'files' not in self.data:
            raise AttributeError("No files to download! Please check if the record id entered is correct! or the data is publically accessible")

        
        if path == None:
            path = pooch.os_cache("tvb")

        for file in self.data["files"]:
            url = file['links']['self']
            known_hash = file['checksum']
            file_name = file['key']
            
            file_path = pooch.retrieve(url= url, known_hash= known_hash, path = path,progressbar = True)

            self.file_loc[f'{file_name}'] = file_path


            print(f"file {file_name} is downloaded at {file_path}")
        

    def get_latest_version(self):        
        return Zenodo().get_record(self.data['links']['latest'].split("/")[-1])
    
    def describe(self) -> str:
        return self.data['metadata']['description']

    def get_record_id(self) -> str:
        return self.data['conceptrecid']

    def is_open_access(self) -> str:
        return self.data['metadata']['access_right'] != "closed"
    
    def __eq__(self, record_b) -> bool:
        return (self.data == record_b.data)

    def __str__(self) -> str:
        return json.dumps(self.data, indent=2) 



class Zenodo:
    def __init__(self, api_key: str = "", base_url: str = BASE_URL) -> None:
        """
        This class handles all the interactions of the user to the zenodo platform. 

        
        """
        self.base_url = base_url
        self._api_key = api_key
    

    def get_record(self, recid: str) -> Record:
        """
        recid: unique id of the data repository
        """
        url = self.base_url + "records/" + recid
        
        return Record(requests.get(url).json())


    def _get_records(self, params: dict[str, str]) -> List[Record]:
        url = self.base_url + "records?" + urlencode(params)

        return [Record(hit) for hit in requests.get(url).json()["hits"]["hits"]]




    def get_versions_info(self, recid) -> dict:
        """
        recid: unique id of the data repository

        """
        # needs ineternet


        recid = self.get_record(recid).data['metadata']['relations']['version'][0]['parent']['pid_value']

        versions = {}

        url = f"{self.base_url}records?q=conceptrecid:{recid}&all_versions=true" 


        for hit in requests.get(url).json()['hits']['hits']:

            version = hit['metadata']['version']
            recid = hit['doi'].split(".")[-1]
            if hit['metadata']['access_right'] == "closed":
                continue
            versions[version] = recid
        

        return versions




