# -*- coding: utf-8 -*-
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
import os
import requests
import json
import pooch
from pathlib import Path
from zipfile import ZipFile

from .base import BaseDataset
from .zenodo import Zenodo, Record, BASE_URL

class TVBZenodoDataset(BaseDataset):

    CONCEPTID = "3417206"
    
    def __init__(self, version= "2.7"):
        """
        Constructor for TVB_Data class 

        parameters
        -----------

        version: str
              - Version number of the dataset, Default value is 2.7

        """
        super().__init__(version)
        self.cached_file = pooch.os_cache("pooch")/ "tvb_cached_responses.txt"
 
        try:
            self.recid = self.read_cached_response()[version]['conceptrecid']
            
        except KeyError:
            self.log.warning(f"Failed to read data from cached response.")
            self.recid = Zenodo().get_versions_info(self.CONCEPTID)[version]            
            self.update_cached_response()
        
        except:
            self.log.warning(f"Failed to get the desired version {version} of TVB_Data, please check if version {version} is available as a public record on zenodo.org or Please check your internet connection")

        #TODO add logging errors method by catching the exact exceptions. 
        self.rec = Record(self.read_cached_response()[self.version])

    def download(self):
        """
        Downloads the dataset to the cached location, skips download is file already present at the path.
        """
        self.rec.download()

    def fetch_data(self, file_name=None, extract_dir=None):        
        """
        Fetches the data 

        parameters:
        -----------
        file_name: str
                - Name of the file from the downloaded zip file to fetch. If `None`, extracts whole archive. Default is `None` 
        extract_dir: str
                - Path where you want to extract the archive, if `None` extracts the archive to current working directory. Default is `None` 


        returns: Pathlib.Path
            path of the file which was extracted
        """

        

        try:
            file_path = self.rec.file_loc['tvb_data.zip']
        except:
            self.download()
            file_path = self.rec.file_loc['tvb_data.zip']

        if (extract_dir==None):
            extract_dir = TvbProfile.current.DATASETS_FOLDER

        extract_dir = Path(extract_dir).expanduser()

        if file_name == None:
            ZipFile(file_path).extractall(path=extract_dir)

            if extract_dir.is_absolute():
                return extract_dir

            return Path.cwd()/ extract_dir

        with ZipFile(file_path) as zf:
            file_names_in_zip = zf.namelist()
        zf.close()

        file_names_in_zip = {str(Path(i).name): i for i in file_names_in_zip}
        if extract_dir==None:
            ZipFile(file_path).extract(file_names_in_zip[file_name])

        ZipFile(file_path).extract(file_names_in_zip[file_name], path = extract_dir)


        if extract_dir.is_absolute():
            return extract_dir / file_names_in_zip[file_name]


        return Path.cwd()/ extract_dir / file_names_in_zip[file_name]


    def update_cached_response(self):
        """
        gets responses from zenodo server and saves them to cache file. 
        """
        
        file_dir = self.cached_file
        
        responses = {}

        url = f"{BASE_URL}records?q=conceptrecid:{self.CONCEPTID}&all_versions=true"

        for hit in requests.get(url).json()['hits']['hits']:
            version = hit['metadata']['version']
            response = hit 

            responses[version] = response

        with open(file_dir, "w") as fp:
            json.dump(responses, fp)
        fp.close()
        self.log.warning("Updated the cache response file")
        return 

    def read_cached_response(self):
        """
        reads responses from the cache file.

        """
        
        file_dir = self.cached_file


        with open(file_dir) as fp:
            responses = json.load(fp)

        fp.close()


        responses = dict(responses)
        return responses

    
    def describe(self):
        return self.rec.describe()

    def get_record(self):
        return self.recid

    def __str__(self):
        return f"TVB Data version : {self.version}"

    def __eq__(self, other):
        if isinstace(other, TVBZenodoDataset):
            return self.rec == tvb_data.rec
        return False

