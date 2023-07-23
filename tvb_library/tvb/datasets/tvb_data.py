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
import shutil

from .base import BaseDataset
from .zenodo import Zenodo, Record, BASE_URL

class TVBZenodoDataset(BaseDataset):

    CONCEPTID = "3417206"
    
    def __init__(self, version= "2.7", extract_dir = None):
        """
        Constructor for TVB_Data class 

        parameters
        -----------

        version: str
              - Version number of the dataset, Default value is 2.7
        
        extract_dir: str
              - path where you want to extract the archive.
              - If `extract_dir` is None, Dataset is downloaded at location according to your profile settings.

        """
        super().__init__(version, extract_dir)
        self.cached_dir = self.extract_dir / ".cache" 
        self.cached_file = self.cached_dir / "tvb_cached_responses.txt"
        self.files_in_zip_dict = None

        if  not self.cached_dir.is_dir():
            self.cached_dir.mkdir(parents=True)

        try:
            self.recid = self.read_cached_response()[version]['conceptrecid']
            
        except :
            self.log.warning(f"Failed to read data from cached response.")
            self.recid = Zenodo().get_versions_info(self.CONCEPTID)[version]            
            self.update_cached_response()
         
        self.rec = Record(self.read_cached_response()[self.version])

    def download(self, path=None, fname=None):
        """
        Downloads the dataset to `path`
        parameters
        -----------
        path: 
            - path where you want to download the Dataset.
            - If `path` is None, Dataset is downloaded at location according to your profile settings. 
        fname: 
            - The name that will be used to save the file. Should NOT include the full the path, just the file name (it will be appended to path). 
            - If fname is None, file will be saved with a unique name that contains hash of the file and the last part of the url from where the file would be fetched. 
        """
        if path == None:
            path = self.cached_dir
        self.rec.download(path, fname)

    def _fetch_data(self, file_name):        
        """
        Function to fetch the file having `file_name` as name of the file. The function checks if the dataset is downloaded or not. If not, function downloads the dataset and then extracts/unzip the file.

        parameters:
        -----------
        file_name: str
                - Name of the file from the downloaded zip file to fetch. Also accepts relative path of the file with respect to tvb_data.zip. This is useful when having multiple files with same name.
    
        returns: str
            path of the extracted/Unzipped file.
        """

        extract_dir = self.extract_dir 

        try:
            file_path = self.rec.file_loc['tvb_data.zip']
        except:
            self.download(path = self.cached_dir, fname=f"tvb_data_{self.version}.zip")
            file_path = self.rec.file_loc['tvb_data.zip']

        if self.files_in_zip_dict == None:
            self.files_in_zip_dict = self.read_zipfile_structure(file_path=file_path)

        file_name = file_name.strip()


        if file_name.startswith("tvb_data"):
            if file_name in self.files_in_zip_dict[str(Path(file_name).name)] :
                ZipFile(file_path).extract(file_name, path=extract_dir)
    
                if extract_dir.is_absolute():
                    return str(extract_dir / file_name)
                return str(Path.cwd()/ extract_dir / file_name)
            else:
                self.log.error("file_name not found, please mention correct relative file path")

        elif len(self.files_in_zip_dict[file_name]) == 1:
            ZipFile(file_path).extract(self.files_in_zip_dict[file_name][0], path=extract_dir)
            
            if extract_dir.is_absolute():
                return str(extract_dir / self.files_in_zip_dict[file_name][0])
            return str(Path.cwd()/ extract_dir / self.files_in_zip_dict[file_name][0])
        
        
        elif len(self.files_in_zip_dict[file_name]) > 1:

            self.log.error(f"""There are more than 1 files with same names in the zip file. Please mention relative path of the file with respect to the tvb_data.zip. 
                           file_name should be one of the following paths: {self.files_in_zip_dict[file_name]}""")
            raise NameError(f"file name should be one of the {self.files_in_zip_dict[file_name]}, but got {file_name}")


    def fetch_all_data(self):

        if self.files_in_zip_dict == None:
            self.download(path = self.cached_dir, fname=f"tvb_data_{self.version}.zip")
            self.files_in_zip_dict = self.read_zipfile_structure(self.rec.file_loc['tvb_data.zip'])
 
        
        for  file_paths in self.files_in_zip_dict.values():
            for file_path in file_paths:
                self.fetch_data(file_path)

        if self.extract_dir.is_absolute():
            return str(self.extract_dir)
        return str(Path.cwd()/self.extract_dir)


    def delete_data(self):
        """
        Deletes the `tvb_data` folder in the `self.extract_dir` directory. 
        """
        _dir = self.extract_dir / "tvb_data"
        shutil.rmtree(_dir)
        self.log.info(f"deleting {self.extract_dir/'tvb_data'} directory.")


    def update_cached_response(self):
        """
        Gets responses from zenodo server and saves them to a cache file. 
        """
        
        file_dir = self.cached_file
        
        responses = {}

        url = f"{BASE_URL}records?q=conceptrecid:{self.CONCEPTID}&all_versions=true"

        for hit in requests.get(url).json()['hits']['hits']:
            version = hit['metadata']['version']
            response = hit 

            responses[version] = response 

        Path(file_dir).touch(exist_ok=True)

        with open(file_dir, "w") as fp:
            json.dump(responses, fp)
        fp.close()
        self.log.warning("Updated the cache response file")
        return 

    def read_cached_response(self):
        """
        Reads responses from the cache file.
        """
        
        file_dir = self.cached_file

        with open(file_dir) as fp:
            responses = json.load(fp)
        fp.close()

        responses = dict(responses)
        return responses

    
    def describe(self):
        """
        Returns the project description mentioned on the zenodo website. 
        """
        return self.rec.describe()

    def get_recordid(self):
        """
        returns record id of the dataset  
        """
        return self.recid

    def __eq__(self, other):
        if isinstance(other, TVBZenodoDataset):
            return self.rec == other.rec
        return False
