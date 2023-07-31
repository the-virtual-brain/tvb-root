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



from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from pathlib import Path
from zipfile import ZipFile

class BaseDataset:

    def __init__(self, version : str , extract_dir : str =None) -> None:

        self.log = get_logger(self.__class__.__module__)
        self.cached_files = None
        self.version = version

        if (extract_dir==None):
            extract_dir = TvbProfile.current.DATASETS_FOLDER

        self.extract_dir = Path(extract_dir).expanduser()
 

    def fetch_data(self, file_name:str) -> str:
        if Path(file_name).is_absolute():
            self.log.warning("Given file name is an absolute path. No operations are done. The path is returned as it is")
            return file_name
        
        return self._fetch_data(file_name)

    def get_version(self) -> str:
        return self.version
    
    def delete_data(self):
        raise NotImplemented

    def _read_zipfile_structure(self, file_path):
        """
        Reads the zipfile structure and returns the dictionary containing file_names as keys and list of relative paths having same file name. 
        """
        with ZipFile(file_path) as zf:
            file_names_in_zip = zf.namelist()
        zf.close()      

        file_names_dict = {}
        for i in file_names_in_zip:
            if str(Path(i).name) not in file_names_dict.keys():
                file_names_dict[str(Path(i).name)] = [i]
            else:
                file_names_dict[str(Path(i).name)].append(i)
        return file_names_dict

    def _fetch_data(self,file_name):
        raise NotImplemented
    
    def _download(self):
        raise NotImplemented
