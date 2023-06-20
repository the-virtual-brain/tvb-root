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

class BaseDataset:

    def __init__(self, version, target_download=None):

        self.log = get_logger(self.__class__.__module__)
        self.cached_files = None
        self.version = version

        if (extract_dir==None):
            extract_dir = TvbProfile.current.DATASETS_FOLDER

        self.extract_dir = Path(extract_dir).expanduser()
 

    def download(self):
        pass

    def fetch_data(self):
        pass
    
    def get_version(self):
        return self.version
