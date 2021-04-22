# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

import pytest
import os

from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.file_storage_factory import FileStorageFactory
from tvb.file.exceptions import UnsupportedFileStorageException
from tvb.file.lab import *


class TestFileStorageFactory:

    def test_invalid_storage_path(self):
        """
        This method will test scenarios where no storage path or storage file is provided
        """
        self.storage_folder = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, "test_hdf5")
        # Test if folder name is None
        with pytest.raises(FileStructureException):
            FileStorageFactory.get_file_storage(None, "test_data.h5")

        # Test if file name is None
        with pytest.raises(FileStructureException):
            FileStorageFactory.get_file_storage(self.storage_folder, None)

    def test_invalid_storage(self):
        """
        This method will test scenarios when an invalid storage is provided
        """
        TvbProfile.current.file_storage = "invalid_storage"
        self.storage_folder = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, "test_hdf5")

        with pytest.raises(UnsupportedFileStorageException):
            FileStorageFactory.get_file_storage(self.storage_folder, "test_data.h5")
