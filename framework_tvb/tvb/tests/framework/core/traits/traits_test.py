# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import numpy
import os
import shutil
from tvb.basic.traits.types_mapped import MappedType
from tvb.basic.profile import TvbProfile


class TestsMappedTypeStorage():
    """
    Test class for testing mapped type data storage into file.
    Most of the storage functionality is tested in the test suite 
    of HDF5StorageManager 
    """
    def setup_method(self):
        """
        Prepare data for tests
        """
        storage_folder = os.path.join(TvbProfile.current.TVB_STORAGE, "test_hdf5")

        if os.path.exists(storage_folder):
            shutil.rmtree(storage_folder)
        os.makedirs(storage_folder)
        
        # Create data type for which to store data
        self.data_type = MappedType()
        self.data_type.storage_path = storage_folder
        
        self.test_2D_array = numpy.random.random((10, 10))
        self.data_name = "vertex"

    def teardown_method(self):
        """
        Clean up tests data
        """
        if os.path.exists(self.data_type.storage_path):
            shutil.rmtree(self.data_type.storage_path)
    
    def test_store_data(self):
        """
        Test data storage into file
        """
        self.data_type.store_data(self.data_name, self.test_2D_array)
        read_data = self.data_type.get_data(self.data_name)
        numpy.testing.assert_array_equal(self.test_2D_array, read_data, "Did not get the expected data")
    
    def test_store_chunked_data(self):
        """
        Test data storage into file, but splitted in chunks
        """
        self.data_type.store_data_chunk(self.data_name, self.test_2D_array)
        read_data = self.data_type.get_data(self.data_name)
        numpy.testing.assert_array_equal(self.test_2D_array, read_data, "Did not get the expected data")
    
    
    def test_set_metadata(self):
        """
        This test checks assignment of metadata to dataset or storage file  
        """
        # First create some data and check if it is stored
        self.data_type.store_data(self.data_name, self.test_2D_array)
        
        key = "meta_key"
        value = "meva_val"
        self.data_type.set_metadata({key: value}, self.data_name)
        read_meta_data = self.data_type.get_metadata(self.data_name)
        assert value == read_meta_data[key], "Meta value is not correct"
        
        # Now we'll store metadata on file /root node
        self.data_type.set_metadata({key: value})
        read_meta_data = self.data_type.get_metadata()
        assert value == read_meta_data[key], "Meta value is not correct"
        
    def test_remove_metadata(self):
        """
        This test checks removal of metadata from dataset 
        """
        # First create some data and check if it is stored
        self.data_type.store_data(self.data_name, self.test_2D_array)
        
        key = "meta_key"
        value = "meva_val"
        self.data_type.set_metadata({key: value}, self.data_name)
        read_meta_data = self.data_type.get_metadata(self.data_name)
        assert value == read_meta_data[key], "Meta value is not correct"
        
        # Now delete metadata
        self.data_type.remove_metadata(key, self.data_name)
        read_meta_data = self.data_type.get_metadata(self.data_name)
        assert 0 == len(read_meta_data), "There should be no metadata on node"
