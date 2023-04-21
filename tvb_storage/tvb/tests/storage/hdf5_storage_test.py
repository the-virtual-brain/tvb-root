# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
    Module used to test storage of TVB data into HDF5 format.
    
    .. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""

import os
import numpy
import shutil
import pytest

from tvb.basic.profile import TvbProfile
from tvb.storage.h5.file.exceptions import MissingDataSetException, IncompatibleFileManagerException, \
    FileStructureException
from tvb.storage.h5.file.hdf5_storage_manager import HDF5StorageManager
from tvb.storage.storage_interface import StorageInterface

# Some constants used by tests

STORAGE_FILE_NAME = "test_data.h5"
DATASET_NAME_1 = "dataset1"
DATASET_NAME_2 = "dataset2"
META_KEY = "meta_key"
META_VALUE = "meta_value"
META_DICT = {META_KEY: META_VALUE}
STORE_PATH = "/node1/node2"


class TestHDF5Storage(object):
    """
    This tests storage of data into HDF5 format (H5 files).
    """

    def setup_method(self):
        """
        Set up the context needed by the tests.
        """
        self.storage_folder = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, "test_hdf5")

        if os.path.exists(self.storage_folder):
            shutil.rmtree(self.storage_folder)
        os.makedirs(self.storage_folder)

        # Now create HDF5 storage instance
        self.storage = HDF5StorageManager(os.path.join(self.storage_folder, STORAGE_FILE_NAME))

        self.test_2D_array = numpy.random.random((10, 10))
        self.test_3D_array = numpy.random.random((3, 3, 3))
        self.test_string_array = numpy.array(
            [["a".encode('utf-8'), "b".encode('utf-8')], ["c".encode('utf-8'), "d".encode('utf-8')]])

    def teardown_method(self):
        """
        Tear down to revert any changes made by a test.
        """
        self.storage.close_file()

        if os.path.exists(self.storage_folder):
            shutil.rmtree(self.storage_folder)

    @staticmethod
    def _assert_arrays_are_equal(expected_arr, current_arr, message=None):
        numpy.testing.assert_array_equal(expected_arr, current_arr, message)

    # -------------- TESTS ---------------------
    def test_invalid_storage_path(self):
        """
        This method will test scenarios where no storage path or storage file is provided
        """
        self.storage_folder = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, "test_hdf5")
        # Test if folder name is None
        with pytest.raises(FileStructureException):
            HDF5StorageManager(None)

    def test_file_creation(self):
        """
        Test if the storage file is created on disk.
        """
        self.storage.store_data(self.test_2D_array, DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)
        full_path = os.path.join(self.storage_folder, STORAGE_FILE_NAME)
        assert os.path.exists(full_path), "Storage file not created."

    def test_simple_data_storage(self):
        """
        Test if simple array data is stored
        """
        self.storage.store_data(self.test_2D_array, DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)
        # Now read data
        read_data = self.storage.get_data(DATASET_NAME_1, None, StorageInterface.ROOT_NODE_PATH, False, True)

        self._assert_arrays_are_equal(self.test_2D_array, read_data, "Did not get the expected data")

    def test_store_data_repeatedly(self):
        """
        Test if simple array data is stored
        """
        self.storage.store_data(self.test_2D_array, DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)
        ## Update with the same shape should work:
        self.storage.store_data(self.test_2D_array, DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)

        try:
            ## But update with a different shape should throw an exception
            self.storage.store_data(self.test_3D_array, DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)
            raise AssertionError("Update with a different shape is expected to fail")
        except IncompatibleFileManagerException:
            pass

        # Now read data
        read_data = self.storage.get_data(DATASET_NAME_1, None, StorageInterface.ROOT_NODE_PATH, False, True)
        self._assert_arrays_are_equal(self.test_2D_array, read_data, "Did not get the expected data")

    def test_simple_data_storage_on_path(self):
        """
        Test if simple array data is stored on a given path
        """
        self.storage.store_data(self.test_2D_array, DATASET_NAME_1, STORE_PATH)
        # Now read data
        read_data = self.storage.get_data(DATASET_NAME_1, None, STORE_PATH, False, True)
        self._assert_arrays_are_equal(self.test_2D_array, read_data)

    def test_string_data_storage(self):
        """
        Test store of string data (array of strings
        """
        self.storage.store_data(self.test_string_array, DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)
        # Now read data
        read_data = self.storage.get_data(DATASET_NAME_1, None, StorageInterface.ROOT_NODE_PATH, False, True)
        self._assert_arrays_are_equal(self.test_string_array, read_data)

    def test_store_none_data(self):
        """
        Test scenario when trying to store None data
        """
        with pytest.raises(FileStructureException):
            self.storage.store_data(None, DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)

    def test_append_data1(self):
        """
        Test data store using append method
        """
        self.storage.append_data(self.test_3D_array, DATASET_NAME_1, 1, True, StorageInterface.ROOT_NODE_PATH)
        read_data = self.storage.get_data(DATASET_NAME_1, None, StorageInterface.ROOT_NODE_PATH, False, True)
        self._assert_arrays_are_equal(self.test_3D_array, read_data)

    def test_append_data2(self):
        """
        Test data store using append method (multiple calls)
        """
        for index in range(self.test_2D_array.shape[-1]):
            slices = (slice(None, None, 1), slice(index, index + 1, 1))

            self.storage.append_data(self.test_2D_array[slices], DATASET_NAME_1, 1, True,
                                     StorageInterface.ROOT_NODE_PATH)

        self.storage.close_file()
        read_data = self.storage.get_data(DATASET_NAME_1, None, StorageInterface.ROOT_NODE_PATH, False, True)
        self._assert_arrays_are_equal(self.test_2D_array, read_data)

    def test_append_data_on_path(self):
        """
        Test data store using append method on a given path
        """
        self.storage.append_data(self.test_3D_array, DATASET_NAME_1, 1, True, STORE_PATH)
        read_data = self.storage.get_data(DATASET_NAME_1, None, STORE_PATH, False, True)
        self._assert_arrays_are_equal(self.test_3D_array, read_data)

    def test_append_other_dimension(self):
        """
        This method test append operation on other dimension
        """
        for index in range(self.test_2D_array.shape[0]):
            slices = (slice(index, index + 1, 1), slice(None, None, 1))

            self.storage.append_data(self.test_2D_array[slices], DATASET_NAME_1, 0, True,
                                     StorageInterface.ROOT_NODE_PATH)

        read_data = self.storage.get_data(DATASET_NAME_1, None, StorageInterface.ROOT_NODE_PATH, False, True)
        self._assert_arrays_are_equal(self.test_2D_array, read_data)

    def test_append_strings(self):
        """
        Test appending strings 
        """
        for index in range(self.test_string_array.shape[-1]):
            slices = (slice(None, None, 1), slice(index, index + 1, 1))

            self.storage.append_data(self.test_string_array[slices], DATASET_NAME_1, 1, True,
                                     StorageInterface.ROOT_NODE_PATH)

        read_data = self.storage.get_data(DATASET_NAME_1, None, StorageInterface.ROOT_NODE_PATH, False, True)
        self._assert_arrays_are_equal(self.test_string_array, read_data)

    def test_append_none_data(self):
        """
        Test appending null value to dataset
        """
        with pytest.raises(FileStructureException):
            self.storage.append_data(None, DATASET_NAME_1, 1, True, StorageInterface.ROOT_NODE_PATH)

    def test_append_without_closing_file(self):
        """
        Test appending data but keeping hdf5 file open. 
        """
        for index in range(self.test_3D_array.shape[-1]):
            slices = (slice(None, None, 1), slice(index, index + 1, 1), slice(None, None, 1))
            self.storage.append_data(self.test_3D_array[slices], DATASET_NAME_1, 1, False,
                                     StorageInterface.ROOT_NODE_PATH)

        self.storage.close_file()

        read_data = self.storage.get_data(DATASET_NAME_1, None, StorageInterface.ROOT_NODE_PATH, False, True)
        self._assert_arrays_are_equal(self.test_3D_array, read_data)

    def test_close_file_multiple_time(self):
        """
        Test closing H5 file multiple times.
        """
        for index in range(self.test_3D_array.shape[-1]):
            slices = (slice(None, None, 1), slice(index, index + 1, 1), slice(None, None, 1))
            self.storage.append_data(self.test_3D_array[slices], DATASET_NAME_1, 1, False,
                                     StorageInterface.ROOT_NODE_PATH)

        self.storage.close_file()
        self.storage.close_file()

        read_data = self.storage.get_data(DATASET_NAME_1, None, StorageInterface.ROOT_NODE_PATH, False, True)
        self._assert_arrays_are_equal(self.test_3D_array, read_data)

    def test_delete_dataset(self):
        """
        This test checks deletion of a dataset from H5 file.
        """
        self.storage.store_data(self.test_2D_array, DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)
        # Now check if data was persisted. 
        read_data = self.storage.get_data(DATASET_NAME_1, None, StorageInterface.ROOT_NODE_PATH, False, True)
        self._assert_arrays_are_equal(self.test_2D_array, read_data)

        # Now delete dataset
        self.storage.remove_data(DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)
        with pytest.raises(MissingDataSetException):
            self.storage.get_data(DATASET_NAME_1, None, StorageInterface.ROOT_NODE_PATH, False, True)

    def test_delete_dataset_on_path(self):
        """
        This test checks deletion of a dataset from H5 file, dataset which is stored 
        under a given path
        """
        self.storage.store_data(self.test_2D_array, DATASET_NAME_1, STORE_PATH)
        # Now check if data was persisted. 
        read_data = self.storage.get_data(DATASET_NAME_1, None, STORE_PATH, False, True)
        self._assert_arrays_are_equal(self.test_2D_array, read_data)

        # Now delete dataset
        self.storage.remove_data(DATASET_NAME_1, STORE_PATH)
        with pytest.raises(MissingDataSetException):
            self.storage.get_data(DATASET_NAME_1, None, StorageInterface.ROOT_NODE_PATH, False, True)

    def test_delete_missing_dataset(self):
        """
        This test checks deletion of non existing  dataset.
        """
        with pytest.raises(FileStructureException):
            self.storage.remove_data(DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)

    def test_read_sliced_data(self):
        """
        This test checks reading of data on slices.
        """
        self.storage.store_data(self.test_2D_array, DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)

        # Now check if data was persisted, but read it using slices
        for index in range(len(self.test_2D_array)):
            sl = slice(index, index + 1, 1)
            read_data = self.storage.get_data(DATASET_NAME_1, sl, StorageInterface.ROOT_NODE_PATH, False, True)
            self._assert_arrays_are_equal(self.test_2D_array[sl], read_data)

    def test_add_metadata(self):
        """
        This method checks metadata add for root or a dataset
        """
        # Create a dataset 
        self.storage.store_data(self.test_2D_array, DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)

        # Now add meta info to it.
        self.storage.set_metadata(META_DICT, DATASET_NAME_1, True, StorageInterface.ROOT_NODE_PATH)
        read_meta_value = self.storage.get_metadata(DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)
        assert META_VALUE == read_meta_value[META_KEY]

        # Now we'll test adding metadata on root node
        self.storage.set_metadata(META_DICT, '', True, StorageInterface.ROOT_NODE_PATH)
        read_meta_value = self.storage.get_metadata('', StorageInterface.ROOT_NODE_PATH)
        assert META_VALUE == read_meta_value[META_KEY]

    def test_add_metadata_on_path(self):
        """
        This method checks metadata add for a dataset stored under a given path
        """
        # Create a dataset 
        self.storage.store_data(self.test_2D_array, DATASET_NAME_1, STORE_PATH)

        # Now add meta info to it.
        self.storage.set_metadata(META_DICT, DATASET_NAME_1, True, STORE_PATH)
        read_meta_value = self.storage.get_metadata(DATASET_NAME_1, STORE_PATH)
        assert META_VALUE == read_meta_value[META_KEY]

    def test_delete_metadata(self):
        """
        Test deletion of metadata for a dataset or root node.
        """
        # Create a dataset 
        self.storage.store_data(self.test_2D_array, DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)

        # Now add meta info to it.
        self.storage.set_metadata(META_DICT, DATASET_NAME_1, True, StorageInterface.ROOT_NODE_PATH)
        read_meta_data = self.storage.get_metadata(DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)
        assert META_VALUE == read_meta_data[META_KEY]
        self.storage.remove_metadata(META_KEY, DATASET_NAME_1, True, StorageInterface.ROOT_NODE_PATH)
        read_meta_data = self.storage.get_metadata(DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)
        assert 0 == len(read_meta_data), "There should be no metadata stored on dataset"

        # Now we'll test removal of metadata on root node
        self.storage.set_metadata(META_DICT, '', True, StorageInterface.ROOT_NODE_PATH)
        read_meta_data = self.storage.get_metadata('', StorageInterface.ROOT_NODE_PATH)
        assert META_VALUE == read_meta_data[META_KEY], "Retrieved meta value is not correct"
        self.storage.remove_metadata(META_KEY, '', True, StorageInterface.ROOT_NODE_PATH)
        read_meta_data = self.storage.get_metadata('', StorageInterface.ROOT_NODE_PATH)
        assert 1 == len(read_meta_data), "There should be no metadata stored on root node, except data version"

    def test_delete_metadata_on_path(self):
        """
        Test deletion of metadata for a dataset stored on a given path
        """
        # Create a dataset 
        self.storage.store_data(self.test_2D_array, DATASET_NAME_1, STORE_PATH)

        # Now add meta info to it.
        self.storage.set_metadata(META_DICT, DATASET_NAME_1, True, STORE_PATH)
        read_meta_data = self.storage.get_metadata(DATASET_NAME_1, STORE_PATH)
        assert META_VALUE == read_meta_data[META_KEY]
        self.storage.remove_metadata(META_KEY, DATASET_NAME_1, True, STORE_PATH)
        read_meta_data = self.storage.get_metadata(DATASET_NAME_1, STORE_PATH)
        assert 0 == len(read_meta_data), "There should be no metadata stored on dataset"

    def test_delete_missing_metadata(self):
        """
        This method tests removal of a missing metadata
        """
        self.storage.store_data(self.test_2D_array, DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)
        # Test delete on a node
        read_meta_data = self.storage.get_metadata(DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)
        assert 0 == len(read_meta_data), "There should be no metadata stored on dataset"

        # Test delete on root node
        read_meta_data = self.storage.get_metadata('', StorageInterface.ROOT_NODE_PATH)
        assert 1, len(read_meta_data) == "There should be no metadata stored on root node, except data version"

    def test_delete_metadata_missing_dataset(self):
        """
        Test retrieval of metadata for a missing dataset.
        """
        # Create
        self.storage.store_data(self.test_2D_array, DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)
        with pytest.raises(MissingDataSetException):
            self.storage.get_metadata(DATASET_NAME_2, StorageInterface.ROOT_NODE_PATH)

    def test_concurent_file_access(self):
        """
        This method tests scenario when HDF5 file is opened concurrent for read & write
        """
        new_storage = HDF5StorageManager(os.path.join(self.storage_folder, STORAGE_FILE_NAME))
        new_storage.store_data(self.test_2D_array, DATASET_NAME_2, StorageInterface.ROOT_NODE_PATH)

        for index in range(self.test_3D_array.shape[-1]):
            slices = (slice(None, None, 1), slice(index, index + 1, 1), slice(None, None, 1))
            # Write into file, but leave file open
            self.storage.append_data(self.test_3D_array[slices], DATASET_NAME_1, 1, False,
                                     StorageInterface.ROOT_NODE_PATH)

            # Now try to read file
            read_data = self.storage.get_data(DATASET_NAME_2, None, StorageInterface.ROOT_NODE_PATH, False, True)
            self._assert_arrays_are_equal(self.test_2D_array, read_data)

    def test_add_metadata_non_tvb_specific(self):
        """
        This method checks metadata add for root or a dataset
        """
        # Create a dataset 
        self.storage.store_data(self.test_2D_array, DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)

        # Now add meta info to it.
        self.storage.set_metadata(META_DICT, DATASET_NAME_1, False, StorageInterface.ROOT_NODE_PATH)
        read_meta_value = self.storage.get_metadata(DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)
        assert META_VALUE == read_meta_value[META_KEY]

        # Now we'll test adding metadata on root node
        self.storage.set_metadata(META_DICT, '', False, StorageInterface.ROOT_NODE_PATH)
        read_meta_value = self.storage.get_metadata('', StorageInterface.ROOT_NODE_PATH)
        assert META_VALUE == read_meta_value[META_KEY]

    def test_delete_metadata_non_tvb_specific(self):
        """
        Test deletion of metadata for a dataset or root node.
        """
        # Create a dataset 
        self.storage.store_data(self.test_2D_array, DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)

        # Now add meta info to it.
        self.storage.set_metadata(META_DICT, DATASET_NAME_1, False, StorageInterface.ROOT_NODE_PATH)
        read_meta_data = self.storage.get_metadata(DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)
        assert META_VALUE == read_meta_data[META_KEY]
        self.storage.remove_metadata(META_KEY, DATASET_NAME_1, False, StorageInterface.ROOT_NODE_PATH)
        read_meta_data = self.storage.get_metadata(DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)
        assert 0 == len(read_meta_data)

        # Now we'll test removal of metadata on root node
        self.storage.set_metadata(META_DICT, '', False, StorageInterface.ROOT_NODE_PATH)
        read_meta_data = self.storage.get_metadata('', StorageInterface.ROOT_NODE_PATH)
        assert META_VALUE == read_meta_data[META_KEY]
        self.storage.remove_metadata(META_KEY, '', False, StorageInterface.ROOT_NODE_PATH)
        read_meta_data = self.storage.get_metadata('', StorageInterface.ROOT_NODE_PATH)
        assert 1 == len(read_meta_data)

    def test_data_version(self):
        """
        Test if data version meta data is assigned when H5 file is created
        """
        self.storage.store_data(self.test_2D_array, DATASET_NAME_1, StorageInterface.ROOT_NODE_PATH)

        # Now read meta data for root node
        read_data = self.storage.get_metadata('', StorageInterface.ROOT_NODE_PATH)
        self._assert_arrays_are_equal(TvbProfile.current.version.DATA_VERSION,
                                      read_data[TvbProfile.current.version.DATA_VERSION_ATTRIBUTE])
