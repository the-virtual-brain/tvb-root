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
Persistence of data in HDF5 format.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""

import os
import copy
import threading
import h5py as hdf5
import numpy as numpy
import tvb.core.utils as utils
from datetime import datetime
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.exceptions import FileStructureException, MissingDataSetException
from tvb.core.entities.file.exceptions import IncompatibleFileManagerException, MissingDataFileException
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.transient.structure_entities import GenericMetaData
from tvb.core.data_encryption_handler import DataEncryptionHandler

# Create logger for this module
LOG = get_logger(__name__)

LOCK_OPEN_FILE = threading.Lock()


class HDF5StorageManager(object):
    """
    This class is responsible for saving / loading data in HDF5 file / format.
    """
    __file_title_ = "TVB data file"
    __storage_full_name = None
    __hfd5_file = None

    TVB_ATTRIBUTE_PREFIX = "TVB_"
    ROOT_NODE_PATH = "/"
    BOOL_VALUE_PREFIX = "bool:"
    DATETIME_VALUE_PREFIX = "datetime:"
    DATE_TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
    LOCKS = {}

    def __init__(self, storage_folder, file_name, buffer_size=600000):
        """
        Creates a new storage manager instance.
        :param buffer_size: the size in Bytes of the amount of data that will be buffered before writing to file.
        """
        if storage_folder is None:
            raise FileStructureException("Please provide the folder where to store data")
        if file_name is None:
            raise FileStructureException("Please provide the file name where to store data")
        self.__storage_full_name = os.path.join(storage_folder, file_name)
        self.__buffer_size = buffer_size
        self.__buffer_array = None
        self.data_buffers = {}

    def is_valid_hdf5_file(self):
        """
        This method checks if specified file exists and if it has correct HDF5 format
        :returns: True is file exists and has HDF5 format. False otherwise.
        """
        try:
            return os.path.exists(self.__storage_full_name) and hdf5.is_hdf5(self.__storage_full_name)
        except RuntimeError:
            return False

    def store_data(self, dataset_name, data_list, where=ROOT_NODE_PATH):
        """
        This method stores provided data list into a data set in the H5 file.
        
        :param dataset_name: Name of the data set where to store data
        :param data_list: Data to be stored
        :param where: represents the path where to store our dataset (e.g. /data/info)
        """
        if dataset_name is None:
            dataset_name = ''
        if where is None:
            where = self.ROOT_NODE_PATH

        data_to_store = self._check_data(data_list)

        try:
            LOG.debug("Saving data into data set: %s" % dataset_name)
            # Open file in append mode ('a') to allow adding multiple data sets in the same file
            hdf5_file = self._open_h5_file()

            full_dataset_name = where + dataset_name
            if full_dataset_name not in hdf5_file:
                hdf5_file.create_dataset(full_dataset_name, data=data_to_store)

            elif hdf5_file[full_dataset_name].shape == data_to_store.shape:
                hdf5_file[full_dataset_name][...] = data_to_store[...]

            else:
                raise IncompatibleFileManagerException("Cannot update existing H5 DataSet %s with a different shape. "
                                                       "Try defining it as chunked!" % full_dataset_name)

        finally:
            # Now close file
            self.close_file()
            DataEncryptionHandler.push_folder_to_sync(FilesHelper.get_project_folder_from_h5(self.__storage_full_name))

    def append_data(self, dataset_name, data_list, grow_dimension=-1, close_file=True, where=ROOT_NODE_PATH):
        """
        This method appends data to an existing data set. If the data set does not exists, create it first.
        
        :param dataset_name: Name of the data set where to store data
        :param data_list: Data to be stored / appended
        :param grow_dimension: The dimension to be used to grow stored array. By default will grow on the LAST dimension
        :param close_file: Specify if the file should be closed automatically after write operation. If not, 
            you have to close file by calling method close_file()
        :param where: represents the path where to store our dataset (e.g. /data/info)
        
        """
        if dataset_name is None:
            dataset_name = ''
        if where is None:
            where = self.ROOT_NODE_PATH
        data_to_store = self._check_data(data_list)
        data_buffer = self.data_buffers.get(where + dataset_name, None)

        if data_buffer is None:
            hdf5_file = self._open_h5_file()
            datapath = where + dataset_name
            if datapath in hdf5_file:
                dataset = hdf5_file[datapath]
                self.data_buffers[datapath] = HDF5StorageManager.H5pyStorageBuffer(dataset,
                                                                                   buffer_size=self.__buffer_size,
                                                                                   buffered_data=data_to_store,
                                                                                   grow_dimension=grow_dimension)
            else:
                data_shape_list = list(data_to_store.shape)
                data_shape_list[grow_dimension] = None
                data_shape = tuple(data_shape_list)
                dataset = hdf5_file.create_dataset(where + dataset_name, data=data_to_store, shape=data_to_store.shape,
                                                   dtype=data_to_store.dtype, maxshape=data_shape)
                self.data_buffers[datapath] = HDF5StorageManager.H5pyStorageBuffer(dataset,
                                                                                   buffer_size=self.__buffer_size,
                                                                                   buffered_data=None,
                                                                                   grow_dimension=grow_dimension)
        else:
            if not data_buffer.buffer_data(data_to_store):
                data_buffer.flush_buffered_data()
        if close_file:
            self.close_file()
            DataEncryptionHandler.push_folder_to_sync(FilesHelper.get_project_folder_from_h5(self.__storage_full_name))

    def remove_data(self, dataset_name, where=ROOT_NODE_PATH):
        """
        Deleting a data set from H5 file.
        
        :param dataset_name:name of the data set to be deleted
        :param where: represents the path where dataset is stored (e.g. /data/info)
          
        """
        LOG.debug("Removing data set: %s" % dataset_name)
        if dataset_name is None:
            dataset_name = ''
        if where is None:
            where = self.ROOT_NODE_PATH
        try:
            # Open file in append mode ('a') to allow data remove
            hdf5_file = self._open_h5_file()
            del hdf5_file[where + dataset_name]

        except KeyError:
            LOG.warn("Trying to delete data set: %s but current file does not contain it." % dataset_name)
            raise FileStructureException("Could not locate dataset: %s" % dataset_name)
        finally:
            self.close_file()

    def get_data(self, dataset_name, data_slice=None, where=ROOT_NODE_PATH, ignore_errors=False, close_file=True):
        """
        This method reads data from the given data set based on the slice specification
        
        :param close_file: Automatically close after reading the current field
        :param ignore_errors: return None in case of error, or throw exception
        :param dataset_name: Name of the data set from where to read data
        :param data_slice: Specify how to retrieve data from array {e.g (slice(1,10,1),slice(1,6,2)) }
        :param where: represents the path where dataset is stored (e.g. /data/info)  
        :returns: a numpy.ndarray containing filtered data
        
        """
        LOG.debug("Reading data from data set: %s" % dataset_name)
        if dataset_name is None:
            dataset_name = ''
        if where is None:
            where = self.ROOT_NODE_PATH

        data_path = where + dataset_name
        try:
            # Open file to read data
            hdf5_file = self._open_h5_file('r')
            if data_path in hdf5_file:
                data_array = hdf5_file[data_path]
                # Now read data
                if data_slice is None:
                    result = data_array[()]
                    if isinstance(result, hdf5.Empty):
                        return numpy.empty([])
                    return result
                else:
                    return data_array[data_slice]
            else:
                if not ignore_errors:
                    LOG.error("Trying to read data from a missing data set: %s" % dataset_name)
                    raise MissingDataSetException("Could not locate dataset: %s" % dataset_name)
                else:
                    return None
        finally:
            if close_file:
                self.close_file()

    def get_data_shape(self, dataset_name, where=ROOT_NODE_PATH):
        """
        This method reads data-size from the given data set 
        
        :param dataset_name: Name of the data set from where to read data
        :param where: represents the path where dataset is stored (e.g. /data/info)  
        :returns: a tuple containing data size
        
        """
        LOG.debug("Reading data from data set: %s" % dataset_name)
        if dataset_name is None:
            dataset_name = ''
        if where is None:
            where = self.ROOT_NODE_PATH

        try:
            # Open file to read data
            hdf5_file = self._open_h5_file('r')
            data_array = hdf5_file[where + dataset_name]
            return data_array.shape
        except KeyError:
            LOG.debug("Trying to read data from a missing data set: %s" % dataset_name)
            raise MissingDataSetException("Could not locate dataset: %s" % dataset_name)

        finally:
            self.close_file()

    def set_metadata(self, meta_dictionary, dataset_name='', tvb_specific_metadata=True, where=ROOT_NODE_PATH):
        """
        Set meta-data information for root node or for a given data set.
        
        :param meta_dictionary: dictionary containing meta info to be stored on node
        :param dataset_name: name of the dataset where to assign metadata. If None, metadata is assigned to ROOT node.
        :param tvb_specific_metadata: specify if the provided metadata is TVB specific (All keys will have a TVB prefix)
        :param where: represents the path where dataset is stored (e.g. /data/info)     
        
        """
        LOG.debug("Setting metadata on node: %s" % dataset_name)
        if dataset_name is None:
            dataset_name = ''
        if where is None:
            where = self.ROOT_NODE_PATH

        # Open file to read data
        hdf5_file = self._open_h5_file()
        try:
            node = hdf5_file[where + dataset_name]
        except KeyError:
            LOG.debug("Trying to set metadata on a missing data set: %s" % dataset_name)
            node = hdf5_file.create_dataset(where + dataset_name, (1,))

        try:
            # Now set meta-data
            for meta_key in meta_dictionary:
                key_to_store = meta_key
                if tvb_specific_metadata:
                    key_to_store = self.TVB_ATTRIBUTE_PREFIX + meta_key

                processed_value = self._serialize_value(meta_dictionary[meta_key])
                node.attrs[key_to_store] = processed_value
        finally:
            self.close_file()
            DataEncryptionHandler.push_folder_to_sync(FilesHelper.get_project_folder_from_h5(self.__storage_full_name))

    @staticmethod
    def serialize_bool(value):
        return HDF5StorageManager.BOOL_VALUE_PREFIX + str(value)

    def _serialize_value(self, value):
        """
        This method takes a value which will be stored as metadata and 
        apply some transformation if necessary
        
        :param value: value which is planned to be stored
        :returns:  value to be stored
        
        """
        if value is None:
            return ''
        # Transform boolean to string and prefix it
        if isinstance(value, bool):
            return self.serialize_bool(value)
        # Transform date to string and append prefix
        elif isinstance(value, datetime):
            return self.DATETIME_VALUE_PREFIX + utils.date2string(value, date_format=self.DATE_TIME_FORMAT)
        else:
            return value

    def remove_metadata(self, meta_key, dataset_name='', tvb_specific_metadata=True, where=ROOT_NODE_PATH):
        """
        Remove meta-data information for root node or for a given data set.
        
        :param meta_key: name of the metadata attribute to be removed
        :param dataset_name: name of the dataset from where to delete metadata. 
            If None, metadata will be removed from ROOT node.
        :param tvb_specific_metadata: specify if the provided metadata is specific to TVB (keys will have a TVB prefix).
        :param where: represents the path where dataset is stored (e.g. /data/info)
             
        """
        LOG.debug("Deleting metadata: %s for dataset: %s" % (meta_key, dataset_name))
        if dataset_name is None:
            dataset_name = ''
        if where is None:
            where = self.ROOT_NODE_PATH
        try:
            # Open file to read data
            hdf5_file = self._open_h5_file()
            node = hdf5_file[where + dataset_name]

            # Now delete metadata
            key_to_remove = meta_key
            if tvb_specific_metadata:
                key_to_remove = self.TVB_ATTRIBUTE_PREFIX + meta_key
            del node.attrs[key_to_remove]
        except KeyError:
            LOG.error("Trying to delete metadata on a missing data set: %s" % dataset_name)
            raise FileStructureException("Could not locate dataset: %s" % dataset_name)
        except AttributeError:
            LOG.error("Trying to delete missing metadata %s" % meta_key)
            raise FileStructureException("There is no metadata named %s on this node" % meta_key)
        finally:
            self.close_file()

    def get_metadata(self, dataset_name='', where=ROOT_NODE_PATH):
        """
        Retrieve ALL meta-data information for root node or for a given data set.

        :param dataset_name: name of the dataset for which to read metadata. If None, read metadata from ROOT node.
        :param where: represents the path where dataset is stored (e.g. /data/info)  
        :returns: a dictionary containing all metadata associated with the node
        
        """
        LOG.debug("Retrieving metadata for dataset: %s" % dataset_name)
        if dataset_name is None:
            dataset_name = ''
        if where is None:
            where = self.ROOT_NODE_PATH

        meta_key = ""
        try:
            # Open file to read data
            hdf5_file = self._open_h5_file('r')
            node = hdf5_file[where + dataset_name]
            # Now retrieve metadata values
            all_meta_data = {}

            for meta_key in node.attrs:
                new_key = meta_key
                if meta_key.startswith(self.TVB_ATTRIBUTE_PREFIX):
                    new_key = meta_key[len(self.TVB_ATTRIBUTE_PREFIX):]
                value = node.attrs[meta_key]
                all_meta_data[new_key] = self._deserialize_value(value)
            return all_meta_data

        except KeyError:
            msg = "Trying to read data from a missing data set: %s" % (where + dataset_name)
            LOG.warning(msg)
            raise MissingDataSetException(msg)
        except AttributeError:
            msg = "Trying to get value for missing metadata %s" % meta_key
            LOG.exception(msg)
            raise FileStructureException(msg)
        except Exception:

            msg = "Failed to read metadata from H5 file! %s" % self.__storage_full_name
            LOG.exception(msg)
            raise FileStructureException(msg)
        finally:
            self.close_file()

    def get_file_data_version(self):
        """
        Checks the data version for the current file.
        """
        if not os.path.exists(self.__storage_full_name):
            raise MissingDataFileException("File storage data not found at path %s" % (self.__storage_full_name,))

        if self.is_valid_hdf5_file():
            metadata = self.get_metadata()
            data_version = TvbProfile.current.version.DATA_VERSION_ATTRIBUTE
            if data_version in metadata:
                return metadata[data_version]
            else:
                raise IncompatibleFileManagerException("Could not find TVB specific data version attribute %s in file: "
                                                       "%s." % (data_version, self.__storage_full_name))
        raise IncompatibleFileManagerException("File %s is not a hdf5 format file. Are you using the correct "
                                               "manager for this file?" % (self.__storage_full_name,))

    def get_gid_attribute(self):
        """
        Used for obtaining the gid of the DataType of
        which data are stored in the current file.
        """
        if self.is_valid_hdf5_file():
            metadata = self.get_metadata()
            if GenericMetaData.KEY_GID in metadata:
                return metadata[GenericMetaData.KEY_GID]
            else:
                raise IncompatibleFileManagerException("Could not find the Gid attribute in the "
                                                       "input file %s." % self.__storage_full_name)
        raise IncompatibleFileManagerException("File %s is not a hdf5 format file. Are you using the correct "
                                               "manager for this file?" % (self.__storage_full_name,))

    def _deserialize_value(self, value):
        """
        This method takes value loaded from H5 file and transform it to TVB data. 
        """
        if value is not None:
            if isinstance(value, numpy.string_):
                if len(value) == 0:
                    value = None
                else:
                    value = str(value)

            if isinstance(value, str):
                if value.startswith(self.BOOL_VALUE_PREFIX):
                    # Remove bool prefix and transform to bool
                    return utils.string2bool(value[len(self.BOOL_VALUE_PREFIX):])
                if value.startswith(self.DATETIME_VALUE_PREFIX):
                    # Remove datetime prefix and transform to datetime
                    return utils.string2date(value[len(self.DATETIME_VALUE_PREFIX):], date_format=self.DATE_TIME_FORMAT)

        return value

    def __aquire_lock(self):
        """
        Aquire a unique lock for each different file path on the system.
        """
        lock = self.LOCKS.get(self.__storage_full_name, None)
        if lock is None:
            lock = threading.Lock()
            self.LOCKS[self.__storage_full_name] = lock
        lock.acquire()

    def __release_lock(self):
        """
        Aquire a unique lock for each different file path on the system.
        """
        lock = self.LOCKS.get(self.__storage_full_name, None)
        if lock is None:
            raise Exception("Some lock was deleted without being released beforehand.")
        lock.release()

    def close_file(self):
        """
        The synchronization of open/close doesn't seem to be needed anymore for h5py in
        contrast to PyTables for concurrent reads. However since it shouldn't add that
        much overhead in most situation we'll leave it like this for now since in case
        of concurrent writes(metadata) this provides extra safety.
        """
        try:
            self.__aquire_lock()
            self.__close_file()
        finally:
            self.__release_lock()

    def _open_h5_file(self, mode='a'):
        """
        The synchronization of open/close doesn't seem to be needed anymore for h5py in
        contrast to PyTables for concurrent reads. However since it shouldn't add that
        much overhead in most situation we'll leave it like this for now since in case
        of concurrent writes(metadata) this provides extra safety.
        """
        try:
            self.__aquire_lock()
            file_obj = self.__open_h5_file(mode)
        finally:
            self.__release_lock()
        return file_obj

    def __close_file(self):
        """
        Close file used to store data.
        """
        hdf5_file = self.__hfd5_file

        # Try to close file only if it was opened before
        if hdf5_file is not None and hdf5_file.id.valid:
            LOG.debug("Closing file: %s" % self.__storage_full_name)
            try:
                for h5py_buffer in self.data_buffers.values():
                    h5py_buffer.flush_buffered_data()
                self.data_buffers = {}
                hdf5_file.close()
            except Exception as excep:
                # Do nothing is this situation.
                # The file is correctly closed, but the list of open files on HDF5 is not updated in a synch manner.
                # del _open_files[filename] might throw KeyError
                LOG.exception(excep)
            if not hdf5_file.id.valid:
                self.__hfd5_file = None

    # -------------- Private methods  --------------
    def __open_h5_file(self, mode='a'):
        """
        Open file for reading, writing or append. 
        
        :param mode: Mode to open file (possible values are w / r / a).
                    Default value is 'a', to allow adding multiple data to the same file.
        :returns: returns the file which stores data in HDF5 format opened for read / write according to mode param
        
        """
        if self.__storage_full_name is None:
            raise FileStructureException("Invalid storage file. Please provide a valid path.")
        try:
            # Check if file is still open from previous writes.
            if self.__hfd5_file is None or not self.__hfd5_file.id.valid:
                file_exists = os.path.exists(self.__storage_full_name)

                # bug in some versions of hdf5 on windows prevent creating file with mode='a'
                if not file_exists and mode == 'a':
                    mode = 'w'

                LOG.debug("Opening file: %s in mode: %s" % (self.__storage_full_name, mode))
                self.__hfd5_file = hdf5.File(self.__storage_full_name, mode, libver='latest')

                # If this is the first time we access file, write data version
                if not file_exists:
                    os.chmod(self.__storage_full_name, TvbProfile.current.ACCESS_MODE_TVB_FILES)
                    attr_name = self.TVB_ATTRIBUTE_PREFIX + TvbProfile.current.version.DATA_VERSION_ATTRIBUTE
                    self.__hfd5_file['/'].attrs[attr_name] = TvbProfile.current.version.DATA_VERSION
        except (IOError, OSError) as err:
            LOG.exception("Could not open storage file.")
            raise FileStructureException("Could not open storage file. %s" % err)

        return self.__hfd5_file

    @staticmethod
    def _check_data(data_list):
        """
        Check if the data to be stores is in a good format. If not adapt it.
        """
        if data_list is None:
            raise FileStructureException("Could not store null data")

        if not (isinstance(data_list, list) or isinstance(data_list, numpy.ndarray)):
            raise FileStructureException("Invalid data type. Could not store data of type:" + str(type(data_list)))

        data_to_store = data_list
        if isinstance(data_to_store, list):
            data_to_store = numpy.array(data_list)
        if data_to_store.shape == ():
            data_to_store = hdf5.Empty("f")
        return data_to_store

    class H5pyStorageBuffer(object):
        """
        Helper class in order to buffer data for append operations, to limit the number of actual
        HDD I/O operations.
        """

        def __init__(self, h5py_dataset, buffer_size=300, buffered_data=None, grow_dimension=-1):
            self.buffered_data = buffered_data
            self.buffer_size = buffer_size
            if h5py_dataset is None:
                raise MissingDataSetException("A H5pyStorageBuffer instance must have a h5py dataset for which the"
                                              "buffering is done. Please supply one to the 'h5py_dataset' parameter.")
            self.h5py_dataset = h5py_dataset
            self.grow_dimension = grow_dimension

        def buffer_data(self, data_list):
            """
            Add data_list to an internal buffer in order to improve performance for append_data type of operations.
            :returns: True if buffer is still fine, \
                      False if a flush is necessary since the buffer is full
            """
            if self.buffered_data is None:
                self.buffered_data = data_list
            else:
                self.buffered_data = self.__custom_numpy_append(self.buffered_data, data_list)
            if self.buffered_data.nbytes > self.buffer_size:
                return False
            else:
                return True

        def __custom_numpy_append(self, array1, array2):
            array_1_shape = numpy.array(array1.shape)
            array_2_shape = numpy.array(array2.shape)
            result_shape = copy.deepcopy(array_1_shape)
            result_shape[self.grow_dimension] += array_2_shape[self.grow_dimension]
            result_array = numpy.empty(shape=tuple(result_shape), dtype=array1.dtype)
            full_slice = slice(None, None, None)
            full_index = [full_slice for _ in array_1_shape]
            full_index[self.grow_dimension] = slice(0, array_1_shape[self.grow_dimension], None)
            result_array[tuple(full_index)] = array1
            full_index[self.grow_dimension] = slice(array_1_shape[self.grow_dimension],
                                                    result_shape[self.grow_dimension], None)
            result_array[tuple(full_index)] = array2
            return result_array

        def flush_buffered_data(self):
            """
            Append the data buffered so far to the input dataset using :param grow_dimension: as the dimension that
            will be expanded. 
            """
            if self.buffered_data is not None:
                current_shape = self.h5py_dataset.shape
                new_shape = list(current_shape)
                new_shape[self.grow_dimension] += self.buffered_data.shape[self.grow_dimension]
                # Create the required slice to which the new data will be added.
                # For example if the 3nd dimension of a 4D datashape (74, 1, 100, 1)
                # we want to get the slice (:, :, 100:200, :) in order to add 100 new entries
                full_slice = slice(None, None, None)
                slice_to_add = slice(current_shape[self.grow_dimension], new_shape[self.grow_dimension], None)
                append2address = [full_slice for _ in new_shape]
                append2address[self.grow_dimension] = slice_to_add
                # Do the data reshape and copy the new data
                self.h5py_dataset.resize(tuple(new_shape))
                self.h5py_dataset[tuple(append2address)] = self.buffered_data
                self.buffered_data = None
