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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: marmaduke <mw@eml.cc>
"""

import os
import six
import json
import numpy
from scipy import sparse
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm.attributes import InstrumentedAttribute
import tvb.basic.traits.types_mapped_light as mapped
from tvb.basic.traits.util import get
from tvb.basic.traits.core import FILE_STORAGE_NONE, KWARG_STORAGE_PATH, FILE_STORAGE_DEFAULT
from tvb.basic.traits.exceptions import ValidationException, MissingEntityException, StorageException
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.core.traits.core import compute_table_name
from tvb.core.entities import model
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.file.hdf5_storage_manager import HDF5StorageManager
from tvb.core.entities.file.exceptions import MissingDataSetException


class MappedType(model.DataType, mapped.MappedTypeLight):
    """
    Mix-in class combining core Traited mechanics with the db'ed DataType class enabling SQLAlchemy.
    """

    #### Transient fields below
    storage_path = None
    _current_metadata = {}
    framework_metadata = None
    logger = get_logger(__name__)
    _ui_complex_datatype = False


    def __init__(self, **kwargs):
        """
        :param kwargs: initialization arguments for generic class.
                       Traited fields are optional to appear here. If not here, default traited value will be taken.
        """
        if KWARG_STORAGE_PATH in kwargs:
            self.storage_path = kwargs.pop(KWARG_STORAGE_PATH)

        self._current_metadata = dict()
        super(MappedType, self).__init__(**kwargs)


    @declared_attr
    def __tablename__(cls):
        """
        Overwrite field __tablename__ for class.
        :returns: None if MappedType itself, custom table name, to recognize Mapped Table in DB.
        """
        if 'MappedType' in cls.__name__:
            return None
        return cls.compute_table_name()


    @classmethod
    def compute_table_name(cls):
        """
        For current class, if to be persisted in DB, compute proper table name.
        """
        return compute_table_name(cls.__name__)


    def __get__(self, inst, cls):
        """
        Called when an attribute of Type is retrieved on another class/instance.
        """
        if inst is None:
            return self
        if self.trait.bound:
            ### Return simple DB field or cached value
            return get(inst, '__' + self.trait.name, None)
        else:
            return self


    def __set__(self, inst, value):
        """
        Add DB code for when an attribute of MappedType class is set on another entity.
        """
        instance_gid, full_instance = None, None
        if value is None or isinstance(value, (str, unicode)):
            #### We consider the string represents a GID
            instance_gid = value
            if value is not None:
                instances_arr = dao.get_generic_entity(self.__class__, instance_gid, 'gid')
                if len(instances_arr) > 0:
                    full_instance = instances_arr[0]
                else:
                    msg = "Could not set '%s' field on '%s' because there is no '%s' with gid: %s in database." \
                          % (self.trait.name, inst.__class__.__name__, self.__class__.__name__, instance_gid)
                    raise MissingEntityException(msg)
        else:
            instance_gid = value.gid
            full_instance = value
        self._put_value_on_instance(inst, instance_gid)

        if self.trait.bound:
            setattr(inst, '__' + self.trait.name, full_instance)


    def initialize(self):
        """
        Method automatically called immediately after DB-Load.
        """
        self.set_operation_id(self.fk_from_operation)
        return self


    def _validate_before_store(self):
        """
        This method checks if the data stored into this entity is valid, 
        and ready to be stored in DB.
        Method automatically called just before saving entity in DB after Type.configure was called.
        In case data is not valid an Exception should be thrown.
        """
        for key, attr in six.iteritems(self.trait):
            if attr.trait.required:
                # In case of fields with data stored on disk check shape
                if isinstance(attr, mapped.Array):
                    if attr.trait.file_storage != FILE_STORAGE_NONE:
                        # Check if any data stored in corresponding dataset
                        try:
                            self.get_data_shape(key)
                        except MissingDataSetException:
                            raise ValidationException("Could not store '%s' because required array '%s' is missing." %
                                                      (self.__class__.__name__, key))
                        except IOError:
                            raise ValidationException("Could not store '%s' because there is no HDF5 file associated." %
                                                      self.__class__.__name__)

                elif not hasattr(self, key) or getattr(self, key) is None:
                    raise ValidationException("Could not store '%s' because required attribute '%s' is missing." %
                                              (self.__class__.__name__, key))


    def set_operation_id(self, operation_id):
        """
        Setter for FK_operation_id.
        """
        self.fk_from_operation = operation_id
        parent_project = dao.get_project_for_operation(operation_id)
        self.storage_path = FilesHelper().get_project_folder(parent_project, str(operation_id))
        self._storage_manager = None


    # ---------------------------- FILE STORAGE -------------------------------
    ROOT_NODE_PATH = "/"


    def store_data(self, data_name, data, where=ROOT_NODE_PATH):
        """
        Store data into a HDF5 file on disk. Each data will be stored into a 
        dataset with the provided name.
            :param data_name: name of the dataset where to store data
            :param data: data to be stored (can be a list / array / numpy array...)
            :param where: represents the path where to store our dataset (e.g. /data/info)
        """
        store_manager = self._get_file_storage_mng()
        store_manager.store_data(data_name, data, where)
        ### Also store Array specific meta-data.
        meta_dictionary = self.__retrieve_array_metadata(data, data_name)
        self.set_metadata(meta_dictionary, data_name, where=where)


    def store_data_chunk(self, data_name, data, grow_dimension=-1, close_file=True, where=ROOT_NODE_PATH):
        """
        Store data into a HDF5 file on disk by writing chunks. 
        Data will be stored into a data-set with the provided name.
            :param data_name: name of the data-set where to store data
            :param data: data to be stored (can be a list / array / numpy array...)
            :param grow_dimension: The dimension to be used to grow stored array.
                                   If not provided a default value = -1 is used (grow on LAST dimension).
            :param close_file: Specify if the file should be closed automatically after write operation.
                                If not, you have to close file by calling method close_file()
            :param where: represents the path where to store our dataset (e.g. /data/info)
        """
        if isinstance(data, list):
            data = numpy.array(data)
        store_manager = self._get_file_storage_mng()
        store_manager.append_data(data_name, data, grow_dimension, close_file, where)

        ### Start updating array meta-data after new chunk of data stored. 
        new_metadata = self.__retrieve_array_metadata(data, data_name)
        previous_meta = dict()
        if data_name in self._current_metadata:
            previous_meta = self._current_metadata[data_name]
        self.__merge_metadata(new_metadata, previous_meta, data)
        self._current_metadata[data_name] = new_metadata


    def get_data(self, data_name, data_slice=None, where=ROOT_NODE_PATH, ignore_errors=False, close_file=True):
        """
        This method reads data from the given data set based on the slice specification
            :param data_name: Name of the data set from where to read data
            :param data_slice: Specify how to retrieve data from array {e.g [slice(1,10,1),slice(1,6,2)] ]
            :param where: represents the path where dataset is stored (e.g. /data/info)
            :returns: a numpy.ndarray containing filtered data
        """
        store_manager = self._get_file_storage_mng()
        return store_manager.get_data(data_name, data_slice, where, ignore_errors, close_file)


    def get_data_shape(self, data_name, where=ROOT_NODE_PATH):
        """
        This method reads data-shape from the given data set
            :param data_name: Name of the data set from where to read size
            :param where: represents the path where dataset is stored (e.g. /data/info)
            :returns: a shape tuple
        """
        if TvbProfile.current.TRAITS_CONFIGURATION.use_storage and self.trait.use_storage:
            try:
                store_manager = self._get_file_storage_mng()
                return store_manager.get_data_shape(data_name, where)
            except IOError as excep:
                self.logger.warning(str(excep))
                self.logger.warning("Could not read shape from file. Most probably because data was not written....")
                return ()
        else:
            return super(MappedType, self).get_data_shape(data_name)


    def get_info_about_array(self, array_name, included_info=None, mask_array_name=None, key_suffix=''):
        """
        :returns: dictionary {label: value} about an attribute of type mapped.Array
                 Generic information, like Max/Min/Mean/Var are to be retrieved for this array_attr
        """
        result = dict()

        try:
            if (TvbProfile.current.TRAITS_CONFIGURATION.use_storage
                    and self.trait.use_storage and mask_array_name is None):
                summary = self._get_summary_info(array_name, included_info, mask_array_name, key_suffix)
            else:
                summary = mapped.MappedTypeLight._get_summary_info(self, array_name, included_info,
                                                                   mask_array_name, key_suffix)

            ### Before return, prepare names for UI display.
            for key, value in six.iteritems(summary):
                result[array_name.capitalize().replace("_", " ") + " - " + key] = value

        except Exception as exc:
            self.logger.warning(exc)

        return result


    def _get_summary_info(self, array_name, included_info, mask_array_name, key_suffix):
        """
        Overwrite get_summary to read from storage.
        """
        if included_info is None:
            included_info = self.trait[array_name].trait.stored_metadata or self.trait[array_name].stored_metadata
        summary = self.__read_storage_array_metadata(array_name, included_info)
        if (self.METADATA_ARRAY_SHAPE in included_info) and (self.METADATA_ARRAY_SHAPE not in summary):
            summary[self.METADATA_ARRAY_SHAPE] = self.get_data_shape(array_name)
        return summary


    def __read_storage_array_metadata(self, array_name, included_info=None):
        """
        Retrieve from HDF5 specific meta-data about an array.
        """
        summary_hdf5 = self.get_metadata(array_name)
        result = dict()
        if included_info is None:
            if array_name in self.trait:
                included_info = self.trait[array_name].trait.stored_metadata or self.trait[array_name].stored_metadata
            else:
                included_info = []
        for key, value in six.iteritems(summary_hdf5):
            if key in included_info:
                result[key] = value
        return result


    def set_metadata(self, meta_dictionary, data_name='', tvb_specific_metadata=True, where=ROOT_NODE_PATH):
        """
        Set meta-data information for root node or for a given data set.
            :param meta_dictionary: dictionary containing meta info to be stored on node
            :param data_name: name of the data-set where to assign metadata.
                                 If None, metadata is assigned to ROOT node.  
            :param tvb_specific_metadata: specify if the provided metadata is
                                 specific to TVB (keys will have a TVB prefix).
            :param where: represents the path where data-set is stored (e.g. /data/info)
             
        """
        if meta_dictionary is None:
            return
        store_manager = self._get_file_storage_mng()
        store_manager.set_metadata(meta_dictionary, data_name, tvb_specific_metadata, where)


    def persist_full_metadata(self):
        """
        Gather all instrumented attributed on current entity, 
        then write them as meta-data in storage.
        """
        meta_dictionary = {}

        # First we process fields store in DB (from Type)
        for attr in dir(self.__class__):
            value = getattr(self.__class__, attr)
            if attr not in self.METADATA_EXCLUDE_PARAMS and isinstance(value, InstrumentedAttribute):
                capitalized_name = attr[0].upper() + attr[1:]
                if not attr.startswith('_'):
                    meta_dictionary[capitalized_name] = getattr(self, str(attr))

        # Now process object traits
        for key, attr in six.iteritems(self.trait):
            kwd = attr.trait.inits.kwd
            if kwd.get('db', True):
                capitalized_name = key[0].upper() + key[1:]
                if isinstance(attr, MappedType):
                    field_value = getattr(self, "_" + key)
                    if field_value is not None:
                        meta_dictionary[capitalized_name] = field_value
                elif isinstance(attr, mapped.Array):
                    pass
                else:
                    field_value = getattr(self, key)
                    if field_value is not None and hasattr(attr, 'to_json'):
                        ### Some traites classes have a specific JSON encoder/encoder
                        meta_dictionary[capitalized_name] = attr.to_json(field_value)
                    else:
                        meta_dictionary[capitalized_name] = json.dumps(field_value)

        # Now store collected meta data
        self.set_metadata(meta_dictionary)


    def load_from_metadata(self, meta_dictionary):
        """
        This method loads data from provided dictionary into current instance
        """
        for attr in dir(self.__class__):
            value = getattr(self.__class__, attr)
            if attr not in self.METADATA_EXCLUDE_PARAMS and isinstance(value, InstrumentedAttribute):
                capitalized_name = attr[0].upper() + attr[1:]
                if not attr.startswith('_'):
                    setattr(self, attr, self._get_meta_value(meta_dictionary, capitalized_name))

        # Now process object traits
        for key, attr in six.iteritems(self.trait):
            kwd = attr.trait.inits.kwd
            if kwd.get('db', True):
                capitalized_name = key[0].upper() + key[1:]
                field_value = self._get_meta_value(meta_dictionary, capitalized_name)
                if field_value is not None:
                    if isinstance(attr, MappedType):
                        setattr(self, "_" + key, field_value)
                    elif isinstance(attr, mapped.Array):
                        pass
                    else:
                        if hasattr(attr, 'from_json'):
                            ### Some traites classes have a specific JSON encoder/encoder
                            field_value = attr.from_json(field_value)
                        else:
                            self.logger.debug("Unpacking " + str(key) + " : " + str(field_value))
                            field_value = json.loads(field_value)
                        setattr(self, key, field_value)


    @staticmethod
    def _get_meta_value(meta_dictionary, meta_key):
        """Utility method. Get meta_key from meta_dictionary, if found key, or None."""
        if meta_key in meta_dictionary.keys():
            return meta_dictionary[meta_key]
        else:
            return None


    def remove_metadata(self, meta_key, data_name='', tvb_specific_metadata=True,
                        where=ROOT_NODE_PATH):
        """
        Remove meta-data information for root node or for a given data set.
            :param meta_key: name of the metadata attribute to be removed
            :param data_name: name of the data-set from where to delete metadata.
                                  If None, metadata will be removed from ROOT node.
            :param tvb_specific_metadata: specify if the provided metadata is
                                  specific to TVB (keys will have a TVB prefix).
            :param where: represents the path where data-set is stored (e.g. /data/info)
        """
        store_manager = self._get_file_storage_mng()
        store_manager.remove_metadata(meta_key, data_name, tvb_specific_metadata, where)


    def get_metadata(self, data_name='', where=ROOT_NODE_PATH):
        """
        Retrieve meta-data information for root node or for a given data set.
            :param data_name: name of the data-set for which to read metadata.
                                 If None, read metadata from ROOT node.
            :param where: represents the path where data-set is stored (e.g. /data/info)
            :returns: a dictionary containing all metadata associated with the node
        """
        store_manager = self._get_file_storage_mng()
        return store_manager.get_metadata(data_name, where)


    def close_file(self):
        """
        Close file used to store data.
        """
        for data_name, new_metadata in six.iteritems(self._current_metadata):
            ## Remove transient metadata, used just for performance issues
            if self._METADATA_ARRAY_SIZE in new_metadata:
                del new_metadata[self._METADATA_ARRAY_SIZE]
            self.set_metadata(new_metadata, data_name)
        store_manager = self._get_file_storage_mng()
        store_manager.close_file()


    def _get_file_storage_mng(self):
        """
        Build the manager responsible for storing data into a file on disk
        """
        if not hasattr(self, "_storage_manager") or self._storage_manager is None:
            file_name = self.get_storage_file_name()
            self._storage_manager = HDF5StorageManager(self.storage_path, file_name)
        return self._storage_manager


    def get_storage_file_name(self):
        """
        This method returns the name of the file where data will be stored.
        """
        return "%s_%s%s" % (self.__class__.__name__, self.gid, FilesHelper.TVB_STORAGE_FILE_EXTENSION)


    def get_storage_file_path(self):
        """
        This method returns FULL path to the file which stores data 
        """
        return os.path.join(self.storage_path, self.get_storage_file_name())


    # ---------------------------- END FILE STORAGE --------------------------------


    # ---------------------------- ARRAY ATTR METADATA ----------------------------
    # -------- see also store_data, store_data_chunk and close_file----------------

    def __retrieve_array_metadata(self, data, data_name):
        """
        :param data: New NumPy array to invoke meta-data methods on.
        :param data_name: String, representing attribute name.  
        """
        if data_name not in self.trait:
            ### Ignore non traited attributes (e.g. sparse-matrix sub-sections).
            return dict()
        traited_attr = self.trait[data_name].trait.stored_metadata or self.trait[data_name].stored_metadata

        if isinstance(data, list):
            data = numpy.array(data)

        meta_dictionary = {}
        for key, value in six.iteritems(self.METADATA_FORMULAS):
            if key != self.METADATA_ARRAY_SHAPE and key in traited_attr:
                try:
                    meta_dictionary[key] = eval(value.replace("$ARRAY$", "data").replace("$MASK$", "data"))
                except Exception:
                    self.logger.exception("Could not evaluate %s on %s" % (value, data_name))

        ## Append size only for chunk computing purposes. It will not be stored in H5 in this form
        ## but we will use H5 native shape support
        meta_dictionary[self._METADATA_ARRAY_SIZE] = data.size
        return meta_dictionary


    def __merge_metadata(self, result_meta, merge_meta, new_data):
        """
        Merge after new chunk added.
        """
        if self.METADATA_ARRAY_VAR in result_meta:
            del result_meta[self.METADATA_ARRAY_VAR]
        if self.METADATA_ARRAY_VAR_NON_ZERO in result_meta:
            del result_meta[self.METADATA_ARRAY_VAR_NON_ZERO]

        if (self.METADATA_ARRAY_MIN in merge_meta
                and merge_meta[self.METADATA_ARRAY_MIN] < result_meta[self.METADATA_ARRAY_MIN]):
            result_meta[self.METADATA_ARRAY_MIN] = merge_meta[self.METADATA_ARRAY_MIN]
        if (self.METADATA_ARRAY_MIN_NON_ZERO in merge_meta
                and merge_meta[self.METADATA_ARRAY_MIN_NON_ZERO] < result_meta[self.METADATA_ARRAY_MIN_NON_ZERO]):
            result_meta[self.METADATA_ARRAY_MIN_NON_ZERO] = merge_meta[self.METADATA_ARRAY_MIN_NON_ZERO]

        if (self.METADATA_ARRAY_MAX in merge_meta
                and merge_meta[self.METADATA_ARRAY_MAX] > result_meta[self.METADATA_ARRAY_MAX]):
            result_meta[self.METADATA_ARRAY_MAX] = merge_meta[self.METADATA_ARRAY_MAX]
        if (self.METADATA_ARRAY_MAX_NON_ZERO in merge_meta
                and merge_meta[self.METADATA_ARRAY_MAX_NON_ZERO] > result_meta[self.METADATA_ARRAY_MAX_NON_ZERO]):
            result_meta[self.METADATA_ARRAY_MAX_NON_ZERO] = merge_meta[self.METADATA_ARRAY_MAX_NON_ZERO]

        if self.METADATA_ARRAY_MEAN in merge_meta and self._METADATA_ARRAY_SIZE in merge_meta:
            prev_no = merge_meta[self._METADATA_ARRAY_SIZE]
            curr_no = new_data.size
            result_meta[self.METADATA_ARRAY_MEAN] = (merge_meta[self.METADATA_ARRAY_MEAN] * prev_no +
                                                     result_meta[self.METADATA_ARRAY_MEAN] * curr_no) / (prev_no +
                                                                                                         curr_no)
            result_meta[self._METADATA_ARRAY_SIZE] = prev_no + curr_no
        if self.METADATA_ARRAY_MEAN_NON_ZERO in merge_meta and self._METADATA_ARRAY_SIZE_NON_ZERO in merge_meta:
            prev_no = merge_meta[self._METADATA_ARRAY_SIZE_NON_ZERO]
            curr_no = new_data.nonzero()[0].shape[0]
            result_meta[self.METADATA_ARRAY_MEAN_NON_ZERO] = (merge_meta[self.METADATA_ARRAY_MEAN_NON_ZERO] * prev_no +
                                                              result_meta[self.METADATA_ARRAY_MEAN_NON_ZERO] * curr_no
                                                              ) / (prev_no + curr_no)
            result_meta[self._METADATA_ARRAY_SIZE] = prev_no + curr_no

    # ---------------------------- END ARRAY ATTR METADATA ------------------------



class Array(mapped.Array):

    def __set__(self, inst, value):
        """
        This is called when an attribute of type Array is set on another class instance.
        :param inst: It is a MappedType instance
        :param value: expected to be of type self.wraps
        :raises Exception: When incompatible type of value is set
        """
        super(Array, self).__set__(inst, value)
        value = getattr(inst, '__' + self.trait.name)
        if (TvbProfile.current.TRAITS_CONFIGURATION.use_storage and inst.trait.use_storage and value is not None
            and (inst is not None and isinstance(inst, mapped.MappedTypeLight)
                 and self.trait.file_storage != FILE_STORAGE_NONE) and value.size > 0):

            if not isinstance(value, self.trait.wraps):
                raise Exception("Invalid DataType!! It expects %s, but is %s for field %s" % (str(self.trait.wraps),
                                                                                              str(type(value)),
                                                                                              str(self.trait.name)))
            self._write_in_storage(inst, value)


    def _get_cached_data(self, inst):
        """
        Overwrite method from library mode array to read from storage when needed.
        """
        cached_data = get(inst, '__' + self.trait.name, None)

        if ((cached_data is None or cached_data.size == 0) and self.trait.file_storage != FILE_STORAGE_NONE
            and TvbProfile.current.TRAITS_CONFIGURATION.use_storage and inst.trait.use_storage
            and isinstance(inst, mapped.MappedTypeLight)):
            ### Data not already loaded, and storage usage
            cached_data = self._read_from_storage(inst)
            setattr(inst, '__' + self.trait.name, cached_data)

        ## Data already loaded, or no storage is used
        return cached_data


    def _write_in_storage(self, inst, value):
        """
        Store value on disk (in h5 file).
        :param inst: Will give us the storage_path, it is a MappedType instance
        :param value: expected to be of type self.wraps
        :raises Exception : when passed value is incompatible (e.g. used with chunks)
        """
        if self.trait.file_storage == FILE_STORAGE_NONE:
            pass
        elif self.trait.file_storage == FILE_STORAGE_DEFAULT:
            inst.store_data(self.trait.name, value)
        else:
            raise StorageException("You should not use SET on attributes-to-be-stored-in-files!")


    def _read_from_storage(self, inst):
        """
        Call correct storage methods, and validation
        :param inst: Will give us the storage_path, it is a MappedType instance
        :returns: entity of self.wraps type
        :raises: Exception when used with chunks
        """
        if self.trait.file_storage == FILE_STORAGE_NONE:
            return None
        elif self.trait.file_storage == FILE_STORAGE_DEFAULT:
            try:
                return inst.get_data(self.trait.name, ignore_errors=True)
            except StorageException as exc:
                self.logger.debug("Missing dataSet " + self.trait.name)
                self.logger.debug(exc)
                return numpy.ndarray(0)
        else:
            raise StorageException("Use get_data(_, slice) not full GET on attributes-stored-in-files!")



class SparseMatrix(mapped.SparseMatrix, Array):
    def _read_from_storage(self, inst):
        """
        Overwrite method from superclass, and call Sparse_Matrix specific reader.
        """
        try:
            return self._read_sparse_matrix(inst, self.trait.name)
        except StorageException as exc:
            self.logger.debug("Missing dataSet " + self.trait.name)
            self.logger.debug(exc)
            return None


    def _write_in_storage(self, inst, value):
        """
        Overwrite method from superclass, and call specific Sparse_Matrix writer.
        """
        self._store_sparse_matrix(inst, value, self.trait.name)


    # ------------------------- STORE and READ sparse matrix to / from HDF5 format 
    ROOT_PATH = "/"

    FORMAT_META = "format"
    DTYPE_META = "dtype"
    DATA_DS = "data"
    INDPTR_DS = "indptr"
    INDICES_DS = "indices"
    ROWS_DS = "rows"
    COLS_DS = "cols"


    @staticmethod
    def extract_sparse_matrix_metadata(mtx):
        info_dict = {SparseMatrix.DTYPE_META: mtx.dtype.str,
                     SparseMatrix.FORMAT_META: mtx.format,
                     MappedType.METADATA_ARRAY_SHAPE: str(mtx.shape),
                     MappedType.METADATA_ARRAY_MAX: mtx.data.max(),
                     MappedType.METADATA_ARRAY_MIN: mtx.data.min(),
                     MappedType.METADATA_ARRAY_MEAN: mtx.mean()}
        return info_dict


    @staticmethod
    def _store_sparse_matrix(inst, mtx, data_name):
        """    
        This method stores sparse matrix into H5 file.
        :param inst: instance on for which to store sparse matrix
        :param mtx: sparse matrix to store
        :param data_name: name of data group which will contain sparse matrix details
        """
        info_dict = SparseMatrix.extract_sparse_matrix_metadata(mtx)
        data_group_path = SparseMatrix.ROOT_PATH + data_name

        # Store data and additional info
        inst.store_data(SparseMatrix.DATA_DS, mtx.data, data_group_path)
        inst.store_data(SparseMatrix.INDPTR_DS, mtx.indptr, data_group_path)
        inst.store_data(SparseMatrix.INDICES_DS, mtx.indices, data_group_path)

        # Store additional info on the group dedicated to sparse matrix
        inst.set_metadata(info_dict, '', True, data_group_path)


    @staticmethod
    def _read_sparse_matrix(inst, data_name):
        """
        Reads SparseMatrix from H5 file and returns an instance of such matrix
        :param inst: instance on for which to read sparse matrix
        :param data_name: name of data group which contains sparse matrix details
        :returns: in instance of sparse matrix with data loaded from H5 file
        """
        constructors = {'csr': sparse.csr_matrix, 'csc': sparse.csc_matrix}

        data_group_path = SparseMatrix.ROOT_PATH + data_name

        info_dict = inst.get_metadata('', data_group_path)

        mtx_format = info_dict[SparseMatrix.FORMAT_META]
        if not isinstance(mtx_format, str):
            mtx_format = mtx_format[0]

        dtype = info_dict[SparseMatrix.DTYPE_META]
        if not isinstance(dtype, str):
            dtype = dtype[0]

        constructor = constructors[mtx_format]
        shape_str = info_dict.get(MappedType.METADATA_ARRAY_SHAPE) or info_dict.get(
            MappedType.METADATA_ARRAY_SHAPE.lower())

        if mtx_format in ['csc', 'csr']:
            data = inst.get_data(SparseMatrix.DATA_DS, where=data_group_path)
            indices = inst.get_data(SparseMatrix.INDICES_DS, where=data_group_path)
            indptr = inst.get_data(SparseMatrix.INDPTR_DS, where=data_group_path)
            shape = eval(shape_str)

            mtx = constructor((data, indices, indptr), shape=shape, dtype=dtype)
            mtx.sort_indices()
        elif mtx_format == 'coo':
            data = inst.get_data(SparseMatrix.DATA_DS, where=data_group_path)
            shape = eval(shape_str)
            rows = inst.get_data(SparseMatrix.ROWS_DS, where=data_group_path)
            cols = inst.get_data(SparseMatrix.COLS_DS, where=data_group_path)

            mtx = constructor((data, sparse.c_[rows, cols].T), shape=shape, dtype=dtype)
        else:
            raise Exception("Unsupported format: %s" % mtx_format)

        return mtx
