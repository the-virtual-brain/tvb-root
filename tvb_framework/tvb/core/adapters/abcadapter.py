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
Root classes for adding custom functionality to the code.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Yann Gordon <yann@tvb.invalid>
"""
from datetime import datetime
import importlib
import json
import os
import typing
import uuid
from abc import ABCMeta, abstractmethod
from enum import Enum
from functools import wraps
import numpy
import psutil
from six import add_metaclass

from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import Attr, HasTraits, List
from tvb.basic.profile import TvbProfile
from tvb.core.adapters.exceptions import IntrospectionException, LaunchException, InvalidParameterException
from tvb.core.adapters.exceptions import NoMemoryAvailableException
from tvb.core.entities.generic_attributes import GenericAttributes
from tvb.core.entities.load import load_entity_by_gid
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.entities.model.model_operation import Algorithm
from tvb.core.entities.storage import dao
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import Form
from tvb.core.neotraits.h5 import H5File
from tvb.core.neotraits.view_model import DataTypeGidAttr, ViewModel
from tvb.core.utils import date2string, LESS_COMPLEX_TIME_FORMAT
from tvb.storage.storage_interface import StorageInterface

LOGGER = get_logger("ABCAdapter")


def nan_not_allowed():
    """
    Annotation that guides NumPy behavior in case of floating point errors.
    The NumPy default is to just print a warning to sys.stdout, this annotation will raise our custom exception.
    This annotation will enforce that an exception is thrown in case a floating point error is produced.

    e.g. If NaN is take as input and not produced inside the context covered by this annotation,
         nothing happens from this method p.o.v.

    e.g. If inside a method annotated with this method we have something like numpy.log(-1),
         then LaunchException is thrown.
    """

    def wrap(func):

        @wraps(func)
        def new_function(*args, **kw):
            old_fp_error_handling = numpy.seterr(divide='raise', invalid='raise')
            try:
                return func(*args, **kw)
            except FloatingPointError:
                raise LaunchException('NaN values were generated during launch. Stopping operation execution.')
            finally:
                numpy.seterr(**old_fp_error_handling)

        return new_function

    return wrap


def nan_allowed():
    """
    Annotation that configures NumPy not to throw an exception in case of floating points errors are computed.
    It should be used on Adapter methods where computation of NaN/ Inf/etc. is allowed.
    """

    def wrap(func):

        @wraps(func)
        def new_function(*args, **kw):
            old_fp_error_handling = numpy.seterr(all='ignore')
            try:
                return func(*args, **kw)
            finally:
                numpy.seterr(**old_fp_error_handling)

        return new_function

    return wrap


class ABCAdapterForm(Form):
    @staticmethod
    def get_required_datatype():
        """
        Each Adapter's computation is based on a main Datatype. This method should keep the class of it.
        This Datatype will be stored to DB at introspection time.
        :return: DataType class
        """
        raise NotImplementedError

    @staticmethod
    def get_filters():
        """
        Should keep filters for the required_datatype. These filters are stored in DB at introspection time.
        :return: FilterChain
        """
        raise NotImplementedError

    @staticmethod
    def get_input_name():
        """
        The Form's input name for the required_datatype. Will be stored in DB at introspection time.
        :return: str
        """
        raise NotImplementedError

    @staticmethod
    def get_view_model():
        """
        Should keep the ViewModel class that corresponds to the current Adapter.
        :return: ViewModel class
        """
        raise NotImplementedError

    def fill_from_post_plus_defaults(self, form_data):
        self.fill_from_trait(self.get_view_model()())
        for field in self.fields:
            if field.name in form_data:
                field.fill_from_post(form_data)


class AdapterLaunchModeEnum(Enum):
    SYNC_SAME_MEM = 'sync_same_mem'
    SYNC_DIFF_MEM = 'sync_diff_mem'
    ASYNC_DIFF_MEM = 'async_diff_mem'


@add_metaclass(ABCMeta)
class ABCAdapter(object):
    """
    Root Abstract class for all TVB Adapters. 
    """
    # model.Algorithm instance that will be set for each adapter class created by in build_adapter method
    stored_adapter = None
    launch_mode = AdapterLaunchModeEnum.ASYNC_DIFF_MEM

    def __init__(self):
        self.generic_attributes = GenericAttributes()
        self.generic_attributes.subject = DataTypeMetaData.DEFAULT_SUBJECT
        self.storage_interface = StorageInterface()
        # Will be populate with current running operation's identifier
        self.operation_id = None
        self.user_id = None
        self.submitted_form = None
        self.log = get_logger(self.__class__.__module__)

    @classmethod
    def get_group_name(cls):
        if hasattr(cls, "_ui_group") and hasattr(cls._ui_group, "name"):
            return cls._ui_group.name
        return None

    @classmethod
    def get_group_description(cls):
        if hasattr(cls, "_ui_group") and hasattr(cls._ui_group, "description"):
            return cls._ui_group.description
        return None

    @classmethod
    def get_ui_name(cls):
        if hasattr(cls, "_ui_name"):
            return cls._ui_name
        else:
            return cls.__name__

    @classmethod
    def get_ui_description(cls):
        if hasattr(cls, "_ui_description"):
            return cls._ui_description

    @classmethod
    def get_ui_subsection(cls):
        if hasattr(cls, "_ui_subsection"):
            return cls._ui_subsection

        if hasattr(cls, "_ui_group") and hasattr(cls._ui_group, "subsection"):
            return cls._ui_group.subsection

    @staticmethod
    def can_be_active():
        """
        To be overridden where needed (e.g. Matlab dependent adapters).
        :return: By default True, and False when the current Adapter can not be executed in the current env
        for various reasons
        """
        return True

    def submit_form(self, form):
        self.submitted_form = form

    # TODO separate usage of get_form_class (returning a class) and return of a submitted instance
    def get_form(self):
        if self.submitted_form is not None:
            return self.submitted_form
        return self.get_form_class()

    @abstractmethod
    def get_form_class(self):
        return None

    def get_adapter_fragments(self, view_model):
        """
        The result will be used for introspecting and checking operation changed input
        params from the defaults, to show in web gui.
        :return: a list of ABCAdapterForm classes, in case the current Adapter GUI
        will be composed of multiple sub-forms.
        """
        return {}

    def get_view_model_class(self):
        return self.get_form_class().get_view_model()

    @abstractmethod
    def get_output(self):
        """
        Describes inputs and outputs of the launch method.
        """

    def configure(self, view_model):
        """
        To be implemented in each Adapter that requires any specific configurations
        before the actual launch.
        """

    @abstractmethod
    def get_required_memory_size(self, view_model):
        """
        Abstract method to be implemented in each adapter. Should return the required memory
        for launching the adapter.
        """

    @abstractmethod
    def get_required_disk_size(self, view_model):
        """
        Abstract method to be implemented in each adapter. Should return the required memory
        for launching the adapter in kilo-Bytes.
        """

    def get_execution_time_approximation(self, view_model):
        """
        Method should approximate based on input arguments, the time it will take for the operation 
        to finish (in seconds).
        """
        return -1

    @abstractmethod
    def launch(self, view_model):
        """
         To be implemented in each Adapter.
         Will contain the logic of the Adapter.
         Takes a ViewModel with data, dependency direction is: Adapter -> Form -> ViewModel
         Any returned DataType will be stored in DB, by the Framework.
        :param view_model: the data model corresponding to the current adapter
        """

    def add_operation_additional_info(self, message):
        """
        Adds additional info on the operation to be displayed in the UI. Usually a warning message.
        """
        current_op = dao.get_operation_by_id(self.operation_id)
        current_op.additional_info = message
        dao.store_entity(current_op)

    def extract_operation_data(self, operation):
        operation = dao.get_operation_by_id(operation.id)
        self.operation_id = operation.id
        self.current_project_id = operation.project.id
        self.user_id = operation.fk_launched_by

    def _ensure_enough_resources(self, available_disk_space, view_model):
        # Compare the amount of memory the current algorithms states it needs,
        # with the average between the RAM available on the OS and the free memory at the current moment.
        # We do not consider only the free memory, because some OSs are freeing late and on-demand only.
        total_free_memory = psutil.virtual_memory().free + psutil.swap_memory().free
        total_existent_memory = psutil.virtual_memory().total + psutil.swap_memory().total
        memory_reference = (total_free_memory + total_existent_memory) / 2
        adapter_required_memory = self.get_required_memory_size(view_model)

        if adapter_required_memory > memory_reference:
            msg = "Machine does not have enough RAM memory for the operation (expected %.2g GB, but found %.2g GB)."
            raise NoMemoryAvailableException(msg % (adapter_required_memory / 2 ** 30, memory_reference / 2 ** 30))

        # Compare the expected size of the operation results with the HDD space currently available for the user
        # TVB defines a quota per user.
        required_disk_space = self.get_required_disk_size(view_model)
        if available_disk_space < 0:
            msg = "You have exceeded you HDD space quota by %.2f MB Stopping execution."
            raise NoMemoryAvailableException(msg % (- available_disk_space / 2 ** 10))
        if available_disk_space < required_disk_space:
            msg = ("You only have %.2f GB of disk space available but the operation you "
                   "launched might require %.2f Stopping execution...")
            raise NoMemoryAvailableException(msg % (available_disk_space / 2 ** 20, required_disk_space / 2 ** 20))
        return required_disk_space

    def _update_operation_entity(self, operation, required_disk_space):
        operation.start_now()
        operation.estimated_disk_size = required_disk_space
        dao.store_entity(operation)

    @nan_not_allowed()
    def _prelaunch(self, operation, view_model, available_disk_space=0):
        """
        Method to wrap LAUNCH.
        Will prepare data, and store results on return.
        """
        self.extract_operation_data(operation)
        self.generic_attributes.fill_from(view_model.generic_attributes)
        self.configure(view_model)
        required_disk_size = self._ensure_enough_resources(available_disk_space, view_model)
        self._update_operation_entity(operation, required_disk_size)

        result = self.launch(view_model)

        if not isinstance(result, (list, tuple)):
            result = [result, ]
        self.__check_integrity(result)
        return self._capture_operation_results(result)

    def _capture_operation_results(self, result):
        """
        After an operation was finished, make sure the results are stored
        in DB storage and the correct meta-data,IDs are set.
        """
        data_type_group_id = None
        operation = dao.get_operation_by_id(self.operation_id)
        if operation.user_group is None or len(operation.user_group) == 0:
            operation.user_group = date2string(datetime.now(), date_format=LESS_COMPLEX_TIME_FORMAT)
            operation = dao.store_entity(operation)
        if self._is_group_launch():
            data_type_group_id = dao.get_datatypegroup_by_op_group_id(operation.fk_operation_group).id

        count_stored = 0
        if result is None:
            return "", count_stored

        group_type = None  # In case of a group, the first not-none type is sufficient to memorize here
        for res in result:
            if res is None:
                continue
            if not res.fixed_generic_attributes:
                res.fill_from_generic_attributes(self.generic_attributes)
            res.fk_from_operation = self.operation_id
            res.fk_datatype_group = data_type_group_id

            associated_file = h5.path_for_stored_index(res)
            if os.path.exists(associated_file):
                if not res.fixed_generic_attributes:
                    with H5File.from_file(associated_file) as f:
                        f.store_generic_attributes(self.generic_attributes)
                # Compute size-on disk, in case file-storage is used
                res.disk_size = self.storage_interface.compute_size_on_disk(associated_file)

            dao.store_entity(res)
            res.after_store()
            group_type = res.type
            count_stored += 1

        if count_stored > 0 and self._is_group_launch():
            # Update the operation group name
            operation_group = dao.get_operationgroup_by_id(operation.fk_operation_group)
            operation_group.fill_operationgroup_name(group_type)
            dao.store_entity(operation_group)

        return 'Operation ' + str(self.operation_id) + ' has finished.', count_stored

    def __check_integrity(self, result):
        """
        Check that the returned parameters for LAUNCH operation
        are of the type specified in the adapter's interface.
        """
        for result_entity in result:
            if result_entity is None:
                continue
            if not self.__is_data_in_supported_types(result_entity):
                msg = "Unexpected output DataType %s"
                raise InvalidParameterException(msg % type(result_entity))

    def __is_data_in_supported_types(self, data):

        if data is None:
            return True
        for supported_type in self.get_output():
            if isinstance(data, supported_type):
                return True
        # Data can't be mapped on any supported type !!
        return False

    def _is_group_launch(self):
        """
        Return true if this adapter is launched from a group of operations
        """
        operation = dao.get_operation_by_id(self.operation_id)
        return operation.fk_operation_group is not None

    def load_entity_by_gid(self, data_gid):
        # type: (typing.Union[uuid.UUID, str]) -> DataType
        """
        Load a generic DataType, specified by GID.
        """
        idx = load_entity_by_gid(data_gid)
        if idx and self.generic_attributes.parent_burst is None:
            # Only in case the BurstConfiguration references hasn't been set already, take it from the current DT
            self.generic_attributes.parent_burst = idx.fk_parent_burst
        return idx

    def load_traited_by_gid(self, data_gid):
        # type: (typing.Union[uuid.UUID, str]) -> HasTraits
        """
        Load a generic HasTraits instance, specified by GID.
        """
        index = self.load_entity_by_gid(data_gid)
        return h5.load_from_index(index)

    def load_with_references(self, dt_gid):
        # type: (typing.Union[uuid.UUID, str]) -> HasTraits
        dt_index = self.load_entity_by_gid(dt_gid)
        h5_path = h5.path_for_stored_index(dt_index)
        dt, _ = h5.load_with_references(h5_path)
        return dt

    def view_model_to_has_traits(self, view_model):
        # type: (ViewModel) -> HasTraits
        has_traits_class = view_model.linked_has_traits
        has_traits = has_traits_class()
        view_model_class = type(view_model)
        if not has_traits_class:
            raise Exception("There is no linked HasTraits for this ViewModel {}".format(type(view_model)))
        for attr_name in has_traits_class.declarative_attrs:
            view_model_class_attr = getattr(view_model_class, attr_name)
            view_model_attr = getattr(view_model, attr_name)
            if isinstance(view_model_class_attr, DataTypeGidAttr) and view_model_attr:
                attr_value = self.load_with_references(view_model_attr)
            elif isinstance(view_model_class_attr, Attr) and isinstance(view_model_attr, ViewModel):
                attr_value = self.view_model_to_has_traits(view_model_attr)
            elif isinstance(view_model_class_attr, List) and len(view_model_attr) > 0 and isinstance(view_model_attr[0],
                                                                                                     ViewModel):
                attr_value = list()
                for view_model_elem in view_model_attr:
                    elem = self.view_model_to_has_traits(view_model_elem)
                    attr_value.append(elem)
            else:
                attr_value = view_model_attr
            setattr(has_traits, attr_name, attr_value)
        return has_traits

    @staticmethod
    def build_adapter_from_class(adapter_class):
        """
        Having a subclass of ABCAdapter, prepare an instance for launching an operation with it.
        """
        if not issubclass(adapter_class, ABCAdapter):
            raise IntrospectionException("Invalid data type: It should extend adapters.ABCAdapter!")
        try:
            stored_adapter = dao.get_algorithm_by_module(adapter_class.__module__, adapter_class.__name__)

            adapter_instance = adapter_class()
            adapter_instance.stored_adapter = stored_adapter
            return adapter_instance
        except Exception as excep:
            LOGGER.exception(excep)
            raise IntrospectionException(str(excep))

    @staticmethod
    def determine_adapter_class(stored_adapter):
        # type: (Algorithm) -> ABCAdapter
        """
        Determine the class of an adapter based on module and classname strings from stored_adapter
        :param stored_adapter: Algorithm or AlgorithmDTO type
        :return: a subclass of ABCAdapter
        """
        ad_module = importlib.import_module(stored_adapter.module)
        adapter_class = getattr(ad_module, stored_adapter.classname)
        return adapter_class

    @staticmethod
    def build_adapter(stored_adapter):
        # type: (Algorithm) -> ABCAdapter
        """
        Having a module and a class name, create an instance of ABCAdapter.
        """
        try:
            adapter_class = ABCAdapter.determine_adapter_class(stored_adapter)
            adapter_instance = adapter_class()
            adapter_instance.stored_adapter = stored_adapter
            return adapter_instance

        except Exception:
            msg = "Could not load Adapter Instance for Stored row %s" % stored_adapter
            LOGGER.exception(msg)
            raise IntrospectionException(msg)

    def load_view_model(self, operation):
        storage_path = self.storage_interface.get_project_folder(operation.project.name, str(operation.id))
        input_gid = operation.view_model_gid
        return h5.load_view_model(input_gid, storage_path)

    @staticmethod
    def array_size2kb(size):
        """
        :param size: size in bytes
        :return: size in kB
        """
        return size * TvbProfile.current.MAGIC_NUMBER / 8 / 2 ** 10

    @staticmethod
    def fill_index_from_h5(analyzer_index, analyzer_h5):
        """
        Method used only by analyzers that write slices of data.
        As they never have the whole array_data in memory, the metadata related to array_data (min, max, etc.) they
        store on the index is not correct, so we need to update them.
        """
        metadata = analyzer_h5.array_data.get_cached_metadata()

        if not metadata.has_complex:
            analyzer_index.array_data_max = float(metadata.max)
            analyzer_index.array_data_min = float(metadata.min)
            analyzer_index.array_data_mean = float(metadata.mean)

        analyzer_index.aray_has_complex = metadata.has_complex
        analyzer_index.array_is_finite = metadata.is_finite
        analyzer_index.shape = json.dumps(analyzer_h5.array_data.shape)
        analyzer_index.ndim = len(analyzer_h5.array_data.shape)

    def path_for(self, h5_file_class, gid, dt_class=None):
        project = dao.get_project_by_id(self.current_project_id)
        return h5.path_for(self.operation_id, h5_file_class, gid, project.name, dt_class)

    def store_complete(self, datatype, generic_attributes=GenericAttributes()):
        project = dao.get_project_by_id(self.current_project_id)
        return h5.store_complete(datatype, self.operation_id, project.name, generic_attributes)

    def get_storage_path(self):
        project = dao.get_project_by_id(self.current_project_id)
        return self.storage_interface.get_project_folder(project.name, str(self.operation_id))

