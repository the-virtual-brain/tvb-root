# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
Root classes for adding custom functionality to the code.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Yann Gordon <yann@tvb.invalid>
"""

import os
import json
import psutil
import numpy
import importlib
from functools import wraps
from datetime import datetime
from copy import copy
from abc import ABCMeta, abstractmethod
from tvb.basic.profile import TvbProfile
from tvb.basic.logger.builder import get_logger
from tvb.basic.traits.types_mapped import MappedType
from tvb.basic.traits.parameters_factory import collapse_params
from tvb.basic.traits.exceptions import TVBException
from tvb.core.utils import date2string, string2array, LESS_COMPLEX_TIME_FORMAT
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.file.files_update_manager import FilesUpdateManager
from tvb.core.entities.file.exceptions import FileVersioningException
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.adapters.exceptions import IntrospectionException, InvalidParameterException, LaunchException
from tvb.core.adapters.exceptions import NoMemoryAvailableException
from tvb.core.adapters.xml_reader import ELEM_OPTIONS, ELEM_OUTPUTS, INPUTS_KEY

import tvb.basic.traits.traited_interface as interface
import tvb.core.adapters.xml_reader as xml_reader

ATT_METHOD = "python_method"
ATT_PARAMETERS = "parameters_prefix"

KEY_EQUATION = "equation"
KEY_FOCAL_POINTS = "focal_points"
KEY_SURFACE_GID = "surface_gid"



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



class ABCAdapter(object):
    """
    Root Abstract class for all TVB Adapters. 
    """

    TYPE_SELECT = xml_reader.TYPE_SELECT
    TYPE_MULTIPLE = xml_reader.TYPE_MULTIPLE
    STATIC_ACCEPTED_TYPES = xml_reader.ALL_TYPES
    KEY_TYPE = xml_reader.ATT_TYPE
    KEY_OPTIONS = xml_reader.ELEM_OPTIONS
    KEY_ATTRIBUTES = xml_reader.ATT_ATTRIBUTES
    KEY_NAME = xml_reader.ATT_NAME
    KEY_DESCRIPTION = xml_reader.ATT_DESCRIPTION
    KEY_VALUE = xml_reader.ATT_VALUE
    KEY_LABEL = xml_reader.ATT_LABEL
    KEY_DEFAULT = "default"
    KEY_DATATYPE = 'datatype'
    KEY_DTYPE = 'elementType'
    KEY_DISABLED = "disabled"
    KEY_ALL = "allValue"
    KEY_CONDITION = "conditions"
    KEY_FILTERABLE = "filterable"
    KEY_REQUIRED = "required"
    KEY_ID = 'id'
    KEY_UI_HIDE = "ui_hidden"

    # TODO: move everything related to parameters PRE + POST into parameters_factory
    KEYWORD_PARAMS = "_parameters_"
    KEYWORD_SEPARATOR = "_"
    KEYWORD_OPTION = "option_"

    INTERFACE_ATTRIBUTES_ONLY = interface.INTERFACE_ATTRIBUTES_ONLY
    INTERFACE_ATTRIBUTES = interface.INTERFACE_ATTRIBUTES

    # Group that will be set for each adapter created by in build_adapter method
    algorithm_group = None

    _ui_display = 1

    __metaclass__ = ABCMeta


    def __init__(self):
        # It will be populate with key from DataTypeMetaData
        self.meta_data = {DataTypeMetaData.KEY_SUBJECT: DataTypeMetaData.DEFAULT_SUBJECT}
        self.file_handler = FilesHelper()
        self.storage_path = '.'
        # Will be populate with current running operation's identifier
        self.operation_id = None
        self.user_id = None
        self.log = get_logger(self.__class__.__module__)


    @abstractmethod
    def get_input_tree(self):
        """
        Describes inputs and outputs of the launch method.
        """


    @abstractmethod
    def get_output(self):
        """
        Describes inputs and outputs of the launch method.
        """


    def configure(self, **kwargs):
        """
        To be implemented in each Adapter that requires any specific configurations
        before the actual launch.
        """


    @abstractmethod
    def get_required_memory_size(self, **kwargs):
        """
        Abstract method to be implemented in each adapter. Should return the required memory
        for launching the adapter.
        """


    @abstractmethod
    def get_required_disk_size(self, **kwargs):
        """
        Abstract method to be implemented in each adapter. Should return the required memory
        for launching the adapter in kilo-Bytes.
        """


    def get_execution_time_approximation(self, **kwargs):
        """
        Method should approximate based on input arguments, the time it will take for the operation 
        to finish (in seconds).
        """
        return -1


    @abstractmethod
    def launch(self):
        """
         To be implemented in each Adapter.
         Will contain the logic of the Adapter.
         Any returned DataType will be stored in DB, by the Framework.
        """


    def add_operation_additional_info(self, message):
        """
        Adds additional info on the operation to be displayed in the UI. Usually a warning message.
        """
        current_op = dao.get_operation_by_id(self.operation_id)
        current_op.additional_info = message
        dao.store_entity(current_op)


    @nan_not_allowed()
    def _prelaunch(self, operation, uid=None, available_disk_space=0, **kwargs):
        """
        Method to wrap LAUNCH.
        Will prepare data, and store results on return. 
        """
        self.meta_data.update(json.loads(operation.meta_data))
        self.storage_path = self.file_handler.get_project_folder(operation.project, str(operation.id))
        self.operation_id = operation.id
        self.current_project_id = operation.project.id
        self.user_id = operation.fk_launched_by

        self.configure(**kwargs)

        # Compare the amount of memory the current algorithms states it needs,
        # with the average between the RAM available on the OS and the free memory at the current moment.
        # We do not consider only the free memory, because some OSs are freeing late and on-demand only.
        total_free_memory = psutil.virtual_memory().free + psutil.swap_memory().free
        total_existent_memory = psutil.virtual_memory().total + psutil.swap_memory().total
        memory_reference = (total_free_memory + total_existent_memory) / 2
        adapter_required_memory = self.get_required_memory_size(**kwargs)

        if adapter_required_memory > memory_reference:
            msg = "Machine does not have enough RAM memory for the operation (expected %.2g GB, but found %.2g GB)."
            raise NoMemoryAvailableException(msg % (adapter_required_memory / 2 ** 30, memory_reference / 2 ** 30))

        # Compare the expected size of the operation results with the HDD space currently available for the user
        # TVB defines a quota per user.
        required_disk_space = self.get_required_disk_size(**kwargs)
        if available_disk_space < 0:
            msg = "You have exceeded you HDD space quota by %.2f MB Stopping execution."
            raise NoMemoryAvailableException(msg % (- available_disk_space / 2 ** 10))
        if available_disk_space < required_disk_space:
            msg = ("You only have %.2f GB of disk space available but the operation you "
                   "launched might require %.2f Stopping execution...")
            raise NoMemoryAvailableException(msg % (available_disk_space / 2 ** 20, required_disk_space / 2 ** 20))

        operation.start_now()
        operation.estimated_disk_size = required_disk_space
        dao.store_entity(operation)

        result = self.launch(**kwargs)

        if not isinstance(result, (list, tuple)):
            result = [result, ]
        self.__check_integrity(result)

        return self._capture_operation_results(result, uid)


    def _capture_operation_results(self, result, user_tag=None):
        """
        After an operation was finished, make sure the results are stored
        in DB storage and the correct meta-data,IDs are set.
        """
        results_to_store = []
        data_type_group_id = None
        operation = dao.get_operation_by_id(self.operation_id)
        if operation.user_group is None or len(operation.user_group) == 0:
            operation.user_group = date2string(datetime.now(), date_format=LESS_COMPLEX_TIME_FORMAT)
            operation = dao.store_entity(operation)
        if self._is_group_launch():
            data_type_group_id = dao.get_datatypegroup_by_op_group_id(operation.fk_operation_group).id
        # All entities will have the same subject and state
        subject = self.meta_data[DataTypeMetaData.KEY_SUBJECT]
        state = self.meta_data[DataTypeMetaData.KEY_STATE]
        burst_reference = None
        if DataTypeMetaData.KEY_BURST in self.meta_data:
            burst_reference = self.meta_data[DataTypeMetaData.KEY_BURST]
        perpetuated_identifier = None
        if DataTypeMetaData.KEY_TAG_1 in self.meta_data:
            perpetuated_identifier = self.meta_data[DataTypeMetaData.KEY_TAG_1]

        for res in result:
            if res is None:
                continue
            res.subject = str(subject)
            res.state = state
            res.fk_parent_burst = burst_reference
            res.fk_from_operation = self.operation_id
            res.framework_metadata = self.meta_data
            if not res.user_tag_1:
                res.user_tag_1 = user_tag if user_tag is not None else perpetuated_identifier
            else:
                res.user_tag_2 = user_tag if user_tag is not None else perpetuated_identifier
            res.fk_datatype_group = data_type_group_id
            ## Compute size-on disk, in case file-storage is used
            if hasattr(res, 'storage_path') and hasattr(res, 'get_storage_file_name'):
                associated_file = os.path.join(res.storage_path, res.get_storage_file_name())
                res.close_file()
                res.disk_size = self.file_handler.compute_size_on_disk(associated_file)
            res = dao.store_entity(res)
            # Write metaData
            res.persist_full_metadata()
            results_to_store.append(res)
        del result[0:len(result)]
        result.extend(results_to_store)

        if len(result) and self._is_group_launch():
            ## Update the operation group name
            operation_group = dao.get_operationgroup_by_id(operation.fk_operation_group)
            operation_group.fill_operationgroup_name(result[0].type)
            dao.store_entity(operation_group)

        return 'Operation ' + str(self.operation_id) + ' has finished.', len(results_to_store)


    def __check_integrity(self, result):
        """
         Check that the returned parameters for LAUNCH operation
        are of the type specified in the adapter's interface.
        """
        entity_id = self.__module__ + '.' + self.__class__.__name__

        for result_entity in result:
            if type(result_entity) == list and len(result_entity) > 0:
                #### Determine the first element not None
                first_item = None
                for res in result_entity:
                    if res is not None:
                        first_item = res
                        break
                if first_item is None:
                    return
                    #### All list items are None
                #### Now check if the first item has a supported type
                if not self.__is_data_in_supported_types(first_item):
                    msg = "Unexpected DataType %s"
                    raise Exception(msg % type(first_item))

                first_item_type = type(first_item)
                for res in result_entity:
                    if not isinstance(res, first_item_type):
                        msg = '%s-Heterogeneous types (%s).Expected %s list.'
                        raise Exception(msg % (entity_id, type(res), first_item_type))
            else:
                if not self.__is_data_in_supported_types(result_entity):
                    msg = "Unexpected DataType %s"
                    raise Exception(msg % type(result_entity))


    def __is_data_in_supported_types(self, data):
        """
        This method checks if the provided data is one of the adapter supported return types 
        """
        if data is None:
            return True
        for supported_type in self.get_output():
            if isinstance(data, supported_type):
                return True
        ##### Data can't be mapped on any supported type !!
        return False


    def _is_group_launch(self):
        """
        Return true if this adapter is launched from a group of operations
        """
        operation = dao.get_operation_by_id(self.operation_id)
        return operation.fk_operation_group is not None


    @staticmethod
    def load_entity_by_gid(data_gid):
        """
        Load a generic DataType, specified by GID.
        """
        datatype = dao.get_datatype_by_gid(data_gid)
        if isinstance(datatype, MappedType):
            datatype_path = datatype.get_storage_file_path()
            files_update_manager = FilesUpdateManager()
            if not files_update_manager.is_file_up_to_date(datatype_path):
                datatype.invalid = True
                dao.store_entity(datatype)
                raise FileVersioningException("Encountered DataType with an incompatible storage or data version. "
                                              "The DataType was marked as invalid.")
        return datatype


    @staticmethod
    def prepare_adapter(adapter_class):
        """
        Having a subclass of ABCAdapter, prepare an instance for launching an operation with it.
        """
        try:
            if not issubclass(adapter_class, ABCAdapter):
                raise IntrospectionException("Invalid data type: It should extend adapters.ABCAdapter!")
            algo_group = dao.find_group(adapter_class.__module__, adapter_class.__name__)

            adapter_instance = adapter_class()
            adapter_instance.algorithm_group = algo_group
            return adapter_instance
        except Exception, excep:
            get_logger("ABCAdapter").exception(excep)
            raise IntrospectionException(str(excep))


    @staticmethod
    def build_adapter(algo_group):
        """
        Having a module and a class name, create an instance of ABCAdapter.
        """
        logger = get_logger("ABCAdapter")
        try:
            ad_module = importlib.import_module(algo_group.module)
            # This does no work for all adapters, so let it for manually choosing by developer
            if TvbProfile.env.IS_WORK_IN_PROGRESS:
                reload(ad_module)
                logger.info("Reloaded %r", ad_module)

            adapter = getattr(ad_module, algo_group.classname)

            if algo_group.init_parameter is not None and len(algo_group.init_parameter) > 0:
                adapter_instance = adapter(str(algo_group.init_parameter))
            else:
                adapter_instance = adapter()
            if not isinstance(adapter_instance, ABCAdapter):
                raise IntrospectionException("Invalid data type: It should extend adapters.ABCAdapter!")
            adapter_instance.algorithm_group = algo_group
            return adapter_instance
        except Exception, excep:
            logger.exception(excep)
            raise IntrospectionException(str(excep))


    ####### METHODS for PROCESSING PARAMETERS start here #############################

    def review_operation_inputs(self, parameters):
        """
        :returns: a list with the inputs from the parameters list that are instances of DataType,\
            and a dictionary with all parameters which are different than the declared defauts
        """
        flat_interface = self.flaten_input_interface()
        return self._review_operation_inputs(parameters, flat_interface)


    def _review_operation_inputs(self, parameters, flat_interface):
        """
        Find out which of the submitted parameters are actually DataTypes and 
        return a list holding all the dataTypes in parameters.
        :returns: list of dataTypes and changed parameters.
        """
        inputs_datatypes = []
        changed_parameters = dict()

        for field_dict in flat_interface:
            eq_flat_interface_name = self.__find_field_submitted_name(parameters, field_dict[self.KEY_NAME])

            if eq_flat_interface_name is not None:
                is_datatype = False
                if field_dict.get(self.KEY_DATATYPE):
                    eq_datatype = ABCAdapter.load_entity_by_gid(parameters.get(str(eq_flat_interface_name)))
                    if eq_datatype is not None:
                        inputs_datatypes.append(eq_datatype)
                        is_datatype = True
                elif type(field_dict[self.KEY_TYPE]) in (str, unicode):
                    point_separator = field_dict[self.KEY_TYPE].rfind('.')
                    if point_separator > 0:
                        module = field_dict[self.KEY_TYPE][:point_separator]
                        classname = field_dict[self.KEY_TYPE][(point_separator + 1):]
                        try:
                            module = __import__(module, [], locals(), globals())
                            class_entity = eval("module." + classname)
                            if issubclass(class_entity, MappedType):
                                data_gid = parameters.get(str(field_dict[self.KEY_NAME]))
                                data_type = ABCAdapter.load_entity_by_gid(data_gid)
                                if data_type:
                                    inputs_datatypes.append(data_type)
                                    is_datatype = True
                        except ImportError, _:
                            pass

                if is_datatype:
                    changed_parameters[field_dict[self.KEY_LABEL]] = inputs_datatypes[-1].display_name
                else:
                    if field_dict[self.KEY_NAME] in parameters and (self.KEY_DEFAULT not in field_dict
                                    or str(field_dict[self.KEY_DEFAULT]) != str(parameters[field_dict[self.KEY_NAME]])):
                        changed_parameters[field_dict[self.KEY_LABEL]] = str(parameters[field_dict[self.KEY_NAME]])

        return inputs_datatypes, changed_parameters


    def prepare_ui_inputs(self, kwargs, validation_required=True):
        """
        Prepare the inputs received from a HTTP Post in a form that will be
        used by the Python adapter.
        """
        algorithm_inputs = self.get_input_tree()
        algorithm_inputs = self.prepare_param_names(algorithm_inputs)
        self._append_required_defaults(kwargs, algorithm_inputs)
        return self.convert_ui_inputs(kwargs, validation_required=validation_required)


    def _append_required_defaults(self, kwargs, algorithm_inputs):
        """
        Add if necessary any parameters marked as required that have a default value
        in the algorithm interface but were not submitted from the UI. For example in 
        operations launched from context-menu or from data structure.
        """
        if algorithm_inputs is None:
            return

        for entry in algorithm_inputs:
            ## First handle this level of the tree, adding defaults where required
            if (entry[self.KEY_NAME] not in kwargs
                    and entry.get(self.KEY_REQUIRED) is True
                    and self.KEY_DEFAULT in entry
                    and entry[self.KEY_TYPE] != xml_reader.TYPE_DICT):
                kwargs[entry[self.KEY_NAME]] = entry[self.KEY_DEFAULT]

        for entry in algorithm_inputs:
            ## Now that first level was handled, go recursively on selected options only          
            if entry.get(self.KEY_REQUIRED) is True and entry.get(ABCAdapter.KEY_OPTIONS) is not None:
                for option in entry[ABCAdapter.KEY_OPTIONS]:
                    #Only go recursive on option that was submitted
                    if option[self.KEY_VALUE] == kwargs[entry[self.KEY_NAME]] and ABCAdapter.KEY_ATTRIBUTES in option:
                        self._append_required_defaults(kwargs, option[self.KEY_ATTRIBUTES])


    def convert_ui_inputs(self, kwargs, validation_required=True):
        """
        Convert HTTP POST parameters into Python parameters.
        """
        kwa = {}
        simple_select_list, to_skip_dict_subargs = [], []
        for row in self.flaten_input_interface():
            row_attr = row[xml_reader.ATT_NAME]
            row_type = row[xml_reader.ATT_TYPE]
            ## If required attribute was submitted empty no point to continue, so just raise exception
            if (validation_required and row.get(xml_reader.ATT_REQUIRED, False)
                    and row_attr in kwargs and kwargs[row_attr] == ""):
                msg = "Parameter %s [%s] is required for %s but no value was submitted! Please relaunch with valid parameters."
                raise InvalidParameterException(msg % (row[self.KEY_LABEL], row[self.KEY_NAME], self.__class__.__name__))

            try:
                if row_type == xml_reader.TYPE_DICT:
                    kwa[row_attr], taken_keys = self.__get_dictionary(row, **kwargs)
                    for key in taken_keys:
                        if key in kwa:
                            del kwa[key]
                        to_skip_dict_subargs.append(key)
                    continue
                ## Dictionary subargs that were previously processed should be ignored
                if row_attr in to_skip_dict_subargs:
                    continue

                if row_attr not in kwargs:
                    ## DataType sub-attributes are not submitted with GID in their name...
                    kwa_name = self.__find_field_submitted_name(kwargs, row_attr, True)
                    if kwa_name is None:
                        ## Do not populate attributes not submitted
                        continue
                    kwargs[row_attr] = kwargs[kwa_name]
                    ## del kwargs[kwa_name] don't remove the original param, as it is useful for retrieving op.input DTs
                elif self.__is_parent_not_submitted(row, kwargs):
                    ## Also do not populate sub-attributes from options not selected
                    del kwargs[row_attr]
                    continue

                if row_type == xml_reader.TYPE_ARRAY:
                    kwa[row_attr] = self.__convert_to_array(kwargs[row_attr], row)
                    if xml_reader.ATT_MINVALUE in row and xml_reader.ATT_MAXVALUE in row:
                        self.__validate_range_for_array_input(kwa[row_attr], row)
                elif row_type == xml_reader.TYPE_LIST:
                    if not isinstance(kwargs[row_attr], list):
                        kwa[row_attr] = json.loads(kwargs[row_attr])
                elif row_type == xml_reader.TYPE_BOOL:
                    kwa[row_attr] = bool(kwargs[row_attr])
                elif row_type == xml_reader.TYPE_INT:
                    if kwargs[row_attr] in [None, '', 'None']:
                        kwa[row_attr] = None
                    else:
                        kwa[row_attr] = int(kwargs[row_attr])
                        if xml_reader.ATT_MINVALUE in row and xml_reader.ATT_MAXVALUE in row:
                            self.__validate_range_for_value_input(kwa[row_attr], row)
                elif row_type == xml_reader.TYPE_FLOAT:
                    if kwargs[row_attr] in ['', 'None']:
                        kwa[row_attr] = None
                    else:
                        kwa[row_attr] = float(kwargs[row_attr])
                        if xml_reader.ATT_MINVALUE in row and xml_reader.ATT_MAXVALUE in row:
                            self.__validate_range_for_value_input(kwa[row_attr], row)
                elif row_type == xml_reader.TYPE_STR:
                    kwa[row_attr] = kwargs[row_attr]
                elif row_type in [xml_reader.TYPE_SELECT, xml_reader.TYPE_MULTIPLE]:
                    val = kwargs[row_attr]
                    if row_type == xml_reader.TYPE_MULTIPLE and not isinstance(val, list):
                        val = [val]
                    kwa[row_attr] = val
                    if row_type == xml_reader.TYPE_SELECT:
                        simple_select_list.append(row_attr)
                elif row_type == xml_reader.TYPE_UPLOAD:
                    kwa[row_attr] = kwargs[row_attr]
                else:
                    ## DataType parameter to be processed:
                    simple_select_list.append(row_attr)
                    datatype_gid = kwargs[row_attr]
                    ## Load filtered and trimmed attribute (e.g. field is applied if specified):
                    kwa[row_attr] = self.__load_entity(row, datatype_gid, kwargs)
                    if xml_reader.ATT_FIELD in row:
                        #Add entity_GID to the parameters to recognize original input
                        kwa[row_attr + '_gid'] = datatype_gid

            except TVBException:
                raise
            except Exception:
                raise InvalidParameterException("Invalid or missing value in field %s [%s]" % (row[self.KEY_LABEL],
                                                                                               row[self.KEY_NAME]))

        return collapse_params(kwa, simple_select_list)


    def __validate_range_for_value_input(self, value, row):
        if value < row[xml_reader.ATT_MINVALUE] or value > row[xml_reader.ATT_MAXVALUE]:
            warning_message = "Field %s [%s] should be between %s and %s but provided value was %s." % (
                row[self.KEY_LABEL], row[self.KEY_NAME], row[xml_reader.ATT_MINVALUE],
                row[xml_reader.ATT_MAXVALUE], value)
            self.log.warning(warning_message)


    def __validate_range_for_array_input(self, array, row):
        min_val = numpy.min(array)
        max_val = numpy.max(array)

        if min_val < row[xml_reader.ATT_MINVALUE] or max_val > row[xml_reader.ATT_MAXVALUE]:
            # As described in TVB-1295, we do no longer raise exception, but only log a warning
            warning_message = "Field %s [%s] should have values between %s and %s but provided array contains min-" \
                              "max:(%s, %s)." % (row[self.KEY_LABEL], row[self.KEY_NAME], row[xml_reader.ATT_MINVALUE],
                                                 row[xml_reader.ATT_MAXVALUE], min_val, max_val)
            self.log.warning(warning_message)


    def __convert_to_array(self, input_data, row):
        """
        Method used when the type of an input is array, to parse or read.

        If the user set an equation for computing a model parameter then the
        value of that parameter will be a dictionary which contains all the data
        needed for computing that parameter for each vertex from the used surface.
        """
        if KEY_EQUATION in str(input_data) and KEY_FOCAL_POINTS in str(input_data) and KEY_SURFACE_GID in str(input_data):
            try:
                input_data = eval(str(input_data))
                # TODO move at a different level
                equation_type = input_data.get(self.KEY_DTYPE, None)
                if equation_type is None:
                    self.log.warning("Cannot figure out type of equation from input dictionary: %s. "
                                     "Returning []." % (str(input_data, )))
                    return []
                splitted_class = equation_type.split('.')
                module = '.'.join(splitted_class[:-1])
                classname = splitted_class[-1]
                eq_module = __import__(module, globals(), locals(), [classname])
                eq_class = eval('eq_module.' + classname)
                equation = eq_class.from_json(input_data[KEY_EQUATION])
                focal_points = json.loads(input_data[KEY_FOCAL_POINTS])
                surface_gid = input_data[KEY_SURFACE_GID]
                surface = self.load_entity_by_gid(surface_gid)
                return surface.compute_equation(focal_points, equation)
            except Exception:
                self.log.exception("The parameter '" + str(row['name']) + "' was ignored. None value was returned.")
                return None

        if xml_reader.ATT_QUATIFIER in row:
            quantifier = row[xml_reader.ATT_QUATIFIER]
            dtype = None
            if self.KEY_DTYPE in row:
                dtype = row[self.KEY_DTYPE]
            if quantifier == xml_reader.QUANTIFIER_MANUAL:
                return string2array(str(input_data), ",", dtype)
            elif quantifier == xml_reader.QUANTIFIER_UPLOAD:
                input_str = open(input_data, 'r').read()
                return string2array(input_str, " ", dtype)
            elif quantifier == xml_reader.QUANTIFIER_FUNTION:
                return input_data

        return None


    def __get_dictionary(self, row, **kwargs):
        """
        Find all key/value pairs for the dictionary represented by name.
        """
        if self.__is_parent_not_submitted(row, kwargs):
            return {}, []
        name = row[xml_reader.ATT_NAME]
        result_dict = {}
        taken_keys = []
        for key in kwargs:
            if name in key and name != key:
                taken_keys.append(key)
                if self.KEY_DTYPE in row:
                    if row[self.KEY_DTYPE] == 'array':
                        val = string2array(kwargs[key], " ", "float")
                    else:
                        val = eval(row[self.KEY_DTYPE] + "('" + kwargs[key] + "')")
                else:
                    val = str(kwargs[key])
                result_dict[key.split(ABCAdapter.KEYWORD_PARAMS[1:])[-1]] = val
        return result_dict, taken_keys


    def __find_field_submitted_name(self, submited_kwargs, flat_name, perform_clean=False):
        """
        Return key as in submitted dictionary for a given flat_name. Also remove from submitted_kwargs parameters like
        surface_parameters_option_DIFFERENT_GID_vertices.
        This won't work when DataType is in selectMultiple !!!!
        :param submited_kwargs: Flat dictionary with  keys in form surface_parameters_option_GID_vertices
        :param flat_name: Name as retrieved from self.flaten_input_interface
                         (in which we are not aware of existing entities in DB - options in select)
        :returns: key from 'submited_kwargs' which corresponds to 'flat_name'
        """
        if ABCAdapter.KEYWORD_PARAMS not in flat_name:
            if flat_name in submited_kwargs.keys():
                return flat_name
            else:
                return None
        prefix = flat_name[0: (flat_name.find(ABCAdapter.KEYWORD_PARAMS) + 12)]
        sufix = flat_name[(flat_name.find(ABCAdapter.KEYWORD_PARAMS) + 12):]
        parent_name = flat_name[0: flat_name.find(ABCAdapter.KEYWORD_PARAMS)]
        submitted_options = ABCAdapter.__compute_submit_option_select(submited_kwargs[parent_name])

        datatype_like_submit = False

        for submitted_option in submitted_options:
            if sufix.startswith(ABCAdapter.KEYWORD_OPTION + str(submitted_option)):
                proposed_name = flat_name
            else:
                datatype_like_submit = True
                proposed_name = prefix + ABCAdapter.KEYWORD_OPTION + str(submitted_option)
                proposed_name = proposed_name + ABCAdapter.KEYWORD_SEPARATOR + sufix

            if perform_clean:
                ## Remove submitted parameters like surface_parameters_option_GID_vertices when surface != GID
                keys_to_remove = []
                for submit_key in submited_kwargs:
                    if (submit_key.startswith(prefix + ABCAdapter.KEYWORD_OPTION)
                            and submit_key.endswith(sufix) and submit_key != proposed_name):
                        keys_to_remove.append(submit_key)
                for submit_key in keys_to_remove:
                    del submited_kwargs[submit_key]
                if datatype_like_submit and len(submitted_options) > 1:
                    self.log.warning("DataType attribute in SELECT_MULTIPLE is not supposed to work!!!")
            if proposed_name in submited_kwargs:
                return proposed_name
        return None


    @staticmethod
    def __is_parent_not_submitted(row, kwargs):
        """
        :returns: True when current attributes should not be considered, because parent option was not selected."""
        att_name = row[xml_reader.ATT_NAME]
        parent_name, option = None, None
        if ABCAdapter.KEYWORD_PARAMS in att_name:
            parent_name = att_name[0: att_name.find(ABCAdapter.KEYWORD_PARAMS)]
            option = att_name[att_name.find(ABCAdapter.KEYWORD_OPTION) + 7:]
            option = option[: option.find(ABCAdapter.KEYWORD_SEPARATOR)]

        if parent_name is None or option is None:
            return False

        submitted_option = ABCAdapter.__compute_submit_option_select(kwargs[parent_name])
        if not submitted_option:
            return True
        if option in submitted_option:
            return False
        return True


    @staticmethod
    def __compute_submit_option_select(submitted_option):
        """ """
        if isinstance(submitted_option, (str, unicode)):
            submitted_option = submitted_option.replace('[', '').replace(']', '').split(',')
        return submitted_option


    def __load_entity(self, row, datatype_gid, kwargs):
        """
        Load specific DataType entities, as specified in DATA_TYPE table. 
        Check if the GID is for the correct DataType sub-class, otherwise throw an exception."""

        entity = self.load_entity_by_gid(datatype_gid)
        if entity is None:
            ## Validate required DT one more time, after actual retrieval from DB:
            if row.get(xml_reader.ATT_REQUIRED, False):
                raise InvalidParameterException("Empty DataType value for required parameter %s [%s]" % (
                    row[self.KEY_LABEL], row[self.KEY_NAME]))

            return None

        expected_dt_class = row[self.KEY_TYPE]
        if isinstance(expected_dt_class, (str, unicode)):
            classname = expected_dt_class.split('.')[-1]
            data_class = __import__(expected_dt_class.replace(classname, ''), globals(), locals(), [classname])
            data_class = eval("data_class." + classname)
            expected_dt_class = data_class
        if not isinstance(entity, expected_dt_class):
            raise InvalidParameterException("Expected param %s [%s] of type %s but got type %s." % (
                row[self.KEY_LABEL], row[self.KEY_NAME], expected_dt_class.__name__, entity.__class__.__name__))

        result = entity

        ## Step 2 of updating Meta-data from parent DataType.
        if entity.fk_parent_burst:
            ## Link just towards the last Burst identified.
            self.meta_data[DataTypeMetaData.KEY_BURST] = entity.fk_parent_burst

        if entity.user_tag_1 and DataTypeMetaData.KEY_TAG_1 not in self.meta_data:
            self.meta_data[DataTypeMetaData.KEY_TAG_1] = entity.user_tag_1

        current_subject = self.meta_data[DataTypeMetaData.KEY_SUBJECT]
        if current_subject == DataTypeMetaData.DEFAULT_SUBJECT:
            self.meta_data[DataTypeMetaData.KEY_SUBJECT] = entity.subject
        else:
            if entity.subject != current_subject and entity.subject not in current_subject.split(','):
                self.meta_data[DataTypeMetaData.KEY_SUBJECT] = current_subject + ',' + entity.subject
        ##  End Step 2 - Meta-data Updates

        ## Validate current entity to be compliant with specified ROW filters.
        dt_filter = row.get(xml_reader.ELEM_CONDITIONS, False)
        if (dt_filter is not None) and (dt_filter is not False) and \
                (entity is not None) and not dt_filter.get_python_filter_equivalent(entity):
            ## If a filter is declared, check that the submitted DataType is in compliance to it.
            raise InvalidParameterException("Field %s [%s] did not pass filters." % (row[self.KEY_LABEL],
                                                                                     row[self.KEY_NAME]))

        # In case a specific field in entity is to be used, use it
        if xml_reader.ATT_FIELD in row:
            val = eval("entity." + row[xml_reader.ATT_FIELD])
            result = val
        if ATT_METHOD in row:
            param_dict = dict()
            #The 'shape' attribute of an arraywrapper is overridden by us
            #the following check is made only to improve performance 
            # (to find data in the dictionary with O(1)) on else the data is found in O(n)
            if hasattr(entity, 'shape'):
                for i in xrange(len(entity.shape)):
                    if not i:
                        continue
                    param_key = (row[xml_reader.ATT_NAME] + "_" + row[ATT_PARAMETERS] + "_" + str(i - 1))
                    if param_key in kwargs:
                        param_dict[param_key] = kwargs[param_key]
            else:
                param_dict = dict((k, v) for k, v in kwargs.items()
                                  if k.startswith(row[xml_reader.ATT_NAME] + "_" + row[ATT_PARAMETERS]))
            val = eval("entity." + row[ATT_METHOD] + "(param_dict)")
            result = val
        return result

    def noise_configurable_parameters(self):
        return [entry[self.KEY_NAME] for entry in self.flaten_input_interface() if 'configurableNoise' in entry]


    def flaten_input_interface(self):
        """ Return a simple dictionary, instead of a Tree."""
        return self._flaten(self.get_input_tree())


    @staticmethod
    def form_prefix(input_param, prefix=None, option_prefix=None):
        """Compute parameter prefix. We need to be able from the flatten  
        submitted values in UI, to be able to re-compose the tree of parameters,
        and to make sure all submitted names are uniquely identified."""
        new_prefix = ""
        if prefix is not None and prefix != '':
            new_prefix = prefix
        if prefix is not None and prefix != '' and not new_prefix.endswith(ABCAdapter.KEYWORD_SEPARATOR):
            new_prefix += ABCAdapter.KEYWORD_SEPARATOR
        new_prefix += input_param + ABCAdapter.KEYWORD_PARAMS
        if option_prefix is not None:
            new_prefix += ABCAdapter.KEYWORD_OPTION + option_prefix + ABCAdapter.KEYWORD_SEPARATOR
        return new_prefix


    def key_parameters(self, parameters_for):
        """ Return the keyword expected for holding parameters 
            for argument 'parameters_for'."""
        return parameters_for + self.KEYWORD_PARAMS[0:11]


    @staticmethod
    def fill_defaults(adapter_interface, data, fill_unselected_branches=False):
        """ Change the default values in the Input Interface Tree."""
        result = []
        for param in adapter_interface:
            # if param[ABCAdapter.KEY_NAME] == 'integrator':
            #     pass
            new_p = copy(param)
            if param[ABCAdapter.KEY_NAME] in data:
                new_p[ABCAdapter.KEY_DEFAULT] = data[param[ABCAdapter.KEY_NAME]]
            if param.get(ABCAdapter.KEY_ATTRIBUTES) is not None:
                new_p[ABCAdapter.KEY_ATTRIBUTES] = ABCAdapter.fill_defaults(param[ABCAdapter.KEY_ATTRIBUTES], data,
                                                                            fill_unselected_branches)
            if param.get(ABCAdapter.KEY_OPTIONS) is not None:
                new_options = param[ABCAdapter.KEY_OPTIONS]
                if param[ABCAdapter.KEY_NAME] in data or fill_unselected_branches:
                    selected_values = []
                    if param[ABCAdapter.KEY_NAME] in data:
                        if param[ABCAdapter.KEY_TYPE] == ABCAdapter.TYPE_MULTIPLE:
                            selected_values = data[param[ABCAdapter.KEY_NAME]]
                        else:
                            selected_values = [data[param[ABCAdapter.KEY_NAME]]]
                    for i, option in enumerate(new_options):
                        if option[ABCAdapter.KEY_VALUE] in selected_values or fill_unselected_branches:
                            new_options[i] = ABCAdapter.fill_defaults([option], data, fill_unselected_branches)[0]
                new_p[ABCAdapter.KEY_OPTIONS] = new_options
            result.append(new_p)
        return result


    def _flaten(self, params_list, prefix=None):
        """ Internal method, to be used recursively, on parameters POST. """
        result = []
        for param in params_list:
            new_param = copy(param)
            new_param[self.KEY_ATTRIBUTES] = None
            new_param[self.KEY_OPTIONS] = None

            param_name = param[ABCAdapter.KEY_NAME]

            if prefix is not None and self.KEY_TYPE in param:
                new_param[self.KEY_NAME] = prefix + param_name
            result.append(new_param)

            if param.get(self.KEY_OPTIONS) is not None:
                for option in param[self.KEY_OPTIONS]:
                    ### SELECT or SELECT_MULTIPLE attributes
                    if option.get(self.KEY_ATTRIBUTES) is not None:
                        new_prefix = ABCAdapter.form_prefix(param_name, prefix, option[self.KEY_VALUE])
                        extra_list = self._flaten(option[self.KEY_ATTRIBUTES], new_prefix)
                        result.extend(extra_list)

            if param.get(self.KEY_ATTRIBUTES) is not None:
                ### DATATYPE attributes
                new_prefix = ABCAdapter.form_prefix(param_name, prefix, None)
                extra_list = self._flaten(param[self.KEY_ATTRIBUTES], new_prefix)
                result.extend(extra_list)
        return result


    @staticmethod
    def prepare_param_names(attributes_list, prefix=None, add_option_prefix=False):
        """
        For a given attribute list, change the name of the attributes where needed.
        Changes refer to adding a prefix, to identify groups.
        Will be used on parameters page GET.
        """
        result = []
        for param in attributes_list:
            prepared_param = copy(param)
            new_name = param[ABCAdapter.KEY_NAME]
            if prefix is not None and ABCAdapter.KEY_TYPE in param:
                new_name = prefix + param[ABCAdapter.KEY_NAME]
                prepared_param[ABCAdapter.KEY_NAME] = new_name

            if ((ABCAdapter.KEY_TYPE not in param or param[ABCAdapter.KEY_TYPE] in ABCAdapter.STATIC_ACCEPTED_TYPES)
                    and param.get(ABCAdapter.KEY_OPTIONS) is not None):
                add_prefix_option = param.get(ABCAdapter.KEY_TYPE) in [xml_reader.TYPE_MULTIPLE, xml_reader.TYPE_SELECT]
                new_prefix = ABCAdapter.form_prefix(param[ABCAdapter.KEY_NAME], prefix)
                prepared_param[ABCAdapter.KEY_OPTIONS] = ABCAdapter.prepare_param_names(param[ABCAdapter.KEY_OPTIONS],
                                                                                        new_prefix, add_prefix_option)

            if param.get(ABCAdapter.KEY_ATTRIBUTES) is not None:
                new_prefix = prefix
                is_dict = param.get(ABCAdapter.KEY_TYPE) == 'dict'
                if add_option_prefix:
                    new_prefix = prefix + ABCAdapter.KEYWORD_OPTION
                    new_prefix = new_prefix + param[ABCAdapter.KEY_VALUE]
                    new_prefix += ABCAdapter.KEYWORD_SEPARATOR
                if is_dict:
                    new_prefix = new_name + ABCAdapter.KEYWORD_PARAMS
                prepared_param[ABCAdapter.KEY_ATTRIBUTES] = ABCAdapter.prepare_param_names(
                                                        param[ABCAdapter.KEY_ATTRIBUTES], new_prefix)
            result.append(prepared_param)
        return result



class ABCGroupAdapter(ABCAdapter):
    """
    Still Abstract class.
    Acts as a notifier that a given adapter has a group of sub-algorithms.
    It is used for multiple simple methods interfaced in TVB through an XML description.
    """


    def __init__(self, xml_file_path):
        ABCAdapter.__init__(self)
        if not os.path.isabs(xml_file_path):
            xml_file_path = os.path.join(TvbProfile.current.web.CURRENT_DIR, xml_file_path)
        ### Find the XML reader (it loads only once in the system per XML file).
        self.xml_reader = xml_reader.XMLGroupReader.get_instance(xml_file_path)


    def get_input_tree(self):
        """ Overwrite empty method from super."""
        interface_result = []
        if self.algorithm_group is None:
            return interface_result
        tree_root = dict()
        tree_root[self.KEY_NAME] = self.xml_reader.get_group_name()
        tree_root[self.KEY_LABEL] = self.xml_reader.get_group_label()
        tree_root[self.KEY_REQUIRED] = True
        tree_root[self.KEY_TYPE] = self.TYPE_SELECT
        tree_root[ELEM_OPTIONS] = self._compute_options_for_group()
        interface_result.append(tree_root)
        return interface_result


    def _compute_options_for_group(self):
        """Sub-Algorithms"""
        result = []
        algorithms = self.xml_reader.get_algorithms_dictionary()
        for identifier in algorithms.keys():
            option = dict()
            option[self.KEY_VALUE] = identifier
            option[self.KEY_NAME] = algorithms[identifier][self.KEY_NAME]
            algorithm = dao.get_algorithm_by_group(self.algorithm_group.id, identifier)
            option[self.KEY_DESCRIPTION] = algorithm.description
            inputs = algorithms[identifier][INPUTS_KEY]
            option[self.KEY_ATTRIBUTES] = [inputs[key] for key in inputs.keys()]
            option[ELEM_OUTPUTS] = self.xml_reader.get_outputs(identifier)
            result.append(option)
        return result


    def get_input_for_algorithm(self, algorithm_identifier=None):
        """For a group, we will return input tree on algorithm base."""
        inputs = self.xml_reader.get_inputs(algorithm_identifier)
        prefix = ABCAdapter.form_prefix(self.get_algorithm_param(), option_prefix=algorithm_identifier)
        result = ABCAdapter.prepare_param_names(inputs, prefix)
        return result


    def get_output(self):
        """For a group, we will return outputs of all sub-algorithms."""
        real_outputs = []
        for output_description in self.xml_reader.get_all_outputs():
            full_type = output_description[xml_reader.ATT_TYPE]
            real_outputs.append(self._import_type(full_type))
        return real_outputs


    def get_output_for_algorithm(self, algorithm_identifier):
        """For this group, we will return input tree on algorithm base."""
        return self.xml_reader.get_outputs(algorithm_identifier)


    def get_algorithms_dictionary(self):
        """Return the list of sub-algorithms in current group"""
        return self.xml_reader.get_algorithms_dictionary()


    def get_algorithm_param(self):
        """
        This string, represents the argument name, 
        where the algorithms selection is submitted.
        """
        return self.xml_reader.root_name


    def get_call_code(self, algorithm_identifier):
        """From the XML interface, read the code for call method."""
        return self.xml_reader.get_code(algorithm_identifier)


    def get_matlab_file(self, algorithm_identifier):
        """From the XML interface read the name of the file that contains the code."""
        return self.xml_reader.get_matlab_file(algorithm_identifier)


    def get_import_code(self, algorithm_identifier):
        """From the XML interface, read the code for Python import. Optional"""
        return self.xml_reader.get_import(algorithm_identifier)


    def _import_type(self, full_type_string):
        """ Execute a dynamic import and return class reverence"""
        module = full_type_string[0: full_type_string.rfind(".")]
        class_name = full_type_string[full_type_string.rfind(".") + 1:]
        reference = __import__(module, globals(), locals(), [class_name])
        self.log.debug("Imported: " + reference.__name__)
        return eval("reference." + class_name)


    def build_result(self, algorithm, result, inputs):
        """
        Build an actual Python object, based on the XML interface description.
        Put inside the resulting Python object, the call result. 
        """
        final_result = []
        self.log.debug("Received results:" + str(result))
        self.log.debug("Received inputs:" + str(inputs))
        python_out_references = self.get_output_for_algorithm(algorithm)
        for output in python_out_references:
            # First prepare output attributes
            kwa = {}
            for field in output[xml_reader.ELEM_FIELD]:
                if xml_reader.ATT_VALUE in field:
                    kwa[field[xml_reader.ATT_NAME]] = field[xml_reader.ATT_VALUE]
                else:
                    expression = field[xml_reader.ATT_REFERENCE]
                    expression = expression.replace("$", 'result[')
                    expression = expression.replace("#", ']')
                    kwa[field[xml_reader.ATT_NAME]] = eval(expression)
            kwa["storage_path"] = self.storage_path
            # Import Output type and call constructor
            out_class = self._import_type(output[xml_reader.ATT_TYPE])
            self.log.warning("Executing INIT with parameters:" + str(kwa))
            final_result.append(out_class(**kwa))
        final_result.append(None)
        return final_result


    def get_algorithm_and_attributes(self, **kwargs):
        """ 
        Read selected Algorithm identifier, from input arguments.
        From the original full dictionary, split Algorithm name, 
        and actual algorithms arguments.
        """
        algorithm = kwargs[self.xml_reader.root_name]
        key_real_args = self.key_parameters(self.xml_reader.root_name)
        algorithm_arguments = {}
        if key_real_args in kwargs:
            algorithm_arguments = kwargs[key_real_args]
        return algorithm, algorithm_arguments


    def prepare_ui_inputs(self, kwargs, validation_required=True):
        """
        Overwrite the method from ABCAdapter to only append the required defaults for
        the selected subalgorithm.
        """
        algorithm_name = self.get_algorithm_param()
        algorithm_inputs = self.get_input_for_algorithm(kwargs[algorithm_name])
        self._append_required_defaults(kwargs, algorithm_inputs)
        return self.convert_ui_inputs(kwargs, validation_required=validation_required)


    def review_operation_inputs(self, parameters):
        """
        Returns a list with the inputs from the parameters list that are instances of DataType.
        """
        algorithm_name = parameters[self.get_algorithm_param()]
        flat_interface = self.get_input_for_algorithm(algorithm_name)
        return self._review_operation_inputs(parameters, flat_interface)



class ABCAsynchronous(ABCAdapter):
    """
      Abstract class, for marking adapters that are prone to be executed 
      on Cluster.
    """
    __metaclass__ = ABCMeta

    def array_size2kb(self, size):
        """
        :param size: size in bytes
        :return: size in kB
        """
        return size * TvbProfile.current.MAGIC_NUMBER / 8 / 2 ** 10



class ABCSynchronous(ABCAdapter):
    """
      Abstract class, for marking adapters that are prone to be NOT executed 
      on Cluster.
    """
    __metaclass__ = ABCMeta
    

