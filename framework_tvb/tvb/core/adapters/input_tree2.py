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
Preparation validation and manipulation of adapter input trees
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
from copy import copy
import json
import numpy

from tvb.basic.filters.chain import FilterChain
from tvb.basic.logger.builder import get_logger
from tvb.basic.traits.exceptions import TVBException
from tvb.basic.traits.parameters_factory import collapse_params
from tvb.basic.traits.types_mapped import MappedType
from tvb.core import utils
from tvb.core.adapters.exceptions import InvalidParameterException
from tvb.core.entities import model
from tvb.core.entities.load import get_class_by_name, load_entity_by_gid, get_filtered_datatypes
from tvb.core.entities.storage import dao
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.portlets.xml_reader import KEY_DYNAMIC
from tvb.core.utils import string2array
import tvb.basic.traits.itree_model as itr

ATT_METHOD = "python_method"
ATT_PARAMETERS = "parameters_prefix"

KEY_EQUATION = "equation"
KEY_FOCAL_POINTS = "focal_points"
KEY_SURFACE_GID = "surface_gid"

from tvb.core.adapters.constants import (
    ALL_TYPES as STATIC_ACCEPTED_TYPES,
    ATT_TYPE as KEY_TYPE,
    ELEM_OPTIONS as KEY_OPTIONS,
    ATT_ATTRIBUTES as KEY_ATTRIBUTES,
    ATT_NAME as KEY_NAME,
    ATT_DESCRIPTION as KEY_DESCRIPTION,
    ATT_VALUE as KEY_VALUE,
    ATT_LABEL as KEY_LABEL,
    ATT_REQUIRED as KEY_REQUIRED,
    ATT_MAXVALUE as KEY_MAXVALUE,
    ATT_MINVALUE as KEY_MINVALUE,
    ATT_FIELD as KEY_FIELD,

    TYPE_SELECT, TYPE_MULTIPLE, TYPE_DICT, TYPE_ARRAY, TYPE_LIST, TYPE_BOOL, TYPE_INT, TYPE_FLOAT,
    TYPE_STR, TYPE_UPLOAD)

KEY_DEFAULT = "default"
KEY_DATATYPE = 'datatype'
KEY_DTYPE = 'elementType'
KEY_DISABLED = "disabled"
KEY_ALL = "allValue"
KEY_CONDITION = "conditions"
KEY_FILTERABLE = "filterable"
KEY_ID = 'id'
KEY_UI_HIDE = "ui_hidden"

KEYWORD_PARAMS = "_parameters_"
KEYWORD_SEPARATOR = "_"
KEYWORD_OPTION = "option_"

KEY_PARAMETER_CHECKED = model.KEY_PARAMETER_CHECKED

MAXIMUM_DATA_TYPES_DISPLAYED = 50
KEY_WARNING = "warning"
WARNING_OVERFLOW = "Too many entities in storage; some of them were not returned, to avoid overcrowding. " \
                   "Use filters, to make the list small enough to fit in here!"


class InputTreeManager(object):

    def __init__(self):
        self.log = get_logger(self.__class__.__module__)


    def append_required_defaults(self, kwargs, algorithm_inputs):
        """
        Add if necessary any parameters marked as required that have a default value
        in the algorithm interface but were not submitted from the UI. For example in
        operations launched from context-menu or from data structure.
        """
        if algorithm_inputs is None:
            return

        for entry in algorithm_inputs:
            ## First handle this level of the tree, adding defaults where required
            if (entry[KEY_NAME] not in kwargs
                    and entry.get(KEY_REQUIRED)
                    and KEY_DEFAULT in entry
                    and entry[KEY_TYPE] != TYPE_DICT):
                kwargs[entry[KEY_NAME]] = entry[KEY_DEFAULT]

        for entry in algorithm_inputs:
            ## Now that first level was handled, go recursively on selected options only
            if entry.get(KEY_REQUIRED) and entry.get(KEY_OPTIONS) is not None:
                for option in entry[KEY_OPTIONS]:
                    # Only go recursive on option that was submitted
                    if option[KEY_VALUE] == kwargs[entry[KEY_NAME]] and KEY_ATTRIBUTES in option:
                        self.append_required_defaults(kwargs, option[KEY_ATTRIBUTES])


    def _validate_range_for_value_input(self, value, row):
        if KEY_MINVALUE in row and KEY_MAXVALUE in row:
            if value < row[KEY_MINVALUE] or value > row[KEY_MAXVALUE]:
                warning_message = "Field %s [%s] should be between %s and %s but provided value was %s." % (
                    row[KEY_LABEL], row[KEY_NAME], row[KEY_MINVALUE],
                    row[KEY_MAXVALUE], value)
                self.log.warning(warning_message)


    def _validate_range_for_array_input(self, array, row):
        if KEY_MINVALUE in row and KEY_MAXVALUE in row:
            min_val = numpy.min(array)
            max_val = numpy.max(array)

            if min_val < row[KEY_MINVALUE] or max_val > row[KEY_MAXVALUE]:
                # As described in TVB-1295, we do no longer raise exception, but only log a warning
                warning_message = "Field %s [%s] should have values between %s and %s but provided array contains min-" \
                                  "max:(%s, %s)." % (row[KEY_LABEL], row[KEY_NAME], row[KEY_MINVALUE],
                                                     row[KEY_MAXVALUE], min_val, max_val)
                self.log.warning(warning_message)


    @staticmethod
    def _get_dictionary(row, **kwargs):
        """
        Find all key/value pairs for the dictionary represented by name.
        """
        if InputTreeManager._is_parent_not_submitted(row, kwargs):
            return {}, []
        name = row[KEY_NAME]
        result_dict = {}
        taken_keys = []
        for key in kwargs:
            if name in key and name != key:
                taken_keys.append(key)
                if KEY_DTYPE in row:
                    if row[KEY_DTYPE] == 'array':
                        val = string2array(kwargs[key], " ", "float")
                    else:
                        val = eval(row[KEY_DTYPE])(kwargs[key])
                else:
                    val = str(kwargs[key])
                result_dict[key.split(KEYWORD_PARAMS[1:])[-1]] = val
        return result_dict, taken_keys


    def _find_field_submitted_name(self, submited_kwargs, flat_name, perform_clean=False):
        """
        Return key as in submitted dictionary for a given flat_name. Also remove from submitted_kwargs parameters like
        surface_parameters_option_DIFFERENT_GID_vertices.
        This won't work when DataType is in selectMultiple !!!!
        :param submited_kwargs: Flat dictionary with  keys in form surface_parameters_option_GID_vertices
        :param flat_name: Name as retrieved from self.flaten_input_interface
                         (in which we are not aware of existing entities in DB - options in select)
        :returns: key from 'submited_kwargs' which corresponds to 'flat_name'
        """
        if KEYWORD_PARAMS not in flat_name:
            if flat_name in submited_kwargs.keys():
                return flat_name
            else:
                return None
        parent_name, sufix = flat_name.split(KEYWORD_PARAMS, 1)
        prefix = parent_name + KEYWORD_PARAMS

        submitted_options = InputTreeManager._compute_submit_option_select(submited_kwargs[parent_name])

        datatype_like_submit = False

        for submitted_option in submitted_options:
            if sufix.startswith(KEYWORD_OPTION + str(submitted_option)):
                proposed_name = flat_name
            else:
                datatype_like_submit = True
                proposed_name = prefix + KEYWORD_OPTION + str(submitted_option)
                proposed_name = proposed_name + KEYWORD_SEPARATOR + sufix

            if perform_clean:
                ## Remove submitted parameters like surface_parameters_option_GID_vertices when surface != GID
                keys_to_remove = []
                for submit_key in submited_kwargs:
                    if (submit_key.startswith(prefix + KEYWORD_OPTION)
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
    def _is_parent_not_submitted(row, kwargs):
        """
        :returns: True when current attributes should not be considered, because parent option was not selected."""
        att_name = row[KEY_NAME]
        if KEYWORD_PARAMS in att_name:
            parent_name = att_name[: att_name.find(KEYWORD_PARAMS)]
            option = att_name[att_name.find(KEYWORD_OPTION) + len(KEYWORD_OPTION):]
            option = option[: option.find(KEYWORD_SEPARATOR)]
        else:
            return False

        submitted_option = InputTreeManager._compute_submit_option_select(kwargs[parent_name])
        if not submitted_option:
            return True
        if option in submitted_option:
            return False
        return True


    @staticmethod
    def _compute_submit_option_select(submitted_option):
        """ """
        if isinstance(submitted_option, basestring):
            submitted_option = submitted_option.replace('[', '').replace(']', '').split(',')
        return submitted_option


    @staticmethod
    def form_prefix(input_param, prefix=None, option_prefix=None):
        """Compute parameter prefix. We need to be able from the flatten
        submitted values in UI, to be able to re-compose the tree of parameters,
        and to make sure all submitted names are uniquely identified."""
        new_prefix = ""
        if prefix is not None and prefix != '':
            new_prefix = prefix
            if not new_prefix.endswith(KEYWORD_SEPARATOR):
                new_prefix += KEYWORD_SEPARATOR
        new_prefix += input_param + KEYWORD_PARAMS
        if option_prefix is not None:
            new_prefix += KEYWORD_OPTION + option_prefix + KEYWORD_SEPARATOR
        return new_prefix


    @staticmethod
    def fill_defaults(adapter_interface, data, fill_unselected_branches=False):
        """ Change the default values in the Input Interface Tree."""
        result = []
        for param in adapter_interface:
            # if param[ABCAdapter.KEY_NAME] == 'integrator':
            #     pass
            new_p = copy(param)
            if param[KEY_NAME] in data:
                new_p[KEY_DEFAULT] = data[param[KEY_NAME]]
            if param.get(KEY_ATTRIBUTES) is not None:
                new_p[KEY_ATTRIBUTES] = InputTreeManager.fill_defaults(param[KEY_ATTRIBUTES], data,
                                                                       fill_unselected_branches)
            if param.get(KEY_OPTIONS) is not None:
                new_options = param[KEY_OPTIONS]
                if param[KEY_NAME] in data or fill_unselected_branches:
                    selected_values = []
                    if param[KEY_NAME] in data:
                        if param[KEY_TYPE] == TYPE_MULTIPLE:
                            selected_values = data[param[KEY_NAME]]
                        else:
                            selected_values = [data[param[KEY_NAME]]]
                    for i, option in enumerate(new_options):
                        if option[KEY_VALUE] in selected_values or fill_unselected_branches:
                            new_options[i] = InputTreeManager.fill_defaults([option], data, fill_unselected_branches)[0]
                new_p[KEY_OPTIONS] = new_options
            result.append(new_p)
        return result


    def flatten(self, params_list, prefix=None):
        """ Internal method, to be used recursively, on parameters POST. """
        result = []
        for param in params_list:
            new_param = copy(param)
            new_param[KEY_ATTRIBUTES] = None
            new_param[KEY_OPTIONS] = None

            param_name = param[KEY_NAME]

            if prefix is not None and KEY_TYPE in param:
                new_param[KEY_NAME] = prefix + param_name
            result.append(new_param)

            if param.get(KEY_OPTIONS) is not None:
                for option in param[KEY_OPTIONS]:
                    ### SELECT or SELECT_MULTIPLE attributes
                    if option.get(KEY_ATTRIBUTES) is not None:
                        new_prefix = InputTreeManager.form_prefix(param_name, prefix, option[KEY_VALUE])
                        extra_list = self.flatten(option[KEY_ATTRIBUTES], new_prefix)
                        result.extend(extra_list)

            if param.get(KEY_ATTRIBUTES) is not None:
                ### DATATYPE attributes
                new_prefix = InputTreeManager.form_prefix(param_name, prefix, None)
                extra_list = self.flatten(param[KEY_ATTRIBUTES], new_prefix)
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
            new_name = param.name
            if prefix is not None: # and KEY_TYPE in param:
                new_name = prefix + param.name
                prepared_param.name = new_name

            if isinstance(param, itr.SelectTypeNode):
                new_prefix = InputTreeManager.form_prefix(param.name, prefix)
                prepared_param.options = InputTreeManager.prepare_param_names(param.options,
                                                                                   new_prefix, True)
            elif isinstance(param, itr.TypeNode):
                new_prefix = prefix
                if add_option_prefix:
                    new_prefix = prefix + KEYWORD_OPTION
                    new_prefix = new_prefix + param.value
                    new_prefix += KEYWORD_SEPARATOR
                prepared_param.attributes = InputTreeManager.prepare_param_names(param.attributes, new_prefix)
            elif isinstance(param, itr.DictNode):
                new_prefix = new_name + KEYWORD_PARAMS
                prepared_param.attributes = InputTreeManager.prepare_param_names(param.attributes, new_prefix)
            result.append(prepared_param)
        return result


    # -- Methods that may load entities from the db

    def review_operation_inputs(self, parameters, flat_interface):
        """
        Find out which of the submitted parameters are actually DataTypes and
        return a list holding all the dataTypes in parameters.
        :returns: list of dataTypes and changed parameters.
        """
        inputs_datatypes = []
        changed_parameters = dict()

        for field_dict in flat_interface:
            eq_flat_interface_name = self._find_field_submitted_name(parameters, field_dict[KEY_NAME])

            if eq_flat_interface_name is not None:
                is_datatype = False
                if field_dict.get(KEY_DATATYPE):
                    eq_datatype = load_entity_by_gid(parameters.get(str(eq_flat_interface_name)))
                    if eq_datatype is not None:
                        inputs_datatypes.append(eq_datatype)
                        is_datatype = True
                elif isinstance(field_dict[KEY_TYPE], basestring):
                    try:
                        class_entity = get_class_by_name(field_dict[KEY_TYPE])
                        if issubclass(class_entity, MappedType):
                            data_gid = parameters.get(str(field_dict[KEY_NAME]))
                            data_type = load_entity_by_gid(data_gid)
                            if data_type:
                                inputs_datatypes.append(data_type)
                                is_datatype = True
                    except ImportError:
                        pass

                if is_datatype:
                    changed_parameters[field_dict[KEY_LABEL]] = inputs_datatypes[-1].display_name
                else:
                    if field_dict[KEY_NAME] in parameters and (KEY_DEFAULT not in field_dict
                                    or str(field_dict[KEY_DEFAULT]) != str(parameters[field_dict[KEY_NAME]])):
                        changed_parameters[field_dict[KEY_LABEL]] = str(parameters[field_dict[KEY_NAME]])

        return inputs_datatypes, changed_parameters


    def _convert_to_array(self, input_data, row):
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
                equation_type = input_data.get(KEY_DTYPE)
                if equation_type is None:
                    self.log.warning("Cannot figure out type of equation from input dictionary: %s. "
                                     "Returning []." % input_data)
                    return []
                eq_class = get_class_by_name(equation_type)
                equation = eq_class.from_json(input_data[KEY_EQUATION])
                focal_points = json.loads(input_data[KEY_FOCAL_POINTS])
                surface_gid = input_data[KEY_SURFACE_GID]
                surface = load_entity_by_gid(surface_gid)
                return surface.compute_equation(focal_points, equation)
            except Exception:
                self.log.exception("The parameter %s was ignored. None value was returned.", row['name'])
                return None

        dtype = None
        if KEY_DTYPE in row:
            dtype = row[KEY_DTYPE]
        return string2array(str(input_data), ",", dtype)


    def _load_entity(self, row, datatype_gid, kwargs, metadata_out):
        """
        Load specific DataType entities, as specified in DATA_TYPE table.
        Check if the GID is for the correct DataType sub-class, otherwise throw an exception.
        Updates metadata_out with the metadata of this entity
        """

        entity = load_entity_by_gid(datatype_gid)
        if entity is None:
            ## Validate required DT one more time, after actual retrieval from DB:
            if row.get(KEY_REQUIRED):
                raise InvalidParameterException("Empty DataType value for required parameter %s [%s]" % (
                    row[KEY_LABEL], row[KEY_NAME]))

            return None

        expected_dt_class = row[KEY_TYPE]
        if isinstance(expected_dt_class, basestring):
            expected_dt_class = get_class_by_name(expected_dt_class)
        if not isinstance(entity, expected_dt_class):
            raise InvalidParameterException("Expected param %s [%s] of type %s but got type %s." % (
                row[KEY_LABEL], row[KEY_NAME], expected_dt_class.__name__, entity.__class__.__name__))

        result = entity

        ## Step 2 of updating Meta-data from parent DataType.
        if entity.fk_parent_burst:
            ## Link just towards the last Burst identified.
            metadata_out[DataTypeMetaData.KEY_BURST] = entity.fk_parent_burst

        if entity.user_tag_1 and DataTypeMetaData.KEY_TAG_1 not in metadata_out:
            metadata_out[DataTypeMetaData.KEY_TAG_1] = entity.user_tag_1

        current_subject = metadata_out[DataTypeMetaData.KEY_SUBJECT]
        if current_subject == DataTypeMetaData.DEFAULT_SUBJECT:
            metadata_out[DataTypeMetaData.KEY_SUBJECT] = entity.subject
        else:
            if entity.subject != current_subject and entity.subject not in current_subject.split(','):
                metadata_out[DataTypeMetaData.KEY_SUBJECT] = current_subject + ',' + entity.subject
        ##  End Step 2 - Meta-data Updates

        ## Validate current entity to be compliant with specified ROW filters.
        dt_filter = row.get(KEY_CONDITION)
        if dt_filter is not None and entity is not None and not dt_filter.get_python_filter_equivalent(entity):
            ## If a filter is declared, check that the submitted DataType is in compliance to it.
            raise InvalidParameterException("Field %s [%s] did not pass filters." % (row[KEY_LABEL],
                                                                                     row[KEY_NAME]))

        # In case a specific field in entity is to be used, use it
        if KEY_FIELD in row:
            # note: this cannot be replaced by getattr(entity, row[KEY_FIELD])
            # at least BCT has 'fields' like scaled_weights()
            result = eval('entity.' + row[KEY_FIELD])
        if ATT_METHOD in row:
            # The 'shape' attribute of an arraywrapper is overridden by us
            # the following check is made only to improve performance
            # (to find data in the dictionary with O(1)) on else the data is found in O(n)
            prefix = row[KEY_NAME] + "_" + row[ATT_PARAMETERS]
            if hasattr(entity, 'shape'):
                param_dict = {}
                for i in range(1, len(entity.shape)):
                    param_key = prefix + "_" + str(i - 1)
                    if param_key in kwargs:
                        param_dict[param_key] = kwargs[param_key]
            else:
                param_dict = dict((k, v) for k, v in kwargs.items() if k.startswith(prefix))
            result = getattr(entity, row[ATT_METHOD])(param_dict)
        return result


    def convert_ui_inputs(self, flat_input_interface, kwargs, metadata_out, validation_required=True):
        """
        Convert HTTP POST parameters into Python parameters.
        """
        kwa = {}
        simple_select_list, to_skip_dict_subargs = [], []
        for row in flat_input_interface:
            row_attr = row[KEY_NAME]
            row_type = row[KEY_TYPE]
            ## If required attribute was submitted empty no point to continue, so just raise exception
            if validation_required and row.get(KEY_REQUIRED) and kwargs.get(row_attr) == "":
                msg = "Parameter %s [%s] is required for %s but no value was submitted! Please relaunch with valid parameters."
                raise InvalidParameterException(msg % (row[KEY_LABEL], row[KEY_NAME], self.__class__.__name__))

            try:
                if row_type == TYPE_DICT:
                    kwa[row_attr], taken_keys = self._get_dictionary(row, **kwargs)
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
                    kwa_name = self._find_field_submitted_name(kwargs, row_attr, True)
                    if kwa_name is None:
                        ## Do not populate attributes not submitted
                        continue
                    kwargs[row_attr] = kwargs[kwa_name]
                    ## del kwargs[kwa_name] don't remove the original param, as it is useful for retrieving op.input DTs
                elif self._is_parent_not_submitted(row, kwargs):
                    ## Also do not populate sub-attributes from options not selected
                    del kwargs[row_attr]
                    continue

                if row_type == TYPE_ARRAY:
                    kwa[row_attr] = self._convert_to_array(kwargs[row_attr], row)
                    self._validate_range_for_array_input(kwa[row_attr], row)
                elif row_type == TYPE_LIST:
                    if not isinstance(kwargs[row_attr], list):
                        kwa[row_attr] = json.loads(kwargs[row_attr])
                elif row_type == TYPE_BOOL:
                    kwa[row_attr] = bool(kwargs[row_attr])
                elif row_type == TYPE_INT:
                    if kwargs[row_attr] in [None, '', 'None']:
                        kwa[row_attr] = None
                    else:
                        kwa[row_attr] = int(kwargs[row_attr])
                        self._validate_range_for_value_input(kwa[row_attr], row)
                elif row_type == TYPE_FLOAT:
                    if kwargs[row_attr] in ['', 'None']:
                        kwa[row_attr] = None
                    else:
                        kwa[row_attr] = float(kwargs[row_attr])
                        self._validate_range_for_value_input(kwa[row_attr], row)
                elif row_type == TYPE_STR:
                    kwa[row_attr] = kwargs[row_attr]
                elif row_type in [TYPE_SELECT, TYPE_MULTIPLE]:
                    val = kwargs[row_attr]
                    if row_type == TYPE_MULTIPLE and not isinstance(val, list):
                        val = [val]
                    kwa[row_attr] = val
                    if row_type == TYPE_SELECT:
                        simple_select_list.append(row_attr)
                elif row_type == TYPE_UPLOAD:
                    kwa[row_attr] = kwargs[row_attr]
                else:
                    ## DataType parameter to be processed:
                    simple_select_list.append(row_attr)
                    datatype_gid = kwargs[row_attr]
                    ## Load filtered and trimmed attribute (e.g. field is applied if specified):
                    kwa[row_attr] = self._load_entity(row, datatype_gid, kwargs, metadata_out)
                    if KEY_FIELD in row:
                        # Add entity_GID to the parameters to recognize original input
                        kwa[row_attr + '_gid'] = datatype_gid

            except TVBException:
                raise
            except Exception:
                self.log.exception('convert_ui_inputs failed')
                raise InvalidParameterException("Invalid or missing value in field %s [%s]" % (row[KEY_LABEL],
                                                                                               row[KEY_NAME]))

        return collapse_params(kwa, simple_select_list)


    @staticmethod
    def _populate_values(data_list, type_, category_key, complex_dt_attributes=None):
        """
        Populate meta-data fields for data_list (list of DataTypes).

        Private method, to be called recursively.
        It will receive a list of Attributes, and it will populate 'options'
        entry with data references from DB.
        """
        values = []
        all_field_values = []
        for id_, _, entity_gid, subject, completion_date, group, gr_name, tag1 in data_list:
            # Here we only populate with DB data, actual
            # XML check will be done after select and submit.
            actual_entity = dao.get_generic_entity(type_, entity_gid, "gid")
            display_name = ''
            if actual_entity is not None and len(actual_entity) > 0 and isinstance(actual_entity[0], model.DataType):
                display_name = actual_entity[0].display_name
            display_name += ' - ' + (subject or "None ")
            if group:
                display_name += ' - From: ' + str(group)
            else:
                display_name += utils.date2string(completion_date)
            if gr_name:
                display_name += ' - ' + str(gr_name)
            display_name += ' - ID:' + str(id_)
            all_field_values.append(str(entity_gid))
            values.append({KEY_NAME: display_name, KEY_VALUE: entity_gid})
            if complex_dt_attributes is not None:
                ### TODO apply filter on sub-attributes
                values[-1][KEY_ATTRIBUTES] = complex_dt_attributes  # this is the copy of complex dtype attributes on all db options
        if category_key is not None:
            category = dao.get_category_by_id(category_key)
            if not category.display and not category.rawinput and len(data_list) > 1:
                values.insert(0, {KEY_NAME: "All", KEY_VALUE: ','.join(all_field_values)})
        return values


    def populate_option_values_for_dtype(self, project_id, type_name, filter_condition=None,
                                         category_key=None, complex_dt_attributes=None):
        '''
        Converts all datatypes that match the project_id, type_name and filter_condition
        to a {name: , value:} dict used to populate options in the input tree ui
        '''
        data_type_cls = get_class_by_name(type_name)
        #todo: send category instead of category_key to avoid redundant queries
        #NOTE these functions are coupled via data_list, _populate_values makes no sense without _get_available_datatypes
        data_list, total_count = get_filtered_datatypes(project_id, data_type_cls,
                                                        filter_condition)
        values = self._populate_values(data_list, data_type_cls,
                                       category_key, complex_dt_attributes)
        return values, total_count


    def fill_input_tree_with_options(self, attributes_list, project_id, category_key):
        """
        For a datatype node in the input tree, load all instances from the db that fit the filters.
        """
        result = []
        for param in attributes_list:
            if getattr(param, KEY_UI_HIDE, False):
                continue
            transformed_param = copy(param)

            if isinstance(param, (itr.DatatypeNode, itr.ComplexDtypeNode)):
                filter_condition = param.conditions
                if filter_condition is None:
                    filter_condition = FilterChain('')
                filter_condition.add_condition(FilterChain.datatype + ".visible", "==", True)

                complex_dt_attributes = None
                if isinstance(param, itr.ComplexDtypeNode):
                    complex_dt_attributes = self.fill_input_tree_with_options(param.attributes,
                                                                    project_id, category_key)
                values, total_count = self.populate_option_values_for_dtype(project_id, param.type, filter_condition,
                                                                    category_key, complex_dt_attributes)
                if total_count > MAXIMUM_DATA_TYPES_DISPLAYED:
                    transformed_param.warning = WARNING_OVERFLOW

                if param.required and len(values) > 0 and param.default is None:
                    transformed_param.default = str(values[-1][KEY_VALUE])

                transformed_param.filterable = FilterChain.get_filters_for_type(param.type)
                transformed_param.type = TYPE_SELECT # todo this type transfer is not nice
                transformed_param.datatype = param.type
                # If Portlet dynamic parameter, don't add the options instead
                # just add the default value.
                if getattr(param, KEY_DYNAMIC, False):
                    dynamic_param = {KEY_NAME: param.default,
                                     KEY_VALUE: param.default}
                    transformed_param.options = [dynamic_param]
                else:
                    transformed_param.options = values

                ### DataType-attributes are no longer necessary, they were already copied on each OPTION
                transformed_param.attributes = [] # todo check if this is ok
            elif isinstance(param, itr.SelectTypeNode):
                transformed_param.options = self.fill_input_tree_with_options(param.options,
                                                                              project_id, category_key)
                if len(param.options) > 0 and param.default is None:
                    transformed_param.default = str(param.options[-1].value)
            elif isinstance(param, (itr.TypeNode, itr.DictNode)):  #ComplexDatatypeNode enters here!
                transformed_param.attributes = self.fill_input_tree_with_options(param.attributes,
                                                                                  project_id, category_key)

            result.append(transformed_param)
        return result


    @staticmethod
    def select_simulator_inputs(full_tree, selection_dictionary, prefix=''):
        """
        Cut Simulator input Tree, to display only user-checked inputs.

        :param full_tree: the simulator input tree
        :param selection_dictionary: a dictionary that keeps for each entry a default value and if it is check or not.
        :param prefix: a prefix to be added to the ui_name in case a select with subtrees is not selected

        """
        if full_tree is None:
            return None
        result = []
        for param in full_tree:
            param_name = param[KEY_NAME]
            if KEY_LABEL in param and len(prefix):
                param[KEY_LABEL] = prefix + '_' + param[KEY_LABEL]

            if param_name in selection_dictionary:
                selection_val = selection_dictionary[param_name][model.KEY_SAVED_VALUE]
                is_checked = selection_dictionary[param_name][KEY_PARAMETER_CHECKED]
            else:
                selection_val = None
                is_checked = False

            if is_checked:
                param[KEY_DEFAULT] = selection_val
                result.append(param)

            if param.get(KEY_OPTIONS) is not None:
                if is_checked:
                    for option in param[KEY_OPTIONS]:
                        if KEY_ATTRIBUTES in option:
                            option[KEY_ATTRIBUTES] = InputTreeManager.select_simulator_inputs(
                                                            option[KEY_ATTRIBUTES], selection_dictionary, prefix)
                            option[KEY_DEFAULT] = selection_val
                else:
                    ## Since entry is not selected, just recurse on the default option and ###
                    ## all it's subtree will come up one level in the input tree         #####
                    for option in param[KEY_OPTIONS]:
                        if (param_name in selection_dictionary and KEY_ATTRIBUTES in option and
                                    option[KEY_VALUE] == selection_val):
                            new_prefix = option[KEY_VALUE] + '_' + prefix
                            recursive_results = InputTreeManager.select_simulator_inputs(option[KEY_ATTRIBUTES],
                                                                             selection_dictionary, new_prefix)
                            result.extend(recursive_results)
        return result
