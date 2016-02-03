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
Preparation validation and manipulation of adapter input trees
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
from copy import copy
import numpy

from tvb.basic.logger.builder import get_logger
from tvb.core.adapters import xml_reader
from tvb.core.utils import string2array

KEY_EQUATION = "equation"
KEY_FOCAL_POINTS = "focal_points"
KEY_SURFACE_GID = "surface_gid"

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

KEYWORD_PARAMS = "_parameters_"
KEYWORD_SEPARATOR = "_"
KEYWORD_OPTION = "option_"


class InputTreeManager(object):

    def __init__(self):
        self.log = get_logger(self.__class__.__module__)


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
            if (entry[KEY_NAME] not in kwargs
                    and entry.get(KEY_REQUIRED) is True
                    and KEY_DEFAULT in entry
                    and entry[KEY_TYPE] != xml_reader.TYPE_DICT):
                kwargs[entry[KEY_NAME]] = entry[KEY_DEFAULT]

        for entry in algorithm_inputs:
            ## Now that first level was handled, go recursively on selected options only
            if entry.get(KEY_REQUIRED) is True and entry.get(KEY_OPTIONS) is not None:
                for option in entry[KEY_OPTIONS]:
                    #Only go recursive on option that was submitted
                    if option[KEY_VALUE] == kwargs[entry[KEY_NAME]] and KEY_ATTRIBUTES in option:
                        self._append_required_defaults(kwargs, option[KEY_ATTRIBUTES])


    def _validate_range_for_value_input(self, value, row):
        if value < row[xml_reader.ATT_MINVALUE] or value > row[xml_reader.ATT_MAXVALUE]:
            warning_message = "Field %s [%s] should be between %s and %s but provided value was %s." % (
                row[KEY_LABEL], row[KEY_NAME], row[xml_reader.ATT_MINVALUE],
                row[xml_reader.ATT_MAXVALUE], value)
            self.log.warning(warning_message)


    def _validate_range_for_array_input(self, array, row):
        min_val = numpy.min(array)
        max_val = numpy.max(array)

        if min_val < row[xml_reader.ATT_MINVALUE] or max_val > row[xml_reader.ATT_MAXVALUE]:
            # As described in TVB-1295, we do no longer raise exception, but only log a warning
            warning_message = "Field %s [%s] should have values between %s and %s but provided array contains min-" \
                              "max:(%s, %s)." % (row[KEY_LABEL], row[KEY_NAME], row[xml_reader.ATT_MINVALUE],
                                                 row[xml_reader.ATT_MAXVALUE], min_val, max_val)
            self.log.warning(warning_message)


    @staticmethod
    def _get_dictionary(row, **kwargs):
        """
        Find all key/value pairs for the dictionary represented by name.
        """
        if InputTreeManager._is_parent_not_submitted(row, kwargs):
            return {}, []
        name = row[xml_reader.ATT_NAME]
        result_dict = {}
        taken_keys = []
        for key in kwargs:
            if name in key and name != key:
                taken_keys.append(key)
                if KEY_DTYPE in row:
                    if row[KEY_DTYPE] == 'array':
                        val = string2array(kwargs[key], " ", "float")
                    else:
                        val = eval(row[KEY_DTYPE] + "('" + kwargs[key] + "')")
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
        prefix = flat_name[0: (flat_name.find(KEYWORD_PARAMS) + 12)]
        sufix = flat_name[(flat_name.find(KEYWORD_PARAMS) + 12):]
        parent_name = flat_name[0: flat_name.find(KEYWORD_PARAMS)]
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
        att_name = row[xml_reader.ATT_NAME]
        parent_name, option = None, None
        if KEYWORD_PARAMS in att_name:
            parent_name = att_name[0: att_name.find(KEYWORD_PARAMS)]
            option = att_name[att_name.find(KEYWORD_OPTION) + 7:]
            option = option[: option.find(KEYWORD_SEPARATOR)]

        if parent_name is None or option is None:
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
        if isinstance(submitted_option, (str, unicode)):
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
        if prefix is not None and prefix != '' and not new_prefix.endswith(KEYWORD_SEPARATOR):
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


    def _flaten(self, params_list, prefix=None):
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
                        extra_list = self._flaten(option[KEY_ATTRIBUTES], new_prefix)
                        result.extend(extra_list)

            if param.get(KEY_ATTRIBUTES) is not None:
                ### DATATYPE attributes
                new_prefix = InputTreeManager.form_prefix(param_name, prefix, None)
                extra_list = self._flaten(param[KEY_ATTRIBUTES], new_prefix)
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
            new_name = param[KEY_NAME]
            if prefix is not None and KEY_TYPE in param:
                new_name = prefix + param[KEY_NAME]
                prepared_param[KEY_NAME] = new_name

            if ((KEY_TYPE not in param or param[KEY_TYPE] in STATIC_ACCEPTED_TYPES)
                    and param.get(KEY_OPTIONS) is not None):
                add_prefix_option = param.get(KEY_TYPE) in [xml_reader.TYPE_MULTIPLE, xml_reader.TYPE_SELECT]
                new_prefix = InputTreeManager.form_prefix(param[KEY_NAME], prefix)
                prepared_param[KEY_OPTIONS] = InputTreeManager.prepare_param_names(param[KEY_OPTIONS],
                                                                                        new_prefix, add_prefix_option)

            if param.get(KEY_ATTRIBUTES) is not None:
                new_prefix = prefix
                is_dict = param.get(KEY_TYPE) == 'dict'
                if add_option_prefix:
                    new_prefix = prefix + KEYWORD_OPTION
                    new_prefix = new_prefix + param[KEY_VALUE]
                    new_prefix += KEYWORD_SEPARATOR
                if is_dict:
                    new_prefix = new_name + KEYWORD_PARAMS
                prepared_param[KEY_ATTRIBUTES] = InputTreeManager.prepare_param_names(
                                                        param[KEY_ATTRIBUTES], new_prefix)
            result.append(prepared_param)
        return result

