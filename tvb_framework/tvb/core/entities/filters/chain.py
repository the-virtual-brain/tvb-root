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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>

The following are the categories that will be available for filters in the UI. 
In case of adding a new category, the first input 'model.DataType.subject' is 
the field from the model that will be used , display represents the
way this field will be visible in the UI to the user, operations mark all the 
possible operations that can be done on that field 
(so far supported are: "==", "!=", "<", ">", "in", "not in"). 
Type represents the type of input that is expected. So far string, date and 
list are supported.

In order to define a default filter, from the adapter interface add::

    'conditions': FilterChain(fields= [FilterChain.datatype + ".subject", FilterChain.datatype + ".some_attribute",
                              values= [["John Doe", "JohnDoe1"], "Some Attr Value"], operations= ["in", "=="])
"""
import importlib
import json
from datetime import datetime

from tvb.basic.logger.builder import get_logger
from tvb.core.entities.filters.exceptions import *

KEY_DISPLAY = 'display_name'
KEY_FIELDS = 'fields'
KEY_VALUES = 'values'
KEY_OPERATIONS = 'operations'
KEY_OPERATOR = 'operator'

LOGGER = get_logger(__name__)


class FilterChain(object):
    """
    Class used to filter tables displayed in UI, based on few criteria.
    Initialize filter with a list of fields, and corresponding values.
    """

    datatype = '$$DATATYPE_INPUT$$'
    algorithm = '$$ALGORITHM_INPUT$$'
    algorithm_category = '$$ALGORITHM_CATEGORY_INPUT$$'
    operation = '$$OPERATION_INPUT$$'

    datatype_replacement = "DataType"
    algorithm_replacement = "Algorithm"
    algorithm_category_replacement = "AlgorithmCategory"
    operation_replacement = "Operation"

    def __init__(self, display_name="", fields=None, values=None, operations=None, operator_between_fields='and'):
        """
        Initialize filter attributes.
        """
        self.display_name = display_name
        self.fields = fields or []
        self.values = values or []
        self.operations = operations or []
        self.operator_between_fields = operator_between_fields

        # Is this filter selected or not in the UI
        self.selected = False
        # How many entries will still remain if we apply this filter
        self.passes_count = ''

    def __setattr__(self, name, value):
        """
        Overwrite so that the custom keys cannot be overwritten.
        """
        if name in ('datatype', 'algorithm', 'algorithm_category', 'operation'):
            raise AttributeError("FilterChain '%s' attribute is used internally and can only be read, not set." % name)
        else:
            object.__setattr__(self, name, value)

    def __add__(self, other):
        """
        Define an addition operation on filters so one can easily do filter1 + filter2
        """
        if other is None:
            return FilterChain(self.display_name, self.fields, self.values,
                               self.operations, self.operator_between_fields)
        if not isinstance(other, FilterChain):
            raise TypeError("unsupported operand type(s) for +: 'FilterChain' and '%s'" % type(other).__name__)

        new_display_name = self.display_name + "__" + other.display_name
        new_fields = self.fields[:]
        new_fields.extend(other.fields)
        new_values = self.values[:]
        new_values.extend(other.values)
        operations = self.operations[:]
        operations.extend(other.operations)
        return FilterChain(new_display_name, new_fields, new_values, operations, self.operator_between_fields)

    def __str__(self):
        return self.__class__.__name__ + "(fields=%s, operations=%s, values=%s, operator_between_fields=%s)" % (
            self.fields, self.operations, self.values, self.operator_between_fields)

    def to_json(self):
        """
        Return a dictionary representation of the filter, to be used when needed.
        """
        dict_equivalent = {KEY_DISPLAY: self.display_name,
                           KEY_FIELDS: self.fields,
                           KEY_VALUES: self.values,
                           KEY_OPERATIONS: self.operations,
                           KEY_OPERATOR: self.operator_between_fields}
        return json.dumps(dict_equivalent)

    def __prepare_filter_string(self, filter_string):
        """
        Do all replacements of place-holders with evaluable strings.
        """
        return filter_string.replace(self.datatype, self.datatype_replacement
                                     ).replace(self.algorithm, self.algorithm_replacement
                                               ).replace(self.algorithm_category, self.algorithm_category_replacement
                                                         ).replace(self.operation, self.operation_replacement)

    @classmethod
    def get_filters_for_type(cls, data_name):
        """
        Return the list of filters to appear in UI.
        """
        data_class = cls._get_class_instance(data_name)
        if data_class is None or not hasattr(data_class, 'accepted_filters'):
            LOGGER.warning("Invalid Class specification:" + str(data_name))
            return []
        else:
            return json.dumps(data_class.accepted_filters())

    @classmethod
    def _get_class_instance(cls, data_name):
        """
        Internal method, to build a DataType class instance, from possible class string-name.
        """
        data_class = data_name
        if isinstance(data_name, str):
            try:
                module_name, class_name = str(data_name).rsplit('.', 1)
                module = importlib.import_module(module_name)
                data_class = getattr(module, class_name)
            except Exception as excep:
                LOGGER.error("Expected DataType full class quantifier! Got:" + str(data_name))
                LOGGER.exception(excep)
                data_class = None
        return data_class

    @classmethod
    def from_json(cls, input_dict):
        """
        From a JSON dictionary create a filter instance.
        """
        if not input_dict or str(input_dict) == 'None':
            return None
        filter_dictionary = json.loads(input_dict)
        return FilterChain(display_name=filter_dictionary[KEY_DISPLAY], fields=filter_dictionary[KEY_FIELDS],
                           values=filter_dictionary[KEY_VALUES], operations=filter_dictionary[KEY_OPERATIONS],
                           operator_between_fields=filter_dictionary[KEY_OPERATOR])

    def get_python_filter_equivalent(self, datatype_to_check=None):
        """
        Python evaluate of the filter against a current given DataType
        Check a filter instance next to a given input.
        
        :param datatype_to_check: the dataType to be checked against this filter instance
        
        :return: true if input passed
                 false if input failed
        """
        passed_test = True
        # set to current method input parameters, as these strings will be use in eval
        self.datatype_replacement = 'datatype_to_check'
        self.algorithm_replacement = 'algorithm_to_check'
        self.algorithm_category_replacement = 'algocategory_to_check'
        self.operation_replacement = 'operation_to_check'

        for i in range(len(self.fields)):
            # Any filter validations checks start here #####
            if self.operations[i] in ('in', 'not in'):
                try:
                    iter(self.values[i])
                except TypeError:
                    raise InvalidFilterEntity("Invalid filter! %s. not applicable for value %s" % (self.operations[i],
                                                                                                   self.values[i]))

            # Any filter validations checks end here #####
            my_filter = self.__prepare_filter_string(self.fields[i]) + ' ' + self.operations[i] + " "
            if isinstance(self.values[i], str):
                prepared_value = self.__prepare_filter_string(self.values[i])
                if prepared_value != self.values[i]:
                    # It's not just some string, but a FilterChain expression.
                    my_filter += str(self.values[i])
                else:
                    # It's just a string, need to add quotes so eval won't search for a variable with that name.
                    my_filter += str('"' + self.values[i] + '"')
            else:
                # Try to replace here as well since we might have a list of place-holders.
                prepared_value = self.__prepare_filter_string(str(self.values[i]))
                my_filter = my_filter + prepared_value

            try:
                my_filter = eval(my_filter)
            except AttributeError:
                raise InvalidFilterChainInput("On %s filtered attribute %s is missing." % (datatype_to_check,
                                                                                           self.fields[i].replace(
                                                                                               'input_to_check.', '')))
            passed_test = eval('passed_test ' + self.operator_between_fields + ' my_filter')
        return passed_test

    def get_sql_filter_equivalent(self, datatype_to_check="DataType", operation_to_check="Operation",
                                  algorithm_to_check="Algorithm",
                                  algocategory_to_check="AlgorithmCategory"):
        """
        Returns the computed SQL string from the given filter.
        The method may return None if the filter is None or the
        fields of the filter are not set.
        
        The input to this method should be a string representing the name under which 
        the variable you want to be filtered was binded in the callers namespace.
        e.g.  

        def foo(var_to_filter, filter_entity):
            filter_entity.get_sql_filter_equivalent('var_to_filter')

        """

        if self.fields is None or len(self.fields) == 0:
            return None

        self.datatype_replacement = datatype_to_check
        self.algorithm_replacement = algorithm_to_check
        self.algorithm_category_replacement = algocategory_to_check
        self.operation_replacement = operation_to_check

        filter_str = self.operator_between_fields + "_("
        for i, field in enumerate(self.fields):
            if i > 0:
                filter_str += ","
            field = self.__prepare_filter_string(field)
            filter_str += self.__get_sql_filter_equivalent(field, self.operations[i], self.values[i])
        filter_str += ")"
        return filter_str

    def __get_sql_filter_equivalent(self, field, operation, value):
        """
        For a field, value and operation get the sql-alchemy specific syntax.
        """
        result = ""
        if operation in ("not in", "in"):
            prepared_value = self.__prepare_filter_string(str(value))
            if isinstance(value, str):
                if prepared_value == value:
                    # It was just a regular string, need to add quotes so it's not evaluated to a variable
                    prepared_value = '"' + prepared_value + '"'
                prepared_value = '[' + prepared_value + ']'
            if operation == "not in":
                result = result + "not_(" + field + ".in_(" + prepared_value + "))"
            elif operation == "in":
                result = result + field + ".in_(" + prepared_value + ")"
        elif operation == "like":
            result = result + field + ".ilike('"
            filter_value = self.__prepare_filter_string(str(value).lstrip().rstrip())
            if '%' not in filter_value:
                filter_value = '%' + filter_value + '%'
            result = result + filter_value + "')"
        else:
            result = result + field
            result = result + operation
            prepared_value = self.__prepare_filter_string(str(value))
            if isinstance(value, (str, datetime)):
                if prepared_value == str(value):
                    # It was just a regular string, need to add quotes so it's not evaluated to a variable
                    prepared_value = '"' + prepared_value + '"'
            result = result + prepared_value
        return result

    def add_condition(self, field_name, operation_string, value):
        """
        Append to the list of conditions.
        """
        self.fields.append(field_name)
        self.operations.append(operation_string)
        self.values.append(value)


class UIFilter(object):
    """
    Helper class for the UI filters.
    """

    def __init__(self, linked_elem_name, linked_elem_field,
                 linked_elem_parent_name=None, linked_elem_parent_option=None):
        self.linked_elem_name = linked_elem_name
        self.linked_elem_field = linked_elem_field
        self.linked_elem_parent_name = linked_elem_parent_name
        self.linked_elem_parent_option = linked_elem_parent_option

    def to_dict(self):
        """
        Prepare for passing in UI
        """
        return {'linked_elem_name': self.linked_elem_name,
                'linked_elem_field': self.linked_elem_field,
                'linked_elem_parent_name': self.linked_elem_parent_name,
                'linked_elem_parent_option': self.linked_elem_parent_option}
