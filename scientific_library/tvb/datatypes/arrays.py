# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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

The Array datatypes. This brings together the scientific and framework 
methods that are associated with the Array datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import numpy
from tvb.basic.exceptions import ValidationException

#TODO: tvb-framework uses MappedArray().parse_selected_items(ui_sel_items) in gettemplatefordimensionselect
from tvb.basic.neotraits.api import HasTraits, Attr, NArray


class MappedArray(HasTraits):
    "An array stored in the database."
    KEY_SIZE = "size"
    KEY_OPERATION = "operation"

    title = Attr(field_type=str)
    label_x, label_y = Attr(field_type=str), Attr(field_type=str)
    aggregation_functions = Attr(dict, required=False)
    dimensions_labels = Attr(dict, required=False)
    nr_dimensions, length_1d, length_2d, length_3d, length_4d = [Attr(field_type=int)] * 5
    array_data = NArray()

    __generate_table__ = True

    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = {"Title:": self.title,
                   "Dimensions:": self.dimensions_labels}
        summary.update(self.get_info_about_array('array_data'))
        return summary

    @property
    def display_name(self):
        """
        Overwrite from superclass and add title field
        """
        previous = super(MappedArray, self).display_name
        if previous is None:
            return str(self.title)
        return str(self.title) + " " + previous

    @property
    def shape(self):
        """
        Shape for current wrapped NumPy array.
        """
        return self.array_data.shape

    def configure_chunk_safe(self):
        """ Configure part which is chunk safe"""
        data_shape = self.get_data_shape('array_data')
        self.nr_dimensions = len(data_shape)
        for i in range(min(self.nr_dimensions, 4)):
            setattr(self, 'length_%dd' % (i + 1), int(data_shape[i]))

    def configure(self):
        """After populating few fields, compute the rest of the fields"""
        super(MappedArray, self).configure()
        if not isinstance(self.array_data, numpy.ndarray):
            return
        self.nr_dimensions = len(self.array_data.shape)
        for i in range(min(self.nr_dimensions, 4)):
            setattr(self, 'length_%dd' % (i + 1), self.array_data.shape[i])

    @staticmethod
    def accepted_filters():
        # filters = MappedType.accepted_filters()
        filters = {'datatype_class._nr_dimensions': {'type': 'int', 'display': 'Dimensionality',
                                                          'operations': ['==', '<', '>']},
                        'datatype_class._length_1d': {'type': 'int', 'display': 'Shape 1',
                                                      'operations': ['==', '<', '>']},
                        'datatype_class._length_2d': {'type': 'int', 'display': 'Shape 2',
                                                      'operations': ['==', '<', '>']}}
        return filters

    def reduce_dimension(self, ui_selected_items):
        """
        ui_selected_items is a dictionary which contains items of form:
        '$name_$D : [$gid_$D_$I,..., func_$FUNC]' where '$D - int dimension', '$gid - a data type gid',
        '$I - index in that dimension' and '$FUNC - an aggregation function'.

        If the user didn't select anything from a certain dimension then it means that the entire
        dimension is selected
        """

        # The fields 'aggregation_functions' and 'dimensions' will be of form:
        # - aggregation_functions = {dimension: agg_func, ...} e.g.: {0: sum, 1: average, 2: none, ...}
        # - dimensions = {dimension: [list_of_indexes], ...} e.g.: {0: [0,1], 1: [5,500],...}
        dimensions, aggregation_functions, required_dimension, shape_restrictions = \
            self.parse_selected_items(ui_selected_items)

        if required_dimension is not None:
            # find the dimension of the resulted array
            dim = len(self.shape)
            for key in aggregation_functions.keys():
                if aggregation_functions[key] != "none":
                    dim -= 1
            for key in dimensions.keys():
                if (len(dimensions[key]) == 1 and
                        (key not in aggregation_functions or aggregation_functions[key] == "none")):
                    dim -= 1
            if dim != required_dimension:
                self.logger.debug("Dimension for selected array is incorrect")
                raise ValidationException("Dimension for selected array is incorrect!")

        result = self.array_data
        full = slice(0, None)
        cut_dimensions = 0
        for i in range(len(self.shape)):
            if i in dimensions.keys():
                my_slice = [full for _ in range(i - cut_dimensions)]
                if len(dimensions[i]) == 1:
                    my_slice.extend(dimensions[i])
                    cut_dimensions += 1
                else:
                    my_slice.append(dimensions[i])
                result = result[tuple(my_slice)]
            if i in aggregation_functions.keys():
                if aggregation_functions[i] != "none":
                    result = eval("numpy." + aggregation_functions[i] + "(result,axis=" + str(i - cut_dimensions) + ")")
                    cut_dimensions += 1

        # check that the shape for the resulted array respects given conditions
        result_shape = result.shape
        for i in range(len(result_shape)):
            if i in shape_restrictions:
                flag = eval(str(result_shape[i]) + shape_restrictions[i][self.KEY_OPERATION] +
                            str(shape_restrictions[i][self.KEY_SIZE]))
                if not flag:
                    msg = ("The condition is not fulfilled: dimension "
                           + str(i + 1) + " "
                           + shape_restrictions[i][self.KEY_OPERATION] + " "
                           + str(shape_restrictions[i][self.KEY_SIZE]) +
                           ". The actual size of dimension " + str(i + 1)
                           + " is " + str(result_shape[i]) + ".")
                    self.logger.debug(msg)
                    raise ValidationException(msg)

        if required_dimension is not None and 1 <= required_dimension != len(result.shape):
            self.logger.debug("Dimensions of the selected array are incorrect")
            raise ValidationException("Dimensions of the selected array are incorrect!")

        return result

    def parse_selected_items(self, ui_selected_items):
        """
        Used for parsing the user selected items.

        ui_selected_items is a dictionary which contains items of form:
        'name_D : [gid_D_I,..., func_FUNC]' where 'D - dimension', 'gid - a data type gid',
        'I - index in that dimension' and 'FUNC - an aggregation function'.
        """
        expected_shape_str = ''
        operations_str = ''
        dimensions = dict()
        aggregation_functions = dict()
        required_dimension = None
        for key in ui_selected_items.keys():
            split_array = str(key).split("_")
            current_dim = split_array[len(split_array) - 1]
            list_values = ui_selected_items[key]
            if list_values is None or len(list_values) == 0:
                list_values = []
            elif not isinstance(list_values, list):
                list_values = [list_values]
            for item in list_values:
                if str(item).startswith("expected_shape_"):
                    expected_shape_str = str(item).split("expected_shape_")[1]
                elif str(item).startswith("operations_"):
                    operations_str = str(item).split("operations_")[1]
                elif str(item).startswith("requiredDim_"):
                    required_dimension = int(str(item).split("requiredDim_")[1])
                elif str(item).startswith("func_"):
                    agg_func = str(item).split("func_")[1]
                    aggregation_functions[int(current_dim)] = agg_func
                else:
                    str_array = str(item).split("_")
                    if int(str_array[1]) in dimensions:
                        dimensions[int(str_array[1])].append(int(str_array[2]))
                    else:
                        dimensions[int(str_array[1])] = [int(str_array[2])]
        return dimensions, aggregation_functions, required_dimension, self._parse_expected_shape(expected_shape_str,
                                                                                                 operations_str)

    def _parse_expected_shape(self, expected_shape_str='', operations_str=''):
        """
        If we have the inputs of form: expected_shape='x,512,x' and operations='x,&lt;,x'.
        The result will be: {1: {'size':512, 'operation':'<'}}
        """
        result = {}
        if len(expected_shape_str.strip()) == 0 or len(operations_str.strip()) == 0:
            return result

        shape_array = str(expected_shape_str).split(",")
        op_array = str(operations_str).split(",")

        operations = self._get_operations()
        for i in range(len(shape_array)):
            if str(shape_array[i]).isdigit() and i < len(op_array) and op_array[i] in operations:
                result[i] = {self.KEY_SIZE: int(shape_array[i]),
                             self.KEY_OPERATION: operations[op_array[i]]}
        return result

    def read_data_shape(self):
        """ Expose shape read on field 'array_data' """
        return self.get_data_shape('array_data')

    def read_data_slice(self, data_slice):
        """ Expose chunked-data access. """
        return self.get_data('array_data', data_slice)

    @staticmethod
    def _get_operations():
        """Return accepted operations"""
        operations = {'&lt;': '<',
                      '&gt;': '>',
                      '&ge;': '>=',
                      '&le;': '<=',
                      '==': '=='}
        return operations
