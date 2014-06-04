# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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

Framework methods for the Array datatypes.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""
import numpy
import tvb.datatypes.arrays_data as arrays_data
from tvb.basic.traits.types_mapped import Array
from tvb.basic.traits.types_mapped import MappedType


class BaseArrayFramework(Array):
    """Basic class for non-mapped arrays."""
    pass


class FloatArrayFramework(arrays_data.FloatArrayData, BaseArrayFramework):
    """ This class exists to add framework methods to FloatArrayData """
    pass


class IntegerArrayFramework(arrays_data.IntegerArrayData, BaseArrayFramework):
    """ This class exists to add framework methods to IntegerArrayData """
    pass


class ComplexArrayFramework(arrays_data.ComplexArrayData, BaseArrayFramework):
    """ This class exists to add framework methods to ComplexArrayData """

    stored_metadata = [key for key in MappedType.DEFAULT_STORED_ARRAY_METADATA if key != MappedType.METADATA_ARRAY_VAR]


class BoolArrayFramework(arrays_data.BoolArrayData, BaseArrayFramework):
    """ This class exists to add framework methods to BoolArrayData """

    stored_metadata = [MappedType.METADATA_ARRAY_SHAPE]


class StringArrayFramework(arrays_data.StringArrayData, BaseArrayFramework):
    """ This class exists to add framework methods to StringArrayData """

    stored_metadata = [MappedType.METADATA_ARRAY_SHAPE]


class PositionArrayFramework(arrays_data.PositionArrayData,
                             FloatArrayFramework):
    """ This class exists to add framework methods to PositionArrayData """
    pass


class OrientationArrayFramework(arrays_data.OrientationArrayData,
                                FloatArrayFramework):
    """ This class exists to add framework methods to OrientationArrayData """
    pass


class IndexArrayFramework(arrays_data.IndexArrayData, IntegerArrayFramework):
    """ This class exists to add framework methods to IndexArrayData """
    pass


class MappedArrayFramework(arrays_data.MappedArrayData):
    """ This class will hold methods that should be available to all 
    array types"""

    KEY_SIZE = "size"
    KEY_OPERATION = "operation"
    __tablename__ = None
    
    
    def configure_chunk_safe(self):
        """ Configure part which is chunk safe"""
        data_shape = self.get_data_shape('array_data')
        self.nr_dimensions = len(data_shape)
        for i in range(min(self.nr_dimensions, 4)): 
            setattr(self, 'length_%dd' % (i + 1), int(data_shape[i]))
    
    
    def configure(self):
        """After populating few fields, compute the rest of the fields"""
        super(MappedArrayFramework, self).configure()
        if not isinstance(self.array_data, numpy.ndarray):
            return
        self.nr_dimensions = len(self.array_data.shape)
        for i in range(min(self.nr_dimensions, 4)): 
            setattr(self, 'length_%dd' % (i + 1), self.array_data.shape[i])


    @staticmethod
    def accepted_filters():
        filters = arrays_data.MappedArrayData.accepted_filters()
        filters.update({'datatype_class._nr_dimensions': {'type': 'int', 'display': 'Dimensionality',
                                                          'operations': ['==', '<', '>']},
                        'datatype_class._length_1d': {'type': 'int', 'display': 'Shape 1',
                                                      'operations': ['==', '<', '>']},
                        'datatype_class._length_2d': {'type': 'int', 'display': 'Shape 2',
                                                      'operations': ['==', '<', '>']}})
        return filters
    
    
    def reduce_dimension(self, ui_selected_items):
        """
        ui_selected_items is a dictionary which contains items of form:
        '$name_$D : [$gid_$D_$I,..., func_$FUNC]' where '$D - int dimension', '$gid - a data type gid',
        '$I - index in that dimension' and '$FUNC - an aggregation function'.

        If the user didn't select anything from a certain dimension then it means that the entire
        dimension is selected
        """

        #The fields 'aggregation_functions' and 'dimensions' will be of form:
        #- aggregation_functions = {dimension: agg_func, ...} e.g.: {0: sum, 1: average, 2: none, ...}
        #- dimensions = {dimension: [list_of_indexes], ...} e.g.: {0: [0,1], 1: [5,500],...}
        dimensions, aggregation_functions, required_dimension, shape_restrictions = \
            self.parse_selected_items(ui_selected_items)

        if required_dimension is not None:
            #find the dimension of the resulted array
            dim = len(self.shape)
            for key in aggregation_functions.keys():
                if aggregation_functions[key] != "none":
                    dim -= 1
            for key in dimensions.keys():
                if (len(dimensions[key]) == 1 and 
                    (key not in aggregation_functions 
                     or aggregation_functions[key] == "none")):
                    dim -= 1
            if dim != required_dimension:
                self.logger.debug("Dimension for selected array is incorrect")
                raise Exception("Dimension for selected array is incorrect!")

        result = self.array_data
        full = slice(0, None)
        cut_dimensions = 0
        for i in xrange(len(self.shape)):
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

        #check that the shape for the resulted array respects given conditions
        result_shape = result.shape
        for i in xrange(len(result_shape)):
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
                    raise Exception(msg)

        if required_dimension is not None and 1 <= required_dimension != len(result.shape):
            self.logger.debug("Dimensions of the selected array are incorrect")
            raise Exception("Dimensions of the selected array are incorrect!")

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
        if len(expected_shape_str.strip()) == 0 or \
           len(operations_str.strip()) == 0:
            return result

        shape_array = str(expected_shape_str).split(",")
        op_array = str(operations_str).split(",")

        operations = self._get_operations()
        for i in xrange(len(shape_array)):
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


    
