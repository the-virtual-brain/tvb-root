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

The Data component of traited array datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

"""

import numpy
import tvb.basic.traits.core as core
import tvb.basic.traits.types_basic as basic
from tvb.basic.traits.types_mapped import MappedType, Array


class FloatArrayData(Array):
    """ A numpy.ndarray of dtype numpy.float64 """
    _ui_name = "Floating-point array"
    dtype = basic.DType(default=numpy.float64)


class IntegerArrayData(Array):
    """ A numpy.ndarray of dtype numpy.int32 """
    _ui_name = "Array of integers"
    dtype = basic.DType(default=numpy.int32)


class ComplexArrayData(Array):
    """ A numpy.ndarray of dtype numpy.complex128 """
    _ui_name = "Array of complex numbers"
    dtype = basic.DType(default=numpy.complex128)


class BoolArrayData(Array):
    """ A numpy.ndarray of dtype numpy.bool """
    _ui_name = "Boolean array"
    dtype = basic.DType(default=numpy.bool)


# if you want variable length strings, you must use dtype=object
# otherwise, must specify max lenth as 'na' where n is integer,
# e.g. dtype='100a' for a string w/ max len 100 characters.
class StringArrayData(Array):
    """ A numpy.ndarray of dtype str """
    _ui_name = "Array of strings"
    dtype = None


class PositionArrayData(FloatArrayData):
    """ An array specifying position. """
    _ui_name = "Array of positions"

    coordinate_system = basic.String(label="Coordinate system",
                                     default="cartesian",
                                     doc="""The coordinate system used to specify the positions.
                                     Eg: 'spherical', 'polar'""")

    coordinate_space = basic.String(label="Coordinate space",
                                    default="None",
                                    doc="The standard space the positions are in, eg, 'MNI', 'colin27'")


class OrientationArrayData(FloatArrayData):
    """ An array specifying orientations. """
    _ui_name = "Array of orientations"

    coordinate_system_or = basic.String(label="Coordinate system",
                                        default="cartesian")


class IndexArrayData(IntegerArrayData):
    """ An array that indexes another array. """
    _ui_name = "Index array"

    target = Array(label="Indexed array",
                   file_storage=core.FILE_STORAGE_NONE,
                   doc="A link to the array that the indices index.")


class MappedArrayData(MappedType):
    """
    Array that will be Mapped as a table in DB.
    """
    
    title = basic.String
    label_x, label_y = basic.String, basic.String
    aggregation_functions = basic.JSONType(required=False)
    dimensions_labels = basic.JSONType(required=False)
    
    nr_dimensions, length_1d, length_2d, length_3d, length_4d = [basic.Integer] * 5
    array_data = Array()  
    
    __generate_table__ = True

    @property
    def display_name(self):
        """
        Overwrite from superclass and add title field
        """
        previous = super(MappedArrayData, self).display_name
        if previous is None:
            return str(self.title)
        return str(self.title) + " " + previous
       
       
    @property
    def shape(self):
        """
        Shape for current wrapped NumPy array.
        """
        return self.array_data.shape
    
     
