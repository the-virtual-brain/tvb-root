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

Scientific methods for the Array datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import numpy
import tvb.datatypes.arrays_data as arrays_data


class FloatArrayScientific(arrays_data.FloatArrayData):
    """ This class exists to add scientific methods to FloatArrayData """
    
    
    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = {"Array type": self.__class__.__name__,
                   "Shape": self.shape,
                   "Maximum": self.value.max(),
                   "Minimum": self.value.min(),
                   "Mean": self.value.mean()}
        return summary


class IntegerArrayScientific(arrays_data.IntegerArrayData):
    """ This class exists to add scientific methods to IntegerArrayData """
    
    
    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = {"Array type": self.__class__.__name__,
                   "Shape": self.shape,
                   "Maximum": self.value.max(),
                   "Minimum": self.value.min(),
                   "Mean": self.value.mean(),
                   "Median": numpy.median(self.value)}
        return summary


class ComplexArrayScientific(arrays_data.ComplexArrayData):
    """ This class exists to add scientific methods to ComplexArrayData """
    
    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = {"Array type": self.__class__.__name__,
                   "Shape": self.shape,
                   "Maximum": self.value.max(),
                   "Minimum": self.value.min(),
                   "Mean": self.value.mean()}
        return summary


class BoolArrayScientific(arrays_data.BoolArrayData):
    """ This class exists to add scientific methods to BoolArrayData """
    
    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = {"Array type": self.__class__.__name__,
                   "Shape": self.shape}
        return summary


class StringArrayScientific(arrays_data.StringArrayData):
    """ This class exists to add scientific methods to StringArrayData """
    
    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = {"Array type": self.__class__.__name__,
                   "Shape": self.shape}
        return summary


class PositionArrayScientific(arrays_data.PositionArrayData, FloatArrayScientific):
    """ This class exists to add scientific methods to PositionArrayData"""
    pass


class OrientationArrayScientific(arrays_data.OrientationArrayData, FloatArrayScientific):
    """ This class exists to add scientific methods to OrientationArrayData """
    pass


class IndexArrayScientific(arrays_data.IndexArrayData, IntegerArrayScientific):
    """ This class exists to add scientific methods to IndexArrayData """
    pass


class MappedArrayScientific(arrays_data.MappedArrayData):
    """This class exists to add scientific methods to MappedArrayData"""
    __tablename__ = None



