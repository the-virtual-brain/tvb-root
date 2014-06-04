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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from tvb.basic.traits.types_mapped import MappedType
from tvb.basic.traits.types_basic import JSONType, String, Dict
from tvb.datatypes.time_series import TimeSeries


# Accepted Value Types to be stored.
ACCEPTED_TYPES = ['float', 'int']


class ValueWrapper(MappedType):
    """
    Class to wrap a singular value storage in DB.
    """
    
    data_value = JSONType()
    data_type = String(default='unknown')  
    data_name = String() 
    
    @property
    def display_name(self):
        """ Simple String to be used for display in UI."""
        return "Value Wrapper - " + self.data_name +" : "+ str(self.data_value) + " ("+ str(self.data_type)+ ")"
            
    
class DatatypeMeasure(MappedType):
    """
    Class to hold the metric for a previous stored DataType.
    E.g. Measure (single value) for any TimeSeries resulted in a group of Simulations
    """
    ### Actual measure (dictionary Algorithm: single Value)
    metrics = Dict
    ### DataType for which the measure was computed.
    analyzed_datatype = TimeSeries
    
    
    @property
    def display_name(self):
        """
        To be implemented in each sub-class which is about to be displayed in UI, 
        and return the text to appear.
        """
        name = "-"
        if self.metrics is not None:
            value = "\n"
            for entry in self.metrics:
                value = value + entry + ' : ' + str(self.metrics[entry]) + '\n'
            name = value
        return name
    
    
    