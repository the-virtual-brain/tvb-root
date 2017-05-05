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
Created on Jul 21, 2011

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

from tvb.core.adapters.abcadapter import ABCSynchronous
from tvb.tests.framework.datatypes.datatype1 import Datatype1

class TestAdapter1(ABCSynchronous):
    """
        This class is used for testing purposes.
    """
    def __init__(self):
        ABCSynchronous.__init__(self)
        
    def get_input_tree(self):
        return [{'name':'test1_val1', 'type':'int', 'default':'0'},
                 {'name':'test1_val2', 'type':'int', 'default':'0'}]
                
    def get_output(self):
        return [Datatype1]
    
    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        # Don't know how much memory is needed.
        return -1
    
    def get_required_disk_size(self, **kwargs):
        """
        Returns the required disk size to be able to run the adapter.
        """
        return 0
        
    def launch(self, test1_val1, test1_val2):
        """
        Tests successful launch of an ABCSynchronous adapter

        :param test1_val1: a dummy integer value
        :param test1_val2: a dummy integer value
        :return: a `Datatype1` object
        """
        int(test1_val1)
        int(test1_val2)
        result = Datatype1()
        result.row1 = 'test'
        result.row2 = 'test'
        result.storage_path = self.storage_path
        return result
    
    
class TestAdapterDatatypeInput(ABCSynchronous):
    """
        This class is used for testing purposes.
    """
    def __init__(self):
        ABCSynchronous.__init__(self)
        
    def get_input_tree(self):
        return [{'name':'test_dt_input', 'type' : Datatype1}, 
                {'name':'test_non_dt_input', 'type': 'int', 'default':'0'}]
                
    def get_output(self):
        return [Datatype1]
    
    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        # Don't know how much memory is needed.
        return -1
    
    def get_required_disk_size(self, **kwargs):
        """
        Returns the required disk size to be able to run the adapter.
        """
        return 0
        
    def launch(self, test_dt_input, test_non_dt_input):
        str(test_dt_input)
        result = Datatype1()
        result.row1 = 'test'
        result.row2 = 'test'
        result.storage_path = self.storage_path
        return result