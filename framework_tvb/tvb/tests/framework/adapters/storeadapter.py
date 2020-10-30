# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.model.model_datatype import DataType


class StoreAdapter(ABCAdapter):
    """
    The purpose of this adapter is only to allow you to
    store into the db a list of data types.
    """
    list_of_entities_to_store = []

    def __init__(self, list_of_entities_to_store):
        """
        Expacts a list of 'DataType' instances.
        """
        ABCAdapter.__init__(self)
        if (list_of_entities_to_store is None 
            or not isinstance(list_of_entities_to_store, list) 
            or len(list_of_entities_to_store) == 0):
            raise Exception("The adapter expacts a list of entities")

        self.list_of_entities_to_store = list_of_entities_to_store

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

    def get_input_tree(self):
        """
        Describes inputs and outputs of the launch method.
        """
        return None

    def get_output(self):
        """
        Describes the outputs of the launch method.
        """
        return [DataType]


    def launch(self):
        """
        Saves in the db the list of entities passed to the constructor.
        """
        return self.list_of_entities_to_store
    
    
    
    