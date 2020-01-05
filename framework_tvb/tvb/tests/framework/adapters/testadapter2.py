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
Created on Jul 21, 2011

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

from time import sleep
import tvb.core.adapters.abcadapter as abcadapter
from tvb.core.neotraits.forms import SimpleIntField
from tvb.tests.framework.datatypes.test_datatype_index import DummyDataTypeIndex


class TestAdapter2Form(abcadapter.ABCAdapterForm):
    """
        This class is used for testing purposes.
    """

    def __init__(self, prefix='', project_id=None):
        super(TestAdapter2Form, self).__init__(prefix, project_id)
        self.test = SimpleIntField(self, name="test", default=0)

    @staticmethod
    def get_required_datatype():
        return DummyDataTypeIndex

    @staticmethod
    def get_input_name():
        return "dummy_data_type"

    @staticmethod
    def get_filters():
        pass


class TestAdapter2(abcadapter.ABCAsynchronous):
    """
        This class is used for testing purposes.
    """    
    def __init__(self):
        super(TestAdapter2, self).__init__()
        
    def get_form_class(self):
        return TestAdapter2Form
                
    def get_output(self):
        return [DummyDataTypeIndex]
    
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
    
    def launch(self, test2=None):
        """
        Mimics launching with a long delay

        :param test2: a dummy parameter
        """
        sleep(15)
        test_var = 1
        int(test_var)


class TestAdapter22Form(abcadapter.ABCAdapterForm):
    """
        This class is used for testing purposes.
    """

    def __init__(self, prefix='', project_id=None):
        super(TestAdapter22Form, self).__init__(prefix, project_id)
        self.test2 = SimpleIntField(self, name="test2", default=0)

    @staticmethod
    def get_required_datatype():
        return DummyDataTypeIndex

    @staticmethod
    def get_input_name():
        return "dummy_data_type"

    @staticmethod
    def get_filters():
        pass


class TestAdapter22(abcadapter.ABCAsynchronous):
    """
        This class is used for testing purposes.
    """

    def __init__(self):
        super(TestAdapter22, self).__init__()
        
    def get_form_class(self):
        return TestAdapter22Form

    def get_output(self):
        return [DummyDataTypeIndex]

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
        
    def launch(self):
        test_var = 1
        str(test_var)