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
from tvb.basic.neotraits.api import Int
from tvb.core.adapters.abcadapter import AdapterLaunchModeEnum
from tvb.core.neotraits.forms import IntField
from tvb.core.neotraits.view_model import ViewModel
from tvb.tests.framework.datatypes.dummy_datatype_index import DummyDataTypeIndex


class TestModel(ViewModel):
    test = Int(default=0)


class TestAdapter2Form(abcadapter.ABCAdapterForm):
    """
        This class is used for testing purposes.
    """

    def __init__(self, project_id=None):
        super(TestAdapter2Form, self).__init__(project_id)
        self.test = IntField(TestModel.test, self.project_id, name='test')

    @staticmethod
    def get_view_model():
        return TestModel

    @staticmethod
    def get_required_datatype():
        return DummyDataTypeIndex

    @staticmethod
    def get_input_name():
        return "dummy_data_type"

    @staticmethod
    def get_filters():
        pass


class TestAdapter2(abcadapter.ABCAdapter):
    """
        This class is used for testing purposes.
    """
    launch_mode = AdapterLaunchModeEnum.ASYNC_DIFF_MEM

    def __init__(self):
        super(TestAdapter2, self).__init__()

    @staticmethod
    def get_view_model():
        return TestModel

    def get_form_class(self):
        return TestAdapter2Form

    def get_output(self):
        return [DummyDataTypeIndex]

    def get_required_memory_size(self, view_model):
        """
        Return the required memory to run this algorithm.
        """
        # Don't know how much memory is needed.
        return -1

    def get_required_disk_size(self, view_model):
        """
        Returns the required disk size to be able to run the adapter.
        """
        return 0

    def launch(self, view_model):
        """
        Mimics launching with a long delay

        :param test2: a dummy parameter
        """
        sleep(15)
        test_var = 1
        int(test_var)
