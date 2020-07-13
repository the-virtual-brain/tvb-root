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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

from tvb.core.adapters import abcadapter
from tvb.basic.neotraits.api import Int
from tvb.core.neotraits.forms import IntField
from tvb.core.neotraits.view_model import ViewModel
from tvb.tests.framework.datatypes.dummy_datatype_index import DummyDataTypeIndex


class TestModel(ViewModel):
    param_5 = Int(default=0)
    param_6 = Int(default=0)


class TestModelRequired(ViewModel):
    test = Int(default=100)


class TestAdapter3Form(abcadapter.ABCAdapterForm):
    """
        This class is used for testing purposes.
    """

    def __init__(self, prefix='', project_id=None):
        super(TestAdapter3Form, self).__init__(prefix, project_id)
        self.param_5 = IntField(TestModel.param_5, self, name='param_5')
        self.param_6 = IntField(TestModel.param_6, self, name='param_6')

    @staticmethod
    def get_view_model():
        return TestModel

    @staticmethod
    def get_required_datatype():
        return DummyDataTypeIndex

    @staticmethod
    def get_input_name():
        return "param_5"

    @staticmethod
    def get_filters():
        pass


class TestAdapter3(abcadapter.ABCAsynchronous):
    """
    This class is used for testing purposes.
    It will be used as an adapter for testing Groups of operations. For ranges to work, it need to be asynchronous.
    """

    def __init__(self):
        super(TestAdapter3, self).__init__()

    @staticmethod
    def get_view_model():
        return TestModel

    def load_view_model(self, operation):
        return TestModel

    def get_form_class(self):
        return TestAdapter3Form

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
        result = DummyDataTypeIndex()
        if view_model.param_5 is not None:
            result.row1 = view_model.param_5
        if view_model.param_6 is not None:
            result.row2 = view_model.param_6
        result.storage_path = self.storage_path
        result.string_data = ["data"]
        return result


class TestAdapterHugeMemoryRequiredForm(abcadapter.ABCAdapterForm):
    """
        This class is used for testing purposes.
    """

    def __init__(self, prefix='', project_id=None):
        super(TestAdapterHugeMemoryRequiredForm, self).__init__(prefix, project_id)
        self.test = IntField(TestModelRequired.test, self, name='test')

    @staticmethod
    def get_view_model():
        return TestModelRequired

    @staticmethod
    def get_required_datatype():
        return DummyDataTypeIndex

    @staticmethod
    def get_input_name():
        return "test"

    @staticmethod
    def get_filters():
        pass


class TestAdapterHugeMemoryRequired(abcadapter.ABCAsynchronous):
    """
    Adapter used for testing launch when a lot of memory is required.
    """

    def __init__(self):
        super(TestAdapterHugeMemoryRequired, self).__init__()

    def get_form_class(self):
        return TestAdapterHugeMemoryRequiredForm

    def get_output(self):
        return [DummyDataTypeIndex]

    def get_required_memory_size(self, view_model):
        """ Huge memory requirement, should fail launch.  """
        return 999999999999999

    def get_required_disk_size(self, view_model):
        """ Returns the required disk size to be able to run the adapter. """
        return 0

    def launch(self, view_model):
        str(view_model.test)


class TestAdapterHDDRequiredForm(abcadapter.ABCAdapterForm):
    """
        This class is used for testing purposes.
    """

    def __init__(self, prefix='', project_id=None):
        super(TestAdapterHDDRequiredForm, self).__init__(prefix, project_id)
        self.test = IntField(TestModelRequired.test, self, name='test')

    @staticmethod
    def get_view_model():
        return TestModelRequired

    @staticmethod
    def get_required_datatype():
        return DummyDataTypeIndex

    @staticmethod
    def get_input_name():
        return "test"

    @staticmethod
    def get_filters():
        pass


class TestAdapterHDDRequired(abcadapter.ABCSynchronous):
    """
    Adapter used for testing launch when a lot of memory is required.
    """

    def __init__(self):
        super(TestAdapterHDDRequired, self).__init__()

    @staticmethod
    def get_view_model():
        return TestModelRequired

    def get_form_class(self):
        return TestAdapterHDDRequiredForm

    def get_output(self):
        return [DummyDataTypeIndex]

    def get_required_memory_size(self, view_model):
        """ Value test to be correctly returned """
        return 42

    def get_required_disk_size(self, view_model):
        """ Returns the required disk size to be able to run the adapter. """
        return int(view_model.test) * 8 / 2 ** 10

    def launch(self, view_model):
        """
        Mimics launching with a lot of memory usage

        :param test: should be a very large integer; the larger, the more memory is used
        :returns: a `Datatype2` object, with big string_data
        """
        result = DummyDataTypeIndex()
        result.row1 = 'param_5'
        result.row2 = 'param_6'
        result.storage_path = self.storage_path
        res_array = []
        for _ in range(int(view_model.test)):
            res_array.append("data")
        result.string_data = res_array
        return result
