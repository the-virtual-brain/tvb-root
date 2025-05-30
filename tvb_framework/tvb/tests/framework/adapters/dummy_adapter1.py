# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
Created on Jul 21, 2011

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import tvb.core.adapters.abcadapter as abcadapter
from tvb.basic.neotraits.api import Int
from tvb.core.neotraits.forms import IntField
from tvb.core.neotraits.view_model import ViewModel
from tvb.tests.framework.datatypes.dummy_datatype_index import DummyDataTypeIndex


class DummyModel(ViewModel):
    test1_val1 = Int(default=0)
    test1_val2 = Int(default=0)


class DummyAdapter1Form(abcadapter.ABCAdapterForm):
    """
        This class is used for testing purposes.
    """

    def __init__(self):
        super(DummyAdapter1Form, self).__init__()
        self.test1_val1 = IntField(DummyModel.test1_val1, name='test1_val1')
        self.test1_val2 = IntField(DummyModel.test1_val2, name='test1_val2')

    @staticmethod
    def get_view_model():
        return DummyModel

    @staticmethod
    def get_required_datatype():
        return DummyDataTypeIndex

    @staticmethod
    def get_input_name():
        return "dummy_data_type"

    @staticmethod
    def get_filters():
        pass


class DummyAdapter1(abcadapter.ABCAdapter):
    """
        This class is used for testing purposes.
    """
    launch_mode = abcadapter.AdapterLaunchModeEnum.SYNC_SAME_MEM

    def __init__(self):
        super(DummyAdapter1, self).__init__()

    @staticmethod
    def get_view_model():
        return DummyModel

    def get_form_class(self):
        return DummyAdapter1Form

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
        Tests successful launch of a synchronous adapter

        :param test1_val1: a dummy integer value
        :param test1_val2: a dummy integer value
        :return: a `Datatype1` object
        """
        result = DummyDataTypeIndex()
        result.row1 = 'test'
        result.row2 = 'test'
        return result
