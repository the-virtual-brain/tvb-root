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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import pytest
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.core.entities import model
from tvb.core.entities.storage import dao
from tvb.core.adapters.abcadapter import ABCSynchronous
from tvb.tests.framework.core.factory import TestFactory


class ComplexInterfaceAdapter(ABCSynchronous):
    """Adapter with a complex interface, target for testing ABCAdapter methods."""

    def get_input_tree(self):
        return [{'name': 'surface', 'type': 'tvb.core.entities.model.DataType', "datatype": True,
                 'attributes': [{'name': 'att1', 'type': 'int', 'default': '0'},
                                {'name': 'att2', 'type': 'float', 'default': '1'}]},
                {'name': 'monitors', 'type': 'selectMultiple', 'default': '["EEG", "MEEG"]',
                 'options': [{'name': 'EEG', 'value': 'EEG',
                              'attributes': [{'name': 'mon_att1', 'type': 'int', 'default': '0'},
                                             {'name': 'mon_att2', 'type': 'float', 'default': '1'}]},
                             {'name': 'MEEG', 'value': 'MEEG',
                              'attributes': [{'name': 'mon_att1', 'type': 'int', 'default': '0'},
                                             {'name': 'mon_att3', 'type': 'str', 'default': '1'}]},
                             {'name': 'BOLD', 'value': 'BOLD',
                              'attributes': [{'name': 'mon_att1', 'type': 'int', 'default': '0'},
                                             {'name': 'mon_att4', 'type': 'str', 'default': '1'}]}]
                 },
                {'name': 'length', 'type': 'int', 'default': '0'}]

    def get_output(self):
        pass

    def launch(self, **kwargs):
        pass

    def get_required_memory_size(self, **kwargs):
        return 0

    def get_required_disk_size(self, **kwargs):
        return 0


class TestAdapterABC(TransactionalTestCase):
    """Unit test for ABCAdapter"""
    EXPECTED_FLAT_NAMES = ["surface", "surface_parameters_att1", "surface_parameters_att2",
                           "monitors", "length",
                           "monitors_parameters_option_EEG_mon_att1", "monitors_parameters_option_EEG_mon_att2",
                           "monitors_parameters_option_MEEG_mon_att1", "monitors_parameters_option_MEEG_mon_att3",
                           "monitors_parameters_option_BOLD_mon_att1", "monitors_parameters_option_BOLD_mon_att4"]

    SUBMIT_DATASET_1 = {"surface": "",
                        "surface_parameters_option_456-GID-1_att1": "10",
                        "surface_parameters_option_456-GID-1_att2": "4.2",
                        "monitors": "EEG",
                        "monitors_parameters_option_EEG_mon_att1": "2",
                        "monitors_parameters_option_EEG_mon_att2": "7.3",
                        "monitors_parameters_option_BOLD_mon_att1": "42",
                        "monitors_parameters_option_BOLD_mon_att4": "string_value",
                        "length": "23"}
    EXPECTED_FILTERED_SET1 = {"surface": None.__class__, "monitors": list,
                              "monitors_parameters": dict, "length": int}

    SUBMIT_DATASET_2 = {"surface": "",
                        "surface_parameters_option_456-GID-1_att1": "10",
                        "surface_parameters_option_456-GID-1_att2": "4.2",
                        "monitors": "EEG",
                        "monitors_parameters_option_EEG_mon_att1": "should_have_been_int",
                        "monitors_parameters_option_EEG_mon_att2": "should_have_been_float",
                        "length": "23"}

    SUBMIT_DATASET_3 = {"surface": "$GID$",
                        "surface_parameters_option_$GID$_att1": "10",
                        "surface_parameters_option_$GID$_att2": "4.2",
                        "monitors": "EEG",
                        "monitors_parameters_option_EEG_mon_att1": "2",
                        "monitors_parameters_option_EEG_mon_att2": "7.3",
                        "length": "23"}
    EXPECTED_FILTERED_SET3 = {"monitors": list, "monitors_parameters": dict,
                              "length": int, "surface": model.DataType, 'surface_parameters': dict}

    SUBMIT_DATASET_4 = {"surface": "",
                        "surface_parameters_option_456-GID-1_att1": "10",
                        "surface_parameters_option_456-GID-1_att2": "4.2",
                        "monitors": ['EEG', 'BOLD'],
                        "monitors_parameters_option_EEG_mon_att1": "43",
                        "monitors_parameters_option_EEG_mon_att2": "7.3",
                        "monitors_parameters_option_BOLD_mon_att1": "42",
                        "monitors_parameters_option_BOLD_mon_att4": "string_value",
                        "length": "23"}
    EXPECTED_FILTERED_SET4 = {"surface": None.__class__, "monitors": list,
                              "monitors_parameters": dict, "length": int}

    def transactional_setup_method(self):
        """
        Reset the database before each test.
        """
        self.test_adapter = ComplexInterfaceAdapter()

    def test_flat_interface(self):
        """
        Test method flaten_input_interface on a complex adapter interface.
        """
        list_flat = self.test_adapter.flaten_input_interface()
        assert len(self.EXPECTED_FLAT_NAMES) == len(list_flat)
        for row in list_flat:
            assert row["name"] in self.EXPECTED_FLAT_NAMES

    def test_prepare_ui_inputs_simple(self):
        """
        Test for ABCAdapter.prepare_ui_inputs on a complex adapter interface.
        We need to make sure that sub-attributes for un-submitted select options are dropped.
        """
        kwargs = self.test_adapter.prepare_ui_inputs(self.SUBMIT_DATASET_1)

        for expected_name, expected_type in self.EXPECTED_FILTERED_SET1.iteritems():
            assert expected_name in kwargs
            assert isinstance(kwargs[expected_name], expected_type)
        assert len(self.EXPECTED_FILTERED_SET1) == len(kwargs)

        assert 2 == len(kwargs["monitors_parameters"]["EEG"])
        assert isinstance(kwargs["monitors_parameters"]["EEG"]["mon_att1"], int)
        assert isinstance(kwargs["monitors_parameters"]["EEG"]["mon_att2"], float)

    def test_prepare_inputs_wrong_type(self):
        """
        Test for ABCAdapter.prepare_ui_inputs, when invalid values passed for numeric fields.
        """
        with pytest.raises(Exception):
            self.test_adapter.prepare_ui_inputs(self.SUBMIT_DATASET_2)

    def test_prepare_inputs_datatype(self):
        """
        Test for ABCAdapter.prepare_ui_inputs method when submitting DataType with sub-attributes.
        """
        parent_op = TestFactory.create_operation()
        test_entity = dao.store_entity(model.DataType(operation_id=parent_op.id))
        dataset_3 = {}
        for key, value in self.SUBMIT_DATASET_3.iteritems():
            dataset_3[key.replace("$GID$", test_entity.gid)] = value.replace("$GID$", test_entity.gid)

        kwargs = self.test_adapter.prepare_ui_inputs(dataset_3)

        for expected_name, expected_type in self.EXPECTED_FILTERED_SET3.iteritems():
            assert expected_name in kwargs
            assert isinstance(kwargs[expected_name], expected_type)
        assert len(self.EXPECTED_FILTERED_SET3) == len(kwargs)

        assert 2 == len(kwargs["surface_parameters"])
        assert isinstance(kwargs["surface_parameters"]["att1"], int)
        assert isinstance(kwargs["surface_parameters"]["att2"], float)

    def test_prepare_select_multiple(self):
        """
        Test for ABCAdapter.prepare_ui_inputs method when submitting 2 values in a multiple-select input.
        """
        kwargs = self.test_adapter.prepare_ui_inputs(self.SUBMIT_DATASET_4)

        for expected_name, expected_type in self.EXPECTED_FILTERED_SET4.iteritems():
            assert expected_name in kwargs
            assert isinstance(kwargs[expected_name], expected_type)
        assert len(self.EXPECTED_FILTERED_SET4) == len(kwargs)

        assert 2 == len(kwargs["monitors_parameters"]["BOLD"])
        assert 2 == len(kwargs["monitors_parameters"]["EEG"])
        assert isinstance(kwargs["monitors_parameters"]["BOLD"]["mon_att1"], int)
        assert 42 == kwargs["monitors_parameters"]["BOLD"]["mon_att1"]
        assert 43 == kwargs["monitors_parameters"]["EEG"]["mon_att1"]
        assert isinstance(kwargs["monitors_parameters"]["BOLD"]["mon_att4"], str)
