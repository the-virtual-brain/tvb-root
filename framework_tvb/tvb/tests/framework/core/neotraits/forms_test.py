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
import os
import uuid

import numpy
import pytest
import tvb_data
from tvb.basic.neotraits.api import Attr, Float, Int, NArray, List
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.file.simulator.view_model import SimulatorAdapterModel
from tvb.core.entities.storage import dao
from tvb.core.neotraits.forms import TraitUploadField, StrField, FloatField, IntField, TraitDataTypeSelectField, \
    BoolField, ArrayField, SelectField, HiddenField, MultiSelectField, FormField
from tvb.core.neotraits.view_model import Str
from tvb.core.services.algorithm_service import AlgorithmService
from tvb.tests.framework.adapters.testadapter1 import TestAdapter1Form
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.tests.framework.core.factory import TestFactory


class TestForms(BaseTestCase):
    def setup_method(self):
        test_user = TestFactory.create_user("Field-User")
        self.test_project = TestFactory.create_project(test_user, "Field-Project")
        self.name = 'dummy_name'

    def teardown_method(self):
        """
        Clean-up tests data
        """
        self.clean_database()
        FilesHelper().remove_project_structure(self.test_project.name)

    def test_upload_field(self):
        connectivity_file = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_96.zip')
        data_file = Str('Test Upload Field')
        required_type = '.zip'
        upload_field = TraitUploadField(data_file, required_type, self.name)

        post_data = {'Data_Subject': 'John Doe', self.name: connectivity_file, 'normalization': 'explicit-None-value'}
        upload_field.fill_from_post(post_data)

        assert post_data[self.name] == upload_field.data, "Path was not set correctly on TraitUploadField!"

    def test_datatype_select_field(self, connectivity_index_factory):
        trait_attribute = SimulatorAdapterModel.connectivity

        connectivity_1 = connectivity_index_factory(2)
        connectivity_2 = connectivity_index_factory(2)
        connectivity_3 = connectivity_index_factory(2)

        op_1 = dao.get_operation_by_id(connectivity_1.fk_from_operation)
        op_1.fk_launched_in = self.test_project.id
        dao.store_entity(op_1)
        op_2 = dao.get_operation_by_id(connectivity_2.fk_from_operation)
        op_2.fk_launched_in = self.test_project.id
        dao.store_entity(op_2)
        op_3 = dao.get_operation_by_id(connectivity_3.fk_from_operation)
        op_3.fk_launched_in = self.test_project.id
        dao.store_entity(op_3)

        datatype_select_field = TraitDataTypeSelectField(trait_attribute, self.name, None, has_all_option=True)
        AlgorithmService().fill_selectfield_with_datatypes(datatype_select_field, self.test_project.id)

        post_data = {self.name: connectivity_1.gid}
        datatype_select_field.fill_from_post(post_data)

        options = datatype_select_field.options()
        conn_1 = next(options)
        conn_2 = next(options)
        conn_3 = next(options)

        next(options)
        with pytest.raises(StopIteration):
            next(options)

        assert conn_1.value == connectivity_3.gid
        assert conn_2.value == connectivity_2.gid
        assert conn_3.value == connectivity_1.gid
        assert uuid.UUID(post_data[self.name]) == datatype_select_field.data, "UUID data was not set correctly on" \
                                                                              " TraitDataTypeSelectField"

    def test_bool_field(self):
        bool_attr = Attr(field_type=bool, default=True, label='Dummy Bool')
        bool_field = BoolField(bool_attr, self.name)

        post_data = {'dummy_name': 'on'}
        bool_field.fill_from_post(post_data)
        assert bool_field.data, "True (boolean) was not set correctly on BoolField!"

        post_data = {}
        bool_field.fill_from_post(post_data)
        assert bool_field.data is False, "False (boolean) was not set correctly on BoolField!"

    def test_str_field_required(self):
        str_attr = Str(label='Dummy Str', default='')
        str_field = StrField(str_attr, self.name)

        post_data = {'dummy_name': 'dummy_str'}
        str_field.fill_from_post(post_data)
        assert str_field.data == post_data[self.name], "Str data was not set correctly on StrField!"

        post_data = {'dummy_name': ''}
        str_field.fill_from_post(post_data)
        assert str_field.validate() is False, "Validation should have failed on StrField!"

    def test_str_field_optional(self):
        str_attr = Str(label='Dummy Str', default='', required=False)
        str_field = StrField(str_attr, self.name)

        post_data = {'dummy_name': ''}
        str_field.fill_from_post(post_data)
        assert str_field.data == '', "Str data was not set correctly on StrField!"
        assert str_field.validate(), "Validation should not have failed on StrField!"

    def test_int_field_required(self):
        int_attr = Int(label='Dummy Int', default=0)
        int_field = IntField(int_attr, self.name)

        post_data = {'dummy_name': '10'}
        int_field.fill_from_post(post_data)
        assert int_field.data == 10, "Int data was not set correctly on IntField!"
        assert int_field.value == int_field.data, "Int data was not set correctly on IntField!"

    def test_int_field_required_empty(self):
        int_attr = Int(label='Dummy Int', default=0)
        int_field = IntField(int_attr, self.name)

        post_data = {'dummy_name': ''}
        int_field.fill_from_post(post_data)
        assert int_field.validate() is False, "Validation should have failed on IntField!"
        assert int_field.value == ''

    def test_int_field_optinal(self):
        int_attr = Int(label='Dummy Int', default=0, required=False)
        int_field = IntField(int_attr, self.name)

        post_data = {'dummy_name': ''}
        int_field.fill_from_post(post_data)
        assert int_field.data is None, "Empty data was not set correctly on IntField!"
        assert int_field.value == ''

    def test_float_field_required(self):
        float_attr = Float(label='Dummy Float', default=0.)
        float_field = FloatField(float_attr, self.name)

        post_data = {'dummy_name': '10.5'}
        float_field.fill_from_post(post_data)
        assert float_field.data == float(post_data[self.name]), "Float data was not set correctly on FloatField!"
        assert float_field.value == float_field.data

    def test_float_field_required_empty(self):
        float_attr = Float(label='Dummy Float', default=0.)
        float_field = FloatField(float_attr, self.name)

        post_data = {'dummy_name': ''}
        float_field.fill_from_post(post_data)
        assert float_field.validate() is False, "Validation should have failed on FloatField!"
        assert float_field.value == ''

    def test_float_field_optional(self):
        float_attr = Float(label='Dummy Float', default=0., required=False)
        float_field = FloatField(float_attr, self.name)

        post_data = {'dummy_name': ''}
        float_field.fill_from_post(post_data)
        assert float_field.data is None, "Empty data was not set correctly on FloatField!"
        assert float_field.value == ''

    def test_array_field_required(self):
        int_attr = NArray(label='Dummy Int', default=0)
        array_field = ArrayField(int_attr, self.name)

        post_data = {'dummy_name': '[1, 2, 3]'}
        array_field.fill_from_post(post_data)
        assert numpy.array_equal(array_field.data, numpy.array([1, 2, 3])), "Data was not set correctly on ArrayField!"
        assert array_field.value == '[1.0, 2.0, 3.0]'

        post_data = {'dummy_name': '[]'}
        array_field.fill_from_post(post_data)
        assert numpy.array_equal(array_field.data, numpy.array([])), "Data was not set correctly on ArrayField!"
        assert array_field.value == '[]'

    def test_array_field_required_empty(self):
        int_attr = NArray(label='Dummy Int', default=0)
        array_field = ArrayField(int_attr, self.name)

        post_data = {'dummy_name': ''}
        array_field.fill_from_post(post_data)
        assert array_field.validate() is False, "Validation should have failed on ArrayField!"
        assert array_field.value == ''

    def test_array_field_optional(self):
        int_attr = NArray(label='Dummy Int', default=0, required=False)
        array_field = ArrayField(int_attr, self.name)

        array_str = ''
        post_data = {'dummy_name': array_str}
        array_field.fill_from_post(post_data)
        assert array_field.data is None, "Empty data was not set correctly on ArrayField!"
        assert array_field.value == ''

    def test_select_field_required(self):
        str_attr = Attr(field_type=str, default='2', label='Dummy Bool', choices=('1', '2', '3'))
        select_field = SelectField(str_attr, self.name)

        post_data = {'dummy_name': '1'}
        select_field.fill_from_post(post_data)
        assert select_field.data == post_data[self.name], "Data was not set correctly on SelectField!"
        assert select_field.validate(), "Validation should have passed on SelectField!"

    def test_select_field_optional_none(self):
        str_attr = Attr(field_type=str, default='2', label='Dummy Bool', choices=('1', '2', '3'), required=False)
        select_field = SelectField(str_attr, self.name)

        post_data = {'dummy_name': 'explicit-None-value'}
        select_field.fill_from_post(post_data)
        assert select_field.data == None, "Data was not set correctly on SelectField!"
        assert select_field.validate(), "Validation should have passed on SelectField!"

    def test_select_field_invalid(self):
        str_attr = Attr(field_type=str, default='2', label='Dummy Bool', choices=('1', '2', '3'))
        select_field = SelectField(str_attr, self.name)

        post_data = {'dummy_name': '4'}
        select_field.fill_from_post(post_data)
        assert select_field.validate() is False, "Validation should have failed on SelectField!"

    def test_multi_select_field(self):
        list_attr = List(of=str, label='Dummy List', choices=('1', '2', '3', '4', '5'))
        multi_select_field = MultiSelectField(list_attr, self.name)

        post_data = {'dummy_name': ['2', '3', '4']}
        multi_select_field.fill_from_post(post_data)
        assert multi_select_field.data == post_data[self.name], "Data was not set correctly on the MultiSelectField!"

    def test_multi_select_field_invalid_data(self):
        list_attr = List(of=str, label='Dummy List', choices=('1', '2', '3', '4', '5'))
        multi_select_field = MultiSelectField(list_attr, self.name)

        post_data = {'dummy_name': ['2', '3', '6']}
        multi_select_field.fill_from_post(post_data)
        assert multi_select_field.validate() is False, "Validation should have failed on MultiSelectField!"

    def test_multi_select_field_no_data(self):
        list_attr = List(of=str, label='Dummy List', choices=('1', '2', '3', '4', '5'))
        multi_select_field = MultiSelectField(list_attr, self.name)

        post_data = {}
        multi_select_field.fill_from_post(post_data)
        assert multi_select_field.validate() is False, "Validation should have failed on MultiSelectField!"

    def test_hidden_field(self):
        hidden_str_attr = Str(label='Dummy Str', default='')
        hidden_field = HiddenField(hidden_str_attr, self.name)

        post_data = {'dummy_name': 'Dummy Hidden Str'}
        hidden_field.fill_from_post(post_data)
        assert hidden_field.data == post_data[self.name], "Hidden data was not set correctly on HiddenField!"
        assert hidden_field.trait_attribute.label == '', "Hidden field's trait attributes should have empty labels!"

    def test_form_field(self):
        form_field = FormField(TestAdapter1Form, self.name)

        test_val1 = 'test1_val1'
        test_val2 = 'test1_val2'
        post_data = {test_val1: '10', test_val2: '15'}
        form_field.fill_from_post(post_data)

        assert form_field.form.test1_val1.data == int(post_data[test_val1]), \
            "test1_val1 was not correctly set on Form!"
        assert form_field.form.test1_val2.data == int(post_data[test_val2]), \
            "test1_val2 was not set correctly set on Form!"

    def test_form(self):
        form = TestAdapter1Form()

        test_val1 = 'test1_val1'
        test_val2 = 'test1_val2'
        post_data = {test_val1: '10', test_val2: '15'}
        form.fill_from_post(post_data)

        assert int(post_data[test_val1]) == form.test1_val1.data, "test1_val1 was not correctly set on Form!"
        assert int(post_data[test_val2]) == form.test1_val2.data, "test1_val2 was not set correctly set on Form!"
