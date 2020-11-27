import pytest
import tvb_data
import os
import numpy
import uuid
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.file.simulator.view_model import SimulatorAdapterModel
from tvb.core.entities.storage import dao
from tvb.core.neotraits.forms import TraitUploadField, StrField, FloatField, IntField, TraitDataTypeSelectField, \
    BoolField, ArrayField, SelectField, HiddenField, MultiSelectField, Form, FormField
from tvb.core.neotraits.view_model import Str
from tvb.basic.neotraits.api import Attr, Float, Int, NArray, List
from tvb.tests.framework.adapters.testadapter1 import TestAdapter1Form
from tvb.tests.framework.core.base_testcase import BaseTestCase, json
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
        temporary_files = []
        upload_field = TraitUploadField(data_file, required_type, self.test_project.id, self.name, temporary_files)

        post_data = {'Data_Subject': 'John Doe', self.name: connectivity_file, 'normalization': 'explicit-None-value'}
        upload_field.fill_from_post(post_data)

        assert post_data[self.name] == upload_field.data, "Path was not set correctly on TraitUploadField!"

    def test_datatype_select_field(self, connectivity_index_factory):
        trait_attribute = SimulatorAdapterModel.connectivity

        datatype_select_field = TraitDataTypeSelectField(trait_attribute, self.test_project.id, self.name, None,
                                                         has_all_option=True)
        connectivity_1 = connectivity_index_factory(2)
        connectivity_2 = connectivity_index_factory(2)
        connectivity_3 = connectivity_index_factory(2)

        post_data = {self.name: connectivity_1.gid}
        datatype_select_field.fill_from_post(post_data)

        op_1 = dao.get_operation_by_id(connectivity_1.fk_from_operation)
        op_1.fk_launched_in = self.test_project.id
        dao.store_entity(op_1)
        op_2 = dao.get_operation_by_id(connectivity_2.fk_from_operation)
        op_2.fk_launched_in = self.test_project.id
        dao.store_entity(op_2)
        op_3 = dao.get_operation_by_id(connectivity_3.fk_from_operation)
        op_3.fk_launched_in = self.test_project.id
        dao.store_entity(op_3)

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
        bool_field = BoolField(bool_attr, self.test_project.id, self.name)

        post_data = {'dummy_name': 'on'}
        bool_field.fill_from_post(post_data)

        assert bool_field.data, "True (boolean) was not set correctly on BoolField!"

    def test_str_field(self):
        str_attr = Str(label='Dummy Str', default='')
        str_field = StrField(str_attr, self.test_project.id, self.name)

        post_data = {'dummy_name': 'dummy_str'}
        str_field.fill_from_post(post_data)

        assert post_data[self.name] == str_field.data, "Str data was not set correctly on StrField!"

    def test_int_field(self):
        int_attr = Int(label='Dummy Int', default=0)
        int_field = IntField(int_attr, self.test_project.id, self.name)

        int_str = '10'
        post_data = {'dummy_name': int_str}
        int_field.fill_from_post(post_data)

        assert int(post_data[self.name]) == int_field.data, "Int data was not set correctly on IntField!"

    def test_float_field(self):
        float_attr = Float(label='Dummy Float', default=0.)
        float_field = FloatField(float_attr, self.test_project.id, self.name)

        float_str = '10.5'
        post_data = {'dummy_name': float_str}
        float_field.fill_from_post(post_data)

        assert float(post_data[self.name]) == float_field.data, "Float data was not set correctly on FloatField!"

    def test_array_field(self):
        int_attr = NArray(label='Dummy Int', default=0)
        array_field = ArrayField(int_attr, self.test_project.id, self.name)

        array_str = '[1, 2, 3]'
        post_data = {'dummy_name': array_str}
        array_field.fill_from_post(post_data)

        result = json.loads(array_str)
        result = numpy.array(result)

        assert (result == array_field.data).all(), "Array data was not set correctly on ArrayField!"
        assert '[1.0, 2.0, 3.0]' == array_field.value, "Array value was not set correctly on ArrayField!"

    def test_select_field(self):
        str_attr = Attr(field_type=str, default=True, required=False, label='Dummy Bool', choices=('1', '2', '3'))
        select_field = SelectField(str_attr, self.test_project.id, self.name, display_none_choice=True)

        options = select_field.options()
        next(options)
        next(options)
        next(options)
        next(options)

        post_data = {'dummy_name': '1'}
        select_field.fill_from_post(post_data)

        assert post_data[self.name] == select_field.data, "Data was not set correctly on SelectField!"

    def test_multi_select_field(self):
        list_attr = List(of=str, label='Dummy List', choices=('1', '2', '3', '4', '5'))
        multi_select_field = MultiSelectField(list_attr, self.test_project.id, self.name)

        post_data = {'dummy_name': ['2', '3', '4']}
        multi_select_field.fill_from_post(post_data)

        options = multi_select_field.options()
        next(options)
        next(options)
        next(options)
        next(options)
        next(options)

        with pytest.raises(StopIteration):
            next(options)

        assert post_data[self.name] == multi_select_field.data, \
            "Data was not set correctly on the MultiSelectField!"

    def test_multi_select_field_unvalid_data(self):
        list_attr = List(of=str, label='Dummy List', choices=('1', '2', '3', '4', '5'))
        multi_select_field = MultiSelectField(list_attr, self.test_project.id, self.name)

        post_data = {'dummy_name': ['2', '3', '6']}
        multi_select_field.fill_from_post(post_data)

        assert multi_select_field.errors is not None, "Data should have not been correctly set on MultiSelectField!"

    def test_hidden_field(self):
        hidden_str_attr = Str(label='Dummy Str', default='')
        hidden_field = HiddenField(hidden_str_attr, self.test_project.id, self.name)

        post_data = {'dummy_name': 'Dummy Hidden Str'}
        hidden_field.fill_from_post(post_data)

        assert post_data[self.name] == hidden_field.data, "Hidden data was not set correctly on HiddenField!"
        assert hidden_field.trait_attribute.label == '', "Hidden field's trait attributes should have empty labels!"

    def test_form_field(self):
        form_field = FormField(TestAdapter1Form, self.test_project.id, self.name)

        test_val1 = 'test1_val1'
        test_val2 = 'test1_val2'
        post_data = {test_val1: '10', test_val2: '15'}
        form_field.fill_from_post(post_data)

        assert int(post_data[test_val1]) == form_field.form.test1_val1.data,\
            "test1_val1 was not correctly set on Form!"
        assert int(post_data[test_val2]) == form_field.form.test1_val2.data,\
            "test1_val2 was not set correctly set on Form!"

    def test_form(self):
        form = TestAdapter1Form()

        test_val1 = 'test1_val1'
        test_val2 = 'test1_val2'
        post_data = {test_val1: '10', test_val2: '15'}
        form.fill_from_post(post_data)

        trait_fields = form.trait_fields

        next(trait_fields)
        next(trait_fields)

        with pytest.raises(StopIteration):
            next(trait_fields)

        assert int(post_data[test_val1]) == form.test1_val1.data, "test1_val1 was not correctly set on Form!"
        assert int(post_data[test_val2]) == form.test1_val2.data, "test1_val2 was not set correctly set on Form!"

    def test_unvalid_data(self):
        int_attr = Int(label='Dummy Int', default=0)
        int_field = IntField(int_attr, self.test_project.id, self.name)

        int_str = 'not an int'
        post_data = {'dummy_name': int_str}
        int_field.fill_from_post(post_data)

        assert int_field.validate() is not None, "Non-Int data should have not been set correctly on IntField!"

