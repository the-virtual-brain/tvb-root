import tvb_data
import os
import numpy
import uuid

from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.file.simulator.view_model import SimulatorAdapterModel
from tvb.core.neotraits.forms import TraitUploadField, StrField, FloatField, IntField, TraitDataTypeSelectField, \
    BoolField, ArrayField, SelectField, HiddenField, MultiSelectField
from tvb.core.neotraits.view_model import Str
from tvb.basic.neotraits.api import Attr, Float, Int, NArray, List
from tvb.tests.framework.core.base_testcase import BaseTestCase, json
from cherrypy._cpreqbody import Part
from cherrypy.lib.httputil import HeaderMap
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
        uploaded = Part(connectivity_file, HeaderMap({}), '')

        upload_field = TraitUploadField(data_file, required_type, self.test_project.id, self.name, temporary_files)
        post_data = {'Data_Subject': 'John Doe', 'uploaded': uploaded, 'normalization': 'explicit-None-value'}

        upload_field.fill_from_post(post_data)

    def test_datatype_select_field(self, connectivity_factory):
        trait_attribute = SimulatorAdapterModel.connectivity

        datatype_select_field = TraitDataTypeSelectField(trait_attribute, self.test_project.id, self.name, None)
        connectivity = connectivity_factory(2)
        post_data = {self.name: connectivity.gid.hex}
        datatype_select_field.fill_from_post(post_data)

        assert uuid.UUID(post_data[self.name]) == datatype_select_field.data, "UUID data was not set correctly on" \
                                                                              " the TraitDataTypeSelectField"

    def test_bool_field(self):
        bool_attr = Attr(field_type=bool, default=True, label='Dummy Bool')
        bool_field = BoolField(bool_attr, self.test_project.id, self.name)

        post_data = {'dummy_name': 'on'}
        bool_field.fill_from_post(post_data)

        assert bool_field.data, "True (boolean) was not set correctly on on BoolField!"

    def test_str_field(self):
        str_attr = Str(label='Dummy Str', default='')
        str_field = StrField(str_attr, self.test_project.id, self.name)

        post_data = {'dummy_name': 'dummy_str'}
        str_field.fill_from_post(post_data)

        assert post_data[self.name] == str_field.data, "Str data was not set correctly on the StrField!"

    def test_int_field(self):
        int_attr = Int(label='Dummy Int', default=0)
        int_field = IntField(int_attr, self.test_project.id, self.name)

        int_str = '10'
        post_data = {'dummy_name': int_str}
        int_field.fill_from_post(post_data)

        assert int(post_data[self.name]) == int_field.data, "Int data was not set correctly on the IntField!"

    def test_float_field(self):
        float_attr = Float(label='Dummy Float', default=0.)
        float_field = FloatField(float_attr, self.test_project.id, self.name)

        float_str = '10.5'
        post_data = {'dummy_name': float_str}
        float_field.fill_from_post(post_data)

        assert float(post_data[self.name]) == float_field.data, "Float data was not set correctly on the FloatField!"

    def test_array_field(self):
        int_attr = NArray(label='Dummy Int', default=0)
        array_field = ArrayField(int_attr, self.test_project.id, self.name)

        array_str = '[1, 2, 3]'
        post_data = {'dummy_name': array_str}
        array_field.fill_from_post(post_data)

        result = json.loads(array_str)
        result = numpy.array(result)

        assert (result == array_field.data).all(), "Array data was not set correctly on the ArrayField!"

    def test_select_field(self):
        str_attr = Attr(field_type=str, default=True, label='Dummy Bool', choices=('1', '2', '3'))
        select_field = SelectField(str_attr, self.test_project.id, self.name)

        post_data = {'dummy_name': '1'}
        select_field.fill_from_post(post_data)

        assert post_data[self.name] == select_field.data, "Data was not set correctly on the SelectField!"

    def test_multi_select_field(self):
        list_attr = List(of=str, label='Dummy List', choices=('1', '2', '3', '4', '5'))
        multi_select_field = MultiSelectField(list_attr, self.test_project.id, self.name)

        post_data = {'dummy_name': ['2', '3', '4']}
        multi_select_field.fill_from_post(post_data)

        assert post_data[self.name] == multi_select_field.data, \
            "Data was not set correctly on the MultiSelectField!"

    def test_hidden_field(self):
        hidden_str_attr = Str(label='Dummy Str', default='')
        hidden_field = HiddenField(hidden_str_attr, self.test_project.id, self.name)

        post_data = {'dummy_name': 'Dummy Hidden Str'}
        hidden_field.fill_from_post(post_data)

        assert post_data[self.name] == hidden_field.data, "Hidden data was not set correctly on the HiddenField!"
        assert hidden_field.trait_attribute.label == '', "Hidden field's trait attributes should have empty labels!"
