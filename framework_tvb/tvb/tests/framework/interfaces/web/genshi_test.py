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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

import os
import re
import numpy
import cherrypy
from tvb.tests.framework.core.base_testcase import BaseTestCase
import tvb.basic.traits as trait
import tvb.interfaces.web.templates.genshi.flow as root_html
from bs4 import BeautifulSoup
from genshi.template.loader import TemplateLoader
from tvb.core.adapters.input_tree import InputTreeManager
from tvb.basic.profile import TvbProfile
from tvb.interfaces.web.controllers import common
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.storage import dao
from tvb.core.services.flow_service import FlowService
from tvb.core.services.operation_service import OperationService, RANGE_PARAMETER_1, RANGE_PARAMETER_2
from tvb.datatypes.arrays import MappedArray
from tvb.interfaces.web.controllers.flow_controller import FlowController
from tvb.interfaces.web.entities.context_selected_adapter import SelectedAdapterContext
from tvb.tests.framework.adapters.ndimensionarrayadapter import NDimensionArrayAdapter
from tvb.tests.framework.core.factory import TestFactory



def _template2string(template_specification):
    """
    Here we use the TemplateLoader from Genshi, so we are linked to this library for comparison.
    """
    template_specification[common.KEY_SHOW_ONLINE_HELP] = False
    path_to_form = os.path.join(os.path.dirname(root_html.__file__), 'genericAdapterFormFields.html')
    loader = TemplateLoader()
    template = loader.load(path_to_form)
    stream = template.generate(**template_specification)
    return stream.render('xhtml').replace('\n', '\t').replace('\'', '"')



class TestTrait(trait.core.Type):
    """ Test class with traited attributes"""

    test_array = trait.types_mapped_light.Array(label="State Variables range [[lo],[hi]]",
                                                default=numpy.array([[-3.0, -6.0], [3.0, 6.0]]), dtype="float")

    test_dict = trait.types_basic.Dict(label="State Variable ranges [lo, hi].", default={"V": -3.0, "W": -6.0})



class TraitAdapter(ABCAdapter):
    """
    Adapter for tests, using a traited defined interface.
    """

    def get_input_tree(self):
        """
        Return a list of lists describing the interface to the simulator. This
        is used by the GUI to generate the menus and fields necessary for
        defining a simulation.
        """
        traited = TestTrait()
        traited.trait.bound = 'attributes-only'
        return traited.interface['attributes']

    def get_output(self):
        return []

    def launch(self, **kwargs):
        pass

    def get_required_memory_size(self, **kwargs):
        return 0

    def get_required_disk_size(self, **kwargs):
        return 0



class GenshiTest(BaseTestCase):
    """
    This class contains the base initialization for tests for the GENSHI TemplateLoader.
    """

    def setup_method(self):
        """
        Define a default template specification.
        """
        self.template_specification = {'submitLink': 'www.google.com',
                                       'section_name': 'test_step',
                                       'HTML': str,
                                       'errors': 'No errors',
                                       'displayControl': True,
                                       'treeSessionKey': SelectedAdapterContext.KEY_TREE_DEFAULT,
                                       common.KEY_PARAMETERS_CONFIG: False,
                                       common.KEY_CURRENT_JS_VERSION: 1}
        TvbProfile.current.web.RENDER_HTML = True


    def teardown_method(self):
        TvbProfile.current.web.RENDER_HTML = False



class TestGenthiTrait(GenshiTest):
    """
    Test HTML generation for a trait based interface.
    """

    def test_multidimensional_array(self):
        """
        Test the generation of a multi-dimensional array.
        """
        input_tree = TraitAdapter().get_input_tree()
        input_tree = InputTreeManager.prepare_param_names(input_tree)
        self.template_specification['inputList'] = input_tree
        resulted_html = _template2string(self.template_specification)
        soup = BeautifulSoup(resulted_html)
        #Find dictionary div which should be dict_+${dict_var_name}
        dict_div = soup.find_all('div', attrs=dict(id="dict_test_dict"))
        assert len(dict_div) == 1, 'Dictionary div not found'
        dict_entries = soup.find_all('input', attrs=dict(name=re.compile('^test_dict_parameters*')))
        assert len(dict_entries) == 2, 'Not all entries found'
        for i in range(2):
            if dict_entries[i]['name'] == "test_dict_parameters_W":
                assert dict_entries[0]['value'] == "-6.0", "Incorrect values"
            if dict_entries[i]['name'] == "test_dict_parameters_V":
                assert dict_entries[1]['value'] == "-3.0", "Incorrect values"
        array_entry = soup.find_all('input', attrs=dict(name='test_array'))
        assert len(array_entry) == 1, 'Array entry not found'
        assert array_entry[0]['value'] == "[[-3.0, -6.0], [3.0, 6.0]]", "Wrong value stored"



class TestGenshiSimulator(GenshiTest):
    """
    For the simulator interface, test that various fields are generated correctly.
    """
    algorithm = dao.get_algorithm_by_module('tvb.adapters.simulator.simulator_adapter', 'SimulatorAdapter')
    adapter_instance = ABCAdapter.build_adapter(algorithm)
    input_tree = adapter_instance.get_input_tree()
    input_tree = InputTreeManager.prepare_param_names(input_tree)


    def setup_method(self):
        """
        Set up any additionally needed parameters.
        """
        super(TestGenshiSimulator, self).setup_method()

        self.template_specification['inputList'] = self.input_tree
        self.template_specification['draw_hidden_ranges'] = True
        self.template_specification[common.KEY_PARAMETERS_CONFIG] = False
        resulted_html = _template2string(self.template_specification)
        self.soup = BeautifulSoup(resulted_html)
        #file = open("output.html", 'w')
        #file.write(self.soup.prettify())
        #file.close()


    def test_sub_algo_inputs(self):
        """
        Check the name of inputs generated for each sub-algorithm is done 
        properly with only one option that is not disabled
        """
        exp = re.compile('model_parameters_option_[a-zA-Z]*')
        all_inputs = self.soup.find_all('input', attrs=dict(name=exp))
        count_disabled = 0
        for one_entry in all_inputs:
            ## Replacing with IN won't work
            if one_entry.has_attr('disabled'):
                count_disabled += 1
        assert len(all_inputs) > 100, "Not enough input fields generated"
        assert count_disabled > 100, "Not enough input fields disabled"


    def test_hidden_ranger_fields(self):
        """ 
        Check that the default ranger hidden fields are generated correctly 
        """
        ranger1 = self.soup.find_all('input', attrs=dict(type="hidden", id=RANGE_PARAMETER_1))
        ranger2 = self.soup.find_all('input', attrs=dict(type="hidden", id=RANGE_PARAMETER_2))
        assert 1 == len(ranger1), "First ranger generated wrong"
        assert 1 == len(ranger2), "Second ranger generated wrong"


    def test_sub_algorithms(self):
        """
        Check that the correct number of sub-algorithms is created
        and that only one of them is not disable
        """
        fail_message = "Something went wrong with generating the sub-algorithms."
        exp = re.compile('data_model[A-Z][a-zA-Z]*')
        enabled_algo = self.soup.find_all('div', attrs=dict(id=exp, style="display:block"))
        all_algo_disabled = self.soup.find_all('div', attrs=dict(id=exp, style="display:none"))
        assert 1 == len(enabled_algo)
        assert 14 == len(all_algo_disabled)
        assert not enabled_algo[0] in all_algo_disabled, fail_message


    def test_normal_ranger(self):
        """
        Check the normal ranger generation. Only one ranger should be created
        because the minValue/ maxValue is specified only for one field. It should
        also be disabled because it is not as default.
        """
        fail_message = "Something went wrong with generating the ranger."

        exp = re.compile('data_model*')
        ranger_parent = self.soup.find_all('table', attrs={'id': exp, 'class': "ranger-div-class"})
        assert len(ranger_parent) > 100, fail_message

        range_expand = self.soup.find_all('input', attrs=dict(
            id="data_modelGeneric2dOscillatormodel_parameters_option_Generic2dOscillator_tau_RANGER_buttonExpand"))
        assert 1 == len(range_expand)


    def test_multiple_select(self):
        """
        Checks the correct creation of a multiple select component.
        """
        fail_message = "Something went wrong with creating multiple select."
        exp = re.compile('data_monitors[A-Z][a-zA-Z]*')
        all_multiple_options = self.soup.find_all('div', attrs=dict(id=exp))
        disabled_options = self.soup.find_all('div', attrs=dict(id=exp, disabled='disabled'))
        assert 9 == len(all_multiple_options), fail_message
        assert 8 == len(disabled_options), fail_message
        exp = re.compile('monitors_parameters*')
        all_multiple_params = self.soup.find_all('input', attrs=dict(name=exp))
        disabled_params = self.soup.find_all('input', attrs=dict(name=exp, disabled='disabled'))
        assert len(all_multiple_params) > 50, fail_message
        assert len(disabled_params) > 50, fail_message



class TestGenshiNDimensionArray(GenshiTest):
    """
    This class tests the generation of the component which allows
    a user to reduce the dimension of an array.
    """


    def setup_method(self):
        """
        Set up any additionally needed parameters.
        """
        self.clean_database()
        super(TestGenshiNDimensionArray, self).setup_method()
        self.test_user = TestFactory.create_user()
        self.test_project = TestFactory.create_project(self.test_user)
        self.operation = TestFactory.create_operation(test_user=self.test_user, test_project=self.test_project)


    def teardown_method(self):
        """
        Reset the database when test is done.
        """
        super(TestGenshiNDimensionArray, self).teardown_method()
        self.clean_database()


    def test_reduce_dimension_component(self):
        """
        Tests the generation of the component which allows the user
        to select one dimension from a multi dimension array
        """
        flow_service = FlowService()
        array_count = self.count_all_entities(MappedArray)
        assert 0 == array_count, "Expected to find no data"
        adapter_instance = NDimensionArrayAdapter()
        PARAMS = {}
        OperationService().initiate_prelaunch(self.operation, adapter_instance, {}, **PARAMS)
        inserted_arrays, array_count = flow_service.get_available_datatypes(self.test_project.id, MappedArray)
        assert 1 == array_count, "Problems when inserting data"

        algorithm = flow_service.get_algorithm_by_module_and_class(
            'tvb.tests.framework.adapters.ndimensionarrayadapter', 'NDimensionArrayAdapter')
        interface = flow_service.prepare_adapter(self.test_project.id, algorithm)
        self.template_specification['inputList'] = interface
        resulted_html = _template2string(self.template_specification)
        self.soup = BeautifulSoup(resulted_html)

        found_divs = self.soup.find_all('p', attrs=dict(id="dimensionsDiv_input_data"))
        assert len(found_divs) == 1, "Data generated incorrect"

        gid = inserted_arrays[0][2]
        cherrypy.session = {'user': self.test_user}
        entity = dao.get_datatype_by_gid(gid)
        component_content = FlowController().gettemplatefordimensionselect(gid, "input_data")
        self.soup = BeautifulSoup(component_content)

        #check dimensions
        found_selects_0 = self.soup.find_all('select', attrs=dict(id="dimId_input_data_dimensions_0"))
        found_selects_1 = self.soup.find_all('select', attrs=dict(id="dimId_input_data_dimensions_1"))
        found_selects_2 = self.soup.find_all('select', attrs=dict(id="dimId_input_data_dimensions_2"))
        assert len(found_selects_0) == 1, "select not found"
        assert len(found_selects_1) == 1, "select not found"
        assert len(found_selects_2) == 1, "select not found"

        #check the aggregation functions selects
        agg_selects_0 = self.soup.find_all('select', attrs=dict(id="funcId_input_data_dimensions_0"))
        agg_selects_1 = self.soup.find_all('select', attrs=dict(id="funcId_input_data_dimensions_1"))
        agg_selects_2 = self.soup.find_all('select', attrs=dict(id="funcId_input_data_dimensions_2"))
        assert len(agg_selects_0), 1 == "incorrect first dim"
        assert len(agg_selects_1), 1 == "incorrect second dim"
        assert len(agg_selects_2), 1 == "incorrect third dim."

        data_shape = entity.shape
        assert len(data_shape) == 3, "Shape of the array is incorrect"
        for i in range(data_shape[0]):
            options = self.soup.find_all('option', attrs=dict(value=gid + "_0_" + str(i)))
            assert len(options) == 1, "Generated option is incorrect"
            assert options[0].text == "Time " + str(i), "The label of the option is not correct"
            assert options[0].parent["name"] == "input_data_dimensions_0"
        for i in range(data_shape[1]):
            options = self.soup.find_all('option', attrs=dict(value=gid + "_1_" + str(i)))
            assert len(options) == 1, "Generated option is incorrect"
            assert options[0].text == "Channel " + str(i), "Option's label incorrect"
            assert options[0].parent["name"] == "input_data_dimensions_1", "incorrect parent"
        for i in range(data_shape[2]):
            options = self.soup.find_all('option', attrs=dict(value=gid + "_2_" + str(i)))
            assert len(options) == 1, "Generated option is incorrect"
            assert options[0].text == "Line " + str(i), "The label of the option is not correct"
            assert options[0].parent["name"] == "input_data_dimensions_2"

        #check the expected hidden fields
        expected_shape = self.soup.find_all('input', attrs=dict(id="input_data_expected_shape"))
        assert len(expected_shape) == 1, "The generated option is not correct"
        assert expected_shape[0]["value"] == "expected_shape_", "The generated option is not correct"
        input_hidden_op = self.soup.find_all('input', attrs=dict(id="input_data_operations"))
        assert len(input_hidden_op) == 1, "The generated option is not correct"
        assert input_hidden_op[0]["value"] == "operations_", "The generated option is not correct"
        input_hidden_dim = self.soup.find_all('input', attrs=dict(id="input_data_expected_dim"))
        assert len(input_hidden_dim) == 1, "The generated option is not correct"
        assert input_hidden_dim[0]["value"] == "requiredDim_1", "The generated option is not correct"
        input_hidden_shape = self.soup.find_all('input', attrs=dict(id="input_data_array_shape"))
        assert len(input_hidden_shape) == 1, "The generated option is not correct"
        assert input_hidden_shape[0]["value"] == "[5, 1, 3]", "The generated option is not correct"

        #check only the first option from the aggregations functions selects
        options = self.soup.find_all('option', attrs=dict(value="func_none"))
        assert len(options) == 3, "The generated option is not correct"
