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

import copy
import numpy
from bs4 import BeautifulSoup
from tvb.adapters.simulator.model_forms import get_ui_name_to_model
from tvb.adapters.simulator.simulator_adapter import SimulatorAdapterForm
from tvb.adapters.simulator.simulator_fragments import SimulatorModelFragment
from tvb.basic.neotraits.api import HasTraits, NArray
from tvb.basic.profile import TvbProfile
from tvb.core.adapters.abcadapter import ABCAdapter, ABCAdapterForm
from tvb.core.neotraits.forms import ArrayField
from tvb.interfaces.web.controllers.decorators import using_template
from tvb.interfaces.web.controllers.simulator_controller import SimulatorController
from tvb.simulator.models import ModelsEnum
from tvb.simulator.simulator import Simulator
from tvb.tests.framework.core.base_testcase import BaseTestCase


class TestTrait(HasTraits):
    """ Test class with traited attributes"""

    test_array = NArray(label="State Variables range [[lo],[hi]]",
                        default=numpy.array([[-3.0, -6.0], [3.0, 6.0]]), dtype="float")


class TraitAdapterForm(ABCAdapterForm):

    def __init__(self):
        super(TraitAdapterForm, self).__init__()
        self.test_array = ArrayField(TestTrait.test_array, self, name='test_array')


class TraitAdapter(ABCAdapter):
    """
    Adapter for tests, using a traited defined interface.
    """

    def submit_form(self, form):
        self.submitted_form = form

    def get_form_class(self):
        return TraitAdapterForm

    def get_output(self):
        return []

    def launch(self, **kwargs):
        pass

    def get_required_memory_size(self, **kwargs):
        return 0

    def get_required_disk_size(self, **kwargs):
        return 0


class Jinja2Test(BaseTestCase):
    def setup_method(self):
        TvbProfile.current.web.RENDER_HTML = True

    def teardown_method(self):
        TvbProfile.current.web.RENDER_HTML = False


class TestTraitAdapterForm(Jinja2Test):
    """
    Test HTML generation for a traited form.
    """

    def prepare_adapter_for_rendering(self):
        adapter = TraitAdapter()
        datatype = TestTrait()

        form = adapter.get_form_class()()
        form.fill_from_trait(datatype)
        adapter.submit_form(form)

        return adapter

    def test_multidimensional_array(self):
        adapter = self.prepare_adapter_for_rendering()
        html = str(adapter.get_form())
        soup = BeautifulSoup(html)

        array_entry = soup.find_all('input', attrs=dict(name='_test_array'))
        assert len(array_entry) == 1, 'Array entry not found'
        assert array_entry[0]['value'] == "[[-3.0, -6.0], [3.0, 6.0]]", "Wrong value stored"


class TestJinja2Simulator(Jinja2Test):

    @using_template('simulator_fragment')
    def dummy_renderer(self, template_dict):
        return template_dict

    def get_dict_to_render(self):
        dict_to_render = copy.deepcopy(SimulatorController.dict_to_render)
        return dict_to_render

    def prepare_simulator_form_for_search(self, dict_to_render):
        sim_adapter_form = SimulatorAdapterForm(project_id=1)
        sim_adapter_form.fill_from_trait(Simulator())
        dict_to_render[SimulatorController.FORM_KEY] = sim_adapter_form

        html = self.dummy_renderer(dict_to_render)
        soup = BeautifulSoup(html)

        return soup

    def test_models_list(self):
        all_models_for_ui = get_ui_name_to_model()
        models_form = SimulatorModelFragment()
        simulator = Simulator()
        simulator.model = ModelsEnum.EPILEPTOR.get_class()()
        models_form.fill_from_trait(simulator)

        html = str(models_form)
        soup = BeautifulSoup(html)

        select_field = soup.find_all('select')
        assert len(select_field) == 1, 'Number of select inputs is different than 1'
        select_field_options = soup.find_all('option')
        assert len(select_field_options) == len(all_models_for_ui), 'Number of select field options != number of models'
        select_field_choice = soup.find_all('option', selected=True)
        assert len(select_field_choice) == 1
        assert 'Epileptor' in select_field_choice[0].attrs['value']

    def test_simulator_adapter_form(self):
        dict_to_render = self.get_dict_to_render()
        soup = self.prepare_simulator_form_for_search(dict_to_render)

        connectivity_select_field = soup.find_all('select', attrs=dict(name='_connectivity'))
        assert len(connectivity_select_field) == 1, 'Number of connectivity select inputs is different than 1'
        conduction_speed_input = soup.find_all('input', attrs=dict(name='_conduction_speed'))
        assert len(conduction_speed_input) == 1, 'Number of conduction speed inputs is different than 1'
        coupling_select_filed = soup.find_all('select', attrs=dict(name='_coupling'))
        assert len(coupling_select_filed) == 1, 'Number of coupling select inputs is different than 1'

    def test_buttons_first_fragment(self):
        dict_to_render = self.get_dict_to_render()
        dict_to_render[SimulatorController.IS_FIRST_FRAGMENT_KEY] = True

        soup = self.prepare_simulator_form_for_search(dict_to_render)

        all_buttons = soup.find_all('button')
        assert len(all_buttons) == 1
        assert all_buttons[0].attrs['name'] == 'next'
        hidden_buttons = soup.find_all('button', attrs=dict(style="visibility: hidden"))
        assert len(hidden_buttons) == 0

    def test_buttons_first_fragment_copy(self):
        dict_to_render = self.get_dict_to_render()
        dict_to_render[SimulatorController.IS_FIRST_FRAGMENT_KEY] = True
        dict_to_render[SimulatorController.IS_COPY] = True

        soup = self.prepare_simulator_form_for_search(dict_to_render)

        all_buttons = soup.find_all('button')
        assert len(all_buttons) == 1
        assert all_buttons[0].attrs['name'] == 'next'
        hidden_buttons = soup.find_all('button', attrs=dict(style="visibility: hidden"))
        assert len(hidden_buttons) == 1

    def test_buttons_last_fragment(self):
        dict_to_render = self.get_dict_to_render()
        dict_to_render[SimulatorController.IS_LAST_FRAGMENT_KEY] = True

        soup = self.prepare_simulator_form_for_search(dict_to_render)

        all_buttons = soup.find_all('button')
        assert len(all_buttons) == 3
        hidden_buttons = soup.find_all('button', attrs=dict(style="visibility: hidden"))
        assert len(hidden_buttons) == 0

    def test_buttons_last_fragment_copy(self):
        dict_to_render = self.get_dict_to_render()
        dict_to_render[SimulatorController.IS_LAST_FRAGMENT_KEY] = True
        dict_to_render[SimulatorController.IS_COPY] = True

        soup = self.prepare_simulator_form_for_search(dict_to_render)

        all_buttons = soup.find_all('button')
        assert len(all_buttons) == 4
        hidden_buttons = soup.find_all('button', attrs=dict(style="visibility: hidden"))
        assert len(hidden_buttons) == 0

    def test_buttons_last_fragment_readonly(self):
        dict_to_render = self.get_dict_to_render()
        dict_to_render[SimulatorController.IS_LAST_FRAGMENT_KEY] = True
        dict_to_render[SimulatorController.IS_LOAD] = True

        soup = self.prepare_simulator_form_for_search(dict_to_render)

        all_buttons = soup.find_all('button')
        assert len(all_buttons) == 1
        hidden_buttons = soup.find_all('button', attrs=dict(style="visibility: hidden"))
        assert len(hidden_buttons) == 1

    def test_buttons_middle_fragment(self):
        dict_to_render = self.get_dict_to_render()
        dict_to_render[SimulatorController.IS_MODEL_FRAGMENT_KEY] = True
        dict_to_render[SimulatorController.IS_SURFACE_SIMULATION_KEY] = True

        soup = self.prepare_simulator_form_for_search(dict_to_render)

        all_buttons = soup.find_all('button')
        assert len(all_buttons) == 4
        hidden_buttons = soup.find_all('button', attrs=dict(style="visibility: hidden"))
        assert len(hidden_buttons) == 0

    def test_buttons_model_fragment(self):
        dict_to_render = self.get_dict_to_render()

        soup = self.prepare_simulator_form_for_search(dict_to_render)

        all_buttons = soup.find_all('button')
        assert len(all_buttons) == 2
        assert all_buttons[0].attrs['name'] == 'previous'
        assert all_buttons[1].attrs['name'] == 'next'
        hidden_buttons = soup.find_all('button', attrs=dict(style="visibility: hidden"))
        assert len(hidden_buttons) == 0
