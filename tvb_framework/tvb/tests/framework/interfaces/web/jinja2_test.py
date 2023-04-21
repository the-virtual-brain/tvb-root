# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

import numpy
from bs4 import BeautifulSoup

from tvb.adapters.forms.model_forms import ModelsEnum
from tvb.adapters.simulator.simulator_adapter import SimulatorAdapterForm
from tvb.adapters.forms.simulator_fragments import SimulatorModelFragment
from tvb.basic.neotraits.api import HasTraits, NArray
from tvb.basic.profile import TvbProfile
from tvb.core.adapters.abcadapter import ABCAdapter, ABCAdapterForm
from tvb.core.entities.model.model_project import User
from tvb.core.neotraits.forms import ArrayField
from tvb.interfaces.web.controllers.decorators import using_template
from tvb.interfaces.web.controllers.simulator.simulator_controller import SimulatorFragmentRenderingRules, \
    SimulatorWizzardURLs
from tvb.simulator.simulator import Simulator
from tvb.tests.framework.core.base_testcase import BaseTestCase


class DummyTrait(HasTraits):
    """ Test class with traited attributes"""

    test_array = NArray(label="State Variables range [[lo],[hi]]",
                        default=numpy.array([[-3.0, -6.0], [3.0, 6.0]]), dtype="float")


class TraitAdapterForm(ABCAdapterForm):

    def __init__(self):
        super(TraitAdapterForm, self).__init__()
        self.test_array = ArrayField(DummyTrait.test_array, name='test_array')


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


class DummyTraitAdapterForm(Jinja2Test):
    """
    Test HTML generation for a traited form.
    """

    @using_template('form_fields/form')
    def prepare_adapter_for_rendering(self):
        adapter = TraitAdapter()
        datatype = DummyTrait()

        form = adapter.get_form_class()()
        form.fill_from_trait(datatype)
        adapter.submit_form(form)
        return {'adapter_form': adapter.get_form()}

    def test_multidimensional_array(self):
        html = self.prepare_adapter_for_rendering()
        soup = BeautifulSoup(html, features="html.parser")

        array_entry = soup.find_all('input', attrs=dict(name='test_array'))
        assert len(array_entry) == 1, 'Array entry not found'
        assert array_entry[0]['value'] == "[[-3.0, -6.0], [3.0, 6.0]]", "Wrong value stored"


class TestJinja2Simulator(Jinja2Test):

    @using_template('simulator_fragment')
    def dummy_renderer(self, template_dict):
        template_dict['showOnlineHelp'] = True
        return template_dict

    def prepare_simulator_form_for_search(self, mocker, rendering_rules, form=None):
        # type: (MockerFixture, SimulatorFragmentRenderingRules, ABCAdapterForm) -> BeautifulSoup
        if form is None:
            form = SimulatorAdapterForm()
            form.fill_from_trait(Simulator())
        rendering_rules.form = form

        def _is_online_help_active(self):
            return True

        def _get_logged_user():
            return User('test', 'test', 'test')

        mocker.patch('tvb.interfaces.web.controllers.common.get_logged_user', _get_logged_user)
        mocker.patch.object(User, 'is_online_help_active', _is_online_help_active)

        html = self.dummy_renderer(rendering_rules.to_dict())
        soup = BeautifulSoup(html, features="html.parser")
        return soup

    def test_models_list(self, mocker):
        models_form = SimulatorModelFragment()
        simulator = Simulator()
        simulator.model = ModelsEnum.EPILEPTOR.instance
        models_form.fill_from_trait(simulator)

        rendering_rules = SimulatorFragmentRenderingRules(is_model_fragment=True)
        soup = self.prepare_simulator_form_for_search(mocker, rendering_rules, form=models_form)

        select_field = soup.find_all('select')
        assert len(select_field) == 1, 'Number of select inputs is different than 1'
        select_field_options = soup.find_all('option')
        assert len(select_field_options) == len(ModelsEnum), 'Number of select field options != number of models'
        select_field_choice = soup.find_all('option', selected=True)
        assert len(select_field_choice) == 1
        assert 'Epileptor' in select_field_choice[0].attrs['value']

    def test_simulator_adapter_form(self, mocker):
        rendering_rules = SimulatorFragmentRenderingRules()
        soup = self.prepare_simulator_form_for_search(mocker, rendering_rules)

        connectivity_select_field = soup.find_all('select', attrs=dict(name='connectivity'))
        assert len(connectivity_select_field) == 1, 'Number of connectivity select inputs is different than 1'
        conduction_speed_input = soup.find_all('input', attrs=dict(name='conduction_speed'))
        assert len(conduction_speed_input) == 1, 'Number of conduction speed inputs is different than 1'
        coupling_select_filed = soup.find_all('select', attrs=dict(name='coupling'))
        assert len(coupling_select_filed) == 1, 'Number of coupling select inputs is different than 1'

    def test_buttons_first_fragment(self, mocker):
        rendering_rules = SimulatorFragmentRenderingRules(form_action_url=SimulatorWizzardURLs.SET_CONNECTIVITY_URL,
                                                          is_first_fragment=True)
        soup = self.prepare_simulator_form_for_search(mocker, rendering_rules)

        all_buttons = soup.find_all('button')
        assert len(all_buttons) == 1
        assert all_buttons[0].attrs['name'] == 'next'
        hidden_buttons = soup.find_all('button', attrs=dict(style="visibility: hidden"))
        assert len(hidden_buttons) == 0

    def test_buttons_first_fragment_copy(self, mocker):
        rendering_rules = SimulatorFragmentRenderingRules(is_first_fragment=True, is_simulation_copy=True)
        soup = self.prepare_simulator_form_for_search(mocker, rendering_rules)

        all_buttons = soup.find_all('button')
        assert len(all_buttons) == 1
        assert all_buttons[0].attrs['name'] == 'next'
        hidden_buttons = soup.find_all('button', attrs=dict(style="visibility: hidden"))
        assert len(hidden_buttons) == 1

    def test_buttons_last_fragment(self, mocker):
        rendering_rules = SimulatorFragmentRenderingRules(form_action_url=SimulatorWizzardURLs.SETUP_PSE_URL,
                                                          last_form_url=SimulatorWizzardURLs.SETUP_PSE_URL,
                                                          is_launch_fragment=True)
        soup = self.prepare_simulator_form_for_search(mocker, rendering_rules)

        all_buttons = soup.find_all('button')
        assert len(all_buttons) == 3
        hidden_buttons = soup.find_all('button', attrs=dict(style="visibility: hidden"))
        assert len(hidden_buttons) == 0

    def test_buttons_last_fragment_copy(self, mocker):
        rendering_rules = SimulatorFragmentRenderingRules(is_launch_fragment=True, is_simulation_copy=True,
                                                          form_action_url=SimulatorWizzardURLs.SETUP_PSE_URL,
                                                          last_form_url=SimulatorWizzardURLs.SETUP_PSE_URL)
        soup = self.prepare_simulator_form_for_search(mocker, rendering_rules)

        all_buttons = soup.find_all('button')
        assert len(all_buttons) == 3
        hidden_buttons = soup.find_all('button', attrs=dict(style="visibility: hidden"))
        assert len(hidden_buttons) == 0

    def test_buttons_last_fragment_readonly(self, mocker):
        rendering_rules = SimulatorFragmentRenderingRules(is_launch_fragment=True, is_simulation_readonly_load=True)
        soup = self.prepare_simulator_form_for_search(mocker, rendering_rules)

        all_buttons = soup.find_all('button')
        assert len(all_buttons) == 1
        hidden_buttons = soup.find_all('button', attrs=dict(style="visibility: hidden"))
        assert len(hidden_buttons) == 1

    def test_buttons_model_fragment(self, mocker):
        rendering_rules = SimulatorFragmentRenderingRules(form_action_url='dummy_url', last_form_url='dummy_url',
                                                          is_model_fragment=True, is_surface_simulation=True)
        soup = self.prepare_simulator_form_for_search(mocker, rendering_rules)

        all_buttons = soup.find_all('button')
        assert len(all_buttons) == 5
        hidden_buttons = soup.find_all('button', attrs=dict(style="visibility: hidden"))
        assert len(hidden_buttons) == 0

    def test_buttons_middle_fragment(self, mocker):
        rendering_rules = SimulatorFragmentRenderingRules(form_action_url='dummy_url', last_form_url='dummy_url')
        soup = self.prepare_simulator_form_for_search(mocker, rendering_rules)

        all_buttons = soup.find_all('button')
        assert len(all_buttons) == 2
        assert all_buttons[0].attrs['name'] == 'previous'
        assert all_buttons[1].attrs['name'] == 'next'
        hidden_buttons = soup.find_all('button', attrs=dict(style="visibility: hidden"))
        assert len(hidden_buttons) == 0
