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

import json
import os
import re
from unittest.mock import patch
from uuid import UUID, uuid4

import cherrypy
import pytest
from cherrypy.lib.sessions import RamSession

from tvb.adapters.forms.model_forms import ModelsEnum
from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.simulator.view_model import HybridSimulatorAdapterModel
from tvb.core.entities.model.model_burst import Dynamic
from tvb.simulator.integrators import HeunDeterministic
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.interfaces.web.controllers.common import KEY_PROJECT, KEY_USER
from tvb.interfaces.web.controllers.simulator.hybrid_simulator_controller import HybridSimulatorController, \
    HybridSimulatorURLs
from tvb.interfaces.web.controllers.simulator.simulator_controller import SimulatorController
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseTransactionalControllerTest


class TestHybridSimulatorController(BaseTransactionalControllerTest):

    def transactional_setup_method(self):
        self.hybrid_controller = HybridSimulatorController()
        self.simulator_controller = SimulatorController()
        self.test_user = TestFactory.create_user('HybridSimulatorController_User')
        self.test_project = TestFactory.create_project(self.test_user, "HybridSimulatorController_Project")
        self.connectivity = TestFactory.import_zip_connectivity(self.test_user, self.test_project)

        self.session_stored_hybrid_simulator = HybridSimulatorAdapterModel()
        self.session_stored_hybrid_simulator.connectivity = UUID(self.connectivity.gid)

        self.sess_mock = RamSession()
        self.sess_mock[KEY_USER] = self.test_user
        self.sess_mock[KEY_PROJECT] = self.test_project

    def _configured_hybrid_simulator(self):
        """
        Put a Hybrid Simulator configuration with the imported Connectivity in session and open the
        Subnetworks wizard step for it.

        Opening a step is a GET: posting it submits the grouping and moves on to the dynamics step. The
        request method is restored afterwards, since cherrypy.request is shared and a caller may have set
        it for what it does next.
        """
        self.hybrid_controller.context.set_hybrid_simulator(self.session_stored_hybrid_simulator)
        previous_method = getattr(cherrypy.request, 'method', None)
        cherrypy.request.method = "GET"
        try:
            return self.hybrid_controller.set_subnetworks()
        finally:
            if previous_method is not None:
                cherrypy.request.method = previous_method

    def _saved_subnetworks(self):
        """The grouping stored on the Hybrid Simulator configuration, as the wizard step lists it."""
        return self.hybrid_controller.context.hybrid_simulator.subnetworks

    def _draft_subnetworks(self):
        """The grouping currently being edited on the board in the third column."""
        return self.hybrid_controller.context.subnetworks_draft

    @staticmethod
    def _names(subnetworks):
        return [subnetwork.name for subnetwork in subnetworks]

    @staticmethod
    def _assert_valid_partition(subnetworks, number_of_regions):
        """
        Every Connectivity node must show up in exactly one Subnetwork, with its original index.
        Accepts both the view models and their JSON ready form.
        """
        assigned = []
        for subnetwork in subnetworks:
            node_indices = subnetwork['node_indices'] if isinstance(subnetwork, dict) else subnetwork.node_indices
            assigned.extend(node_indices)

        assert sorted(assigned) == list(range(number_of_regions))
        assert len(assigned) == len(set(assigned))

    def test_index(self):
        cherrypy.request.method = "GET"

        with patch('cherrypy.session', self.sess_mock, create=True), \
                patch('tvb.interfaces.web.controllers.decorators.TvbProfile.is_first_run', return_value=False):
            result_dict = self.hybrid_controller.index()

        assert result_dict['mainContent'] == 'burst/main_hybrid_simulator'
        assert result_dict['subsection_name'] == 'hybrid'
        assert result_dict['renderer'].form_action_url == HybridSimulatorURLs.SET_CONNECTIVITY_URL
        assert result_dict['renderer'].fragment_title == "Connectivity"
        assert result_dict['burst_list'] == []
        assert result_dict['last_form_url'] == HybridSimulatorURLs.SET_CONNECTIVITY_URL
        assert not result_dict['errors']

    def _index(self):
        cherrypy.request.method = "GET"
        with patch('cherrypy.session', self.sess_mock, create=True), \
                patch('tvb.interfaces.web.controllers.decorators.TvbProfile.is_first_run', return_value=False):
            return self.hybrid_controller.index()

    def test_index_after_reload_points_at_the_step_being_configured(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()

        result_dict = self._index()

        # the page renders the first step, the client then brings back the rest up to this one
        assert result_dict['renderer'].form_action_url == HybridSimulatorURLs.SET_CONNECTIVITY_URL
        assert result_dict['last_form_url'] == HybridSimulatorURLs.SET_SUBNETWORKS_URL

    def test_stepping_back_is_remembered_for_the_next_reload(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.set_fragment_url(url=HybridSimulatorURLs.SET_CONNECTIVITY_URL)

        assert self._index()['last_form_url'] == HybridSimulatorURLs.SET_CONNECTIVITY_URL

    def test_set_fragment_url_ignores_unknown_steps(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.set_fragment_url(url="/burst/set_model")
            self.hybrid_controller.set_fragment_url()

        assert self._index()['last_form_url'] == HybridSimulatorURLs.SET_SUBNETWORKS_URL

    @pytest.mark.parametrize('previous_step', [HybridSimulatorURLs.SET_PROJECTIONS_URL,
                                               HybridSimulatorURLs.SET_MONITORS_URL,
                                               HybridSimulatorURLs.SET_MONITOR_PARAMS_URL + '/BoldViewModel'])
    def test_stepping_back_to_a_later_step_is_remembered(self, previous_step):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.set_fragment_url(url=previous_step)

        assert self._index()['last_form_url'] == previous_step

    @pytest.mark.parametrize('previous_step', [HybridSimulatorURLs.SET_SUBNETWORK_DYNAMICS_URL,
                                               HybridSimulatorURLs.SAVE_SUBNETWORK_DYNAMICS_URL])
    def test_index_after_reload_on_the_dynamics_points_at_the_editor(self, previous_step):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            # the dynamics step is remembered while its editor is open, and Previous from the
            # Projections goes back to the row of actions closing that editor
            self.hybrid_controller.set_fragment_url(url=previous_step)

        assert self._index()['last_form_url'] == HybridSimulatorURLs.SUBNETWORK_EDITOR_URL

    def _get_step(self, url):
        """Request a wizard step by its url, the way the client does when it follows a step's next url."""
        cherrypy.request.method = "GET"
        for monitor_url, method in ((HybridSimulatorURLs.SET_MONITOR_PARAMS_URL, 'set_monitor_params'),
                                    (HybridSimulatorURLs.SET_MONITOR_EQUATION_URL, 'set_monitor_equation')):
            if url.startswith(monitor_url + '/'):
                return getattr(self.hybrid_controller, method)(url[len(monitor_url) + 1:])
        method = {HybridSimulatorURLs.SET_SUBNETWORKS_URL: 'set_subnetworks',
                  HybridSimulatorURLs.SET_SUBNETWORK_DYNAMICS_URL: 'set_subnetwork_dynamics',
                  HybridSimulatorURLs.SUBNETWORK_EDITOR_URL: 'subnetwork_editor',
                  HybridSimulatorURLs.SET_PROJECTIONS_URL: 'set_projections',
                  HybridSimulatorURLs.SET_MONITORS_URL: 'set_monitors',
                  HybridSimulatorURLs.SET_SIMULATION_SUMMARY_URL: 'set_simulation_summary'}[url]
        return getattr(self.hybrid_controller, method)()

    def test_each_step_names_the_one_after_it_up_to_the_closing_step(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_subnetwork()
            self._choose_monitors(['Temporal average', 'BOLD'], simulation_length='6000.0')

            # what the client does after a reload: follow each step's next url from the first step
            steps = []
            renderer = self._index()['renderer']
            while renderer.next_step_url is not None:
                renderer = self._get_step(renderer.next_step_url)['renderer']
                steps.append(renderer.form_action_url)

        assert steps == [HybridSimulatorURLs.SET_SUBNETWORKS_URL, HybridSimulatorURLs.SET_SUBNETWORK_DYNAMICS_URL,
                         HybridSimulatorURLs.SUBNETWORK_EDITOR_URL, HybridSimulatorURLs.SET_PROJECTIONS_URL,
                         HybridSimulatorURLs.SET_MONITORS_URL,
                         HybridSimulatorURLs.SET_MONITOR_PARAMS_URL + '/TemporalAverageViewModel',
                         HybridSimulatorURLs.SET_MONITOR_PARAMS_URL + '/BoldViewModel',
                         HybridSimulatorURLs.SET_MONITOR_EQUATION_URL + '/BoldViewModel',
                         HybridSimulatorURLs.SET_SIMULATION_SUMMARY_URL]

    def test_the_monitors_step_names_the_summary_when_no_monitor_has_parameters(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_subnetwork()
            self._choose_monitors(['Raw recording'])
            renderer = self._get_step(HybridSimulatorURLs.SET_MONITORS_URL)['renderer']

        assert renderer.next_step_url == HybridSimulatorURLs.SET_SIMULATION_SUMMARY_URL

    def test_connectivity_step_configures_nothing_in_the_third_column(self):
        cherrypy.request.method = "GET"

        with patch('cherrypy.session', self.sess_mock, create=True):
            rendering_rules = self.hybrid_controller.set_connectivity()

        # nothing declared here is what empties the third column when stepping back to this step
        assert rendering_rules['renderer'].context_form_url is None

    def test_set_connectivity(self):
        cherrypy.request.method = "POST"
        self.sess_mock['connectivity'] = self.connectivity.gid

        with patch('cherrypy.session', self.sess_mock, create=True):
            self.hybrid_controller.context.set_hybrid_simulator(self.session_stored_hybrid_simulator)
            rendering_rules = self.hybrid_controller.set_connectivity(**self.sess_mock._data)
            hybrid_simulator = self.hybrid_controller.context.hybrid_simulator

        assert hybrid_simulator.connectivity.hex == self.connectivity.gid
        assert rendering_rules['renderer'].form_action_url == HybridSimulatorURLs.SET_SUBNETWORKS_URL
        assert rendering_rules['renderer'].previous_form_action_url == HybridSimulatorURLs.SET_CONNECTIVITY_URL
        assert rendering_rules['renderer'].is_subnetworks_summary_fragment

    def test_set_connectivity_missing_value_stays_on_connectivity_fragment(self):
        cherrypy.request.method = "POST"

        with patch('cherrypy.session', self.sess_mock, create=True):
            self.hybrid_controller.context.set_hybrid_simulator(HybridSimulatorAdapterModel())
            rendering_rules = self.hybrid_controller.set_connectivity()

        renderer = rendering_rules['renderer']
        assert renderer.form_action_url == HybridSimulatorURLs.SET_CONNECTIVITY_URL
        assert renderer.is_first_fragment
        assert renderer.form.connectivity.errors

    def test_set_subnetworks_creates_one_subnetwork_with_all_regions(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            rendering_rules = self._configured_hybrid_simulator()
            saved = self._saved_subnetworks()

        renderer = rendering_rules['renderer']
        assert renderer.is_subnetworks_summary_fragment
        assert renderer.form_action_url == HybridSimulatorURLs.SET_SUBNETWORKS_URL
        assert len(renderer.region_labels) == self.connectivity.number_of_regions

        assert len(saved) == 1
        assert saved[0].name == 'Subnetwork A'
        assert list(saved[0].node_indices) == list(range(self.connectivity.number_of_regions))
        self._assert_valid_partition(saved, self.connectivity.number_of_regions)

    def test_subnetworks_step_configures_the_board_in_the_third_column(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            rendering_rules = self._configured_hybrid_simulator()

        renderer = rendering_rules['renderer']
        # this is what makes the third column show the grouping board for this step
        assert renderer.context_form_url == HybridSimulatorURLs.CONFIGURE_SUBNETWORKS_URL
        assert renderer.context_title == "Subnetworks"
        assert renderer.include_previous_button
        assert renderer.previous_button_label == 'Previous'
        # the grouping on the board is the saved one, so the dynamics step can be opened over it
        assert renderer.next_button_enabled
        # nothing was edited yet, so the board holds exactly what the step lists
        assert not renderer.is_modified

    def test_configure_subnetworks_renders_the_board_for_the_current_grouping(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            rendering_rules = self.hybrid_controller.configure_subnetworks()

        renderer = rendering_rules['renderer']
        assert renderer.load_error is None
        assert len(renderer.region_labels) == self.connectivity.number_of_regions
        assert self._names(renderer.subnetworks) == ['Subnetwork A']
        assert not renderer.is_modified

        rendered_state = json.loads(renderer.subnetworks_json)
        assert rendered_state['is_modified'] is False
        self._assert_valid_partition(rendered_state['subnetworks'], self.connectivity.number_of_regions)

    def test_configure_subnetworks_without_connectivity_reports_the_reason(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self.hybrid_controller.context.set_hybrid_simulator(HybridSimulatorAdapterModel())
            rendering_rules = self.hybrid_controller.configure_subnetworks()

        # the third column says why it can not configure anything, the wizard is left alone
        assert rendering_rules['renderer'].load_error
        assert rendering_rules['renderer'].subnetworks is None

    def test_the_rendered_step_points_the_third_column_at_the_board(self):
        """
        What actually goes on the wire: the wizard step carries the attribute hybrid_simulator.js reads
        to fill the third column, and that url answers with the board. Rendering is normally off during
        tests, so a broken template would otherwise only show up in the browser.
        """
        cherrypy.request.method = "GET"
        with patch.object(TvbProfile.current.web, 'RENDER_HTML', True), \
                patch('cherrypy.session', self.sess_mock, create=True):
            self.hybrid_controller.context.set_hybrid_simulator(self.session_stored_hybrid_simulator)
            step_html = self.hybrid_controller.set_subnetworks()
            board_html = self.hybrid_controller.configure_subnetworks()

        assert 'data-hybrid-context-url="{}"'.format(
            HybridSimulatorURLs.CONFIGURE_SUBNETWORKS_URL) in step_html
        assert 'data-hybrid-context-title="Subnetworks"' in step_html
        # the removed page and the action that used to open it must not come back
        assert 'Configure Subnetworks</button>' not in step_html

        assert 'id="hybrid-subnetworks-board"' in board_html
        assert 'HYBRID_SUBNETWORKS.init(' in board_html
        assert 'hybridSaveSubnetworks()' in board_html

    def test_connectivity_step_declares_no_third_column_configuration_when_rendered(self):
        cherrypy.request.method = "GET"
        with patch.object(TvbProfile.current.web, 'RENDER_HTML', True), \
                patch('cherrypy.session', self.sess_mock, create=True):
            step_html = self.hybrid_controller.set_connectivity()

        # an empty url is what makes hybrid_simulator.js clear the third column on this step
        assert 'data-hybrid-context-url=""' in step_html

    def test_set_subnetworks_without_connectivity_returns_to_first_fragment(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self.hybrid_controller.context.set_hybrid_simulator(HybridSimulatorAdapterModel())
            rendering_rules = self.hybrid_controller.set_subnetworks()

        assert rendering_rules['renderer'].form_action_url == HybridSimulatorURLs.SET_CONNECTIVITY_URL
        assert rendering_rules['renderer'].is_first_fragment

    def _select_unavailable_connectivity(self):
        """Keep in session a Connectivity GID which is no longer found in the database."""
        hybrid_simulator = HybridSimulatorAdapterModel()
        hybrid_simulator.connectivity = uuid4()
        self.hybrid_controller.context.set_hybrid_simulator(hybrid_simulator)

    def test_unavailable_connectivity_returns_to_first_fragment(self):
        cherrypy.request.method = "GET"

        with patch('cherrypy.session', self.sess_mock, create=True):
            self._select_unavailable_connectivity()
            subnetworks_step = self.hybrid_controller.set_subnetworks()
            saved_step = self.hybrid_controller.save_subnetworks()
            board = self.hybrid_controller.configure_subnetworks()
            state = json.loads(self.hybrid_controller.add_subnetwork())

        for rendering_rules in (subnetworks_step, saved_step):
            assert rendering_rules['renderer'].form_action_url == HybridSimulatorURLs.SET_CONNECTIVITY_URL
            assert rendering_rules['renderer'].is_first_fragment
        assert board['renderer'].load_error
        assert board['renderer'].subnetworks is None
        assert state['status'] == 'error'
        assert state['subnetworks'] == []

    def test_connectivity_without_data_file_returns_to_first_fragment(self):
        cherrypy.request.method = "GET"
        os.remove(h5.path_for_stored_index(self.connectivity))

        with patch('cherrypy.session', self.sess_mock, create=True):
            rendering_rules = self._configured_hybrid_simulator()
            state = json.loads(self.hybrid_controller.add_subnetwork())

        assert rendering_rules['renderer'].form_action_url == HybridSimulatorURLs.SET_CONNECTIVITY_URL
        assert rendering_rules['renderer'].is_first_fragment
        assert state['status'] == 'error'

    # ---------------------------------------------------------------- editing the board

    def test_editing_the_board_leaves_the_saved_configuration_alone(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.add_subnetwork()
            result = json.loads(self.hybrid_controller.move_regions(subnetwork_index='1',
                                                                    node_indices=json.dumps([1, 3])))
            rendering_rules = self.hybrid_controller.set_subnetworks()
            saved, draft = self._saved_subnetworks(), self._draft_subnetworks()

        # the board holds the new grouping...
        assert result['status'] == 'ok'
        assert result['is_modified'] is True
        assert result['subnetworks'][1]['node_indices'] == [1, 3]
        assert self._names(draft) == ['Subnetwork A', 'Subnetwork B']

        # ...while the wizard step still lists the saved one
        assert self._names(saved) == ['Subnetwork A']
        assert list(saved[0].node_indices) == list(range(self.connectivity.number_of_regions))
        assert [row['name'] for row in rendering_rules['renderer'].subnetwork_rows] == ['Subnetwork A']
        assert rendering_rules['renderer'].is_modified

    def test_save_updates_the_summary_with_the_grouping_from_the_board(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.add_subnetwork()
            self.hybrid_controller.move_regions(subnetwork_index='1', node_indices=json.dumps([1, 3]))
            rendering_rules = self.hybrid_controller.save_subnetworks()
            saved = self._saved_subnetworks()

        renderer = rendering_rules['renderer']
        assert renderer.form_action_url == HybridSimulatorURLs.SET_SUBNETWORKS_URL
        assert renderer.is_subnetworks_summary_fragment
        # saving is what makes the board and the summary agree again
        assert not renderer.is_modified

        rows = renderer.subnetwork_rows
        assert [row['name'] for row in rows] == ['Subnetwork A', 'Subnetwork B']
        assert rows[0]['count'] == self.connectivity.number_of_regions - 2
        assert rows[1]['count'] == 2
        # the rows keep the original Connectivity indices next to the labels
        assert [region['index'] for region in rows[1]['regions']] == [1, 3]
        assert all(region['label'] for region in rows[1]['regions'])
        assert 1 not in [region['index'] for region in rows[0]['regions']]

        assert list(saved[1].node_indices) == [1, 3]
        self._assert_valid_partition(saved, self.connectivity.number_of_regions)

    def test_the_board_keeps_empty_subnetworks_and_saving_discards_them(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            # two extra Subnetworks, only one of which is ever filled
            self.hybrid_controller.add_subnetwork()
            self.hybrid_controller.add_subnetwork()
            self.hybrid_controller.move_regions(subnetwork_index='1', node_indices=json.dumps([1, 3]))

            board = self.hybrid_controller.configure_subnetworks()
            rendering_rules = self.hybrid_controller.save_subnetworks()
            saved, draft = self._saved_subnetworks(), self._draft_subnetworks()

        # on the board an empty Subnetwork must survive, it is the drop target being prepared
        assert self._names(board['renderer'].subnetworks) == ['Subnetwork A', 'Subnetwork B', 'Subnetwork C']
        assert list(board['renderer'].subnetworks[2].node_indices) == []

        # saving is where the ones left empty are dropped, from the configuration and from the board
        assert [row['name'] for row in rendering_rules['renderer'].subnetwork_rows] == ['Subnetwork A',
                                                                                        'Subnetwork B']
        assert self._names(saved) == ['Subnetwork A', 'Subnetwork B']
        assert self._names(draft) == ['Subnetwork A', 'Subnetwork B']
        self._assert_valid_partition(saved, self.connectivity.number_of_regions)

    def test_an_empty_subnetwork_alone_is_not_reported_as_an_unsaved_change(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            result = json.loads(self.hybrid_controller.add_subnetwork())

        # saving would discard it again, so there is nothing pending to save yet
        assert result['status'] == 'ok'
        assert result['is_modified'] is False

    def test_a_single_empty_subnetwork_is_never_discarded(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self.hybrid_controller.context.set_hybrid_simulator(self.session_stored_hybrid_simulator)
            rendering_rules = self.hybrid_controller.set_subnetworks()

        # the default configuration holds every region, so nothing is dropped and one always remains
        assert len(rendering_rules['renderer'].subnetwork_rows) == 1

    def test_save_without_connectivity_returns_to_first_fragment(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self.hybrid_controller.context.set_hybrid_simulator(HybridSimulatorAdapterModel())
            rendering_rules = self.hybrid_controller.save_subnetworks()

        assert rendering_rules['renderer'].is_first_fragment

    def test_add_subnetwork(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            result = json.loads(self.hybrid_controller.add_subnetwork())

        assert result['status'] == 'ok'
        assert [subnetwork['name'] for subnetwork in result['subnetworks']] == ['Subnetwork A', 'Subnetwork B']
        assert result['subnetworks'][1]['node_indices'] == []
        self._assert_valid_partition(result['subnetworks'], self.connectivity.number_of_regions)

    def test_rename_subnetwork(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            result = json.loads(self.hybrid_controller.rename_subnetwork(subnetwork_index='0', name=' Cortex '))
            saved_before, draft = self._saved_subnetworks(), self._draft_subnetworks()

            self.hybrid_controller.save_subnetworks()
            saved_after = self._saved_subnetworks()

        assert result['status'] == 'ok'
        assert result['subnetworks'][0]['name'] == 'Cortex'
        assert draft[0].name == 'Cortex'
        # renaming on the board does not reach the configuration before it is saved
        assert saved_before[0].name == 'Subnetwork A'
        assert saved_after[0].name == 'Cortex'

    def test_rename_subnetwork_rejects_empty_and_duplicated_names(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.add_subnetwork()

            empty_name = json.loads(self.hybrid_controller.rename_subnetwork(subnetwork_index='1', name='  '))
            duplicated = json.loads(
                self.hybrid_controller.rename_subnetwork(subnetwork_index='1', name='Subnetwork A'))
            missing_subnetwork = json.loads(
                self.hybrid_controller.rename_subnetwork(subnetwork_index='7', name='Whatever'))

        assert empty_name['status'] == 'error'
        assert duplicated['status'] == 'error'
        assert missing_subnetwork['status'] == 'error'
        assert [subnetwork['name'] for subnetwork in duplicated['subnetworks']] == ['Subnetwork A', 'Subnetwork B']

    def test_move_regions_keeps_original_connectivity_indices(self):
        moved = [1, 3, 4]

        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.add_subnetwork()
            result = json.loads(self.hybrid_controller.move_regions(subnetwork_index='1',
                                                                    node_indices=json.dumps(moved)))
            draft = self._draft_subnetworks()

            self.hybrid_controller.save_subnetworks()
            saved = self._saved_subnetworks()

        assert result['status'] == 'ok'
        assert result['subnetworks'][1]['node_indices'] == moved
        for node_index in moved:
            assert node_index not in result['subnetworks'][0]['node_indices']
        self._assert_valid_partition(result['subnetworks'], self.connectivity.number_of_regions)

        # the same grouping is on the board and, after saving, in the configuration, both still using
        # the original Connectivity indices
        assert list(draft[1].node_indices) == moved
        assert list(saved[1].node_indices) == moved

    def test_move_regions_back_leaves_no_region_unassigned(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.add_subnetwork()
            self.hybrid_controller.move_regions(subnetwork_index='1', node_indices=json.dumps([0, 2]))
            result = json.loads(self.hybrid_controller.move_regions(subnetwork_index='0',
                                                                    node_indices=json.dumps([2])))

        assert result['status'] == 'ok'
        assert result['subnetworks'][1]['node_indices'] == [0]
        assert 2 in result['subnetworks'][0]['node_indices']
        self._assert_valid_partition(result['subnetworks'], self.connectivity.number_of_regions)

    def test_move_regions_rejects_invalid_input(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.add_subnetwork()

            no_selection = json.loads(self.hybrid_controller.move_regions(subnetwork_index='1',
                                                                          node_indices=json.dumps([])))
            unknown_node = json.loads(self.hybrid_controller.move_regions(
                subnetwork_index='1', node_indices=json.dumps([self.connectivity.number_of_regions])))
            unknown_subnetwork = json.loads(self.hybrid_controller.move_regions(subnetwork_index='5',
                                                                                node_indices=json.dumps([0])))

        for result in (no_selection, unknown_node, unknown_subnetwork):
            assert result['status'] == 'error'
            self._assert_valid_partition(result['subnetworks'], self.connectivity.number_of_regions)

    def test_remove_subnetwork_reassigns_its_regions(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.add_subnetwork()
            self.hybrid_controller.move_regions(subnetwork_index='1', node_indices=json.dumps([0, 1]))
            result = json.loads(self.hybrid_controller.remove_subnetwork(subnetwork_index='1'))

        assert result['status'] == 'ok'
        assert len(result['subnetworks']) == 1
        self._assert_valid_partition(result['subnetworks'], self.connectivity.number_of_regions)

    def test_remove_last_subnetwork_is_prevented(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            result = json.loads(self.hybrid_controller.remove_subnetwork(subnetwork_index='0'))

        assert result['status'] == 'error'
        assert len(result['subnetworks']) == 1
        self._assert_valid_partition(result['subnetworks'], self.connectivity.number_of_regions)

    # ---------------------------------------------------------------- Subnetwork dynamics

    MODEL_PARAMS = {'tau': '[1.0]', 'I': '[0.0]', 'a': '[-2.0]', 'b': '[-10.0]', 'c': '[0.0]',
                    'd': '[0.02]', 'e': '[3.0]', 'f': '[1.0]', 'g': '[0.0]', 'alpha': '[1.0]',
                    'beta': '[1.0]', 'gamma': '[1.0]', 'variables_of_interest': 'V'}

    def _open_dynamics_step(self, **data):
        """Press Next on the Subnetworks step, which opens the dynamics step over a saved grouping."""
        cherrypy.request.method = "POST"
        return self.hybrid_controller.set_subnetworks(**data)

    def _open_subnetwork_editor(self, **data):
        """
        Press Next on the Subnetwork dynamics step, which applies the shared dt and opens the editor of
        the selected Subnetwork under it.
        """
        cherrypy.request.method = "POST"
        return self.hybrid_controller.set_subnetwork_dynamics(**data)

    def _dynamics_step(self):
        """Re-read the Subnetwork dynamics step, which carries the summary and the Subnetwork selector."""
        previous_method = getattr(cherrypy.request, 'method', None)
        cherrypy.request.method = "GET"
        try:
            return self.hybrid_controller.set_subnetwork_dynamics()
        finally:
            if previous_method is not None:
                cherrypy.request.method = previous_method

    def _dynamics_draft(self):
        """The Model/Integrator configuration currently being edited, keyed by Subnetwork identifier."""
        return self.hybrid_controller.context.dynamics_draft

    def _two_subnetworks(self):
        """A saved grouping of two Subnetworks, so per Subnetwork behaviour can be told apart."""
        self._configured_hybrid_simulator()
        self.hybrid_controller.add_subnetwork()
        self.hybrid_controller.move_regions(subnetwork_index='1', node_indices=json.dumps([0, 1]))
        self.hybrid_controller.save_subnetworks()
        return self._saved_subnetworks()

    def test_next_on_the_subnetworks_step_opens_the_dynamics_step(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            rendering_rules = self._open_dynamics_step()

        renderer = rendering_rules['renderer']
        assert renderer.form_action_url == HybridSimulatorURLs.SET_SUBNETWORK_DYNAMICS_URL
        assert renderer.previous_form_action_url == HybridSimulatorURLs.SET_SUBNETWORKS_URL
        assert renderer.is_dynamics_summary_fragment
        # the Model and Integrator are configured in this same column, under this step, so the third
        # column configures nothing here and goes back to showing the Results
        assert renderer.context_form_url is None
        # and this step offers the Subnetwork selector for the steps stacked under it
        assert len(renderer.subnetwork_choices) == 1
        assert renderer.subnetwork_choices[0]['is_selected']

    def test_next_on_the_dynamics_step_opens_the_editor_of_the_selected_subnetwork(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            rendering_rules = self._open_subnetwork_editor()
            name = self._saved_subnetworks()[0].name

        renderer = rendering_rules['renderer']
        # the editor of the selected Subnetwork opens right under the dynamics step, not its first step
        assert renderer.is_subnetwork_editor
        assert renderer.editor_subnetwork_name == name
        assert self._editor_urls(rendering_rules)[0] == HybridSimulatorURLs.SET_SUBNETWORK_MODEL_URL

    def test_next_is_refused_while_the_grouping_is_unsaved(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.add_subnetwork()
            self.hybrid_controller.move_regions(subnetwork_index='1', node_indices=json.dumps([0, 1]))
            rendering_rules = self._open_dynamics_step()

        renderer = rendering_rules['renderer']
        # the dynamics step configures the saved Subnetworks, so it stays shut over an unsaved grouping
        assert renderer.form_action_url == HybridSimulatorURLs.SET_SUBNETWORKS_URL
        assert renderer.is_subnetworks_summary_fragment
        assert not renderer.next_button_enabled

    def test_every_subnetwork_is_seeded_with_a_model_and_an_integrator(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._two_subnetworks()
            self._open_dynamics_step()
            draft = self._dynamics_draft()
            saved = self._saved_subnetworks()

        assert len(draft) == 2
        for subnetwork in saved:
            assert subnetwork.dynamics.model is not None
            assert subnetwork.dynamics.integrator is not None
            assert draft[subnetwork.id].model is not None
            assert draft[subnetwork.id].integrator is not None

    def _edit_step_size(self, dt):
        """Change the integration step size on the Integrator of the selected Subnetwork."""
        cherrypy.request.method = "POST"
        return self.hybrid_controller.set_subnetwork_integrator_params(dt=dt)

    def _integrator_params_dt(self):
        """The step size the Integrator section of the selected Subnetwork shows."""
        cherrypy.request.method = "GET"
        editor = self.hybrid_controller.subnetwork_editor()
        section = editor['renderer'].editor_sections[3]
        assert section.form_action_url == HybridSimulatorURLs.SET_SUBNETWORK_INTEGRATOR_PARAMS_URL
        return section.form.dt.value

    def test_the_dynamics_step_has_no_step_size_of_its_own(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            rendering_rules = self._open_dynamics_step()

        # the step size is edited on the Subnetworks' Integrators, not on the step above them
        assert not hasattr(rendering_rules['renderer'].form, 'dt')

    def test_a_step_size_edited_on_one_subnetwork_reaches_every_subnetwork(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            saved = self._two_subnetworks()
            self._open_dynamics_step()
            self._open_subnetwork_editor()

            self._edit_step_size('0.5')
            draft = self._dynamics_draft()
            hybrid_simulator = self.hybrid_controller.context.hybrid_simulator
            saved_dts = [subnetwork.dynamics.integrator.dt for subnetwork in self._saved_subnetworks()]

            # and the other Subnetwork's Integrator shows it too
            cherrypy.request.method = "POST"
            self.hybrid_controller.select_subnetwork(subnetwork_id=saved[1].id)
            shown_on_the_other = self._integrator_params_dt()

        assert hybrid_simulator.dt == 0.5
        # every Integrator, the ones being edited and the saved ones alike
        assert [dynamics.integrator.dt for dynamics in draft.values()] == [0.5, 0.5]
        assert saved_dts == [0.5, 0.5]
        assert shown_on_the_other == 0.5

    def test_the_latest_step_size_wins_on_every_subnetwork(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            saved = self._two_subnetworks()
            self._open_dynamics_step()
            self._open_subnetwork_editor()

            self._edit_step_size('0.5')
            cherrypy.request.method = "POST"
            self.hybrid_controller.select_subnetwork(subnetwork_id=saved[1].id)
            self._edit_step_size('0.2')
            draft = self._dynamics_draft()
            saved_dts = [subnetwork.dynamics.integrator.dt for subnetwork in self._saved_subnetworks()]
            hybrid_simulator = self.hybrid_controller.context.hybrid_simulator

        assert hybrid_simulator.dt == 0.2
        assert [dynamics.integrator.dt for dynamics in draft.values()] == [0.2, 0.2]
        assert saved_dts == [0.2, 0.2]

    def test_a_new_integrator_class_takes_the_shared_step_size(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            saved = self._two_subnetworks()
            self._open_dynamics_step()
            self._open_subnetwork_editor()

            self._edit_step_size('0.5')
            cherrypy.request.method = "POST"
            self.hybrid_controller.set_subnetwork_integrator(integrator='Stochastic Heun')
            draft = self._dynamics_draft()

        # switching class does not bring back the new class's default step size
        assert draft[saved[0].id].integrator.dt == 0.5

    def test_editing_the_step_size_is_not_reported_as_an_unsaved_change(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._two_subnetworks()
            self._open_dynamics_step()
            self._open_subnetwork_editor()
            self._edit_step_size('0.5')
            column = self._dynamics_step()

        # dt is a simulation wide setting, stored as soon as it is edited, not a pending per Subnetwork edit
        assert not column['renderer'].is_modified
        assert not any(choice['is_modified'] for choice in column['renderer'].subnetwork_choices)

    def test_a_step_size_that_is_not_positive_is_refused(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._two_subnetworks()
            self._open_dynamics_step()
            self._open_subnetwork_editor()
            before = self.hybrid_controller.context.hybrid_simulator.dt

            answers = [self._edit_step_size(dt) for dt in ('0', '-1')]
            after = self.hybrid_controller.context.hybrid_simulator.dt
            draft = self._dynamics_draft()

        # the section answers with itself and the error, and no Subnetwork's step size changes
        for answer in answers:
            renderer = answer['renderer']
            assert renderer.form_action_url == HybridSimulatorURLs.SET_SUBNETWORK_INTEGRATOR_PARAMS_URL
            assert renderer.form.dt.errors
        assert after == before
        assert [dynamics.integrator.dt for dynamics in draft.values()] == [before, before]

    def test_integrator_params_keep_the_shared_dt_when_it_is_not_posted(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self._open_subnetwork_editor()
            self._edit_step_size('0.25')

            self.hybrid_controller.set_subnetwork_integrator(integrator='Heun')
            # no dt in the posted data: the shared step size is left as it is
            self.hybrid_controller.set_subnetwork_integrator_params()
            draft = self._dynamics_draft()
            saved = self._saved_subnetworks()

        assert draft[saved[0].id].integrator.dt == 0.25

    def test_resubmitting_the_same_integrator_class_keeps_its_edited_params(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self._open_subnetwork_editor()
            saved = self._saved_subnetworks()

            cherrypy.request.method = "POST"
            self.hybrid_controller.set_subnetwork_integrator(integrator='Stochastic Heun')
            self.hybrid_controller.set_subnetwork_integrator_params(noise='Additive')
            self.hybrid_controller.set_subnetwork_noise_params(nsig='[7.0]', ntau='0.0', noise_seed='42')
            edited = self._dynamics_draft()[saved[0].id].integrator.noise.nsig

            # the same class again: SimulatorIntegratorFragment.fill_trait would have replaced the
            # instance and thrown the edited value away
            self.hybrid_controller.set_subnetwork_integrator(integrator='Stochastic Heun')
            after = self._dynamics_draft()[saved[0].id].integrator.noise.nsig

        assert list(edited) == [7.0]
        assert list(after) == [7.0]

    def test_switching_the_integrator_class_resets_its_params(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self._open_subnetwork_editor()
            saved = self._saved_subnetworks()

            cherrypy.request.method = "POST"
            self.hybrid_controller.set_subnetwork_integrator(integrator='Stochastic Heun')
            self.hybrid_controller.set_subnetwork_integrator_params(noise='Additive')
            self.hybrid_controller.set_subnetwork_noise_params(nsig='[7.0]', ntau='0.0', noise_seed='42')

            self.hybrid_controller.set_subnetwork_integrator(integrator='Euler-Maruyama')
            reset = self._dynamics_draft()[saved[0].id].integrator

        assert type(reset).__name__ == 'EulerStochasticViewModel'
        assert list(reset.noise.nsig) != [7.0]

    def test_dynamics_edits_stay_in_the_draft_until_they_are_saved(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self._open_subnetwork_editor()
            saved = self._saved_subnetworks()

            cherrypy.request.method = "POST"
            self.hybrid_controller.set_subnetwork_integrator(integrator='Heun')
            self.hybrid_controller.set_subnetwork_integrator_params()

            before_save = type(self._saved_subnetworks()[0].dynamics.integrator).__name__
            column = self._dynamics_step()

            rendering_rules = self.hybrid_controller.save_subnetwork_dynamics()
            after_save = type(self._saved_subnetworks()[0].dynamics.integrator).__name__

        # the wizard step keeps summarising the saved dynamics while the column edits its own copy
        assert before_save == 'HeunDeterministicViewModel'
        assert after_save == 'HeunDeterministicViewModel'
        assert not column['renderer'].is_modified
        assert not rendering_rules['renderer'].is_modified
        assert rendering_rules['renderer'].form_action_url == HybridSimulatorURLs.SET_SUBNETWORK_DYNAMICS_URL

    def test_saving_reports_a_pending_change_before_it_happens(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self._open_subnetwork_editor()

            cherrypy.request.method = "POST"
            self.hybrid_controller.set_subnetwork_integrator(integrator='Stochastic Heun')
            unsaved = self._dynamics_step()

            saved_rules = self.hybrid_controller.save_subnetwork_dynamics()
            saved_column = self._dynamics_step()
            stored_integrator = type(self._saved_subnetworks()[0].dynamics.integrator).__name__

        assert unsaved['renderer'].is_modified
        assert not saved_rules['renderer'].is_modified
        assert not saved_column['renderer'].is_modified
        assert stored_integrator == 'HeunStochasticViewModel'

    def test_configuring_one_subnetwork_leaves_the_other_alone(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            saved = self._two_subnetworks()
            first_id, second_id = saved[0].id, saved[1].id
            self._open_dynamics_step()
            self._open_subnetwork_editor()

            cherrypy.request.method = "POST"
            self.hybrid_controller.set_subnetwork_integrator(integrator='Stochastic Heun')

            self.hybrid_controller.select_subnetwork(subnetwork_id=second_id)
            second_before = type(self._dynamics_draft()[second_id].integrator).__name__

            self.hybrid_controller.set_subnetwork_model(model='Kuramoto Oscillator')
            draft = self._dynamics_draft()

        assert second_before == 'HeunDeterministicViewModel'
        # the Integrator was changed for the first Subnetwork, the Model for the second one
        assert type(draft[first_id].integrator).__name__ == 'HeunStochasticViewModel'
        assert type(draft[first_id].model).__name__ == 'Generic2dOscillator'
        assert type(draft[second_id].integrator).__name__ == 'HeunDeterministicViewModel'
        assert type(draft[second_id].model).__name__ == 'Kuramoto'

    @staticmethod
    def _editor_urls(rendering_rules):
        return [section.form_action_url for section in rendering_rules['renderer'].editor_sections]

    def _editor(self):
        cherrypy.request.method = "GET"
        return self.hybrid_controller.subnetwork_editor()

    def test_selecting_a_subnetwork_answers_with_its_editor(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            saved = self._two_subnetworks()
            self._open_dynamics_step()
            self._open_subnetwork_editor()

            cherrypy.request.method = "POST"
            rendering_rules = self.hybrid_controller.select_subnetwork(subnetwork_id=saved[1].id)

        # every section of that Subnetwork's dynamics, in order, closed by the row of actions
        assert rendering_rules['renderer'].is_subnetwork_editor
        assert rendering_rules['renderer'].editor_subnetwork_name == saved[1].name
        assert self._editor_urls(rendering_rules) == [
            HybridSimulatorURLs.SET_SUBNETWORK_MODEL_URL,
            HybridSimulatorURLs.SET_SUBNETWORK_MODEL_PARAMS_URL,
            HybridSimulatorURLs.SET_SUBNETWORK_INTEGRATOR_URL,
            HybridSimulatorURLs.SET_SUBNETWORK_INTEGRATOR_PARAMS_URL,
            HybridSimulatorURLs.SAVE_SUBNETWORK_DYNAMICS_URL]

    def test_every_section_of_the_editor_is_editable(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            saved = self._two_subnetworks()
            self._open_dynamics_step()
            self._open_subnetwork_editor()

            cherrypy.request.method = "POST"
            rendering_rules = self.hybrid_controller.select_subnetwork(subnetwork_id=saved[1].id)

        sections = rendering_rules['renderer'].editor_sections
        # nothing arrives read only: the whole configuration is edited at once
        assert [section.is_read_only for section in sections] == [False] * 5
        # the sections carry no Next or Previous of their own
        assert all(section.is_editor_section for section in sections[:-1])
        # the row of actions goes back, or stores the dynamics and moves on
        actions = sections[-1]
        assert not actions.is_editor_section
        assert actions.is_dynamics_save_fragment
        # Previous skips the Subnetwork dynamics step, which only holds the tabs, back to the Subnetworks
        assert actions.previous_form_action_url == HybridSimulatorURLs.SET_SUBNETWORKS_URL
        assert actions.next_form_action_url == HybridSimulatorURLs.SET_PROJECTIONS_URL
        # Set up region Model is offered on the Model parameters, the section it edits
        assert [section.include_region_model_button for section in sections] == [
            False, True, False, False, False]

    def test_the_editor_follows_the_integrator_and_noise_classes(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._two_subnetworks()
            self._open_dynamics_step()
            self._open_subnetwork_editor()

            cherrypy.request.method = "POST"
            self.hybrid_controller.set_subnetwork_integrator(integrator='Stochastic Heun')
            stochastic = self._editor()

            cherrypy.request.method = "POST"
            self.hybrid_controller.set_subnetwork_integrator_params(noise='Multiplicative')
            multiplicative = self._editor()

            cherrypy.request.method = "POST"
            self.hybrid_controller.set_subnetwork_integrator(integrator='Heun')
            deterministic = self._editor()

        # a stochastic Integrator brings in its Noise section
        assert self._editor_urls(stochastic)[-2] == HybridSimulatorURLs.SET_SUBNETWORK_NOISE_PARAMS_URL
        # a Multiplicative Noise its Equation section on top of that
        assert self._editor_urls(multiplicative)[-3:-1] == [
            HybridSimulatorURLs.SET_SUBNETWORK_NOISE_PARAMS_URL,
            HybridSimulatorURLs.SET_SUBNETWORK_NOISE_EQUATION_PARAMS_URL]
        # and a deterministic one takes both away again
        assert HybridSimulatorURLs.SET_SUBNETWORK_NOISE_PARAMS_URL not in self._editor_urls(deterministic)
        assert len(self._editor_urls(deterministic)) == 5

    def test_a_model_change_reloads_its_parameters(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self._open_subnetwork_editor()

            cherrypy.request.method = "POST"
            answer = self.hybrid_controller.set_subnetwork_model(model='Kuramoto Oscillator')
            editor = self._editor()

        # the Model section answers with the parameters of the newly selected class
        assert answer['renderer'].form_action_url == HybridSimulatorURLs.SET_SUBNETWORK_MODEL_PARAMS_URL
        params_section = editor['renderer'].editor_sections[1]
        assert type(params_section.form).__name__ == 'KuramotoModelForm'

    def test_a_refused_section_answers_with_itself(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self._open_subnetwork_editor()

            cherrypy.request.method = "POST"
            params = dict(self.MODEL_PARAMS)
            params['a'] = 'not a number'
            rendering_rules = self.hybrid_controller.set_subnetwork_model_params(**params)

        # the client replaces the section on screen with this one, errors included
        renderer = rendering_rules['renderer']
        assert renderer.form_action_url == HybridSimulatorURLs.SET_SUBNETWORK_MODEL_PARAMS_URL
        assert renderer.is_editor_section
        assert renderer.form.a.errors

    def test_the_selector_reports_which_subnetworks_are_unsaved(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            saved = self._two_subnetworks()
            self._open_dynamics_step()
            self._open_subnetwork_editor()

            cherrypy.request.method = "POST"
            self.hybrid_controller.select_subnetwork(subnetwork_id=saved[1].id)
            self.hybrid_controller.set_subnetwork_model(model='Kuramoto Oscillator')
            edited = self._dynamics_step()['renderer'].subnetwork_choices

            cherrypy.request.method = "POST"
            self.hybrid_controller.save_subnetwork_dynamics()
            stored = self._dynamics_step()['renderer'].subnetwork_choices

        # only the Subnetwork that was edited is marked, and saving clears the mark
        assert [choice['is_modified'] for choice in edited] == [False, True]
        assert [choice['is_modified'] for choice in stored] == [False, False]

    def test_building_the_editor_does_not_change_the_configuration(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self._open_subnetwork_editor()
            saved = self._saved_subnetworks()

            cherrypy.request.method = "POST"
            self.hybrid_controller.set_subnetwork_integrator(integrator='Stochastic Heun')
            self.hybrid_controller.set_subnetwork_integrator_params(noise='Additive')
            self.hybrid_controller.set_subnetwork_noise_params(nsig='[7.0]', ntau='0.0', noise_seed='42')
            before = list(self._dynamics_draft()[saved[0].id].integrator.noise.nsig)

            # showing the configuration must not be able to alter it
            self.hybrid_controller.select_subnetwork(subnetwork_id=saved[0].id)
            after = list(self._dynamics_draft()[saved[0].id].integrator.noise.nsig)

        assert before == [7.0]
        assert after == [7.0]

    def test_selecting_a_subnetwork_points_the_selector_at_it(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            saved = self._two_subnetworks()
            self._open_dynamics_step()
            self._open_subnetwork_editor()

            cherrypy.request.method = "POST"
            self.hybrid_controller.select_subnetwork(subnetwork_id=saved[1].id)
            dynamics_step = self._dynamics_step()

        selector = dynamics_step['renderer']
        assert selector.selected_subnetwork == saved[1].id
        selected = [choice for choice in selector.subnetwork_choices if choice['is_selected']]
        assert len(selected) == 1
        assert selected[0]['name'] == saved[1].name

    def test_dynamics_survive_renaming_the_subnetwork(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self._open_subnetwork_editor()

            cherrypy.request.method = "POST"
            self.hybrid_controller.set_subnetwork_model(model='Kuramoto Oscillator')
            self.hybrid_controller.rename_subnetwork(subnetwork_index='0', name='Cortex')
            saved = self._saved_subnetworks()
            draft = self._dynamics_draft()

        # the identifier is what keeps the configuration attached across a rename
        assert type(draft[saved[0].id].model).__name__ == 'Kuramoto'

    def test_dynamics_are_dropped_together_with_their_subnetwork(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            saved = self._two_subnetworks()
            removed_id = saved[1].id
            self._open_dynamics_step()
            self._open_subnetwork_editor()

            cherrypy.request.method = "POST"
            self.hybrid_controller.select_subnetwork(subnetwork_id=removed_id)
            self.hybrid_controller.set_subnetwork_model(model='Kuramoto Oscillator')

            self.hybrid_controller.remove_subnetwork(subnetwork_index='1')
            self.hybrid_controller.save_subnetworks()
            self._open_subnetwork_editor()
            draft = self._dynamics_draft()
            selected = self.hybrid_controller.context.selected_subnetwork

        assert removed_id not in draft
        # and the selection falls back to a Subnetwork that still exists
        assert selected in draft

    def test_saving_rejects_a_model_parameter_that_does_not_fit_the_subnetwork(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self._open_subnetwork_editor()

            cherrypy.request.method = "POST"
            self.hybrid_controller.set_subnetwork_model(model='Generic 2D Oscillator')
            params = dict(self.MODEL_PARAMS)
            # three values for a Subnetwork owning every region of the Connectivity
            params['a'] = '[-2.0, -1.0, 0.0]'
            self.hybrid_controller.set_subnetwork_model_params(**params)

            rendering_rules = self.hybrid_controller.save_subnetwork_dynamics()
            saved = self._saved_subnetworks()

        # refused, and the saved configuration is left as it was
        assert list(saved[0].dynamics.model.a) == [-2.0]
        assert rendering_rules['renderer'].form_action_url == HybridSimulatorURLs.SET_SUBNETWORK_DYNAMICS_URL
        # the reason travels with the answer, which is how the client knows not to move on
        assert rendering_rules['renderer'].error_message

    # ---------------------------------------------------------------- Set up region Model

    @staticmethod
    def _g2d_parameters(a):
        """Generic2dOscillator parameters, as a Dynamic stores them: a JSON list of name/value pairs."""
        values = {'tau': 1.0, 'a': a, 'b': -10.0, 'c': 0.0, 'I': 0.0, 'd': 0.02, 'e': 3.0, 'f': 1.0,
                  'g': 0.0, 'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0}
        return json.dumps([[name, value] for name, value in values.items()])

    def _saved_dynamics(self):
        """Two model configurations on the Subnetwork's Model class, and one on another class."""
        self.dynamic_fast = dao.store_entity(Dynamic(
            'hybrid_fast', self.test_user.id, ModelsEnum.GENERIC_2D_OSCILLATOR.value.__name__,
            self._g2d_parameters(-2.0), HeunDeterministic.__name__, None))
        self.dynamic_slow = dao.store_entity(Dynamic(
            'hybrid_slow', self.test_user.id, ModelsEnum.GENERIC_2D_OSCILLATOR.value.__name__,
            self._g2d_parameters(-4.0), HeunDeterministic.__name__, None))
        self.dynamic_other = dao.store_entity(Dynamic(
            'hybrid_kuramoto', self.test_user.id, ModelsEnum.KURAMOTO.value.__name__,
            '[["omega", 1.0]]', HeunDeterministic.__name__, None))

    def _region_model_panel(self):
        previous_method = getattr(cherrypy.request, 'method', None)
        cherrypy.request.method = "GET"
        try:
            return self.hybrid_controller.configure_region_model()
        finally:
            if previous_method is not None:
                cherrypy.request.method = previous_method

    def _apply_region_model(self, dynamic_id, node_indices):
        cherrypy.request.method = "POST"
        return json.loads(self.hybrid_controller.apply_region_model(
            dynamic_id=str(dynamic_id), node_indices=json.dumps(node_indices)))

    def _submit_region_model(self):
        """Press Submit, which puts the placed values into the Model parameters of the middle column."""
        cherrypy.request.method = "POST"
        return self.hybrid_controller.submit_region_model()

    def test_model_params_step_offers_the_region_model_action(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self._open_subnetwork_editor()
            cherrypy.request.method = "POST"
            rendering_rules = self.hybrid_controller.set_subnetwork_model(model='Generic 2D Oscillator')

        # the same action the classic Cockpit offers next to the Model parameters
        assert rendering_rules['renderer'].include_region_model_button

    def test_region_model_lists_only_this_subnetworks_regions(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._saved_dynamics()
            saved = self._two_subnetworks()
            self._open_dynamics_step()
            self._open_subnetwork_editor()
            panel = self._region_model_panel()

        renderer = panel['renderer']
        # the first Subnetwork is selected by default, and its Model applies to the nodes it owns
        assert [row['index'] for row in renderer.region_model_rows] == list(saved[0].node_indices)
        assert renderer.region_model_unassigned == len(saved[0].node_indices)

    def test_region_model_offers_only_matching_model_configurations(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._saved_dynamics()
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self._open_subnetwork_editor()
            panel = self._region_model_panel()

        offered = [dynamic.name for dynamic in panel['renderer'].region_model_dynamics]
        # the Subnetwork is configured with Generic2dOscillator, so the Kuramoto configuration is not
        # offered; the Model class stays decided by the wizard step alone
        assert sorted(offered) == ['hybrid_fast', 'hybrid_slow']

    def test_placing_on_some_regions_leaves_the_model_alone(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._saved_dynamics()
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self._open_subnetwork_editor()
            self._region_model_panel()
            saved = self._saved_subnetworks()

            answer = self._apply_region_model(self.dynamic_fast.id, [0, 1])
            model_a = list(self._dynamics_draft()[saved[0].id].model.a)

        assert answer['status'] == 'ok'
        assert answer['unassigned'] == self.connectivity.number_of_regions - 2
        assert [row['dynamic_name'] for row in answer['rows'][:2]] == ['hybrid_fast', 'hybrid_fast']
        assert answer['rows'][2]['dynamic_name'] == ''
        # placing only records where the configuration goes, the Model is written by Submit
        assert model_a == [-2.0]

    def test_submitting_writes_one_value_per_node_into_the_model(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._saved_dynamics()
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self._open_subnetwork_editor()
            self._region_model_panel()
            saved = self._saved_subnetworks()
            every_region = list(saved[0].node_indices)

            self._apply_region_model(self.dynamic_fast.id, every_region)
            answer = self._apply_region_model(self.dynamic_slow.id, every_region[:2])
            before_submit = list(self._dynamics_draft()[saved[0].id].model.a)

            rendering_rules = self._submit_region_model()
            model = self._dynamics_draft()[saved[0].id].model

        assert answer['unassigned'] == 0
        # placing only records where each configuration goes, Submit is what writes it
        assert before_submit == [-2.0]
        # two regions carry the slow configuration and the rest the fast one, in node order
        assert list(model.a) == [-4.0, -4.0] + [-2.0] * (len(every_region) - 2)
        assert len(model.a) == len(every_region)
        # a parameter both configurations agree on stays a single shared value
        assert list(model.tau) == [1.0]
        # and the answer is the Model parameters step, so those values show up in the middle column
        assert rendering_rules['renderer'].form_action_url == \
               HybridSimulatorURLs.SET_SUBNETWORK_MODEL_PARAMS_URL
        assert rendering_rules['renderer'].include_region_model_button

    def test_submitting_an_incomplete_placement_is_refused(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._saved_dynamics()
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self._open_subnetwork_editor()
            self._region_model_panel()
            saved = self._saved_subnetworks()

            self._apply_region_model(self.dynamic_slow.id, [0, 1])
            rendering_rules = self._submit_region_model()
            model_a = list(self._dynamics_draft()[saved[0].id].model.a)

        # a Model parameter needs a value for every node, so nothing is written
        assert model_a == [-2.0]
        # and the step comes back unchanged, so what is on screen still matches the configuration
        assert rendering_rules['renderer'].form_action_url == \
               HybridSimulatorURLs.SET_SUBNETWORK_MODEL_PARAMS_URL

    def test_region_model_reaches_the_configuration_only_once_saved(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._saved_dynamics()
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self._open_subnetwork_editor()
            self._region_model_panel()
            saved = self._saved_subnetworks()

            self._apply_region_model(self.dynamic_slow.id, list(saved[0].node_indices))
            self._submit_region_model()
            before_save = list(self._saved_subnetworks()[0].dynamics.model.a)

            self.hybrid_controller.save_subnetwork_dynamics()
            after_save = list(self._saved_subnetworks()[0].dynamics.model.a)

        # the panel edits the draft, like every other Phase 3 step
        assert before_save == [-2.0]
        assert after_save == [-4.0]

    def test_placing_a_configuration_of_another_model_is_refused(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._saved_dynamics()
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self._open_subnetwork_editor()
            self._region_model_panel()

            answer = self._apply_region_model(self.dynamic_other.id, [0, 1])

        assert answer['status'] == 'error'
        assert answer['rows'][0]['dynamic_name'] == ''

    def test_placing_on_no_region_is_refused(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._saved_dynamics()
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self._open_subnetwork_editor()
            self._region_model_panel()

            answer = self._apply_region_model(self.dynamic_fast.id, [])

        assert answer['status'] == 'error'

    def test_regions_moved_away_lose_their_configuration(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._saved_dynamics()
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self._open_subnetwork_editor()
            self._region_model_panel()

            self._apply_region_model(self.dynamic_fast.id, [0, 1, 2])

            # regroup: regions 0 and 1 go to a second Subnetwork
            self.hybrid_controller.add_subnetwork()
            self.hybrid_controller.move_regions(subnetwork_index='1', node_indices=json.dumps([0, 1]))
            self.hybrid_controller.save_subnetworks()

            panel = self._region_model_panel()

        rows = {row['index']: row['dynamic_name'] for row in panel['renderer'].region_model_rows}
        # what is left of the first Subnetwork keeps its configuration, and the regions it no longer
        # owns are simply not listed for it any more
        assert 0 not in rows and 1 not in rows
        assert rows[2] == 'hybrid_fast'

    # ---------------------------------------------------------------- Projections

    def _reach_projections(self):
        """Walk the wizard to the closing step of the Subnetwork configuration, then press Next."""
        cherrypy.request.method = "POST"
        return self.hybrid_controller.set_projections()

    def _configured_subnetwork(self):
        """A saved grouping whose dynamics are saved too, which is what the Projections need."""
        self._configured_hybrid_simulator()
        self._open_dynamics_step()
        self._open_subnetwork_editor()
        cherrypy.request.method = "POST"
        self.hybrid_controller.save_subnetwork_dynamics()

    def test_next_reaches_the_projections_step(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_subnetwork()
            rendering_rules = self._reach_projections()

        renderer = rendering_rules['renderer']
        assert renderer.form_action_url == HybridSimulatorURLs.SET_PROJECTIONS_URL
        assert renderer.previous_form_action_url == HybridSimulatorURLs.SAVE_SUBNETWORK_DYNAMICS_URL
        assert renderer.is_projections_fragment
        assert renderer.fragment_title == "Projections"
        # its Next opens the Monitors step, rather than submitting it with nothing chosen
        assert renderer.next_button_enabled
        assert renderer.next_form_action_url == HybridSimulatorURLs.OPEN_MONITORS_URL

    def test_the_closing_step_offers_a_way_onward(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self._open_subnetwork_editor()
            cherrypy.request.method = "POST"
            self.hybrid_controller.set_subnetwork_model(model='Generic 2D Oscillator')
            self.hybrid_controller.set_subnetwork_model_params(**self.MODEL_PARAMS)
            self.hybrid_controller.set_subnetwork_integrator(integrator='Heun')
            rendering_rules = self.hybrid_controller.set_subnetwork_integrator_params()

        renderer = rendering_rules['renderer']
        assert renderer.is_dynamics_save_fragment
        # its own action stores the dynamics, so Save and continue opens the Projections afterwards
        assert renderer.form_action_url == HybridSimulatorURLs.SAVE_SUBNETWORK_DYNAMICS_URL
        assert renderer.next_form_action_url == HybridSimulatorURLs.SET_PROJECTIONS_URL

    def test_a_single_subnetwork_generates_one_intra_projection(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_subnetwork()
            rendering_rules = self._reach_projections()

        rows = rendering_rules['renderer'].projection_rows
        assert len(rows) == 1
        assert rows[0]['kind'] == 'Intra'
        assert rows[0]['shape'] == '{n} x {n}'.format(n=self.connectivity.number_of_regions)
        # nothing to connect to, so no pair is left out either
        assert rendering_rules['renderer'].unconnected_pairs == []

    def test_two_subnetworks_generate_intra_and_inter_projections(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.add_subnetwork()
            self.hybrid_controller.move_regions(subnetwork_index='1', node_indices=json.dumps([0, 1, 2]))
            self.hybrid_controller.save_subnetworks()
            self._open_dynamics_step()
            self._open_subnetwork_editor()
            cherrypy.request.method = "POST"
            self.hybrid_controller.save_subnetwork_dynamics()

            rendering_rules = self._reach_projections()

        rows = rendering_rules['renderer'].projection_rows
        intra = [row for row in rows if row['kind'] == 'Intra']
        inter = [row for row in rows if row['kind'] == 'Inter']

        # one IntraProjection per Subnetwork, sized to the nodes it owns
        assert len(intra) == 2
        assert sorted(row['shape'] for row in intra) == \
               sorted(['{n} x {n}'.format(n=self.connectivity.number_of_regions - 3), '3 x 3'])
        # and the pairs this Connectivity actually connects
        assert len(inter) >= 1
        for row in inter:
            assert row['source'] != row['target']
            # the coupling variables are named: the first one of each Model by default
            assert row['source_cvar'] == ['V']
            assert row['target_cvar'] == ['V']

    def test_projections_are_refused_while_the_dynamics_are_unsaved(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_subnetwork()

            # change the Integrator without saving it
            cherrypy.request.method = "POST"
            self.hybrid_controller.set_subnetwork_integrator(integrator='Stochastic Heun')
            rendering_rules = self._reach_projections()

        renderer = rendering_rules['renderer']
        # the Projections are built from the saved dynamics, so the wizard stays on the dynamics step
        assert renderer.form_action_url == HybridSimulatorURLs.SET_SUBNETWORK_DYNAMICS_URL
        assert renderer.is_dynamics_summary_fragment
        assert renderer.is_modified
        # and says why, since a message set on the session does not reach the fragment
        assert 'Save the Subnetwork configuration' in renderer.error_message

    def test_projections_follow_the_grouping(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_subnetwork()
            before = self._reach_projections()['renderer'].projection_rows

            # regroup and save, then generate again
            self.hybrid_controller.add_subnetwork()
            self.hybrid_controller.move_regions(subnetwork_index='1', node_indices=json.dumps([0, 1]))
            self.hybrid_controller.save_subnetworks()
            self.hybrid_controller.save_subnetwork_dynamics()
            after = self._reach_projections()['renderer'].projection_rows

        # they are derived on every render, so the new grouping is reflected without anything stored
        assert len(before) == 1
        assert len([row for row in after if row['kind'] == 'Intra']) == 2

    # ---------------------------------------------------------------- navigation

    # ---------------------------------------------------------------- Projection editing

    def _two_saved_subnetworks_at_the_projections(self):
        """Two saved Subnetworks with saved dynamics, and the Projections step opened over them."""
        saved = self._two_subnetworks()
        self._open_dynamics_step()
        self._open_subnetwork_editor()
        cherrypy.request.method = "POST"
        self.hybrid_controller.save_subnetwork_dynamics()
        return saved, self._reach_projections()

    def _stored_projections(self):
        return list(self.hybrid_controller.context.hybrid_simulator.projections)

    def _key(self, projection):
        return self.hybrid_controller.hybrid_simulator_service.projection_key(projection)

    def _inter_keys(self):
        return [self._key(projection) for projection in self._stored_projections() if not projection.is_intra]

    def _inter_key(self):
        return self._inter_keys()[0]

    def _post_projection(self, projection_key, **data):
        posted = {'source_cvar': 'V', 'target_cvar': 'V', 'scale': '1.0', 'cv': '3.0', 'coupling': 'Linear',
                  'a': '[0.00390625]', 'b': '[0.0]'}
        posted.update(data)
        cherrypy.request.method = "POST"
        return self.hybrid_controller.set_projection(projection_key=projection_key, **posted)

    def _projections_editor(self):
        """The Projections editor the step declares for the third column."""
        cherrypy.request.method = "GET"
        return self.hybrid_controller.configure_projections()['renderer']

    def _projection(self, projection_key):
        return [projection for projection in self._stored_projections()
                if self._key(projection) == projection_key][0]

    def test_the_projections_step_stores_the_configuration_and_opens_its_editor(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            _, rendering_rules = self._two_saved_subnetworks_at_the_projections()
            stored = self._stored_projections()
            editor = self._projections_editor()

        renderer = rendering_rules['renderer']
        # every generated Projection is configured, and stored on the configuration
        assert len(stored) == len(renderer.projection_rows) >= 3
        assert all(row['coupling'] == 'Linear' for row in renderer.projection_rows)
        # none of them part of the simulation until the user adds it
        assert not any(row['included'] for row in renderer.projection_rows)
        # the step lists them, and declares the editor for the third column
        assert renderer.context_form_url == HybridSimulatorURLs.CONFIGURE_PROJECTIONS_URL
        # which edits the selected Projection, the first one to begin with
        assert editor.selected_projection == editor.projection_rows[0]['key']
        assert editor.projection_row['key'] == editor.selected_projection
        assert editor.projection_form.coupling.value.__name__ == 'Linear'

    def test_the_editor_shows_the_selected_projection(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._two_saved_subnetworks_at_the_projections()
            inter_key = self._inter_key()
            editor = self.hybrid_controller.select_projection(projection_key=inter_key)['renderer']
            # and the editor rendered again keeps it selected
            reopened = self._projections_editor()

        # the editor answers with that Projection's tab selected
        assert editor.selected_projection == inter_key
        assert reopened.selected_projection == inter_key
        assert editor.projection_row['key'] == inter_key
        assert editor.projection_row['kind'] == 'Inter'
        assert editor.projection_form.coupling.value.__name__ == 'Linear'
        assert list(editor.coupling_form.a.data) == [0.00390625]

    def test_an_edit_reaches_the_configuration(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._two_saved_subnetworks_at_the_projections()
            inter_key = self._inter_key()
            answer = self._post_projection(inter_key, scale='2.5', cv='6.0', a='[0.2]', b='[0.1]')
            projection = self._projection(inter_key)

        assert answer['renderer'].projection_applied
        assert projection.scale == 2.5
        assert projection.cv == 6.0
        assert list(projection.coupling.a) == [0.2]
        assert list(projection.coupling.b) == [0.1]

    def test_the_coupling_variables_are_chosen_one_per_side_among_the_models_own(self):
        from tvb.adapters.forms.hybrid_projection_forms import CouplingVariableSelectField, HybridProjectionForm

        with patch('cherrypy.session', self.sess_mock, create=True):
            self._two_saved_subnetworks_at_the_projections()
            inter_key = self._inter_key()
            editor = self.hybrid_controller.select_projection(projection_key=inter_key)['renderer']
            answer = self._post_projection(inter_key, source_cvar='V', target_cvar='V')
            projection = self._projection(inter_key)

        form = editor.projection_form
        for field in (form.source_cvar, form.target_cvar):
            assert isinstance(field, CouplingVariableSelectField)
            # the coupling variables of Generic2dOscillator, W being only a state variable of it
            assert field.choices == ['V']
            assert field.value == 'V'
        assert answer['renderer'].projection_applied
        # a single variable each side, kept as the list the library takes
        assert list(projection.source_cvar) == ['V'] and list(projection.target_cvar) == ['V']

        # always a drop down, the names offered as they are, the stored one selected
        jansen_rit = ['y1', 'y2']
        html = str(HybridProjectionForm(jansen_rit, ['V']).source_cvar)
        assert '<select name="source_cvar"' in html
        assert 'type="radio"' not in html and 'type="checkbox"' not in html
        assert '>y1</option>' in html and '>y2</option>' in html
        form = HybridProjectionForm(jansen_rit, ['V'])
        form.source_cvar.data = 'y2'
        assert re.search(r'<option value="y2" selected', str(form.source_cvar))

    def test_a_new_coupling_function_starts_from_its_defaults(self):
        from tvb.simulator.hybrid.coupling import Kuramoto, Linear

        with patch('cherrypy.session', self.sess_mock, create=True):
            self._two_saved_subnetworks_at_the_projections()
            inter_key = self._inter_key()
            # the parameters on screen still belong to Linear when Kuramoto is picked
            answer = self._post_projection(inter_key, coupling='Kuramoto', a='[0.2]')
            kuramoto = self._projection(inter_key).coupling
            self._post_projection(inter_key, coupling='Linear', a='[0.7]')
            linear = self._projection(inter_key).coupling

        assert answer['renderer'].projection_applied
        assert type(kuramoto) is Kuramoto and list(kuramoto.a) == list(Kuramoto.a.default)
        # and Linear comes back with the classic Simulator's defaults, not the hybrid Linear's
        assert type(linear) is Linear and list(linear.a) == [0.00390625]
        # the editor answers with the new function's parameters
        assert [field.name for field in answer['renderer'].coupling_form.fields] == ['a']

    def test_a_refused_edit_leaves_the_configuration_alone(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._two_saved_subnetworks_at_the_projections()
            inter_key = self._inter_key()
            answers = [self._post_projection(inter_key, cv='0', scale='9.0'),
                       self._post_projection(inter_key, scale='nine'),
                       self._post_projection(inter_key, a='not an array', scale='9.0'),
                       self._post_projection(inter_key, source_cvar='W', scale='9.0')]
            projection = self._projection(inter_key)

        for answer in answers:
            assert not answer['renderer'].projection_applied
        assert answers[0]['renderer'].projection_form.cv.errors
        assert answers[1]['renderer'].projection_form.scale.errors
        assert answers[2]['renderer'].coupling_form.a.errors
        # W is a state variable of Generic2dOscillator, but not one of its coupling variables
        assert answers[3]['renderer'].projection_form.source_cvar.errors
        assert projection.scale == 1.0 and projection.cv > 0

    def test_adding_and_removing_a_projection_keeps_its_settings(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._two_saved_subnetworks_at_the_projections()
            inter_key = self._inter_key()
            self._post_projection(inter_key, scale='2.5')
            nothing_added_step = self._reach_projections()['renderer']

            cherrypy.request.method = "POST"
            added = self.hybrid_controller.set_projection_included(projection_key=inter_key, included='true')
            was_added = self._projection(inter_key).included
            added_step = self._reach_projections()['renderer']
            removed = self.hybrid_controller.set_projection_included(projection_key=inter_key, included='false')
            was_removed = not self._projection(inter_key).included
            added_again = self.hybrid_controller.set_projection_included(projection_key=inter_key, included='true')
            added_again_projection = self._projection(inter_key)

        # nothing is added to begin with, which the step says once rather than per Projection
        assert not any(row['included'] for row in nothing_added_step.projection_rows)
        assert nothing_added_step.projection_warnings == []
        # added, and listed on the step
        added_row = [row for row in added['renderer'].projection_rows if row['key'] == inter_key][0]
        assert added_row['included'] and was_added
        assert [row['key'] for row in added_step.projection_rows if row['included']] == [inter_key]
        # with what is still uncoupled: the regions of each Subnetwork, whose IntraProjection is not added
        assert sum('are not coupled to one another' in warning for warning in added_step.projection_warnings) == 2
        # every answer follows a change of the configuration, which the list on the step follows
        assert added['renderer'].is_modified and removed['renderer'].is_modified
        # removed, then added again, with what it was configured with
        assert was_removed
        assert added_again_projection.included and added_again_projection.scale == 2.5
        assert added_again['renderer'].is_modified

    def test_the_inter_tab_picks_among_the_inter_projections(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            saved, _ = self._two_saved_subnetworks_at_the_projections()
            first, second = saved[0], saved[1]
            opened = self._projections_editor()
            cherrypy.request.method = "POST"
            chosen = self.hybrid_controller.select_inter_projection(source_id=second.id, target_id=first.id)
            # a source alone, with a target it has no InterProjection towards, opens its first one
            fallback = self.hybrid_controller.select_inter_projection(source_id=second.id, target_id=second.id)

        # while an IntraProjection is selected, the Inter tab points at the first InterProjection
        assert opened.projection_row['kind'] == 'Intra'
        assert opened.inter_projection == '{}:{}'.format(first.id, second.id)
        assert [source['name'] for source in opened.inter_sources] == [first.name, second.name]
        assert [target['name'] for target in opened.inter_targets] == [second.name]
        assert opened.inter_total == 2 and opened.inter_added == 0

        # and the Intra tab offers one IntraProjection per Subnetwork, pointing at the selected one
        assert [option['name'] for option in opened.intra_options] == [first.name, second.name]
        assert opened.intra_projection == '{}:{}'.format(first.id, first.id) == opened.selected_projection
        assert opened.intra_added == 0

        for answer in (chosen, fallback):
            renderer = answer['renderer']
            # while an InterProjection is selected, the Intra tab keeps pointing at the last one opened
            assert renderer.intra_projection == '{}:{}'.format(first.id, first.id)
            assert renderer.selected_projection == '{}:{}'.format(second.id, first.id)
            assert renderer.projection_row['kind'] == 'Inter'
            assert renderer.inter_projection == renderer.selected_projection
            assert [source['is_selected'] for source in renderer.inter_sources] == [False, True]
            assert [target['name'] for target in renderer.inter_targets] == [first.name]

    def test_each_tab_points_at_the_last_projection_it_opened(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            saved, _ = self._two_saved_subnetworks_at_the_projections()
            first, second = saved[0], saved[1]
            cherrypy.request.method = "POST"
            self.hybrid_controller.select_projection(projection_key='{}:{}'.format(second.id, second.id))
            self.hybrid_controller.select_inter_projection(source_id=second.id, target_id=first.id)
            back_to_intra = self.hybrid_controller.select_projection(
                projection_key='{}:{}'.format(first.id, first.id))['renderer']
            inter_open = self.hybrid_controller.select_inter_projection(
                source_id=first.id, target_id=second.id)['renderer']

        # the Inter tab still points at the pair opened last, while an IntraProjection is selected
        assert back_to_intra.inter_projection == '{}:{}'.format(second.id, first.id)
        # and the Intra tab at the Subnetwork opened last, while an InterProjection is selected
        assert inter_open.intra_projection == '{}:{}'.format(first.id, first.id)

    def test_the_inter_tab_is_rendered_with_its_pickers(self):
        with patch.object(TvbProfile.current.web, 'RENDER_HTML', True), \
                patch('cherrypy.session', self.sess_mock, create=True):
            saved, _ = self._two_saved_subnetworks_at_the_projections()
            cherrypy.request.method = "POST"
            inter_key = '{}:{}'.format(saved[0].id, saved[1].id)
            self.hybrid_controller.set_projection_included(projection_key=inter_key, included='true')
            html = self.hybrid_controller.select_inter_projection(source_id=saved[0].id, target_id=saved[1].id)

        # the InterProjections share one tab, selected and in the Inter colour, with its From and To
        inter_tab = re.search(r'<div role="tab"[^>]*hybrid-inter-tab[^>]*>', html).group(0)
        assert 'is-inter' in inter_tab and 'is-selected' in inter_tab
        assert 'id="hybrid-inter-source"' in html and 'id="hybrid-inter-target"' in html
        assert 'hybridSelectInterProjection()' in html
        # there is no tab per Projection: one Intra tab and one Inter tab, each with its picker
        assert html.count('role="tab"') == 2
        assert 'id="hybrid-intra-subnetwork"' in html
        intra_tab = re.search(r'<div role="tab"[^>]*hybrid-intra-tab[^>]*>', html).group(0)
        assert 'is-selected' not in intra_tab
        # an added target is checked, and counted
        assert '✓ {}'.format(saved[1].name) in html
        assert '1 of 2 added' in html
        # and the editor offers to remove it again
        assert 'Remove from the simulation' in html
        # and the editor under it is framed in the Inter colour
        assert 'class="hybrid-projection-block is-inter"' in html

    def test_projection_settings_survive_leaving_and_reopening_the_step(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._two_saved_subnetworks_at_the_projections()
            inter_key = self._inter_key()
            self._post_projection(inter_key, scale='2.5')
            reopened = self._reach_projections()

        row = [row for row in reopened['renderer'].projection_rows if row['key'] == inter_key][0]
        assert row['scale'] == 2.5

    def test_the_launched_configuration_carries_the_projections(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._two_saved_subnetworks_at_the_projections()
            cherrypy.request.method = "POST"
            # every IntraProjection added, and no InterProjection
            for projection in self._stored_projections():
                if projection.is_intra:
                    self.hybrid_controller.set_projection_included(projection_key=self._key(projection),
                                                                   included='true')
            hybrid_simulator = self.hybrid_controller.context.hybrid_simulator
            connectivity = h5.load_from_gid(hybrid_simulator.connectivity)
            network_set = self.hybrid_controller.hybrid_simulator_service.build_network_set(
                connectivity, hybrid_simulator.subnetworks, hybrid_simulator.dt, hybrid_simulator.projections)

        # what the adapter builds holds the added Projections only
        assert network_set.projections == []
        assert all(len(subnet.projections) == 1 for subnet in network_set.subnets)

    def test_saved_subnetworks_survive_navigation_between_steps(self):
        cherrypy.request.method = "POST"
        self.sess_mock['connectivity'] = self.connectivity.gid

        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.add_subnetwork()
            self.hybrid_controller.move_regions(subnetwork_index='1', node_indices=json.dumps([2, 5]))
            self.hybrid_controller.save_subnetworks()

            # go back to the Connectivity step and forward again, without changing the Connectivity
            cherrypy.request.method = "GET"
            previous_rules = self.hybrid_controller.set_connectivity()
            cherrypy.request.method = "POST"
            rendering_rules = self.hybrid_controller.set_connectivity(**self.sess_mock._data)
            board = self.hybrid_controller.configure_subnetworks()
            saved = self._saved_subnetworks()

        assert previous_rules['renderer'].is_first_fragment
        assert rendering_rules['renderer'].is_subnetworks_summary_fragment
        assert len(saved) == 2
        assert list(saved[1].node_indices) == [2, 5]
        # the third column comes back showing the same grouping
        assert list(board['renderer'].subnetworks[1].node_indices) == [2, 5]
        assert not board['renderer'].is_modified

    def test_unsaved_grouping_survives_navigation_between_steps(self):
        cherrypy.request.method = "POST"
        self.sess_mock['connectivity'] = self.connectivity.gid

        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.add_subnetwork()
            self.hybrid_controller.move_regions(subnetwork_index='1', node_indices=json.dumps([2, 5]))

            cherrypy.request.method = "GET"
            self.hybrid_controller.set_connectivity()
            cherrypy.request.method = "POST"
            rendering_rules = self.hybrid_controller.set_connectivity(**self.sess_mock._data)
            board = self.hybrid_controller.configure_subnetworks()
            saved = self._saved_subnetworks()

        # what was never saved is still on the board, and still only on the board
        assert list(board['renderer'].subnetworks[1].node_indices) == [2, 5]
        assert board['renderer'].is_modified
        assert rendering_rules['renderer'].is_modified
        assert len(saved) == 1

    def test_subnetworks_are_reset_when_connectivity_changes(self):
        other_connectivity = TestFactory.import_zip_connectivity(self.test_user, self.test_project,
                                                                 subject='HybridSubject')
        cherrypy.request.method = "POST"

        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.add_subnetwork()
            self.hybrid_controller.move_regions(subnetwork_index='1', node_indices=json.dumps([0, 2]))

            self.sess_mock['connectivity'] = other_connectivity.gid
            self.hybrid_controller.set_connectivity(**self.sess_mock._data)
            hybrid_simulator = self.hybrid_controller.context.hybrid_simulator
            draft = self._draft_subnetworks()

        assert hybrid_simulator.connectivity.hex == other_connectivity.gid
        assert len(hybrid_simulator.subnetworks) == 1
        assert list(hybrid_simulator.subnetworks[0].node_indices) == list(
            range(other_connectivity.number_of_regions))
        # the board was grouping the regions of the Connectivity that is no longer selected
        assert self._names(draft) == ['Subnetwork A']
        assert list(draft[0].node_indices) == list(range(other_connectivity.number_of_regions))

    def test_reset_drops_the_board_as_well(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.add_subnetwork()
            self.hybrid_controller.move_regions(subnetwork_index='1', node_indices=json.dumps([0, 2]))

            rendering_rules = self.hybrid_controller.reset_hybrid_simulator_configuration()
            draft = self._draft_subnetworks()

        assert rendering_rules['renderer'].is_first_fragment
        assert not draft

    def test_switching_project_drops_the_hybrid_configuration(self):
        other_project = TestFactory.create_project(self.test_user, "HybridSimulatorController_Other_Project")

        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.add_subnetwork()
            self.hybrid_controller._mark_selected(other_project)

            hybrid_simulator = self.hybrid_controller.context.hybrid_simulator
            draft = self._draft_subnetworks()
            last_loaded_fragment_url = self.hybrid_controller.context.last_loaded_fragment_url

        assert hybrid_simulator is None
        assert draft is None
        assert last_loaded_fragment_url is None

    def test_reselecting_the_same_project_keeps_the_hybrid_configuration(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.add_subnetwork()
            self.hybrid_controller._mark_selected(self.test_project)

            saved = self._saved_subnetworks()
            draft = self._draft_subnetworks()

        assert len(saved) == 1
        assert len(draft) == 2

    def test_hybrid_history_excludes_classic_bursts_for_now(self):
        classic_burst = BurstConfiguration(self.test_project.id)
        classic_burst.name = 'classic_burst'
        dao.store_entity(classic_burst)

        with patch('cherrypy.session', self.sess_mock, create=True):
            result = self.hybrid_controller.load_hybrid_history()

        assert result['burst_list'] == []

    def test_classic_simulator_index_is_unchanged(self):
        cherrypy.request.method = "GET"

        with patch('cherrypy.session', self.sess_mock, create=True), \
                patch('tvb.interfaces.web.controllers.decorators.TvbProfile.is_first_run', return_value=False):
            result_dict = self.simulator_controller.index()

        assert result_dict['mainContent'] == 'burst/main_burst'

    # ---------------------------------------------------------------- Monitors

    def _monitors_step(self):
        """Open the Monitors step, the way pressing Next on the Projections step opens it."""
        cherrypy.request.method = "POST"
        return self.hybrid_controller.open_monitors()

    def _choose_monitors(self, names, simulation_length='100.0'):
        """
        Submit the Monitors step, which answers with the first Monitor that has parameters.

        The simulation length is asked on the closing step, which comes after; it is set on the
        configuration here, the way an earlier pass through that step leaves it, since a Monitor's
        sampling period is checked against it.
        """
        self.hybrid_controller.context.hybrid_simulator.simulation_length = float(simulation_length)
        cherrypy.request.method = "POST"
        return self.hybrid_controller.set_monitors(monitors=names)

    def _submit_monitor_params(self, monitor_name, **data):
        cherrypy.request.method = "POST"
        return self.hybrid_controller.set_monitor_params(monitor_name, **data)

    def _configured_monitors(self):
        return self.hybrid_controller.context.hybrid_simulator.monitors

    def test_the_monitors_step_follows_the_projections(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_subnetwork()
            rendering_rules = self._monitors_step()

        renderer = rendering_rules['renderer']
        assert renderer.form_action_url == HybridSimulatorURLs.SET_MONITORS_URL
        assert renderer.previous_form_action_url == HybridSimulatorURLs.SET_PROJECTIONS_URL
        assert renderer.is_monitor_step
        # what to record: how long for is asked last, on the closing step
        names = [field.name for field in renderer.form.ordered_fields]
        assert names == ['monitors']
        # opened, not submitted: the configured Monitors are shown, and nothing is refused
        assert renderer.form.monitors.data == ['Temporal average']
        assert not renderer.form.monitors.errors

    def test_every_region_monitor_is_on_offer(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_subnetwork()
            rendering_rules = self._monitors_step()

        choices = rendering_rules['renderer'].form.monitors.trait_attribute.element_choices
        assert 'Raw recording' in choices
        assert 'EEG' in choices and 'MEG' in choices and 'BOLD' in choices
        # a surface only Monitor, and the Hybrid Simulator has no surface
        assert 'BOLD Region ROI' not in choices

    def test_choosing_monitors_stores_them_and_opens_the_first(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_subnetwork()
            rendering_rules = self._choose_monitors(['Temporal average', 'BOLD'], simulation_length='6000.0')
            stored = [type(monitor).__name__ for monitor in self._configured_monitors()]

        assert stored == ['TemporalAverageViewModel', 'BoldViewModel']

        renderer = rendering_rules['renderer']
        assert renderer.form_action_url == \
               HybridSimulatorURLs.SET_MONITOR_PARAMS_URL + '/TemporalAverageViewModel'
        assert renderer.previous_form_action_url == HybridSimulatorURLs.SET_MONITORS_URL
        assert renderer.monitor_name == 'Temporal average monitor'

    def test_a_monitor_step_does_not_offer_the_variables_of_interest(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_subnetwork()
            rendering_rules = self._choose_monitors(['Temporal average'])

        # the Hybrid Simulator assigns voi = slice(None) to every Monitor, so a selection made here
        # would be discarded before the first step is integrated
        names = [field.name for field in rendering_rules['renderer'].form.fields]
        assert 'variables_of_interest' not in names
        assert names == ['period']

    def test_each_chosen_monitor_gets_its_own_step(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_subnetwork()
            self._choose_monitors(['Temporal average', 'Global average'])
            rendering_rules = self._submit_monitor_params('TemporalAverageViewModel', period='1.0')
            period = self._configured_monitors()[0].period

        renderer = rendering_rules['renderer']
        assert renderer.form_action_url == \
               HybridSimulatorURLs.SET_MONITOR_PARAMS_URL + '/GlobalAverageViewModel'
        assert renderer.previous_form_action_url == \
               HybridSimulatorURLs.SET_MONITOR_PARAMS_URL + '/TemporalAverageViewModel'
        assert period == 1.0

    def test_the_raw_monitor_has_no_step_of_its_own(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_subnetwork()
            rendering_rules = self._choose_monitors(['Raw recording', 'Temporal average'])

        # Raw records every integration step and documents its sampling period as ignored
        assert rendering_rules['renderer'].form_action_url == \
               HybridSimulatorURLs.SET_MONITOR_PARAMS_URL + '/TemporalAverageViewModel'

    def test_a_bold_monitor_takes_an_equation_step(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_subnetwork()
            self._choose_monitors(['BOLD'], simulation_length='6000.0')
            rendering_rules = self._submit_monitor_params(
                'BoldViewModel', period='2000.0', hrf_kernel='Hrf Kernel: Mixture Of Gammas')
            kernel = type(self._configured_monitors()[0].hrf_kernel).__name__

        renderer = rendering_rules['renderer']
        assert renderer.form_action_url == \
               HybridSimulatorURLs.SET_MONITOR_EQUATION_URL + '/BoldViewModel'
        assert renderer.monitor_name == 'BOLD monitor - haemodynamic response'
        assert kernel == 'MixtureOfGammas'

    def test_a_sampling_period_below_the_step_size_is_refused(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_subnetwork()
            self._choose_monitors(['Temporal average'])
            rendering_rules = self._submit_monitor_params('TemporalAverageViewModel', period='0.001')

        # the step hands itself back, which is what the client reads as a refusal of this step, and
        # says why on the field
        renderer = rendering_rules['renderer']
        assert renderer.form_action_url == \
               HybridSimulatorURLs.SET_MONITOR_PARAMS_URL + '/TemporalAverageViewModel'
        assert 'integration step size' in str(renderer.form.period.errors)

    def test_a_sampling_period_longer_than_the_simulation_waits_for_the_length(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_subnetwork()
            # a BOLD Monitor samples every 2000 ms by default, longer than the default 1000 ms simulation
            self._choose_monitors(['Temporal average'], simulation_length='1000.0')
            rendering_rules = self._submit_monitor_params('TemporalAverageViewModel', period='2000.0')

        # the length is asked last, on the closing step, so the Monitor's step does not refuse it: the
        # launch does, if the length given there is still shorter
        assert rendering_rules['renderer'].form_action_url == HybridSimulatorURLs.SET_SIMULATION_SUMMARY_URL

    def test_the_last_monitor_reaches_the_summary(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_subnetwork()
            self._choose_monitors(['Temporal average'], simulation_length='100.0')
            rendering_rules = self._submit_monitor_params('TemporalAverageViewModel', period='1.0')

        renderer = rendering_rules['renderer']
        assert renderer.form_action_url == HybridSimulatorURLs.SET_SIMULATION_SUMMARY_URL
        assert renderer.is_simulation_summary_fragment
        assert not renderer.is_monitor_step
        assert renderer.fragment_title == "Simulation"
        # and it closes the wizard: the simulation length and name are asked last, next to Launch
        assert renderer.is_launch_fragment
        assert [field.name for field in renderer.form.ordered_fields] == \
               ['simulation_length', 'input_simulation_name_id']
        assert renderer.form.simulation_length.value == 100.0

    def test_the_summary_reports_the_output_layout(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_subnetwork()
            self._choose_monitors(['Temporal average'], simulation_length='100.0')
            rendering_rules = self._submit_monitor_params('TemporalAverageViewModel', period='1.0')

        layout = rendering_rules['renderer'].output_layout
        # one Subnetwork holding everything, so the counts trivially agree
        assert layout['is_merged'] is True
        assert layout['nodes'] == self.connectivity.number_of_regions

    def test_the_closing_step_warns_when_the_output_is_not_region_ordered(self):
        with patch.object(TvbProfile.current.web, 'RENDER_HTML', True), \
                patch('cherrypy.session', self.sess_mock, create=True):
            saved, _ = self._two_saved_subnetworks_at_the_projections()
            # the first Subnetwork watches two variables, the second one
            saved[0].dynamics.model.variables_of_interest = ('V', 'W')
            self._choose_monitors(['Temporal average'])
            html = self._submit_monitor_params('TemporalAverageViewModel', period='1.0')

        assert 'The Subnetworks watch different numbers of variables' in html
        assert '{}: 2'.format(saved[0].name) in html and '{}: 1'.format(saved[1].name) in html
        # and nothing else is summarised above the simulation length
        assert 'Samples every' not in html
        assert html.index('different numbers of variables') < html.index('name="simulation_length"')

    # ---------------------------------------------------------------- Launch

    def _reach_the_launch_step(self, simulation_length='20.0'):
        """Walk the whole wizard, ending on the step that carries the Launch button."""
        self._configured_subnetwork()
        self._choose_monitors(['Temporal average'], simulation_length=simulation_length)
        return self._submit_monitor_params('TemporalAverageViewModel', period='1.0')

    def test_a_valid_launch_is_answered_as_not_available_yet(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._reach_the_launch_step()
            bursts_before = dao.get_bursts_for_project(self.test_project.id, count=True)
            with patch.object(self.hybrid_controller.simulator_service,
                              'async_launch_and_prepare_simulation') as launch:
                # the endpoint answers the browser, so what comes back is serialized JSON
                answer = json.loads(self.hybrid_controller.launch_simulation(
                    simulation_length='30.0', input_simulation_name_id='my hybrid run'))
            simulation_length = self.hybrid_controller.context.hybrid_simulator.simulation_length

        assert 'not available yet' in answer['error']
        assert not launch.called
        assert dao.get_bursts_for_project(self.test_project.id, count=True) == bursts_before
        # the length given with the launch is still kept on the configuration
        assert simulation_length == 30.0

    def test_a_simulation_length_the_monitors_can_not_record_is_refused_at_launch(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._reach_the_launch_step()
            with patch.object(self.hybrid_controller.simulator_service,
                              'async_launch_and_prepare_simulation') as launch:
                answers = [json.loads(self.hybrid_controller.launch_simulation(
                               simulation_length=length, input_simulation_name_id='a hybrid run'))
                           for length in ('0', '-5', '0.5')]
            simulation_length = self.hybrid_controller.context.hybrid_simulator.simulation_length

        # not positive, and shorter than the Temporal average's 1 ms sampling period
        assert all('error' in answer for answer in answers), answers
        assert 'positive' in answers[0]['error'] and 'positive' in answers[1]['error']
        assert 'TemporalAverage' in answers[2]['error'] and 'record nothing' in answers[2]['error']
        assert not launch.called
        assert simulation_length == 20.0

    def test_an_unusable_simulation_name_is_refused(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._reach_the_launch_step()
            with patch.object(self.hybrid_controller.simulator_service,
                              'async_launch_and_prepare_simulation'):
                answer = json.loads(self.hybrid_controller.launch_simulation(
                    simulation_length='20.0', input_simulation_name_id='bad/name'))

        assert 'error' in answer
        assert 'Invalid simulation name' in answer['error']
