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
from unittest.mock import patch
from uuid import UUID

import cherrypy
from cherrypy.lib.sessions import RamSession

from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.simulator.view_model import HybridSimulatorAdapterModel
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.entities.storage import dao
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
        assert not result_dict['errors']

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
        # the third column follows this step and configures each Subnetwork's dynamics
        assert renderer.context_form_url == HybridSimulatorURLs.CONFIGURE_SUBNETWORK_DYNAMICS_URL
        assert renderer.context_title == "Subnetwork dynamics"

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

    def test_the_shared_dt_reaches_every_subnetwork_integrator(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            saved = self._two_subnetworks()
            self._open_dynamics_step()
            # the shared dt is the dynamics step's own field, applied by submitting that step
            cherrypy.request.method = "POST"
            self.hybrid_controller.set_subnetwork_dynamics(dt='0.5')
            draft = self._dynamics_draft()
            hybrid_simulator = self.hybrid_controller.context.hybrid_simulator
            saved_dts = [subnetwork.dynamics.integrator.dt for subnetwork in saved]

        assert hybrid_simulator.dt == 0.5
        # including the Subnetworks configured before the value was last changed
        for dynamics in draft.values():
            assert dynamics.integrator.dt == 0.5
        assert saved_dts == [0.5, 0.5]

    def test_applying_the_shared_dt_is_not_reported_as_an_unsaved_change(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._two_subnetworks()
            self._open_dynamics_step()
            cherrypy.request.method = "POST"
            rendering_rules = self.hybrid_controller.set_subnetwork_dynamics(dt='0.5')
            column = self.hybrid_controller.configure_subnetwork_dynamics()

        # dt is a simulation wide setting applied on its own step, not a pending per Subnetwork edit
        assert not rendering_rules['renderer'].is_modified
        assert not column['renderer'].is_modified

    def test_integrator_params_keep_the_shared_dt_when_it_is_not_posted(self):
        """
        dt is rendered disabled, and a disabled input is not submitted, so the step has to put the shared
        value back into the posted data itself.
        """
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            cherrypy.request.method = "POST"
            self.hybrid_controller.set_subnetwork_dynamics(dt='0.25')
            self.hybrid_controller.configure_subnetwork_dynamics()

            self.hybrid_controller.set_subnetwork_integrator(integrator='Heun')
            # deliberately no dt in the posted data, the way the browser submits that step
            self.hybrid_controller.set_subnetwork_integrator_params()
            draft = self._dynamics_draft()
            saved = self._saved_subnetworks()

        assert draft[saved[0].id].integrator.dt == 0.25

    def test_resubmitting_the_same_integrator_class_keeps_its_edited_params(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self.hybrid_controller.configure_subnetwork_dynamics()
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
            self.hybrid_controller.configure_subnetwork_dynamics()
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
            self.hybrid_controller.configure_subnetwork_dynamics()
            saved = self._saved_subnetworks()

            cherrypy.request.method = "POST"
            self.hybrid_controller.set_subnetwork_integrator(integrator='Heun')
            self.hybrid_controller.set_subnetwork_integrator_params()

            before_save = type(self._saved_subnetworks()[0].dynamics.integrator).__name__
            column = self.hybrid_controller.configure_subnetwork_dynamics()

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
            self.hybrid_controller.configure_subnetwork_dynamics()

            cherrypy.request.method = "POST"
            self.hybrid_controller.set_subnetwork_integrator(integrator='Stochastic Heun')
            unsaved = self.hybrid_controller.configure_subnetwork_dynamics()

            saved_rules = self.hybrid_controller.save_subnetwork_dynamics()
            saved_column = self.hybrid_controller.configure_subnetwork_dynamics()
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
            self.hybrid_controller.configure_subnetwork_dynamics()

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

    def test_selecting_a_subnetwork_loads_its_own_configuration(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            saved = self._two_subnetworks()
            self._open_dynamics_step()
            self.hybrid_controller.configure_subnetwork_dynamics()

            cherrypy.request.method = "POST"
            column = self.hybrid_controller.select_subnetwork(subnetwork_id=saved[1].id)

        renderer = column['renderer']
        assert renderer.selected_subnetwork == saved[1].id
        selected = [choice for choice in renderer.subnetwork_choices if choice['is_selected']]
        assert len(selected) == 1
        assert selected[0]['name'] == saved[1].name
        # the sub wizard reopens on its first step for the newly selected Subnetwork
        assert renderer.step_renderer.form_action_url == HybridSimulatorURLs.SET_SUBNETWORK_MODEL_URL

    def test_dynamics_survive_renaming_the_subnetwork(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self.hybrid_controller.configure_subnetwork_dynamics()

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
            self.hybrid_controller.configure_subnetwork_dynamics()

            cherrypy.request.method = "POST"
            self.hybrid_controller.select_subnetwork(subnetwork_id=removed_id)
            self.hybrid_controller.set_subnetwork_model(model='Kuramoto Oscillator')

            self.hybrid_controller.remove_subnetwork(subnetwork_index='1')
            self.hybrid_controller.save_subnetworks()
            self.hybrid_controller.configure_subnetwork_dynamics()
            draft = self._dynamics_draft()
            selected = self.hybrid_controller.context.selected_subnetwork

        assert removed_id not in draft
        # and the selection falls back to a Subnetwork that still exists
        assert selected in draft

    def test_saving_rejects_a_model_parameter_that_does_not_fit_the_subnetwork(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self._open_dynamics_step()
            self.hybrid_controller.configure_subnetwork_dynamics()

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

    # ---------------------------------------------------------------- navigation

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
