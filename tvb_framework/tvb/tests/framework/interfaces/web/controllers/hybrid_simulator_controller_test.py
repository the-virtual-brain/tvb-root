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
        Subnetwork configuration step for it.
        """
        self.hybrid_controller.context.set_hybrid_simulator(self.session_stored_hybrid_simulator)
        return self.hybrid_controller.set_subnetworks()

    @staticmethod
    def _assert_valid_partition(subnetworks, number_of_regions):
        """
        Every Connectivity node must show up in exactly one Subnetwork, with its original index.
        """
        assigned = []
        for subnetwork in subnetworks:
            assigned.extend(subnetwork['node_indices'])

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
            hybrid_simulator = self.hybrid_controller.context.hybrid_simulator

        renderer = rendering_rules['renderer']
        assert renderer.is_subnetworks_summary_fragment
        assert renderer.form_action_url == HybridSimulatorURLs.SET_SUBNETWORKS_URL
        assert len(renderer.region_labels) == self.connectivity.number_of_regions

        assert len(hybrid_simulator.subnetworks) == 1
        assert hybrid_simulator.subnetworks[0].name == 'Subnetwork A'
        assert list(hybrid_simulator.subnetworks[0].node_indices) == list(
            range(self.connectivity.number_of_regions))

        rendered_state = json.loads(renderer.subnetworks_json)
        self._assert_valid_partition(rendered_state['subnetworks'], self.connectivity.number_of_regions)

    def test_subnetworks_step_lists_every_subnetwork_with_its_regions(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.add_subnetwork()
            self.hybrid_controller.move_regions(subnetwork_index='1', node_indices=json.dumps([1, 3]))
            rendering_rules = self.hybrid_controller.set_subnetworks()

        rows = rendering_rules['renderer'].subnetwork_rows
        assert [row['name'] for row in rows] == ['Subnetwork A', 'Subnetwork B']
        assert rows[0]['count'] == self.connectivity.number_of_regions - 2
        assert rows[1]['count'] == 2

        # the rows keep the original Connectivity indices next to the labels
        assert [region['index'] for region in rows[1]['regions']] == [1, 3]
        assert all(region['label'] for region in rows[1]['regions'])
        assert 1 not in [region['index'] for region in rows[0]['regions']]

    def test_subnetworks_step_shows_an_empty_subnetwork_as_empty(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.add_subnetwork()
            rendering_rules = self.hybrid_controller.set_subnetworks()

        rows = rendering_rules['renderer'].subnetwork_rows
        assert rows[1]['count'] == 0
        assert rows[1]['regions'] == []

    def test_subnetworks_step_offers_configure_and_a_not_yet_available_next(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            rendering_rules = self._configured_hybrid_simulator()

        renderer = rendering_rules['renderer']
        assert renderer.include_configure_subnetworks_button
        assert renderer.include_previous_button
        assert renderer.previous_button_label == 'Previous'
        assert renderer.include_next_button
        # Model and Integrator configuration is a later phase, so Next has nowhere to go yet
        assert not renderer.next_button_enabled

    def test_configure_subnetworks_opens_the_grouping_board(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            rendering_rules = self.hybrid_controller.configure_subnetworks()

        renderer = rendering_rules['renderer']
        assert renderer.is_subnetwork_fragment
        assert not renderer.is_subnetworks_summary_fragment
        assert renderer.form_action_url == HybridSimulatorURLs.CONFIGURE_SUBNETWORKS_URL
        # the board leads back to the Subnetworks step, under a label that says so
        assert renderer.previous_form_action_url == HybridSimulatorURLs.SET_SUBNETWORKS_URL
        assert renderer.previous_button_label == 'Back to Simulator'
        assert not renderer.include_next_button
        assert not renderer.include_configure_subnetworks_button
        assert len(renderer.region_labels) == self.connectivity.number_of_regions

    def test_grouping_on_the_board_shows_up_on_the_subnetworks_step(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.configure_subnetworks()
            self.hybrid_controller.add_subnetwork()
            self.hybrid_controller.move_regions(subnetwork_index='1', node_indices=json.dumps([4, 5]))
            # "Back to Simulator" is a plain load of the Subnetworks step
            rendering_rules = self.hybrid_controller.set_subnetworks()

        rows = rendering_rules['renderer'].subnetwork_rows
        assert len(rows) == 2
        assert [region['index'] for region in rows[1]['regions']] == [4, 5]

    def test_configure_subnetworks_without_connectivity_returns_to_first_fragment(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self.hybrid_controller.context.set_hybrid_simulator(HybridSimulatorAdapterModel())
            rendering_rules = self.hybrid_controller.configure_subnetworks()

        assert rendering_rules['renderer'].is_first_fragment

    def test_set_subnetworks_without_connectivity_returns_to_first_fragment(self):
        with patch('cherrypy.session', self.sess_mock, create=True):
            self.hybrid_controller.context.set_hybrid_simulator(HybridSimulatorAdapterModel())
            rendering_rules = self.hybrid_controller.set_subnetworks()

        assert rendering_rules['renderer'].form_action_url == HybridSimulatorURLs.SET_CONNECTIVITY_URL
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
            hybrid_simulator = self.hybrid_controller.context.hybrid_simulator

        assert result['status'] == 'ok'
        assert result['subnetworks'][0]['name'] == 'Cortex'
        assert hybrid_simulator.subnetworks[0].name == 'Cortex'

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
            hybrid_simulator = self.hybrid_controller.context.hybrid_simulator

        assert result['status'] == 'ok'
        assert result['subnetworks'][1]['node_indices'] == moved
        for node_index in moved:
            assert node_index not in result['subnetworks'][0]['node_indices']
        self._assert_valid_partition(result['subnetworks'], self.connectivity.number_of_regions)

        # the same grouping is stored in session, still using the original Connectivity indices
        assert list(hybrid_simulator.subnetworks[1].node_indices) == moved

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

    def test_subnetworks_survive_navigation_between_steps(self):
        cherrypy.request.method = "POST"
        self.sess_mock['connectivity'] = self.connectivity.gid

        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.add_subnetwork()
            self.hybrid_controller.move_regions(subnetwork_index='1', node_indices=json.dumps([2, 5]))

            # go back to the Connectivity step and forward again, without changing the Connectivity
            cherrypy.request.method = "GET"
            previous_rules = self.hybrid_controller.set_connectivity()
            cherrypy.request.method = "POST"
            rendering_rules = self.hybrid_controller.set_connectivity(**self.sess_mock._data)
            hybrid_simulator = self.hybrid_controller.context.hybrid_simulator

        assert previous_rules['renderer'].is_first_fragment
        assert rendering_rules['renderer'].is_subnetworks_summary_fragment
        assert len(hybrid_simulator.subnetworks) == 2
        assert list(hybrid_simulator.subnetworks[1].node_indices) == [2, 5]

    def test_subnetworks_are_reset_when_connectivity_changes(self):
        other_connectivity = TestFactory.import_zip_connectivity(self.test_user, self.test_project,
                                                                 subject='HybridSubject')
        cherrypy.request.method = "POST"

        with patch('cherrypy.session', self.sess_mock, create=True):
            self._configured_hybrid_simulator()
            self.hybrid_controller.add_subnetwork()

            self.sess_mock['connectivity'] = other_connectivity.gid
            self.hybrid_controller.set_connectivity(**self.sess_mock._data)
            hybrid_simulator = self.hybrid_controller.context.hybrid_simulator

        assert hybrid_simulator.connectivity.hex == other_connectivity.gid
        assert len(hybrid_simulator.subnetworks) == 1
        assert list(hybrid_simulator.subnetworks[0].node_indices) == list(
            range(other_connectivity.number_of_regions))

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
