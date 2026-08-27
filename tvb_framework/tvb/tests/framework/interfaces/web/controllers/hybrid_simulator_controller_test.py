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
        assert rendering_rules['renderer'].is_subnetwork_fragment

    def test_set_connectivity_missing_value_stays_on_connectivity_fragment(self):
        cherrypy.request.method = "POST"

        with patch('cherrypy.session', self.sess_mock, create=True):
            self.hybrid_controller.context.set_hybrid_simulator(HybridSimulatorAdapterModel())
            rendering_rules = self.hybrid_controller.set_connectivity()

        renderer = rendering_rules['renderer']
        assert renderer.form_action_url == HybridSimulatorURLs.SET_CONNECTIVITY_URL
        assert renderer.is_first_fragment
        assert renderer.form.connectivity.errors

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
