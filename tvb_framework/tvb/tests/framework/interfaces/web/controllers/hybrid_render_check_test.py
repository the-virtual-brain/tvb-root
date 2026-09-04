# -*- coding: utf-8 -*-
"""Temporary check: what the Hybrid Simulator endpoints actually put on the wire."""

from unittest.mock import patch
from uuid import UUID

import cherrypy
from cherrypy.lib.sessions import RamSession

from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.simulator.view_model import HybridSimulatorAdapterModel
from tvb.interfaces.web.controllers.common import KEY_PROJECT, KEY_USER
from tvb.interfaces.web.controllers.simulator.hybrid_simulator_controller import HybridSimulatorController
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseTransactionalControllerTest


class TestHybridRendering(BaseTransactionalControllerTest):

    def transactional_setup_method(self):
        self.hybrid_controller = HybridSimulatorController()
        self.test_user = TestFactory.create_user('HybridRender_User')
        self.test_project = TestFactory.create_project(self.test_user, "HybridRender_Project")
        self.connectivity = TestFactory.import_zip_connectivity(self.test_user, self.test_project)

        self.hybrid_simulator = HybridSimulatorAdapterModel()
        self.hybrid_simulator.connectivity = UUID(self.connectivity.gid)

        self.sess_mock = RamSession()
        self.sess_mock[KEY_USER] = self.test_user
        self.sess_mock[KEY_PROJECT] = self.test_project

    def test_what_the_endpoints_return(self):
        cherrypy.request.method = "GET"
        with patch.object(TvbProfile.current.web, 'RENDER_HTML', True), \
                patch('cherrypy.session', self.sess_mock, create=True):
            self.hybrid_controller.context.set_hybrid_simulator(self.hybrid_simulator)
            step_html = self.hybrid_controller.set_subnetworks()
            board_html = self.hybrid_controller.configure_subnetworks()

        print("\n########## SET_SUBNETWORKS (column 2) ##########")
        print(step_html[:700])
        print("\n########## CONFIGURE_SUBNETWORKS (column 3) ##########")
        print(board_html[:1500])

        assert isinstance(step_html, str), "set_subnetworks did not render HTML"
        assert isinstance(board_html, str), "configure_subnetworks did not render HTML"
        assert 'data-hybrid-context-url="/burst/hybrid/configure_subnetworks"' in step_html
        assert 'id="hybrid-subnetworks-board"' in board_html
        assert 'HYBRID_SUBNETWORKS.init(' in board_html
        assert 'hybridSaveSubnetworks()' in board_html
