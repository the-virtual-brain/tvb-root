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

    # The reused Cockpit forms declare every parameter required, so a step has to be posted whole.
    # These are the defaults of the classes the Hybrid Simulator seeds a Subnetwork with.
    MODEL_PARAMS = {'tau': '[1.0]', 'I': '[0.0]', 'a': '[-2.0]', 'b': '[-10.0]', 'c': '[0.0]',
                    'd': '[0.02]', 'e': '[3.0]', 'f': '[1.0]', 'g': '[0.0]', 'alpha': '[1.0]',
                    'beta': '[1.0]', 'gamma': '[1.0]', 'variables_of_interest': 'V'}
    NOISE_PARAMS = {'nsig': '[1.0]', 'ntau': '0.0', 'noise_seed': '42', 'equation': 'Linear'}
    EQUATION_PARAMS = {'a': '1.0', 'b': '0.0'}

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

    def _first_subnetwork_id(self):
        return self.hybrid_controller.context.hybrid_simulator.subnetworks[0].id

    def test_what_the_dynamics_endpoints_return(self):
        """
        Render the whole per Subnetwork dynamics chain. There is no JavaScript test infrastructure here,
        so this is what catches a template that does not render and a step that loses the shared dt.
        """
        with patch.object(TvbProfile.current.web, 'RENDER_HTML', True), \
                patch('cherrypy.session', self.sess_mock, create=True):
            self.hybrid_controller.context.set_hybrid_simulator(self.hybrid_simulator)

            cherrypy.request.method = "GET"
            self.hybrid_controller.set_subnetworks()

            # Next on the Subnetworks step opens the dynamics step
            cherrypy.request.method = "POST"
            dynamics_step_html = self.hybrid_controller.set_subnetworks()

            # Next on the dynamics step applies the shared dt and opens the Model of the selected
            # Subnetwork, in this same column
            model_html = self.hybrid_controller.set_subnetwork_dynamics(dt='0.1')

            # step 1 -> Model parameters
            model_params_html = self.hybrid_controller.set_subnetwork_model(model='Generic 2D Oscillator')
            # step 2 -> Integrator class. Every ModelForm parameter is required, so all are posted.
            integrator_html = self.hybrid_controller.set_subnetwork_model_params(**self.MODEL_PARAMS)
            # step 3 -> Integrator parameters, for a stochastic Integrator
            integrator_params_html = self.hybrid_controller.set_subnetwork_integrator(
                integrator='Stochastic Heun')
            # step 4 -> Noise parameters. dt is rendered disabled, so it is deliberately not posted here
            noise_params_html = self.hybrid_controller.set_subnetwork_integrator_params(
                noise='Multiplicative')
            # step 5 -> Equation parameters, for a Multiplicative Noise
            equation_html = self.hybrid_controller.set_subnetwork_noise_params(**self.NOISE_PARAMS)
            # step 6 -> the closing step
            save_step_html = self.hybrid_controller.set_subnetwork_noise_equation_params(
                **self.EQUATION_PARAMS)

            # switching Subnetwork answers with the first step of the newly selected one
            selected_html = self.hybrid_controller.select_subnetwork(subnetwork_id=self._first_subnetwork_id())

        for name, html in [('dynamics step', dynamics_step_html), ('model', model_html),
                           ('model params', model_params_html), ('integrator', integrator_html),
                           ('integrator params', integrator_params_html),
                           ('noise params', noise_params_html), ('equation', equation_html),
                           ('save step', save_step_html)]:
            assert isinstance(html, str) and html.strip(), '{} did not render HTML'.format(name)

        # the Model and Integrator are configured in this same column, so the dynamics step declares no
        # configuration for the third column, which hands it back to the Results view
        assert 'data-hybrid-context-url=""' in dynamics_step_html
        # it lists the shared dt and offers the Subnetwork selector for the steps stacked under it
        assert 'Integration step size' in dynamics_step_html
        assert 'hybridSelectSubnetwork(' in dynamics_step_html
        assert 'Subnetwork A' in dynamics_step_html
        # the selector keeps working once this step is locked, which is what that marker is read for
        assert 'data-hybrid-keep-enabled="true"' in dynamics_step_html

        # the steps post to their own urls
        assert 'action="/burst/hybrid/set_subnetwork_model"' in model_html
        assert 'action="/burst/hybrid/set_subnetwork_model_params"' in model_params_html
        assert 'action="/burst/hybrid/set_subnetwork_integrator"' in integrator_html
        assert 'action="/burst/hybrid/set_subnetwork_integrator_params"' in integrator_params_html
        assert 'action="/burst/hybrid/set_subnetwork_noise_params"' in noise_params_html
        assert 'action="/burst/hybrid/set_subnetwork_noise_equation_params"' in equation_html

        # dt shows on the Integrator parameters step, and cannot be edited there
        assert 'name="dt"' in integrator_params_html
        assert 'disabled' in integrator_params_html

        # the closing step offers the save action rather than another Next
        assert 'hybridSaveSubnetworkDynamics()' in save_step_html
        assert 'Save Configuration' in save_step_html

        # switching Subnetwork restarts its configuration at the Model step, rendered once
        assert isinstance(selected_html, str)
        assert 'action="/burst/hybrid/set_subnetwork_model"' in selected_html
