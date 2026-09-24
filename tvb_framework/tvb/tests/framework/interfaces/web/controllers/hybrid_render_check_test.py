# -*- coding: utf-8 -*-
"""Temporary check: what the Hybrid Simulator endpoints actually put on the wire."""

import re
from unittest.mock import patch
from uuid import UUID

import cherrypy
from cherrypy.lib.sessions import RamSession

from tvb.adapters.forms.model_forms import ModelsEnum
from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.simulator.view_model import HybridSimulatorAdapterModel
from tvb.core.entities.model.model_burst import Dynamic
from tvb.core.entities.storage import dao
from tvb.simulator.integrators import HeunDeterministic
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

            # the dynamics step again, now that this Subnetwork carries a stochastic Integrator
            cherrypy.request.method = "GET"
            stochastic_step_html = self.hybrid_controller.set_subnetwork_dynamics()
            cherrypy.request.method = "POST"

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

        # each box describes its Subnetwork with the names the selectors offered, and it is the only
        # place this step does so - a second summary above the boxes used to repeat all of it
        assert 'Generic 2D Oscillator' in dynamics_step_html
        assert 'Heun' in dynamics_step_html
        assert 'hybrid-dynamics-summary' not in dynamics_step_html
        assert dynamics_step_html.count('Generic 2D Oscillator') == 1

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

        # a stochastic Integrator names its Noise in the box as well
        assert 'Stochastic Heun' in stochastic_step_html
        assert 'Multiplicative noise' in stochastic_step_html

        # the closing step offers the save action rather than another Next
        assert 'hybridSaveSubnetworkDynamics()' in save_step_html
        assert 'Save Configuration' in save_step_html

        # Switching Subnetwork answers with its whole configuration: every step, in order, all but the
        # last read only. This is the state the wizard leaves finished steps in, produced server side,
        # so what reaches the browser can be asserted here rather than only in a browser.
        assert isinstance(selected_html, str)
        actions = re.findall(r'<form[^>]*action="([^"]*)"', selected_html)
        assert actions == ['/burst/hybrid/set_subnetwork_model',
                           '/burst/hybrid/set_subnetwork_model_params',
                           '/burst/hybrid/set_subnetwork_integrator',
                           '/burst/hybrid/set_subnetwork_integrator_params',
                           '/burst/hybrid/set_subnetwork_noise_params',
                           '/burst/hybrid/set_subnetwork_noise_equation_params',
                           '/burst/hybrid/save_subnetwork_dynamics']
        # every step but the closing one has its fields disabled and its buttons hidden
        assert len(re.findall(r'<fieldset\s+disabled', selected_html)) == 6
        assert 'visibility: hidden' in selected_html
        # and the closing one is the live step, so nothing in it is hidden
        last_form = selected_html[selected_html.rindex('<form'):]
        assert 'visibility: hidden' not in last_form
        assert 'hybridSaveSubnetworkDynamics()' in last_form
        # the values on screen are the configured ones, not defaults
        assert 'Stochastic Heun' in selected_html

    def test_what_the_region_model_panel_returns(self):
        """
        Render the Set up region Model panel, in both states it can be in: with a matching model
        configuration on offer, and with none.
        """
        with patch.object(TvbProfile.current.web, 'RENDER_HTML', True), \
                patch('cherrypy.session', self.sess_mock, create=True):
            self.hybrid_controller.context.set_hybrid_simulator(self.hybrid_simulator)

            cherrypy.request.method = "GET"
            self.hybrid_controller.set_subnetworks()
            cherrypy.request.method = "POST"
            self.hybrid_controller.set_subnetworks()
            self.hybrid_controller.set_subnetwork_dynamics(dt='0.1')
            # the action sits on the Model parameters step, which follows the Model class one
            model_html = self.hybrid_controller.set_subnetwork_model(model='Generic 2D Oscillator')

            # nothing saved yet, so nothing can be placed on the regions
            cherrypy.request.method = "GET"
            empty_html = self.hybrid_controller.configure_region_model()

            dao.store_entity(Dynamic(
                'render_check_dyn', self.test_user.id, ModelsEnum.GENERIC_2D_OSCILLATOR.value.__name__,
                '[["tau", 1.0], ["a", -2.0]]', HeunDeterministic.__name__, None))
            # and one built on a Model class this Subnetwork is not configured with
            dao.store_entity(Dynamic(
                'render_check_other', self.test_user.id, ModelsEnum.KURAMOTO.value.__name__,
                '[["omega", 1.0]]', HeunDeterministic.__name__, None))
            panel_html = self.hybrid_controller.configure_region_model()

        # the Model parameters step offers the action, which fills the third column rather than opening
        # a page of its own
        assert 'Set up region Model' in model_html
        assert 'hybridConfigureRegionModel()' in model_html

        # with no matching configuration saved, the panel links to where they are defined
        assert 'HYBRID_REGION_MODEL.init(' not in empty_html
        assert 'Phase plane page</a>' in empty_html
        assert 'href="/burst/dynamic"' in empty_html
        assert 'target="_blank"' in empty_html
        # and it names the Model the way the Model selector named it
        assert 'Generic 2D Oscillator' in empty_html

        # with one saved, the region list is drawn from it
        assert 'HYBRID_REGION_MODEL.init(' in panel_html
        assert 'id="hybrid-region-model-list"' in panel_html
        assert 'render_check_dyn' in panel_html
        # only configurations on this Subnetwork's Model class are offered
        assert 'render_check_other' not in panel_html
        # placing, putting the result into the Model parameters, and selecting every region at once
        assert 'id="hybrid-region-apply"' in panel_html
        assert 'id="hybrid-region-submit"' in panel_html
        assert 'id="hybrid-region-select-all"' in panel_html

    def test_what_the_projections_step_returns(self):
        """
        Render the Projections step. This is the Phase 3 checkpoint and the Phase 4 one at once: it only
        renders if the configuration really does translate into a NetworkSet.
        """
        with patch.object(TvbProfile.current.web, 'RENDER_HTML', True), \
                patch('cherrypy.session', self.sess_mock, create=True):
            self.hybrid_controller.context.set_hybrid_simulator(self.hybrid_simulator)

            cherrypy.request.method = "GET"
            self.hybrid_controller.set_subnetworks()
            cherrypy.request.method = "POST"
            self.hybrid_controller.set_subnetworks()
            self.hybrid_controller.set_subnetwork_dynamics(dt='0.1')
            self.hybrid_controller.set_subnetwork_model(model='Generic 2D Oscillator')
            self.hybrid_controller.set_subnetwork_model_params(**self.MODEL_PARAMS)
            self.hybrid_controller.set_subnetwork_integrator(integrator='Heun')
            closing_html = self.hybrid_controller.set_subnetwork_integrator_params()
            self.hybrid_controller.save_subnetwork_dynamics()

            projections_html = self.hybrid_controller.set_projections()

        # the closing step of the Subnetwork configuration offers both storing and moving on
        assert 'hybridSaveSubnetworkDynamics()' in closing_html
        assert "hybridSubmitTo(this.parentElement, '/burst/hybrid/set_projections')" in closing_html

        # and the Projections step lists what the configuration produced
        assert 'action="/burst/hybrid/set_projections"' in projections_html
        assert 'Intra' in projections_html
        assert 'Coupling variables' in projections_html
        assert 'hybrid-projections-summary' in projections_html

    def _reach_the_projections(self):
        """Walk the wizard up to the Projections step, which is where the global configuration starts."""
        cherrypy.request.method = "GET"
        self.hybrid_controller.set_subnetworks()
        cherrypy.request.method = "POST"
        self.hybrid_controller.set_subnetworks()
        self.hybrid_controller.set_subnetwork_dynamics(dt='0.1')
        self.hybrid_controller.set_subnetwork_model(model='Generic 2D Oscillator')
        self.hybrid_controller.set_subnetwork_model_params(**self.MODEL_PARAMS)
        self.hybrid_controller.set_subnetwork_integrator(integrator='Heun')
        self.hybrid_controller.set_subnetwork_integrator_params()
        self.hybrid_controller.save_subnetwork_dynamics()
        return self.hybrid_controller.set_projections()

    def test_what_the_monitor_steps_return(self):
        """
        Render the whole global configuration chain, a projection Monitor and a BOLD one included. There
        is no JavaScript test infrastructure here, so this is what catches a step that does not render,
        one that posts to the wrong url, and a Monitor form still offering the variables of interest.
        """
        with patch.object(TvbProfile.current.web, 'RENDER_HTML', True), \
                patch('cherrypy.session', self.sess_mock, create=True):
            self.hybrid_controller.context.set_hybrid_simulator(self.hybrid_simulator)

            projections_html = self._reach_the_projections()

            # Next on the Projections step opens the global configuration
            monitors_html = self.hybrid_controller.set_monitors()

            # a plain Monitor followed by one carrying datatype fields
            tavg_html = self.hybrid_controller.set_monitors(
                simulation_length='6000.0', monitors=['Temporal average', 'EEG'])
            eeg_html = self.hybrid_controller.set_monitor_params(
                'TemporalAverageViewModel', period='1.0')

            # and a BOLD Monitor, which takes an Equation step of its own after its parameters
            bold_html = self.hybrid_controller.set_monitors(
                simulation_length='6000.0', monitors=['BOLD'])
            equation_html = self.hybrid_controller.set_monitor_params(
                'BoldViewModel', period='2000.0', hrf_kernel='Hrf Kernel: Mixture Of Gammas')

        # the Projections step now carries a way onward
        assert "hybridSubmitTo(this.parentElement, '/burst/hybrid/set_monitors')" in projections_html

        # the global configuration step asks what to record and for how long
        assert 'action="/burst/hybrid/set_monitors"' in monitors_html
        assert 'simulation_length' in monitors_html
        assert 'BOLD' in monitors_html

        # the first chosen Monitor's own step, with a legend and without the variables of interest
        assert 'action="/burst/hybrid/set_monitor_params/TemporalAverageViewModel"' in tavg_html
        assert 'Temporal average monitor' in tavg_html
        assert 'name="variables_of_interest"' not in tavg_html

        # the EEG step renders the datatype fields it needs
        assert 'action="/burst/hybrid/set_monitor_params/EEGViewModel"' in eeg_html
        assert 'EEG monitor' in eeg_html
        assert 'name="sensors"' in eeg_html
        assert 'name="projection"' in eeg_html
        assert 'name="region_mapping"' in eeg_html
        assert 'name="variables_of_interest"' not in eeg_html

        # BOLD carries its haemodynamic response kernel, and its Equation follows on its own step
        assert 'action="/burst/hybrid/set_monitor_params/BoldViewModel"' in bold_html
        assert 'name="hrf_kernel"' in bold_html
        assert 'name="variables_of_interest"' not in bold_html
        assert 'action="/burst/hybrid/set_monitor_equation/BoldViewModel"' in equation_html
        assert 'haemodynamic response' in equation_html

    def test_what_the_simulation_summary_returns(self):
        """
        Render the closing step of the global configuration, which reports what the Monitors record.
        """
        with patch.object(TvbProfile.current.web, 'RENDER_HTML', True), \
                patch('cherrypy.session', self.sess_mock, create=True):
            self.hybrid_controller.context.set_hybrid_simulator(self.hybrid_simulator)

            self._reach_the_projections()
            self.hybrid_controller.set_monitors(simulation_length='100.0',
                                                monitors=['Raw recording', 'Temporal average'])
            summary_html = self.hybrid_controller.set_monitor_params(
                'TemporalAverageViewModel', period='1.0')

        assert 'action="/burst/hybrid/set_simulation_summary"' in summary_html
        assert 'hybrid-simulation-summary' in summary_html
        assert 'Raw' in summary_html
        assert 'every integration step' in summary_html
        assert '100.0 ms of simulated time' in summary_html
        # one Subnetwork, so the variable counts trivially agree and the output is region ordered
        assert 'in the original region order' in summary_html
        # and the step closes the wizard: a name for this simulation, and the Launch button
        assert 'name="input_simulation_name_id"' in summary_html
        assert 'hybridLaunchSimulation(this.parentElement)' in summary_html
        assert 'disabled="disabled"' not in summary_html
