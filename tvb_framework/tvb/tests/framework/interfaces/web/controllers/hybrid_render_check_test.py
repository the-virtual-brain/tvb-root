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
from tvb.interfaces.web.controllers.decorators import env
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

            # Next on the dynamics step applies the shared dt and opens the editor of the selected
            # Subnetwork, in this same column
            editor_html = self.hybrid_controller.set_subnetwork_dynamics()
            model_html = editor_html

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

            # switching Subnetwork answers with the editor of the newly selected one
            selected_html = self.hybrid_controller.select_subnetwork(subnetwork_id=self._first_subnetwork_id())

            # the editor as re-rendered after a class change
            cherrypy.request.method = "GET"
            reloaded_html = self.hybrid_controller.subnetwork_editor()

            # a Model parameters section rendered read only, as the record of a finished step
            dynamics, _ = self.hybrid_controller._selected_dynamics()
            read_only_rules = self.hybrid_controller._model_params_step_rules(dynamics)
            read_only_rules.is_read_only = True
            read_only_html = env.get_template('hybrid_simulator_fragment.html').render(**read_only_rules.to_dict())

        for name, html in [('dynamics step', dynamics_step_html), ('model', model_html),
                           ('model params', model_params_html), ('integrator', integrator_html),
                           ('integrator params', integrator_params_html),
                           ('noise params', noise_params_html), ('equation', equation_html),
                           ('save step', save_step_html)]:
            assert isinstance(html, str) and html.strip(), '{} did not render HTML'.format(name)

        # the Model and Integrator are configured in this same column, so the dynamics step declares no
        # configuration for the third column, which hands it back to the Results view
        assert 'data-hybrid-context-url=""' in dynamics_step_html
        # it offers the Subnetwork selector for the editor stacked under it, and has no field of its own:
        # the step size is edited on the Subnetworks' Integrators
        assert 'name="dt"' not in dynamics_step_html
        assert '<fieldset' not in dynamics_step_html
        assert 'hybridSelectSubnetwork(' in dynamics_step_html
        assert 'Subnetwork A' in dynamics_step_html
        # the selector keeps working once this step is locked, which is what that marker is read for
        assert 'data-hybrid-keep-enabled="true"' in dynamics_step_html
        # and it is a row of tabs, the Subnetwork being configured the selected one
        assert 'role="tablist"' in dynamics_step_html
        assert dynamics_step_html.count('role="tab"') == 1
        assert 'aria-selected="true"' in dynamics_step_html

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

        # The client re-renders the editor when a class chooser changes, and tells those apart from the
        # parameter fields by name (hybrid_simulator.js HYBRID_CLASS_FIELDS). The Noise is chosen with
        # radio buttons, the Equation with a select.
        assert '<select name="model"' in model_html
        assert '<select name="integrator"' in integrator_html
        assert re.search(r'<input[^>]*type="radio"[^>]*name="noise"|<input[^>]*name="noise"[^>]*type="radio"',
                         integrator_params_html)
        assert '<select name="equation"' in noise_params_html

        # dt is edited on the Integrator parameters section, as the step size of every Subnetwork
        dt_input = re.search(r'<input[^>]*name="dt"[^>]*>', integrator_params_html)
        assert dt_input
        assert 'disabled' not in dt_input.group(0)

        # a stochastic Integrator names its Noise in the box as well
        assert 'Stochastic Heun' in stochastic_step_html
        assert 'Multiplicative noise' in stochastic_step_html

        # the row closing the editor stores the dynamics, and moves on only by storing them first
        assert 'hybridSaveSubnetworkDynamics()' in save_step_html
        assert 'Save Configuration' in save_step_html
        assert "hybridSaveSubnetworkDynamics('/burst/hybrid/set_projections')" in save_step_html
        assert 'Save and continue' in save_step_html
        assert "hybridSubmitTo(this.parentElement, '/burst/hybrid/set_projections')" not in save_step_html
        assert 'This Subnetwork is configured' not in save_step_html

        # Next on the dynamics step answers with the editor: every section, editable, then the actions
        assert 'data-hybrid-editor="true"' in editor_html
        assert re.findall(r'<form[^>]*action="([^"]*)"', editor_html) == [
            '/burst/hybrid/set_subnetwork_model', '/burst/hybrid/set_subnetwork_model_params',
            '/burst/hybrid/set_subnetwork_integrator', '/burst/hybrid/set_subnetwork_integrator_params',
            '/burst/hybrid/save_subnetwork_dynamics']

        # Switching Subnetwork answers with its editor: every section of its configuration at once, in
        # order, all of them editable, and no Next or Previous between them.
        for html in (selected_html, reloaded_html):
            assert isinstance(html, str)
            actions = re.findall(r'<form[^>]*action="([^"]*)"', html)
            assert actions == ['/burst/hybrid/set_subnetwork_model',
                               '/burst/hybrid/set_subnetwork_model_params',
                               '/burst/hybrid/set_subnetwork_integrator',
                               '/burst/hybrid/set_subnetwork_integrator_params',
                               '/burst/hybrid/set_subnetwork_noise_params',
                               '/burst/hybrid/set_subnetwork_noise_equation_params',
                               '/burst/hybrid/save_subnetwork_dynamics']
            assert not re.findall(r'<fieldset\s+disabled', html)
            assert 'visibility: hidden' not in html
            sections = html[:html.rindex('<form')]
            assert 'name="next"' not in sections
            assert 'name="previous"' not in sections
            assert sections.count('class="hybrid-editor-section ') == 6
            # the Noise is nested one level into the Integrator, its Equation one level further
            levels = re.findall(r'action="/burst/hybrid/([a-z_]+)"[^>]*\s+class="hybrid-editor-section hybrid-editor-level-(\d)"', html)
            assert dict(levels) == {'set_subnetwork_model': '0', 'set_subnetwork_model_params': '0',
                                    'set_subnetwork_integrator': '0', 'set_subnetwork_integrator_params': '0',
                                    'set_subnetwork_noise_params': '1',
                                    'set_subnetwork_noise_equation_params': '2'}
            # the region Model action is offered, on the section it edits
            assert 'hybridConfigureRegionModel()' in sections
            # and the row closing it carries the three actions
            last_form = html[html.rindex('<form'):]
            assert "hybridPreviousStep(this.parentElement, '/burst/hybrid/set_subnetworks')" in last_form
            assert 'hybridSaveSubnetworkDynamics()' in last_form
            assert 'Save and continue' in last_form
        # the editor carries no title of its own: the selected tab names the Subnetwork
        assert 'Dynamics of' not in selected_html
        # the Model and Integrator sections are framed apart from the simulation wide fields, and the
        # row of actions, which stores every Subnetwork, is left outside that frame
        frame = selected_html[selected_html.index('class="hybrid-editor-fields"'):selected_html.rindex('<form')]
        assert frame.count('class="hybrid-editor-section ') == 6
        assert 'hybrid-editor-actions' not in frame

        # a read only Model parameters section does not offer Set up region Model
        region_model_button = read_only_html[read_only_html.index('id="configRegionModelParam"') - 200:]
        region_model_button = region_model_button[:region_model_button.index('</button>')]
        assert 'visibility: hidden' in region_model_button

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
            self.hybrid_controller.set_subnetwork_dynamics()
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
            self.hybrid_controller.set_subnetwork_dynamics()
            self.hybrid_controller.set_subnetwork_model(model='Generic 2D Oscillator')
            self.hybrid_controller.set_subnetwork_model_params(**self.MODEL_PARAMS)
            self.hybrid_controller.set_subnetwork_integrator(integrator='Heun')
            closing_html = self.hybrid_controller.set_subnetwork_integrator_params()
            self.hybrid_controller.save_subnetwork_dynamics()

            projections_html = self.hybrid_controller.set_projections()
            (projection,) = self.hybrid_controller.context.hybrid_simulator.projections
            key = self.hybrid_controller.hybrid_simulator_service.projection_key(projection)

            # the editor the step declares for the third column, answering a click on the Projection's
            # tab, and its answers to an edit
            cherrypy.request.method = "GET"
            configure_html = self.hybrid_controller.configure_projections()
            cherrypy.request.method = "POST"
            editor_html = self.hybrid_controller.select_projection(projection_key=key)
            edit = {'source_cvar': 'V', 'target_cvar': 'V', 'scale': '2.0', 'coupling': 'Linear',
                    'a': '[0.1]', 'b': '[0.0]'}
            applied_html = self.hybrid_controller.set_projection(projection_key=key, cv='3.0', **edit)
            refused_html = self.hybrid_controller.set_projection(projection_key=key, cv='0', **edit)
            kuramoto_html = self.hybrid_controller.set_projection(
                projection_key=key, cv='3.0', **dict(edit, coupling='Kuramoto'))

            # the editor, and the step, with that Projection added, and removed again
            added_html = self.hybrid_controller.set_projection_included(projection_key=key, included='true')
            added_step_html = self.hybrid_controller.set_projections()
            excluded_html = self.hybrid_controller.set_projection_included(projection_key=key, included='false')
            excluded_step_html = self.hybrid_controller.set_projections()

        # the row closing the Subnetwork editor offers storing, and moving on by storing first
        assert 'hybridSaveSubnetworkDynamics()' in closing_html
        assert "hybridSaveSubnetworkDynamics('/burst/hybrid/set_projections')" in closing_html

        # and the Projections step lists what the simulation includes - nothing, until a Projection is
        # added - and declares the editor for the third column
        assert 'action="/burst/hybrid/set_projections"' in projections_html
        assert 'data-hybrid-context-url="/burst/hybrid/configure_projections"' in projections_html
        assert '<table' not in projections_html
        assert 'No Projection is added yet' in projections_html
        # without editing anything itself
        assert 'hybrid-projection-editor' not in projections_html
        assert 'hybridSelectProjection(' not in projections_html

        # once added, it is listed by name, as it was set up before being added
        assert '<span class="hybrid-projection-kind is-intra">Intra</span>' in added_step_html
        assert 'V &rarr; V' in added_step_html
        # the list shows what connects what, nothing of how: that stays in the editor
        for header in ('Kind', 'From', 'To', 'Coupling variables'):
            assert '<th scope="col">{}</th>'.format(header) in added_step_html, header
        assert '<th scope="col">Coupling</th>' not in added_step_html
        assert '<th scope="col">Speed</th>' not in added_step_html
        assert 'Kuramoto &times;' not in added_step_html and 'mm/ms' not in added_step_html
        # and the editor checks it in the Intra picker, counts it, and offers to remove it
        assert re.search(r'<option value="{}"[^>]*>✓ Subnetwork A</option>'.format(key), added_html)
        assert '1 of 1 added' in added_html
        assert "hybridSetProjectionIncluded('{}', false)".format(key) in added_html
        assert 'Remove from the simulation' in added_html
        assert 'data-hybrid-changed="true"' in added_html

        # the editor, in the third column: a single Intra tab, its picker choosing the Subnetwork
        assert 'id="hybrid-projections-config"' in configure_html
        assert 'role="tablist"' in configure_html
        assert '<span class="hybrid-projection-group is-intra">Intra</span>' in configure_html
        tab = re.search(r'<div role="tab"[^>]*hybrid-intra-tab[^>]*>', configure_html).group(0)
        assert 'is-intra' in tab and 'is-selected' in tab and 'aria-selected="true"' in tab
        assert "hybridSelectProjection('{}')".format(key) in tab
        assert 'id="hybrid-intra-subnetwork"' in configure_html
        assert re.search(r'<option value="{}" selected>Subnetwork A</option>'.format(key), configure_html)
        assert '0 of 1 added' in configure_html
        # no tab of its own for any Projection
        assert '<button type="button" tabindex="1" role="tab"' not in configure_html
        # and the Projection's editor is framed under it, in the colour of its kind
        assert 'class="hybrid-projection-block is-intra" id="hybrid-projection-editor"' in configure_html
        assert 'Source coupling variable' in configure_html
        assert '<select name="source_cvar"' in configure_html
        # opening it changes nothing, so the list on the step is not reloaded
        assert 'data-hybrid-changed="false"' in configure_html
        assert 'data-hybrid-changed="false"' in editor_html

        # the editor holds the Projection's own fields, its coupling function's parameters, and the button
        # adding it to the simulation - not added, to begin with
        assert 'data-projection-key="{}"'.format(key) in editor_html
        assert "hybridSetProjectionIncluded('{}', true)".format(key) in editor_html
        assert 'Add to the simulation' in editor_html
        assert 'not in the simulation' in editor_html
        for field in ('source_cvar', 'target_cvar', 'scale', 'cv', 'coupling', 'a', 'b'):
            assert 'name="{}"'.format(field) in editor_html, field
        assert 'Linear parameters' in editor_html
        assert '<div class="hybrid-editor-level-1">' in editor_html
        assert 'data-hybrid-applied="false"' in editor_html

        # an applied edit says so, and asks for the list to be reloaded; a refused one does neither, and
        # shows why
        assert 'data-hybrid-applied="true"' in applied_html
        assert 'data-hybrid-changed="true"' in applied_html
        assert 'data-hybrid-applied="false"' in refused_html
        assert 'data-hybrid-changed="false"' in refused_html
        assert 'The conduction speed has to be positive.' in refused_html
        # a new coupling function brings its own parameters
        assert 'Kuramoto parameters' in kuramoto_html
        assert 'name="b"' not in kuramoto_html

        # removed again, it is no longer checked in the editor, and leaves the list on the step
        assert '0 of 1 added' in excluded_html
        assert '✓ Subnetwork A' not in excluded_html
        assert 'Add to the simulation' in excluded_html
        assert 'data-hybrid-changed="true"' in excluded_html
        assert '<table' not in excluded_step_html
        assert 'No Projection is added yet' in excluded_step_html

    def _reach_the_projections(self):
        """Walk the wizard up to the Projections step, which is where the global configuration starts."""
        cherrypy.request.method = "GET"
        self.hybrid_controller.set_subnetworks()
        cherrypy.request.method = "POST"
        self.hybrid_controller.set_subnetworks()
        self.hybrid_controller.set_subnetwork_dynamics()
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
            monitors_html = self.hybrid_controller.open_monitors()

            # the simulation length is asked on the closing step; the BOLD period below is checked
            # against it, so a long one is set, the way an earlier pass through that step leaves it
            self.hybrid_controller.context.hybrid_simulator.simulation_length = 6000.0

            # a plain Monitor followed by one carrying datatype fields
            tavg_html = self.hybrid_controller.set_monitors(monitors=['Temporal average', 'EEG'])
            eeg_html = self.hybrid_controller.set_monitor_params(
                'TemporalAverageViewModel', period='1.0')

            # and a BOLD Monitor, which takes an Equation step of its own after its parameters
            bold_html = self.hybrid_controller.set_monitors(monitors=['BOLD'])
            equation_html = self.hybrid_controller.set_monitor_params(
                'BoldViewModel', period='2000.0', hrf_kernel='Hrf Kernel: Mixture Of Gammas')

        # the Projections step now carries a way onward, which opens the Monitors step
        assert "hybridSubmitTo(this.parentElement, '/burst/hybrid/open_monitors')" in projections_html

        # the global configuration step asks what to record - how long for is asked last - and opens
        # without refusing anything
        assert 'action="/burst/hybrid/set_monitors"' in monitors_html
        assert 'simulation_length' not in monitors_html
        assert 'BOLD' in monitors_html
        assert 'Field required' not in monitors_html
        # it opens the Monitors section, which each Monitor's step continues
        assert 'class="hybrid-monitor-step"' in monitors_html
        assert 'class="hybrid-monitor-step"' in tavg_html
        assert 'class="hybrid-monitor-step"' in equation_html

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
            self.hybrid_controller.context.hybrid_simulator.simulation_length = 100.0
            self.hybrid_controller.set_monitors(monitors=['Raw recording', 'Temporal average'])
            summary_html = self.hybrid_controller.set_monitor_params(
                'TemporalAverageViewModel', period='1.0')

        assert 'action="/burst/hybrid/set_simulation_summary"' in summary_html
        # no summary of the Monitors any more
        assert 'hybrid-simulation-summary' not in summary_html
        assert 'Samples every' not in summary_html
        # one Subnetwork, so the variable counts trivially agree: the output is region ordered, and there
        # is nothing to warn about
        assert 'different numbers of variables' not in summary_html
        # it comes after the Monitors section, outside it
        assert 'class="hybrid-monitor-step"' not in summary_html
        # and the step closes the wizard: the simulation length and a name for it, asked last, in that
        # order, and the Launch button, labelled
        length_at = summary_html.index('name="simulation_length"')
        name_at = summary_html.index('name="input_simulation_name_id"')
        assert length_at < name_at
        assert re.search(r'<input[^>]*name="simulation_length"[^>]*value="100.0"', summary_html)
        launch = re.search(r'<button name="launch_simulation"[^>]*>\s*Launch', summary_html)
        assert launch and 'action-launch' not in launch.group(0)
        assert 'hybridLaunchSimulation(this.parentElement)' in summary_html
        assert 'disabled="disabled"' not in summary_html
