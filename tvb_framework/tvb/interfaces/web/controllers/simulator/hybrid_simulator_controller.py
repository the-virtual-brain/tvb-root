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
import cherrypy
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.forms.equation_forms import get_form_for_equation
from tvb.adapters.forms.hybrid_simulator_fragments import HybridConnectivityFragment, \
    HybridSubnetworkDynamicsFragment, HybridSubnetworksFragment
from tvb.adapters.forms.integrator_forms import get_form_for_integrator
from tvb.adapters.forms.model_forms import get_form_for_model
from tvb.adapters.forms.noise_forms import get_form_for_noise
from tvb.adapters.forms.simulator_fragments import SimulatorIntegratorFragment, SimulatorModelFragment
from tvb.core.entities.file.simulator.view_model import HybridSimulatorAdapterModel, \
    IntegratorStochasticViewModel, MultiplicativeNoiseViewModel
from tvb.core.services.hybrid_simulator_service import HybridSimulatorService, HybridSubnetworkException
from tvb.core.services.simulator_service import SimulatorService
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.burst.base_controller import BurstBaseController
from tvb.interfaces.web.controllers.decorators import expose_fragment, expose_page, expose_json, settings, \
    context_selected
from tvb.interfaces.web.controllers.simulator.simulator_fragment_rendering_rules import POST_REQUEST
from tvb.interfaces.web.entities.context_hybrid_simulator import HybridSimulatorContext


class HybridSimulatorURLs(object):
    SET_CONNECTIVITY_URL = '/burst/hybrid/set_connectivity'
    # the wizard step listing the saved Subnetworks, shown in the cockpit configuration column
    SET_SUBNETWORKS_URL = '/burst/hybrid/set_subnetworks'
    # the board on which the regions are grouped, shown in the contextual configuration column
    CONFIGURE_SUBNETWORKS_URL = '/burst/hybrid/configure_subnetworks'
    SAVE_SUBNETWORKS_URL = '/burst/hybrid/save_subnetworks'
    ADD_SUBNETWORK_URL = '/burst/hybrid/add_subnetwork'
    REMOVE_SUBNETWORK_URL = '/burst/hybrid/remove_subnetwork'
    RENAME_SUBNETWORK_URL = '/burst/hybrid/rename_subnetwork'
    MOVE_REGIONS_URL = '/burst/hybrid/move_regions'
    # the wizard step under which every Subnetwork's dynamics are configured, holding the shared dt
    SET_SUBNETWORK_DYNAMICS_URL = '/burst/hybrid/set_subnetwork_dynamics'
    SELECT_SUBNETWORK_URL = '/burst/hybrid/select_subnetwork'
    SAVE_SUBNETWORK_DYNAMICS_URL = '/burst/hybrid/save_subnetwork_dynamics'
    # the steps configuring the selected Subnetwork, mirroring the classic Cockpit chain. They are
    # wizard steps of the configuration column, accumulating under the Subnetwork dynamics step.
    SET_SUBNETWORK_MODEL_URL = '/burst/hybrid/set_subnetwork_model'
    SET_SUBNETWORK_MODEL_PARAMS_URL = '/burst/hybrid/set_subnetwork_model_params'
    SET_SUBNETWORK_INTEGRATOR_URL = '/burst/hybrid/set_subnetwork_integrator'
    SET_SUBNETWORK_INTEGRATOR_PARAMS_URL = '/burst/hybrid/set_subnetwork_integrator_params'
    SET_SUBNETWORK_NOISE_PARAMS_URL = '/burst/hybrid/set_subnetwork_noise_params'
    SET_SUBNETWORK_NOISE_EQUATION_PARAMS_URL = '/burst/hybrid/set_subnetwork_noise_equation_params'


class HybridSimulatorFragmentRenderingRules(object):
    FIRST_FORM_URL = HybridSimulatorURLs.SET_CONNECTIVITY_URL

    def __init__(self, form, form_action_url, previous_form_action_url=None, is_first_fragment=False,
                 is_subnetworks_summary_fragment=False, fragment_title=None, next_button_label='Next',
                 previous_button_label='Previous', next_button_enabled=True, region_labels=None,
                 subnetworks=None, context_form_url=None, context_title=None, is_modified=False,
                 load_error=None, is_dynamics_summary_fragment=False, is_dynamics_save_fragment=False,
                 selected_subnetwork=None, dynamics_by_id=None):
        self.form = form
        self.form_action_url = form_action_url
        self.previous_form_action_url = previous_form_action_url
        self.is_first_fragment = is_first_fragment
        # the wizard step listing what the grouping board produced
        self.is_subnetworks_summary_fragment = is_subnetworks_summary_fragment
        self.fragment_title = fragment_title
        self.next_button_label = next_button_label
        self.previous_button_label = previous_button_label
        self.next_button_enabled = next_button_enabled
        self.region_labels = region_labels
        self.subnetworks = subnetworks
        # The configuration this step exposes in the third column, when it has one. That column follows
        # the step being configured, and is emptied again for the steps declaring nothing here.
        self.context_form_url = context_form_url
        self.context_title = context_title
        # True while the board holds a grouping that was not saved onto the configuration yet
        self.is_modified = is_modified
        # set instead of a board when the grouping can not be configured, e.g. without a Connectivity
        self.load_error = load_error
        # the wizard step summarising the dynamics configured for every Subnetwork
        self.is_dynamics_summary_fragment = is_dynamics_summary_fragment
        # the closing step of the per Subnetwork sub wizard, offering Save Configuration instead of Next
        self.is_dynamics_save_fragment = is_dynamics_save_fragment
        self.selected_subnetwork = selected_subnetwork
        # the dynamics currently being edited, keyed by Subnetwork identifier
        self.dynamics_by_id = dynamics_by_id or {}

    @property
    def include_previous_button(self):
        return not self.is_first_fragment

    @property
    def subnetwork_rows(self):
        """
        One row per Subnetwork for the summary step: its name and the regions assigned to it, keeping
        the original Connectivity indices next to the labels.
        """
        labels = self.region_labels or []
        rows = []
        for subnetwork in self.subnetworks or []:
            node_indices = list(subnetwork.node_indices)
            rows.append({
                'name': subnetwork.name,
                'count': len(node_indices),
                'regions': [{'index': node_index,
                             'label': labels[node_index] if node_index < len(labels) else str(node_index)}
                            for node_index in node_indices]
            })
        return rows

    @property
    def subnetwork_choices(self):
        """
        One entry per Subnetwork for the selector of the contextual column: what it holds and what is
        configured for it, so the user can tell the Subnetworks apart without opening each one.
        """
        choices = []
        for subnetwork in self.subnetworks or []:
            dynamics = self.dynamics_by_id.get(subnetwork.id) or subnetwork.dynamics
            choices.append({
                'id': subnetwork.id,
                'name': subnetwork.name,
                'count': len(subnetwork.node_indices),
                'model': type(dynamics.model).__name__ if dynamics and dynamics.model else '',
                'integrator': type(dynamics.integrator).__name__ if dynamics and dynamics.integrator else '',
                'is_selected': subnetwork.id == self.selected_subnetwork
            })
        return choices

    @property
    def dynamics_rows(self):
        """
        One row per Subnetwork for the dynamics wizard step, describing the saved configuration.
        """
        rows = []
        for subnetwork in self.subnetworks or []:
            dynamics = subnetwork.dynamics
            integrator = dynamics.integrator if dynamics else None
            noise = getattr(integrator, 'noise', None)
            rows.append({
                'name': subnetwork.name,
                'count': len(subnetwork.node_indices),
                'model': type(dynamics.model).__name__ if dynamics and dynamics.model else '',
                'integrator': type(integrator).__name__.replace('ViewModel', '') if integrator else '',
                'noise': type(noise).__name__.replace('ViewModel', '') if noise is not None else ''
            })
        return rows

    @property
    def subnetworks_json(self):
        """
        The Subnetwork configuration, as consumed by the hybrid_subnetworks.js client side component.
        """
        payload = json.dumps({
            'region_labels': self.region_labels or [],
            'subnetworks': HybridSimulatorService.to_json_ready(self.subnetworks or []),
            'is_modified': self.is_modified
        })
        # the result is inlined inside a <script> tag, so no region label may close it
        return payload.replace('<', '\\u003c')

    def to_dict(self):
        return {"renderer": self, "isCallout": False}


@traced
class HybridSimulatorController(BurstBaseController):

    def __init__(self):
        BurstBaseController.__init__(self)
        self.context = HybridSimulatorContext()
        self.simulator_service = SimulatorService()
        self.hybrid_simulator_service = HybridSimulatorService()

    @staticmethod
    def get_available_hybrid_bursts(project_id):
        return []

    def _prepare_connectivity_form(self):
        self.context.set_hybrid_simulator()
        form = self.algorithm_service.prepare_adapter_form(form_instance=HybridConnectivityFragment(),
                                                           project_id=self.context.project.id)
        self.simulator_service.validate_first_fragment(form, self.context.project.id, ConnectivityIndex)
        form.fill_from_trait(self.context.hybrid_simulator)
        return form

    @staticmethod
    def _connectivity_rendering_rules(form):
        # No context_form_url: the Connectivity step configures nothing in the third column, which is
        # what empties that column again when the user steps back to it.
        return HybridSimulatorFragmentRenderingRules(
            form, HybridSimulatorURLs.SET_CONNECTIVITY_URL, is_first_fragment=True, fragment_title="Connectivity")

    @expose_page
    @settings
    @context_selected
    def index(self):
        template_specification = dict(mainContent="burst/main_hybrid_simulator", title="Hybrid Simulator",
                                      includedResources='project/included_resources')

        if not self.context.last_loaded_fragment_url:
            self.context.add_last_loaded_form_url_to_session(HybridSimulatorURLs.SET_CONNECTIVITY_URL)

        form = self._prepare_connectivity_form()
        rendering_rules = self._connectivity_rendering_rules(form)

        template_specification['burst_list'] = self.get_available_hybrid_bursts(self.context.project.id)
        template_specification.update(**rendering_rules.to_dict())

        cherrypy.response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        cherrypy.response.headers['Pragma'] = 'no-cache'
        cherrypy.response.headers['Expires'] = '0'

        return self.fill_default_attributes(template_specification, subsection='hybrid')

    @expose_fragment('burst/hybrid_burst_history')
    def load_hybrid_history(self):
        return {'burst_list': self.get_available_hybrid_bursts(self.context.project.id)}

    @expose_fragment('hybrid_simulator_fragment')
    def reset_hybrid_simulator_configuration(self):
        self.context.reset_hybrid_simulator()
        self.context.add_last_loaded_form_url_to_session(HybridSimulatorURLs.SET_CONNECTIVITY_URL)
        form = self._prepare_connectivity_form()
        return self._connectivity_rendering_rules(form).to_dict()

    @expose_fragment('hybrid_simulator_fragment')
    def set_connectivity(self, **data):
        if cherrypy.request.method == POST_REQUEST:
            form = self.algorithm_service.prepare_adapter_form(form_instance=HybridConnectivityFragment(),
                                                               project_id=self.context.project.id)
            form.fill_from_post(data)
            if not form.validate():
                self.context.add_last_loaded_form_url_to_session(HybridSimulatorURLs.SET_CONNECTIVITY_URL)
                return self._connectivity_rendering_rules(form).to_dict()

            # keep the already configured Subnetworks, they are only dropped when the Connectivity changes
            hybrid_simulator = self.context.hybrid_simulator or HybridSimulatorAdapterModel()
            previous_connectivity = self._configured_connectivity(hybrid_simulator)
            form.fill_trait(hybrid_simulator)
            if hybrid_simulator.connectivity != previous_connectivity:
                hybrid_simulator.subnetworks = []
                # the board was grouping the regions of the Connectivity that is no longer selected
                self.context.clear_subnetworks_draft()
            self.context.set_hybrid_simulator(hybrid_simulator)

            # the undecorated helper, since an exposed method answers with a rendered fragment
            return self._subnetworks_step()

        form = self._prepare_connectivity_form()
        return self._connectivity_rendering_rules(form).to_dict()

    @expose_fragment('hybrid_simulator_fragment')
    def set_subnetworks(self, **data):
        """
        The wizard step listing the saved Subnetworks, rendered in the cockpit configuration column.
        It declares the grouping board as the configuration belonging to the third column.

        Pressing Next moves on to configuring each Subnetwork's dynamics. A grouping that was edited but
        not saved is refused there: the following step configures the saved Subnetworks, so letting it
        open over an unsaved grouping would configure Subnetworks the configuration does not hold.
        """
        if cherrypy.request.method == POST_REQUEST:
            try:
                _, region_labels, subnetworks, draft = self._load_subnetworks_configuration()
            except HybridSubnetworkException as excep:
                return self._back_to_connectivity(str(excep))

            if self._is_modified(subnetworks, draft):
                common.set_error_message("Save the Subnetwork configuration before configuring its dynamics.")
                return self._subnetworks_step_rules(region_labels, subnetworks, draft).to_dict()

            return self._subnetwork_dynamics_step()

        return self._subnetworks_step()

    def _subnetworks_step(self):
        self.context.add_last_loaded_form_url_to_session(HybridSimulatorURLs.SET_SUBNETWORKS_URL)
        try:
            _, region_labels, subnetworks, draft = self._load_subnetworks_configuration()
        except HybridSubnetworkException as excep:
            return self._back_to_connectivity(str(excep))

        return self._subnetworks_step_rules(region_labels, subnetworks, draft).to_dict()

    @expose_fragment('burst/hybrid_subnetworks')
    def configure_subnetworks(self, **data):
        """
        The board on which the Connectivity regions are grouped into Subnetworks, shown in the third
        column while the Subnetworks step is being configured. It edits a draft: nothing reaches the
        Hybrid Simulator configuration until the grouping is saved.
        """
        try:
            _, region_labels, subnetworks, draft = self._load_subnetworks_configuration()
        except HybridSubnetworkException as excep:
            return HybridSimulatorFragmentRenderingRules(
                None, HybridSimulatorURLs.CONFIGURE_SUBNETWORKS_URL, load_error=str(excep)).to_dict()

        return HybridSimulatorFragmentRenderingRules(
            None, HybridSimulatorURLs.CONFIGURE_SUBNETWORKS_URL, fragment_title="Subnetworks",
            region_labels=region_labels, subnetworks=draft,
            is_modified=self._is_modified(subnetworks, draft)).to_dict()

    @expose_fragment('hybrid_simulator_fragment')
    def save_subnetworks(self, **data):
        """
        Store the grouping currently on the board onto the Hybrid Simulator configuration and answer with
        the refreshed Subnetworks wizard step. This is the only place writing that grouping, which is what
        keeps the summary showing the saved configuration rather than the one being edited.
        """
        try:
            hybrid_simulator, region_labels, _, draft = self._load_subnetworks_configuration()
        except HybridSubnetworkException as excep:
            return self._back_to_connectivity(str(excep))

        # Empty Subnetworks are allowed on the board, as somewhere to drag regions into, but they can not
        # take part in a simulation, so saving is where they are dropped.
        subnetworks = self.hybrid_simulator_service.discard_empty_subnetworks(draft)
        hybrid_simulator.subnetworks = subnetworks
        self.context.set_hybrid_simulator(hybrid_simulator)
        # the board must show what was saved, the discarded Subnetworks included
        draft = self.hybrid_simulator_service.copy_subnetworks(subnetworks)
        self.context.set_subnetworks_draft(draft)

        self.context.add_last_loaded_form_url_to_session(HybridSimulatorURLs.SET_SUBNETWORKS_URL)
        return self._subnetworks_step_rules(region_labels, subnetworks, draft).to_dict()

    # ---------------------------------------------------------------- Subnetwork editing

    @expose_json
    def add_subnetwork(self, **data):
        return self._change_subnetworks(self.hybrid_simulator_service.add_subnetwork,
                                        "New Subnetwork created.")

    @expose_json
    def remove_subnetwork(self, subnetwork_index=None, **data):
        index = self._parse_index(subnetwork_index)
        return self._change_subnetworks(
            lambda subnetworks: self.hybrid_simulator_service.remove_subnetwork(subnetworks, index),
            "Subnetwork removed.")

    @expose_json
    def rename_subnetwork(self, subnetwork_index=None, name=None, **data):
        index = self._parse_index(subnetwork_index)
        return self._change_subnetworks(
            lambda subnetworks: self.hybrid_simulator_service.rename_subnetwork(subnetworks, index, name),
            "Subnetwork renamed.")

    @expose_json
    def move_regions(self, subnetwork_index=None, node_indices=None, **data):
        index = self._parse_index(subnetwork_index)
        try:
            nodes = json.loads(node_indices) if node_indices else []
        except ValueError:
            nodes = None
        if not isinstance(nodes, list):
            nodes = None

        return self._change_subnetworks(
            lambda subnetworks: self.hybrid_simulator_service.move_regions(subnetworks, nodes, index),
            "Connectivity regions moved.")

    # ---------------------------------------------------------------- Subnetwork dynamics

    @expose_fragment('hybrid_simulator_fragment')
    def set_subnetwork_dynamics(self, **data):
        """
        The wizard step under which each Subnetwork's Model and Integrator are configured. Its own field
        is the shared integration step size; the per Subnetwork configuration lives in the third column.
        """
        if cherrypy.request.method == POST_REQUEST:
            # applying the shared dt is what submitting this step does, and then the configuration of the
            # selected Subnetwork begins under it
            self._subnetwork_dynamics_step(data)
            dynamics, _ = self._selected_dynamics()
            return self._model_step_rules(dynamics).to_dict()

        return self._subnetwork_dynamics_step()

    def _subnetwork_dynamics_step(self, data=None):
        self.context.add_last_loaded_form_url_to_session(HybridSimulatorURLs.SET_SUBNETWORK_DYNAMICS_URL)
        try:
            hybrid_simulator, _, subnetworks, _ = self._load_subnetworks_configuration()
        except HybridSubnetworkException as excep:
            return self._back_to_connectivity(str(excep))

        form = HybridSubnetworkDynamicsFragment()
        if data is not None:
            form.fill_from_post(data)
            if form.validate():
                form.fill_trait(hybrid_simulator)
                self.context.set_hybrid_simulator(hybrid_simulator)
            else:
                form.fill_from_trait(hybrid_simulator)
        else:
            form.fill_from_trait(hybrid_simulator)

        # the shared dt reaches every Subnetwork Integrator, the ones configured before it last changed
        # included, which is what keeps tvb.simulator.hybrid.Simulator from refusing the NetworkSet
        draft = self._prepare_dynamics_draft(subnetworks, hybrid_simulator.dt)

        return self._dynamics_step_rules(form, subnetworks, draft).to_dict()

    @expose_fragment('hybrid_simulator_fragment')
    def select_subnetwork(self, subnetwork_id=None, **data):
        """
        Configure another Subnetwork. Answers with the first step of its configuration, which is what the
        client puts in place of the steps that were configuring the Subnetwork being left. What was
        edited there is kept: the draft holds every Subnetwork at once.
        """
        try:
            _, _, subnetworks, _ = self._load_subnetworks_configuration()
            self.hybrid_simulator_service.find_subnetwork(subnetworks, subnetwork_id)
        except HybridSubnetworkException as excep:
            return self._back_to_connectivity(str(excep))

        self.context.set_selected_subnetwork(subnetwork_id)
        dynamics, _ = self._selected_dynamics()
        return self._model_step_rules(dynamics).to_dict()

    @expose_fragment('hybrid_simulator_fragment')
    def set_subnetwork_model(self, **data):
        """
        Step 1: the Model class of the selected Subnetwork. Answers with its parameters, the way the
        classic Cockpit's set_model answers with the Model parameters form.
        """
        dynamics, _ = self._selected_dynamics()

        if cherrypy.request.method == POST_REQUEST:
            form = SimulatorModelFragment()
            form.fill_from_post(data)
            if not form.validate():
                # an unknown class would otherwise reach fill_trait as a plain string and raise there
                return self._step_rules(form, HybridSimulatorURLs.SET_SUBNETWORK_MODEL_URL,
                                        HybridSimulatorURLs.SET_SUBNETWORK_DYNAMICS_URL).to_dict()
            # fill_trait only replaces the Model when the selected class actually changed, so switching
            # class resets the parameters while re-submitting the same one keeps the edited values
            form.fill_trait(dynamics)

        return self._model_params_step_rules(dynamics).to_dict()

    @expose_fragment('hybrid_simulator_fragment')
    def set_subnetwork_model_params(self, **data):
        """
        Step 2: the Model parameters. Answers with the Integrator class selection.
        """
        dynamics, _ = self._selected_dynamics()

        if cherrypy.request.method == POST_REQUEST:
            form = get_form_for_model(type(dynamics.model))()
            form.fill_from_post(data)
            if not form.validate():
                return self._model_params_step_rules(dynamics, form).to_dict()
            form.fill_trait(dynamics.model)

        return self._integrator_step_rules(dynamics).to_dict()

    @expose_fragment('hybrid_simulator_fragment')
    def set_subnetwork_integrator(self, **data):
        """
        Step 3: the Integrator class of the selected Subnetwork. Answers with its parameters.
        """
        dynamics, hybrid_simulator = self._selected_dynamics()

        if cherrypy.request.method == POST_REQUEST:
            form = SimulatorIntegratorFragment()
            form.fill_from_post(data)
            if not form.validate():
                form.integrator.display_subform = False
                return self._step_rules(form, HybridSimulatorURLs.SET_SUBNETWORK_INTEGRATOR_URL,
                                        HybridSimulatorURLs.SET_SUBNETWORK_MODEL_PARAMS_URL).to_dict()
            # SimulatorIntegratorFragment.fill_trait replaces the Integrator unconditionally, which would
            # discard the edited parameters every time this step is submitted again. Only the class change
            # is applied here, so re-submitting the same class keeps them.
            selected_class = form.integrator.value
            if selected_class is not None and type(dynamics.integrator) != selected_class.value:
                dynamics.integrator = selected_class.instance
            dynamics.integrator.dt = hybrid_simulator.dt

        return self._integrator_params_step_rules(dynamics).to_dict()

    @expose_fragment('hybrid_simulator_fragment')
    def set_subnetwork_integrator_params(self, **data):
        """
        Step 4: the Integrator parameters. Answers with the Noise parameters for a stochastic Integrator,
        and with the closing step otherwise.
        """
        dynamics, hybrid_simulator = self._selected_dynamics()

        if cherrypy.request.method == POST_REQUEST:
            # dt is rendered disabled, and a disabled input is not submitted, so the shared value is put
            # back into the posted data the way the classic set_integrator_params does for a branch
            data['dt'] = str(hybrid_simulator.dt)
            form = get_form_for_integrator(type(dynamics.integrator))(is_dt_disabled=True)
            form.fill_from_post(data)
            if not form.validate():
                return self._integrator_params_step_rules(dynamics, form).to_dict()
            form.fill_trait(dynamics.integrator)
            dynamics.integrator.dt = hybrid_simulator.dt

        if not isinstance(dynamics.integrator, IntegratorStochasticViewModel):
            return self._save_step_rules(HybridSimulatorURLs.SET_SUBNETWORK_INTEGRATOR_PARAMS_URL).to_dict()

        return self._noise_params_step_rules(dynamics).to_dict()

    @expose_fragment('hybrid_simulator_fragment')
    def set_subnetwork_noise_params(self, **data):
        """
        Step 5: the Noise parameters. Answers with the Equation parameters for a Multiplicative Noise,
        and with the closing step otherwise.
        """
        dynamics, _ = self._selected_dynamics()
        noise = dynamics.integrator.noise

        if cherrypy.request.method == POST_REQUEST:
            form = get_form_for_noise(type(noise))()
            form.fill_from_post(data)
            if not form.validate():
                return self._noise_params_step_rules(dynamics, form).to_dict()
            form.fill_trait(noise)

        if not isinstance(noise, MultiplicativeNoiseViewModel):
            return self._save_step_rules(HybridSimulatorURLs.SET_SUBNETWORK_NOISE_PARAMS_URL).to_dict()

        return self._noise_equation_step_rules(dynamics).to_dict()

    @expose_fragment('hybrid_simulator_fragment')
    def set_subnetwork_noise_equation_params(self, **data):
        """
        Step 6: the parameters of the Equation of a Multiplicative Noise. Answers with the closing step.
        """
        dynamics, _ = self._selected_dynamics()
        equation = dynamics.integrator.noise.b

        if cherrypy.request.method == POST_REQUEST:
            form = get_form_for_equation(type(equation))()
            form.fill_from_post(data)
            if not form.validate():
                return self._noise_equation_step_rules(dynamics, form).to_dict()
            form.fill_trait(equation)

        return self._save_step_rules(HybridSimulatorURLs.SET_SUBNETWORK_NOISE_EQUATION_PARAMS_URL).to_dict()

    @expose_fragment('hybrid_simulator_fragment')
    def save_subnetwork_dynamics(self, **data):
        """
        Store the dynamics currently being edited onto the Subnetworks and answer with the refreshed
        wizard step. This is the only place writing them, which is what keeps the step summarising the
        saved configuration rather than the one being edited.
        """
        try:
            hybrid_simulator, _, subnetworks, _ = self._load_subnetworks_configuration()
        except HybridSubnetworkException as excep:
            return self._back_to_connectivity(str(excep))

        draft = self._prepare_dynamics_draft(subnetworks, hybrid_simulator.dt)

        # a Model parameter has to broadcast onto the nodes of the Subnetwork it belongs to
        try:
            for subnetwork in subnetworks:
                candidate = self.hybrid_simulator_service.copy_subnetworks([subnetwork])[0]
                self.hybrid_simulator_service.store_dynamics([candidate], draft)
                self.hybrid_simulator_service.validate_model_parameters(candidate)
        except HybridSubnetworkException as excep:
            common.set_error_message(str(excep))
            form = HybridSubnetworkDynamicsFragment()
            form.fill_from_trait(hybrid_simulator)
            return self._dynamics_step_rules(form, subnetworks, draft).to_dict()

        self.hybrid_simulator_service.store_dynamics(subnetworks, draft)
        hybrid_simulator.subnetworks = subnetworks
        self.context.set_hybrid_simulator(hybrid_simulator)

        form = HybridSubnetworkDynamicsFragment()
        form.fill_from_trait(hybrid_simulator)
        self.context.add_last_loaded_form_url_to_session(HybridSimulatorURLs.SET_SUBNETWORK_DYNAMICS_URL)
        return self._dynamics_step_rules(form, subnetworks, draft).to_dict()

    # ---------------------------------------------------------------- Dynamics helpers

    def _prepare_dynamics_draft(self, subnetworks, dt):
        """
        The dynamics being edited, seeded from the saved ones, with the entries of Subnetworks that no
        longer exist dropped and the shared dt applied to every Integrator.
        """
        draft = self.hybrid_simulator_service.prepare_dynamics_draft(subnetworks, self.context.dynamics_draft)
        self.hybrid_simulator_service.apply_shared_dt(draft, dt)
        # The shared dt is a simulation wide setting applied on its own step, not part of the per
        # Subnetwork draft, so the saved Integrators get it too. Leaving them behind would report a
        # pending change that the user cannot save away.
        self.hybrid_simulator_service.apply_shared_dt(
            {subnetwork.id: subnetwork.dynamics for subnetwork in subnetworks or []}, dt)
        self.context.set_dynamics_draft(draft)
        return draft

    def _selected_subnetwork(self, subnetworks):
        """
        :return: the Subnetwork being configured, defaulting to the first one and falling back to it when
                 the remembered selection is no longer part of the configuration
        """
        try:
            return self.hybrid_simulator_service.find_subnetwork(subnetworks, self.context.selected_subnetwork)
        except HybridSubnetworkException:
            self.context.set_selected_subnetwork(subnetworks[0].id)
            return subnetworks[0]

    def _selected_dynamics(self):
        """
        :return: the draft dynamics of the Subnetwork being configured, and the Hybrid Simulator
                 configuration holding the shared dt
        """
        hybrid_simulator, _, subnetworks, _ = self._load_subnetworks_configuration()
        draft = self._prepare_dynamics_draft(subnetworks, hybrid_simulator.dt)
        selected = self._selected_subnetwork(subnetworks)
        return draft[selected.id], hybrid_simulator

    def _dynamics_step_rules(self, form, subnetworks, draft):
        return HybridSimulatorFragmentRenderingRules(
            form, HybridSimulatorURLs.SET_SUBNETWORK_DYNAMICS_URL, HybridSimulatorURLs.SET_SUBNETWORKS_URL,
            is_dynamics_summary_fragment=True, fragment_title="Subnetwork dynamics", subnetworks=subnetworks,
            # No context_form_url: the Model and Integrator of each Subnetwork are configured in this
            # column, under this step, so the third column is handed back to the Results view here.
            dynamics_by_id=draft, selected_subnetwork=self._selected_subnetwork(subnetworks).id,
            is_modified=not self.hybrid_simulator_service.same_dynamics(subnetworks, draft))

    @staticmethod
    def _step_rules(form, form_action_url, previous_form_action_url):
        return HybridSimulatorFragmentRenderingRules(form, form_action_url, previous_form_action_url)

    def _model_step_rules(self, dynamics):
        form = self.algorithm_service.prepare_adapter_form(form_instance=SimulatorModelFragment())
        form.fill_from_trait(dynamics)
        return self._step_rules(form, HybridSimulatorURLs.SET_SUBNETWORK_MODEL_URL,
                                HybridSimulatorURLs.SET_SUBNETWORK_DYNAMICS_URL)

    def _model_params_step_rules(self, dynamics, form=None):
        if form is None:
            form = self.algorithm_service.prepare_adapter_form(
                form_instance=get_form_for_model(type(dynamics.model))())
            form.fill_from_trait(dynamics.model)
        return self._step_rules(form, HybridSimulatorURLs.SET_SUBNETWORK_MODEL_PARAMS_URL,
                                HybridSimulatorURLs.SET_SUBNETWORK_MODEL_URL)

    def _integrator_step_rules(self, dynamics):
        form = self.algorithm_service.prepare_adapter_form(form_instance=SimulatorIntegratorFragment())
        # the Integrator parameters are the next step, they are not nested inside this one
        form.integrator.display_subform = False
        form.fill_from_trait(dynamics)
        return self._step_rules(form, HybridSimulatorURLs.SET_SUBNETWORK_INTEGRATOR_URL,
                                HybridSimulatorURLs.SET_SUBNETWORK_MODEL_PARAMS_URL)

    def _integrator_params_step_rules(self, dynamics, form=None):
        if form is None:
            form = self.algorithm_service.prepare_adapter_form(
                form_instance=get_form_for_integrator(type(dynamics.integrator))(is_dt_disabled=True))
            if hasattr(form, 'noise'):
                # the Noise parameters are the next step
                form.noise.display_subform = False
            form.fill_from_trait(dynamics.integrator)
        return self._step_rules(form, HybridSimulatorURLs.SET_SUBNETWORK_INTEGRATOR_PARAMS_URL,
                                HybridSimulatorURLs.SET_SUBNETWORK_INTEGRATOR_URL)

    def _noise_params_step_rules(self, dynamics, form=None):
        if form is None:
            form = self.algorithm_service.prepare_adapter_form(
                form_instance=get_form_for_noise(type(dynamics.integrator.noise))())
            if hasattr(form, 'equation'):
                # the Equation parameters are the next step
                form.equation.display_subform = False
            form.fill_from_trait(dynamics.integrator.noise)
        return self._step_rules(form, HybridSimulatorURLs.SET_SUBNETWORK_NOISE_PARAMS_URL,
                                HybridSimulatorURLs.SET_SUBNETWORK_INTEGRATOR_PARAMS_URL)

    def _noise_equation_step_rules(self, dynamics, form=None):
        if form is None:
            form = self.algorithm_service.prepare_adapter_form(
                form_instance=get_form_for_equation(type(dynamics.integrator.noise.b))())
            form.fill_from_trait(dynamics.integrator.noise.b)
        return self._step_rules(form, HybridSimulatorURLs.SET_SUBNETWORK_NOISE_EQUATION_PARAMS_URL,
                                HybridSimulatorURLs.SET_SUBNETWORK_NOISE_PARAMS_URL)

    @staticmethod
    def _save_step_rules(previous_form_action_url):
        """
        The closing step of the sub wizard: nothing left to fill in for this Subnetwork, only the action
        writing what was configured onto the Hybrid Simulator configuration.
        """
        return HybridSimulatorFragmentRenderingRules(
            None, HybridSimulatorURLs.SAVE_SUBNETWORK_DYNAMICS_URL, previous_form_action_url,
            is_dynamics_save_fragment=True, next_button_label='Save Configuration')

    # ---------------------------------------------------------------- Helpers

    def _subnetworks_step_rules(self, region_labels, subnetworks, draft):
        return HybridSimulatorFragmentRenderingRules(
            HybridSubnetworksFragment(), HybridSimulatorURLs.SET_SUBNETWORKS_URL,
            HybridSimulatorURLs.SET_CONNECTIVITY_URL, is_subnetworks_summary_fragment=True,
            fragment_title="Subnetworks", region_labels=region_labels, subnetworks=subnetworks,
            context_form_url=HybridSimulatorURLs.CONFIGURE_SUBNETWORKS_URL, context_title="Subnetworks",
            is_modified=self._is_modified(subnetworks, draft),
            # The dynamics step configures the saved Subnetworks, so it only opens over a saved grouping.
            next_button_enabled=not self._is_modified(subnetworks, draft))

    def _back_to_connectivity(self, message):
        common.set_error_message(message)
        self.context.add_last_loaded_form_url_to_session(HybridSimulatorURLs.SET_CONNECTIVITY_URL)
        return self._connectivity_rendering_rules(self._prepare_connectivity_form()).to_dict()

    def _is_modified(self, subnetworks, draft):
        """
        :return: True when saving the board would change the stored configuration. The comparison is done
                 against what saving would actually store, so an empty Subnetwork prepared as a drop
                 target is not on its own reported as an unsaved change.
        """
        if not draft:
            return False
        return not self.hybrid_simulator_service.same_grouping(
            subnetworks, self.hybrid_simulator_service.discard_empty_subnetworks(draft))

    def _load_subnetworks_configuration(self):
        """
        :return: the session stored Hybrid Simulator configuration, the Connectivity region labels, the
                 saved Subnetworks and the ones currently being edited on the board, after making sure
                 both still describe a valid partition of the Connectivity
        """
        hybrid_simulator = self.context.hybrid_simulator
        connectivity_gid = self._configured_connectivity(hybrid_simulator)
        if connectivity_gid is None:
            raise HybridSubnetworkException("Select a Connectivity before configuring the Subnetworks.")

        region_labels = self.hybrid_simulator_service.get_region_labels(connectivity_gid)
        subnetworks = self.hybrid_simulator_service.prepare_subnetworks(hybrid_simulator, len(region_labels))
        self.context.set_hybrid_simulator(hybrid_simulator)

        draft = self.context.subnetworks_draft
        if not self.hybrid_simulator_service.is_valid_partition(draft or [], len(region_labels)):
            # nothing was edited yet, or the draft groups a Connectivity that is no longer selected
            draft = self.hybrid_simulator_service.copy_subnetworks(subnetworks)
            self.context.set_subnetworks_draft(draft)

        return hybrid_simulator, region_labels, subnetworks, draft

    def _change_subnetworks(self, change, success_message):
        """
        Apply one Subnetwork change on the draft being edited and describe the resulting state. A change
        that would make the grouping invalid is refused, and the current draft is returned unchanged.
        """
        try:
            _, region_labels, subnetworks, draft = self._load_subnetworks_configuration()
        except HybridSubnetworkException as excep:
            return self._subnetworks_state([], [], [], str(excep), is_error=True)

        try:
            changed = change(draft)
        except HybridSubnetworkException as excep:
            return self._subnetworks_state(region_labels, subnetworks, draft, str(excep), is_error=True)

        self.context.set_subnetworks_draft(changed)
        return self._subnetworks_state(region_labels, subnetworks, changed, success_message)

    def _subnetworks_state(self, region_labels, subnetworks, draft, message, is_error=False):
        return {'status': 'error' if is_error else 'ok',
                'message': message,
                'region_labels': list(region_labels),
                'subnetworks': HybridSimulatorService.to_json_ready(draft),
                'is_modified': self._is_modified(subnetworks, draft)}

    @staticmethod
    def _configured_connectivity(hybrid_simulator):
        """
        :return: the GID of the selected Connectivity, or None when none was selected yet. The trait raises
                 instead of answering None while the required attribute was never assigned.
        """
        if hybrid_simulator is None:
            return None
        return getattr(hybrid_simulator, 'connectivity', None)

    @staticmethod
    def _parse_index(subnetwork_index):
        try:
            return int(subnetwork_index)
        except (TypeError, ValueError):
            return -1
