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
from tvb.adapters.forms.hybrid_monitor_forms import HybridSpatialAverageMonitorForm, \
    get_form_for_hybrid_monitor
from tvb.adapters.forms.hybrid_simulator_fragments import HybridConnectivityFragment, \
    HybridMonitorsFragment, HybridSubnetworkDynamicsFragment, HybridSubnetworksFragment
from tvb.adapters.forms.monitor_forms import get_monitor_to_ui_name_dict
from tvb.adapters.forms.integrator_forms import NoiseTypesEnum, get_form_for_integrator
from tvb.adapters.forms.model_forms import ModelsEnum, get_form_for_model
from tvb.adapters.forms.noise_forms import get_form_for_noise
from tvb.adapters.forms.simulator_fragments import SimulatorIntegratorFragment, SimulatorModelFragment
from tvb.core.entities.file.simulator.view_model import BoldViewModel, HybridSimulatorAdapterModel, \
    IntegratorStochasticViewModel, IntegratorViewModelsEnum, MultiplicativeNoiseViewModel, RawViewModel
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.services.hybrid_simulator_service import HybridSimulatorService, HybridSubnetworkException
from tvb.core.services.simulator_service import SimulatorService
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.burst.base_controller import BurstBaseController
from tvb.interfaces.web.controllers.decorators import expose_fragment, expose_page, expose_json, settings, \
    context_selected
from tvb.interfaces.web.controllers.simulator.simulator_fragment_rendering_rules import POST_REQUEST
from tvb.interfaces.web.entities.context_hybrid_simulator import HybridSimulatorContext


# The Phase plane page, where the Dynamics placed on regions by Set up region Model are defined.
PHASE_PLANE_PATH = '/burst/dynamic'


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
    # placing saved Dynamics on the regions of the Subnetwork being configured, in the third column
    CONFIGURE_REGION_MODEL_URL = '/burst/hybrid/configure_region_model'
    APPLY_REGION_MODEL_URL = '/burst/hybrid/apply_region_model'
    SUBMIT_REGION_MODEL_URL = '/burst/hybrid/submit_region_model'
    # the wizard step generating the Projections out of the Connectivity and the Subnetwork grouping
    SET_PROJECTIONS_URL = '/burst/hybrid/set_projections'
    # the global configuration: what the simulation records, for how long, and what that produces
    SET_MONITORS_URL = '/burst/hybrid/set_monitors'
    SET_MONITOR_PARAMS_URL = '/burst/hybrid/set_monitor_params'
    SET_MONITOR_EQUATION_URL = '/burst/hybrid/set_monitor_equation'
    SET_SIMULATION_SUMMARY_URL = '/burst/hybrid/set_simulation_summary'


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
        # set on the Model parameters step, which offers the Set up region Model action
        self.include_region_model_button = False
        # Where Next posts, when that is not this step's own action url. The closing step of the
        # Subnetwork configuration needs this: its own url stores the dynamics, so Next has to go
        # somewhere else, and it may not equal the answer's url or the client would take the answer
        # for a rejection of this step rather than for the next one.
        self.next_form_action_url = None
        # True for a step rendered as the record of something already configured: fields disabled and
        # buttons hidden, which is what the client's own locking does at runtime. Rendering it this way
        # lets the whole configuration of a Subnetwork arrive already read only.
        self.is_read_only = False
        # The steps of one Subnetwork's configuration, in order, when the whole thing is rendered at
        # once instead of being stepped through. All but the last are read only.
        self.chain_renderers = []
        # the wizard step listing the generated Projections
        self.is_projections_fragment = False
        self.projection_rows = []
        self.unconnected_pairs = []
        # the legend a Monitor parameters step carries, the way the classic Cockpit titles its own
        self.monitor_name = None
        # the closing step of the global configuration, describing what the simulation will record
        self.is_simulation_summary_fragment = False
        self.monitor_rows = []
        self.output_layout = None
        self.simulation_length = None
        # the Region Model panel: the Subnetwork being configured, its regions and the Dynamics on offer
        self.region_model_subnetwork = None
        self.region_model_rows = []
        self.region_model_dynamics = []
        self.region_model_unassigned = 0
        # Where model configurations are defined. Built by the controller because deploy_context is only
        # put in the template context of a full page, not of a fragment.
        self.phase_plane_url = None

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
    def region_model_model_label(self):
        """
        The Model this Subnetwork is configured with, named the way the Model selector named it.
        """
        subnetwork = self.region_model_subnetwork
        dynamics = subnetwork.dynamics if subnetwork is not None else None
        return self._label_for(self.MODEL_LABELS, dynamics.model if dynamics else None)

    @property
    def region_model_json(self):
        """
        The Region Model panel state, as consumed by the hybrid_region_model.js client side component.
        """
        payload = json.dumps({
            'rows': self.region_model_rows,
            'unassigned': self.region_model_unassigned
        })
        # the result is inlined inside a <script> tag, so no region label may close it
        return payload.replace('<', '\\u003c')

    # The names the Model and Integrator selectors offered, by class, so a Subnetwork is described with
    # the same words it was configured with rather than with a class name.
    MODEL_LABELS = {member.value: str(member) for member in ModelsEnum}
    INTEGRATOR_LABELS = {member.value: str(member) for member in IntegratorViewModelsEnum}
    NOISE_LABELS = {member.value: str(member) for member in NoiseTypesEnum}

    @classmethod
    def _label_for(cls, labels, instance):
        if instance is None:
            return ''
        return labels.get(type(instance), type(instance).__name__.replace('ViewModel', ''))

    @property
    def subnetwork_choices(self):
        """
        One entry per Subnetwork for the selector: what it holds and what is configured for it. This is
        the only place the Subnetwork dynamics step describes them, so it carries the Model, the
        Integrator and, for a stochastic one, its Noise.
        """
        choices = []
        for subnetwork in self.subnetworks or []:
            dynamics = self.dynamics_by_id.get(subnetwork.id) or subnetwork.dynamics
            integrator = dynamics.integrator if dynamics else None
            noise = getattr(integrator, 'noise', None)
            choices.append({
                'id': subnetwork.id,
                'name': subnetwork.name,
                'count': len(subnetwork.node_indices),
                'model': self._label_for(self.MODEL_LABELS, dynamics.model if dynamics else None),
                'integrator': self._label_for(self.INTEGRATOR_LABELS, integrator),
                'noise': self._label_for(self.NOISE_LABELS, noise),
                'is_selected': subnetwork.id == self.selected_subnetwork
            })
        return choices

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

    @expose_fragment('burst/hybrid_subnetwork_chain')
    def select_subnetwork(self, subnetwork_id=None, **data):
        """
        Configure another Subnetwork.

        Answers with its whole configuration at once - every step, all but the last already read only -
        rather than with the first step, so a Subnetwork that is already set up is shown rather than
        stepped through again. What was edited for the Subnetwork being left is kept: the draft holds
        every Subnetwork at once.
        """
        try:
            _, _, subnetworks, _ = self._load_subnetworks_configuration()
            self.hybrid_simulator_service.find_subnetwork(subnetworks, subnetwork_id)
        except HybridSubnetworkException as excep:
            common.set_error_message(str(excep))
            return HybridSimulatorFragmentRenderingRules(
                None, HybridSimulatorURLs.SELECT_SUBNETWORK_URL, load_error=str(excep)).to_dict()

        self.context.set_selected_subnetwork(subnetwork_id)
        dynamics, _ = self._selected_dynamics()
        return self._subnetwork_chain_rules(dynamics).to_dict()

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

    # ---------------------------------------------------------------- Projections

    @expose_fragment('hybrid_simulator_fragment')
    def set_projections(self, **data):
        """
        Generate the Projections the configuration describes and list them, so the resulting NetworkSet
        can be inspected before anything is launched.

        They are derived from the Connectivity and the Subnetwork grouping every time this step is
        rendered rather than stored: nothing here is a user choice yet, and keeping sparse matrices in
        the session would only let them fall out of step with the grouping.
        """
        try:
            hybrid_simulator, _, subnetworks, _ = self._load_subnetworks_configuration()
        except HybridSubnetworkException as excep:
            return self._back_to_connectivity(str(excep))

        draft = self._prepare_dynamics_draft(subnetworks, hybrid_simulator.dt)

        if not self.hybrid_simulator_service.same_dynamics(subnetworks, draft):
            # the Projections are built from the saved dynamics, so they would not describe what is on
            # screen; the same rule the Subnetworks step applies to an unsaved grouping
            return self._back_to_dynamics(
                hybrid_simulator, subnetworks, draft,
                "Save the Subnetwork configuration before generating the Projections.")

        try:
            connectivity = h5.load_from_gid(hybrid_simulator.connectivity)
            network_set = self.hybrid_simulator_service.build_network_set(
                connectivity, subnetworks, hybrid_simulator.dt)
        except Exception as excep:
            self.logger.exception("Could not generate the Hybrid Simulator Projections")
            return self._back_to_dynamics(
                hybrid_simulator, subnetworks, draft,
                "The Projections could not be generated: {}".format(excep))

        self.context.add_last_loaded_form_url_to_session(HybridSimulatorURLs.SET_PROJECTIONS_URL)

        rules = HybridSimulatorFragmentRenderingRules(
            None, HybridSimulatorURLs.SET_PROJECTIONS_URL,
            HybridSimulatorURLs.SAVE_SUBNETWORK_DYNAMICS_URL, fragment_title="Projections")
        rules.is_projections_fragment = True
        rules.projection_rows = self.hybrid_simulator_service.describe_network_set(network_set)
        rules.unconnected_pairs = self.hybrid_simulator_service.unconnected_pairs(network_set)
        # This step is entered by posting to its own url, so its Next has to post somewhere else, the
        # same way the closing step of the Subnetwork configuration does.
        rules.next_form_action_url = HybridSimulatorURLs.SET_MONITORS_URL
        return rules.to_dict()

    def _back_to_dynamics(self, hybrid_simulator, subnetworks, draft, message):
        """
        Refuse to move on and hand the Subnetwork dynamics step back, so the wizard never shows a step
        built from a configuration the user has not saved.
        """
        common.set_error_message(message)
        self.context.add_last_loaded_form_url_to_session(HybridSimulatorURLs.SET_SUBNETWORK_DYNAMICS_URL)
        form = HybridSubnetworkDynamicsFragment()
        form.fill_from_trait(hybrid_simulator)
        return self._dynamics_step_rules(form, subnetworks, draft).to_dict()

    # ---------------------------------------------------------------- Monitors

    @expose_fragment('hybrid_simulator_fragment')
    def set_monitors(self, **data):
        """
        The wizard step saying what the simulation records and for how long.

        Answers with the parameters of the first Monitor that has any, and with the closing summary when
        none of them has - a Raw Monitor on its own is the whole of that case.
        """
        try:
            hybrid_simulator, _, _, _ = self._load_subnetworks_configuration()
        except HybridSubnetworkException as excep:
            return self._back_to_connectivity(str(excep))

        if cherrypy.request.method == POST_REQUEST:
            form = HybridMonitorsFragment()
            form.fill_from_post(data)
            if not form.validate():
                return self._monitors_step_rules(form).to_dict()

            form.fill_trait(hybrid_simulator)
            # the Monitor selector is built on an ad hoc List carrying no field_name, so fill_trait
            # skips it and the Monitors are built here, the way the classic Cockpit builds its own
            hybrid_simulator.monitors = form.monitors_from_post()
            self.context.set_hybrid_simulator(hybrid_simulator)

            return self._monitor_chain_step(hybrid_simulator, 0)

        return self._monitors_step()

    @expose_fragment('hybrid_simulator_fragment')
    def set_monitor_params(self, current_monitor_name, **data):
        """
        The parameters of one Monitor. Answers with the next step of the Monitor chain, which is this
        Monitor's Equation for a BOLD one, the next Monitor's parameters, or the closing summary.
        """
        return self._handle_monitor_step(current_monitor_name, data, is_equation=False)

    @expose_fragment('hybrid_simulator_fragment')
    def set_monitor_equation(self, current_monitor_name, **data):
        """
        The parameters of the haemodynamic response Equation of a BOLD Monitor.
        """
        return self._handle_monitor_step(current_monitor_name, data, is_equation=True)

    @expose_fragment('hybrid_simulator_fragment')
    def set_simulation_summary(self, **data):
        """
        The closing step of the global configuration: what each Monitor records, over how long, and the
        shape of the array that produces.
        """
        return self._simulation_summary_step()

    def _handle_monitor_step(self, current_monitor_name, data, is_equation):
        try:
            hybrid_simulator, _, _, _ = self._load_subnetworks_configuration()
        except HybridSubnetworkException as excep:
            return self._back_to_connectivity(str(excep))

        index = self._monitor_step_index(hybrid_simulator, current_monitor_name, is_equation)
        if index is None:
            # the Monitor selection changed under this step, so start the chain again
            return self._monitors_step()

        _, monitor, _ = self._monitor_chain(hybrid_simulator)[index]

        if cherrypy.request.method == POST_REQUEST:
            form = self._monitor_form(hybrid_simulator, monitor, is_equation)
            form.fill_from_post(data)
            if not form.validate():
                return self._monitor_chain_step(hybrid_simulator, index, form)

            form.fill_trait(monitor.hrf_kernel if is_equation else monitor)

            try:
                # checked on this Monitor's own step rather than on the closing summary: a refusal has
                # to hand back the step it is about, and the summary would instead answer with a step
                # already on screen
                self.hybrid_simulator_service.validate_monitors(
                    [monitor], hybrid_simulator.simulation_length, hybrid_simulator.dt)
            except HybridSubnetworkException as excep:
                common.set_error_message(str(excep))
                return self._monitor_chain_step(hybrid_simulator, index)

            self.context.set_hybrid_simulator(hybrid_simulator)
            return self._monitor_chain_step(hybrid_simulator, index + 1)

        return self._monitor_chain_step(hybrid_simulator, index)

    # ---------------------------------------------------------------- Monitor chain

    @staticmethod
    def _build_monitor_url(url, monitor):
        """
        The action url of one Monitor's step. The class name is a path segment, which is how the exposed
        method receives it - the same shape the classic Cockpit gives its own Monitor steps.
        """
        return '{}/{}'.format(url, type(monitor).__name__)

    def _monitor_chain(self, hybrid_simulator):
        """
        The steps configuring the chosen Monitors, in the order they were chosen.

        :return: one ``(url, monitor, is_equation)`` per step: the parameters of every Monitor that has
                 any, each BOLD Monitor followed by its Equation. A Raw Monitor contributes none - it
                 records every integration step and documents its sampling period as ignored.
        """
        chain = []
        for monitor in hybrid_simulator.monitors or []:
            if isinstance(monitor, RawViewModel):
                continue
            chain.append((self._build_monitor_url(HybridSimulatorURLs.SET_MONITOR_PARAMS_URL, monitor),
                          monitor, False))
            if isinstance(monitor, BoldViewModel):
                chain.append((self._build_monitor_url(HybridSimulatorURLs.SET_MONITOR_EQUATION_URL, monitor),
                              monitor, True))
        return chain

    def _monitor_step_index(self, hybrid_simulator, monitor_name, is_equation):
        """
        :return: the position of the given Monitor step in the chain, or None when the Monitor selection
                 no longer holds it
        """
        for index, (_, monitor, step_is_equation) in enumerate(self._monitor_chain(hybrid_simulator)):
            if type(monitor).__name__ == monitor_name and step_is_equation == is_equation:
                return index
        return None

    def _monitor_form(self, hybrid_simulator, monitor, is_equation):
        if is_equation:
            return get_form_for_equation(type(monitor.hrf_kernel))()

        form_class = get_form_for_hybrid_monitor(type(monitor))
        if issubclass(form_class, HybridSpatialAverageMonitorForm):
            # which default masks are on offer depends on what this Connectivity carries
            form = form_class(connectivity_gid=hybrid_simulator.connectivity)
        else:
            form = form_class()
        return self.algorithm_service.prepare_adapter_form(form_instance=form,
                                                           project_id=self.context.project.id)

    def _monitor_chain_step(self, hybrid_simulator, index, form=None):
        """
        Render the Monitor chain step on the given position, or the closing summary once the chain is
        exhausted.
        """
        chain = self._monitor_chain(hybrid_simulator)
        if index >= len(chain):
            return self._simulation_summary_step()

        url, monitor, is_equation = chain[index]
        previous_url = chain[index - 1][0] if index > 0 else HybridSimulatorURLs.SET_MONITORS_URL

        if form is None:
            form = self._monitor_form(hybrid_simulator, monitor, is_equation)
            form.fill_from_trait(monitor.hrf_kernel if is_equation else monitor)

        self.context.add_last_loaded_form_url_to_session(url)
        rules = self._step_rules(form, url, previous_url)
        rules.monitor_name = self._monitor_legend(monitor, is_equation)
        return rules.to_dict()

    @staticmethod
    def _monitor_legend(monitor, is_equation):
        name = get_monitor_to_ui_name_dict(HybridMonitorsFragment.IS_SURFACE_SIMULATION).get(
            type(monitor), type(monitor).__name__.replace('ViewModel', ''))
        if is_equation:
            return '{} monitor - haemodynamic response'.format(name)
        return '{} monitor'.format(name)

    # ---------------------------------------------------------------- Global configuration steps

    def _monitors_step(self, form=None):
        try:
            hybrid_simulator, _, _, _ = self._load_subnetworks_configuration()
        except HybridSubnetworkException as excep:
            return self._back_to_connectivity(str(excep))

        if form is None:
            form = HybridMonitorsFragment()
            form.fill_from_trait(hybrid_simulator)
        return self._monitors_step_rules(form).to_dict()

    def _monitors_step_rules(self, form):
        self.context.add_last_loaded_form_url_to_session(HybridSimulatorURLs.SET_MONITORS_URL)
        return self._step_rules(form, HybridSimulatorURLs.SET_MONITORS_URL,
                                HybridSimulatorURLs.SET_PROJECTIONS_URL)

    def _simulation_summary_step(self):
        """
        Describe what the configured Monitors will record.

        Purely descriptive: a sampling period that could not record anything is refused on the step of
        the Monitor it belongs to, on the way here.
        """
        try:
            hybrid_simulator, _, subnetworks, _ = self._load_subnetworks_configuration()
        except HybridSubnetworkException as excep:
            return self._back_to_connectivity(str(excep))

        chain = self._monitor_chain(hybrid_simulator)
        previous_url = chain[-1][0] if chain else HybridSimulatorURLs.SET_MONITORS_URL

        self.context.add_last_loaded_form_url_to_session(HybridSimulatorURLs.SET_SIMULATION_SUMMARY_URL)
        rules = HybridSimulatorFragmentRenderingRules(
            None, HybridSimulatorURLs.SET_SIMULATION_SUMMARY_URL, previous_url,
            fragment_title="Simulation",
            # launching is the next wizard step, it does not exist yet
            next_button_enabled=False)
        rules.is_simulation_summary_fragment = True
        rules.simulation_length = hybrid_simulator.simulation_length
        rules.monitor_rows = self._monitor_rows(hybrid_simulator)
        rules.output_layout = self.hybrid_simulator_service.output_layout(subnetworks)
        return rules.to_dict()

    def _monitor_rows(self, hybrid_simulator):
        """
        One row per configured Monitor: what it is called and how often it samples.
        """
        rows = []
        for monitor in hybrid_simulator.monitors or []:
            rows.append({
                'name': self._monitor_legend(monitor, False).replace(' monitor', ''),
                # Raw ignores its own period and records every integration step
                'period': hybrid_simulator.dt if isinstance(monitor, RawViewModel) else float(monitor.period),
                'is_raw': isinstance(monitor, RawViewModel)
            })
        return rows

    # ---------------------------------------------------------------- Region Model

    @expose_fragment('burst/hybrid_region_model')
    def configure_region_model(self, **data):
        """
        Place saved Dynamics on the regions of the Subnetwork being configured, shown in the third
        column. Only that Subnetwork's own regions are listed: its Model applies to the nodes it owns, so
        a parameter value is needed for each of those and for no other.
        """
        try:
            rules = self._region_model_rules()
        except HybridSubnetworkException as excep:
            rules = HybridSimulatorFragmentRenderingRules(
                None, HybridSimulatorURLs.CONFIGURE_REGION_MODEL_URL, load_error=str(excep))
            rules.phase_plane_url = self.build_path(PHASE_PLANE_PATH)
            return rules.to_dict()
        return rules.to_dict()

    @expose_json
    def apply_region_model(self, dynamic_id=None, node_indices=None, **data):
        """
        Put the given Model configuration on the given regions. This only records the placement, the way
        the classic page's Apply to selected nodes does; Submit is what writes it onto the Model.
        """
        try:
            subnetwork, dynamics, assignment, region_labels = self._region_model_state()
        except HybridSubnetworkException as excep:
            return {'status': 'error', 'message': str(excep), 'rows': [], 'unassigned': 0}

        dynamics_by_id = {dynamic.id: dynamic for dynamic in dynamics}
        try:
            chosen_id = int(dynamic_id)
        except (TypeError, ValueError):
            chosen_id = None

        if chosen_id not in dynamics_by_id:
            return self._region_model_state_answer(
                subnetwork, dynamics_by_id, assignment, region_labels,
                "This Model configuration is not available for this Subnetwork.", is_error=True)

        nodes = self._parse_node_indices(node_indices)
        if nodes is None:
            return self._region_model_state_answer(
                subnetwork, dynamics_by_id, assignment, region_labels,
                "The regions to configure could not be read.", is_error=True)

        owned = [node_index for node_index in nodes if node_index in set(subnetwork.node_indices)]
        if not owned:
            return self._region_model_state_answer(
                subnetwork, dynamics_by_id, assignment, region_labels,
                "Select the regions this Model configuration should be placed on.", is_error=True)

        assignment = self.hybrid_simulator_service.place_dynamic_on_regions(assignment, owned, chosen_id)
        self._store_region_model(subnetwork, assignment)

        message = "Model configuration placed on {} region{}.".format(
            len(owned), 's' if len(owned) != 1 else '')
        unassigned = self.hybrid_simulator_service.unassigned_count(list(subnetwork.node_indices), assignment)
        if unassigned == 0:
            message += " Press Submit to put these values in the Model parameters."
        else:
            message += " {} still without one.".format(unassigned)

        return self._region_model_state_answer(subnetwork, dynamics_by_id, assignment, region_labels, message)

    @expose_fragment('hybrid_simulator_fragment')
    def submit_region_model(self, **data):
        """
        Write what was placed on the regions onto the Subnetwork's Model, as one array per parameter, and
        answer with the refreshed Model parameters step so the values show up there.

        Like every other Phase 3 edit this reaches the draft only; the existing Save Configuration is
        what stores it on the Hybrid Simulator configuration.
        """
        try:
            subnetwork, dynamics, assignment, _ = self._region_model_state()
        except HybridSubnetworkException as excep:
            return self._back_to_connectivity(str(excep))

        dynamics_by_id = {dynamic.id: dynamic for dynamic in dynamics}
        try:
            self.hybrid_simulator_service.apply_dynamics_to_model(
                subnetwork.dynamics.model, list(subnetwork.node_indices), assignment, dynamics_by_id)
        except HybridSubnetworkException as excep:
            # answers with the step unchanged, so the values on screen keep matching the configuration
            common.set_error_message(str(excep))
            return self._model_params_step_rules(subnetwork.dynamics).to_dict()

        common.set_info_message(
            "The Model parameters of '{}' now hold one value per region.".format(subnetwork.name))
        return self._model_params_step_rules(subnetwork.dynamics).to_dict()

    # ---------------------------------------------------------------- Region Model helpers

    def _region_model_state(self):
        """
        :return: the draft Subnetwork being configured, the Dynamics that can be placed on it, what is
                 placed on its regions already, and the Connectivity region labels
        """
        hybrid_simulator, region_labels, subnetworks, _ = self._load_subnetworks_configuration()
        draft = self._prepare_dynamics_draft(subnetworks, hybrid_simulator.dt)
        selected = self._selected_subnetwork(subnetworks)

        # the panel edits the draft Model, so the existing Save Configuration is what commits it
        subnetwork = self.hybrid_simulator_service.copy_subnetworks([selected])[0]
        subnetwork.dynamics = draft[selected.id]

        dynamics = self.hybrid_simulator_service.dynamics_for_model(
            dao.get_dynamics_for_user(self.context.logged_user.id), subnetwork.dynamics.model)

        assignment = (self.context.region_model or {}).get(selected.id, {})
        # a regrouping may have taken regions away from this Subnetwork since this was last edited
        assignment = self.hybrid_simulator_service.restrict_assignment(assignment, subnetwork.node_indices)

        return subnetwork, dynamics, assignment, region_labels

    def _store_region_model(self, subnetwork, assignment):
        region_model = dict(self.context.region_model or {})
        region_model[subnetwork.id] = assignment
        self.context.set_region_model(region_model)

    def _region_model_rules(self):
        subnetwork, dynamics, assignment, region_labels = self._region_model_state()
        dynamics_by_id = {dynamic.id: dynamic for dynamic in dynamics}

        rules = HybridSimulatorFragmentRenderingRules(
            None, HybridSimulatorURLs.CONFIGURE_REGION_MODEL_URL, fragment_title="Region Model")
        rules.phase_plane_url = self.build_path(PHASE_PLANE_PATH)
        rules.region_model_subnetwork = subnetwork
        rules.region_model_dynamics = dynamics
        rules.region_model_rows = self.hybrid_simulator_service.region_model_rows(
            list(subnetwork.node_indices), region_labels, assignment, dynamics_by_id)
        rules.region_model_unassigned = self.hybrid_simulator_service.unassigned_count(
            list(subnetwork.node_indices), assignment)
        return rules

    def _region_model_state_answer(self, subnetwork, dynamics_by_id, assignment, region_labels,
                                   message, is_error=False):
        return {
            'status': 'error' if is_error else 'ok',
            'message': message,
            'rows': self.hybrid_simulator_service.region_model_rows(
                list(subnetwork.node_indices), region_labels, assignment, dynamics_by_id),
            'unassigned': self.hybrid_simulator_service.unassigned_count(
                list(subnetwork.node_indices), assignment)
        }

    @staticmethod
    def _parse_node_indices(node_indices):
        try:
            nodes = json.loads(node_indices) if node_indices else []
        except ValueError:
            return None
        if not isinstance(nodes, list):
            return None
        try:
            return [int(node_index) for node_index in nodes]
        except (TypeError, ValueError):
            return None

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

    def _subnetwork_chain_rules(self, dynamics):
        """
        The whole configuration of one Subnetwork, as the ordered steps that make it up: Model, Model
        parameters, Integrator, Integrator parameters and, where they apply, Noise and its Equation,
        closed by the step that stores it.

        Every step but the last is marked read only, so the configuration arrives on screen already in
        the state the wizard leaves a finished step in. Stepping back through it with Previous is what
        makes any of it editable again, which the client already does.
        """
        steps = [self._model_step_rules(dynamics),
                 self._model_params_step_rules(dynamics),
                 self._integrator_step_rules(dynamics),
                 self._integrator_params_step_rules(dynamics)]

        if isinstance(dynamics.integrator, IntegratorStochasticViewModel):
            steps.append(self._noise_params_step_rules(dynamics))
            if isinstance(dynamics.integrator.noise, MultiplicativeNoiseViewModel):
                steps.append(self._noise_equation_step_rules(dynamics))

        steps.append(self._save_step_rules(steps[-1].form_action_url))

        for step in steps[:-1]:
            step.is_read_only = True

        rules = HybridSimulatorFragmentRenderingRules(
            None, HybridSimulatorURLs.SELECT_SUBNETWORK_URL)
        rules.chain_renderers = steps
        return rules

    def _model_params_step_rules(self, dynamics, form=None):
        if form is None:
            form = self.algorithm_service.prepare_adapter_form(
                form_instance=get_form_for_model(type(dynamics.model))())
            form.fill_from_trait(dynamics.model)
        rules = self._step_rules(form, HybridSimulatorURLs.SET_SUBNETWORK_MODEL_PARAMS_URL,
                                 HybridSimulatorURLs.SET_SUBNETWORK_MODEL_URL)
        # the same action the classic Cockpit offers next to the Model parameters, except that it fills
        # the third column instead of opening a page of its own
        rules.include_region_model_button = True
        return rules

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
        The closing step of the Subnetwork configuration: nothing left to fill in for this Subnetwork,
        the action writing what was configured onto the Hybrid Simulator configuration, and Next onto
        the Projections.
        """
        rules = HybridSimulatorFragmentRenderingRules(
            None, HybridSimulatorURLs.SAVE_SUBNETWORK_DYNAMICS_URL, previous_form_action_url,
            is_dynamics_save_fragment=True, next_button_label='Save Configuration')
        rules.next_form_action_url = HybridSimulatorURLs.SET_PROJECTIONS_URL
        return rules

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
