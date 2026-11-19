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

from tvb.adapters.forms.integrator_forms import NoiseTypesEnum
from tvb.adapters.forms.model_forms import ModelsEnum
from tvb.core.entities.file.simulator.view_model import IntegratorViewModelsEnum
from tvb.core.services.hybrid_simulator_service import HybridSimulatorService
from tvb.interfaces.web.controllers.simulator.hybrid_simulator_wizzard_urls import HybridSimulatorURLs


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
        # The url answering with the step after this one, followed after a page reload to bring the
        # steps back on screen, the way the classic Cockpit follows each step's action url. None for
        # the last step.
        self.next_step_url = None
        # True for a step rendered as the record of something already configured: fields disabled and
        # buttons hidden, which is what the client's own locking does at runtime.
        self.is_read_only = False
        # The Subnetwork editor: every section of one Subnetwork's dynamics on screen at once, closed by
        # the row of actions storing them. Each section is its own form, applied as soon as it changes.
        self.is_subnetwork_editor = False
        self.editor_sections = []
        self.editor_subnetwork_name = None
        # set on a section of the Subnetwork editor, which carries no Next or Previous of its own
        self.is_editor_section = False
        # how deep a section of the Subnetwork editor is nested: 1 for the Noise of the Integrator,
        # 2 for the Equation of that Noise, indented as many tabs
        self.section_level = 0
        # a refusal the client has to show, since a message set on the session only reaches a full page
        self.error_message = None
        # the wizard step listing the generated Projections
        self.is_projections_fragment = False
        self.projection_rows = []
        self.unconnected_pairs = []
        self.projection_warnings = []
        # the Inter tab's pickers: the Subnetworks sending an InterProjection, the targets of the one
        # chosen, the InterProjection they point at, and how many of them are added to the simulation
        self.inter_sources = []
        self.inter_targets = []
        self.inter_projection = None
        self.inter_added = 0
        self.inter_total = 0
        # the Intra tab's picker: one option per Subnetwork, the IntraProjection it points at, and how
        # many of them are added to the simulation
        self.intra_options = []
        self.intra_projection = None
        self.intra_added = 0
        # the Projection whose tab is selected, by its key
        self.selected_projection = None
        # the editor under that tab: the Projection's own form, its coupling function's parameters, the
        # pair it connects, whether the last change reached the configuration and why not
        self.projection_form = None
        self.coupling_form = None
        self.projection_row = None
        self.projection_applied = False
        self.projection_error = None
        # the legend a Monitor parameters step carries, the way the classic Cockpit titles its own
        self.monitor_name = None
        # a step of the Monitors section: the Monitor selection and each Monitor's parameters, outlined
        # as one section
        self.is_monitor_step = False
        # the closing step of the global configuration: the simulation length and name, and Launch
        self.is_simulation_summary_fragment = False
        # True on the step carrying the Launch button
        self.is_launch_fragment = False
        # how the Monitors will lay out what they record, for the warning when it is not region ordered
        self.output_layout = None
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
                'is_selected': subnetwork.id == self.selected_subnetwork,
                # the draft of this Subnetwork holds something Save Configuration has not stored yet
                'is_modified': (HybridSimulatorService.dynamics_signature(dynamics) !=
                                HybridSimulatorService.dynamics_signature(subnetwork.dynamics))
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
