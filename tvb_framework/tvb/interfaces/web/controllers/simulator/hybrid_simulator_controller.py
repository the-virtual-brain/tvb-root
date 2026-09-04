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
from tvb.adapters.forms.hybrid_simulator_fragments import HybridConnectivityFragment, HybridSubnetworksFragment
from tvb.core.entities.file.simulator.view_model import HybridSimulatorAdapterModel
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


class HybridSimulatorFragmentRenderingRules(object):
    FIRST_FORM_URL = HybridSimulatorURLs.SET_CONNECTIVITY_URL

    def __init__(self, form, form_action_url, previous_form_action_url=None, is_first_fragment=False,
                 is_subnetworks_summary_fragment=False, fragment_title=None, next_button_label='Next',
                 previous_button_label='Previous', next_button_enabled=True, region_labels=None,
                 subnetworks=None, context_form_url=None, context_title=None, is_modified=False,
                 load_error=None):
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
        """
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

    # ---------------------------------------------------------------- Helpers

    def _subnetworks_step_rules(self, region_labels, subnetworks, draft):
        return HybridSimulatorFragmentRenderingRules(
            HybridSubnetworksFragment(), HybridSimulatorURLs.SET_SUBNETWORKS_URL,
            HybridSimulatorURLs.SET_CONNECTIVITY_URL, is_subnetworks_summary_fragment=True,
            fragment_title="Subnetworks", region_labels=region_labels, subnetworks=subnetworks,
            context_form_url=HybridSimulatorURLs.CONFIGURE_SUBNETWORKS_URL, context_title="Subnetworks",
            is_modified=self._is_modified(subnetworks, draft),
            # Model and Integrator configuration is the next wizard step, it does not exist yet
            next_button_enabled=False)

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
