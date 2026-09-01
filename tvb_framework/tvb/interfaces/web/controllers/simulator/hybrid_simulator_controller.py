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
    # the wizard step listing the configured Subnetworks, shown in the cockpit layout
    SET_SUBNETWORKS_URL = '/burst/hybrid/set_subnetworks'
    # the full width board on which the regions are actually grouped
    CONFIGURE_SUBNETWORKS_URL = '/burst/hybrid/configure_subnetworks'
    ADD_SUBNETWORK_URL = '/burst/hybrid/add_subnetwork'
    REMOVE_SUBNETWORK_URL = '/burst/hybrid/remove_subnetwork'
    RENAME_SUBNETWORK_URL = '/burst/hybrid/rename_subnetwork'
    MOVE_REGIONS_URL = '/burst/hybrid/move_regions'


class HybridSimulatorFragmentRenderingRules(object):
    FIRST_FORM_URL = HybridSimulatorURLs.SET_CONNECTIVITY_URL

    CONFIGURE_SUBNETWORKS_URL = HybridSimulatorURLs.CONFIGURE_SUBNETWORKS_URL

    def __init__(self, form, form_action_url, previous_form_action_url=None, is_first_fragment=False,
                 is_subnetwork_fragment=False, is_subnetworks_summary_fragment=False, fragment_title=None,
                 next_button_label='Next', previous_button_label='Previous', next_button_enabled=True,
                 region_labels=None, subnetworks=None):
        self.form = form
        self.form_action_url = form_action_url
        self.previous_form_action_url = previous_form_action_url
        self.is_first_fragment = is_first_fragment
        # the full width Subnetwork grouping board
        self.is_subnetwork_fragment = is_subnetwork_fragment
        # the wizard step listing what the board produced
        self.is_subnetworks_summary_fragment = is_subnetworks_summary_fragment
        self.fragment_title = fragment_title
        self.next_button_label = next_button_label
        self.previous_button_label = previous_button_label
        self.next_button_enabled = next_button_enabled
        self.region_labels = region_labels
        self.subnetworks = subnetworks

    @property
    def include_next_button(self):
        # the grouping board is a detour from the wizard, it only leads back to the Subnetworks step
        return not self.is_subnetwork_fragment

    @property
    def include_previous_button(self):
        return not self.is_first_fragment

    @property
    def include_configure_subnetworks_button(self):
        return self.is_subnetworks_summary_fragment

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
            'subnetworks': HybridSimulatorService.to_json_ready(self.subnetworks or [])
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
            self.context.set_hybrid_simulator(hybrid_simulator)

            self.context.add_last_loaded_form_url_to_session(HybridSimulatorURLs.SET_SUBNETWORKS_URL)
            return self._prepare_subnetworks_fragment(is_summary=True)

        form = self._prepare_connectivity_form()
        return self._connectivity_rendering_rules(form).to_dict()

    @expose_fragment('hybrid_simulator_fragment')
    def set_subnetworks(self, **data):
        """
        The wizard step listing the configured Subnetworks, rendered in the usual cockpit layout.
        """
        self.context.add_last_loaded_form_url_to_session(HybridSimulatorURLs.SET_SUBNETWORKS_URL)
        return self._prepare_subnetworks_fragment(is_summary=True)

    @expose_fragment('hybrid_simulator_fragment')
    def configure_subnetworks(self, **data):
        """
        The full width board on which the Connectivity regions are grouped into Subnetworks.
        """
        self.context.add_last_loaded_form_url_to_session(HybridSimulatorURLs.CONFIGURE_SUBNETWORKS_URL)
        return self._prepare_subnetworks_fragment(is_summary=False)

    def _prepare_subnetworks_fragment(self, is_summary):
        try:
            _, region_labels, subnetworks = self._load_subnetworks_configuration()
        except HybridSubnetworkException as excep:
            common.set_error_message(str(excep))
            self.context.add_last_loaded_form_url_to_session(HybridSimulatorURLs.SET_CONNECTIVITY_URL)
            return self._connectivity_rendering_rules(self._prepare_connectivity_form()).to_dict()

        if is_summary:
            rendering_rules = HybridSimulatorFragmentRenderingRules(
                HybridSubnetworksFragment(), HybridSimulatorURLs.SET_SUBNETWORKS_URL,
                HybridSimulatorURLs.SET_CONNECTIVITY_URL, is_subnetworks_summary_fragment=True,
                fragment_title="Subnetworks", region_labels=region_labels, subnetworks=subnetworks,
                # Model and Integrator configuration is the next wizard step, it does not exist yet
                next_button_enabled=False)
        else:
            rendering_rules = HybridSimulatorFragmentRenderingRules(
                HybridSubnetworksFragment(), HybridSimulatorURLs.CONFIGURE_SUBNETWORKS_URL,
                HybridSimulatorURLs.SET_SUBNETWORKS_URL, is_subnetwork_fragment=True,
                fragment_title="Subnetworks", previous_button_label="Back to Simulator",
                region_labels=region_labels, subnetworks=subnetworks)
        return rendering_rules.to_dict()

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

    def _load_subnetworks_configuration(self):
        """
        :return: the session stored Hybrid Simulator configuration, the Connectivity region labels and the
                 Subnetworks, after making sure these still describe a valid partition of the Connectivity
        """
        hybrid_simulator = self.context.hybrid_simulator
        connectivity_gid = self._configured_connectivity(hybrid_simulator)
        if connectivity_gid is None:
            raise HybridSubnetworkException("Select a Connectivity before configuring the Subnetworks.")

        region_labels = self.hybrid_simulator_service.get_region_labels(connectivity_gid)
        subnetworks = self.hybrid_simulator_service.prepare_subnetworks(hybrid_simulator, len(region_labels))
        self.context.set_hybrid_simulator(hybrid_simulator)
        return hybrid_simulator, region_labels, subnetworks

    def _change_subnetworks(self, change, success_message):
        """
        Apply one Subnetwork change on the session stored configuration and describe the resulting state.
        A change that would make the configuration invalid is refused, and the current state is returned.
        """
        try:
            hybrid_simulator, region_labels, subnetworks = self._load_subnetworks_configuration()
        except HybridSubnetworkException as excep:
            return self._subnetworks_state([], [], str(excep), is_error=True)

        try:
            subnetworks = change(subnetworks)
        except HybridSubnetworkException as excep:
            return self._subnetworks_state(region_labels, subnetworks, str(excep), is_error=True)

        hybrid_simulator.subnetworks = subnetworks
        self.context.set_hybrid_simulator(hybrid_simulator)
        return self._subnetworks_state(region_labels, subnetworks, success_message)

    @staticmethod
    def _subnetworks_state(region_labels, subnetworks, message, is_error=False):
        return {'status': 'error' if is_error else 'ok',
                'message': message,
                'region_labels': list(region_labels),
                'subnetworks': HybridSimulatorService.to_json_ready(subnetworks)}

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
