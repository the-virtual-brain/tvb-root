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

import cherrypy

from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.forms.hybrid_simulator_fragments import HybridConnectivityFragment, \
    HybridSubnetworksPlaceholderFragment
from tvb.core.entities.file.simulator.view_model import HybridSimulatorAdapterModel
from tvb.core.services.simulator_service import SimulatorService
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.burst.base_controller import BurstBaseController
from tvb.interfaces.web.controllers.decorators import expose_fragment, expose_page, settings, context_selected
from tvb.interfaces.web.controllers.simulator.simulator_fragment_rendering_rules import POST_REQUEST
from tvb.interfaces.web.entities.context_hybrid_simulator import HybridSimulatorContext


class HybridSimulatorURLs(object):
    SET_CONNECTIVITY_URL = '/burst/hybrid/set_connectivity'
    SET_SUBNETWORKS_URL = '/burst/hybrid/set_subnetworks'


class HybridSimulatorFragmentRenderingRules(object):
    FIRST_FORM_URL = HybridSimulatorURLs.SET_CONNECTIVITY_URL

    def __init__(self, form, form_action_url, previous_form_action_url=None, is_first_fragment=False,
                 is_subnetwork_fragment=False, fragment_title=None):
        self.form = form
        self.form_action_url = form_action_url
        self.previous_form_action_url = previous_form_action_url
        self.is_first_fragment = is_first_fragment
        self.is_subnetwork_fragment = is_subnetwork_fragment
        self.fragment_title = fragment_title

    @property
    def include_next_button(self):
        return not self.is_subnetwork_fragment

    @property
    def include_previous_button(self):
        return not self.is_first_fragment

    def to_dict(self):
        return {"renderer": self, "isCallout": False}


@traced
class HybridSimulatorController(BurstBaseController):

    def __init__(self):
        BurstBaseController.__init__(self)
        self.context = HybridSimulatorContext()
        self.simulator_service = SimulatorService()

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

    @expose_page
    @settings
    @context_selected
    def index(self):
        template_specification = dict(mainContent="burst/main_hybrid_simulator", title="Hybrid Simulator",
                                      includedResources='project/included_resources')

        if not self.context.last_loaded_fragment_url:
            self.context.add_last_loaded_form_url_to_session(HybridSimulatorURLs.SET_CONNECTIVITY_URL)

        form = self._prepare_connectivity_form()
        rendering_rules = HybridSimulatorFragmentRenderingRules(
            form, HybridSimulatorURLs.SET_CONNECTIVITY_URL, is_first_fragment=True, fragment_title="Connectivity")

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
        rendering_rules = HybridSimulatorFragmentRenderingRules(
            form, HybridSimulatorURLs.SET_CONNECTIVITY_URL, is_first_fragment=True, fragment_title="Connectivity")
        return rendering_rules.to_dict()

    @expose_fragment('hybrid_simulator_fragment')
    def set_connectivity(self, **data):
        if cherrypy.request.method == POST_REQUEST:
            form = self.algorithm_service.prepare_adapter_form(form_instance=HybridConnectivityFragment(),
                                                               project_id=self.context.project.id)
            form.fill_from_post(data)
            if not form.validate():
                self.context.add_last_loaded_form_url_to_session(HybridSimulatorURLs.SET_CONNECTIVITY_URL)
                rendering_rules = HybridSimulatorFragmentRenderingRules(
                    form, HybridSimulatorURLs.SET_CONNECTIVITY_URL, is_first_fragment=True,
                    fragment_title="Connectivity")
                return rendering_rules.to_dict()

            self.context.set_hybrid_simulator(HybridSimulatorAdapterModel())
            form.fill_trait(self.context.hybrid_simulator)
            self.context.add_last_loaded_form_url_to_session(HybridSimulatorURLs.SET_SUBNETWORKS_URL)
            return self._prepare_subnetworks_fragment()

        form = self._prepare_connectivity_form()
        rendering_rules = HybridSimulatorFragmentRenderingRules(
            form, HybridSimulatorURLs.SET_CONNECTIVITY_URL, is_first_fragment=True, fragment_title="Connectivity")
        return rendering_rules.to_dict()

    @expose_fragment('hybrid_simulator_fragment')
    def set_subnetworks(self, **data):
        self.context.add_last_loaded_form_url_to_session(HybridSimulatorURLs.SET_SUBNETWORKS_URL)
        return self._prepare_subnetworks_fragment()

    @staticmethod
    def _prepare_subnetworks_fragment():
        form = HybridSubnetworksPlaceholderFragment()
        rendering_rules = HybridSimulatorFragmentRenderingRules(
            form, HybridSimulatorURLs.SET_SUBNETWORKS_URL, HybridSimulatorURLs.SET_CONNECTIVITY_URL,
            is_subnetwork_fragment=True, fragment_title="Subnetworks")
        return rendering_rules.to_dict()
