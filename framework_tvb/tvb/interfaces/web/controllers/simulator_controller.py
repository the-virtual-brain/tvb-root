# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

import threading
from cherrypy.lib.static import serve_file
from tvb.adapters.datatypes.db.simulation_history import SimulationHistoryIndex
from tvb.adapters.exporters.export_manager import ExportManager
from tvb.adapters.simulator.equation_forms import get_form_for_equation
from tvb.adapters.simulator.model_forms import get_form_for_model
from tvb.adapters.simulator.noise_forms import get_form_for_noise
from tvb.adapters.simulator.range_parameters import SimulatorRangeParameters
from tvb.adapters.simulator.simulator_adapter import SimulatorAdapterForm
from tvb.adapters.simulator.simulator_fragments import *
from tvb.adapters.simulator.monitor_forms import get_form_for_monitor
from tvb.adapters.simulator.integrator_forms import get_form_for_integrator
from tvb.adapters.simulator.coupling_forms import get_form_for_coupling
from tvb.core.entities.file.simulator.view_model import CortexViewModel, SimulatorAdapterModel
from tvb.core.services.simulator_serializer import SimulatorSerializer
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.entities.storage import dao
from tvb.core.services.burst_service import BurstService
from tvb.core.services.exceptions import BurstServiceException, ServicesBaseException
from tvb.core.services.simulator_service import SimulatorService
from tvb.core.neocom import h5
from tvb.config.init.introspector_registry import IntrospectionRegistry
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.burst.base_controller import BurstBaseController
from tvb.interfaces.web.controllers.flow_controller import FlowController
from tvb.interfaces.web.controllers.decorators import *
from tvb.simulator.integrators import IntegratorStochastic
from tvb.simulator.monitors import Bold, Projection, Raw
from tvb.simulator.noise import Additive


class SimulatorWizzardURLs(object):
    SET_CONNECTIVITY_URL = '/burst/set_connectivity'
    SET_COUPLING_PARAMS_URL = '/burst/set_coupling_params'
    SET_SURFACE_URL = '/burst/set_surface'
    SET_STIMULUS_URL = '/burst/set_stimulus'
    SET_CORTEX_URL = '/burst/set_cortex'
    SET_MODEL_URL = '/burst/set_model'
    SET_MODEL_PARAMS_URL = '/burst/set_model_params'
    SET_INTEGRATOR_URL = '/burst/set_integrator'
    SET_INTEGRATOR_PARAMS_URL = '/burst/set_integrator_params'
    SET_NOISE_PARAMS_URL = '/burst/set_noise_params'
    SET_NOISE_EQUATION_PARAMS_URL = '/burst/set_noise_equation_params'
    SET_MONITORS_URL = '/burst/set_monitors'
    SET_MONITOR_PARAMS_URL = '/burst/set_monitor_params'
    SET_MONITOR_EQUATION_URL = '/burst/set_monitor_equation'
    SETUP_PSE_URL = '/burst/setup_pse'
    SET_PSE_PARAMS_URL = '/burst/set_pse_params'
    LAUNCH_PSE_URL = '/burst/launch_pse'


class SimulatorFragmentRenderingRules(object):
    """
    This class gathers the rendering rules for simulator_fragment template.
    TVB > 2.0 brings a change of UI inside the simulator configuration page.
    Instead of a pre-loaded huge form with all the configurations, we want to have a wizzard-like page.
    Thus, the configurations are grouped into fragments and each fragment is rendered as a separate form.
    Each form provides the user with Next/Previous buttons, and some of them bring some extras:
        - model form provides the buttons: setup region model parameters/setup surface model parameters;
        - last fragment provides extra Launch/Setup PSE buttons, and a Branch button when it's the case.
    There are several cases that should be taken into consideration for the UX:
        - for a normal configuration, the user would just select the proper configurations and click Next to go to the
        next form. When the Next button is clicked, the current form is made read-only. If the user need to make changes
        to a previous form that is already read-only, he might use the Previous buttons to get there, make the change,
        and come back clicking Next again.
        - from the history list, the user could choose to inspect an existing simulator configuration, by clicking on
        it. This will load the full configuration as read-only forms, without any buttons.
        - from the history list, the user also has the option to copy an existing simulator configuration A and edit it,
        or start a new simulation B using the results from A as initial conditions. This results, in a semi-read-only
        load of simulator configuration A, because the user will have the options:
            - to use the Previous buttons in order to edit configuration A and the Launch/Setup PSE buttons to start a
            new simulation.
            - to use the Branch button in order to start a new simulation B that will use results from A as initial
            conditions.
    So, we need a series of rendering rules, to know when to display/hide each button, and also make fields read-only.
    """

    FIRST_FORM_URL = SimulatorWizzardURLs.SET_CONNECTIVITY_URL

    def __init__(self, form=None, form_action_url=None, previous_form_action_url=None, is_simulation_copy=False,
                 is_simulation_readonly_load=False, last_form_url=SimulatorWizzardURLs.SET_CONNECTIVITY_URL,
                 last_request_type='GET', is_first_fragment=False, is_launch_fragment=False, is_model_fragment=False,
                 is_surface_simulation=False, is_noise_fragment=False, is_launch_pse_fragment=False,
                 is_pse_launch=False, monitor_name=None):
        """
        :param is_first_fragment: True only for the first form in the wizzard, to hide Previous button
        :param is_launch_fragment: True only for the last form in the wizzard to diplay Launch/SetupPSE/Branch, hide Next
        :param is_model_fragment: True only for the model form, to display SetupRegionModelParams/SetupSurfaceModelParams
        :param is_surface_simulation: True only for the model form, if the user is configuring a surface simulation
        :param is_simulation_copy: True only when the user chooses to copy an existing configuration from the history
        :param is_simulation_readonly_load: True when a GET request comes for a certain form, and that means we need to
                                            display it as read-only. Applicable at read-only full configuration load,
                                            at configuration copy, but also, at refresh/redirect time, to keep the most
                                            recent configured state.
        """
        self.form = form
        self.form_action_url = form_action_url
        self.previous_form_action_url = previous_form_action_url
        self.is_simulation_copy = is_simulation_copy
        self._is_simulation_readonly_load = is_simulation_readonly_load
        self.last_form_url = last_form_url
        self.last_request_type = last_request_type
        self.is_first_fragment = is_first_fragment
        self.is_launch_fragment = is_launch_fragment
        self.is_model_fragment = is_model_fragment
        self.is_surface_simulation = is_surface_simulation
        self.is_noise_fragment = is_noise_fragment
        self.is_launch_pse_fragment = is_launch_pse_fragment
        self.is_pse_launch = is_pse_launch
        self.monitor_name = monitor_name

    @property
    def load_readonly(self):
        if self.last_request_type == 'GET' and self.form_action_url != self.last_form_url:
            return True
        return self._is_simulation_readonly_load

    @property
    def disable_fields(self):
        if self.load_readonly:
            return True
        return False

    @property
    def include_next_button(self):
        if self.is_launch_fragment or self.is_launch_pse_fragment:
            return False
        return True

    @property
    def include_previous_button(self):
        if self.is_first_fragment:
            return False
        return True

    @property
    def hide_previous_button(self):
        if self.load_readonly:
            return True
        return False

    @property
    def include_setup_region_model(self):
        if self.is_model_fragment:
            return True
        return False

    @property
    def include_setup_surface_model(self):
        if self.is_model_fragment and self.is_surface_simulation:
            return True
        return False

    @property
    def include_setup_noise(self):
        if self.is_noise_fragment:
            return True
        return False

    @property
    def include_launch_button(self):
        if self.is_launch_fragment and (not self.load_readonly):
            return True
        return False

    @property
    def hide_launch_and_setup_pse_button(self):
        if self.last_form_url != SimulatorWizzardURLs.SETUP_PSE_URL and (not self.load_readonly):
            return True
        return False

    @property
    def include_branch_button(self):
        if self.is_launch_fragment and self.is_simulation_copy and not self.is_pse_launch:
            return True
        return False

    @property
    def include_setup_pse(self):
        if self.is_launch_fragment and (not self.load_readonly):
            return True
        return False

    @property
    def include_launch_pse_button(self):
        if self.is_launch_pse_fragment and (not self.load_readonly):
            return True
        return False

    def to_dict(self):
        return {"renderer": self}


@traced
class SimulatorController(BurstBaseController):
    KEY_IS_LOAD_AFTER_REDIRECT = "is_load_after_redirect"
    DEFAULT_COPY_PREFIX = "copy_of_"

    def __init__(self):
        BurstBaseController.__init__(self)
        self.last_loaded_form_url = SimulatorWizzardURLs.SET_CONNECTIVITY_URL
        self.range_parameters = SimulatorRangeParameters()
        self.burst_service = BurstService()
        self.simulator_service = SimulatorService()
        self.files_helper = FilesHelper()
        self.cached_simulator_algorithm = self.algorithm_service.get_algorithm_by_module_and_class(
            IntrospectionRegistry.SIMULATOR_MODULE, IntrospectionRegistry.SIMULATOR_CLASS)

    def _update_last_loaded_fragment_url(self, current_url):
        self.last_loaded_form_url = current_url
        common.add2session(common.KEY_LAST_LOADED_FORM_URL, self.last_loaded_form_url)

    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def cancel_or_remove_burst(self, burst_id):
        """
        Cancel or Remove the burst entity given by burst_id (and all linked entities: op, DTs)
        :returns True: if the op was successfully.
        """
        burst_id = int(burst_id)
        burst_configuration = self.burst_service.load_burst_configuration(burst_id)
        remove_after_stop = burst_configuration.status != burst_configuration.BURST_RUNNING

        if burst_configuration.fk_operation_group is None:
            op_id = burst_configuration.fk_simulation
            is_group = 0
        else:
            op_id = burst_configuration.fk_operation_group
            is_group = 1

        return FlowController().cancel_or_remove_operation(op_id, is_group, remove_after_stop)

    @expose_page
    @settings
    @context_selected
    def index(self):
        """Get on burst main page"""
        template_specification = dict(mainContent="burst/main_burst", title="Simulation Cockpit",
                                      includedResources='project/included_resources')
        project = common.get_current_project()

        self.last_loaded_form_url = common.get_from_session(common.KEY_LAST_LOADED_FORM_URL)
        if not self.last_loaded_form_url:
            self.last_loaded_form_url = SimulatorWizzardURLs.SET_CONNECTIVITY_URL
            common.add2session(common.KEY_LAST_LOADED_FORM_URL, SimulatorWizzardURLs.SET_CONNECTIVITY_URL)

        burst_config = common.get_from_session(common.KEY_BURST_CONFIG)
        if not burst_config:
            burst_config = BurstConfiguration(project.id)
            common.add2session(common.KEY_BURST_CONFIG, burst_config)

        if burst_config.start_time is not None:
            is_simulator_load = True
            common.add2session(common.KEY_IS_SIMULATOR_LOAD, True)
        else:
            is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD) or False

        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY) or False

        template_specification['burstConfig'] = burst_config
        template_specification['burst_list'] = self.burst_service.get_available_bursts(common.get_current_project().id)
        portlets_list = []  # self.burst_service.get_available_portlets()
        template_specification['portletList'] = portlets_list
        template_specification['selectedPortlets'] = json.dumps(portlets_list)

        form = self.prepare_first_fragment()
        rendering_rules = SimulatorFragmentRenderingRules(form, SimulatorWizzardURLs.SET_CONNECTIVITY_URL, None,
                                                          is_simulator_copy, is_simulator_load,
                                                          last_form_url=self.last_loaded_form_url,
                                                          last_request_type=cherrypy.request.method,
                                                          is_first_fragment=True)

        template_specification.update(**rendering_rules.to_dict())
        return self.fill_default_attributes(template_specification)

    def prepare_first_fragment(self):
        adapter_instance = ABCAdapter.build_adapter(self.cached_simulator_algorithm)
        form = adapter_instance.get_form()('', common.get_current_project().id)

        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        if session_stored_simulator is None:
            session_stored_simulator = SimulatorAdapterModel()
            common.add2session(common.KEY_SIMULATOR_CONFIG, session_stored_simulator)

        form.fill_from_trait(session_stored_simulator)
        return form

    @cherrypy.expose
    @using_template("simulator_fragment")
    @handle_error(redirect=False)
    @check_user
    def set_connectivity(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY) or False
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD) or False
        form = SimulatorAdapterForm()

        if cherrypy.request.method == 'POST':
            self._update_last_loaded_fragment_url(SimulatorWizzardURLs.SET_COUPLING_PARAMS_URL)
            is_simulator_copy = False
            form.fill_from_post(data)

            session_stored_simulator.connectivity = uuid.UUID(form.connectivity.value)
            session_stored_simulator.conduction_speed = form.conduction_speed.value
            coupling = form.coupling.value
            session_stored_simulator.coupling = coupling()

        next_form = get_form_for_coupling(type(session_stored_simulator.coupling))()
        self.range_parameters.coupling_parameters = next_form.get_range_parameters()
        next_form.fill_from_trait(session_stored_simulator.coupling)

        rendering_rules = SimulatorFragmentRenderingRules(next_form, SimulatorWizzardURLs.SET_COUPLING_PARAMS_URL,
                                                          SimulatorWizzardURLs.SET_CONNECTIVITY_URL, is_simulator_copy,
                                                          is_simulator_load, self.last_loaded_form_url,
                                                          cherrypy.request.method)
        return rendering_rules.to_dict()

    @cherrypy.expose
    @using_template("simulator_fragment")
    @handle_error(redirect=False)
    @check_user
    def set_coupling_params(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY) or False
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD) or False

        if cherrypy.request.method == 'POST':
            self._update_last_loaded_fragment_url(SimulatorWizzardURLs.SET_SURFACE_URL)
            is_simulator_copy = False
            form = get_form_for_coupling(type(session_stored_simulator.coupling))()
            form.fill_from_post(data)
            form.fill_trait(session_stored_simulator.coupling)

        surface_fragment = SimulatorSurfaceFragment('', common.get_current_project().id)
        surface_fragment.fill_from_trait(session_stored_simulator)

        rendering_rules = SimulatorFragmentRenderingRules(surface_fragment, SimulatorWizzardURLs.SET_SURFACE_URL,
                                                          SimulatorWizzardURLs.SET_COUPLING_PARAMS_URL,
                                                          is_simulator_copy, is_simulator_load,
                                                          self.last_loaded_form_url, cherrypy.request.method)
        return rendering_rules.to_dict()

    @cherrypy.expose
    @using_template("simulator_fragment")
    @handle_error(redirect=False)
    @check_user
    def set_surface(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY) or False
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD) or False

        surface_index = None
        rendering_rules = SimulatorFragmentRenderingRules(previous_form_action_url=SimulatorWizzardURLs.SET_SURFACE_URL,
                                                          is_simulation_copy=is_simulator_copy,
                                                          is_simulation_readonly_load=is_simulator_load,
                                                          last_form_url=self.last_loaded_form_url,
                                                          last_request_type=cherrypy.request.method)
        if cherrypy.request.method == 'POST':
            is_simulator_copy = False
            form = SimulatorSurfaceFragment()
            form.fill_from_post(data)

            surface_index_gid = form.surface.value
            # surface_index_gid = data['_surface']
            if surface_index_gid is None:
                session_stored_simulator.surface = None
                self._update_last_loaded_fragment_url(SimulatorWizzardURLs.SET_STIMULUS_URL)
            else:
                self._update_last_loaded_fragment_url(SimulatorWizzardURLs.SET_CORTEX_URL)
                surface_index = ABCAdapter.load_entity_by_gid(surface_index_gid)
                session_stored_simulator.surface = CortexViewModel()
                session_stored_simulator.surface.surface_gid = uuid.UUID(surface_index_gid)

        if session_stored_simulator.surface is None:
            stimuli_fragment = SimulatorStimulusFragment('', common.get_current_project().id, False)
            stimuli_fragment.fill_from_trait(session_stored_simulator)

            rendering_rules.form = stimuli_fragment
            rendering_rules.form_action_url = SimulatorWizzardURLs.SET_STIMULUS_URL
            rendering_rules.is_simulation_copy = is_simulator_copy
            return rendering_rules.to_dict()

        # TODO: work-around this situation: surf_index filter
        rm_fragment = SimulatorRMFragment('', common.get_current_project().id, surface_index)
        rm_fragment.fill_from_trait(session_stored_simulator)

        rendering_rules.form = rm_fragment
        rendering_rules.form_action_url = SimulatorWizzardURLs.SET_CORTEX_URL
        rendering_rules.is_simulation_copy = is_simulator_copy
        return rendering_rules.to_dict()

    @cherrypy.expose
    @using_template("simulator_fragment")
    @handle_error(redirect=False)
    @check_user
    def set_cortex(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY) or False
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD) or False

        if cherrypy.request.method == 'POST':
            self._update_last_loaded_fragment_url(SimulatorWizzardURLs.SET_STIMULUS_URL)
            is_simulator_copy = False
            rm_fragment = SimulatorRMFragment()
            rm_fragment.fill_from_post(data)

            session_stored_simulator.surface.coupling_strength = rm_fragment.coupling_strength.data

            lc_gid = rm_fragment.lc.value
            if lc_gid:
                session_stored_simulator.surface.local_connectivity = uuid.UUID(lc_gid)

            rm_gid = rm_fragment.rm.value
            session_stored_simulator.surface.region_mapping_data = uuid.UUID(rm_gid)

        stimuli_fragment = SimulatorStimulusFragment('', common.get_current_project().id, True)
        stimuli_fragment.fill_from_trait(session_stored_simulator)

        rendering_rules = SimulatorFragmentRenderingRules(stimuli_fragment, SimulatorWizzardURLs.SET_STIMULUS_URL,
                                                          SimulatorWizzardURLs.SET_CORTEX_URL, is_simulator_copy,
                                                          is_simulator_load, self.last_loaded_form_url,
                                                          cherrypy.request.method)
        return rendering_rules.to_dict()

    @cherrypy.expose
    @using_template("simulator_fragment")
    @handle_error(redirect=False)
    @check_user
    def set_stimulus(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY) or False
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD) or False

        if cherrypy.request.method == 'POST':
            self._update_last_loaded_fragment_url(SimulatorWizzardURLs.SET_MODEL_URL)
            is_simulator_copy = False
            stimuli_fragment = SimulatorStimulusFragment('', common.get_current_project().id,
                                                         session_stored_simulator.is_surface_simulation)
            stimuli_fragment.fill_from_post(data)
            stimulus_gid = stimuli_fragment.stimulus.value
            if stimulus_gid:
                session_stored_simulator.stimulus = stimulus_gid

        model_fragment = SimulatorModelFragment('', common.get_current_project().id)
        model_fragment.fill_from_trait(session_stored_simulator)

        rendering_rules = SimulatorFragmentRenderingRules(model_fragment, SimulatorWizzardURLs.SET_MODEL_URL,
                                                          SimulatorWizzardURLs.SET_STIMULUS_URL, is_simulator_copy,
                                                          is_simulator_load, self.last_loaded_form_url,
                                                          cherrypy.request.method, is_model_fragment=True,
                                                          is_surface_simulation=session_stored_simulator.is_surface_simulation)
        return rendering_rules.to_dict()

    @cherrypy.expose
    @using_template("simulator_fragment")
    @handle_error(redirect=False)
    @check_user
    def set_model(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY) or False
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD) or False

        if cherrypy.request.method == 'POST':
            self._update_last_loaded_fragment_url(SimulatorWizzardURLs.SET_MODEL_PARAMS_URL)
            is_simulator_copy = False
            form = SimulatorModelFragment()
            form.fill_from_post(data)
            session_stored_simulator.model = form.model.value()

        form = get_form_for_model(type(session_stored_simulator.model))()
        self.range_parameters.model_parameters = form.get_range_parameters()
        form.fill_from_trait(session_stored_simulator.model)

        rendering_rules = SimulatorFragmentRenderingRules(form, SimulatorWizzardURLs.SET_MODEL_PARAMS_URL,
                                                          SimulatorWizzardURLs.SET_MODEL_URL, is_simulator_copy,
                                                          is_simulator_load, self.last_loaded_form_url,
                                                          cherrypy.request.method)
        return rendering_rules.to_dict()

    @cherrypy.expose
    @using_template("simulator_fragment")
    @handle_error(redirect=False)
    @check_user
    def set_model_params(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY) or False
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD) or False

        if cherrypy.request.method == 'POST':
            self._update_last_loaded_fragment_url(SimulatorWizzardURLs.SET_INTEGRATOR_URL)
            is_simulator_copy = False
            form = get_form_for_model(type(session_stored_simulator.model))()
            form.fill_from_post(data)
            form.fill_trait(session_stored_simulator.model)

        integrator_fragment = SimulatorIntegratorFragment('', common.get_current_project().id)
        integrator_fragment.integrator.display_subform = False
        integrator_fragment.fill_from_trait(session_stored_simulator)

        rendering_rules = SimulatorFragmentRenderingRules(integrator_fragment, SimulatorWizzardURLs.SET_INTEGRATOR_URL,
                                                          SimulatorWizzardURLs.SET_MODEL_PARAMS_URL, is_simulator_copy,
                                                          is_simulator_load, self.last_loaded_form_url,
                                                          cherrypy.request.method)
        return rendering_rules.to_dict()

    @cherrypy.expose
    @using_template("simulator_fragment")
    @handle_error(redirect=False)
    @check_user
    def set_integrator(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY) or False
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD) or False

        if cherrypy.request.method == 'POST':
            self._update_last_loaded_fragment_url(SimulatorWizzardURLs.SET_INTEGRATOR_PARAMS_URL)
            is_simulator_copy = False
            fragment = SimulatorIntegratorFragment()
            fragment.fill_from_post(data)
            session_stored_simulator.integrator = fragment.integrator.value()

        form = get_form_for_integrator(type(session_stored_simulator.integrator))()
        if hasattr(form, 'noise'):
            form.noise.display_subform = False
        form.fill_from_trait(session_stored_simulator.integrator)

        rendering_rules = SimulatorFragmentRenderingRules(form, SimulatorWizzardURLs.SET_INTEGRATOR_PARAMS_URL,
                                                          SimulatorWizzardURLs.SET_INTEGRATOR_URL, is_simulator_copy,
                                                          is_simulator_load, self.last_loaded_form_url,
                                                          cherrypy.request.method)
        return rendering_rules.to_dict()

    @cherrypy.expose
    @using_template("simulator_fragment")
    @handle_error(redirect=False)
    @check_user
    def set_integrator_params(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY) or False
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD) or False

        if cherrypy.request.method == 'POST':
            is_simulator_copy = False
            form = get_form_for_integrator(type(session_stored_simulator.integrator))()
            form.fill_from_post(data)
            form.fill_trait(session_stored_simulator.integrator)
            if isinstance(session_stored_simulator.integrator, IntegratorStochastic):
                self._update_last_loaded_fragment_url(SimulatorWizzardURLs.SET_NOISE_PARAMS_URL)
            else:
                self._update_last_loaded_fragment_url(SimulatorWizzardURLs.SET_MONITORS_URL)

        if isinstance(session_stored_simulator.integrator, IntegratorStochastic):
            integrator_noise_fragment = get_form_for_noise(type(session_stored_simulator.integrator.noise))()
            if hasattr(integrator_noise_fragment, 'equation'):
                integrator_noise_fragment.equation.display_subform = False
            self.range_parameters.integrator_noise_parameters = integrator_noise_fragment.get_range_parameters()
            integrator_noise_fragment.fill_from_trait(session_stored_simulator.integrator.noise)

            rendering_rules = SimulatorFragmentRenderingRules(integrator_noise_fragment,
                                                              SimulatorWizzardURLs.SET_NOISE_PARAMS_URL,
                                                              SimulatorWizzardURLs.SET_INTEGRATOR_PARAMS_URL,
                                                              is_simulator_copy, is_simulator_load,
                                                              self.last_loaded_form_url, cherrypy.request.method,
                                                              is_noise_fragment=True)
            return rendering_rules.to_dict()

        monitor_fragment = SimulatorMonitorFragment('', common.get_current_project().id,
                                                    session_stored_simulator.is_surface_simulation)
        monitor_fragment.fill_from_trait(session_stored_simulator.monitors)

        rendering_rules = SimulatorFragmentRenderingRules(monitor_fragment, SimulatorWizzardURLs.SET_MONITORS_URL,
                                                          SimulatorWizzardURLs.SET_INTEGRATOR_PARAMS_URL,
                                                          is_simulator_copy, is_simulator_load,
                                                          self.last_loaded_form_url, cherrypy.request.method)
        return rendering_rules.to_dict()

    @cherrypy.expose
    @using_template("simulator_fragment")
    @handle_error(redirect=False)
    @check_user
    def set_noise_params(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY) or False
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD) or False

        if cherrypy.request.method == 'POST':
            is_simulator_copy = False
            form = get_form_for_noise(type(session_stored_simulator.integrator.noise))()
            form.fill_from_post(data)
            form.fill_trait(session_stored_simulator.integrator.noise)
            if isinstance(session_stored_simulator.integrator.noise, Additive):
                self._update_last_loaded_fragment_url(SimulatorWizzardURLs.SET_MONITORS_URL)
            else:
                self._update_last_loaded_fragment_url(SimulatorWizzardURLs.SET_NOISE_EQUATION_PARAMS_URL)

        if isinstance(session_stored_simulator.integrator.noise, Additive):
            monitor_fragment = SimulatorMonitorFragment('', common.get_current_project().id,
                                                        session_stored_simulator.is_surface_simulation)
            monitor_fragment.fill_from_trait(session_stored_simulator.monitors)

            rendering_rules = SimulatorFragmentRenderingRules(monitor_fragment, SimulatorWizzardURLs.SET_MONITORS_URL,
                                                              SimulatorWizzardURLs.SET_NOISE_PARAMS_URL,
                                                              is_simulator_copy, is_simulator_load,
                                                              self.last_loaded_form_url, cherrypy.request.method)
            return rendering_rules.to_dict()

        equation_form = get_form_for_equation(type(session_stored_simulator.integrator.noise.b))()
        equation_form.equation.data = session_stored_simulator.integrator.noise.b.equation
        equation_form.fill_from_trait(session_stored_simulator.integrator.noise.b)

        rendering_rules = SimulatorFragmentRenderingRules(equation_form,
                                                          SimulatorWizzardURLs.SET_NOISE_EQUATION_PARAMS_URL,
                                                          SimulatorWizzardURLs.SET_NOISE_PARAMS_URL, is_simulator_copy,
                                                          is_simulator_load, self.last_loaded_form_url,
                                                          cherrypy.request.method)
        return rendering_rules.to_dict()

    @cherrypy.expose
    @using_template("simulator_fragment")
    @handle_error(redirect=False)
    @check_user
    def set_noise_equation_params(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY) or False
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD) or False

        if cherrypy.request.method == 'POST':
            self._update_last_loaded_fragment_url(SimulatorWizzardURLs.SET_MONITORS_URL)
            is_simulator_copy = False
            form = get_form_for_equation(type(session_stored_simulator.integrator.noise.b))()
            form.fill_from_post(data)
            form.fill_trait(session_stored_simulator.integrator.noise.b)

        monitor_fragment = SimulatorMonitorFragment('', common.get_current_project().id,
                                                    session_stored_simulator.is_surface_simulation)
        monitor_fragment.fill_from_trait(session_stored_simulator.monitors)

        rendering_rules = SimulatorFragmentRenderingRules(monitor_fragment, SimulatorWizzardURLs.SET_MONITORS_URL,
                                                          SimulatorWizzardURLs.SET_NOISE_EQUATION_PARAMS_URL,
                                                          is_simulator_copy, is_simulator_load,
                                                          self.last_loaded_form_url, cherrypy.request.method)
        return rendering_rules.to_dict()

    @staticmethod
    def _get_variables_of_interest_indexes(all_variables, chosen_variables):
        indexes = {}

        if not isinstance(chosen_variables, list):
            chosen_variables = [chosen_variables]

        for variable in chosen_variables:
            indexes[variable] = all_variables.index(variable)
        return indexes

    @staticmethod
    def _build_list_of_monitors(monitor_names, is_surface_simulation):
        monitor_dict = get_ui_name_to_monitor_dict(is_surface_simulation)
        monitor_classes = []

        for monitor in monitor_names:
            monitor_classes.append(monitor_dict[monitor]())

        return monitor_classes

    @staticmethod
    def _prepare_monitor_legend(is_surface_simulation, monitor):
        return get_monitor_to_ui_name_dict(
            is_surface_simulation)[type(monitor)] + ' monitor'

    @staticmethod
    def build_monitor_url(fragment_url, monitor):
        url_regex = '{}/{}'
        url = url_regex.format(fragment_url, monitor)
        return url

    def _skip_raw_monitor(self, monitors):
        # if the first monitor is Raw, it must be skipped because it does not have parameters
        # also if the only monitor is Raw, the parameters setting phase must be skipped entirely
        first_monitor_index = 0
        if len(monitors) == 1 and isinstance(monitors[0], Raw):
            return first_monitor_index, SimulatorWizzardURLs.SETUP_PSE_URL

        if isinstance(monitors[0], Raw):
            first_monitor_index = 1
        last_loaded_fragment_url = self.build_monitor_url(SimulatorWizzardURLs.SET_MONITOR_PARAMS_URL,
                                                          type(monitors[first_monitor_index]).__name__)
        return first_monitor_index, last_loaded_fragment_url

    @cherrypy.expose
    @using_template("simulator_fragment")
    @handle_error(redirect=False)
    @check_user
    def set_monitors(self, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        session_stored_burst = common.get_from_session(common.KEY_BURST_CONFIG)
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY) or False
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD) or False

        if cherrypy.request.method == 'POST':
            is_simulator_copy = False
            fragment = SimulatorMonitorFragment(is_surface_simulation=session_stored_simulator.is_surface_simulation)
            fragment.fill_from_post(data)

            session_stored_simulator.monitors = self._build_list_of_monitors(
                fragment.monitors.value, session_stored_simulator.is_surface_simulation)

        first_monitor_index, last_loaded_fragment_url = self._skip_raw_monitor(session_stored_simulator.monitors)

        if cherrypy.request.method == 'POST':
            self._update_last_loaded_fragment_url(last_loaded_fragment_url)

        all_variables = session_stored_simulator.model.__class__.variables_of_interest.element_choices
        chosen_variables = session_stored_simulator.model.variables_of_interest
        indexes = self._get_variables_of_interest_indexes(all_variables, chosen_variables)

        monitor = session_stored_simulator.monitors[first_monitor_index]
        form = get_form_for_monitor(type(monitor))(indexes, '', common.get_current_project().id)
        form.fill_from_trait(monitor)

        default_simulation_name, simulation_number = BurstService.prepare_name(session_stored_burst,
                                                                               common.get_current_project().id)

        if isinstance(monitor, Raw) and len(session_stored_simulator.monitors) == 1:
            form = SimulatorFinalFragment(default_simulation_name=default_simulation_name)

            if cherrypy.request.method != 'POST':
                simulation_name = session_stored_burst.name
                if simulation_name is None:
                    simulation_name = 'simulation_' + str(simulation_number)
                form.fill_from_post({'input_simulation_name_id': simulation_name,
                                     'simulation_length': str(session_stored_simulator.simulation_length)})

            form.fill_from_trait(session_stored_simulator)
            rendering_rules = SimulatorFragmentRenderingRules(form, SimulatorWizzardURLs.SETUP_PSE_URL,
                                                              SimulatorWizzardURLs.SET_MONITORS_URL,
                                                              is_simulator_copy, is_simulator_load,
                                                              self.last_loaded_form_url, cherrypy.request.method,
                                                              is_launch_fragment=True,
                                                              is_pse_launch=session_stored_burst.is_pse_burst())
        else:
            monitor_name = self._prepare_monitor_legend(session_stored_simulator.is_surface_simulation, monitor)
            rendering_rules = SimulatorFragmentRenderingRules(form, last_loaded_fragment_url,
                                                              SimulatorWizzardURLs.SET_MONITORS_URL, is_simulator_copy,
                                                              is_simulator_load, self.last_loaded_form_url,
                                                              cherrypy.request.method, monitor_name=monitor_name)

        return rendering_rules.to_dict()

    def _handle_next_fragment_for_monitors(self, session_stored_simulator, next_monitor, session_stored_burst):
        if next_monitor is not None:
            all_variables = session_stored_simulator.model.__class__.variables_of_interest.element_choices
            chosen_variables = session_stored_simulator.model.variables_of_interest
            indexes = self._get_variables_of_interest_indexes(all_variables, chosen_variables)

            next_form = get_form_for_monitor(type(next_monitor))(indexes, '', common.get_current_project().id)
            next_form.fill_from_trait(next_monitor)

            form_action_url = self.build_monitor_url(SimulatorWizzardURLs.SET_MONITOR_PARAMS_URL,
                                                     type(next_monitor).__name__)
            is_launch_fragment = False

            monitor_name = self._prepare_monitor_legend(session_stored_simulator.is_surface_simulation, next_monitor)
        else:
            default_simulation_name, simulation_number = BurstService.prepare_name(session_stored_burst,
                                                                                   common.get_current_project().id)
            next_form = SimulatorFinalFragment(default_simulation_name=default_simulation_name)
            next_form.fill_from_trait(session_stored_simulator)

            form_action_url = SimulatorWizzardURLs.SETUP_PSE_URL
            is_launch_fragment = True

            if cherrypy.request.method != 'POST':
                simulation_name = common.get_from_session(common.KEY_BURST_CONFIG).name
                if simulation_name is None:
                    simulation_name = 'simulation_' + str(simulation_number)
                next_form.fill_from_post({'input_simulation_name_id': simulation_name,
                                          'simulation_length': str(session_stored_simulator.simulation_length)})

            monitor_name = None

        return next_form, form_action_url, is_launch_fragment, monitor_name

    @staticmethod
    def _get_current_index_and_next_monitor(monitors, current_monitor_name):
        for monitor in monitors:
            if type(monitor).__name__ == current_monitor_name:
                index = monitors.index(monitor)
                if index < len(monitors) - 1:
                    return monitors[index + 1], index

        # Currently at the last monitor
        return None, len(monitors) - 1

    @cherrypy.expose
    @using_template("simulator_fragment")
    @handle_error(redirect=False)
    @check_user
    def set_monitor_params(self, current_monitor, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        session_stored_burst = common.get_from_session(common.KEY_BURST_CONFIG)
        next_monitor, current_monitor_index = self._get_current_index_and_next_monitor(
            session_stored_simulator.monitors, current_monitor)
        monitor = session_stored_simulator.monitors[current_monitor_index]
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY) or False
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD) or False

        if cherrypy.request.method == 'POST':
            is_simulator_copy = False
            chosen_variables = data['variables_of_interest']
            all_variables = session_stored_simulator.model.variables_of_interest
            indexes = self._get_variables_of_interest_indexes(all_variables, chosen_variables)
            form = get_form_for_monitor(type(monitor))(indexes)
            form.fill_from_post(data)
            form.fill_trait(monitor)

            if isinstance(monitor, Bold):
                last_loaded_fragment_url = self.build_monitor_url(SimulatorWizzardURLs.SET_MONITOR_EQUATION_URL,
                                                                  current_monitor)
            elif next_monitor is not None:
                last_loaded_fragment_url = self.build_monitor_url(SimulatorWizzardURLs.SET_MONITOR_PARAMS_URL,
                                                                  type(next_monitor).__name__)
            else:
                last_loaded_fragment_url = SimulatorWizzardURLs.SETUP_PSE_URL
            self._update_last_loaded_fragment_url(last_loaded_fragment_url)
        if isinstance(monitor, Bold):
            next_form = get_form_for_equation(type(monitor.hrf_kernel))()
            next_form.fill_from_trait(monitor.hrf_kernel)

            form_action_url = self.build_monitor_url(SimulatorWizzardURLs.SET_MONITOR_EQUATION_URL, current_monitor)
            previous_form_action_url = self.build_monitor_url(SimulatorWizzardURLs.SET_MONITOR_PARAMS_URL,
                                                              current_monitor)
            rendering_rules = SimulatorFragmentRenderingRules(next_form, form_action_url,
                                                              previous_form_action_url,
                                                              is_simulator_copy, is_simulator_load,
                                                              self.last_loaded_form_url, cherrypy.request.method)
            return rendering_rules.to_dict()

        if isinstance(monitor, Projection) and cherrypy.request.method == 'POST':
            # load region mapping
            region_mapping_index = ABCAdapter.load_entity_by_gid(data['region_mapping'])
            region_mapping = h5.load_from_index(region_mapping_index)
            monitor.region_mapping = region_mapping

            # load sensors and projection
            sensors_index = ABCAdapter.load_entity_by_gid(data['sensors'])
            sensors_class = monitor.projection_class().sensors.field_type
            sensors = h5.load_from_index(sensors_index, dt_class=sensors_class)

            projection_surface_index = ABCAdapter.load_entity_by_gid(data['projection'])
            projection_class = monitor.projection_class()
            projection = h5.load_from_index(projection_surface_index, dt_class=projection_class)

            monitor.sensors = sensors
            monitor.projection = projection

        next_form, form_action_url, is_launch_fragment, monitor_name = \
            self._handle_next_fragment_for_monitors(session_stored_simulator, next_monitor, session_stored_burst)

        previous_form_action_url = self.build_monitor_url(SimulatorWizzardURLs.SET_MONITOR_PARAMS_URL,
                                                          type(monitor).__name__)
        rendering_rules = SimulatorFragmentRenderingRules(next_form, form_action_url,
                                                          previous_form_action_url,
                                                          is_simulator_copy, is_simulator_load,
                                                          self.last_loaded_form_url, cherrypy.request.method,
                                                          is_launch_fragment=is_launch_fragment,
                                                          is_pse_launch=session_stored_burst.is_pse_burst(),
                                                          monitor_name=monitor_name)
        return rendering_rules.to_dict()

    @cherrypy.expose
    @using_template("simulator_fragment")
    @handle_error(redirect=False)
    @check_user
    def set_monitor_equation(self, current_monitor, **data):
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        session_stored_burst = common.get_from_session(common.KEY_BURST_CONFIG)
        next_monitor, current_monitor_index = self._get_current_index_and_next_monitor(
            session_stored_simulator.monitors, current_monitor)
        monitor = session_stored_simulator.monitors[current_monitor_index]
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY) or False
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD) or False

        if cherrypy.request.method == 'POST':
            if next_monitor is not None:
                last_loaded_fragment_url = self.build_monitor_url(SimulatorWizzardURLs.SET_MONITOR_PARAMS_URL,
                                                                  type(next_monitor).__name__)
                self._update_last_loaded_fragment_url(last_loaded_fragment_url)
            else:
                self._update_last_loaded_fragment_url(SimulatorWizzardURLs.SETUP_PSE_URL)

            is_simulator_copy = False
            form = get_form_for_equation(type(monitor.hrf_kernel))()
            form.fill_from_post(data)
            form.fill_trait(monitor.hrf_kernel)

        next_form, form_action_url, is_launch_fragment, monitor_name = \
            self._handle_next_fragment_for_monitors(session_stored_simulator, next_monitor, session_stored_burst)

        previous_form_action_url = self.build_monitor_url(SimulatorWizzardURLs.SET_MONITOR_EQUATION_URL,
                                                          current_monitor)
        rendering_rules = SimulatorFragmentRenderingRules(next_form, form_action_url,
                                                          previous_form_action_url,
                                                          is_simulator_copy, is_simulator_load,
                                                          self.last_loaded_form_url, cherrypy.request.method,
                                                          is_launch_fragment=is_launch_fragment,
                                                          is_pse_launch=session_stored_burst.is_pse_burst(),
                                                          monitor_name=monitor_name)
        return rendering_rules.to_dict()

    @cherrypy.expose
    @using_template("simulator_fragment")
    @handle_error(redirect=False)
    @check_user
    def setup_pse(self, **data):
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY) or False
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD) or False
        next_form = SimulatorPSEConfigurationFragment(self.range_parameters.get_all_range_parameters())

        if cherrypy.request.method == 'POST':
            session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
            session_stored_simulator.simulation_length = float(data['simulation_length'])
            burst_config = common.get_from_session(common.KEY_BURST_CONFIG)
            burst_config.name = data['input_simulation_name_id']
            self._update_last_loaded_fragment_url(SimulatorWizzardURLs.SET_PSE_PARAMS_URL)
            is_simulator_copy = False

        param1, param2 = self._handle_range_params_at_loading()
        if param1:
            param_dict = {'pse_param1': param1.name}
            if param2 is not None:
                param_dict['pse_param2'] = param2.name
            next_form.fill_from_post(param_dict)

        rendering_rules = SimulatorFragmentRenderingRules(next_form, SimulatorWizzardURLs.SET_PSE_PARAMS_URL,
                                                          SimulatorWizzardURLs.SETUP_PSE_URL,
                                                          is_simulator_copy, is_simulator_load,
                                                          self.last_loaded_form_url, cherrypy.request.method)
        return rendering_rules.to_dict()

    def _handle_range_params_at_loading(self):
        burst_config = common.get_from_session(common.KEY_BURST_CONFIG)
        all_range_parameters = self.range_parameters.get_all_range_parameters()
        param1, param2 = None, None
        if burst_config.range1:
            param1 = RangeParameter.from_json(burst_config.range1)
            param1.fill_from_default(all_range_parameters[param1.name])
            if burst_config.range2 is not None:
                param2 = RangeParameter.from_json(burst_config.range2)
                param2.fill_from_default(all_range_parameters[param2.name])

        return param1, param2

    @cherrypy.expose
    @using_template("simulator_fragment")
    @handle_error(redirect=False)
    @check_user
    def set_pse_params(self, **data):
        is_simulator_copy = common.get_from_session(common.KEY_IS_SIMULATOR_COPY) or False
        is_simulator_load = common.get_from_session(common.KEY_IS_SIMULATOR_LOAD) or False
        form = SimulatorPSEConfigurationFragment(self.range_parameters.get_all_range_parameters())
        burst_config = common.get_from_session(common.KEY_BURST_CONFIG)

        if cherrypy.request.method == 'POST':
            is_simulator_copy = False
            self._update_last_loaded_fragment_url(SimulatorWizzardURLs.LAUNCH_PSE_URL)
            form.fill_from_post(data)

            param1 = form.pse_param1.value
            burst_config.range1 = param1.to_json()
            param2 = None
            if form.pse_param2.value:
                param2 = form.pse_param2.value
                burst_config.range2 = param2.to_json()
        else:
            param1, param2 = self._handle_range_params_at_loading()
        project_id = common.get_current_project().id
        next_form = SimulatorPSERangeFragment(param1, param2, project_id=project_id)

        rendering_rules = SimulatorFragmentRenderingRules(next_form, SimulatorWizzardURLs.LAUNCH_PSE_URL,
                                                          SimulatorWizzardURLs.SET_PSE_PARAMS_URL,
                                                          is_simulator_copy, is_simulator_load,
                                                          last_form_url=self.last_loaded_form_url,
                                                          is_launch_pse_fragment=True)
        return rendering_rules.to_dict()

    @expose_json
    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def launch_pse(self, **data):
        all_range_parameters = self.range_parameters.get_all_range_parameters()
        range_param1, range_param2 = SimulatorPSERangeFragment.fill_from_post(all_range_parameters, **data)
        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)

        project = common.get_current_project()
        user = common.get_logged_user()

        burst_config = common.get_from_session(common.KEY_BURST_CONFIG)
        burst_config.start_time = datetime.now()
        burst_config.range1 = range_param1.to_json()
        if range_param2:
            burst_config.range2 = range_param2.to_json()
        burst_config = self.burst_service.prepare_burst_for_pse(burst_config)

        try:
            thread = threading.Thread(target=self.simulator_service.async_launch_and_prepare_pse,
                                      kwargs={'burst_config': burst_config,
                                              'user': user,
                                              'project': project,
                                              'simulator_algo': self.cached_simulator_algorithm,
                                              'range_param1': range_param1,
                                              'range_param2': range_param2,
                                              'session_stored_simulator': session_stored_simulator})
            thread.start()
            return {'id': burst_config.id}
        except BurstServiceException as e:
            self.logger.exception("Could not launch burst!")
            return {'error': e.message}

    @expose_json
    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def launch_simulation(self, launch_mode, **data):
        current_form = SimulatorFinalFragment()
        session_burst_config = common.get_from_session(common.KEY_BURST_CONFIG)
        session_burst_config.range1 = None
        session_burst_config.range2 = None

        try:
            current_form.fill_from_post(data)
        except Exception as exc:
            self.logger.exception(exc)
            return {'error': str(exc)}

        burst_name = current_form.simulation_name.value

        session_stored_simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)
        session_stored_simulator.simulation_length = current_form.simulation_length.value

        project = common.get_current_project()
        user = common.get_logged_user()

        if burst_name != 'none_undefined':
            session_burst_config.name = burst_name

        simulation_state_index_gid = None
        if launch_mode == self.burst_service.LAUNCH_BRANCH:
            parent_burst = session_burst_config.parent_burst_object
            count = dao.count_bursts_with_name(parent_burst.name, session_burst_config.fk_project)
            session_burst_config.name = parent_burst.name + "_" + launch_mode + str(count + 1)
            simulation_state_index = dao.get_generic_entity(SimulationHistoryIndex,
                                                            parent_burst.id, "fk_parent_burst")
            if simulation_state_index is None or len(simulation_state_index) < 1:
                exc = BurstServiceException("Simulation State not found for %s, thus we are unable to branch from "
                                            "it!" % session_burst_config.name)
                self.logger.error(exc)
                raise exc
            simulation_state_index_gid = simulation_state_index[0].gid

        session_burst_config.start_time = datetime.now()
        session_burst_config = dao.store_entity(session_burst_config)

        try:
            thread = threading.Thread(target=self.simulator_service.async_launch_and_prepare_simulation,
                                      kwargs={'burst_config': session_burst_config,
                                              'user': user,
                                              'project': project,
                                              'simulator_algo': self.cached_simulator_algorithm,
                                              'session_stored_simulator': session_stored_simulator,
                                              'simulation_state_gid': simulation_state_index_gid})
            thread.start()
            return {'id': session_burst_config.id}
        except BurstServiceException as e:
            self.logger.exception('Could not launch burst!')
            return {'error': e.message}

    @expose_fragment('burst/burst_history')
    def load_burst_history(self):
        """
        Load the available burst that are stored in the database at this time.
        This is one alternative to 'chrome-back problem'.
        """
        session_burst = common.get_from_session(common.KEY_BURST_CONFIG)
        bursts = self.burst_service.get_available_bursts(common.get_current_project().id)
        self.burst_service.populate_burst_disk_usage(bursts)
        return {'burst_list': bursts,
                'selectedBurst': session_burst.id,
                'first_fragment_url': SimulatorFragmentRenderingRules.FIRST_FORM_URL}

    def _prepare_last_fragment_by_burst_type(self, burst_config):
        if burst_config.is_pse_burst():
            return SimulatorWizzardURLs.LAUNCH_PSE_URL
        else:
            return SimulatorWizzardURLs.SETUP_PSE_URL

    @cherrypy.expose
    def get_last_fragment_url(self, burst_config_id):
        burst_config = self.burst_service.load_burst_configuration(burst_config_id)
        common.add2session(common.KEY_BURST_CONFIG, burst_config)
        return self._prepare_last_fragment_by_burst_type(burst_config)

    @cherrypy.expose
    @using_template("simulator_fragment")
    @handle_error(redirect=False)
    @check_user
    def load_burst_read_only(self, burst_config_id):
        try:
            burst_config = self.burst_service.load_burst_configuration(burst_config_id)
            common.add2session(common.KEY_BURST_CONFIG, burst_config)
            project = common.get_current_project()
            storage_path = self.files_helper.get_project_folder(project, str(burst_config.fk_simulation))
            simulator = SimulatorSerializer().deserialize_simulator(burst_config.simulator_gid, storage_path)

            common.add2session(common.KEY_SIMULATOR_CONFIG, simulator)
            common.add2session(common.KEY_IS_SIMULATOR_LOAD, True)
            common.add2session(common.KEY_IS_SIMULATOR_COPY, False)

            self._update_last_loaded_fragment_url(self._prepare_last_fragment_by_burst_type(burst_config))
            form = self.prepare_first_fragment()
            rendering_rules = SimulatorFragmentRenderingRules(form, SimulatorWizzardURLs.SET_CONNECTIVITY_URL,
                                                              is_simulation_readonly_load=True, is_first_fragment=True)
            return rendering_rules.to_dict()
        except Exception:
            # Most probably Burst was removed. Delete it from session, so that client
            # has a good chance to get a good response on refresh
            self.logger.exception("Error loading burst")
            common.remove_from_session(common.KEY_BURST_CONFIG)
            raise

    @cherrypy.expose
    @using_template("simulator_fragment")
    @handle_error(redirect=False)
    @check_user
    def copy_simulator_configuration(self, burst_config_id):
        burst_config = self.burst_service.load_burst_configuration(burst_config_id)
        burst_config_copy = burst_config.clone()
        burst_config_copy.name = self.DEFAULT_COPY_PREFIX + burst_config.name

        project = common.get_current_project()
        storage_path = self.files_helper.get_project_folder(project, str(burst_config.fk_simulation))
        simulator = SimulatorSerializer().deserialize_simulator(burst_config.simulator_gid, storage_path)

        common.add2session(common.KEY_SIMULATOR_CONFIG, simulator)
        common.add2session(common.KEY_IS_SIMULATOR_COPY, True)
        common.add2session(common.KEY_IS_SIMULATOR_LOAD, False)
        common.add2session(common.KEY_BURST_CONFIG, burst_config_copy)

        self._update_last_loaded_fragment_url(self._prepare_last_fragment_by_burst_type(burst_config_copy))
        form = self.prepare_first_fragment()
        rendering_rules = SimulatorFragmentRenderingRules(form, SimulatorWizzardURLs.SET_CONNECTIVITY_URL,
                                                          is_simulation_copy=True, is_simulation_readonly_load=True,
                                                          is_first_fragment=True)
        return rendering_rules.to_dict()

    @cherrypy.expose
    @using_template("simulator_fragment")
    @handle_error(redirect=False)
    @check_user
    def reset_simulator_configuration(self):
        common.add2session(common.KEY_SIMULATOR_CONFIG, None)
        common.add2session(common.KEY_IS_SIMULATOR_COPY, False)
        common.add2session(common.KEY_IS_SIMULATOR_LOAD, False)

        self._update_last_loaded_fragment_url(SimulatorWizzardURLs.SET_CONNECTIVITY_URL)
        project = common.get_current_project()
        common.add2session(common.KEY_BURST_CONFIG, BurstConfiguration(project.id))

        form = self.prepare_first_fragment()
        rendering_rules = SimulatorFragmentRenderingRules(form, SimulatorWizzardURLs.SET_CONNECTIVITY_URL,
                                                          is_first_fragment=True)
        return rendering_rules.to_dict()

    @expose_json
    def rename_burst(self, burst_id, burst_name):
        """
        Rename the burst given by burst_id, setting it's new name to
        burst_name.
        """
        validation_result = SimulatorFinalFragment.is_burst_name_ok(burst_name)
        if validation_result is True:
            self.burst_service.rename_burst(burst_id, burst_name)
            return {'success': "Simulation successfully renamed!"}
        else:
            self.logger.exception(validation_result)
            return {'error': validation_result}

    @expose_json
    def get_history_status(self, **data):
        """
        For each burst id received, get the status and return it.
        """
        return self.burst_service.update_history_status(json.loads(data['burst_ids']))

    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def export(self, burst_id):
        export_manager = ExportManager()
        export_zip = export_manager.export_simulator_configuration(burst_id)

        result_name = "tvb_simulation_" + str(burst_id) + ".zip"
        return serve_file(export_zip, "application/x-download", "attachment", result_name)

    @expose_fragment("overlay")
    def get_upload_overlay(self):
        template_specification = self.fill_overlay_attributes(None, "Upload", "Simulation ZIP",
                                                              "burst/upload_burst_overlay", "dialog-upload")
        template_specification['first_fragment_url'] = SimulatorWizzardURLs.SET_CONNECTIVITY_URL
        return self.fill_default_attributes(template_specification)

    @cherrypy.expose
    @handle_error(redirect=True)
    @check_user
    @settings
    def load_simulator_configuration_from_zip(self, **data):
        """Upload Simulator from previously exported ZIP file"""
        self.logger.debug("Uploading ..." + str(data))

        try:
            upload_param = "uploadedfile"
            if upload_param in data and data[upload_param]:
                project = common.get_current_project()
                simulator, burst_config = self.simulator_service.load_from_zip(data[upload_param], project)

                common.add2session(common.KEY_BURST_CONFIG, burst_config)
                common.add2session(common.KEY_SIMULATOR_CONFIG, simulator)
                common.add2session(common.KEY_IS_SIMULATOR_COPY, True)
                common.add2session(common.KEY_IS_SIMULATOR_LOAD, False)
                if burst_config.is_pse_burst():
                    self._update_last_loaded_fragment_url(SimulatorWizzardURLs.LAUNCH_PSE_URL)
                else:
                    self._update_last_loaded_fragment_url(SimulatorWizzardURLs.SETUP_PSE_URL)
        except IOError as ioexcep:
            self.logger.exception(ioexcep)
            common.set_warning_message("This ZIP does not contain a complete simulator configuration")
        except ServicesBaseException as excep:
            self.logger.warning(excep.message)
            common.set_warning_message(excep.message)

        raise cherrypy.HTTPRedirect('/burst/')
