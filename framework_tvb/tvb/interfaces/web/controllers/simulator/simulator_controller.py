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
from tvb.adapters.exporters.export_manager import ExportManager
from tvb.adapters.simulator.coupling_forms import get_form_for_coupling
from tvb.adapters.simulator.equation_forms import get_form_for_equation
from tvb.adapters.simulator.model_forms import get_form_for_model
from tvb.adapters.simulator.monitor_forms import get_form_for_monitor
from tvb.adapters.simulator.noise_forms import get_form_for_noise
from tvb.adapters.simulator.range_parameters import SimulatorRangeParameters
from tvb.adapters.simulator.simulator_adapter import SimulatorAdapterForm
from tvb.adapters.simulator.simulator_fragments import *
from tvb.config.init.introspector_registry import IntrospectionRegistry
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.file.simulator.view_model import AdditiveNoiseViewModel, BoldViewModel, RawViewModel
from tvb.core.entities.file.simulator.view_model import IntegratorStochasticViewModel
from tvb.core.neocom import h5
from tvb.core.services.burst_service import BurstService
from tvb.core.services.exceptions import BurstServiceException, ServicesBaseException
from tvb.core.services.operation_service import OperationService
from tvb.core.services.simulator_service import SimulatorService
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.burst.base_controller import BurstBaseController
from tvb.interfaces.web.controllers.decorators import *
from tvb.interfaces.web.controllers.simulator.simulator_fragment_rendering_rules import \
    SimulatorFragmentRenderingRules, POST_REQUEST
from tvb.interfaces.web.controllers.simulator.simulator_wizzard_urls import SimulatorWizzardURLs
from tvb.interfaces.web.entities.context_simulator import SimulatorContext


@traced
class SimulatorController(BurstBaseController):
    KEY_IS_LOAD_AFTER_REDIRECT = "is_load_after_redirect"
    COPY_NAME_FORMAT = "copy_of_{}"
    BRANCH_NAME_FORMAT = "{}_branch{}"

    def __init__(self):
        BurstBaseController.__init__(self)
        self.range_parameters = SimulatorRangeParameters()
        self.burst_service = BurstService()
        self.simulator_service = SimulatorService()
        self.files_helper = FilesHelper()
        self.cached_simulator_algorithm = self.algorithm_service.get_algorithm_by_module_and_class(
            IntrospectionRegistry.SIMULATOR_MODULE, IntrospectionRegistry.SIMULATOR_CLASS)
        self.context = SimulatorContext()

    @expose_json
    def cancel_or_remove_burst(self, burst_id):
        """
        Cancel or Remove the burst entity given by burst_id (and all linked entities: op, DTs)
        :returns True: if the op was successfully.
        """
        burst_id = int(burst_id)
        op_id, is_group, remove_after_stop = self.burst_service.cancel_or_remove_burst(burst_id)
        return self.cancel_or_remove_operation(op_id, is_group, remove_after_stop)

    def cancel_or_remove_operation(self, operation_id, is_group, remove_after_stop=False):
        """
        Stop the operation given by operation_id. If is_group is true stop all the
        operations from that group.
        """
        # Load before we remove, to have its data in memory here
        burst_config = BurstService.get_burst_for_operation_id(operation_id, is_group)
        result = OperationService.stop_operation(operation_id, is_group, remove_after_stop)

        if remove_after_stop:
            current_burst = self.context.burst_config
            if (current_burst is not None and burst_config is not None and current_burst.id == burst_config.id and
                    ((current_burst.fk_simulation == operation_id and not is_group) or
                     (current_burst.fk_operation_group == operation_id and is_group))):
                self.reset_simulator_configuration()
        return result

    @expose_page
    @settings
    @context_selected
    def index(self):
        """Get on burst main page"""
        template_specification = dict(mainContent="burst/main_burst", title="Simulation Cockpit",
                                      includedResources='project/included_resources')
        self.context.get_current_project()
        self.context.get_session_params()

        if not self.context.last_loaded_form_url:
            self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_CONNECTIVITY_URL)

        self.context.add_burst_config_to_session()
        if self.context.burst_config.start_time is not None:
            self.context.add_simulator_load_to_session(True)

        template_specification['burstConfig'] = self.context.burst_config
        template_specification['burst_list'] = self.burst_service.get_available_bursts(self.context.project.id)
        portlets_list = []  # self.burst_service.get_available_portlets()
        template_specification['portletList'] = portlets_list
        template_specification['selectedPortlets'] = json.dumps(portlets_list)

        form = self.prepare_first_fragment()
        rendering_rules = SimulatorFragmentRenderingRules(self.context, form, SimulatorWizzardURLs.SET_CONNECTIVITY_URL,
                                                          None, last_request_type=cherrypy.request.method,
                                                          is_first_fragment=True)
        template_specification.update(**rendering_rules.to_dict())
        return self.fill_default_attributes(template_specification)

    def prepare_first_fragment(self):
        form = self.simulator_service.prepare_first_simulation_fragment(self.cached_simulator_algorithm,
                                                                        self.context.project.id)
        self.filter_connectivity(form)
        self.context.init_session_stored_simulator()
        form.fill_from_trait(self.context.session_stored_simulator)
        return form

    def filter_connectivity(self, form):
        if self.context.is_branch:
            self.simulator_service.filter_connectivity(form, self.context.session_stored_simulator.connectivity.hex)

    @expose_fragment('simulator_fragment')
    def set_connectivity(self, **data):
        if cherrypy.request.method == POST_REQUEST:
            self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_COUPLING_PARAMS_URL)
            form = SimulatorAdapterForm()
            form.fill_from_post(data)
            self.simulator_service.reset_at_connectivity_change(self.context.is_simulator_copy,
                                                                form, self.context.session_stored_simulator)
            form.fill_trait(self.context.session_stored_simulator)

        next_form = self.algorithm_service.prepare_adapter_form(
            form_instance=get_form_for_coupling(type(self.context.session_stored_simulator.coupling))())
        self.range_parameters.coupling_parameters = next_form.get_range_parameters()
        next_form.fill_from_trait(self.context.session_stored_simulator.coupling)

        rendering_rules = SimulatorFragmentRenderingRules(
            self.context, next_form, SimulatorWizzardURLs.SET_COUPLING_PARAMS_URL,
            SimulatorWizzardURLs.SET_CONNECTIVITY_URL, cherrypy.request.method)
        return rendering_rules.to_dict()

    @expose_fragment('simulator_fragment')
    def set_coupling_params(self, **data):
        if cherrypy.request.method == POST_REQUEST:
            self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_SURFACE_URL)
            form = get_form_for_coupling(type(self.context.session_stored_simulator.coupling))()
            form.fill_from_post(data)
            form.fill_trait(self.context.session_stored_simulator.coupling)

        surface_fragment = self.algorithm_service.prepare_adapter_form(form_instance=SimulatorSurfaceFragment(),
                                                                       project_id=common.get_current_project().id)
        surface_fragment.fill_from_trait(self.context.session_stored_simulator.surface)

        rendering_rules = SimulatorFragmentRenderingRules(
            self.context, surface_fragment, SimulatorWizzardURLs.SET_SURFACE_URL,
            SimulatorWizzardURLs.SET_COUPLING_PARAMS_URL, cherrypy.request.method)
        return rendering_rules.to_dict()

    @expose_fragment('simulator_fragment')
    def set_surface(self, **data):
        rendering_rules = SimulatorFragmentRenderingRules(self.context,
                                                          previous_form_action_url=SimulatorWizzardURLs.SET_SURFACE_URL,
                                                          last_request_type=cherrypy.request.method)
        if cherrypy.request.method == POST_REQUEST:
            form = SimulatorSurfaceFragment()
            form.fill_from_post(data)
            self.simulator_service.reset_at_surface_change(self.context.is_simulator_copy, form,
                                                           self.context.session_stored_simulator)
            form.fill_trait(self.context.session_stored_simulator)

            if self.context.session_stored_simulator.surface:
                self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_CORTEX_URL)
            else:
                self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_STIMULUS_URL)

        rendering_rules.is_simulation_copy = self.context.is_simulator_copy
        return self.simulator_service.prepare_next_fragment_if_surface(self.context, rendering_rules)

    @expose_fragment('simulator_fragment')
    def set_cortex(self, **data):
        if cherrypy.request.method == POST_REQUEST:
            self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_STIMULUS_URL)
            rm_fragment = SimulatorRMFragment()
            rm_fragment.fill_from_post(data)
            rm_fragment.fill_trait(self.context.session_stored_simulator.surface)

        rendering_rules = SimulatorFragmentRenderingRules(self.context, None, None, SimulatorWizzardURLs.SET_CORTEX_URL,
                                                          cherrypy.request.method)
        return self.simulator_service.prepare_stimulus_fragment(self.context.session_stored_simulator, rendering_rules,
                                                                True, self.context.project.id,
                                                                SimulatorWizzardURLs.SET_STIMULUS_URL)

    @expose_fragment('simulator_fragment')
    def set_stimulus(self, **data):
        if cherrypy.request.method == POST_REQUEST:
            self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_MODEL_URL)
            stimuli_fragment = SimulatorStimulusFragment(self.context.session_stored_simulator.is_surface_simulation)
            stimuli_fragment.fill_from_post(data)
            stimuli_fragment.fill_trait(self.context.session_stored_simulator)

        model_fragment = self.algorithm_service.prepare_adapter_form(form_instance=SimulatorModelFragment())

        model_fragment.fill_from_trait(self.context.session_stored_simulator)

        rendering_rules = SimulatorFragmentRenderingRules(
            self.context, model_fragment, SimulatorWizzardURLs.SET_MODEL_URL, SimulatorWizzardURLs.SET_STIMULUS_URL,
            cherrypy.request.method, is_model_fragment=True, is_surface_simulation=
            self.context.session_stored_simulator.is_surface_simulation)
        return rendering_rules.to_dict()

    @expose_fragment('simulator_fragment')
    def set_model(self, **data):
        if cherrypy.request.method == POST_REQUEST:
            self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_MODEL_PARAMS_URL)
            form = SimulatorModelFragment()
            form.fill_from_post(data)
            form.fill_trait(self.context.session_stored_simulator)

        form = self.algorithm_service.prepare_adapter_form(
            form_instance=get_form_for_model(type(self.context.session_stored_simulator.model))())
        self.range_parameters.model_parameters = form.get_range_parameters()
        form.fill_from_trait(self.context.session_stored_simulator.model)

        rendering_rules = SimulatorFragmentRenderingRules(
            self.context, form, SimulatorWizzardURLs.SET_MODEL_PARAMS_URL, SimulatorWizzardURLs.SET_MODEL_URL,
            cherrypy.request.method)
        return rendering_rules.to_dict()

    @expose_fragment('simulator_fragment')
    def set_model_params(self, **data):
        if cherrypy.request.method == POST_REQUEST:
            self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_INTEGRATOR_URL)
            form = get_form_for_model(type(self.context.session_stored_simulator.model))()
            form.fill_from_post(data)
            form.fill_trait(self.context.session_stored_simulator.model)

        integrator_fragment = self.algorithm_service.prepare_adapter_form(form_instance=SimulatorIntegratorFragment())
        integrator_fragment.integrator.display_subform = False
        integrator_fragment.fill_from_trait(self.context.session_stored_simulator)
        rendering_rules = SimulatorFragmentRenderingRules(
            self.context, integrator_fragment, SimulatorWizzardURLs.SET_INTEGRATOR_URL,
            SimulatorWizzardURLs.SET_MODEL_PARAMS_URL, cherrypy.request.method)
        return rendering_rules.to_dict()

    @expose_fragment('simulator_fragment')
    def set_integrator(self, **data):
        if cherrypy.request.method == POST_REQUEST:
            self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_INTEGRATOR_PARAMS_URL)
            fragment = SimulatorIntegratorFragment()
            fragment.fill_from_post(data)
            fragment.fill_trait(self.context.session_stored_simulator)

        form = get_form_for_integrator(type(self.context.session_stored_simulator.integrator))()
        form.fill_from_trait(self.context.session_stored_simulator.integrator)

        rendering_rules = SimulatorFragmentRenderingRules(
            self.context, form, SimulatorWizzardURLs.SET_INTEGRATOR_PARAMS_URL, SimulatorWizzardURLs.SET_INTEGRATOR_URL,
            cherrypy.request.method)
        return rendering_rules.to_dict()

    @expose_fragment('simulator_fragment')
    def set_integrator_params(self, **data):
        if cherrypy.request.method == POST_REQUEST:
            form = get_form_for_integrator(type(self.context.session_stored_simulator.integrator))()
            form.fill_from_post(data)
            form.fill_trait(self.context.session_stored_simulator.integrator)
            if isinstance(self.context.session_stored_simulator.integrator, IntegratorStochasticViewModel):
                self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_NOISE_PARAMS_URL)
            else:
                self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_MONITORS_URL)

        rendering_rules = SimulatorFragmentRenderingRules(self.context, None, None,
                                                          SimulatorWizzardURLs.SET_INTEGRATOR_PARAMS_URL,
                                                          cherrypy.request.method, is_noise_fragment=False)

        if not isinstance(self.context.session_stored_simulator.integrator, IntegratorStochasticViewModel):
            return self.simulator_service.prepare_monitor_form(self.context, rendering_rules,
                                                               SimulatorWizzardURLs.SET_MONITORS_URL)

        integrator_noise_fragment = get_form_for_noise(type(self.context.session_stored_simulator.integrator.noise))()

        self.range_parameters.integrator_noise_parameters = integrator_noise_fragment.get_range_parameters()
        integrator_noise_fragment.fill_from_trait(self.context.session_stored_simulator.integrator.noise)

        rendering_rules.form = integrator_noise_fragment
        rendering_rules.form_action_url = SimulatorWizzardURLs.SET_NOISE_PARAMS_URL
        rendering_rules.is_noise_fragment = True
        return rendering_rules.to_dict()

    @expose_fragment('simulator_fragment')
    def set_noise_params(self, **data):
        if cherrypy.request.method == POST_REQUEST:
            form = get_form_for_noise(type(self.context.session_stored_simulator.integrator.noise))()
            form.fill_from_post(data)
            form.fill_trait(self.context.session_stored_simulator.integrator.noise)
            if isinstance(self.context.session_stored_simulator.integrator.noise, AdditiveNoiseViewModel):
                self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_MONITORS_URL)
            else:
                self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_NOISE_EQUATION_PARAMS_URL)

        rendering_rules = SimulatorFragmentRenderingRules(
            self.context, None, None, SimulatorWizzardURLs.SET_NOISE_PARAMS_URL, cherrypy.request.method)

        return self.simulator_service.prepare_next_fragment_if_noise(self.context, rendering_rules)

    @expose_fragment('simulator_fragment')
    def set_noise_equation_params(self, **data):
        if cherrypy.request.method == POST_REQUEST:
            self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_MONITORS_URL)
            form = get_form_for_equation(type(self.context.session_stored_simulator.integrator.noise.b))()
            form.fill_from_post(data)
            form.fill_trait(session_stored_simulator.integrator.noise.b)

        rendering_rules = SimulatorFragmentRenderingRules(
            self.context, None, None, SimulatorWizzardURLs.SET_NOISE_EQUATION_PARAMS_URL, cherrypy.request.method)

        return self.simulator_service.prepare_monitor_form(self.context, rendering_rules,
                                                           SimulatorWizzardURLs.SET_MONITORS_URL)

    def _prepare_final_fragment(self, session_stored_simulator, rendering_rules):
        session_stored_burst = common.get_from_session(common.KEY_BURST_CONFIG)
        default_simulation_name, simulation_number = BurstService.prepare_name(session_stored_burst,
                                                                               common.get_current_project().id)
        form = self.algorithm_service.prepare_adapter_form(
            form_instance=SimulatorFinalFragment(default_simulation_name=default_simulation_name))

        if cherrypy.request.method != POST_REQUEST:
            simulation_name = session_stored_burst.name
            if simulation_name is None:
                simulation_name = 'simulation_' + str(simulation_number)
            form.fill_from_post({'input_simulation_name_id': simulation_name,
                                 'simulation_length': str(session_stored_simulator.simulation_length)})

        form.fill_from_trait(session_stored_simulator)

        rendering_rules.form = form
        rendering_rules.form_action_url = SimulatorWizzardURLs.SETUP_PSE_URL
        rendering_rules.is_launch_fragment = True
        rendering_rules.is_branch = self.context.is_branch
        rendering_rules.is_pse_launch = session_stored_burst.is_pse_burst()
        return rendering_rules.to_dict()

    @expose_fragment('simulator_fragment')
    def set_monitors(self, **data):
        if cherrypy.request.method == POST_REQUEST:
            fragment = SimulatorMonitorFragment(is_surface_simulation=
                                                self.context.session_stored_simulator.is_surface_simulation)
            fragment.fill_from_post(data)

            self.context.session_stored_simulator.monitors = self.simulator_service.build_list_of_monitors(
                fragment.monitors.value, self.context.session_stored_simulator)

        first_monitor_index, last_loaded_fragment_url = self.simulator_service.skip_raw_monitor(
            self.context.session_stored_simulator.monitors, SimulatorWizzardURLs.SETUP_PSE_URL,
            SimulatorWizzardURLs.SET_MONITOR_PARAMS_URL)

        if cherrypy.request.method == POST_REQUEST:
            self.context.add_last_loaded_form_url_to_session(last_loaded_fragment_url)

        monitor = self.context.session_stored_simulator.monitors[first_monitor_index]
        form = self.algorithm_service.prepare_adapter_form(
            form_instance=get_form_for_monitor(type(monitor))(self.context.session_stored_simulator),
            project_id=common.get_current_project().id)
        form.fill_from_trait(monitor)

        rendering_rules = SimulatorFragmentRenderingRules(self.context, form, SimulatorWizzardURLs.SET_MONITORS_URL,
                                                          cherrypy.request.method)

        if isinstance(monitor, RawViewModel) and len(self.context.session_stored_simulator.monitors) == 1:
            return self._prepare_final_fragment(self.context.session_stored_simulator, rendering_rules)

        monitor_name = self.simulator_service.prepare_monitor_legend(
            self.context.session_stored_simulator.is_surface_simulation, monitor)

        rendering_rules.form = form
        rendering_rules.form_action_url = last_loaded_fragment_url
        rendering_rules.monitor_name = monitor_name
        return rendering_rules.to_dict()

    def _handle_next_fragment_for_monitors(self, context, next_monitor, rendering_rules):
        if not next_monitor:
            return self._prepare_final_fragment(context.session_stored_simulator, rendering_rules)
        return self.simulator_service.handle_next_fragment_for_monitors(context, rendering_rules, next_monitor)

    @expose_fragment('simulator_fragment')
    def set_monitor_params(self, current_monitor, **data):
        next_monitor, current_monitor_index = self.simulator_service.get_current_index_and_next_monitor(
            self.context.session_stored_simulator.monitors, current_monitor)
        monitor = self.context.session_stored_simulator.monitors[current_monitor_index]

        if cherrypy.request.method == POST_REQUEST:
            form = get_form_for_monitor(type(monitor))(self.context.session_stored_simulator)
            form.fill_from_post(data)
            form.fill_trait(monitor)

            last_loaded_fragment_url = self.simulator_service.prepare_final_monitor_fragment(monitor, current_monitor,
                                                                                             next_monitor)
            self.context.add_last_loaded_form_url_to_session(last_loaded_fragment_url)

        rendering_rules = SimulatorFragmentRenderingRules(self.context, cherrypy.request.method)

        if isinstance(monitor, BoldViewModel):
            return self.simulator_service.prepare_next_fragment_if_bold(monitor, rendering_rules, current_monitor)

        rendering_rules.previous_form_action_url = self.simulator_service.build_monitor_url(
            SimulatorWizzardURLs.SET_MONITOR_PARAMS_URL, type(monitor).__name__)
        return self._handle_next_fragment_for_monitors(self.context, next_monitor, rendering_rules)

    @expose_fragment('simulator_fragment')
    def set_monitor_equation(self, current_monitor, **data):
        next_monitor, current_monitor_index = self.simulator_service.get_current_index_and_next_monitor(
            self.context.session_stored_simulator.monitors, current_monitor)
        monitor = self.context.session_stored_simulator.monitors[current_monitor_index]

        if cherrypy.request.method == POST_REQUEST:
            if next_monitor is not None:
                last_loaded_fragment_url = self.simulator_service.build_monitor_url(
                    SimulatorWizzardURLs.SET_MONITOR_PARAMS_URL, type(next_monitor).__name__)
                self.context.add_last_loaded_form_url_to_session(last_loaded_fragment_url)
            else:
                self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SETUP_PSE_URL)

            form = get_form_for_equation(type(monitor.hrf_kernel))()
            form.fill_from_post(data)
            form.fill_trait(monitor.hrf_kernel)

        previous_form_action_url = self.simulator_service.build_monitor_url(
            SimulatorWizzardURLs.SET_MONITOR_EQUATION_URL, current_monitor)
        rendering_rules = SimulatorFragmentRenderingRules(self.context, None, None, previous_form_action_url,
                                                          cherrypy.request.method)

        return self._handle_next_fragment_for_monitors(self.context, next_monitor, rendering_rules)

    @expose_fragment('simulator_fragment')
    def setup_pse(self, **data):
        next_form = self.algorithm_service.prepare_adapter_form(
            form_instance=SimulatorPSEConfigurationFragment(self.range_parameters.get_all_range_parameters()))

        if cherrypy.request.method == POST_REQUEST:
            self.context.session_stored_simulator.simulation_length = float(data['simulation_length'])
            self.context.burst_config.name = data['input_simulation_name_id']
            self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_PSE_PARAMS_URL)

        param1, param2 = self._handle_range_params_at_loading()
        if param1:
            param_dict = {'pse_param1': param1.name}
            if param2 is not None:
                param_dict['pse_param2'] = param2.name
            next_form.fill_from_post(param_dict)

        rendering_rules = SimulatorFragmentRenderingRules(self.context, next_form,
                                                          SimulatorWizzardURLs.SET_PSE_PARAMS_URL,
                                                          SimulatorWizzardURLs.SETUP_PSE_URL, cherrypy.request.method)
        return rendering_rules.to_dict()

    def _handle_range_params_at_loading(self):
        burst_config = common.get_from_session(common.KEY_BURST_CONFIG)
        all_range_parameters = self.range_parameters.get_all_range_parameters()

        param1, param2 = self.burst_service.handle_range_params_at_loading(self.context.burst_config,
                                                                           all_range_parameters)
        return param1, param2

    @expose_fragment('simulator_fragment')
    def set_pse_params(self, **data):
        form = SimulatorPSEConfigurationFragment(self.range_parameters.get_all_range_parameters())

        if cherrypy.request.method == POST_REQUEST:
            self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.LAUNCH_PSE_URL)
            form.fill_from_post(data)

            param1 = form.pse_param1.value
            self.context.burst_config.range1 = param1.to_json()
            param2 = None
            if form.pse_param2.value != form.pse_param2.missing_value:
                param2 = form.pse_param2.value
                self.context.burst_config.range2 = param2.to_json()
        else:
            param1, param2 = self._handle_range_params_at_loading()
        next_form = self.algorithm_service.prepare_adapter_form(form_instance=SimulatorPSERangeFragment(param1, param2))

        rendering_rules = SimulatorFragmentRenderingRules(
            self.context, next_form, SimulatorWizzardURLs.LAUNCH_PSE_URL, SimulatorWizzardURLs.SET_PSE_PARAMS_URL,
            is_launch_pse_fragment=True)
        return rendering_rules.to_dict()

    @expose_json
    def launch_pse(self, **data):
        all_range_parameters = self.range_parameters.get_all_range_parameters()
        range_param1, range_param2 = SimulatorPSERangeFragment.fill_from_post(all_range_parameters, **data)

        self.context.burst_config.start_time = datetime.now()
        self.context.burst_config.range1 = range_param1.to_json()
        if range_param2:
            self.context.burst_config.range2 = range_param2.to_json()
        burst_config = self.burst_service.prepare_burst_for_pse(self.context.burst_config)
        self.context.session_stored_simulator.operation_group_gid = uuid.UUID(burst_config.operation_group.gid)
        self.context.session_stored_simulator.ranges = json.dumps(burst_config.ranges)

        try:
            thread = threading.Thread(target=self.simulator_service.async_launch_and_prepare_pse,
                                      kwargs={'burst_config': burst_config,
                                              'user': self.context.get_logged_user(),
                                              'project': self.context.project,
                                              'simulator_algo': self.cached_simulator_algorithm,
                                              'range_param1': range_param1,
                                              'range_param2': range_param2,
                                              'session_stored_simulator': self.context.session_stored_simulator})
            thread.start()
            return {'id': burst_config.id}
        except BurstServiceException as e:
            self.logger.exception("Could not launch burst!")
            return {'error': e.message}

    @expose_json
    def launch_simulation(self, launch_mode, **data):
        current_form = SimulatorFinalFragment()
        self.context.burst_config.range1 = None
        self.context.burst_config.range2 = None

        try:
            current_form.fill_from_post(data)
        except Exception as exc:
            self.logger.exception(exc)
            return {'error': str(exc)}

        burst_name = current_form.simulation_name.value
        self.context.session_stored_simulator.simulation_length = current_form.simulation_length.value

        if burst_name != 'none_undefined':
            self.context.burst_config.name = burst_name

        if launch_mode == self.burst_service.LAUNCH_BRANCH:
            simulation_state_index = self.simulator_service.get_simulation_state_index(self.context.burst_config)
            self.context.session_stored_simulator.history_gid = simulation_state_index[0].gid

        self.context.burst_config.start_time = datetime.now()
        session_burst_config = self.burst_service.store_burst(self.context.burst_config)

        try:
            thread = threading.Thread(target=self.simulator_service.async_launch_and_prepare_simulation,
                                      kwargs={'burst_config': session_burst_config,
                                              'user': self.context.get_logged_user(),
                                              'project': self.context.project,
                                              'simulator_algo': self.cached_simulator_algorithm,
                                              'session_stored_simulator': self.context.session_stored_simulator})
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
        bursts = self.burst_service.get_available_bursts(self.context.project.id)
        self.burst_service.populate_burst_disk_usage(bursts)
        return {'burst_list': bursts,
                'selectedBurst': self.context.burst_config.id,
                'first_fragment_url': SimulatorFragmentRenderingRules.FIRST_FORM_URL}

    @staticmethod
    def _prepare_last_fragment_by_burst_type(burst_config):
        if burst_config.is_pse_burst():
            return SimulatorWizzardURLs.LAUNCH_PSE_URL
        else:
            return SimulatorWizzardURLs.SETUP_PSE_URL

    @cherrypy.expose
    def get_last_fragment_url(self, burst_config_id):
        burst_config = self.burst_service.load_burst_configuration(burst_config_id)
        self.context.add_burst_config_to_session(burst_config)
        return self._prepare_last_fragment_by_burst_type(burst_config)

    @expose_fragment('simulator_fragment')
    def load_burst_read_only(self, burst_config_id):
        try:
            burst_config = self.burst_service.load_burst_configuration(burst_config_id)
            self.context.add_burst_config_to_session(burst_config)
            storage_path = self.files_helper.get_project_folder(self.context.project, str(burst_config.fk_simulation))
            simulator = h5.load_view_model(burst_config.simulator_gid, storage_path)
            self.context.init_session_at_burst_loading(simulator)

            self.context.add_last_loaded_form_url_to_session(self._prepare_last_fragment_by_burst_type(burst_config))
            form = self.prepare_first_fragment()
            rendering_rules = SimulatorFragmentRenderingRules(
                self.context, form, SimulatorWizzardURLs.SET_CONNECTIVITY_URL, is_first_fragment=True)
            return rendering_rules.to_dict()
        except Exception:
            # Most probably Burst was removed. Delete it from session, so that client
            # has a good chance to get a good response on refresh
            self.logger.exception("Error loading burst")
            self.context.remove_burst_config_from_session()
            raise

    @expose_fragment('simulator_fragment')
    def copy_simulator_configuration(self, burst_config_id):
        self.context.add_branch_and_copy_to_session(False, True)
        form = self._prepare_first_fragment_for_burst_copy(burst_config_id, self.COPY_NAME_FORMAT)
        rendering_rules = SimulatorFragmentRenderingRules(
            self.context, form, SimulatorWizzardURLs.SET_CONNECTIVITY_URL, is_first_fragment=True)
        return rendering_rules.to_dict()

    @expose_fragment('simulator_fragment')
    def branch_simulator_configuration(self, burst_config_id):
        self.context.add_branch_and_copy_to_session(True, False)
        form = self._prepare_first_fragment_for_burst_copy(burst_config_id, self.BRANCH_NAME_FORMAT)
        rendering_rules = SimulatorFragmentRenderingRules(self.context, form, SimulatorWizzardURLs.SET_CONNECTIVITY_URL,
                                                          is_first_fragment=True)

        return rendering_rules.to_dict()

    def _prepare_first_fragment_for_burst_copy(self, burst_config_id, burst_name_format):
        simulator, burst_config_copy = self.burst_service.prepare_first_fragment_for_burst_copy(
            burst_config_id, burst_name_format, self.context.project)

        self.context.init_session_at_copy_preparation(simulator, burst_config_copy)
        self.context.add_last_loaded_form_url_to_session(self._prepare_last_fragment_by_burst_type(burst_config_copy))
        return self.prepare_first_fragment()

    @expose_fragment('simulator_fragment')
    def reset_simulator_configuration(self):
        self.context.init_session_at_sim_reset()
        self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_CONNECTIVITY_URL)
        self.context.add_burst_config_to_session()

        form = self.prepare_first_fragment()
        rendering_rules = SimulatorFragmentRenderingRules(self.context, form, SimulatorWizzardURLs.SET_CONNECTIVITY_URL,
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
                simulator, burst_config = self.simulator_service.load_from_zip(data[upload_param], self.context.project)

                self.context.init_session_at_sim_config_from_zip(simulator, burst_config)
                if burst_config.is_pse_burst():
                    self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.LAUNCH_PSE_URL)
                else:
                    self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SETUP_PSE_URL)
        except IOError as ioexcep:
            self.logger.exception(ioexcep)
            self.context.set_warning_message("This ZIP does not contain a complete simulator configuration")
        except ServicesBaseException as excep:
            self.logger.warning(excep.message)
            self.context.set_warning_message(excep.message)

        raise cherrypy.HTTPRedirect('/burst/')
