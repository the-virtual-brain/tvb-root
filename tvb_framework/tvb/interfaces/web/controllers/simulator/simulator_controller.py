# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
import os
import threading
from cherrypy.lib.static import serve_file

from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.simulation_history import SimulationHistoryIndex
from tvb.adapters.exporters.export_manager import ExportManager
from tvb.adapters.forms.coupling_forms import get_form_for_coupling
from tvb.adapters.forms.equation_forms import get_form_for_equation
from tvb.adapters.forms.model_forms import get_form_for_model
from tvb.adapters.forms.monitor_forms import get_form_for_monitor
from tvb.adapters.forms.noise_forms import get_form_for_noise
from tvb.adapters.simulator.range_parameters import RangeParametersCollector
from tvb.adapters.simulator.simulator_adapter import SimulatorAdapterForm, SimulatorAdapter
from tvb.adapters.forms.simulator_fragments import *
from tvb.config.init.introspector_registry import IntrospectionRegistry
from tvb.core.entities.file.simulator.view_model import AdditiveNoiseViewModel, BoldViewModel
from tvb.core.entities.file.simulator.view_model import IntegratorStochasticViewModel
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.neocom import h5
from tvb.core.services.burst_service import BurstService
from tvb.core.services.exceptions import BurstServiceException, ServicesBaseException
from tvb.core.services.import_service import ImportService
from tvb.core.services.operation_service import OperationService, GROUP_BURST_PENDING
from tvb.core.services.simulator_service import SimulatorService
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.burst.base_controller import BurstBaseController
from tvb.interfaces.web.controllers.decorators import *
from tvb.interfaces.web.controllers.simulator.monitors_wizard_handler import MonitorsWizardHandler
from tvb.interfaces.web.controllers.simulator.simulator_fragment_rendering_rules import \
    SimulatorFragmentRenderingRules, POST_REQUEST
from tvb.interfaces.web.controllers.simulator.simulator_wizzard_urls import SimulatorWizzardURLs
from tvb.interfaces.web.entities.context_simulator import SimulatorContext
from tvb.storage.storage_interface import StorageInterface


@traced
class SimulatorController(BurstBaseController):
    KEY_IS_LOAD_AFTER_REDIRECT = "is_load_after_redirect"
    COPY_NAME_FORMAT = "copy_of_{}"
    BRANCH_NAME_FORMAT = "{}_branch{}"
    KEY_KEEP_SAME_SIM_WIZARD = "keep_same_wizard"

    def __init__(self):
        BurstBaseController.__init__(self)
        self.range_parameters = RangeParametersCollector()
        self.burst_service = BurstService()
        self.simulator_service = SimulatorService()
        self.cached_simulator_algorithm = self.algorithm_service.get_algorithm_by_module_and_class(
            IntrospectionRegistry.SIMULATOR_MODULE, IntrospectionRegistry.SIMULATOR_CLASS)
        self.context = SimulatorContext()
        self.monitors_handler = MonitorsWizardHandler()

    @expose_json
    def cancel_or_remove_burst(self, burst_id):
        """
        Cancel or Remove the burst entity given by burst_id (and all linked entities: op, DTs)
        :returns True: if the op was successfully.
        """
        burst_config = BurstService.load_burst_configuration(int(burst_id))
        op_id, is_group = burst_config.operation_info_for_burst_removal

        return self.cancel_or_remove_operation(op_id, is_group, burst_config.is_finished)

    def cancel_or_remove_operation(self, operation_id, is_group, remove_after_stop=False):
        """
        Stop the operation given by operation_id. If is_group is true stop all the
        operations from that group.
        """
        # Load before we remove, to have its data in memory here
        burst_config = BurstService.get_burst_for_operation_id(operation_id, is_group)
        if burst_config is not None:
            self.burst_service.mark_burst_finished(burst_config, BurstConfiguration.BURST_CANCELED, store_h5_file=False)
            while GROUP_BURST_PENDING.get(burst_config.id, False):
                pass
            GROUP_BURST_PENDING.pop(burst_config.id, False)
        result = OperationService.stop_operation(operation_id, is_group, remove_after_stop)

        if remove_after_stop:
            current_burst = self.context.burst_config
            if (current_burst is not None and burst_config is not None and current_burst.id == burst_config.id and
                    ((current_burst.fk_simulation == operation_id and not is_group) or
                     (current_burst.fk_operation_group == operation_id and is_group))):
                self.reset_simulator_configuration()
            if burst_config is not None:
                burst_config = BurstService.load_burst_configuration(burst_config.id)
                if burst_config:
                    BurstService.remove_burst_configuration(burst_config.id)
        return result

    @expose_page
    @settings
    @context_selected
    def index(self):
        """Get on burst main page"""
        template_specification = dict(mainContent="burst/main_burst", title="Simulation Cockpit",
                                      includedResources='project/included_resources')

        if not self.context.last_loaded_fragment_url:
            self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_CONNECTIVITY_URL)

        self.context.set_burst_config()

        _, is_simulation_copy, is_simulation_load, is_branch = self.context.get_common_params()
        if self.context.burst_config.start_time is not None:
            is_simulation_load = True
            self.context.add_simulator_load_to_session(True)

        template_specification['burstConfig'] = self.context.burst_config
        template_specification['burst_list'] = self.burst_service.get_available_bursts(self.context.project.id)

        form = self.prepare_first_fragment()
        rendering_rules = SimulatorFragmentRenderingRules(
            form, SimulatorWizzardURLs.SET_CONNECTIVITY_URL, None, is_simulation_copy, is_simulation_load,
            last_form_url=self.context.last_loaded_fragment_url,
            last_request_type=cherrypy.request.method, is_first_fragment=True, is_branch=is_branch)
        template_specification.update(**rendering_rules.to_dict())

        cherrypy.response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        cherrypy.response.headers['Pragma'] = 'no-cache'
        cherrypy.response.headers['Expires'] = '0'

        return self.fill_default_attributes(template_specification)

    def prepare_first_fragment(self):
        self.context.set_simulator()
        simulator, _, _, is_branch = self.context.get_common_params()
        branch_conditions = self.simulator_service.compute_conn_branch_conditions(is_branch, simulator)
        form = self.algorithm_service.prepare_adapter_form(form_instance=SimulatorAdapterForm(),
                                                           project_id=self.context.project.id,
                                                           extra_conditions=branch_conditions)

        self.simulator_service.validate_first_fragment(form, self.context.project.id, ConnectivityIndex)
        form.fill_from_trait(self.context.simulator)
        return form

    @expose_json
    def set_fragment_url(self, **data):
        try:
            self.context.add_last_loaded_form_url_to_session(data['url'])
        except KeyError:
            self.logger.error("Cannot set last loaded url to session because the required data was not found.")

    @expose_fragment('simulator_fragment')
    def set_connectivity(self, **data):
        session_stored_simulator, is_simulation_copy, is_simulation_load, _ = self.context.get_common_params()

        if cherrypy.request.method == POST_REQUEST:
            self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_COUPLING_PARAMS_URL)
            form = SimulatorAdapterForm()
            form.fill_from_post(data)
            self.simulator_service.reset_at_connectivity_change(is_simulation_copy, form, session_stored_simulator)
            form.fill_trait(session_stored_simulator)

        next_form = self.algorithm_service.prepare_adapter_form(
            form_instance=get_form_for_coupling(type(session_stored_simulator.coupling))())
        self.range_parameters.range_param_forms[FormWithRanges.COUPLING_FRAGMENT_KEY] = next_form
        next_form.fill_from_trait(session_stored_simulator.coupling)

        rendering_rules = SimulatorFragmentRenderingRules(
            next_form, SimulatorWizzardURLs.SET_COUPLING_PARAMS_URL, SimulatorWizzardURLs.SET_CONNECTIVITY_URL,
            is_simulation_copy, is_simulation_load, self.context.last_loaded_fragment_url, cherrypy.request.method)
        return rendering_rules.to_dict()

    @expose_fragment('simulator_fragment')
    def set_coupling_params(self, **data):
        session_stored_simulator, is_simulation_copy, is_simulation_load, is_branch = self.context.get_common_params()

        if cherrypy.request.method == POST_REQUEST:
            self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_SURFACE_URL)
            form = get_form_for_coupling(type(session_stored_simulator.coupling))()
            form.fill_from_post(data)
            form.fill_trait(session_stored_simulator.coupling)

        surface_fragment = self.algorithm_service.prepare_adapter_form(form_instance=SimulatorSurfaceFragment(),
                                                                       project_id=self.context.project.id)
        surface_fragment.fill_from_trait(session_stored_simulator.surface)

        rendering_rules = SimulatorFragmentRenderingRules(
            surface_fragment, SimulatorWizzardURLs.SET_SURFACE_URL, SimulatorWizzardURLs.SET_COUPLING_PARAMS_URL,
            is_simulation_copy, is_simulation_load, self.context.last_loaded_fragment_url, cherrypy.request.method,
            is_branch=is_branch)
        return rendering_rules.to_dict()

    @expose_fragment('simulator_fragment')
    def set_surface(self, **data):
        session_stored_simulator, is_simulation_copy, is_simulation_load, is_branch = self.context.get_common_params()
        rendering_rules = SimulatorFragmentRenderingRules(
            previous_form_action_url=SimulatorWizzardURLs.SET_SURFACE_URL, is_simulation_copy=is_simulation_copy,
            is_simulation_readonly_load=is_simulation_load, last_form_url=self.context.last_loaded_fragment_url,
            last_request_type=cherrypy.request.method, is_branch=is_branch)

        if cherrypy.request.method == POST_REQUEST:
            form = SimulatorSurfaceFragment()
            form.fill_from_post(data)
            self.simulator_service.reset_at_surface_change(is_simulation_copy, form, session_stored_simulator)
            self.range_parameters.range_param_forms.pop(FormWithRanges.CORTEX_FRAGMENT_KEY, None)
            form.fill_trait(session_stored_simulator)

            if session_stored_simulator.surface:
                self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_CORTEX_URL)
            else:
                self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_STIMULUS_URL)

        return SimulatorSurfaceFragment.prepare_next_fragment_after_surface(
            session_stored_simulator, rendering_rules, self.context.project.id, SimulatorWizzardURLs.SET_CORTEX_URL,
            SimulatorWizzardURLs.SET_STIMULUS_URL)

    @expose_fragment('simulator_fragment')
    def set_cortex(self, **data):
        session_stored_simulator, is_simulation_copy, is_simulation_load, _ = self.context.get_common_params()
        rm_fragment = SimulatorRMFragment()

        if cherrypy.request.method == POST_REQUEST:
            self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_STIMULUS_URL)
            rm_fragment.fill_from_post(data)
            rm_fragment.fill_trait(session_stored_simulator.surface)

        self.range_parameters.range_param_forms[FormWithRanges.CORTEX_FRAGMENT_KEY] = rm_fragment
        rendering_rules = SimulatorFragmentRenderingRules(
            None, None, SimulatorWizzardURLs.SET_CORTEX_URL, is_simulation_copy, is_simulation_load,
            self.context.last_loaded_fragment_url, cherrypy.request.method)

        return SimulatorStimulusFragment.prepare_stimulus_fragment(session_stored_simulator, rendering_rules,
                                                                   True, SimulatorWizzardURLs.SET_STIMULUS_URL,
                                                                   self.context.project.id)

    @expose_fragment('simulator_fragment')
    def set_stimulus(self, **data):
        session_stored_simulator, is_simulation_copy, is_simulation_load, is_branch = self.context.get_common_params()

        if cherrypy.request.method == POST_REQUEST:
            self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_MODEL_URL)
            stimuli_fragment = SimulatorStimulusFragment(session_stored_simulator.is_surface_simulation)
            stimuli_fragment.fill_from_post(data)
            stimuli_fragment.fill_trait(session_stored_simulator)

        model_fragment = self.algorithm_service.prepare_adapter_form(form_instance=SimulatorModelFragment())
        model_fragment.fill_from_trait(session_stored_simulator)

        rendering_rules = SimulatorFragmentRenderingRules(
            model_fragment, SimulatorWizzardURLs.SET_MODEL_URL, SimulatorWizzardURLs.SET_STIMULUS_URL,
            is_simulation_copy, is_simulation_load, self.context.last_loaded_fragment_url, cherrypy.request.method,
            is_model_fragment=True, is_surface_simulation=session_stored_simulator.is_surface_simulation,
            is_branch=is_branch)
        return rendering_rules.to_dict()

    @expose_fragment('simulator_fragment')
    def set_model(self, **data):
        session_stored_simulator, is_simulation_copy, is_simulation_load, is_branch = self.context.get_common_params()

        if cherrypy.request.method == POST_REQUEST:
            set_next_wizard = True
            if SimulatorController.KEY_KEEP_SAME_SIM_WIZARD in data:
                set_next_wizard = False
            if set_next_wizard:
                self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_MODEL_PARAMS_URL)
            form = SimulatorModelFragment()
            form.fill_from_post(data)
            form.fill_trait(session_stored_simulator)

        form = self.algorithm_service.prepare_adapter_form(
            form_instance=get_form_for_model(type(session_stored_simulator.model))(is_branch))
        self.range_parameters.range_param_forms[FormWithRanges.MODEL_FRAGMENT_KEY] = form
        form.fill_from_trait(session_stored_simulator.model)

        rendering_rules = SimulatorFragmentRenderingRules(
            form, SimulatorWizzardURLs.SET_MODEL_PARAMS_URL, SimulatorWizzardURLs.SET_MODEL_URL, is_simulation_copy,
            is_simulation_load, self.context.last_loaded_fragment_url, cherrypy.request.method)
        return rendering_rules.to_dict()

    @expose_fragment('simulator_fragment')
    def set_model_params(self, **data):
        session_stored_simulator, is_simulation_copy, is_simulation_load, is_branch = self.context.get_common_params()

        if cherrypy.request.method == POST_REQUEST:
            self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_INTEGRATOR_URL)
            form = get_form_for_model(type(session_stored_simulator.model))(is_branch)

            if is_branch:
                data['variables_of_interest'] = list(session_stored_simulator.model.variables_of_interest)

            form.fill_from_post(data)
            form.fill_trait(session_stored_simulator.model)

        integrator_fragment = self.algorithm_service.prepare_adapter_form(form_instance=SimulatorIntegratorFragment())
        integrator_fragment.integrator.display_subform = False
        integrator_fragment.fill_from_trait(session_stored_simulator)

        rendering_rules = SimulatorFragmentRenderingRules(
            integrator_fragment, SimulatorWizzardURLs.SET_INTEGRATOR_URL,
            SimulatorWizzardURLs.SET_MODEL_PARAMS_URL, is_simulation_copy, is_simulation_load,
            self.context.last_loaded_fragment_url, cherrypy.request.method, is_branch=is_branch)
        return rendering_rules.to_dict()

    @expose_fragment('simulator_fragment')
    def set_integrator(self, **data):
        session_stored_simulator, is_simulation_copy, is_simulation_load, is_branch = self.context.get_common_params()

        if cherrypy.request.method == POST_REQUEST:
            self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_INTEGRATOR_PARAMS_URL)
            fragment = SimulatorIntegratorFragment()
            fragment.fill_from_post(data)
            fragment.fill_trait(session_stored_simulator)

        form = self.algorithm_service.prepare_adapter_form(form_instance=get_form_for_integrator(
            type(session_stored_simulator.integrator))(is_branch))

        if hasattr(form, 'noise'):
            form.noise.display_subform = False

        form.fill_from_trait(session_stored_simulator.integrator)

        rendering_rules = SimulatorFragmentRenderingRules(
            form, SimulatorWizzardURLs.SET_INTEGRATOR_PARAMS_URL, SimulatorWizzardURLs.SET_INTEGRATOR_URL,
            is_simulation_copy, is_simulation_load, self.context.last_loaded_fragment_url, cherrypy.request.method)
        return rendering_rules.to_dict()

    @expose_fragment('simulator_fragment')
    def set_integrator_params(self, **data):
        session_stored_simulator, is_simulation_copy, is_simulation_load, is_branch = self.context.get_common_params()

        if cherrypy.request.method == POST_REQUEST:
            form = get_form_for_integrator(type(session_stored_simulator.integrator))(is_branch)

            if is_branch:
                data['dt'] = str(session_stored_simulator.integrator.dt)

            form.fill_from_post(data)
            form.fill_trait(session_stored_simulator.integrator)

            if isinstance(session_stored_simulator.integrator, IntegratorStochasticViewModel):
                self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_NOISE_PARAMS_URL)
            else:
                self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_MONITORS_URL)

            self.range_parameters.range_param_forms.pop(FormWithRanges.NOISE_FRAGMENT_KEY, None)

        rendering_rules = SimulatorFragmentRenderingRules(
            None, None, SimulatorWizzardURLs.SET_INTEGRATOR_PARAMS_URL, is_simulation_copy,
            is_simulation_load, self.context.last_loaded_fragment_url, cherrypy.request.method,
            is_noise_fragment=False)

        if not isinstance(session_stored_simulator.integrator, IntegratorStochasticViewModel):
            return self.monitors_handler.prepare_monitor_fragment(session_stored_simulator, rendering_rules,
                                                                  SimulatorWizzardURLs.SET_MONITORS_URL)

        integrator_noise_fragment = get_form_for_noise(type(session_stored_simulator.integrator.noise))()
        if hasattr(integrator_noise_fragment, 'equation'):
            integrator_noise_fragment.equation.display_subform = False

        self.range_parameters.range_param_forms[FormWithRanges.NOISE_FRAGMENT_KEY] = integrator_noise_fragment
        integrator_noise_fragment.fill_from_trait(session_stored_simulator.integrator.noise)

        rendering_rules.form = integrator_noise_fragment
        rendering_rules.form_action_url = SimulatorWizzardURLs.SET_NOISE_PARAMS_URL
        rendering_rules.is_noise_fragment = True
        return rendering_rules.to_dict()

    @expose_fragment('simulator_fragment')
    def set_noise_params(self, **data):
        session_stored_simulator, is_simulation_copy, is_simulation_load, is_branch = self.context.get_common_params()

        if cherrypy.request.method == POST_REQUEST:
            form = get_form_for_noise(type(session_stored_simulator.integrator.noise))()
            form.fill_from_post(data)
            form.fill_trait(session_stored_simulator.integrator.noise)
            if isinstance(session_stored_simulator.integrator.noise, AdditiveNoiseViewModel):
                self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_MONITORS_URL)
            else:
                self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_NOISE_EQUATION_PARAMS_URL)

        rendering_rules = SimulatorFragmentRenderingRules(
            None, None, SimulatorWizzardURLs.SET_NOISE_PARAMS_URL, is_simulation_copy, is_simulation_load,
            self.context.last_loaded_fragment_url, cherrypy.request.method)

        return self.monitors_handler.prepare_next_fragment_after_noise(
            session_stored_simulator, is_branch, rendering_rules, SimulatorWizzardURLs.SET_MONITORS_URL,
            SimulatorWizzardURLs.SET_NOISE_EQUATION_PARAMS_URL)

    @expose_fragment('simulator_fragment')
    def set_noise_equation_params(self, **data):
        session_stored_simulator, is_simulation_copy, is_simulation_load, _ = self.context.get_common_params()

        if cherrypy.request.method == POST_REQUEST:
            self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_MONITORS_URL)
            form = get_form_for_equation(type(session_stored_simulator.integrator.noise.b))()
            form.fill_from_post(data)
            form.fill_trait(session_stored_simulator.integrator.noise.b)

        rendering_rules = SimulatorFragmentRenderingRules(
            None, None, SimulatorWizzardURLs.SET_NOISE_EQUATION_PARAMS_URL, is_simulation_copy, is_simulation_load,
            self.context.last_loaded_fragment_url, cherrypy.request.method)

        return self.monitors_handler.prepare_monitor_fragment(session_stored_simulator, rendering_rules,
                                                              SimulatorWizzardURLs.SET_MONITORS_URL)

    @staticmethod
    def build_monitor_url(fragment_url, monitor):
        url_regex = '{}/{}'
        url = url_regex.format(fragment_url, monitor)
        return url

    def get_first_monitor_fragment_url(self, simulator, monitors_url):
        first_monitor = simulator.first_monitor
        if first_monitor is not None:
            monitor_vm_name = type(first_monitor).__name__
            return self.build_monitor_url(monitors_url, monitor_vm_name)
        return SimulatorWizzardURLs.SETUP_PSE_URL

    @expose_fragment('simulator_fragment')
    def set_monitors(self, **data):
        session_stored_simulator, is_simulation_copy, is_simulation_load, is_branch = self.context.get_common_params()
        if cherrypy.request.method == POST_REQUEST:
            fragment = SimulatorMonitorFragment(is_surface_simulation=session_stored_simulator.is_surface_simulation)
            fragment.fill_from_post(data)
            self.monitors_handler.set_monitors_list_on_simulator(session_stored_simulator, fragment.monitors.value)

        last_loaded_fragment_url = self.get_first_monitor_fragment_url(
            session_stored_simulator, SimulatorWizzardURLs.SET_MONITOR_PARAMS_URL)

        if cherrypy.request.method == POST_REQUEST:
            self.context.add_last_loaded_form_url_to_session(last_loaded_fragment_url)

        rendering_rules = SimulatorFragmentRenderingRules(
            is_simulation_copy=is_simulation_copy, is_simulation_readonly_load=is_simulation_load,
            last_request_type=cherrypy.request.method, form_action_url=last_loaded_fragment_url,
            previous_form_action_url=SimulatorWizzardURLs.SET_MONITORS_URL)
        return self.monitors_handler.get_fragment_after_monitors(
            session_stored_simulator, self.context.burst_config, self.context.project.id,
            is_branch, rendering_rules, SimulatorWizzardURLs.SETUP_PSE_URL)

    def get_url_after_monitors(self, current_monitor, monitor_name, next_monitor):
        if isinstance(current_monitor, BoldViewModel):
            return self.build_monitor_url(SimulatorWizzardURLs.SET_MONITOR_EQUATION_URL, monitor_name)
        if next_monitor is not None:
            return self.build_monitor_url(SimulatorWizzardURLs.SET_MONITOR_PARAMS_URL, type(next_monitor).__name__)
        return SimulatorWizzardURLs.SETUP_PSE_URL

    @staticmethod
    def get_url_for_final_fragment(burst_config):
        if burst_config.is_pse_burst():
            return SimulatorWizzardURLs.LAUNCH_PSE_URL
        return SimulatorWizzardURLs.SETUP_PSE_URL

    def get_urls_for_next_monitor_fragment(self, next_monitor, current_monitor):
        form_action_url = self.build_monitor_url(SimulatorWizzardURLs.SET_MONITOR_PARAMS_URL,
                                                 type(next_monitor).__name__)
        if_bold_url = self.build_monitor_url(SimulatorWizzardURLs.SET_MONITOR_EQUATION_URL,
                                             type(current_monitor).__name__)
        return form_action_url, if_bold_url

    @expose_fragment('simulator_fragment')
    def set_monitor_params(self, current_monitor_name, **data):
        session_stored_simulator, is_simulation_copy, is_simulator_load, is_branch = self.context.get_common_params()

        current_monitor, next_monitor = self.monitors_handler.get_current_and_next_monitor_form(
            current_monitor_name, session_stored_simulator)

        if cherrypy.request.method == POST_REQUEST:
            form = get_form_for_monitor(type(current_monitor))(session_stored_simulator, is_branch)

            if is_branch:
                data['period'] = str(current_monitor.period)
                data['variables_of_interest'] = [session_stored_simulator.model.variables_of_interest[i] for i
                                                 in current_monitor.variables_of_interest]

            form.fill_from_post(data)
            form.fill_trait(current_monitor)

            last_loaded_form_url = self.get_url_after_monitors(
                current_monitor, current_monitor_name, next_monitor)
            self.context.add_last_loaded_form_url_to_session(last_loaded_form_url)

        previous_form_action_url = self.build_monitor_url(SimulatorWizzardURLs.SET_MONITOR_PARAMS_URL,
                                                          current_monitor_name)
        rendering_rules = SimulatorFragmentRenderingRules(
            is_simulation_copy=is_simulation_copy, is_simulation_readonly_load=is_simulator_load,
            last_request_type=cherrypy.request.method, last_form_url=self.context.last_loaded_fragment_url,
            previous_form_action_url=previous_form_action_url)

        form_action_url, if_bold_url = self.get_urls_for_next_monitor_fragment(next_monitor, current_monitor)
        self.monitors_handler.update_monitor(current_monitor, session_stored_simulator)
        return self.monitors_handler.handle_next_fragment_for_monitors(self.context, rendering_rules, current_monitor,
                                                                       next_monitor, False, form_action_url,
                                                                       if_bold_url)

    def get_url_after_monitor_equation(self, next_monitor):
        if next_monitor is None:
            return SimulatorWizzardURLs.SETUP_PSE_URL

        last_loaded_fragment_url = self.build_monitor_url(SimulatorWizzardURLs.SET_MONITOR_PARAMS_URL,
                                                          type(next_monitor).__name__)
        return last_loaded_fragment_url

    @expose_fragment('simulator_fragment')
    def set_monitor_equation(self, current_monitor_name, **data):
        session_stored_simulator, is_simulation_copy, is_simulator_load, is_branch = self.context.get_common_params()
        current_monitor, next_monitor = self.monitors_handler.get_current_and_next_monitor_form(
            current_monitor_name, session_stored_simulator)

        if cherrypy.request.method == POST_REQUEST:
            form = get_form_for_equation(type(current_monitor.hrf_kernel))()
            form.fill_from_post(data)
            form.fill_trait(current_monitor.hrf_kernel)

            last_loaded_fragment_url = self.get_url_after_monitor_equation(next_monitor)
            self.context.add_last_loaded_form_url_to_session(last_loaded_fragment_url)

        previous_form_action_url = self.build_monitor_url(
            SimulatorWizzardURLs.SET_MONITOR_EQUATION_URL, current_monitor_name)
        rendering_rules = SimulatorFragmentRenderingRules(
            None, None, previous_form_action_url, is_simulation_copy, is_simulator_load,
            self.context.last_loaded_fragment_url, cherrypy.request)

        form_action_url, if_bold_url = self.get_urls_for_next_monitor_fragment(next_monitor, current_monitor)
        return self.monitors_handler.handle_next_fragment_for_monitors(self.context, rendering_rules, current_monitor,
                                                                       next_monitor, True, form_action_url, if_bold_url)

    @expose_fragment('simulator_fragment')
    def setup_pse(self, **data):
        session_stored_simulator, is_simulation_copy, is_simulator_load, _ = self.context.get_common_params()
        burst_config = self.context.burst_config
        all_range_parameters = self.range_parameters.get_all_range_parameters()
        next_form = self.algorithm_service.prepare_adapter_form(
            form_instance=SimulatorPSEConfigurationFragment(all_range_parameters))

        if cherrypy.request.method == POST_REQUEST:
            session_stored_simulator.simulation_length = float(data['simulation_length'])
            burst_config.name = data['input_simulation_name_id']
            self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_PSE_PARAMS_URL)

        param1, param2 = self.burst_service.handle_range_params_at_loading(burst_config, all_range_parameters)
        if param1:
            param_dict = {'pse_param1': param1.name}
            if param2 is not None:
                param_dict['pse_param2'] = param2.name
            next_form.fill_from_post(param_dict)

        rendering_rules = SimulatorFragmentRenderingRules(next_form, SimulatorWizzardURLs.SET_PSE_PARAMS_URL,
                                                          SimulatorWizzardURLs.SETUP_PSE_URL, is_simulation_copy,
                                                          is_simulator_load, self.context.last_loaded_fragment_url,
                                                          cherrypy.request.method)
        return rendering_rules.to_dict()

    @expose_fragment('simulator_fragment')
    def set_pse_params(self, **data):
        session_stored_simulator, is_simulation_copy, is_simulation_load, _ = self.context.get_common_params()
        burst_config = self.context.burst_config
        form = SimulatorPSEConfigurationFragment(self.range_parameters.get_all_range_parameters())

        if cherrypy.request.method == POST_REQUEST:
            self.context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.LAUNCH_PSE_URL)
            form.fill_from_post(data)

            param1 = form.pse_param1.value
            burst_config.range1 = param1.to_json()
            param2 = None
            if form.pse_param2.value:
                param2 = form.pse_param2.value
                burst_config.range2 = param2.to_json()
        else:
            all_range_parameters = self.range_parameters.get_all_range_parameters()
            param1, param2 = self.burst_service.handle_range_params_at_loading(burst_config, all_range_parameters)
        next_form = self.algorithm_service.prepare_adapter_form(form_instance=SimulatorPSERangeFragment(param1, param2))

        rendering_rules = SimulatorFragmentRenderingRules(
            next_form, SimulatorWizzardURLs.LAUNCH_PSE_URL, SimulatorWizzardURLs.SET_PSE_PARAMS_URL, is_simulation_copy,
            is_simulation_load, last_form_url=self.context.last_loaded_fragment_url, is_launch_pse_fragment=True)
        return rendering_rules.to_dict()

    @expose_json
    def launch_pse(self, **data):
        session_stored_simulator = self.context.simulator
        all_range_parameters = self.range_parameters.get_all_range_parameters()
        range_param1, range_param2 = SimulatorPSERangeFragment.fill_from_post(all_range_parameters, **data)

        burst_config = self.context.burst_config
        burst_config.start_time = datetime.now()
        burst_config.range1 = range_param1.to_json()
        len_range = len(range_param1.get_range_values())
        if range_param2:
            burst_config.range2 = range_param2.to_json()
            len_range *= len(range_param2.get_range_values())

        if not self.simulator_service.operation_service.fits_max_operation_size(SimulatorAdapter(), session_stored_simulator,
                                                                                self.context.project.id, len_range):
            return {'error': self.MAX_SIZE_ERROR_MSG}

        burst_config = self.burst_service.prepare_burst_for_pse(burst_config)
        session_stored_simulator.operation_group_gid = uuid.UUID(burst_config.operation_group.gid)
        session_stored_simulator.ranges = json.dumps(burst_config.ranges)

        try:
            thread = threading.Thread(target=self.simulator_service.async_launch_and_prepare_pse,
                                      kwargs={'burst_config': burst_config,
                                              'user': self.context.logged_user,
                                              'project': self.context.project,
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
    def launch_simulation(self, launch_mode, **data):
        current_form = SimulatorFinalFragment()
        burst_config = self.context.burst_config
        burst_config.range1 = None
        burst_config.range2 = None

        try:
            current_form.fill_from_post(data)
        except Exception as exc:
            self.logger.exception(exc)
            return {'error': str(exc)}

        burst_name = current_form.simulation_name.value
        session_stored_simulator = self.context.simulator
        session_stored_simulator.simulation_length = current_form.simulation_length.value

        if burst_name != 'none_undefined':
            burst_config.name = burst_name

        if launch_mode == self.burst_service.LAUNCH_BRANCH:
            simulation_state_index = self.simulator_service.get_simulation_state_index(burst_config,
                                                                                       SimulationHistoryIndex)
            session_stored_simulator.history_gid = simulation_state_index[0].gid

        if not self.simulator_service.operation_service.fits_max_operation_size(SimulatorAdapter(), session_stored_simulator,
                                                                                self.context.project.id):
            return {'error': self.MAX_SIZE_ERROR_MSG}

        burst_config.start_time = datetime.now()
        session_burst_config = self.burst_service.store_burst(burst_config)

        try:
            thread = threading.Thread(target=self.simulator_service.async_launch_and_prepare_simulation,
                                      kwargs={'burst_config': session_burst_config,
                                              'user': self.context.logged_user,
                                              'project': self.context.project,
                                              'simulator_algo': self.cached_simulator_algorithm,
                                              'simulator': session_stored_simulator})
            thread.start()
            return {'id': session_burst_config.id}
        except BurstServiceException as e:
            self.logger.exception('Could not launch burst!')
            return {'error': e.message}

    @expose_fragment('burst/burst_history')
    def load_burst_history(self, initBurst=None):
        """
        Load the available burst that are stored in the database at this time.
        This is one alternative to 'chrome-back problem'.
        """
        bursts = self.burst_service.get_available_bursts(self.context.project.id)
        self.burst_service.populate_burst_disk_usage(bursts)
        fromInit = False
        if initBurst is not None:
            fromInit = True
        return {'burst_list': bursts,
                'fromInit': fromInit,
                'selectedBurst': self.context.burst_config.id,
                'first_fragment_url': SimulatorFragmentRenderingRules.FIRST_FORM_URL}

    @cherrypy.expose
    def get_last_fragment_url(self, burst_config_id):
        burst_config = self.burst_service.load_burst_configuration(burst_config_id)
        return self.get_url_for_final_fragment(burst_config)

    @expose_fragment('simulator_fragment')
    def load_burst_read_only(self, burst_config_id):
        try:
            burst_config = self.burst_service.load_burst_configuration(burst_config_id)
            storage_path = StorageInterface().get_project_folder(self.context.project.name,
                                                                 str(burst_config.fk_simulation))
            simulator = h5.load_view_model(burst_config.simulator_gid, storage_path)
            last_loaded_form_url = self.get_url_for_final_fragment(burst_config)
            self.context.init_session_at_burst_loading(burst_config, simulator, last_loaded_form_url)

            form = self.prepare_first_fragment()
            rendering_rules = SimulatorFragmentRenderingRules(form, SimulatorWizzardURLs.SET_CONNECTIVITY_URL,
                                                              is_simulation_readonly_load=True, is_first_fragment=True)
            return rendering_rules.to_dict()
        except Exception:
            # Most probably Burst was removed. Delete it from session, so that client
            # has a good chance to get a good response on refresh
            self.logger.exception("Error loading burst")
            self.context.remove_burst_config_from_session()
            raise

    def _prepare_first_fragment_for_burst_copy(self, burst_config_id, burst_name_format):
        simulator, burst_config_copy = self.burst_service.prepare_data_for_burst_copy(
            burst_config_id, burst_name_format, self.context.project)

        last_loaded_form_url = self.get_url_for_final_fragment(burst_config_copy)
        self.context.init_session_at_copy_preparation(burst_config_copy, simulator, last_loaded_form_url)
        return self.prepare_first_fragment()

    @expose_fragment('simulator_fragment')
    def copy_simulator_configuration(self, burst_config_id):
        self.context.add_branch_and_copy_to_session(False, True)
        form = self._prepare_first_fragment_for_burst_copy(burst_config_id, self.COPY_NAME_FORMAT)
        rendering_rules = SimulatorFragmentRenderingRules(form, SimulatorWizzardURLs.SET_CONNECTIVITY_URL,
                                                          is_simulation_copy=True, is_simulation_readonly_load=True,
                                                          is_first_fragment=True)
        return rendering_rules.to_dict()

    @expose_fragment('simulator_fragment')
    def branch_simulator_configuration(self, burst_config_id):
        self.context.add_branch_and_copy_to_session(True, False)
        form = self._prepare_first_fragment_for_burst_copy(burst_config_id, self.BRANCH_NAME_FORMAT)
        rendering_rules = SimulatorFragmentRenderingRules(form, SimulatorWizzardURLs.SET_CONNECTIVITY_URL,
                                                          is_simulation_copy=True, is_simulation_readonly_load=True,
                                                          is_first_fragment=True)

        return rendering_rules.to_dict()

    @expose_fragment('simulator_fragment')
    def reset_simulator_configuration(self):
        burst_config = BurstConfiguration(self.context.project.id)
        self.context.init_session_at_sim_reset(burst_config, SimulatorWizzardURLs.SET_CONNECTIVITY_URL)

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
        last_loaded_form_url = SimulatorWizzardURLs.SETUP_PSE_URL

        try:
            upload_param = "uploadedfile"
            if upload_param in data and data[upload_param]:
                simulator, burst_config, sim_folder = self.burst_service.load_simulation_from_zip(data[upload_param],
                                                                                                  self.context.project)

                dts_folder = os.path.join(sim_folder, StorageInterface.EXPORTED_SIMULATION_DTS_DIR)
                ImportService().import_list_of_operations(self.context.project, dts_folder, False, None)

                if burst_config.is_pse_burst():
                    last_loaded_form_url = SimulatorWizzardURLs.LAUNCH_PSE_URL
                self.context.init_session_at_sim_config_from_zip(burst_config, simulator, last_loaded_form_url)
        except IOError as ioexcep:
            self.logger.exception(ioexcep)
            self.context.set_warning_message("This ZIP does not contain a complete simulator configuration")
        except ServicesBaseException as excep:
            self.logger.warning(excep.message)
            self.context.set_warning_message(excep.message)

        self.redirect('/burst/')
