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

"""
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

from tvb.adapters.forms.equation_forms import get_form_for_equation
from tvb.adapters.forms.monitor_forms import AdditiveNoiseViewModel, get_ui_name_to_monitor_dict, \
    get_form_for_monitor, BoldViewModel, get_monitor_to_ui_name_dict
from tvb.adapters.forms.simulator_fragments import SimulatorMonitorFragment, SimulatorFinalFragment
from tvb.basic.logger.builder import get_logger
from tvb.core.services.algorithm_service import AlgorithmService
from tvb.interfaces.web.controllers.simulator.simulator_wizzard_urls import SimulatorWizzardURLs


class MonitorsWizardHandler(object):

    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)
        self.all_monitors_dict = get_ui_name_to_monitor_dict(True)

    def set_monitors_list_on_simulator(self, session_stored_simulator, monitor_names):
        list_vms = []
        for monitor in monitor_names:
            if monitor not in self.all_monitors_dict:
                self.logger.error("Trying to configure not matched monitor " + str(monitor))
            else:
                list_vms.append(self.all_monitors_dict[monitor]())
        session_stored_simulator.monitors = list_vms

    @staticmethod
    def get_current_and_next_monitor_form(current_monitor_vm_name, simulator):

        current_monitor_index = 0
        for idx, m in enumerate(simulator.monitors):
            if type(m).__name__ == current_monitor_vm_name:
                current_monitor_index = idx
                break

        current_monitor = simulator.monitors[current_monitor_index]
        next_monitor_index = current_monitor_index + 1

        if next_monitor_index < len(simulator.monitors):
            return current_monitor, simulator.monitors[next_monitor_index]
        return current_monitor, None

    @staticmethod
    def prepare_monitor_fragment(simulator, rendering_rules, form_action_url):
        monitor_fragment = SimulatorMonitorFragment(simulator.is_surface_simulation)
        monitor_fragment.fill_from_trait(simulator.monitors)

        rendering_rules.form = monitor_fragment
        rendering_rules.form_action_url = form_action_url
        return rendering_rules.to_dict()

    @staticmethod
    def _prepare_next_fragment_if_bold(monitor, rendering_rules, form_action_url):
        next_form = get_form_for_equation(type(monitor.hrf_kernel))()
        next_form.fill_from_trait(monitor.hrf_kernel)
        rendering_rules.form = next_form
        rendering_rules.form_action_url = form_action_url
        return rendering_rules.to_dict()

    def handle_next_fragment_for_monitors(self, context, rendering_rules, current_monitor, next_monitor, is_noise_form,
                                          form_action_url, if_bold_url):
        simulator, _, _, is_branch = context.get_common_params()
        if isinstance(current_monitor, BoldViewModel) and is_noise_form is False:
            return self._prepare_next_fragment_if_bold(current_monitor, rendering_rules, if_bold_url)
        if not next_monitor:
            rendering_rules.is_branch = is_branch
            return SimulatorFinalFragment.prepare_final_fragment(simulator, context.burst_config, context.project.id,
                                                                 rendering_rules, SimulatorWizzardURLs.SETUP_PSE_URL)

        next_form = get_form_for_monitor(type(next_monitor))(simulator, is_branch)
        next_form = AlgorithmService().prepare_adapter_form(form_instance=next_form, project_id=context.project.id)
        next_form.fill_from_trait(next_monitor)
        monitor_name = self.prepare_monitor_legend(simulator.is_surface_simulation, next_monitor)
        rendering_rules.form = next_form
        rendering_rules.form_action_url = form_action_url
        rendering_rules.monitor_name = monitor_name
        return rendering_rules.to_dict()

    def prepare_next_fragment_after_noise(self, simulator, is_branch, rendering_rules, monitors_url,
                                          noise_equation_params_url):
        if isinstance(simulator.integrator.noise, AdditiveNoiseViewModel):
            rendering_rules.is_branch = is_branch
            return self.prepare_monitor_fragment(simulator, rendering_rules, monitors_url)

        equation_form = get_form_for_equation(type(simulator.integrator.noise.b))()
        equation_form.equation.data = simulator.integrator.noise.b.equation
        equation_form.fill_from_trait(simulator.integrator.noise.b)

        rendering_rules.form = equation_form
        rendering_rules.form_action_url = noise_equation_params_url
        return rendering_rules.to_dict()

    @staticmethod
    def get_fragment_after_monitors(simulator, burst_config, project_id, is_branch, rendering_rules, setup_pse_url):
        first_monitor = simulator.first_monitor
        if first_monitor is None:
            rendering_rules.is_branch = is_branch
            return SimulatorFinalFragment.prepare_final_fragment(simulator, burst_config, project_id, rendering_rules,
                                                                 setup_pse_url)

        form = get_form_for_monitor(type(first_monitor))(simulator, is_branch)
        form = AlgorithmService().prepare_adapter_form(form_instance=form)
        form.fill_from_trait(first_monitor)

        monitor_name = MonitorsWizardHandler.prepare_monitor_legend(simulator.is_surface_simulation, first_monitor)
        rendering_rules.monitor_name = monitor_name
        rendering_rules.form = form
        return rendering_rules.to_dict()

    @staticmethod
    def prepare_monitor_legend(is_surface_simulation, monitor):
        return get_monitor_to_ui_name_dict(is_surface_simulation)[type(monitor)] + ' monitor'

    @staticmethod
    def update_monitor(monitor_vm_instance, simulator):
        for idx, m in enumerate(simulator.monitors):
            if type(m) == type(monitor_vm_instance):
                simulator.monitors[idx] = monitor_vm_instance
