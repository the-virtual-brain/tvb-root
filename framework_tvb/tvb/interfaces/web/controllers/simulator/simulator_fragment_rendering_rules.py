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
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.simulator.simulator_wizzard_urls import SimulatorWizzardURLs

GET_REQUEST = 'GET'
POST_REQUEST = 'POST'


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
                 last_request_type=GET_REQUEST, is_first_fragment=False, is_launch_fragment=False,
                 is_model_fragment=False, is_surface_simulation=False, is_noise_fragment=False,
                 is_launch_pse_fragment=False, is_pse_launch=False, monitor_name=None,
                 is_branch=False):
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
        self.is_branch = is_branch

    @property
    def load_readonly(self):
        if self.last_request_type == GET_REQUEST and self.form_action_url != self.last_form_url:
            return True
        return self._is_simulation_readonly_load

    @property
    def disable_fields(self):
        if self.is_branch and not self.include_launch_button:
            return True
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
        if self.is_branch and self.is_launch_fragment:
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
        return {"renderer": self,
                "showOnlineHelp": common.get_logged_user().is_online_help_active(),
                "isCallout": False}

