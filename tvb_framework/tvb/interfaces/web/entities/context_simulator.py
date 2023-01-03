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

from tvb.core.entities.file.simulator.view_model import SimulatorAdapterModel
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.interfaces.web.controllers import common


class SimulatorContext(object):
    KEY_BURST_CONFIG = 'burst_configuration'
    KEY_SIMULATOR_CONFIG = 'simulator_configuration'
    KEY_LAST_LOADED_FORM_URL = 'last_loaded_form_url'
    KEY_IS_SIMULATOR_COPY = 'is_simulator_copy'
    KEY_IS_SIMULATOR_LOAD = 'is_simulator_load'
    KEY_IS_SIMULATOR_BRANCH = 'is_simulator_branch'

    @property
    def project(self):
        return common.get_current_project()

    @property
    def logged_user(self):
        return common.get_logged_user()

    @property
    def last_loaded_fragment_url(self):
        return common.get_from_session(self.KEY_LAST_LOADED_FORM_URL)

    @property
    def simulator(self):
        return common.get_from_session(self.KEY_SIMULATOR_CONFIG)

    @property
    def burst_config(self):
        return common.get_from_session(self.KEY_BURST_CONFIG)

    def get_common_params(self):
        session_stored_simulator = common.get_from_session(self.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(self.KEY_IS_SIMULATOR_COPY) or False
        is_simulator_load = common.get_from_session(self.KEY_IS_SIMULATOR_LOAD) or False
        is_branch = common.get_from_session(self.KEY_IS_SIMULATOR_BRANCH)

        return session_stored_simulator, is_simulator_copy, is_simulator_load, is_branch

    def set_simulator(self, simulator=None):
        # type: (SimulatorAdapterModel) -> None
        """
        Create a new simulator instance only if one does not exist in the context.
        """
        if not simulator and not self.simulator:
            simulator = SimulatorAdapterModel()
        if simulator:
            common.add2session(self.KEY_SIMULATOR_CONFIG, simulator)

    def set_burst_config(self, burst_config=None):
        # type: (BurstConfiguration) -> None
        """
        Create a new burst instance only if one does not exist in the context.
        """
        if not burst_config and not self.burst_config:
            burst_config = BurstConfiguration(self.project.id)
        if burst_config:
            common.add2session(self.KEY_BURST_CONFIG, burst_config)

    def add_last_loaded_form_url_to_session(self, last_loaded_form_url):
        # type: (str) -> None
        common.add2session(self.KEY_LAST_LOADED_FORM_URL, last_loaded_form_url)

    def remove_burst_config_from_session(self):
        common.remove_from_session(self.KEY_BURST_CONFIG)

    def add_simulator_load_to_session(self, is_simulator_load):
        # type: (bool) -> None
        common.add2session(self.KEY_IS_SIMULATOR_LOAD, is_simulator_load)

    def init_session_at_burst_loading(self, burst_config, simulator, last_loaded_form_url):
        # type: (BurstConfiguration, SimulatorAdapterModel, str) -> None
        self.set_burst_config(burst_config)
        self.set_simulator(simulator)
        self.add_last_loaded_form_url_to_session(last_loaded_form_url)
        common.add2session(self.KEY_IS_SIMULATOR_LOAD, True)
        common.add2session(self.KEY_IS_SIMULATOR_COPY, False)
        common.add2session(self.KEY_IS_SIMULATOR_BRANCH, False)

    def add_branch_and_copy_to_session(self, is_simulator_branch, is_simulator_copy):
        # type: (bool, bool) -> None
        common.add2session(self.KEY_IS_SIMULATOR_BRANCH, is_simulator_branch)
        common.add2session(self.KEY_IS_SIMULATOR_COPY, is_simulator_copy)

    def init_session_at_copy_preparation(self, burst_config, simulator, last_loaded_form_url):
        # type: (BurstConfiguration, SimulatorAdapterModel, str) -> None
        self.set_burst_config(burst_config)
        self.set_simulator(simulator)
        self.add_last_loaded_form_url_to_session(last_loaded_form_url)
        common.add2session(self.KEY_IS_SIMULATOR_LOAD, False)

    def init_session_at_sim_reset(self, burst_config, last_loaded_form_url):
        # type: (BurstConfiguration, str) -> None
        self.set_burst_config(burst_config)
        self.set_simulator(SimulatorAdapterModel())
        self.add_last_loaded_form_url_to_session(last_loaded_form_url)
        common.add2session(self.KEY_IS_SIMULATOR_COPY, False)
        common.add2session(self.KEY_IS_SIMULATOR_LOAD, False)
        common.add2session(self.KEY_IS_SIMULATOR_BRANCH, False)

    def init_session_at_sim_config_from_zip(self, burst_config, simulator, last_loaded_form_url):
        # type: (BurstConfiguration, SimulatorAdapterModel, str) -> None
        self.set_burst_config(burst_config)
        self.set_simulator(simulator)
        self.add_last_loaded_form_url_to_session(last_loaded_form_url)
        common.add2session(self.KEY_IS_SIMULATOR_COPY, True)
        common.add2session(self.KEY_IS_SIMULATOR_LOAD, False)
        common.add2session(self.KEY_IS_SIMULATOR_BRANCH, False)

    @staticmethod
    def set_warning_message(message):
        common.set_warning_message(message)

    def clean_project_data_from_session(self, remove_project=True):
        common.remove_from_session(self.KEY_SIMULATOR_CONFIG)
        common.remove_from_session(self.KEY_LAST_LOADED_FORM_URL)
        common.remove_from_session(self.KEY_BURST_CONFIG)
        common.remove_from_session(self.KEY_IS_SIMULATOR_BRANCH)
        common.add2session(self.KEY_IS_SIMULATOR_LOAD, False)
        if remove_project:
            common.remove_project_from_session()
