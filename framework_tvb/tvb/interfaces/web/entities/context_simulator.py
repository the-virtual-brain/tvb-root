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

"""
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

from tvb.core.entities.file.simulator.view_model import SimulatorAdapterModel
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.interfaces.web.controllers import common


class SimulatorContext(object):
    KEY_BURST_CONFIG = 'burst_configuration'
    KEY_SIMULATOR_CONFIG = 'simulator_configuration'
    KEY_IS_SIMULATOR_COPY = 'is_simulator_copy'
    KEY_IS_SIMULATOR_LOAD = 'is_simulator_load'
    KEY_LAST_LOADED_FORM_URL = 'last_loaded_form_url'
    KEY_IS_SIMULATOR_BRANCH = "is_branch"

    def __init__(self):
        self.project = None
        self.last_loaded_fragment_url = None

    def init_session_stored_simulator(self):
        session_stored_simulator = common.get_from_session(self.KEY_SIMULATOR_CONFIG)
        if session_stored_simulator is None:
            session_stored_simulator = SimulatorAdapterModel()
            common.add2session(self.KEY_SIMULATOR_CONFIG, session_stored_simulator)

        return session_stored_simulator

    def get_session_stored_simulator(self):
        return common.get_from_session(self.KEY_SIMULATOR_CONFIG)

    def get_current_project(self):
        self.project = common.get_current_project()

    @staticmethod
    def get_logged_user():
        return common.get_logged_user()

    def get_common_params(self):
        session_stored_simulator = common.get_from_session(self.KEY_SIMULATOR_CONFIG)
        is_simulator_copy = common.get_from_session(self.KEY_IS_SIMULATOR_COPY) or False
        is_simulator_load = common.get_from_session(self.KEY_IS_SIMULATOR_LOAD) or False
        is_branch = common.get_from_session(self.KEY_IS_SIMULATOR_BRANCH)

        return session_stored_simulator, is_simulator_copy, is_simulator_load, is_branch

    def add_last_loaded_form_url_to_session(self, last_loaded_form_url):
        self.last_loaded_fragment_url = last_loaded_form_url
        common.add2session(self.KEY_LAST_LOADED_FORM_URL, last_loaded_form_url)

    def get_last_loaded_form_url_from_session(self):
        return common.get_from_session(self.KEY_LAST_LOADED_FORM_URL)

    def add_burst_config_to_session(self, burst_config):
        common.add2session(self.KEY_BURST_CONFIG, burst_config)

    def get_burst_config_from_session(self):
        return common.get_from_session(self.KEY_BURST_CONFIG)

    def remove_burst_config_from_session(self):
        common.remove_from_session(self.KEY_BURST_CONFIG)

    def add_simulator_load_to_session(self, is_simulator_load):
        common.add2session(self.KEY_IS_SIMULATOR_LOAD, is_simulator_load)

    def get_branch_from_session(self):
        return common.get_from_session(self.KEY_IS_SIMULATOR_BRANCH)

    def init_session_at_burst_loading(self, simulator):
        common.add2session(self.KEY_SIMULATOR_CONFIG, simulator)
        common.add2session(self.KEY_IS_SIMULATOR_LOAD, True)
        common.add2session(self.KEY_IS_SIMULATOR_COPY, False)

    def add_branch_and_copy_to_session(self, branch, copy):
        common.add2session(self.KEY_IS_SIMULATOR_BRANCH, branch)
        common.add2session(self.KEY_IS_SIMULATOR_COPY, copy)

    def init_session_at_copy_preparation(self, simulator, burst_config_copy):
        common.add2session(self.KEY_SIMULATOR_CONFIG, simulator)
        common.add2session(self.KEY_IS_SIMULATOR_LOAD, False)
        common.add2session(self.KEY_BURST_CONFIG, burst_config_copy)

    def init_session_at_sim_reset(self):
        common.add2session(self.KEY_SIMULATOR_CONFIG, None)
        common.add2session(self.KEY_IS_SIMULATOR_COPY, False)
        common.add2session(self.KEY_IS_SIMULATOR_LOAD, False)
        common.add2session(self.KEY_IS_SIMULATOR_BRANCH, False)

    def init_session_at_sim_config_from_zip(self, burst_config, simulator):
        common.add2session(self.KEY_BURST_CONFIG, burst_config)
        common.add2session(self.KEY_SIMULATOR_CONFIG, simulator)
        common.add2session(self.KEY_IS_SIMULATOR_COPY, True)
        common.add2session(self.KEY_IS_SIMULATOR_LOAD, False)

    @staticmethod
    def set_warning_message(message):
        common.set_warning_message(message)
