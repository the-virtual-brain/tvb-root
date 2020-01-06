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

import tempfile
from tvb.interfaces.rest.client.datatype.datatype_api import DataTypeApi
from tvb.interfaces.rest.client.operation.operation_api import OperationApi
from tvb.interfaces.rest.client.project.project_api import ProjectApi
from tvb.interfaces.rest.client.simulator.simulation_api import SimulationApi
from tvb.interfaces.rest.client.user.user_api import UserApi


class MainClient:

    def __init__(self, server_url):
        self.temp_folder = tempfile.gettempdir()
        self.user_api = UserApi(server_url)
        self.project_api = ProjectApi(server_url)
        self.datatype_api = DataTypeApi(server_url)
        self.simulation_api = SimulationApi(server_url)
        self.operation_api = OperationApi(server_url)

    def get_users(self):
        return self.user_api.get_users()

    def get_project_list(self, username):
        return self.user_api.get_projects_list(username)

    def get_data_in_project(self, project_gid):
        return self.project_api.get_data_in_project(project_gid)

    def get_operations_in_project(self, project_gid):
        return self.project_api.get_operations_in_project(project_gid)

    def retrieve_datatype(self, datatype_gid, download_folder):
        return self.datatype_api.retrieve_datatype(datatype_gid, download_folder)

    def get_operations_for_datatype(self, datatype_gid):
        return self.datatype_api.get_operations_for_datatype(datatype_gid)

    def fire_simulation(self, project_gid, session_stored_simulator, burst_config):
        return self.simulation_api.fire_simulation(project_gid, session_stored_simulator,
                                                   burst_config, self.temp_folder)

    def launch_operation(self, project_gid, algorithm_module, algorithm_classname):
        return self.operation_api.launch_operation(project_gid, algorithm_module, algorithm_classname,
                                                   self.temp_folder)

    def get_operation_status(self, operation_gid):
        return self.operation_api.get_operation_status(operation_gid)

    def get_operation_results(self, operation_gid):
        return self.operation_api.get_operations_results(operation_gid)
