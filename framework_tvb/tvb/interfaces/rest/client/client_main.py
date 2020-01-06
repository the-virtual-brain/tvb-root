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
