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
import socket
import tempfile
import webbrowser

from tvb.basic.neotraits.api import HasTraits
from tvb.config.init.datatypes_registry import populate_datatypes_registry
from tvb.interfaces.rest.client.datatype.datatype_api import DataTypeApi
from tvb.interfaces.rest.client.operation.operation_api import OperationApi
from tvb.interfaces.rest.client.project.project_api import ProjectApi
from tvb.interfaces.rest.client.simulator.simulation_api import SimulationApi
from tvb.interfaces.rest.client.user.user_api import UserApi
from tvb.interfaces.rest.commons.dtos import OperationDto, UserDto
from tvb.interfaces.rest.commons.exceptions import ClientException


class TVBClient:
    """
    TVB-BrainX3 client class which expose the whole API.
    Initializing this class with the correct rest server url is mandatory.
    The methods for loading datatypes are not intended to be used for datatypes with expandable fields (eg. TimeSeries).
    Those should be loaded in chunks, because they might be to large to be loaded in memory at once.
    """

    def __init__(self, server_url, auth_token='', login_callback_port=8888):
        # type: (str,str,int) -> None
        """
        TVBClient init method
        :param server_url: REST server URL
        :param auth_token: Authorization Bearer token (optional). It is required if you do an external login
        :param login_callback_port: Port where cherrypy login callback server will run on 127.0.0.1
        """
        populate_datatypes_registry()
        self._test_free_port(login_callback_port)
        self.temp_folder = tempfile.gettempdir()
        self.login_callback_port = login_callback_port
        self.user_api = UserApi(server_url, auth_token)
        self.project_api = ProjectApi(server_url, auth_token)
        self.datatype_api = DataTypeApi(server_url, auth_token)
        self.simulation_api = SimulationApi(server_url, auth_token)
        self.operation_api = OperationApi(server_url, auth_token)
        self.is_data_encrypted = None

    @staticmethod
    def _test_free_port(login_callback_port):
        try:
            test_socket = socket.socket()
            test_socket.connect(("127.0.0.1", login_callback_port))
            test_socket.close()
            raise ClientException("Port {} is already in use.".format(login_callback_port))
        except socket.error:
            return

    def browser_login(self):
        login_response = self.user_api.browser_login(self.login_callback_port, self.open_browser)
        self._update_token(login_response)
        self.is_data_encrypted = self.datatype_api.is_data_encrypted()

    @staticmethod
    def open_browser(url):
        """
        Open given URL in a browser. If you want to open the URL in an embedded browser for example you will have to
        override this method
        :param url: URL to open
        """
        web = webbrowser.get()
        web.open_new(url)

    def logout(self):
        """
        Logout user by invalidating the keycloak token
        """
        self.user_api.logout()

    def update_auth_token(self, auth_token):
        """
        Set the authorization token for API requests. Use this method if you handle the login and token refreshing
        processes by yourself.
        :param auth_token:
        """
        self.user_api.authorization_token = auth_token
        self.project_api.authorization_token = auth_token
        self.datatype_api.authorization_token = auth_token
        self.simulation_api.authorization_token = auth_token
        self.operation_api.authorization_token = auth_token

    def _update_token(self, response):
        self.user_api.update_tokens(response)
        self.project_api.update_tokens(response)
        self.datatype_api.update_tokens(response)
        self.simulation_api.update_tokens(response)
        self.operation_api.update_tokens(response)

    def get_users(self, page=1):
        """
        Return all TVB users
        """
        response = self.user_api.get_users(page)
        user_list, pages_no = response
        return [UserDto(**user) for user in response[user_list]], response[pages_no]

    def get_project_list(self):
        """
        Return all projects for the current logged user.
        """
        return self.user_api.get_projects_list()

    def create_project(self, project_name, project_description=''):
        """
        Create a new project which will be linked to the logged user
        :param project_name: Project name (mandatory)
        :param project_description: Project description
        :return: GID of the new project
        """
        return self.user_api.create_project(project_name, project_description)

    def add_members_to_project(self, project_gid, new_members_gid):
        # type: (str, []) -> None
        """
        Add members to the given project. Logged user must be the project administrator
        :param project_gid: Given project GID
        :param new_members_gid: List of user which will be project members
        """
        self.project_api.add_members_to_project(project_gid, new_members_gid)

    def get_data_in_project(self, project_gid):
        """
        Given a project_gid, this function will return a list of DataTypeDTO instances associated with the current project.
        """
        return self.project_api.get_data_in_project(project_gid)

    def get_operations_in_project(self, project_gid, page_number):
        """
        Given a project_gid and a page number (default page size is 20), this function will return the list of OperationDTO entities
        """
        response = self.project_api.get_operations_in_project(project_gid, page_number)
        operations = [OperationDto(**operation) for operation in response["operations"]]
        pages = response["pages"]
        return operations, pages

    def retrieve_datatype(self, datatype_gid, download_folder):
        """
        Given a guid, this function will download locally the H5 full data from the server to the given folder.
        If encryption was activated, the function will also decrypt the H5 data.
        """
        return self.datatype_api.retrieve_datatype(datatype_gid, download_folder, self.is_data_encrypted)

    def load_datatype_from_file(self, datatype_path):
        """
        Given a local H5 file location, where previously a valid H5 file has been downloaded from TVB server, load in
        memory a HasTraits subclass instance (e.g. Connectivity).
        """
        return self.datatype_api.load_datatype_from_file(datatype_path)

    def load_datatype_with_full_references(self, datatype_gid, download_folder):
        # type: (str, str) -> HasTraits
        """
        Given a datatype GID, download the entire tree of dependencies and load them in memory.
        :param datatype_gid: GID of datatype to load
        :return: datatype object with all references fully loaded
        """
        return self.datatype_api.load_datatype_with_full_references(datatype_gid, download_folder,
                                                                    self.is_data_encrypted)

    def load_datatype_with_links(self, datatype_gid, download_folder):
        # type: (str, str) -> HasTraits
        """
        Given a datatype GID, download only the corresponding H5 file and load it in memory.
        Also, instantiate empty objects as its references only for the purpose to load the GIDs on them.
        :param datatype_gid: GID of datatype to load
        :return: datatype object with correct GIDs for references
        """
        return self.datatype_api.load_datatype_with_links(datatype_gid, download_folder, self.is_data_encrypted)

    def get_operations_for_datatype(self, datatype_gid):
        """
        Given a guid, this function will return the available operations for that datatype, as a list of AlgorithmDTO instances
        """
        return self.datatype_api.get_operations_for_datatype(datatype_gid)

    def fire_simulation(self, project_gid, session_stored_simulator):
        """
        Given a project to execute the operation, and a configuration for the Simulator’s inputs, this will launch the
        simulation and return its gid
        """
        return self.simulation_api.fire_simulation(project_gid, session_stored_simulator, self.temp_folder)

    def quick_launch_operation(self, project_gid, algorithm_dto, datatype_gid):
        return self.operation_api.quick_launch_operation(project_gid, algorithm_dto, datatype_gid, self.temp_folder)

    def launch_operation(self, project_gid, algorithm_class, view_model):
        """
        This is a more generic method of launching Analyzers. Given a project id, algorithm module, algorithm classname
        and a view model instance, this function will serialize the view model and will launch the analyzer.
        """
        return self.operation_api.launch_operation(project_gid, algorithm_class, view_model, self.temp_folder)

    def get_operation_status(self, operation_gid):
        """
        Given an operation gid, this function returns the status of that operation in TVB. The status of operations can be:
        STATUS_FINISHED = "5-FINISHED"
        STATUS_PENDING = "4-PENDING"
        STATUS_STARTED = "3-STARTED"
        STATUS_CANCELED = "2-CANCELED"
        STATUS_ERROR = "1-ERROR"
        """
        return self.operation_api.get_operation_status(operation_gid)

    def get_operation_results(self, operation_gid):
        """
        Given an operation gid, this function returns a list of DataType instances (subclasses), representing the results of that
        operation if it has finished and an empty list, if the operation is still running, has failed or simply has no results.
        """
        return self.operation_api.get_operations_results(operation_gid)

    def get_extra_info(self, datatype_gid):
        """
        Given an datatype gid, this function returns a dict containing the extra information of the datatype.
        """
        return self.datatype_api.get_extra_info(datatype_gid)
