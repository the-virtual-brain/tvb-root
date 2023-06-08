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

from enum import Enum


class Strings(Enum):
    PAGE_NUMBER = "page"
    BASE_PATH = "api"
    BEARER = "Bearer "
    AUTH_HEADER = "Authorization"
    AUTH_URL = "auth_url"
    ACCOUNT_URL = "account_url"


class RequestFileKey(Enum):
    SIMULATION_FILE_KEY = "simulation_zip_file"
    SIMULATION_FILE_NAME = "SimulationData.zip"
    LAUNCH_ANALYZERS_MODEL_FILE = "model_file"


class FormKeyInput(Enum):
    CREATE_PROJECT_NAME = 'project_name'
    CREATE_PROJECT_DESCRIPTION = 'project_description'
    CODE = 'code'
    REDIRECT_URI = 'redirect_uri'
    KEYCLOAK_REFRESH_TOKEN = 'refresh_token'
    NEW_MEMBERS_GID = 'new_members_gid'


class RestNamespace(Enum):
    USERS = "/users"
    PROJECTS = "/projects"
    DATATYPES = "/datatypes"
    OPERATIONS = "/operations"
    SIMULATION = "/simulation"


class LinkPlaceholder(Enum):
    USERNAME = "username"
    PROJECT_GID = "project_gid"
    DATATYPE_GID = "datatype_gid"
    OPERATION_GID = "operation_gid"
    ALG_MODULE = "algorithm_module"
    ALG_CLASSNAME = "algorithm_classname"


class RestLink(Enum):
    # USERS
    LOGIN = "/login"
    PROJECTS = "/logged/projects"
    USEFUL_URLS = "/kc-urls"

    # PROJECTS
    DATA_IN_PROJECT = "/{" + LinkPlaceholder.PROJECT_GID.value + "}/data"
    OPERATIONS_IN_PROJECT = "/{" + LinkPlaceholder.PROJECT_GID.value + "}/operations"
    PROJECT_MEMBERS = "/{" + LinkPlaceholder.PROJECT_GID.value + "}/members"

    # DATATYPES
    GET_DATATYPE = "/{" + LinkPlaceholder.DATATYPE_GID.value + "}"
    DATATYPE_OPERATIONS = "/{" + LinkPlaceholder.DATATYPE_GID.value + "}/operations"
    DATATYPE_EXTRA_INFO = "/{" + LinkPlaceholder.DATATYPE_GID.value + "}/extra_info"
    IS_DATA_ENCRYPTED = "/is_data_encrypted"

    # OPERATIONS
    LAUNCH_OPERATION = "/{" + LinkPlaceholder.PROJECT_GID.value + "}/algorithm/{" + LinkPlaceholder.ALG_MODULE.value + "}/{" + LinkPlaceholder.ALG_CLASSNAME.value + "}"
    OPERATION_STATUS = "/{" + LinkPlaceholder.OPERATION_GID.value + "}/status"
    OPERATION_RESULTS = "/{" + LinkPlaceholder.OPERATION_GID.value + "}/results"

    # SIMULATION
    FIRE_SIMULATION = "/{" + LinkPlaceholder.PROJECT_GID.value + "}"

    def compute_url(self, include_namespace=False, values=None):
        if values is None:
            values = {}
        result = ""
        if include_namespace:
            for namespace, links in _namespace_url_dict.items():
                if self in links:
                    result += namespace.value
                    break
        result += self.value
        return result.format(**values)


_namespace_url_dict = {
    RestNamespace.USERS: [RestLink.LOGIN, RestLink.PROJECTS, RestLink.USEFUL_URLS],
    RestNamespace.PROJECTS: [RestLink.DATA_IN_PROJECT, RestLink.OPERATIONS_IN_PROJECT, RestLink.PROJECT_MEMBERS],
    RestNamespace.DATATYPES: [RestLink.IS_DATA_ENCRYPTED, RestLink.GET_DATATYPE, RestLink.DATATYPE_OPERATIONS,
                              RestLink.DATATYPE_EXTRA_INFO],
    RestNamespace.OPERATIONS: [RestLink.LAUNCH_OPERATION, RestLink.OPERATION_STATUS, RestLink.OPERATION_RESULTS],
    RestNamespace.SIMULATION: [RestLink.FIRE_SIMULATION]
}
