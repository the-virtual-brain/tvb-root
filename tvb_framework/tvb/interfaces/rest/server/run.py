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
import sys

from flask import Flask
from flask_restx.apidoc import apidoc
from gevent.pywsgi import WSGIServer
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.config.init.initializer import initialize
from tvb.core.services.authorization import AuthorizationManager
from tvb.core.services.exceptions import InvalidSettingsException
from tvb.interfaces.rest.commons.strings import RestNamespace, RestLink, LinkPlaceholder, Strings
from tvb.interfaces.rest.server.decorators.encoders import CustomFlaskEncoder
from tvb.interfaces.rest.server.resources.datatype.datatype_resource import RetrieveDatatypeResource, \
    GetOperationsForDatatypeResource, GetExtraInfoForDatatypeResource, IsDataEncryptedResource
from tvb.interfaces.rest.server.resources.operation.operation_resource import GetOperationStatusResource, \
    GetOperationResultsResource, LaunchOperationResource
from tvb.interfaces.rest.server.resources.project.project_resource import GetOperationsInProjectResource, \
    GetDataInProjectResource, ProjectMembersResource
from tvb.interfaces.rest.server.resources.simulator.simulation_resource import FireSimulationResource
from tvb.interfaces.rest.server.resources.user.user_resource import LoginUserResource, GetProjectsListResource, \
    GetUsersResource, LinksResource
from tvb.interfaces.rest.server.rest_api import RestApi
from tvb.storage.storage_interface import StorageInterface
from werkzeug.middleware.proxy_fix import ProxyFix

TvbProfile.set_profile(TvbProfile.COMMAND_PROFILE)

LOGGER = get_logger('tvb.interfaces.rest.server.run')
LOGGER.info("TVB application will be running using encoding: " + sys.getdefaultencoding())

FLASK_PORT = 9090


def initialize_tvb_flask():
    if not os.path.exists(TvbProfile.current.TVB_STORAGE):
        try:
            os.makedirs(TvbProfile.current.TVB_STORAGE)
        except Exception:
            sys.exit("You do not have enough rights to use TVB storage folder:" + str(TvbProfile.current.TVB_STORAGE))
    try:
        initialize(skip_updates=True)
    except InvalidSettingsException as excep:
        LOGGER.exception(excep)
        sys.exit()


def build_path(namespace):
    return Strings.BASE_PATH.value + namespace.value


def initialize_flask():
    if TvbProfile.current.web.REST_DEPLOY_CONTEXT:
        apidoc.url_prefix = TvbProfile.current.web.REST_DEPLOY_CONTEXT

    # creating the flask app
    app = Flask(__name__)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
    app.json_encoder = CustomFlaskEncoder

    # creating an API object
    api = RestApi(app,
                  prefix=TvbProfile.current.web.REST_DEPLOY_CONTEXT,
                  title="Rest services for TVB",
                  doc=TvbProfile.current.web.REST_DEPLOY_CONTEXT + "/doc/",
                  version=TvbProfile.current.version.CURRENT_VERSION)

    # Users namespace
    name_space_users = api.namespace(build_path(RestNamespace.USERS), description="TVB-REST APIs for users management")
    name_space_users.add_resource(GetUsersResource, "/")
    name_space_users.add_resource(LoginUserResource, RestLink.LOGIN.compute_url())
    name_space_users.add_resource(GetProjectsListResource, RestLink.PROJECTS.compute_url())
    name_space_users.add_resource(LinksResource, RestLink.USEFUL_URLS.compute_url())

    # Projects namespace
    name_space_projects = api.namespace(build_path(RestNamespace.PROJECTS),
                                        description="TVB-REST APIs for projects management")
    name_space_projects.add_resource(GetDataInProjectResource, RestLink.DATA_IN_PROJECT.compute_url(
        values={LinkPlaceholder.PROJECT_GID.value: "<string:project_gid>"}))
    name_space_projects.add_resource(GetOperationsInProjectResource, RestLink.OPERATIONS_IN_PROJECT.compute_url(
        values={LinkPlaceholder.PROJECT_GID.value: "<string:project_gid>"}))
    name_space_projects.add_resource(ProjectMembersResource, RestLink.PROJECT_MEMBERS.compute_url(
        values={LinkPlaceholder.PROJECT_GID.value: "<string:project_gid>"}))

    # Datatypes namepsace
    name_space_datatypes = api.namespace(build_path(RestNamespace.DATATYPES),
                                         description="TVB-REST APIs for datatypes management")
    name_space_datatypes.add_resource(RetrieveDatatypeResource, RestLink.GET_DATATYPE.compute_url(
        values={LinkPlaceholder.DATATYPE_GID.value: '<string:datatype_gid>'}))
    name_space_datatypes.add_resource(GetOperationsForDatatypeResource, RestLink.DATATYPE_OPERATIONS.compute_url(
        values={LinkPlaceholder.DATATYPE_GID.value: '<string:datatype_gid>'}))
    name_space_datatypes.add_resource(GetExtraInfoForDatatypeResource, RestLink.DATATYPE_EXTRA_INFO.compute_url(
        values={LinkPlaceholder.DATATYPE_GID.value: '<string:datatype_gid>'}))
    name_space_datatypes.add_resource(IsDataEncryptedResource, RestLink.IS_DATA_ENCRYPTED.compute_url())

    # Operations namespace
    name_space_operations = api.namespace(build_path(RestNamespace.OPERATIONS),
                                          description="TVB-REST APIs for operations management")
    name_space_operations.add_resource(LaunchOperationResource, RestLink.LAUNCH_OPERATION.compute_url(values={
        LinkPlaceholder.PROJECT_GID.value: '<string:project_gid>',
        LinkPlaceholder.ALG_MODULE.value: '<string:algorithm_module>',
        LinkPlaceholder.ALG_CLASSNAME.value: '<string:algorithm_classname>'
    }))
    name_space_operations.add_resource(GetOperationStatusResource, RestLink.OPERATION_STATUS.compute_url(values={
        LinkPlaceholder.OPERATION_GID.value: '<string:operation_gid>'
    }))
    name_space_operations.add_resource(GetOperationResultsResource, RestLink.OPERATION_RESULTS.compute_url(values={
        LinkPlaceholder.OPERATION_GID.value: '<string:operation_gid>'
    }))

    # Simulation namespace
    name_space_simulation = api.namespace(build_path(RestNamespace.SIMULATION),
                                          description="TVB-REST APIs for simulation management")
    name_space_simulation.add_resource(FireSimulationResource, RestLink.FIRE_SIMULATION.compute_url(
        values={LinkPlaceholder.PROJECT_GID.value: '<string:project_gid>'}))

    api.add_namespace(name_space_users)
    api.add_namespace(name_space_projects)
    api.add_namespace(name_space_datatypes)
    api.add_namespace(name_space_operations)
    api.add_namespace(name_space_simulation)

    if StorageInterface.encryption_enabled() and StorageInterface.app_encryption_handler():
        storage_interface = StorageInterface()
        storage_interface.start()
        storage_interface.startup_cleanup()

    # Register keycloak authorization manager
    AuthorizationManager(TvbProfile.current.KEYCLOAK_CONFIG)

    http_server = WSGIServer(("0.0.0.0", FLASK_PORT), app)
    http_server.serve_forever()


if __name__ == '__main__':
    # Prepare parameters and fire Flask
    # Remove not-relevant parameter, 0 should point towards this "run.py" file, 1 to the profile
    initialize_tvb_flask()
    initialize_flask()
