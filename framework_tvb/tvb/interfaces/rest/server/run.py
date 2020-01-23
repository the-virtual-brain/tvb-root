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

import os
import sys

from flask import Flask
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.config.init.initializer import initialize
from tvb.core.services.exceptions import InvalidSettingsException
from tvb.interfaces.rest import BASE_PATH
from tvb.interfaces.rest.server.resources.datatype.datatype_resource import RetrieveDatatypeResource, \
    GetOperationsForDatatypeResource
from tvb.interfaces.rest.server.resources.operation.operation_resource import GetOperationStatusResource, \
    GetOperationResultsResource, LaunchOperationResource
from tvb.interfaces.rest.server.resources.project.project_resource import GetOperationsInProjectResource, \
    GetDataInProjectResource
from tvb.interfaces.rest.server.resources.simulator.simulation_resource import FireSimulationResource
from tvb.interfaces.rest.server.resources.user.user_resource import GetUsersResource, GetProjectsListResource
from tvb.interfaces.rest.server.rest_api import RestApi

TvbProfile.set_profile(TvbProfile.COMMAND_PROFILE)

LOGGER = get_logger('tvb.interfaces.rest.server.run')
LOGGER.info("TVB application will be running using encoding: " + sys.getdefaultencoding())

FLASK_PORT = 9090


def initialize_tvb(arguments):
    if not os.path.exists(TvbProfile.current.TVB_STORAGE):
        try:
            os.makedirs(TvbProfile.current.TVB_STORAGE)
        except Exception:
            sys.exit("You do not have enough rights to use TVB storage folder:" + str(TvbProfile.current.TVB_STORAGE))
    try:
        initialize(arguments)
    except InvalidSettingsException as excep:
        LOGGER.exception(excep)
        sys.exit()


def build_path(path):
    return BASE_PATH + path


def initialize_flask():
    # creating the flask app
    app = Flask(__name__)
    # creating an API object
    api = RestApi(app, title="Rest services for TVB", doc="/doc/")

    # Users namespace
    name_space_users = api.namespace(build_path('/users'), description="TVB-REST APIs for users management")
    name_space_users.add_resource(GetUsersResource, '/')
    name_space_users.add_resource(GetProjectsListResource, '/<string:username>/projects')

    # Projects namespace
    name_space_projects = api.namespace(build_path('/projects'), description="TVB-REST APIs for projects management")
    name_space_projects.add_resource(GetDataInProjectResource, '/<string:project_gid>/data')
    name_space_projects.add_resource(GetOperationsInProjectResource, '/<string:project_gid>/operations/'
                                                                     '<int:page_number>')

    # Datatypes namepsace
    name_space_datatypes = api.namespace(build_path('/datatypes'), description="TVB-REST APIs for datatypes management")
    name_space_datatypes.add_resource(RetrieveDatatypeResource, '/<string:datatype_gid>')
    name_space_datatypes.add_resource(GetOperationsForDatatypeResource, '/<string:datatype_gid>/operations')

    # Operations namespace
    name_space_operations = api.namespace(build_path('/operations'),
                                          description="TVB-REST APIs for operations management")
    name_space_operations.add_resource(LaunchOperationResource, '/<string:project_gid>/algorithm'
                                                                '/<string:algorithm_module>/<string:algorithm_classname>')
    name_space_operations.add_resource(GetOperationStatusResource, '/<string:operation_gid>/status')
    name_space_operations.add_resource(GetOperationResultsResource, '/<string:operation_gid>/results')

    # Simulation namespace
    name_space_simulation = api.namespace(build_path('/simulation'),
                                          description="TVB-REST APIs for simulation management")
    name_space_simulation.add_resource(FireSimulationResource, '/<string:project_gid>')

    api.add_namespace(name_space_users)
    api.add_namespace(name_space_projects)
    api.add_namespace(name_space_datatypes)
    api.add_namespace(name_space_operations)
    api.add_namespace(name_space_simulation)

    app.run(debug=True, port=FLASK_PORT)


if __name__ == '__main__':
    # Prepare parameters and fire Flask
    # Remove not-relevant parameter, 0 should point towards this "run.py" file, 1 to the profile
    initialize_tvb(sys.argv[2:])
    initialize_flask()
