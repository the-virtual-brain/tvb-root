import os
import sys

from flask import Flask
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.config.init.initializer import initialize
from tvb.core.services.exceptions import InvalidSettingsException
from tvb.interfaces.rest.server.resources.datatype.datatype_resource import RetrieveDatatypeResource, \
    GetOperationsForDatatypeResource
from tvb.interfaces.rest.server.resources.operation.operation_resource import GetOperationStatusResource, \
    GetOperationResultsResource, LaunchOperationResource
from tvb.interfaces.rest.server.resources.project.project_resource import GetOperationsInProjectResource, \
    GetDataInProjectResource
from tvb.interfaces.rest.server.resources.simulator.fire_simulation import FireSimulationResource
from tvb.interfaces.rest.server.resources.user.user_resource import GetUsersResource, GetProjectsListResource
from tvb.interfaces.rest.server.rest_api import RestApi

TvbProfile.set_profile(TvbProfile.COMMAND_PROFILE)

LOGGER = get_logger('tvb.interfaces.rest.server.run')
LOGGER.info("TVB application will be running using encoding: " + sys.getdefaultencoding())

FLASK_PORT = 9090
BASE_PATH = "/api"


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
    api = RestApi(app)

    api.add_resource(GetUsersResource, build_path('/users'))
    api.add_resource(GetProjectsListResource, build_path('/users/<string:username>/projects'))
    api.add_resource(GetDataInProjectResource, build_path('/projects/<string:project_gid>/data'))
    api.add_resource(GetOperationsInProjectResource, build_path('/projects/<string:project_gid>/operations'))
    api.add_resource(RetrieveDatatypeResource, build_path('/datatypes/<string:datatype_gid>'))
    api.add_resource(GetOperationsForDatatypeResource, build_path('/datatypes/<string:datatype_gid>/operations'))
    api.add_resource(FireSimulationResource, build_path('/simulation/<string:project_gid>'))
    api.add_resource(LaunchOperationResource, build_path('/operations/<string:project_gid>/algorithm'
                                                         '/<string:algorithm_module>/<string:algorithm_classname>'))
    api.add_resource(GetOperationStatusResource, build_path('/operations/<string:operation_gid>/status'))
    api.add_resource(GetOperationResultsResource, build_path('/operations/<string:operation_gid>/results'))

    app.run(debug=True, port=FLASK_PORT)


if __name__ == '__main__':
    # Prepare parameters and fire Flask
    # Remove not-relevant parameter, 0 should point towards this "run.py" file, 1 to the profile
    initialize_tvb(sys.argv[2:])
    initialize_flask()
