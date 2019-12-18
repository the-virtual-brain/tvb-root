import os
import sys

from flask import Flask
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.config.init.initializer import initialize
from tvb.core.services.exceptions import InvalidSettingsException
from tvb.interfaces.rest.server.resources.datatype.datatype_resource import RetrieveDatatypeResource
from tvb.interfaces.rest.server.resources.operation.operation_resource import GetOperationStatusResource, \
    GetOperationResultsResource
from tvb.interfaces.rest.server.resources.project.project_resource import GetProjectsListResource, \
    GetOperationsInProjectResource, GetDataInProjectResource, GetOperationsForDatatypeResource
from tvb.interfaces.rest.server.resources.simulator.fire_simulation import FireSimulationResource
from tvb.interfaces.rest.server.resources.user.user_resource import GetUsersResource
from tvb.interfaces.rest.server.rest_api import RestApi

TvbProfile.set_profile(TvbProfile.COMMAND_PROFILE)

LOGGER = get_logger('tvb.interfaces.rest.server.run')
LOGGER.info("TVB application will be running using encoding: " + sys.getdefaultencoding())

FLASK_PORT = 9090
BASE_PATH = "/api"

UPLOAD_FOLDER = TvbProfile.current.TVB_TEMP_FOLDER


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
    api.add_resource(GetProjectsListResource, build_path('/projects/<int:user_id>'))
    api.add_resource(GetDataInProjectResource, build_path('/datatypes/project/<int:project_id>'))
    api.add_resource(RetrieveDatatypeResource, build_path('/datatypes/<string:guid>'))
    api.add_resource(FireSimulationResource, build_path('/simulation/<int:project_id>'))
    api.add_resource(GetOperationsInProjectResource, build_path('/operations/<int:project_id>'))
    api.add_resource(GetOperationsForDatatypeResource, build_path('/operations/datatype/<string:guid>'))
    api.add_resource(GetOperationStatusResource, build_path('/operations/<int:operation_id>/status'))
    api.add_resource(GetOperationResultsResource, build_path('/operations/<int:operation_id>/results'))

    app.run(debug=True, port=FLASK_PORT)


if __name__ == '__main__':
    # Prepare parameters and fire Flask
    # Remove not-relevant parameter, 0 should point towards this "run.py" file, 1 to the profile
    initialize_tvb(sys.argv[2:])
    initialize_flask()
