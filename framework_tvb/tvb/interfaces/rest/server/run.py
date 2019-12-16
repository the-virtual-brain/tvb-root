import os
import sys
from flask import Flask
from flask_restful import Api
from tvb.core.services.exceptions import InvalidSettingsException
from tvb.interfaces.rest.server.resources.datatype.datatype_resource import GetDatatypeResource
from tvb.interfaces.rest.server.resources.operation.operation_resource import GetOperationStatusResource, \
    GetOperationResultsResource
from tvb.interfaces.rest.server.resources.project.project_resource import GetProjectsOfAUserResource, \
    GetOperationsFromProjectResource, GetDataFromProjectResource, GetOperationsForDatatypeResource
from tvb.interfaces.rest.server.resources.simulator.fire_simulation import FireSimulationResource
from tvb.interfaces.rest.server.resources.user.user_resource import GetUsersResource
from tvb.basic.logger.builder import get_logger
from tvb.config.init.initializer import initialize
from tvb.basic.profile import TvbProfile

TvbProfile.set_profile(TvbProfile.COMMAND_PROFILE)

LOGGER = get_logger('tvb.interfaces.rest.server.run')
LOGGER.info("TVB application will be running using encoding: " + sys.getdefaultencoding())

FLASK_PORT = 9090

UPLOAD_FOLDER = TvbProfile.current.TVB_TEMP_FOLDER

app = Flask(__name__)

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


def initialize_flask():
    # creating the flask app
    app = Flask(__name__)
    # creating an API object
    api = Api(app)

    api.add_resource(GetUsersResource, '/users')
    api.add_resource(GetProjectsOfAUserResource, '/projects/<int:user_id>')
    api.add_resource(GetDataFromProjectResource, '/project/datatypes/<int:project_id>')
    api.add_resource(GetOperationsFromProjectResource, '/operations/<int:project_id>')
    api.add_resource(GetDatatypeResource, '/datatypes/<string:guid>')
    api.add_resource(GetOperationsForDatatypeResource, '/operations/datatype/<string:guid>')
    api.add_resource(FireSimulationResource, '/simulation/<int:project_id>')
    api.add_resource(GetOperationStatusResource, '/operation/status/<int:operation_id>')
    api.add_resource(GetOperationResultsResource, '/operation/results/<int:operation_id>')

    app.run(debug=True, port=FLASK_PORT)


if __name__ == '__main__':
    # Prepare parameters and fire Flask
    # Remove not-relevant parameter, 0 should point towards this "run.py" file, 1 to the profile
    initialize_tvb(sys.argv[2:])
    initialize_flask()
