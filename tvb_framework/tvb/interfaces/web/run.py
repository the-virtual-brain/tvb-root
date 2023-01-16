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

"""
Launches the web server and configure the controllers for UI.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""
import time


STARTUP_TIC = time.time()

import os
import importlib
import sys
import webbrowser
import cherrypy
from cherrypy import Tool
from cherrypy.lib.sessions import RamSession
from subprocess import Popen, PIPE
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.config.init.initializer import initialize, reset
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.core.decorators import user_environment_execution
from tvb.core.services.exceptions import InvalidSettingsException
from tvb.core.services.hpc_operation_service import HPCOperationService
from tvb.core.services.backend_clients.standalone_client import StandAloneClient
from tvb.interfaces.web.controllers.base_controller import BaseController
from tvb.interfaces.web.controllers.burst.dynamic_model_controller import DynamicModelController
from tvb.interfaces.web.controllers.burst.exploration_controller import ParameterExplorationController
from tvb.interfaces.web.controllers.burst.noise_configuration_controller import NoiseConfigurationController
from tvb.interfaces.web.controllers.burst.region_model_parameters_controller import RegionsModelParametersController
from tvb.interfaces.web.controllers.common import KEY_PROJECT
from tvb.interfaces.web.controllers.flow_controller import FlowController
from tvb.interfaces.web.controllers.help.help_controller import HelpController
from tvb.interfaces.web.controllers.hpc_controller import HPCController
from tvb.interfaces.web.controllers.kube_controller import KubeController
from tvb.interfaces.web.controllers.project.figure_controller import FigureController
from tvb.interfaces.web.controllers.project.project_controller import ProjectController
from tvb.interfaces.web.controllers.settings_controller import SettingsController
from tvb.interfaces.web.controllers.simulator.simulator_controller import SimulatorController
from tvb.interfaces.web.controllers.spatial.base_spatio_temporal_controller import SpatioTemporalController
from tvb.interfaces.web.controllers.spatial.local_connectivity_controller import LocalConnectivityController
from tvb.interfaces.web.controllers.spatial.region_stimulus_controller import RegionStimulusController
from tvb.interfaces.web.controllers.spatial.surface_model_parameters_controller import SurfaceModelParametersController
from tvb.interfaces.web.controllers.spatial.surface_stimulus_controller import SurfaceStimulusController
from tvb.interfaces.web.controllers.burst.transfer_vector_controller import TransferVectorController
from tvb.interfaces.web.controllers.users_controller import UserController
from tvb.interfaces.web.request_handler import RequestHandler
from tvb.storage.storage_interface import StorageInterface

if __name__ == '__main__':
    if len(sys.argv) < 2:
        TvbProfile.set_profile(TvbProfile.WEB_PROFILE)
    else:
        TvbProfile.set_profile(sys.argv[1])

LOGGER = get_logger('tvb.interfaces.web.run')
CONFIG_EXISTS = not TvbProfile.is_first_run()
PARAM_RESET_DB = "reset"
LOGGER.info("TVB application will be running using encoding: " + sys.getdefaultencoding())


class CleanupSessionHandler(RamSession):
    def __init__(self, id=None, **kwargs):
        super(CleanupSessionHandler, self).__init__(id, **kwargs)

    def clean_up(self):
        """Clean up expired sessions."""

        now = self.now()
        for _id, (data, expiration_time) in self.cache.copy().items():
            if expiration_time <= now:
                if KEY_PROJECT in data:
                    selected_project = data[KEY_PROJECT]
                    StorageInterface().set_project_inactive(selected_project)
                try:
                    del self.cache[_id]
                except KeyError:
                    pass
                try:
                    if self.locks[_id].acquire(blocking=False):
                        lock = self.locks.pop(_id)
                        lock.release()
                except KeyError:
                    pass

        # added to remove obsolete lock objects
        for _id in list(self.locks):
            locked = (
                    _id not in self.cache
                    and self.locks[_id].acquire(blocking=False)
            )
            if locked:
                lock = self.locks.pop(_id)
                lock.release()


def init_cherrypy(arguments=None):
    # Mount static folders from modules marked for introspection
    arguments = arguments or []
    CONFIGUER = TvbProfile.current.web.CHERRYPY_CONFIGURATION
    if StorageInterface.encryption_enabled():
        CONFIGUER["/"]["tools.sessions.storage_class"] = CleanupSessionHandler
    for module in arguments:
        module_inst = importlib.import_module(str(module))
        module_path = os.path.dirname(os.path.abspath(module_inst.__file__))
        CONFIGUER["/static_" + str(module)] = {'tools.staticdir.on': True,
                                               'tools.staticdir.dir': '.',
                                               'tools.staticdir.root': module_path}

    # Mount controllers, and specify the root URL for them.
    cherrypy.tree.mount(BaseController(), BaseController.build_path("/"), config=CONFIGUER)
    cherrypy.tree.mount(UserController(), BaseController.build_path("/user/"), config=CONFIGUER)
    cherrypy.tree.mount(ProjectController(), BaseController.build_path("/project/"), config=CONFIGUER)
    cherrypy.tree.mount(FigureController(), BaseController.build_path("/project/figure/"), config=CONFIGUER)
    cherrypy.tree.mount(FlowController(), BaseController.build_path("/flow/"), config=CONFIGUER)
    cherrypy.tree.mount(SettingsController(), BaseController.build_path("/settings/"), config=CONFIGUER)
    cherrypy.tree.mount(HelpController(), BaseController.build_path("/help/"), config=CONFIGUER)
    cherrypy.tree.mount(SimulatorController(), BaseController.build_path("/burst/"), config=CONFIGUER)
    cherrypy.tree.mount(ParameterExplorationController(), BaseController.build_path("/burst/explore/"),
                        config=CONFIGUER)
    cherrypy.tree.mount(DynamicModelController(), BaseController.build_path("/burst/dynamic/"), config=CONFIGUER)
    cherrypy.tree.mount(SpatioTemporalController(), BaseController.build_path("/spatial/"), config=CONFIGUER)
    cherrypy.tree.mount(RegionsModelParametersController(),
                        BaseController.build_path("/burst/modelparameters/regions/"),
                        config=CONFIGUER)
    cherrypy.tree.mount(SurfaceModelParametersController(),
                        BaseController.build_path("/spatial/modelparameters/surface/"),
                        config=CONFIGUER)
    cherrypy.tree.mount(RegionStimulusController(), BaseController.build_path("/spatial/stimulus/region/"),
                        config=CONFIGUER)
    cherrypy.tree.mount(SurfaceStimulusController(), BaseController.build_path("/spatial/stimulus/surface/"),
                        config=CONFIGUER)
    cherrypy.tree.mount(LocalConnectivityController(), BaseController.build_path("/spatial/localconnectivity/"),
                        config=CONFIGUER)
    cherrypy.tree.mount(NoiseConfigurationController(), BaseController.build_path("/burst/noise/"), config=CONFIGUER)
    cherrypy.tree.mount(TransferVectorController(), TransferVectorController.build_path("/burst/transfer/"), config=CONFIGUER)
    cherrypy.tree.mount(HPCController(), BaseController.build_path("/hpc/"), config=CONFIGUER)
    cherrypy.tree.mount(KubeController(), BaseController.build_path("/kube/"), config=CONFIGUER)

    cherrypy.config.update(CONFIGUER)

    # ----------------- Register additional request handlers -----------------
    # This tool checks for MAX upload size
    cherrypy.tools.upload = Tool('on_start_resource', RequestHandler.check_upload_size)
    # This tools clean up files on disk (mainly after export)
    cherrypy.tools.cleanup = Tool('on_end_request', RequestHandler.clean_files_on_disk)
    # ----------------- End register additional request handlers ----------------

    # Register housekeeping job
    if TvbProfile.current.hpc.IS_HPC_RUN and TvbProfile.current.hpc.CAN_RUN_HPC:
        cherrypy.engine.housekeeper = cherrypy.process.plugins.BackgroundTask(
            TvbProfile.current.hpc.BACKGROUND_JOB_INTERVAL, HPCOperationService.check_operations_job)
        cherrypy.engine.housekeeper.start()

    if not TvbProfile.current.web.OPENSHIFT_DEPLOY:
        operations_job = cherrypy.process.plugins.BackgroundTask(
            TvbProfile.current.OPERATIONS_BACKGROUND_JOB_INTERVAL, StandAloneClient.process_queued_operations,
            bus=cherrypy.engine)
        operations_job.start()

    # HTTP Server is fired now #
    cherrypy.engine.start()


def expose_rest_api():
    if not TvbProfile.current.KEYCLOAK_CONFIG:
        LOGGER.info("REST server will not start because KEYCLOAK CONFIG path is not set.")
        return

    if not os.path.exists(TvbProfile.current.KEYCLOAK_CONFIG):
        LOGGER.warning("Cannot start REST server because the KEYCLOAK CONFIG file {} does not exist.".format(
            TvbProfile.current.KEYCLOAK_CONFIG))
        return

    if CONFIG_EXISTS:
        LOGGER.info("Starting Flask server with REST API...")
        run_params = [TvbProfile.current.PYTHON_INTERPRETER_PATH, '-m', 'tvb.interfaces.rest.server.run',
                      TvbProfile.CURRENT_PROFILE_NAME]
        flask_process = Popen(run_params, stderr=PIPE)
        stdout, stderr = flask_process.communicate()
        if flask_process.returncode != 0:
            LOGGER.warning("Failed to start the Flask server with REST API. Stderr: {}".format(stderr))
        else:
            LOGGER.info("Finished starting Flask server with REST API...")


def start_tvb(arguments, browser=True):
    """
    Fire CherryPy server and listen on a free port
    """

    if PARAM_RESET_DB in arguments:
        # When specified, clean everything in DB
        reset()
        arguments.remove(PARAM_RESET_DB)

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

    # Mark that the interface is Web
    ABCDisplayer.VISUALIZERS_ROOT = TvbProfile.current.web.VISUALIZERS_ROOT

    init_cherrypy(arguments)
    if StorageInterface.encryption_enabled() and StorageInterface.app_encryption_handler():
        storage_interface = StorageInterface()
        storage_interface.start()
        storage_interface.startup_cleanup()

    # Fire a browser page at the end.
    if browser:
        run_browser()

    expose_rest_api()

    # Launch CherryPy loop forever.
    LOGGER.info("Finished starting TVB version %s in %.3f s",
                TvbProfile.current.version.CURRENT_VERSION, time.time() - STARTUP_TIC)
    cherrypy.engine.block()
    cherrypy.log.error_log


@user_environment_execution
def run_browser():
    try:
        if TvbProfile.env.is_windows():
            browser_app = webbrowser.get('windows-default')
        elif TvbProfile.env.is_mac():
            browser_app = webbrowser.get('macosx')
        else:
            browser_app = webbrowser

        url_to_open = TvbProfile.current.web.BASE_LOCAL_URL
        if not CONFIG_EXISTS:
            url_to_open += 'settings/settings'

        LOGGER.info("We will try to open in a browser: " + url_to_open)
        browser_app.open(url_to_open)

    except Exception:
        LOGGER.warning("Browser could not be fired!  Please manually type in your "
                       "preferred browser: %s" % TvbProfile.current.web.BASE_LOCAL_URL)


if __name__ == '__main__':
    # Prepare parameters and fire CherryPy
    # Remove not-relevant parameter, 0 should point towards this "run.py" file, 1 to the profile
    start_tvb(sys.argv[2:])
