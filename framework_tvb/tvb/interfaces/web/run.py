# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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

"""
Launches the web server and configure the controllers for UI.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import time
STARTUP_TIC = time.time()

import os
import sys
import cherrypy
import webbrowser
from cherrypy import Tool

### This will set running profile from arguments.
### Reload modules, only when running, thus avoid problems when sphinx generates documentation
from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(sys.argv[1], try_reload=(__name__ == '__main__'))

### For Linux Distribution, correctly set MatplotLib Path, before start.
from tvb.basic.config.settings import TVBSettings
if TvbProfile.is_linux_deployment():
    mpl_data_path_maybe = os.path.join(TVBSettings().get_library_folder(), 'mpl-data')
    try:
        os.stat(mpl_data_path_maybe)
        os.environ['MATPLOTLIBDATA'] = mpl_data_path_maybe
    except:
        pass

### Import MPLH5 asap, to have the back-end Thread started before other pylab/matplotlib import
from tvb.basic.logger.builder import get_logger
if __name__ == "__main__":
    from tvb.interfaces.web.mplh5 import mplh5_server
    LOGGER = get_logger('tvb.interfaces.web.mplh5.mplh5_server')
    mplh5_server.start_server(LOGGER)

from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.core.decorators import user_environment_execution
from tvb.core.services.initializer import initialize, reset
from tvb.core.services.exceptions import InvalidSettingsException
from tvb.interfaces.web.request_handler import RequestHandler
from tvb.interfaces.web.controllers.base_controller import BaseController
from tvb.interfaces.web.controllers.users_controller import UserController
from tvb.interfaces.web.controllers.help.help_controller import HelpController
from tvb.interfaces.web.controllers.project.project_controller import ProjectController
from tvb.interfaces.web.controllers.project.figure_controller import FigureController
from tvb.interfaces.web.controllers.flow_controller import FlowController
from tvb.interfaces.web.controllers.settings_controller import SettingsController
from tvb.interfaces.web.controllers.burst.burst_controller import BurstController
from tvb.interfaces.web.controllers.burst.region_model_parameters_controller import RegionsModelParametersController
from tvb.interfaces.web.controllers.burst.exploration_controller import ParameterExplorationController
from tvb.interfaces.web.controllers.burst.dynamic_model_controller import DynamicModelController
from tvb.interfaces.web.controllers.spatial.base_spatio_temporal_controller import SpatioTemporalController
from tvb.interfaces.web.controllers.spatial.surface_model_parameters_controller import SurfaceModelParametersController
from tvb.interfaces.web.controllers.spatial.region_stimulus_controller import RegionStimulusController
from tvb.interfaces.web.controllers.spatial.surface_stimulus_controller import SurfaceStimulusController
from tvb.interfaces.web.controllers.spatial.local_connectivity_controller import LocalConnectivityController
from tvb.interfaces.web.controllers.burst.noise_configuration_controller import NoiseConfigurationController
from tvb.interfaces.web.controllers.api.simulator_controller import SimulatorController


LOGGER = get_logger('tvb.interfaces.web.run')
CONFIG_EXISTS = not TVBSettings.is_first_run()
PARAM_RESET_DB = "reset"
LOGGER.info("TVB application will be running using encoding: " + sys.getdefaultencoding())


def init_cherrypy(arguments=None):
    #### Mount static folders from modules marked for introspection
    arguments = arguments or []
    CONFIGUER = TVBSettings.CHERRYPY_CONFIGURATION
    for module in arguments:
        module_inst = __import__(str(module), globals(), locals(), ["__init__"])
        module_path = os.path.dirname(os.path.abspath(module_inst.__file__))
        CONFIGUER["/static_" + str(module)] = {'tools.staticdir.on': True,
                                               'tools.staticdir.dir': '.',
                                               'tools.staticdir.root': module_path}

    #### Mount controllers, and specify the root URL for them.
    cherrypy.tree.mount(BaseController(), "/", config=CONFIGUER)
    cherrypy.tree.mount(UserController(), "/user/", config=CONFIGUER)
    cherrypy.tree.mount(ProjectController(), "/project/", config=CONFIGUER)
    cherrypy.tree.mount(FigureController(), "/project/figure/", config=CONFIGUER)
    cherrypy.tree.mount(FlowController(), "/flow/", config=CONFIGUER)
    cherrypy.tree.mount(SettingsController(), "/settings/", config=CONFIGUER)
    cherrypy.tree.mount(HelpController(), "/help/", config=CONFIGUER)
    cherrypy.tree.mount(BurstController(), "/burst/", config=CONFIGUER)
    cherrypy.tree.mount(ParameterExplorationController(), "/burst/explore/", config=CONFIGUER)
    cherrypy.tree.mount(DynamicModelController(), "/burst/dynamic/", config=CONFIGUER)
    cherrypy.tree.mount(SpatioTemporalController(), "/spatial/", config=CONFIGUER)
    cherrypy.tree.mount(RegionsModelParametersController(), "/burst/modelparameters/regions/", config=CONFIGUER)
    cherrypy.tree.mount(SurfaceModelParametersController(), "/spatial/modelparameters/surface/", config=CONFIGUER)
    cherrypy.tree.mount(RegionStimulusController(), "/spatial/stimulus/region/", config=CONFIGUER)
    cherrypy.tree.mount(SurfaceStimulusController(), "/spatial/stimulus/surface/", config=CONFIGUER)
    cherrypy.tree.mount(LocalConnectivityController(), "/spatial/localconnectivity/", config=CONFIGUER)
    cherrypy.tree.mount(NoiseConfigurationController(), "/burst/noise/", config=CONFIGUER)
    cherrypy.tree.mount(SimulatorController(), "/api/simulator/", config=CONFIGUER)

    cherrypy.config.update(CONFIGUER)

    #----------------- Register additional request handlers -----------------
    # This tool checks for MAX upload size
    cherrypy.tools.upload = Tool('on_start_resource', RequestHandler.check_upload_size)
    # This tools clean up files on disk (mainly after export)
    cherrypy.tools.cleanup = Tool('on_end_request', RequestHandler.clean_files_on_disk)
    #----------------- End register additional request handlers ----------------

    #### HTTP Server is fired now ######  
    cherrypy.engine.start()



def start_tvb(arguments, browser=True):
    """
    Fire CherryPy server and listen on a free port
    """

    if PARAM_RESET_DB in arguments:
    ##### When specified, clean everything in DB
        reset()
        arguments.remove(PARAM_RESET_DB)

    if not os.path.exists(TVBSettings.TVB_STORAGE):
        try:
            os.makedirs(TVBSettings.TVB_STORAGE)
        except Exception:
            sys.exit("You do not have enough rights to use TVB storage folder:" + str(TVBSettings.TVB_STORAGE))

    try:
        initialize(arguments)
    except InvalidSettingsException, excep:
        LOGGER.exception(excep)
        sys.exit()

    #### Mark that the interface is Web
    ABCDisplayer.VISUALIZERS_ROOT = TVBSettings.WEB_VISUALIZERS_ROOT
    ABCDisplayer.VISUALIZERS_URL_PREFIX = TVBSettings.WEB_VISUALIZERS_URL_PREFIX

    init_cherrypy(arguments)

    #### Fire a browser page at the end.
    if browser:
        run_browser()

    ## Launch CherryPy loop forever.
    LOGGER.info("Finished starting TVB version %s in %.3f s", TVBSettings.CURRENT_VERSION, time.time() - STARTUP_TIC)
    cherrypy.engine.block()
    cherrypy.log.error_log



@user_environment_execution
def run_browser():
    try:
        if TvbProfile.is_windows():
            browser_app = webbrowser.get('windows-default')
        elif TvbProfile.is_mac():
            browser_app = webbrowser.get('macosx')
        else:
            browser_app = webbrowser

        url_to_open = TVBSettings.BASE_LOCAL_URL
        if not CONFIG_EXISTS:
            url_to_open += 'settings/settings'

        LOGGER.info("We will try to open in a browser: " + url_to_open)
        browser_app.open(url_to_open)

    except Exception:
        LOGGER.warning("Browser could not be fired!  Please manually type in your "
                       "preferred browser: %s" % TVBSettings.BASE_LOCAL_URL)



if __name__ == '__main__':
    #### Prepare parameters and fire CherryPy
    #### Remove not-relevant parameter, 0 should point towards this "run.py" file, 1 to the profile
    start_tvb(sys.argv[2:])
