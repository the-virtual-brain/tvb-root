# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
Prepare TVB settings to be grouped under various profile classes.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import os
import sys
from tvb.basic.config import stored
from tvb.basic.config.environment import Environment
from tvb.basic.config.settings import ClusterSettings, DBSettings, VersionSettings, WebSettings, HPCSettings


class BaseSettingsProfile(object):
    TVB_USER_HOME = os.environ.get('TVB_USER_HOME', '~')

    TVB_CONFIG_FILE = os.path.expanduser(os.path.join(TVB_USER_HOME, '.tvb.configuration'))

    DEFAULT_STORAGE = os.path.expanduser(os.path.join(TVB_USER_HOME, 'TVB' + os.sep))
    FIRST_RUN_STORAGE = os.path.expanduser(os.path.join(TVB_USER_HOME, '.tvb-temp'))

    LOGGER_CONFIG_FILE_NAME = "logger_config.conf"

    # Access rights for TVB generated files/folders.
    ACCESS_MODE_TVB_FILES = 0o744

    # Number used for estimation of TVB used storage space
    MAGIC_NUMBER = 9

    def __init__(self):

        self.manager = stored.SettingsManager(self.TVB_CONFIG_FILE)

        # Actual storage of all TVB related files
        self.KEYCLOAK_CONFIG = self.manager.get_attribute(stored.KEY_KC_CONFIGURATION, '')
        self.KEYCLOAK_LOGIN_ENABLED = self.manager.get_attribute(stored.KEY_ENABLE_KC_LOGIN, False, eval)
        self.KEYCLOAK_WEB_CONFIG = self.manager.get_attribute(stored.KEY_KC_WEB_CONFIGURATION, '')
        self.TVB_STORAGE = self.manager.get_attribute(stored.KEY_STORAGE, self.FIRST_RUN_STORAGE, str)
        self.UPLOAD_KEY_PATH = self.manager.get_attribute(stored.KEY_UPLOAD_PRIVATE_KEY_PATH, None, str)
        self.TRACE_USER_ACTIONS = self.manager.get_attribute(stored.KEY_TRACE_USER_ACTIONS, False, eval)
        self.TVB_LOG_FOLDER = os.path.join(self.TVB_STORAGE, "logs")
        self.TVB_TEMP_FOLDER = os.path.join(self.TVB_STORAGE, "TEMP")

        self.env = Environment()
        self.cluster = ClusterSettings(self.manager)
        self.hpc = HPCSettings(self.manager)
        self.web = WebSettings(self.manager)
        self.db = DBSettings(self.manager, self.DEFAULT_STORAGE, self.TVB_STORAGE)
        self.version = VersionSettings(self.manager, self.BIN_FOLDER)
        self.file_storage = self.manager.get_attribute(stored.KEY_FILE_STORAGE, 'h5', str)

        # Maximum number of vertices acceptable o be part of a surface at import time.
        self.MAX_SURFACE_VERTICES_NUMBER = self.manager.get_attribute(stored.KEY_MAX_NR_SURFACE_VERTEX, 300000, int)
        # Max number of ops that can be scheduled from UI in a PSE. To be correlated with the oarsub limitations
        self.MAX_RANGE_NUMBER = self.manager.get_attribute(stored.KEY_MAX_RANGE_NR, 2000, int)
        # Max number of threads in the pool of ops running in parallel. TO be correlated with CPU cores
        self.MAX_THREADS_NUMBER = self.manager.get_attribute(stored.KEY_MAX_THREAD_NR, 4, int)
        self.OPERATIONS_BACKGROUND_JOB_INTERVAL = self.manager.get_attribute(stored.KEY_OP_BACKGROUND_INTERVAL, 60, int)
        # The maximum disk space that can be used by one single user, in KB.
        self.MAX_DISK_SPACE = self.manager.get_attribute(stored.KEY_MAX_DISK_SPACE_USR, 5 * 1024 * 1024, int)

        # The url of the elasticsearch server
        self.ELASTICSEARCH_URL = self.manager.get_attribute(stored.KEY_ELASTICSEARCH_URL, "", str)
        # The security key that is used to connect to the server
        self.ELASTICSEARCH_API_KEY = self.manager.get_attribute(stored.KEY_ELASTICSEARCH_API_KEY, "", str)
        # The request timeout for the elasticsearch rest calls
        self.ELASTICSEARCH_LOGGING_INDEX = self.manager.get_attribute(stored.KEY_ELASTICSEARCH_LOGGING_INDEX, "", str)
        self.ELASTICSEARCH_REQUEST_TIMEOUT = self.manager.get_attribute(stored.KEY_ELASTICSEARCH_REQUEST_TIMEOUT, 30, int)
        # The number of logs in a message batch that are sent to the server
        self.ELASTICSEARCH_BUFFER_THRESHOLD = self.manager.get_attribute(stored.KEY_ELASTICSEARCH_BUFFER_THRESHOLD, 1000000, int)

    @property
    def BIN_FOLDER(self):
        """
        Return path towards tvb_bin location. It will be used in some environment for determining the starting point
        """
        try:
            import tvb_bin
            return os.path.dirname(os.path.abspath(tvb_bin.__file__))
        except (ImportError, TypeError):
            return "."

    @property
    def PYTHON_INTERPRETER_PATH(self):
        """
        Get Python path, based on current environment.
        """
        if self.env.is_mac_deployment():
            return os.path.join(os.path.dirname(sys.executable), "python")

        return sys.executable

    def prepare_for_operation_mode(self):
        """
        Overwrite PostgreSQL number of connections when executed in the context of a node.
        """
        self.db.MAX_CONNECTIONS = self.db.MAX_ASYNC_CONNECTIONS
        self.cluster.IN_OPERATION_EXECUTION_PROCESS = True
        self.hpc.IN_OPERATION_EXECUTION_PROCESS = True

    def initialize_profile(self):
        """
        Make sure tvb folders are created.
        """
        if not os.path.exists(self.TVB_LOG_FOLDER):
            os.makedirs(self.TVB_LOG_FOLDER)

        if not os.path.exists(self.TVB_TEMP_FOLDER):
            os.makedirs(self.TVB_TEMP_FOLDER)

        if not os.path.exists(self.TVB_STORAGE):
            os.makedirs(self.TVB_STORAGE)

    def initialize_for_deployment(self):

        library_folder = self.env.get_library_folder(self.BIN_FOLDER)

        if self.env.is_windows_deployment():
            self.env.setup_python_path(library_folder, os.path.join(library_folder, 'lib-tk'))
            self.env.append_to_path(library_folder)
            self.env.setup_tk_tcl_environ(library_folder)

        if self.env.is_mac_deployment():
            # MacOS package structure is in the form:
            # Contents/Resorces/lib/python2.7/tvb . PYTHONPATH needs to be set
            # at the level Contents/Resources/lib/python2.7/ and the root path
            # from where to start looking for TK and TCL up to Contents/
            tcl_root = os.path.dirname(os.path.dirname(os.path.dirname(library_folder)))
            self.env.setup_tk_tcl_environ(tcl_root)

            self.env.setup_python_path(library_folder,
                                       os.path.join(library_folder, 'site-packages'),
                                       os.path.join(library_folder, 'site-packages.zip'),
                                       os.path.join(library_folder, 'lib-dynload'))

        if self.env.is_linux_deployment():
            # Note that for the Linux package some environment variables like LD_LIBRARY_PATH,
            # LD_RUN_PATH, PYTHONPATH and PYTHONHOME are set also in the startup scripts.
            self.env.setup_python_path(library_folder, os.path.join(library_folder, 'lib-tk'))
            self.env.setup_tk_tcl_environ(library_folder)

            # Correctly set MatplotLib Path, before start.
            mpl_data_path_maybe = os.path.join(library_folder, 'mpl-data')
            try:
                os.stat(mpl_data_path_maybe)
                os.environ['MATPLOTLIBDATA'] = mpl_data_path_maybe
            except:
                pass


class LibrarySettingsProfile(BaseSettingsProfile):
    """
    Profile used when scientific library is used without storage and without web UI.
    """

    TVB_STORAGE = os.path.expanduser(os.path.join("~", "TVB" + os.sep))
    LOGGER_CONFIG_FILE_NAME = "library_logger.conf"

    def __init__(self):
        super(LibrarySettingsProfile, self).__init__()


class TestLibraryProfile(LibrarySettingsProfile):
    """
    Profile for library unit-tests.
    """

    LOGGER_CONFIG_FILE_NAME = "library_logger_test.conf"

    def __init__(self):
        super(TestLibraryProfile, self).__init__()
        self.TVB_LOG_FOLDER = "TEST_OUTPUT"


class MATLABLibraryProfile(LibrarySettingsProfile):
    """
    Profile use library use from MATLAB.
    """

    LOGGER_CONFIG_FILE_NAME = None
