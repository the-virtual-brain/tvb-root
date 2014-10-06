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
TVB global configurations are predefined/read from here.
"""

import os
import sys
from tvb.basic.config import stored
from tvb.basic.config.settings import BaseSettingsProfile



class DevelopmentProfile(BaseSettingsProfile):
    """
    Custom settings for development profile.
    """

    LOGGER_CONFIG_FILE_NAME = "dev_logger_config.conf"



class TestSQLiteProfile(BaseSettingsProfile):
    """
    Defines settings for running tests on an SQLite database.
    """

    #Use a different configuration file, to make it possible to run multiple instances in the same time
    TVB_CONFIG_FILE = os.path.expanduser(os.path.join("~", '.test.tvb.configuration'))

    DEFAULT_STORAGE = os.path.expanduser(os.path.join('~', 'TVB_TEST'))

    SVN_VERSION = 1
    CODE_CHECKED_TO_VERSION = sys.maxint
    TRADE_CRASH_SAFETY_FOR_SPEED = True


    def __init__(self):
        super(TestSQLiteProfile, self).__init__()

        self.web.RENDER_HTML = False
        self.MAX_THREADS_NUMBER = self.manager.get_attribute(stored.KEY_MAX_THREAD_NR, 2, int)

        self.TVB_STORAGE = self.manager.get_attribute(stored.KEY_STORAGE, self.DEFAULT_STORAGE, unicode)
        # For tests we will place logs in workspace, to have them visible from Hudson.
        self.TVB_LOG_FOLDER = os.path.join(self.BIN_FOLDER, "TEST_OUTPUT")

        self.db.DB_URL = 'sqlite:///' + os.path.join(self.TVB_STORAGE, "tvb-database.db")
        self.db.SELECTED_DB = 'sqlite'



class TestPostgresProfile(TestSQLiteProfile):
    """
    Defines settings for running tests on a Postgres database.
    """

    def __init__(self):
        super(TestPostgresProfile, self).__init__()
        # Used DB url: IP,PORT. The DB needs to be created in advance.
        self.db.DB_URL = 'postgresql+psycopg2://postgres:root@127.0.0.1:5432/tvb-test?user=postgres&password=postgres'
        self.db.SELECTED_DB = 'postgres'



class DeploymentProfile(BaseSettingsProfile):
    """
    Profile for deployed TVB packages.
    """

    def __init__(self):
        super(DeploymentProfile, self).__init__()

        inside_static_folder = os.path.join(self.EXTERNALS_FOLDER_PARENT, 'tvb')
        self.web.CHERRYPY_CONFIGURATION['/statichelp']['tools.staticdir.root'] = inside_static_folder


    @property
    def _LIBRARY_FOLDER(self):
        """
        Return top level library folder. Will be use for setting paths
        """
        if self.env.is_windows_deployment():
            return os.path.dirname(os.path.dirname(sys.executable))
        if self.env.is_mac_deployment():
            return os.path.dirname(self.BIN_FOLDER)
        if self.env.is_linux_deployment():
            return os.path.dirname(os.path.dirname(sys.executable))


    def _setup_tk_tcl_environ(self, root_folder):
        """
        Given a root folder to look in, find the required configuration files for TCL/TK and set the proper
        environmental variables so everything works fine in the distribution package.

        :param root_folder: the top folder from which to start looking for the required configuration files
        """
        tk_folder = self._find_file('tk.tcl', root_folder)
        if tk_folder:
            os.environ['TK_LIBRARY'] = tk_folder
        tcl_folder = self._find_file('init.tcl', root_folder)
        if tcl_folder:
            os.environ['TCL_LIBRARY'] = tcl_folder


    def _find_file(self, target_file, root_folder):
        """
        Search for a file in a folder directory. Return the folder in which the file can be found.

        :param target_file: the name of the file that is searched
        :param root_folder: the top lever folder from which to start searching in all it's subdirectories
        :returns: the name of the folder in which the file can be found
        """
        for root, _, files in os.walk(root_folder):
            for file_n in files:
                if file_n == target_file:
                    return root


    def initialize_profile(self):
        """
        This method is called at the time the config.py module is first imported. Any specific
        initializations for the profile should be placed here.
        """
        super(DeploymentProfile, self).initialize_profile()

        #We want to disable warnings we get from sqlalchemy for traited attributes
        #when we are in deployment mode.
        import warnings
        from sqlalchemy import exc as sa_exc

        warnings.simplefilter("ignore", category=sa_exc.SAWarning)
        data_path = self._LIBRARY_FOLDER
        if self.env.is_windows_deployment():
            # Add root folder as first in PYTHONPATH so we can find TVB there in case of GIT contributors
            new_python_path = self.TVB_PATH + os.pathsep
            new_python_path += data_path + os.pathsep + os.path.join(data_path, 'lib-tk')
            os.environ['PYTHONPATH'] = new_python_path
            os.environ['PATH'] = data_path + os.pathsep + os.environ.get('PATH', '')
            self._setup_tk_tcl_environ(data_path)

        if self.env.is_mac_deployment():
            # MacOS package structure is in the form:
            # Contents/Resorces/lib/python2.7/tvb . PYTHONPATH needs to be set
            # at the level Contents/Resources/lib/python2.7/ and the root path
            # from where to start looking for TK and TCL up to Contents/
            tcl_root = os.path.split(os.path.split(os.path.split(data_path)[0])[0])[0]
            self._setup_tk_tcl_environ(tcl_root)

            #Add root folder as first in PYTHONPATH so we can find TVB there in case of GIT contributors
            new_python_path = data_path + os.pathsep + os.path.join(data_path, 'site-packages.zip')
            new_python_path += os.pathsep + os.path.join(data_path, 'lib-dynload')
            new_python_path = self.TVB_PATH + os.pathsep + new_python_path
            os.environ['PYTHONPATH'] = new_python_path

        if self.env.is_linux_deployment():
            # Note that for the Linux package some environment variables like LD_LIBRARY_PATH,
            # LD_RUN_PATH, PYTHONPATH and PYTHONHOME are set also in the startup scripts.
            # Add root folder as first in PYTHONPATH so we can find TVB there in case of GIT contributors
            new_python_path = self.TVB_PATH + os.pathsep + data_path
            new_python_path += os.pathsep + os.path.join(data_path, 'lib-tk')
            os.environ['PYTHONPATH'] = new_python_path
            self._setup_tk_tcl_environ(data_path)

            ### Correctly set MatplotLib Path, before start.
            mpl_data_path_maybe = os.path.join(self._LIBRARY_FOLDER, 'mpl-data')
            try:
                os.stat(mpl_data_path_maybe)
                os.environ['MATPLOTLIBDATA'] = mpl_data_path_maybe
            except:
                pass



class CommandProfile(DeploymentProfile):
    """
    Profile which allows you to work with tvb in console mode.
    """
