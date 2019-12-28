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

"""
TVB global configurations are predefined/read from here, for working with Framework.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import os
import sys
from tvb.basic.config import stored
from tvb.basic.config.settings import DBSettings
from tvb.basic.config.profile_settings import BaseSettingsProfile
from tvb.basic.logger import builder
from tvb.basic.logger.builder import LoggerBuilder
from tvb.config.init.datatypes_registry import populate_datatypes_registry



class WebSettingsProfile(BaseSettingsProfile):
    """
    Setting for working with storage and web interface
    """
    LOGGER_CONFIG_FILE_NAME = "logger_config.conf"

    def initialize_profile(self, change_logger_in_dev=True):
        """
        Specific initialization when functioning with storage
        """
        super(WebSettingsProfile, self).initialize_profile()

        if change_logger_in_dev and not self.env.is_distribution():
            self.LOGGER_CONFIG_FILE_NAME = "dev_logger_config.conf"

        populate_datatypes_registry()
        builder.GLOBAL_LOGGER_BUILDER = LoggerBuilder('tvb.config.logger')

    def initialize_for_deployment(self):
        """
        Specific initialization for deployment and framework settings
        """
        super(WebSettingsProfile, self).initialize_for_deployment()

        inside_static_folder = os.path.join(self.EXTERNALS_FOLDER_PARENT, 'tvb')
        self.web.CHERRYPY_CONFIGURATION['/statichelp']['tools.staticdir.root'] = inside_static_folder

        # We want to disable warnings we get from sqlalchemy for traited attributes when we are in deployment mode.
        import warnings
        from sqlalchemy import exc as sa_exc

        warnings.simplefilter("ignore", category=sa_exc.SAWarning)


class CommandSettingsProfile(WebSettingsProfile):
    """
    Profile which allows you to work in tvb with storage enable, but in console mode.
    """


class TestSQLiteProfile(WebSettingsProfile):
    """
    Defines settings for running tests on an SQLite database.
    """

    # Use a different configuration file, to make it possible to run multiple instances in the same time
    TVB_CONFIG_FILE = os.path.expanduser(os.path.join("~", '.test.tvb.configuration'))

    DEFAULT_STORAGE = os.path.expanduser(os.path.join('~', 'TVB_TEST'))

    SVN_VERSION = 1
    CODE_CHECKED_TO_VERSION = sys.maxsize
    TRADE_CRASH_SAFETY_FOR_SPEED = True


    def __init__(self):
        super(TestSQLiteProfile, self).__init__()

        self.web.RENDER_HTML = False
        self.MAX_THREADS_NUMBER = self.manager.get_attribute(stored.KEY_MAX_THREAD_NR, 2, int)

        self.TVB_STORAGE = self.manager.get_attribute(stored.KEY_STORAGE, self.DEFAULT_STORAGE, str)
        # For tests we will place logs in workspace (current folder), to have them visible from Hudson.
        self.TVB_LOG_FOLDER = "TEST_OUTPUT"
        self.TVB_TEMP_FOLDER = os.path.join(self.TVB_STORAGE, "TEMP")

        self.db = DBSettings(self.manager, self.DEFAULT_STORAGE, self.TVB_STORAGE)
        self.db.DB_URL = 'sqlite:///' + os.path.join(self.TVB_STORAGE, "tvb-database.db")
        self.db.SELECTED_DB = 'sqlite'


    def initialize_profile(self, change_logger_in_dev=False):
        super(TestSQLiteProfile, self).initialize_profile(change_logger_in_dev=change_logger_in_dev)

        from tvb.core.utils import get_matlab_executable
        self.MATLAB_EXECUTABLE = get_matlab_executable()


    def initialize_for_deployment(self):
        """
        This profile will not be used in deployment, so we try to avoid problems due to too much initialization
        """
        pass


class TestPostgresProfile(TestSQLiteProfile):
    """
    Defines settings for running tests on a Postgres database.
    """

    def __init__(self):
        super(TestPostgresProfile, self).__init__()

        # Used DB url: IP,PORT. The DB needs to be created in advance.
        self.db.DB_URL = 'postgresql+psycopg2://postgres:root@127.0.0.1:5432/tvb-test?user=postgres&password=postgres'
        self.db.SELECTED_DB = 'postgres'
