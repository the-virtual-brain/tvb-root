# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
TVB Raw Settings are defined here, grouped by their category of usage (e.g. cluster related, web related, etc).
Do not instantiate these classes directly, but rather use them through TvpProfile.current instance.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import os
from tvb.basic.config import stored


class VersionSettings(object):
    """
    Gather settings related to various version numbers of TVB application
    """

    # Current release number
    BASE_VERSION = "2.0.7"

    # Current DB version. Increment this and create a new xxx_update_db.py migrate script
    DB_STRUCTURE_VERSION = 18

    # This is the version of the data stored in H5 and XML files
    # and should be used by next versions to know how to import
    # data in TVB format, in case data structure changes.
    # Value should be updated every time data structure is changed.
    DATA_VERSION = 5
    DATA_VERSION_ATTRIBUTE = "Data_version"

    # This is the version of the tvb project.
    # It should be updated every time the project structure changes
    # Should this be sync-ed with data version changes?
    PROJECT_VERSION = 3

    def __init__(self, manager, bin_folder):

        # Used for reading the version file from it
        self.BIN_FOLDER = bin_folder

        # Concatenate BASE_VERSION with svn revision number
        self.CURRENT_VERSION = self.BASE_VERSION + '-' + str(self.SVN_VERSION)

        # The version up until we done the upgrade properly for the file data storage.
        self.DATA_CHECKED_TO_VERSION = manager.get_attribute(stored.KEY_LAST_CHECKED_FILE_VERSION, 1, int)

        # The version up until we done the upgrade properly for the file data storage.
        self.CODE_CHECKED_TO_VERSION = manager.get_attribute(stored.KEY_LAST_CHECKED_CODE_VERSION, -1, int)

    @property
    def SVN_VERSION(self):
        try:
            import tvb.basic.config
            config_folder = os.path.dirname(os.path.abspath(tvb.basic.config.__file__))
            with open(os.path.join(config_folder, 'tvb.version'), 'r') as version_file:
                return self.parse_svn_version(version_file.read())
        except Exception:
            pass

        return 42

    @staticmethod
    def parse_svn_version(version_string):
        if ':' in version_string:
            version_string = version_string.split(':')[1]

        number = ''.join([ch for ch in version_string if ch.isdigit()])
        return int(number)


class ClusterSettings(object):
    """
    Cluster related settings.
    """
    SCHEDULER_OAR = "oar"
    SCHEDULER_SLURM = "slurm"

    # Specify if the current process is executing an operation (via clusterLauncher)
    IN_OPERATION_EXECUTION_PROCESS = False

    _CACHED_IS_RUNNING_ON_CLUSTER = None
    _CACHED_NODE_NAME = None

    def __init__(self, manager):
        self.IS_DEPLOY = manager.get_attribute(stored.KEY_CLUSTER, False, eval)
        self.CLUSTER_SCHEDULER = manager.get_attribute(stored.KEY_CLUSTER_SCHEDULER, self.SCHEDULER_OAR)
        self.ACCEPTED_SCHEDULERS = {self.SCHEDULER_OAR: self.SCHEDULER_OAR,
                                    self.SCHEDULER_SLURM: self.SCHEDULER_SLURM}

    @property
    def SCHEDULE_COMMAND(self):
        if self.CLUSTER_SCHEDULER == self.SCHEDULER_OAR:
            return 'oarsub -q tvb -S "/home/tvbadmin/clusterLauncher %s %s" -l walltime=%s'
        return 'sbatch /home/tvbadmin/clusterLauncher %s %s %s'

    @property
    def STOP_COMMAND(self):
        if self.CLUSTER_SCHEDULER == self.SCHEDULER_OAR:
            return 'oardel %s'
        return 'scancel %s'

    @property
    def STATUS_COMMAND(self):
        if self.CLUSTER_SCHEDULER == self.SCHEDULER_OAR:
            return 'oarstat %s'
        return 'squeue -j %s'

    @property
    def JOB_ID_STRING(self):
        if self.CLUSTER_SCHEDULER == self.SCHEDULER_OAR:
            return 'OAR_JOB_ID='
        return 'Submitted batch job '

    @property
    def NODE_ENV(self):
        if self.CLUSTER_SCHEDULER == self.SCHEDULER_OAR:
            return 'OAR_NODEFILE'
        return 'SLURM_NODEID'

    @property
    def IS_RUNNING_ON_CLUSTER_NODE(self):
        """
        Returns True if current execution happens on cluster node.
        Even when IS_DEPLOY is True, this call will return False for the web machine.
        """
        if self._CACHED_IS_RUNNING_ON_CLUSTER is None:
            self._CACHED_IS_RUNNING_ON_CLUSTER = self.CLUSTER_NODE_NAME is not None

        return self._CACHED_IS_RUNNING_ON_CLUSTER

    @property
    def CLUSTER_NODE_NAME(self):
        """
        :return the name of the cluster on which TVB code is executed.
            If code is executed on a normal machine (not cluster node) returns None
        """
        # Check if the name wasn't computed before.
        if self._CACHED_NODE_NAME is None:
            # Read env variable which contains path the the file containing node name
            env_oar_nodefile = os.getenv(self.NODE_ENV)
            if env_oar_nodefile is not None and len(env_oar_nodefile) > 0 and os.path.exists(env_oar_nodefile):
                # Read node name from file (valid for OAR cluster)
                with open(env_oar_nodefile, 'r') as f:
                    node_name = f.read()
            else:
                # Valid for SLURM clusters
                node_name = os.getenv(self.NODE_ENV)

            if node_name is not None and len(node_name.strip()) > 0:
                self._CACHED_NODE_NAME = node_name.strip()
                return self._CACHED_NODE_NAME
        else:
            return self._CACHED_NODE_NAME

        return None


class WebSettings(object):
    """
    Web related specifications
    """

    LOCALHOST = "localhost"
    RENDER_HTML = True
    VISUALIZERS_ROOT = "tvb.interfaces.web.templates.jinja2.visualizers"

    def __init__(self, manager):

        self.admin = WebAdminSettings(manager)

        self.CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        try:
            import tvb.interfaces
            self.CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(tvb.interfaces.__file__)))
        except ImportError:
            pass

        self.SERVER_PORT = manager.get_attribute(stored.KEY_PORT, 8080, int)

        # Compute reference towards the current web application, valid FROM localhost
        server_IP = manager.get_attribute(stored.KEY_IP, self.LOCALHOST)
        self.BASE_LOCAL_URL = "http://%s:%s/" % (server_IP, str(self.SERVER_PORT))

        # Compute PUBLIC reference towards the current web application, it will be used in the emails text
        self.BASE_URL = ""

        # URL for reading current available version information.
        default = "http://www.thevirtualbrain.org/tvb/zwei/action/serialize-version?version=1&type=json"
        self.URL_TVB_VERSION = manager.get_attribute(stored.KEY_URL_VERSION, default)

        self.TEMPLATE_ROOT = os.path.join(self.CURRENT_DIR, 'interfaces', 'web', 'templates', 'jinja2')
        self.CHERRYPY_CONFIGURATION = {'global': {'server.socket_host': '0.0.0.0',
                                                  'server.socket_port': self.SERVER_PORT,
                                                  'server.thread_pool': 20,
                                                  'engine.autoreload_on': False,
                                                  'server.max_request_body_size': 3221225472  # 3 GB
                                                  },
                                       '/': {'tools.encode.on': True,
                                             'tools.encode.encoding': 'utf-8',
                                             'tools.decode.on': True,
                                             'tools.gzip.on': True,
                                             'tools.gzip.mime_types': ['text/html', 'text/plain',
                                                                       'text/javascript', 'text/css',
                                                                       'application/x.ndarray'],
                                             'tools.sessions.on': True,
                                             'tools.sessions.storage_type': 'ram',
                                             'tools.sessions.timeout': 600,  # 10 hours
                                             'response.timeout': 1000000,
                                             'tools.sessions.locking': 'explicit',
                                             'tools.upload.on': True,  # Tool to check upload content size
                                             'tools.cleanup.on': True  # Tool to clean up files on disk
                                             },
                                       '/static': {'tools.staticdir.root': self.CURRENT_DIR,
                                                   'tools.staticdir.on': True,
                                                   'tools.staticdir.dir': os.path.join('interfaces', 'web', 'static')
                                                   },
                                       '/statichelp': {'tools.staticdir.root': self.CURRENT_DIR,
                                                       'tools.staticdir.on': True,
                                                       'tools.staticdir.dir': os.path.join('interfaces', 'web',
                                                                                           'static', 'help')
                                                       },
                                       '/static_view': {'tools.staticdir.root': self.CURRENT_DIR,
                                                        'tools.staticdir.on': True,
                                                        'tools.staticdir.dir': os.path.join('interfaces', 'web',
                                                                                            'templates', 'jinja2',
                                                                                            'visualizers'),
                                                        },
                                       }


class WebAdminSettings(object):
    """
    Setting related to the default users of web-tvb
    """

    SYSTEM_USER_NAME = 'TVB system'
    DEFAULT_ADMIN_EMAIL = 'jira.tvb@gmail.com'
    ADMINISTRATOR_BLANK_PWD = 'pass'

    def __init__(self, manager):
        # Give name for the Admin user, first created.
        self.ADMINISTRATOR_NAME = manager.get_attribute(stored.KEY_ADMIN_NAME, 'admin')

        # Give display name for the Admin user, first created.
        self.ADMINISTRATOR_DISPLAY_NAME = manager.get_attribute(stored.KEY_ADMIN_DISPLAY_NAME, 'Administrator')

        # Admin's password used when creating first user (default is MD5 for 'pass')
        self.ADMINISTRATOR_PASSWORD = manager.get_attribute(stored.KEY_ADMIN_PWD, '1a1dc91c907325c69271ddf0c944bc72')

        # Admin's email used when creating first user
        self.ADMINISTRATOR_EMAIL = manager.get_attribute(stored.KEY_ADMIN_EMAIL, self.DEFAULT_ADMIN_EMAIL)

        # Admins GID
        self.ADMINISTRATOR_GIDS = manager.get_attribute(stored.KEY_ADMIN_GIDS, "").split(",")


class DBSettings(object):
    # Overwrite number of connections to the DB.
    # Otherwise might reach PostgreSQL limit when launching multiple concurrent operations.
    # MAX_CONNECTION default value will be used for WEB
    # When launched on cluster, the MAX_ASYNC_CONNECTIONS overwrites MAX_ONNECTIONS value
    MAX_CONNECTIONS = 20
    MAX_ASYNC_CONNECTIONS = 2

    # Nested transactions are not supported by all databases and not really necessary in TVB so far so
    # we don't support them yet. However when running tests we can use them to out advantage to rollback
    # any database changes between tests.
    ALLOW_NESTED_TRANSACTIONS = False

    def __init__(self, manager, default_storage, current_storage):
        # A dictionary with accepted db's and their default URLS
        default_pg = 'postgresql+psycopg2://postgres:root@127.0.0.1:5432/tvb?user=postgres&password=postgres'
        default_lite = 'sqlite:///' + os.path.join(default_storage, 'tvb-database.db')
        self.ACEEPTED_DBS = {'postgres': manager.get_attribute(stored.KEY_DB_URL, default_pg),
                             'sqlite': manager.get_attribute(stored.KEY_DB_URL, default_lite)}

        # Currently selected database (must be a key in ACCEPTED_DBS)
        self.SELECTED_DB = manager.get_attribute(stored.KEY_SELECTED_DB, 'sqlite')

        # Used DB url: IP,PORT. The DB  needs to be created in advance.
        default_lite = 'sqlite:///' + os.path.join(current_storage, "tvb-database.db")
        self.DB_URL = manager.get_attribute(stored.KEY_DB_URL, default_lite)

        # Upgrade/Downgrade repository
        self.DB_VERSIONING_REPO = os.path.join(current_storage, 'db_repo')
