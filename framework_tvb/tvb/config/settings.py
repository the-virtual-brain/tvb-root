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
from copy import copy
from sys import platform
from subprocess import Popen, PIPE
from tvb.basic.profile import TvbProfile
from tvb.basic.config.utils import ClassProperty, EnhancedDictionary
from functools import wraps


def settings_loaded():
    """
    Annotation to check if file settings are loaded before returning attribute.
    """

    def dec(func):
        @wraps(func)        
        def deco(*a, **b):            
            if FrameworkSettings.FILE_SETTINGS is None:
                FrameworkSettings.read_config_file()
            return func(*a, **b)
        return deco
    return dec



def find_file(target_file, root_folder):
    """
    Search for a file in a folder directory. Return the folder in which
    the file can be found.
    :param target_file: the name of the file that is searched
    :param root_folder: the top lever folder from which to start searching in all it's \
                        subdirectories
        
    :returns: the name of the folder in which the file can be found
    """
    for root, _, files in os.walk(root_folder):
        for file_n in files:
            if file_n == target_file:
                return root



def setup_tk_tcl_environ(root_folder):
    """
    Given a root folder to look in, find the required configuration files
    for TCL/TK and set the proper environmental variables so everything works
    fine in the distribution package.
    
    :param root_folder: the top folder from which to start looking for the
        required configuration files
    """
    tk_folder = find_file('tk.tcl', root_folder)
    if tk_folder:
        os.environ['TK_LIBRARY'] = tk_folder
    tcl_folder = find_file('init.tcl', root_folder)
    if tcl_folder:
        os.environ['TCL_LIBRARY'] = tcl_folder



class BaseProfile():
    """
    Class handling correct settings to be loaded.
    Contains default values for all settings.
    """
    # I. Attributes that are going to be set at initialization time:
    MPLH5_Server_Thread = None
    #The settings loaded from the configuration file. At first import, if this
    #variable is None, it will try to load the settings from the TVB_CONFIG_FILE
    FILE_SETTINGS = None

    MAGIC_NUMBER = 9


    # II. Attributes with value not changeable from settings page:
    DB_CURRENT_VERSION = 10
    # Overwrite number of connections to the DB. 
    # Otherwise might reach PostgreSQL limit when launching multiple concurrent operations.
    # MAX_DB_CONNECTION default value will be used for WEB  
    # When launched on cluster, the MAX_DB_ASYNC_CONNECTIONS overwrites MAX_DB_CONNECTIONS value 
    MAX_DB_CONNECTIONS = 20
    MAX_DB_ASYNC_CONNECTIONS = 2
    BASE_VERSION = "1.2"
    # Nested transactions are not supported by all databases and not really necessary in TVB so far so
    # we don't support them yet. However when running tests we can use them to out advantage to rollback 
    # any database changes between tests.
    ALLOW_NESTED_TRANSACTIONS = False

    # This is the version of the data stored in H5 and XML files
    # and should be used by next versions to know how to import
    # data in TVB format, in case data structure changes.
    # Value should be updated every time data structure is changed.
    DATA_VERSION = 2
    DATA_VERSION_ATTRIBUTE = "Data_version"

    # This is the version of the tvb project.
    # It should be updated every time the project structure changes
    # Should this be sync-ed with data version changes?
    PROJECT_VERSION = 1

    @ClassProperty
    @staticmethod
    @settings_loaded()
    def DATA_CHECKED_TO_VERSION():
        """The version up until we done the upgrade properly for the file data storage."""
        return FrameworkSettings.get_attribute(FrameworkSettings.KEY_LAST_CHECKED_FILE_VERSION, 1, int)


    @ClassProperty
    @staticmethod
    @settings_loaded()
    def FILE_STORAGE_UPGRADE_STATUS():
        """Keeps track if the storage upgrade was successfull or not. This is set to `invalid` in case
        some of the datatypes upgrade failed. It's not used thus far but we might want to have it so 
        we can re-run an improved upgrade script on all datatypes that are invalid."""
        return FrameworkSettings.get_attribute(FrameworkSettings.KEY_FILE_STORAGE_UPDATE_STATUS, 'valid')


    @ClassProperty
    @staticmethod
    def SVN_VERSION():
        """Current SVN version in the package running now."""
        svn_variable = 'SVN_REVISION'
        if svn_variable in os.environ:
            return os.environ[svn_variable]

        try:
            with open(os.path.join(FrameworkSettings.BIN_FOLDER, 'tvb.version'), 'r') as version_file:
                return BaseProfile.parse_svn_version(version_file.read())
        except Exception:
            pass

        try:
            _proc = Popen(["svnversion", "."], stdout=PIPE)
            return BaseProfile.parse_svn_version(_proc.communicate()[0])
        except Exception:
            pass

        try:
            proc = Popen(['git', 'rev-parse', 'HEAD'], stdout=PIPE)
            return proc.stdout.read().strip()
        except Exception:
            pass

        raise ValueError('cannot determine svn version')


    @ClassProperty
    @staticmethod
    def CURRENT_VERSION():
        """ Concatenate BASE_VERSION with svn revision number"""
        return FrameworkSettings.BASE_VERSION + '-' + str(FrameworkSettings.SVN_VERSION)


    @ClassProperty
    @staticmethod
    @settings_loaded()
    def CODE_CHECKED_TO_VERSION():
        """The version up until we done the upgrade properly for the file data storage."""
        default = -1
        version_string = FrameworkSettings.get_attribute(FrameworkSettings.KEY_LAST_CHECKED_CODE_VERSION, str(default))
        try:
            return BaseProfile.parse_svn_version(version_string)
        except Exception:
            return default


    # Access rights for TVB generated files/folders.
    ACCESS_MODE_TVB_FILES = 0744

    LOCALHOST = "127.0.0.1"
    SYSTEM_USER_NAME = 'TVB system'
    DEFAULT_ADMIN_EMAIL = 'jira.tvb@gmail.com'


    CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


    @ClassProperty
    @staticmethod
    def BIN_FOLDER():
        """
        :return:
        """
        try:
            import tvb_bin
            return os.path.dirname(os.path.abspath(tvb_bin.__file__))
        except ImportError:
            return FrameworkSettings.CURRENT_DIR


    @ClassProperty
    @staticmethod
    def EXTERNALS_FOLDER_PARENT():
        return os.path.dirname(FrameworkSettings.BIN_FOLDER)

    # Specify if the current process is executing an operation (via clusterLauncher)
    OPERATION_EXECUTION_PROCESS = False

    CLUSTER_SCHEDULE_COMMAND = 'oarsub ' \
                               '-p "host>\'n02\' AND host>\'n02\'" ' \
                               '-l walltime=%s ' \
                               '-q tvb ' \
                               '-S "/home/tvbadmin/clusterLauncher %s %s"'
    CLUSTER_STOP_COMMAND = 'oardel %s'

    _CACHED_RUNNING_ON_CLUSTER_NODE = None


    @ClassProperty
    @staticmethod
    def RUNNING_ON_CLUSTER_NODE():
        """
        Returns True if current execution happens on cluster node.
        """
        if FrameworkSettings._CACHED_RUNNING_ON_CLUSTER_NODE is None:
            FrameworkSettings._CACHED_RUNNING_ON_CLUSTER_NODE = FrameworkSettings.CLUSTER_NODE_NAME is not None

        return FrameworkSettings._CACHED_RUNNING_ON_CLUSTER_NODE


    _CACHED_CLUSTER_NODE_NAME = None


    @ClassProperty
    @staticmethod
    def CLUSTER_NODE_NAME():
        """
        Returns the name of the cluster on which TVB code is executed.
        If code is executed on a normal machine (not cluster node) returns None
        """
        # Check if the name wasn't computed before.
        if FrameworkSettings._CACHED_CLUSTER_NODE_NAME is None:
            # Read env variable which contains path the the file containing node name
            env_oar_nodefile = os.getenv('OAR_NODEFILE')
            if env_oar_nodefile is not None and len(env_oar_nodefile) > 0:
                # Read node name from file
                with open(env_oar_nodefile, 'r') as f:
                    node_name = f.read()

                if node_name is not None and len(node_name.strip()) > 0:
                    FrameworkSettings._CACHED_CLUSTER_NODE_NAME = node_name.strip()
                    return FrameworkSettings._CACHED_CLUSTER_NODE_NAME
        else:
            return FrameworkSettings._CACHED_CLUSTER_NODE_NAME

        return None


    TVB_CONFIG_FILE = os.path.expanduser(os.path.join("~", '.tvb.configuration'))
    #the folder containing the XSD files used for validating applications XMLs
    SCHEMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core', 'schema')
    # Web Specifications:
    RENDER_HTML = True
    TEMPLATE_ROOT = os.path.join(CURRENT_DIR, 'interfaces', 'web', 'templates', 'genshi')
    WEB_VISUALIZERS_ROOT = "tvb.interfaces.web.templates.genshi.visualizers"
    WEB_VISUALIZERS_URL_PREFIX = "/flow/read_datatype_attribute/"
    # Traits Specific
    TRAITS_CONFIGURATION = EnhancedDictionary()
    TRAITS_CONFIGURATION.interface_method_name = 'interface'
    TRAITS_CONFIGURATION.use_storage = True
    #Logger specific
    LOGGER_CONFIG_FILE_NAME = "logger_config.conf"


    @classmethod
    def initialize_profile(cls):
        """
        This method is called at the time the config.py module is first imported. Any specific
        initializations for the profile should be placed here.
        """
        pass

    FIRST_RUN_STORAGE = os.path.expanduser(os.path.join('~', '.tvb-temp'))
    DEFAULT_STORAGE = os.path.expanduser(os.path.join('~', 'TVB'))

    # III. Attributes that can be overwritten from config file.
    #     Will have only default values in here.
    @ClassProperty
    @staticmethod
    @settings_loaded()
    def TVB_STORAGE():
        """Root folder for all projects and users."""
        default = FrameworkSettings.FIRST_RUN_STORAGE
        return FrameworkSettings.get_attribute(FrameworkSettings.KEY_STORAGE, default, unicode)


    @ClassProperty
    @staticmethod
    @settings_loaded()
    def MAX_DISK_SPACE():
        """ The maximum disk space that can be used by one single user, in KB. """
        return FrameworkSettings.get_attribute(FrameworkSettings.KEY_MAX_DISK_SPACE_USR, 5 * 1024 * 1024, int)


    @ClassProperty
    @staticmethod
    @settings_loaded()
    def MATLAB_EXECUTABLE():
        """ The path to the matlab executable (if existend). Otherwise just return an empty string. """
        value = FrameworkSettings.get_attribute(FrameworkSettings.KEY_MATLAB_EXECUTABLE, '', str) or ''
        if value == 'None':
            value = ''
        return value


    @ClassProperty
    @staticmethod
    def TVB_TEMP_FOLDER():
        """ 
        Represents a temporary folder, where to store things for a while.
        Content of this folder can be deleted at any time.
        """
        tmp_path = os.path.join(FrameworkSettings.TVB_STORAGE, "TEMP")
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        return tmp_path


    @ClassProperty
    @staticmethod
    def TVB_LOG_FOLDER():
        """ 
        Represents a folder, where all log files are stored.
        """
        tmp_path = os.path.join(FrameworkSettings.TVB_STORAGE, "logs")
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        return tmp_path


    # DB related attributes:
    @ClassProperty
    @staticmethod
    def ACEEPTED_DBS():
        """A dictionary with accepted db's and their default URLS"""
        return {'postgres': FrameworkSettings.get_attribute(FrameworkSettings.KEY_DB_URL,
                            'postgresql+psycopg2://postgres:root@127.0.0.1:5432/tvb?user=postgres&password=postgres'),
                'sqlite': FrameworkSettings.get_attribute(FrameworkSettings.KEY_DB_URL,
                            'sqlite:///' + os.path.join(FrameworkSettings.DEFAULT_STORAGE, "tvb-database.db"))}


    @ClassProperty
    @staticmethod
    @settings_loaded()
    def SELECTED_DB():
        """Selected DB should be a key that exists in 
        the ACCEPTED_DBS dictionary"""
        return FrameworkSettings.get_attribute(FrameworkSettings.KEY_SELECTED_DB, 'sqlite')


    @ClassProperty
    @staticmethod
    @settings_loaded()
    def DB_URL():
        """Used DB url: IP,PORT. The DB  needs to be created in advance."""
        default = 'sqlite:///' + os.path.join(FrameworkSettings.TVB_STORAGE, "tvb-database.db")
        return FrameworkSettings.get_attribute(FrameworkSettings.KEY_DB_URL, default)


    @ClassProperty
    @staticmethod
    def DB_VERSIONING_REPO():
        """Upgrade/Downgrade repository"""
        return os.path.join(FrameworkSettings.TVB_STORAGE, 'db_repo')


    # IP and Ports

    # The maximum number of threads to allocate in case TVB is ran locally instead
    # of cluster. This represents the maximum number of operations that can be executed
    # in parallel.
    @ClassProperty
    @staticmethod
    @settings_loaded()
    def MAX_THREADS_NUMBER():
        """Maximum number of threads in the pool of simulations range."""
        return FrameworkSettings.get_attribute(FrameworkSettings.KEY_MAX_THREAD_NR, 4, int)


    # The maximum number of operations that can be launched with a PSE mechanism.
    # when setting ranges with a bigger number of resulting operations, an exception will be thrown.
    # oarsub on the cluster has a maximum number of entries in the queue also set. 
    # This number should not be bigger than the oar setting, or ops will be lost.
    @ClassProperty
    @staticmethod
    @settings_loaded()
    def MAX_RANGE_NUMBER():
        """Maximum number of operations that can be scheduled from UI."""
        return FrameworkSettings.get_attribute(FrameworkSettings.KEY_MAX_RANGE_NR, 2000, int)


    # The maximum number of vertices that are allowed for a surface.
    # System will not allow import of surfaces with more vertices than this value.
    @ClassProperty
    @staticmethod
    @settings_loaded()
    def MAX_SURFACE_VERTICES_NUMBER():
        """Maximum number of vertices acceptable o be part of a surface at import time."""
        return FrameworkSettings.get_attribute(FrameworkSettings.KEY_MAX_NR_SURFACE_VERTEX, 300000, int)


    @ClassProperty
    @staticmethod
    @settings_loaded()
    def WEB_SERVER_PORT():
        """CherryPy port"""
        return FrameworkSettings.get_attribute(FrameworkSettings.KEY_PORT, 8080, int)


    @ClassProperty
    @staticmethod
    @settings_loaded()
    def MPLH5_SERVER_PORT():
        """Post for the Matplotlib HTML5 backend"""
        return FrameworkSettings.get_attribute(FrameworkSettings.KEY_PORT_MPLH5, 9000, int)


    @ClassProperty
    @staticmethod
    @settings_loaded()
    def DEPLOY_CLUSTER():
        """
        Only when DEPLOY_CLUSTER, CLUSTER_SCHEDULE_COMMAND is used."""
        return FrameworkSettings.get_attribute(FrameworkSettings.KEY_CLUSTER, False, eval)


    @ClassProperty
    @staticmethod
    def BASE_LOCAL_URL():
        """PUBLIC WEB reference towards the web site TVB."""
        server_IP = FrameworkSettings.get_attribute(FrameworkSettings.KEY_IP, FrameworkSettings.LOCALHOST)
        return "http://%s:%s/" % (server_IP, str(FrameworkSettings.WEB_SERVER_PORT))


    @ClassProperty
    @staticmethod
    def BASE_URL():
        """PUBLIC WEB reference towards the web site TVB."""
        default = FrameworkSettings.BASE_LOCAL_URL
        return FrameworkSettings.get_attribute(FrameworkSettings.KEY_URL_WEB, default)


    @ClassProperty
    @staticmethod
    @settings_loaded()
    def MPLH5_SERVER_URL():
        """URL for accessing the Matplotlib HTML5 backend"""
        server_IP = FrameworkSettings.get_attribute(FrameworkSettings.KEY_IP, FrameworkSettings.LOCALHOST)
        default = "ws://%s:%s/" % (server_IP, str(FrameworkSettings.MPLH5_SERVER_PORT))
        return FrameworkSettings.get_attribute(FrameworkSettings.KEY_URL_MPLH5, default)


    @ClassProperty
    @staticmethod
    def URL_TVB_VERSION():
        """URL for reading current available version information."""
        default = "http://www.thevirtualbrain.org/tvb/zwei/action/serialize-version?version=1&type=json"
        return FrameworkSettings.get_attribute(FrameworkSettings.KEY_URL_VERSION, default)


    # ADMINISTRATOR user:
    @ClassProperty
    @staticmethod
    @settings_loaded()
    def ADMINISTRATOR_NAME():
        """Give name for the Admin user, first created."""
        return FrameworkSettings.get_attribute(FrameworkSettings.KEY_ADMIN_NAME, 'admin')


    @ClassProperty
    @staticmethod
    @settings_loaded()
    def ADMINISTRATOR_PASSWORD():
        """Admin's password used when creating first user"""
        return FrameworkSettings.get_attribute(FrameworkSettings.KEY_ADMIN_PWD, '1a1dc91c907325c69271ddf0c944bc72')
        # MD5 for 'pass'


    @ClassProperty
    @staticmethod
    def ADMINISTRATOR_BLANK_PWD():
        """Unencrypted default password"""
        return 'pass'


    @ClassProperty
    @staticmethod
    @settings_loaded()
    def ADMINISTRATOR_EMAIL():
        """Admin's email used when creating first user"""
        return FrameworkSettings.get_attribute(FrameworkSettings.KEY_ADMIN_EMAIL, FrameworkSettings.DEFAULT_ADMIN_EMAIL)


    @ClassProperty
    @staticmethod
    @settings_loaded()
    def TVB_PATH():
        return FrameworkSettings.get_attribute(FrameworkSettings.KEY_TVB_PATH, '')


    # CherryPy settings:
    @ClassProperty
    @staticmethod
    def CHERRYPY_CONFIGURATION():
        """CherryPy startup configuration"""
        return {
            'global': {
                'server.socket_host': '0.0.0.0',
                'server.socket_port': FrameworkSettings.WEB_SERVER_PORT,
                'server.thread_pool': 20,
                'engine.autoreload_on': False,
                'server.max_request_body_size': 1932735283  # 1.8 GB
            },
            '/': {
                'tools.encode.on': True,
                'tools.encode.encoding': 'utf-8',
                'tools.decode.on': True,
                'tools.gzip.on': True,
                'tools.sessions.on': True,
                'tools.sessions.storage_type': 'ram',
                'tools.sessions.timeout': 6000,  # 100 hours
                'response.timeout': 1000000,
                'tools.sessions.locking': 'explicit',
                'tools.upload.on': True,    # Tool to check upload content size
                'tools.cleanup.on': True    # Tool to clean up files on disk
            },
            '/static': {
                'tools.staticdir.root': FrameworkSettings.CURRENT_DIR,
                'tools.staticdir.on': True,
                'tools.staticdir.dir': os.path.join('interfaces', 'web', 'static')
            },
            '/statichelp': {
                'tools.staticdir.root': FrameworkSettings.CURRENT_DIR,
                'tools.staticdir.on': True,
                'tools.staticdir.dir': os.path.join('interfaces', 'web', 'static', 'help')
            },
            '/static_view': {
                'tools.staticdir.root': FrameworkSettings.CURRENT_DIR,
                'tools.staticdir.on': True,
                'tools.staticdir.dir': os.path.join('interfaces', 'web', 'templates', 'genshi', 'visualizers'),
            },
            '/schema': {
                'tools.staticdir.root': FrameworkSettings.CURRENT_DIR,
                'tools.staticdir.on': True,
                'tools.staticdir.dir': os.path.join('core', 'schema'),
            },
        }


    @staticmethod
    def parse_svn_version(version_string):
        if ':' in version_string:
            version_string = version_string.split(':')[1]

        number = ''.join([ch for ch in version_string if ch.isdigit()])
        return int(number)


    @staticmethod
    def read_config_file():
        """
        Get data from the configurations file in the form of a dictionary.
        Return None if file not present.
        """
        if not os.path.exists(FrameworkSettings.TVB_CONFIG_FILE):
            return None
        config_dict = {}
        with open(FrameworkSettings.TVB_CONFIG_FILE, 'r') as cfg_file:
            data = cfg_file.read()
            entries = [line for line in data.split('\n') if not line.startswith('#') and len(line.strip()) > 0]
            for one_entry in entries:
                name, value = one_entry.split('=', 1)
                config_dict[name] = value
            FrameworkSettings.FILE_SETTINGS = config_dict
        return config_dict


    @classmethod
    def add_entries_to_config_file(cls, input_data):
        """
        Set the LAST_CHECKED_FILE_VERSION from the settings file to the current DATA_VERSION.
        
        :param input_data: A dictionary of pairs that need to be added to the config file.
        """
        config_dict = FrameworkSettings.read_config_file()
        if config_dict is None:
            config_dict = {}
        for entry in input_data:
            config_dict[entry] = input_data[entry]
        with open(cls.TVB_CONFIG_FILE, 'w') as file_writer:
            for key in config_dict:
                file_writer.write(key + '=' + str(config_dict[key]) + '\n')


    @classmethod
    def update_config_file(cls, data_dict):
        """
        Update data from configuration file, without restart. Used by
        methods like change password, or change email.
        """
        config_dict = FrameworkSettings.read_config_file()
        if config_dict is None:
            config_dict = data_dict
        else:
            for key in config_dict:
                if key in data_dict:
                    config_dict[key] = data_dict[key]
        with open(cls.TVB_CONFIG_FILE, 'w') as file_writer:
            for key in config_dict:
                file_writer.write(key + '=' + str(config_dict[key]) + '\n')


    @staticmethod
    def get_attribute(attr_name, default=None, dtype=str):
        """
        Get a cfg attribute that could also be found in the settings file.
        """
        try:
            if FrameworkSettings.FILE_SETTINGS and attr_name in FrameworkSettings.FILE_SETTINGS:
                return dtype(FrameworkSettings.FILE_SETTINGS[attr_name])
        except ValueError:
            ## Invalid convert operation.
            return default
        return default


    #File keys
    KEY_ADMIN_NAME = 'ADMINISTRATOR_NAME'
    KEY_ADMIN_PWD = 'ADMINISTRATOR_PASSWORD'
    KEY_ADMIN_EMAIL = 'ADMINISTRATOR_EMAIL'
    KEY_TVB_PATH = 'TVB_PATH'
    KEY_STORAGE = 'TVB_STORAGE'
    KEY_MAX_DISK_SPACE_USR = 'USR_DISK_SPACE'
    #During the introspection phase, it is checked if either Matlab or
    #octave are installed and available trough the system PATH variable
    #If so, they will be used for some analyzers
    KEY_MATLAB_EXECUTABLE = 'MATLAB_EXECUTABLE'
    KEY_IP = 'SERVER_IP'
    KEY_PORT = 'WEB_SERVER_PORT'
    KEY_PORT_MPLH5 = 'MPLH5_SERVER_PORT'
    KEY_URL_WEB = 'URL_WEB'
    KEY_URL_MPLH5 = 'URL_MPLH5'
    KEY_SELECTED_DB = 'SELECTED_DB'
    KEY_DB_URL = 'URL_VALUE'
    KEY_URL_VERSION = 'URL_TVB_VERSION'
    KEY_CLUSTER = 'DEPLOY_CLUSTER'
    KEY_MAX_THREAD_NR = 'MAXIMUM_NR_OF_THREADS'
    KEY_MAX_RANGE_NR = 'MAXIMUM_NR_OF_OPS_IN_RANGE'
    KEY_MAX_NR_SURFACE_VERTEX = 'MAXIMUM_NR_OF_VERTICES_ON_SURFACE'
    KEY_LAST_CHECKED_FILE_VERSION = 'LAST_CHECKED_FILE_VERSION'
    KEY_LAST_CHECKED_CODE_VERSION = 'LAST_CHECKED_CODE_VERSION'
    KEY_FILE_STORAGE_UPDATE_STATUS = 'FILE_STORAGE_UPDATE_STATUS'
    # Keep a mapping of how the python executable will look on different os versions
    PYTHON_EXE_MAPPING = {'windows': 'python.exe',
                          'linux': 'python',
                          'macos': 'python'}


    @staticmethod
    def is_development():
        """
        Return True when TVB  is used with Python installed natively.
        """
        tvb_root = os.path.dirname(BaseProfile.CURRENT_DIR)
        return (os.path.exists(os.path.join(tvb_root, 'AUTHORS'))
                and os.path.exists(os.path.join(os.path.dirname(tvb_root), 'third_party_licenses'))
                and os.path.exists(os.path.join(os.path.dirname(tvb_root), 'externals'))
                and os.path.exists(os.path.join(os.path.dirname(tvb_root), 'tvb_documentation')))


    def is_windows(self):
        """
        Return True if current run is not development and is running on Windows.
        """
        return platform.startswith('win') and not self.is_development()


    def is_linux(self):
        """ 
        Return True if current run is not development and is running on Linux.
        """
        return not (platform.startswith('win') or platform == 'darwin' or self.is_development())


    def is_mac(self):
        """
        Return True if current run is not development and is running on Mac OS X
        """
        return platform == 'darwin' and not self.is_development()


    def get_python_exe_name(self):
        """ Returns the name of the python executable depending on the specific OS """
        if platform.startswith('win'):
            return self.PYTHON_EXE_MAPPING['windows']
        elif platform == 'darwin':
            return self.PYTHON_EXE_MAPPING['macos']
        else:
            return self.PYTHON_EXE_MAPPING['linux']


    def get_library_folder(self):
        """ Return top level library folder """
        if self.is_windows():
            return os.path.dirname(os.path.dirname(sys.executable))
        if self.is_mac():
            return os.path.dirname(self.BIN_FOLDER)
        if self.is_linux():
            return os.path.dirname(os.path.dirname(sys.executable))
        

    def get_python_path(self):
        """Get Python path, based on running options."""

        if self.is_development():
            python_path = 'python'
        elif self.is_windows():
            python_path = os.path.join(os.path.dirname(FrameworkSettings.BIN_FOLDER), 'exe', self.get_python_exe_name())
        elif self.is_mac():
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(FrameworkSettings.BIN_FOLDER))))
            python_path = os.path.join(root_dir, 'MacOS', self.get_python_exe_name())
        elif self.is_linux():
            python_path = os.path.join(os.path.dirname(FrameworkSettings.BIN_FOLDER), 'exe', self.get_python_exe_name())
        else:
            python_path = 'python'

        try:
            # check if file actually exists
            os.stat(python_path)
            return python_path
        except:
            # otherwise best guess is the current interpreter!
            return sys.executable



class DevelopmentProfile(BaseProfile):
    """
    Custom settings for development profile.
    """

    LOGGER_CONFIG_FILE_NAME = "dev_logger_config.conf"
    TRADE_CRASH_SAFETY_FOR_SPEED = False



class TestSQLiteProfile(BaseProfile):
    """
    Defines settings for running tests on an SQLite database.
    """
    #Use a different configuration file, to make it possible to run multiple instances in the same time
    TVB_CONFIG_FILE = os.path.expanduser(os.path.join("~", '.test.tvb.configuration'))

    RENDER_HTML = False
    TRADE_CRASH_SAFETY_FOR_SPEED = True
    DEFAULT_STORAGE = os.path.expanduser(os.path.join('~', 'TVB_TEST'))


    @ClassProperty
    @staticmethod
    @settings_loaded()
    def TVB_STORAGE():
        """Root folder for all projects and users."""
        default = FrameworkSettings.DEFAULT_STORAGE
        current_storage = FrameworkSettings.get_attribute(FrameworkSettings.KEY_STORAGE, default)

        if not os.path.exists(current_storage):
            os.makedirs(current_storage)
        return current_storage


    @ClassProperty
    @staticmethod
    def DB_URL():
        """Used DB url: IP,PORT. The DB  needs to be created in advance."""
        db_url = os.path.join(TestSQLiteProfile.TVB_STORAGE, "tvb-database.db")
        return 'sqlite:///' + db_url


    @ClassProperty
    @staticmethod
    @settings_loaded()
    def SELECTED_DB():
        """The selected DB will be SQLite."""
        return 'sqlite'


    @ClassProperty
    @staticmethod
    @settings_loaded()
    def CODE_CHECKED_TO_VERSION():
        return sys.maxint


    @ClassProperty
    @staticmethod
    def SVN_VERSION():
        """Current SVN version in the package running now."""
        return 1


    @ClassProperty
    @staticmethod
    def TVB_LOG_FOLDER():
        """
        Represents a folder, where all log files are stored.
        For tests we will place them in the workspace, to have them visible from Hudson.
        """
        tmp_path = os.path.join(FrameworkSettings.BIN_FOLDER, "TEST_OUTPUT")
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        return tmp_path


    @ClassProperty
    @staticmethod
    @settings_loaded()
    def MAX_THREADS_NUMBER():
        """Maximum number of threads in the pool of simulations range."""
        return FrameworkSettings.get_attribute(FrameworkSettings.KEY_MAX_THREAD_NR, 2, int)



class TestPostgresProfile(TestSQLiteProfile):
    """
    Defines settings for running tests on a Postgres database.
    """
    RENDER_HTML = False


    @ClassProperty
    @staticmethod
    def DB_URL():
        """Used DB url: IP,PORT. The DB  needs to be created in advance."""
        default = 'postgresql+psycopg2://postgres:root@127.0.0.1:5432/tvb-test?user=postgres&password=postgres'
        return default


    @ClassProperty
    @staticmethod
    @settings_loaded()
    def SELECTED_DB():
        """The selected DB will be SQLite."""
        return 'postgres'



class DeploymentProfile(BaseProfile):
    """
    Profile for packages deployed already.
    """


    @classmethod
    def initialize_profile(cls):
        """
        This method is called at the time the config.py module is first imported. Any specific
        initializations for the profile should be placed here.
        """
        #We want to disable warnings we get from sqlalchemy for traited attributes
        #when we are in deployment mode.
        import warnings
        from sqlalchemy import exc as sa_exc

        warnings.simplefilter("ignore", category=sa_exc.SAWarning)
        cfg = FrameworkSettings()
        data_path = cfg.get_library_folder()
        if cfg.is_windows():
            # Add root folder as first in PYTHONPATH so we can find TVB there in case of GIT contributors
            new_python_path = cfg.TVB_PATH + os.pathsep
            new_python_path += data_path + os.pathsep + os.path.join(data_path, 'lib-tk')
            os.environ['PYTHONPATH'] = new_python_path
            os.environ['PATH'] = data_path + os.pathsep + os.environ.get('PATH', '')
            setup_tk_tcl_environ(data_path)

        if cfg.is_mac():
            # MacOS package structure is in the form:
            # Contents/Resorces/lib/python2.7/tvb . PYTHONPATH needs to be set
            # at the level Contents/Resources/lib/python2.7/ and the root path
            # from where to start looking for TK and TCL up to Contents/
            tcl_root = os.path.split(os.path.split(os.path.split(data_path)[0])[0])[0]
            setup_tk_tcl_environ(tcl_root)

            #Add root folder as first in PYTHONPATH so we can find TVB there in case of GIT contributors
            new_python_path = data_path + os.pathsep + os.path.join(data_path, 'site-packages.zip')
            new_python_path += os.pathsep + os.path.join(data_path, 'lib-dynload')
            new_python_path = cfg.TVB_PATH + os.pathsep + new_python_path
            os.environ['PYTHONPATH'] = new_python_path

        if cfg.is_linux():
            # Note that for the Linux package some environment variables like LD_LIBRARY_PATH,
            # LD_RUN_PATH, PYTHONPATH and PYTHONHOME are set also in the startup scripts.
            # Add root folder as first in PYTHONPATH so we can find TVB there in case of GIT contributors
            new_python_path = cfg.TVB_PATH + os.pathsep + data_path
            new_python_path += os.pathsep + os.path.join(data_path, 'lib-tk')
            os.environ['PYTHONPATH'] = new_python_path
            setup_tk_tcl_environ(data_path)


    @ClassProperty
    @staticmethod
    def CHERRYPY_CONFIGURATION():

        inside_static_folder = os.path.join(FrameworkSettings.EXTERNALS_FOLDER_PARENT, 'tvb')

        default_configuration = BaseProfile.CHERRYPY_CONFIGURATION
        default_configuration['/statichelp']['tools.staticdir.root'] = inside_static_folder
        return default_configuration


class ConsoleProfile(DeploymentProfile):
    """
    Profile which allows you to work with tvb in console mode.
    """
    TRAITS_CONFIGURATION = copy(BaseProfile.TRAITS_CONFIGURATION)
    TRAITS_CONFIGURATION.use_storage = True



if TvbProfile.CURRENT_SELECTED_PROFILE == TvbProfile.TEST_POSTGRES_PROFILE:
    FrameworkSettings = TestPostgresProfile

elif TvbProfile.CURRENT_SELECTED_PROFILE == TvbProfile.TEST_SQLITE_PROFILE:
    FrameworkSettings = TestSQLiteProfile

elif TvbProfile.CURRENT_SELECTED_PROFILE == TvbProfile.CONSOLE_PROFILE:
    FrameworkSettings = ConsoleProfile

elif BaseProfile.is_development() or TvbProfile.CURRENT_SELECTED_PROFILE == TvbProfile.DEVELOPMENT_PROFILE:
    FrameworkSettings = DevelopmentProfile

else:
    FrameworkSettings = DeploymentProfile

FrameworkSettings.initialize_profile()
