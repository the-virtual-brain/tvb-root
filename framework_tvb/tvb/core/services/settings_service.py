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
Service layer for saving/editing TVB settings.

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""
import os
import sys
import shutil
from hashlib import md5
from sqlalchemy import create_engine
from tvb.basic.config.settings import TVBSettings as cfg
from tvb.basic.logger.builder import get_logger
from tvb.core.utils import get_matlab_executable
from tvb.core.services.exceptions import InvalidSettingsException



class SettingsService():
    """
    Handle all TVB Setting related problems, at the service level.
    """

    KEY_ADMIN_NAME = cfg.KEY_ADMIN_NAME
    KEY_ADMIN_PWD = cfg.KEY_ADMIN_PWD
    KEY_ADMIN_EMAIL = cfg.KEY_ADMIN_EMAIL
    KEY_STORAGE = cfg.KEY_STORAGE
    KEY_MAX_DISK_SPACE_USR = cfg.KEY_MAX_DISK_SPACE_USR
    KEY_MATLAB_EXECUTABLE = cfg.KEY_MATLAB_EXECUTABLE
    KEY_PORT = cfg.KEY_PORT
    KEY_PORT_MPLH5 = cfg.KEY_PORT_MPLH5
    KEY_URL_WEB = cfg.KEY_URL_WEB
    KEY_URL_MPLH5 = cfg.KEY_URL_MPLH5
    KEY_SELECTED_DB = cfg.KEY_SELECTED_DB
    KEY_DB_URL = cfg.KEY_DB_URL
    KEY_CLUSTER = cfg.KEY_CLUSTER
    KEY_MAX_NR_THREADS = cfg.KEY_MAX_THREAD_NR
    KEY_MAX_RANGE = cfg.KEY_MAX_RANGE_NR
    KEY_MAX_NR_SURFACE_VERTEX = cfg.KEY_MAX_NR_SURFACE_VERTEX

    #Display order for the keys. None means a separator/new line will be added
    KEYS_DISPLAY_ORDER = [KEY_ADMIN_NAME, KEY_ADMIN_PWD, KEY_ADMIN_EMAIL, None,
                          KEY_STORAGE, KEY_MAX_DISK_SPACE_USR, KEY_MATLAB_EXECUTABLE, KEY_SELECTED_DB, KEY_DB_URL, None,
                          KEY_PORT, KEY_PORT_MPLH5, KEY_URL_WEB, KEY_URL_MPLH5, None,
                          KEY_CLUSTER, KEY_MAX_NR_THREADS, KEY_MAX_RANGE, KEY_MAX_NR_SURFACE_VERTEX]


    def __init__(self):
        self.logger = get_logger(__name__)
        self.configurable_keys = {
            self.KEY_STORAGE: {'label': 'Root folder for all projects',
                               'value': cfg.TVB_STORAGE if not self.is_first_run() else cfg.DEFAULT_STORAGE,
                               'readonly': not self.is_first_run(), 'type': 'text'},
            self.KEY_MAX_DISK_SPACE_USR: {'label': 'Max hard disk space per user (MBytes)',
                                          'value': cfg.MAX_DISK_SPACE / 2 ** 10, 'type': 'text'},
            self.KEY_MATLAB_EXECUTABLE: {'label': 'Optional Matlab or Octave path', 'type': 'text',
                                         'value': cfg.MATLAB_EXECUTABLE or get_matlab_executable() or '',
                                         'description': 'Some analyzers will not be available when '
                                                        'matlab/octave are not found'},
            self.KEY_SELECTED_DB: {'label': 'Select one DB engine', 'value': cfg.SELECTED_DB,
                                   'type': 'select', 'readonly': not self.is_first_run(),
                                   'options': cfg.ACEEPTED_DBS},
            self.KEY_DB_URL: {'label': "DB connection URL", 'value': cfg.ACEEPTED_DBS[cfg.SELECTED_DB],
                              'type': 'text', 'readonly': cfg.SELECTED_DB == 'sqlite'},

            self.KEY_PORT: {'label': 'Port to run Cherrypy on',
                            'value': cfg.WEB_SERVER_PORT, 'dtype': 'primitive', 'type': 'text'},
            self.KEY_PORT_MPLH5: {'label': 'Port to run Matplotlib on',
                                  'value': cfg.MPLH5_SERVER_PORT, 'type': 'text', 'dtype': 'primitive'},
            self.KEY_URL_WEB: {'label': 'URL for accessing web',
                               'value': cfg.BASE_URL, 'type': 'text', 'dtype': 'primitive'},
            self.KEY_URL_MPLH5: {'label': 'URL for accessing MPLH5 visualizers',
                                 'value': cfg.MPLH5_SERVER_URL, 'type': 'text', 'dtype': 'primitive'},

            self.KEY_MAX_NR_THREADS: {'label': 'Maximum no. of threads for local installations',
                                      'value': cfg.MAX_THREADS_NUMBER, 'type': 'text', 'dtype': 'primitive'},
            self.KEY_MAX_RANGE: {'label': 'Maximum no. of operations in one PSE',
                                 'description': "Parameters Space Exploration (PSE) maximum number of operations",
                                 'value': cfg.MAX_RANGE_NUMBER, 'type': 'text', 'dtype': 'primitive'},
            self.KEY_MAX_NR_SURFACE_VERTEX: {'label': 'Maximum no. of vertices in a surface', 'type': 'text',
                                             'value': cfg.MAX_SURFACE_VERTICES_NUMBER, 'dtype': 'primitive'},
            self.KEY_CLUSTER: {'label': 'Deploy on cluster', 'value': cfg.DEPLOY_CLUSTER,
                               'description': 'Check this only if on the web-server machine OARSUB command is enabled.',
                               'dtype': 'primitive', 'type': 'boolean'},
            self.KEY_ADMIN_NAME: {'label': 'Administrator User Name', 'value': cfg.ADMINISTRATOR_NAME,
                                  'type': 'text', 'readonly': not self.is_first_run(),
                                  'description': ('Password and Email can be edited after first run, '
                                                  'from the profile page directly.')},
            self.KEY_ADMIN_PWD: {'label': 'Password',
                                 'value': cfg.ADMINISTRATOR_BLANK_PWD if self.is_first_run()
                                 else cfg.ADMINISTRATOR_PASSWORD,
                                 'type': 'password', 'readonly': not self.is_first_run()},
            self.KEY_ADMIN_EMAIL: {'label': 'Administrator Email', 'value': cfg.ADMINISTRATOR_EMAIL,
                                   'readonly': not self.is_first_run(), 'type': 'text'}}


    def check_db_url(self, url):
        """Validate DB URL, that a connection can be done."""
        try:
            engine = create_engine(url)
            connection = engine.connect()
            connection.close()
        except Exception, excep:
            self.logger.exception(excep)
            raise InvalidSettingsException('Could not connect to DB! ' 'Invalid URL:' + str(url))


    @staticmethod
    def is_first_run():
        """
        Check if this is the first time TVB was started.
        """
        file_dict = cfg.read_config_file()
        return file_dict is None or len(file_dict) <= 2


    @staticmethod
    def get_disk_free_space(storage_path):
        """
        :returns: the available HDD space in KB in TVB_STORAGE folder.
        """
        if sys.platform.startswith('win'):
            import ctypes
            drive = unicode(storage_path.split(':')[0] + ':')
            freeuser = ctypes.c_int64()
            total = ctypes.c_int64()
            free = ctypes.c_int64()
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(drive, ctypes.byref(freeuser),
                                                       ctypes.byref(total), ctypes.byref(free))
            bytes_value = freeuser.value
        else:
            mem_stat = os.statvfs(storage_path)
            bytes_value = mem_stat.f_frsize * mem_stat.f_bavail
            ## Occupied memory would be:
            #bytes_value = mem_stat.f_bsize * mem_stat.f_bavail
        return bytes_value / 2 ** 10


    def save_settings(self, **data):
        """
        Check if new settings are correct.  Make necessary changes, then save new data in configuration file.
        
        :returns: two boolean values
                    -there were any changes to the configuration;
                    -a reset should be performed on the TVB relaunch.
        """
        new_storage = data[self.KEY_STORAGE]
        previous_storage = cfg.TVB_STORAGE

        new_db = data[self.KEY_SELECTED_DB]
        previous_db = cfg.SELECTED_DB
        db_changed = new_db != previous_db
        storage_changed = new_storage != previous_storage

        matlab_exec = data[self.KEY_MATLAB_EXECUTABLE]
        if matlab_exec == 'None':
            data[self.KEY_MATLAB_EXECUTABLE] = ''
        #Storage changed but DB didn't, just copy TVB storage to new one.
        if storage_changed and not db_changed:
            if os.path.exists(new_storage):
                if os.access(new_storage, os.W_OK):
                    shutil.rmtree(new_storage)
                else:
                    raise InvalidSettingsException("No Write access on storage folder!!")
            shutil.copytree(previous_storage, new_storage)
            
        if not os.path.isdir(new_storage):
            os.makedirs(new_storage)
        max_space = data[self.KEY_MAX_DISK_SPACE_USR]
        available_mem_kb = SettingsService.get_disk_free_space(new_storage)
        kb_value = int(max_space) * 2 ** 10
        if not (0 < kb_value < available_mem_kb):
            raise InvalidSettingsException(
                "Not enough disk space. There is a maximum of %d MB available "
                "on this disk or partition." % (available_mem_kb >> 10, )
            )
        data[self.KEY_MAX_DISK_SPACE_USR] = kb_value

        #Save data to file, all while checking if any data has changed
        first_run = self.is_first_run()
        if first_run:
            data[cfg.KEY_LAST_CHECKED_FILE_VERSION] = cfg.DATA_VERSION
            data[cfg.KEY_LAST_CHECKED_CODE_VERSION] = cfg.SVN_VERSION
            file_data = data
            if self.KEY_ADMIN_PWD in data:
                data[self.KEY_ADMIN_PWD] = md5(data[self.KEY_ADMIN_PWD]).hexdigest()
            anything_changed = True
        else:
            file_data = cfg.read_config_file()
            anything_changed = False
            for key in file_data:
                if key in data and str(data[key]) != str(file_data[key]):
                    anything_changed = True
                    file_data[key] = data[key]
            if db_changed:
                file_data[self.KEY_DB_URL] = cfg.DB_URL
            for key in data:
                if key not in file_data:
                    anything_changed = True
                    file_data[key] = data[key]
        # Write in file new data
        if anything_changed:
            with open(cfg.TVB_CONFIG_FILE, 'w') as file_writer:
                for key in file_data:
                    file_writer.write(key + '=' + str(file_data[key]) + '\n')
            os.chmod(cfg.TVB_CONFIG_FILE, 0644)
        return anything_changed, first_run or db_changed

