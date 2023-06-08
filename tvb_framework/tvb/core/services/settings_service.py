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
Service layer for saving/editing TVB settings.

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import os
import shutil
import sys
from sqlalchemy import create_engine
from tvb.basic.config import stored
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.core.services.exceptions import InvalidSettingsException
from tvb.core.utils import hash_password


class SettingsService(object):
    """
    Handle all TVB Setting related problems, at the service level.
    """

    KEY_ADMIN_NAME = stored.KEY_ADMIN_NAME
    KEY_ADMIN_DISPLAY_NAME = stored.KEY_ADMIN_DISPLAY_NAME
    KEY_ADMIN_PWD = stored.KEY_ADMIN_PWD
    KEY_ADMIN_EMAIL = stored.KEY_ADMIN_EMAIL
    KEY_STORAGE = stored.KEY_STORAGE
    KEY_KC_CONFIG = stored.KEY_KC_CONFIGURATION
    KEY_KC_WEB_CONFIG = stored.KEY_KC_WEB_CONFIGURATION
    KEY_ENABLE_KC_LOGIN = stored.KEY_ENABLE_KC_LOGIN
    KEY_MAX_DISK_SPACE_USR = stored.KEY_MAX_DISK_SPACE_USR
    KEY_PORT = stored.KEY_PORT
    KEY_SELECTED_DB = stored.KEY_SELECTED_DB
    KEY_DB_URL = stored.KEY_DB_URL
    KEY_CLUSTER = stored.KEY_CLUSTER
    KEY_CLUSTER_SCHEDULER = stored.KEY_CLUSTER_SCHEDULER
    KEY_MAX_NR_THREADS = stored.KEY_MAX_THREAD_NR
    KEY_MAX_RANGE = stored.KEY_MAX_RANGE_NR
    KEY_MAX_NR_SURFACE_VERTEX = stored.KEY_MAX_NR_SURFACE_VERTEX

    # Display order for the keys. None means a separator/new line will be added
    KEYS_DISPLAY_ORDER = [KEY_ADMIN_DISPLAY_NAME, KEY_ADMIN_NAME, KEY_ADMIN_PWD, KEY_ADMIN_EMAIL, None,
                          KEY_KC_CONFIG, KEY_ENABLE_KC_LOGIN, KEY_KC_WEB_CONFIG, None, KEY_STORAGE,
                          KEY_MAX_DISK_SPACE_USR, KEY_SELECTED_DB,
                          KEY_DB_URL, None,
                          KEY_PORT, None,
                          KEY_CLUSTER, KEY_CLUSTER_SCHEDULER,
                          KEY_MAX_NR_THREADS, KEY_MAX_RANGE, KEY_MAX_NR_SURFACE_VERTEX]

    def __init__(self):
        self.logger = get_logger(__name__)
        first_run = TvbProfile.is_first_run()
        storage = TvbProfile.current.TVB_STORAGE if not first_run else TvbProfile.current.DEFAULT_STORAGE
        self.configurable_keys = {
            self.KEY_KC_CONFIG: {'label': 'Rest API Keycloak configuration file',
                                 'value': TvbProfile.current.KEYCLOAK_CONFIG,
                                 'readonly': False, 'type': 'text'},
            self.KEY_ENABLE_KC_LOGIN: {'label': 'Enable Keycloak login',
                                       'value': TvbProfile.current.KEYCLOAK_LOGIN_ENABLED,
                                       'readonly': False, 'type': 'boolean'},
            self.KEY_KC_WEB_CONFIG: {'label': 'Web Keycloak configuration file',
                                     'value': TvbProfile.current.KEYCLOAK_WEB_CONFIG,
                                     'readonly': False, 'type': 'text'},
            self.KEY_STORAGE: {'label': 'Root folder for all projects', 'value': storage,
                               'readonly': not first_run, 'type': 'text'},
            self.KEY_MAX_DISK_SPACE_USR: {'label': 'Max hard disk space per user (MBytes)',
                                          'value': int(TvbProfile.current.MAX_DISK_SPACE / 2 ** 10), 'type': 'text'},
            self.KEY_SELECTED_DB: {'label': 'Select one DB engine', 'value': TvbProfile.current.db.SELECTED_DB,
                                   'type': 'select', 'readonly': not first_run,
                                   'options': TvbProfile.current.db.ACEEPTED_DBS},
            self.KEY_DB_URL: {'label': "DB connection URL",
                              'value': TvbProfile.current.db.ACEEPTED_DBS[TvbProfile.current.db.SELECTED_DB],
                              'type': 'text', 'readonly': TvbProfile.current.db.SELECTED_DB == 'sqlite'},

            self.KEY_PORT: {'label': 'Port to run Cherrypy on',
                            'value': TvbProfile.current.web.SERVER_PORT, 'dtype': 'primitive', 'type': 'text'},

            self.KEY_MAX_NR_THREADS: {'label': 'Maximum no. of threads for local installations', 'type': 'text',
                                      'value': TvbProfile.current.MAX_THREADS_NUMBER, 'dtype': 'primitive'},
            self.KEY_MAX_RANGE: {'label': 'Maximum no. of operations in one PSE',
                                 'description': "Parameters Space Exploration (PSE) maximum number of operations",
                                 'value': TvbProfile.current.MAX_RANGE_NUMBER, 'type': 'text', 'dtype': 'primitive'},
            self.KEY_MAX_NR_SURFACE_VERTEX: {'label': 'Maximum no. of vertices in a surface',
                                             'type': 'text', 'dtype': 'primitive',
                                             'value': TvbProfile.current.MAX_SURFACE_VERTICES_NUMBER},
            self.KEY_CLUSTER: {'label': 'Deploy on cluster', 'value': TvbProfile.current.cluster.IS_DEPLOY,
                               'description': 'Check this only if on the web-server machine OARSUB command is enabled.',
                               'dtype': 'primitive', 'type': 'boolean'},
            self.KEY_CLUSTER_SCHEDULER: {'label': 'Cluster Scheduler', 'readonly': False,
                                         'value': TvbProfile.current.cluster.CLUSTER_SCHEDULER, 'type': 'select',
                                         'options': TvbProfile.current.cluster.ACCEPTED_SCHEDULERS},
            self.KEY_ADMIN_DISPLAY_NAME: {'label': 'Administrator Display Name',
                                          'value': TvbProfile.current.web.admin.ADMINISTRATOR_DISPLAY_NAME,
                                          'type': 'text', 'readonly': not first_run},
            self.KEY_ADMIN_NAME: {'label': 'Administrator User Name',
                                  'value': TvbProfile.current.web.admin.ADMINISTRATOR_NAME,
                                  'type': 'text', 'readonly': not first_run,
                                  'description': ('Password and Email can be edited after first run, '
                                                  'from the profile page directly.')},
            self.KEY_ADMIN_PWD: {'label': 'Password',
                                 'value': TvbProfile.current.web.admin.ADMINISTRATOR_BLANK_PWD if first_run
                                 else TvbProfile.current.web.admin.ADMINISTRATOR_PASSWORD,
                                 'type': 'password', 'readonly': not first_run},
            self.KEY_ADMIN_EMAIL: {'label': 'Administrator Email',
                                   'value': TvbProfile.current.web.admin.ADMINISTRATOR_EMAIL,
                                   'readonly': not first_run, 'type': 'text'}}

    def check_db_url(self, url):
        """Validate DB URL, that a connection can be done."""
        try:
            engine = create_engine(url)
            connection = engine.connect()
            connection.close()
        except Exception as excep:
            self.logger.exception(excep)
            raise InvalidSettingsException('Could not connect to DB! Invalid URL:' + str(url))

    @staticmethod
    def get_disk_free_space(storage_path):
        """
        :returns: the available HDD space in KB in TVB_STORAGE folder.
        """
        if sys.platform.startswith('win'):
            import ctypes
            storage_path = os.path.abspath(storage_path)
            drive = storage_path.split(':')[0] + ':'
            freeuser = ctypes.c_int64()
            total = ctypes.c_int64()
            free = ctypes.c_int64()
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(drive, ctypes.byref(freeuser),
                                                       ctypes.byref(total), ctypes.byref(free))
            bytes_value = freeuser.value
        else:
            mem_stat = os.statvfs(storage_path)
            bytes_value = mem_stat.f_frsize * mem_stat.f_bavail
            # Occupied memory would be:
            # bytes_value = mem_stat.f_bsize * mem_stat.f_bavail
        return bytes_value / 2 ** 10

    def save_settings(self, **data):
        """
        Check if new settings are correct.  Make necessary changes, then save new data in configuration file.

        :returns: two boolean values
                    -there were any changes to the configuration;
                    -a reset should be performed on the TVB relaunch.
        """
        new_storage = data[self.KEY_STORAGE]
        previous_storage = TvbProfile.current.TVB_STORAGE

        new_db = data[self.KEY_SELECTED_DB]
        previous_db = TvbProfile.current.db.SELECTED_DB
        db_changed = new_db != previous_db
        storage_changed = new_storage != previous_storage

        if TvbProfile.is_first_run() or storage_changed:
            self._check_tvb_folder(new_storage)

        # Storage changed but DB didn't, just copy TVB storage to new one.
        if storage_changed and not db_changed:
            shutil.copytree(previous_storage, new_storage)

        if not os.path.isdir(new_storage):
            os.makedirs(new_storage)
        max_space = data[self.KEY_MAX_DISK_SPACE_USR]
        available_mem_kb = SettingsService.get_disk_free_space(new_storage)
        kb_value = int(max_space) * 2 ** 10
        if not (0 < kb_value < available_mem_kb):
            raise InvalidSettingsException("Not enough disk space. There is a maximum of %d MB available on this disk "
                                           "or partition. Wanted %d" % (available_mem_kb / (2 ** 10), max_space))
        data[self.KEY_MAX_DISK_SPACE_USR] = kb_value

        # Save data to file, all while checking if any data has changed
        first_run = TvbProfile.is_first_run()
        if first_run:
            data[stored.KEY_LAST_CHECKED_FILE_VERSION] = TvbProfile.current.version.DATA_VERSION
            data[stored.KEY_LAST_CHECKED_CODE_VERSION] = TvbProfile.current.version.REVISION_NUMBER
            data[stored.KEY_FILE_STORAGE] = TvbProfile.current.file_storage
            file_data = data
            if self.KEY_ADMIN_PWD in data:
                data[self.KEY_ADMIN_PWD] = hash_password(data[self.KEY_ADMIN_PWD])
            anything_changed = True
        else:
            file_data = TvbProfile.current.manager.stored_settings
            anything_changed = False
            for key in file_data:
                if key in data and str(data[key]) != str(file_data[key]):
                    anything_changed = True
                    file_data[key] = data[key]
            if db_changed:
                file_data[self.KEY_DB_URL] = TvbProfile.current.db.DB_URL
            for key in data:
                if key not in file_data:
                    anything_changed = True
                    file_data[key] = data[key]
        # Write in file new data
        if anything_changed:
            TvbProfile.current.manager.write_config_data(file_data)
            os.chmod(TvbProfile.current.TVB_CONFIG_FILE, 0o644)
        return anything_changed, first_run or db_changed

    def _check_tvb_folder(self, storage_path):
        """
        Check if the storage folder is compatible (should be an empty or new folder, with rights to write inside).
        """
        if not os.path.exists(storage_path):
            return True

        if not os.path.isdir(storage_path):
            raise InvalidSettingsException('TVB Storage should be a folder')

        if not os.access(storage_path, os.W_OK):
            raise InvalidSettingsException('TVB Storage folder should have write access for tvb process')

        list_content = os.listdir(storage_path)
        if "TEMP" in list_content:
            list_content.remove("TEMP")
        if len(list_content) > 0:
            raise InvalidSettingsException(
                'TVB Storage should be empty, please set another folder than {}.'.format(storage_path))

        return True
