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
Manages reading and writing settings in file

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>

"""

import os

# File keys
KEY_ADMIN_DISPLAY_NAME = 'ADMINISTRATOR_DISPLAY_NAME'
KEY_ADMIN_NAME = 'ADMINISTRATOR_NAME'
KEY_ADMIN_PWD = 'ADMINISTRATOR_PASSWORD'
KEY_ADMIN_EMAIL = 'ADMINISTRATOR_EMAIL'
KEY_ADMINS_GROUP = 'ADMINISTRATORS_GROUP'
KEY_STORAGE = 'TVB_STORAGE'
KEY_UPLOAD_PRIVATE_KEY_PATH = 'UPLOAD_KEY_PATH'
KEY_KC_CONFIGURATION = 'KEYCLOAK_CONFIGURATION'
KEY_KC_WEB_CONFIGURATION = 'KEYCLOAK_WEB_CONFIGURATION'
KEY_ENABLE_KC_LOGIN = 'ENABLE_KEYCLOAK_LOGIN'
KEY_MAX_DISK_SPACE_USR = 'USR_DISK_SPACE'
# During the introspection phase, it is checked if either Matlab or
# octave are installed and available trough the system PATH variable
# If so, they will be used for some analyzers
KEY_MATLAB_EXECUTABLE = 'MATLAB_EXECUTABLE'
KEY_IP = 'SERVER_IP'
KEY_PORT = 'WEB_SERVER_PORT'
KEY_SELECTED_DB = 'SELECTED_DB'
KEY_DB_URL = 'URL_VALUE'
KEY_URL_VERSION = 'URL_TVB_VERSION'
KEY_CLUSTER = 'DEPLOY_CLUSTER'
KEY_CLUSTER_SCHEDULER = 'CLUSTER_SCHEDULER'
KEY_HPC = 'RUN_ON_HPC'
KEY_HPC_INTERVAL = 'HPC_SYNC_BACKGROUND_JOB_INTERVAL'
KEY_CRYPT_PASSDIR = 'CRYPT_PASSDIR'
KEY_CRYPT_DATADIR = 'CRYPT_DATADIR'
KEY_HPC_COMPUTE_SITE = 'HPC_COMPUTE_SITE'
KEY_MAX_THREAD_NR = 'MAXIMUM_NR_OF_THREADS'
KEY_MAX_RANGE_NR = 'MAXIMUM_NR_OF_OPS_IN_RANGE'
KEY_MAX_NR_SURFACE_VERTEX = 'MAXIMUM_NR_OF_VERTICES_ON_SURFACE'
KEY_LAST_CHECKED_FILE_VERSION = 'LAST_CHECKED_FILE_VERSION'
KEY_LAST_CHECKED_CODE_VERSION = 'LAST_CHECKED_CODE_VERSION'
KEY_FILE_STORAGE_UPDATE_STATUS = 'FILE_STORAGE_UPDATE_STATUS'
KEY_TRACE_USER_ACTIONS = "TRACE_USER_ACTIONS"
KEY_ENCRYPT_STORAGE = "ENCRYPT_STORAGE"
KEY_DECRYPT_PATH = "DECRYPT_PATH"


class SettingsManager(object):
    def __init__(self, config_file_location):
        self.config_file_location = config_file_location
        self.stored_settings = self._read_config_file()

    def _read_config_file(self):
        """
        Get data from the configurations file in the form of a dictionary.
        Return empty dictionary if file not present.
        """
        if not os.path.exists(self.config_file_location):
            return {}

        config_dict = {}
        with open(self.config_file_location, 'r') as cfg_file:
            data = cfg_file.read()
            entries = [line for line in data.split('\n') if not line.startswith('#') and len(line.strip()) > 0]
            for one_entry in entries:
                name, value = one_entry.split('=', 1)
                config_dict[name] = value
        return config_dict

    def _store_config_file(self, config_dict):

        with open(self.config_file_location, 'w') as file_writer:
            for key in config_dict:
                file_writer.write(key + '=' + str(config_dict[key]) + '\n')

        self.stored_settings = self._read_config_file()

    def add_entries_to_config_file(self, input_data):
        """
        Add to the dictionary of settings already existent in the settings file.

        :param input_data: A dictionary of pairs that need to be added to the config file.
        """
        config_dict = self._read_config_file()
        if config_dict is None:
            config_dict = {}

        for entry in input_data:
            config_dict[entry] = input_data[entry]

        self._store_config_file(config_dict)

    def delete_entries_from_config_file(self, data_to_be_deleted):
        """
        Delete from the dictionary of settings existent in the settings file.

        :param data_to_be_deleted: A list of attributes that need to be deleted from the config file.
        """
        config_dict = self._read_config_file()

        for entry in data_to_be_deleted:
            if entry in config_dict:
                config_dict.pop(entry)

        self._store_config_file(config_dict)

    def write_config_data(self, config_dict):
        """
        Overwrite anything already existent in the config file
        """
        with open(self.config_file_location, 'w') as file_writer:
            for key in config_dict:
                file_writer.write(key + '=' + str(config_dict[key]) + '\n')

        self.stored_settings = self._read_config_file()

    def get_attribute(self, key, default=None, dtype=str):
        """
        Get a cfg attribute that could also be found in the settings file.
        """
        try:
            if key in self.stored_settings:
                return dtype(self.stored_settings[key])
        except ValueError:
            # Invalid convert operation.
            return default
        return default

    def is_first_run(self):
        return self.stored_settings is None or len(self.stored_settings) <= 2
