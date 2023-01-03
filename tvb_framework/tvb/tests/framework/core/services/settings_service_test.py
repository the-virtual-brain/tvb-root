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
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import os

import pytest
from tvb.basic.profile import TvbProfile
from tvb.config.profile_settings import TestSQLiteProfile
from tvb.core.services.settings_service import SettingsService, InvalidSettingsException
from tvb.storage.storage_interface import StorageInterface
from tvb.tests.framework.core.base_testcase import BaseTestCase

TEST_CONFIG_FILE = os.path.expanduser(os.path.join("~", 'tvb.tests.framework.configuration'))


class TestSettingsService(BaseTestCase):
    """
    This class contains tests for the tvb.core.services.settings_service module.
    """
    TEST_SETTINGS = {SettingsService.KEY_ADMIN_NAME: 'test_name',
                     SettingsService.KEY_ADMIN_EMAIL: 'my@yahoo.com',
                     SettingsService.KEY_PORT: 8081,
                     SettingsService.KEY_MAX_DISK_SPACE_USR: 2 ** 8}

    def setup_method(self):
        """
        Prepare the usage of a different config file for this class only.
        """
        StorageInterface.remove_files([TEST_CONFIG_FILE, TestSQLiteProfile.DEFAULT_STORAGE])

        self.old_config_file = TvbProfile.current.TVB_CONFIG_FILE
        TvbProfile.current.__class__.TVB_CONFIG_FILE = TEST_CONFIG_FILE
        TvbProfile._build_profile_class(TvbProfile.CURRENT_PROFILE_NAME)
        self.settings_service = SettingsService()

    def teardown_method(self):
        """
        Restore configuration file
        """
        if os.path.exists(TEST_CONFIG_FILE):
            os.remove(TEST_CONFIG_FILE)

        TvbProfile.current.__class__.TVB_CONFIG_FILE = self.old_config_file
        TvbProfile._build_profile_class(TvbProfile.CURRENT_PROFILE_NAME)

    def test_check_db_url_invalid(self):
        """
        Make sure a proper exception is raised in case an invalid database url is passed.
        """
        with pytest.raises(InvalidSettingsException):
            self.settings_service.check_db_url("this-url-should-be-invalid")

    def test_get_free_disk_space(self):
        """
        Check that no exception is raised during the query for free disk space.
        Also do a check that returned value is greater than 0.
        Not most precise check but other does not seem possible so far.
        """
        disk_space = self.settings_service.get_disk_free_space(TvbProfile.current.TVB_STORAGE)
        assert disk_space > 0, "Disk space should never be negative."

    def test_first_run_save(self):
        """
        Check that before setting something, all flags are pointing towards empty.
        After storing some configurations, check that flags are changed.
        """
        initial_configurations = self.settings_service.configurable_keys
        first_run = TvbProfile.is_first_run()
        assert first_run, "Invalid First_Run flag!!"
        assert not os.path.exists(TEST_CONFIG_FILE)
        assert len(TvbProfile.current.manager.stored_settings) == 0

        to_store_data = {key: value['value'] for key, value in initial_configurations.items()}
        for key, value in self.TEST_SETTINGS.items():
            to_store_data[key] = value
        _, shoud_reset = self.settings_service.save_settings(**to_store_data)

        assert shoud_reset
        first_run = TvbProfile.is_first_run()
        assert not first_run, "Invalid First_Run flag!!"
        assert os.path.exists(TEST_CONFIG_FILE)
        assert not len(TvbProfile.current.manager.stored_settings) == 0

    def test_read_stored_settings(self):
        """
        Test to see that keys from the configuration dict is updated with
        the value from the configuration file after store.
        """
        initial_configurations = self.settings_service.configurable_keys
        to_store_data = {key: value['value'] for key, value in initial_configurations.items()}
        for key, value in self.TEST_SETTINGS.items():
            to_store_data[key] = value

        is_changed, shoud_reset = self.settings_service.save_settings(**to_store_data)
        assert shoud_reset and is_changed

        # enforce keys to get repopulated:
        TvbProfile._build_profile_class(TvbProfile.CURRENT_PROFILE_NAME)
        self.settings_service = SettingsService()

        updated_configurations = self.settings_service.configurable_keys
        for key, value in updated_configurations.items():
            if key in self.TEST_SETTINGS:
                assert self.TEST_SETTINGS[key] == value['value']
            elif key == SettingsService.KEY_ADMIN_PWD:
                assert TvbProfile.current.web.admin.ADMINISTRATOR_PASSWORD == value['value']
                assert TvbProfile.current.web.admin.ADMINISTRATOR_BLANK_PWD == initial_configurations[key]['value']
            else:
                assert initial_configurations[key]['value'] == value['value']

    def test_update_settings(self):
        """
        Test update of settings: correct flags should be returned, and check storage folder renamed
        """
        # 1. save on empty config-file:
        to_store_data = {key: value['value'] for key, value in self.settings_service.configurable_keys.items()}
        for key, value in self.TEST_SETTINGS.items():
            to_store_data[key] = value

        is_changed, shoud_reset = self.settings_service.save_settings(**to_store_data)
        assert shoud_reset and is_changed

        # 2. Reload and save with the same values (is_changed expected to be False)
        TvbProfile._build_profile_class(TvbProfile.CURRENT_PROFILE_NAME)
        self.settings_service = SettingsService()
        to_store_data = {key: value['value'] for key, value in self.settings_service.configurable_keys.items()}

        is_changed, shoud_reset = self.settings_service.save_settings(**to_store_data)
        assert not is_changed
        assert not shoud_reset

        # 3. Reload and check that changing TVB_STORAGE is done correctly
        TvbProfile._build_profile_class(TvbProfile.CURRENT_PROFILE_NAME)
        self.settings_service = SettingsService()
        to_store_data = {key: value['value'] for key, value in self.settings_service.configurable_keys.items()}
        to_store_data[SettingsService.KEY_STORAGE] = os.path.join(TvbProfile.current.TVB_STORAGE, 'RENAMED')

        # Write a test-file and check that it is moved
        file_writer = open(os.path.join(TvbProfile.current.TVB_STORAGE, "test_rename-xxx43"), 'w')
        file_writer.write('test-content')
        file_writer.close()

        is_changed, shoud_reset = self.settings_service.save_settings(**to_store_data)
        assert is_changed
        assert not shoud_reset
        # Check that the file was correctly moved:
        data = open(os.path.join(TvbProfile.current.TVB_STORAGE, 'RENAMED', "test_rename-xxx43"), 'r').read()
        assert data == 'test-content'

        StorageInterface.remove_files([os.path.join(TvbProfile.current.TVB_STORAGE, 'RENAMED'),
                                       os.path.join(TvbProfile.current.TVB_STORAGE, "test_rename-xxx43")])
