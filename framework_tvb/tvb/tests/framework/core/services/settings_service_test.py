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
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import os
import shutil
import unittest
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.basic.profile import TvbProfile
from tvb.core.services.settings_service import SettingsService, InvalidSettingsException


TEST_CONFIG_FILE = os.path.expanduser(os.path.join("~", 'tvb.tests.framework.configuration'))


class SettingsServiceTest(BaseTestCase):
    """
    This class contains tests for the tvb.core.services.settings_service module.
    """
    TEST_SETTINGS = {SettingsService.KEY_ADMIN_NAME: 'test_name',
                     SettingsService.KEY_ADMIN_EMAIL: 'my@yahoo.com',
                     SettingsService.KEY_PORT: 8081,
                     SettingsService.KEY_URL_WEB: "http://192.168.123.11:8081/",
                     SettingsService.KEY_MAX_DISK_SPACE_USR: 2 ** 8}


    def setUp(self):
        """
        Prepare the usage of a different config file for this class only.
        """
        if os.path.exists(TEST_CONFIG_FILE):
            os.remove(TEST_CONFIG_FILE)

        self.old_config_file = TvbProfile.current.TVB_CONFIG_FILE
        TvbProfile.current.__class__.TVB_CONFIG_FILE = TEST_CONFIG_FILE
        TvbProfile._build_profile_class(TvbProfile.CURRENT_PROFILE_NAME)
        self.settings_service = SettingsService()
        
            
    def tearDown(self):
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
        self.assertRaises(InvalidSettingsException, self.settings_service.check_db_url, "this-url-should-be-invalid")
        
    
    def test_get_free_disk_space(self):
        """
        Check that no exception is raised during the query for free disk space.
        Also do a check that returned value is greater than 0.
        Not most precise check but other does not seem possible so far.
        """
        disk_space = self.settings_service.get_disk_free_space(TvbProfile.current.TVB_STORAGE)
        self.assertTrue(disk_space > 0, "Disk space should never be negative.")
    
            
    def test_first_run_save(self):
        """
        Check that before setting something, all flags are pointing towards empty.
        After storing some configurations, check that flags are changed.
        """
        initial_configurations = self.settings_service.configurable_keys
        first_run = TvbProfile.is_first_run()
        self.assertTrue(first_run, "Invalid First_Run flag!!")
        self.assertFalse(os.path.exists(TEST_CONFIG_FILE))
        self.assertTrue(len(TvbProfile.current.manager.stored_settings) == 0)

        to_store_data = {key: value['value'] for key, value in initial_configurations.iteritems()}
        for key, value in self.TEST_SETTINGS.iteritems():
            to_store_data[key] = value
        _, shoud_reset = self.settings_service.save_settings(**to_store_data)

        self.assertTrue(shoud_reset)
        first_run = TvbProfile.is_first_run()
        self.assertFalse(first_run, "Invalid First_Run flag!!")
        self.assertTrue(os.path.exists(TEST_CONFIG_FILE))
        self.assertFalse(len(TvbProfile.current.manager.stored_settings) == 0)
        
        
    def test_read_stored_settings(self):
        """
        Test to see that keys from the configuration dict is updated with
        the value from the configuration file after store.
        """
        initial_configurations = self.settings_service.configurable_keys
        to_store_data = {key: value['value'] for key, value in initial_configurations.iteritems()}
        for key, value in self.TEST_SETTINGS.iteritems():
            to_store_data[key] = value

        is_changed, shoud_reset = self.settings_service.save_settings(**to_store_data)
        self.assertTrue(shoud_reset and is_changed)

        # enforce keys to get repopulated:
        TvbProfile._build_profile_class(TvbProfile.CURRENT_PROFILE_NAME)
        self.settings_service = SettingsService()

        updated_configurations = self.settings_service.configurable_keys
        for key, value in updated_configurations.iteritems():
            if key in self.TEST_SETTINGS:
                self.assertEqual(self.TEST_SETTINGS[key], value['value'])
            elif key == SettingsService.KEY_ADMIN_PWD:
                self.assertEqual(TvbProfile.current.web.admin.ADMINISTRATOR_PASSWORD, value['value'])
                self.assertEqual(TvbProfile.current.web.admin.ADMINISTRATOR_BLANK_PWD,
                                 initial_configurations[key]['value'])
            else:
                self.assertEqual(initial_configurations[key]['value'], value['value'])

                    
    def test_update_settings(self):
        """
        Test update of settings: correct flags should be returned, and check storage folder renamed
        """
        # 1. save on empty config-file:
        to_store_data = {key: value['value'] for key, value in self.settings_service.configurable_keys.iteritems()}
        for key, value in self.TEST_SETTINGS.iteritems():
            to_store_data[key] = value

        is_changed, shoud_reset = self.settings_service.save_settings(**to_store_data)
        self.assertTrue(shoud_reset and is_changed)

        # 2. Reload and save with the same values (is_changed expected to be False)
        TvbProfile._build_profile_class(TvbProfile.CURRENT_PROFILE_NAME)
        self.settings_service = SettingsService()
        to_store_data = {key: value['value'] for key, value in self.settings_service.configurable_keys.iteritems()}

        is_changed, shoud_reset = self.settings_service.save_settings(**to_store_data)
        self.assertFalse(is_changed)
        self.assertFalse(shoud_reset)

        # 3. Reload and check that changing TVB_STORAGE is done correctly
        TvbProfile._build_profile_class(TvbProfile.CURRENT_PROFILE_NAME)
        self.settings_service = SettingsService()
        to_store_data = {key: value['value'] for key, value in self.settings_service.configurable_keys.iteritems()}
        to_store_data[SettingsService.KEY_STORAGE] = os.path.join(TvbProfile.current.TVB_STORAGE, 'RENAMED')

        # Write a test-file and check that it is moved
        file_writer = open(os.path.join(TvbProfile.current.TVB_STORAGE, "test_rename-xxx43"), 'w')
        file_writer.write('test-content')
        file_writer.close()

        is_changed, shoud_reset = self.settings_service.save_settings(**to_store_data)
        self.assertTrue(is_changed)
        self.assertFalse(shoud_reset)
        # Check that the file was correctly moved:
        data = open(os.path.join(TvbProfile.current.TVB_STORAGE, 'RENAMED', "test_rename-xxx43"), 'r').read()
        self.assertEqual(data, 'test-content')

        shutil.rmtree(os.path.join(TvbProfile.current.TVB_STORAGE, 'RENAMED'))
        os.remove(os.path.join(TvbProfile.current.TVB_STORAGE, "test_rename-xxx43"))



def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(SettingsServiceTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)
    
    
    