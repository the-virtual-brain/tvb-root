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
from tvb.basic.config.settings import TVBSettings as cfg
from tvb.core.services.settings_service import SettingsService, InvalidSettingsException


TEST_CONFIG_FILE = os.path.expanduser(os.path.join("~", 'tvb.tests.framework.configuration'))


class SettingsServiceTest(unittest.TestCase):
    """
    This class contains tests for the tvb.core.services.settings_service module.
    """
    
    def _write_cfg_file(self, initial_data):
        """ Write in CFG file, a dictionary of properties"""
        with open(cfg.TVB_CONFIG_FILE, 'w') as file_writer:
            for key in initial_data:
                file_writer.write(key + '=' + str(initial_data[key]) + '\n')
            file_writer.close()
        cfg.read_config_file()
        self.settings_service = SettingsService()
    
    
    def setUp(self):
        """
        Sets up the environment for testing;
        saves config file;
        creates initial TVB settings and a `SettingsService`
        """
        self.initial_settings = {}
        self.old_config_file = cfg.TVB_CONFIG_FILE
        cfg.TVB_CONFIG_FILE = TEST_CONFIG_FILE
        if os.path.exists(cfg.TVB_CONFIG_FILE):
            os.remove(cfg.TVB_CONFIG_FILE)
        for key in dir(cfg):
            self.initial_settings[key] = getattr(cfg, key)
        self.settings_service = SettingsService()
        
            
    def tearDown(self):
        """
        Clean up after testing is done;
        restore config file
        """
        if os.path.exists(cfg.TVB_CONFIG_FILE):
            os.remove(cfg.TVB_CONFIG_FILE)
        cfg.FILE_SETTINGS = None
        cfg.TVB_CONFIG_FILE = self.old_config_file
    
    
    def test_check_db_url_invalid(self):
        """
        Make sure a proper exception is raised in case an invalid database url is passed.
        """
        self.assertRaises(InvalidSettingsException, self.settings_service.check_db_url, "this-url-should-be-invalid")
        
    
    def test_get_free_disk_space(self):
        """
        Check that no unexpected exception is raised during the query for free disk space.
        Also do a check that returned value is greater than 0. Not most precise check but other
        does not seem possible so far.
        """
        disk_space = self.settings_service.get_disk_free_space(cfg.TVB_STORAGE)
        self.assertTrue(disk_space > 0, "Disk space should never be negative.")
    
            
    def test_getsettings_no_configfile(self):
        """
        If getting the interface with no configuration file present, the
        configurations dictionary should not change and the first_run parameter
        should be true.
        """
        initial_configurations = self.settings_service.configurable_keys
        updated = self.settings_service.configurable_keys
        first_run = self.settings_service.is_first_run()
        self.assertEqual(initial_configurations, updated, "Configuration changed even with no config file.")
        self.assertTrue(first_run, "Invalid First_Run flag!!")
        
        
    def test_getsettings_with_config(self):
        """
        Test to see that keys from the configuration dict is updated with
        the value from the configuration file.
        """
        initial_configurations = self.settings_service.configurable_keys
        #Simulate the encrypted value is stored
        initial_configurations[self.settings_service.KEY_ADMIN_PWD] = {'value': cfg.ADMINISTRATOR_PASSWORD}
        test_dict = {self.settings_service.KEY_ADMIN_NAME: 'test_name',
                     self.settings_service.KEY_ADMIN_EMAIL: 'my@yahoo.com',
                     self.settings_service.KEY_PORT: 8081,
                     self.settings_service.KEY_URL_WEB: "http://192.168.123.11:8081/"}
        
        self._write_cfg_file(test_dict)
        updated_cfg = self.settings_service.configurable_keys
        isfirst = self.settings_service.is_first_run()
        self.assertFalse(isfirst, "Invalid First_Run flag!!")
        for key in updated_cfg:
            if key in test_dict:
                self.assertEqual(updated_cfg[key]['value'], test_dict[key])
            else:
                self.assertEqual(updated_cfg[key]['value'], initial_configurations[key]['value'])
                
    
    def test_updatesets_with_config(self):
        """
        Test that the config.py entries are updated accordingly when a 
        configuration file is present.
        """
        test_storage = os.path.join(cfg.TVB_STORAGE, 'test_storage')
        test_dict = {self.settings_service.KEY_STORAGE: test_storage,
                     self.settings_service.KEY_ADMIN_NAME: 'test_name',
                     self.settings_service.KEY_ADMIN_EMAIL: 'my@yahoo.com',
                     self.settings_service.KEY_PORT: 8081}
        old_settings = {}
        for attr in dir(cfg):
            if not attr.startswith('__'):
                old_settings[attr] = getattr(cfg, attr)
        self._write_cfg_file(test_dict)   
        for attr in dir(cfg):
            if not attr.startswith('__'):
                if attr in test_dict:
                    self.assertEqual(test_dict[attr], getattr(cfg, attr),
                                     'For some reason attribute %s did not change' % attr)
                    
                    
    def test_savesettings_no_change(self):
        """
        Test than when nothing changes in the settings file, the correct flags
        are returned.
        """
        test_storage = os.path.join(cfg.TVB_STORAGE, 'test_storage')
        disk_storage = 100
        initial_data = {self.settings_service.KEY_STORAGE: test_storage,
                        self.settings_service.KEY_ADMIN_NAME: 'test_name',
                        self.settings_service.KEY_SELECTED_DB: cfg.SELECTED_DB,
                        self.settings_service.KEY_DB_URL: cfg.DB_URL,
                        self.settings_service.KEY_ADMIN_EMAIL: 'my@yahoo.com',
                        self.settings_service.KEY_MAX_DISK_SPACE_USR: disk_storage * (2 ** 10),
                        self.settings_service.KEY_PORT: 8081,
                        self.settings_service.KEY_MATLAB_EXECUTABLE: 'test'}

        self._write_cfg_file(initial_data)
        initial_data[self.settings_service.KEY_MAX_DISK_SPACE_USR] = disk_storage
        changes, restart = self.settings_service.save_settings(**initial_data)
        self.assertFalse(changes)
        self.assertFalse(restart)

        
    def test_savesettings_changedir(self):
        """
        Test than storage is changed, the data is copied in proper place.
        """
        #Add some additional entries that would normaly come from the UI.
        old_storage = os.path.join(cfg.TVB_STORAGE, 'tvb.tests.framework_old')
        new_storage = os.path.join(cfg.TVB_STORAGE, 'tvb.tests.framework_new')
        test_data = 'tvb.tests.framework_data'
        if os.path.exists(old_storage):
            shutil.rmtree(old_storage)
        os.makedirs(old_storage)
        file_writer = open(os.path.join(old_storage, test_data), 'w')
        file_writer.write('test')
        file_writer.close() 
        initial_data = {self.settings_service.KEY_STORAGE: old_storage,
                        self.settings_service.KEY_ADMIN_NAME: 'test_name',
                        self.settings_service.KEY_SELECTED_DB: cfg.SELECTED_DB,
                        self.settings_service.KEY_DB_URL: cfg.DB_URL,
                        self.settings_service.KEY_MAX_DISK_SPACE_USR: 100,
                        self.settings_service.KEY_MATLAB_EXECUTABLE: 'test'}
        self._write_cfg_file(initial_data)
        initial_data[self.settings_service.KEY_STORAGE] = new_storage
        anything_changed, is_reset = self.settings_service.save_settings(**initial_data)
        self.assertTrue(anything_changed)
        self.assertFalse(is_reset)
        copied_file_path = os.path.join(new_storage, test_data)
        data = open(copied_file_path, 'r').read()
        self.assertEqual(data, 'test')
        shutil.rmtree(old_storage)
        shutil.rmtree(new_storage)
        

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
    
    
    