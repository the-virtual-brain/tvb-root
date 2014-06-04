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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import os
import json
import unittest
from tvb.interfaces.web.controllers.settings_controller import SettingsController
from tvb.basic.config.settings import TVBSettings as cfg
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseControllersTest


class SettingsControllerTest(TransactionalTestCase, BaseControllersTest): 
    """Unit tests for SettingsController"""
    
    def setUp(self):
        """
        Sets up the environment for testing;
        creates a `SettingsController`
        """
        BaseControllersTest.init(self)
        self.settings_c = SettingsController()
        if os.path.exists(cfg.TVB_CONFIG_FILE):
            os.remove(cfg.TVB_CONFIG_FILE)
    
    
    def tearDown(self):
        """ Clean testing environment """
        BaseControllersTest.cleanup(self)

            
#    TODO: This sees settings as being changed so it will restart a tvb process. See if there is any way around else drop test.
#    def test_settings(self):
#        """
#        Test that settings are indeed saved in config file.
#        """
#        self.test_user.role = "ADMINISTRATOR"
#        self.test_user = dao.store_entity(self.test_user)
#        cherrypy.session[b_c.KEY_USER] = self.test_user
#        
#        form_data = {'ADMINISTRATOR_NAME' : 'test_admin',
#                     'ADMINISTRATOR_PASSWORD' : 'test_pass',
#                     'ADMINISTRATOR_EMAIL' : "email@test.test",
#                     'TVB_STORAGE' : "test_storage",
#                     'USR_DISK_SPACE' : 1,
#                     'SELECTED_DB' : 'postgres',
#                     'URL_VALUE' : cfg.DB_URL,
#                     'SERVER_IP' : 'localtesthost',
#                     'MAXIMUM_NR_OF_OPS_IN_RANGE' : 6,
#                     'MATLAB_EXECUTABLE' : '',
#                     'MAXIMUM_NR_OF_THREADS' : 9,
#                     'MAXIMUM_NR_OF_VERTICES_ON_SURFACE' : 151,
#                     'MPLH5_SERVER_PORT' : 111,
#                     'WEB_SERVER_PORT' : 888
#                     }
#        self.settings_c.settings(save_settings=True, **form_data)
#        self.assertEqual(cfg.ADMINISTRATOR_NAME, 'test_admin')
#        self.assertEqual(cfg.ADMINISTRATOR_PASSWORD, 'test_pass')
#        self.assertEqual(cfg.ADMINISTRATOR_EMAIL, "email@test.test")
#        self.assertEqual(cfg.TVB_STORAGE, "test_storage")
#        self.assertEqual(cfg.MAX_DISK_SPACE, 1)
#        self.assertEqual(cfg.SELECTED_DB, 'postgres')
#        self.assertEqual(cfg.SERVER_IP, 'localtesthost')
#        self.assertEqual(cfg.MAX_RANGE_NUMBER, 6)
#        self.assertEqual(cfg.MAX_THREADS_NUMBER, 9)
#        self.assertEqual(cfg.MAX_SURFACE_VERTICES_NUMBER, 151)
#        self.assertEqual(cfg.MPLH5_SERVER_PORT, 111)
#        self.assertEqual(cfg.WEB_SERVER_PORT, 888)
    
    
    def test_check_db_url(self):
        """
        Test that for a valid url the correct status is returned.
        """
        valid_db_data = {cfg.KEY_STORAGE : cfg.TVB_STORAGE, 
                         cfg.KEY_DB_URL : cfg.DB_URL}
        result = json.loads(self.settings_c.check_db_url(**valid_db_data))
        self.assertEqual(result['status'], 'ok')
        
    
    def test_check_db_url_invalid(self):
        """
        Test that for invalid url the proper message is returned.
        """
        valid_db_data = {cfg.KEY_STORAGE : cfg.TVB_STORAGE, 
                         cfg.KEY_DB_URL : "this URL should be invalid"}
        result = json.loads(self.settings_c.check_db_url(**valid_db_data))
        self.assertEqual(result['status'], 'not ok')
        
        
def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(SettingsControllerTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)