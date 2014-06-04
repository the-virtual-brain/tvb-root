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
""""
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import unittest
import os
import time
import tvb.core.services.event_handlers as event_handlers
from tvb.core.entities.storage import dao
from tvb.core.services.project_service import ProjectService
from tvb.tests.framework.core.test_factory import TestFactory
from tvb.tests.framework.core.base_testcase import BaseTestCase


class EventHandlerTest(BaseTestCase):
    """
    This class contains tests for the tvb.core.services.event_handler module.
    """ 
       
    def setUp(self):
        """
        Reset the database before each test.
        """
        self.project_service = ProjectService()
        self.test_user = TestFactory.create_user()
        
        
    def tearDown(self):
        """
        Cleans the environment after testing (database and executors dictionary)
        """
        self.clean_database()
        event_handlers.EXECUTORS_DICT = {}
        
        
    def test_handle_event(self):
        """
        Test a defined handler for the store project method.
        """
        path_to_events = os.path.dirname(__file__)
        event_handlers.read_events([path_to_events])
        data = dict(name="test_project", description="test_description", users=[])
        initial_projects = dao.get_projects_for_user(self.test_user.id)
        self.assertEqual(len(initial_projects), 0, "Database reset probably failed!")
        test_project = self.project_service.store_project(self.test_user, True, None, **data)
        # Operations will start asynchronously; Give them time.
        time.sleep(1)
        gid = dao.get_last_data_with_uid("test_uid")
        self.assertTrue(gid is not None, "Nothing was stored in database!")
        datatype = dao.get_datatype_by_gid(gid)
        self.assertEqual(datatype.type, "Datatype1", "Wrong data stored!")
        self.project_service._remove_project_node_files(test_project.id, gid)
        

def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(EventHandlerTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)
        
        
        