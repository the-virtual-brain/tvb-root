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
.. moduleauthor:: bogdan.neacsa <bogdan.neacsa@codemart.ro>
"""

import json
import unittest
from tvb.core.entities import model
from tvb.core.entities.storage import dao
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.adapters.exceptions import NoMemoryAvailableException
from tvb.core.services.operation_service import OperationService
from tvb.core.services.flow_service import FlowService
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.test_factory import TestFactory


class AdapterMemoryUsageTest(TransactionalTestCase):
    """
    Test class for the module handling methods computing required memory for an adapter to run.
    """
    
    def setUp(self):
        """
        Reset the database before each test.
        """
        self.test_user = TestFactory.create_user()
        self.test_project = TestFactory.create_project(admin=self.test_user)
    
    
    def test_adapter_memory(self):
        """
        Test that a method not implemented exception is raised in case the
        get_required_memory_size method is not implemented.
        """
        algo_group = dao.find_group("tvb.tests.framework.adapters.testadapter3", "TestAdapterHDDRequired")
        adapter = FlowService().build_adapter_instance(algo_group)
        self.assertEqual(42, adapter.get_required_memory_size())
        
        
    def test_adapter_huge_memory_requirement(self):
        """
        Test that an MemoryException is raised in case adapter cant launch due to lack of memory.
        """
        module = "tvb.tests.framework.adapters.testadapter3"
        class_name = "TestAdapterHugeMemoryRequired"
        algo_group = dao.find_group(module, class_name)
        adapter = FlowService().build_adapter_instance(algo_group)
        data = {"test": 5}

        operation = model.Operation(self.test_user.id, self.test_project.id, algo_group.id,
                                    json.dumps(data), json.dumps({}), status=model.STATUS_STARTED,
                                    method_name=ABCAdapter.LAUNCH_METHOD)
        operation = dao.store_entity(operation)
        self.assertRaises(NoMemoryAvailableException, OperationService().initiate_prelaunch, operation, adapter, {})



def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(AdapterMemoryUsageTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    unittest.main()     
        