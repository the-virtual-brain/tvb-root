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
Created on Jul 21, 2011

.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import os
import unittest
from tvb.core.entities.storage import dao
from tvb.core.adapters.introspector import Introspector
from tvb.core.services.project_service import initialize_storage
from tvb.basic.config.settings import TVBSettings as cfg
from tvb.tests.framework.core.base_testcase import BaseTestCase


class IntrospectorTest(BaseTestCase):
    """
    Test class for the introspection module.
    """
    
    def setUp(self):
        """
        Reset the database before each test.
        """
        self.reset_database()
        #tvb.tests.framework path
        core_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.old_path = cfg.CURRENT_DIR
        cfg.CURRENT_DIR = os.path.dirname(core_path)
        self.introspector = Introspector("tvb.tests.framework")
        
        
    def tearDown(self):
        """
        Reset the database when test is done.
        """
        cfg.CURRENT_DIR = self.old_path
        self.reset_database()
        
    def test_introspect(self):
        """
        Test the actual introspect module on a test structure created in:
        tvb.tests.framework.dummy_adapters and tvb.tests.framework.dummy_datatypes
        """ 
        self.introspector.introspect(True)
        initialize_storage()
        
        all_categories = dao.get_algorithm_categories()
        category_ids = [cat.id for cat in all_categories if cat.displayname == "AdaptersTest"]
        groups = dao.get_groups_by_categories(category_ids)
        self.assertEqual(10, len(groups), "Introspection failed!")
        nr_adapters_mod2 = 0
        for algorithm in groups:
            self.assertTrue(algorithm.module in ['tvb.tests.framework.adapters.testadapter1', 'tvb.tests.framework.adapters.testadapter2',
                                                 'tvb.tests.framework.adapters.testadapter3',
                                                 'tvb.tests.framework.adapters.ndimensionarrayadapter',
                                                 "tvb.adapters.analyzers.group_python_adapter",
                                                 "tvb.tests.framework.adapters.testgroupadapter"],
                            "Unknown Adapter module:" + str(algorithm.module))
            self.assertTrue(algorithm.classname in ["TestAdapter1", "TestAdapterDatatypeInput", "TestAdapter2",
                                                    "TestAdapter22", "TestAdapter3", "TestGroupAdapter",
                                                    "NDimensionArrayAdapter", "PythonAdapter", "TestAdapterHDDRequired",
                                                    "TestAdapterHugeMemoryRequired"],
                            "Unknown Adapter Class:" + str(algorithm.classname))
            if algorithm.module == 'tvb.tests.framework.adapters.testadapter2':
                nr_adapters_mod2 += 1
        self.assertEqual(nr_adapters_mod2, 2, "Two adapters should have been loaded from module tvb.tests.framework.adapters2!")


    def test_xml_introspection(self):
        """
        Check the XML introspection. The folders which are introspected
        are defined in the variable __xml_folders__ from tvb.tests.framework/adapters/_init.py
        """
        self.introspector.introspect(True)
        initialize_storage()
        init_parameter = os.path.join("core", "adapters", "test_group.xml")
        group = dao.find_group("tvb.tests.framework.adapters.testgroupadapter", "TestGroupAdapter", init_parameter)
        self.assertTrue(group is not None, "The group was not found")
        self.assertEqual(group.init_parameter, init_parameter, "Wrong init_parameter:" + str(group.init_parameter))
        self.assertEqual(group.displayname, "Simple Python Analyzers", "The display-name of the group is not valid")
        self.assertEqual(group.algorithm_param_name, "simple", "The algorithm_param_name of the group is not valid")
        self.assertEqual(group.classname, "TestGroupAdapter", "The class-name of the group is not valid")
        self.assertEqual(group.module, "tvb.tests.framework.adapters.testgroupadapter", "Group Module invalid")

        
def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(IntrospectorTest))
    return test_suite


if __name__ == "__main__":
    #To run tests individually.
    unittest.main()  
    
