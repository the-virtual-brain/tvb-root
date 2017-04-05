# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

"""
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import copy
import json
import unittest
import cherrypy
from time import sleep
from tvb.tests.framework.core.test_factory import TestFactory
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseControllersTest
from tvb.core.entities import model
from tvb.core.entities.storage import dao
from tvb.core.services.operation_service import OperationService
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.flow_controller import FlowController
from tvb.interfaces.web.controllers.burst.burst_controller import BurstController
from tvb.tests.framework.adapters.testadapter1 import TestAdapter1
from tvb.tests.framework.datatypes.datatypes_factory import DatatypesFactory
from tvb.tests.framework.adapters.simulator.simulator_adapter_test import SIMULATOR_PARAMETERS


class FlowContollerTest(BaseControllersTest):
    """ Unit tests for FlowController """
    
    def setUp(self):
        """
        Sets up the environment for testing;
        creates a `FlowController`
        """
        self.init()
        self.flow_c = FlowController()
        self.burst_c = BurstController()
        self.operation_service = OperationService()
    
    
    def tearDown(self):
        """ Cleans up the testing environment """
        self.cleanup()
        self.clean_database()
            
            
    def test_context_selected(self):
        """
        Remove the project from CherryPy session and check that you are redirected to projects page.
        """
        del cherrypy.session[common.KEY_PROJECT]
        self._expect_redirect('/project/viewall', self.flow_c.step_analyzers)
        
        
    def test_valid_step(self):
        """
        For all algorithm categories check that a submenu is generated and the result
        page has it's title given by category name.
        """
        result_dict = self.flow_c.step_analyzers()
        self.assertTrue(common.KEY_SUBMENU_LIST in result_dict,
                        "Expect to have a submenu with available algorithms for category.")
        self.assertEqual(result_dict["section_name"], 'analyze')


    def test_step_connectivity(self):
        """
        Check that the correct section name and connectivity sub-menu are returned for the connectivity step.
        """
        result_dict = self.flow_c.step_connectivity()
        self.assertEqual(result_dict['section_name'], 'connectivity')
        self.assertEqual(result_dict['submenu_list'], self.flow_c.connectivity_submenu)


    def test_default(self):
        """
        Test default method from step controllers. Check that the submit link is ok, that a mainContent
        is present in result dict and that the isAdapter flag is set to true.
        """
        cherrypy.request.method = "GET"
        categories = dao.get_algorithm_categories()
        for categ in categories:
            algo_groups = dao.get_adapters_from_categories([categ.id])
            for algo in algo_groups:
                result_dict = self.flow_c.default(categ.id, algo.id)
                self.assertEqual(result_dict[common.KEY_SUBMIT_LINK], '/flow/%i/%i' % (categ.id, algo.id))
                self.assertTrue('mainContent' in result_dict)
                self.assertTrue(result_dict['isAdapter'])
                
                
    def test_default_cancel(self):
        """
        On cancel we should get a redirect to the back page link.
        """
        cherrypy.request.method = "POST"
        categories = dao.get_algorithm_categories()
        algo_groups = dao.get_adapters_from_categories([categories[0].id])
        self._expect_redirect('/project/viewoperations/%i' % self.test_project.id, self.flow_c.default,
                              categories[0].id, algo_groups[0].id, cancel=True, back_page='operations')
        
        
    def test_default_invalid_key(self):
        """
        Pass invalid keys for adapter and step and check you get redirect to tvb entry
        page with error set.
        """
        self._expect_redirect('/tvb?error=True', self.flow_c.default, 'invalid', 'invalid')
        
        
    def test_read_datatype_attribute(self):
        """
        Read an attribute from a datatype.
        """
        dt = DatatypesFactory().create_datatype_with_storage("test_subject", "RAW_STATE",
                                                             'this is the stored data'.split())
        returned_data = self.flow_c.read_datatype_attribute(dt.gid, "string_data")
        self.assertEqual(returned_data, '["this", "is", "the", "stored", "data"]')
        
        
    def test_read_datatype_attribute_method_call(self):
        """
        Call method on given datatype.
        """
        dt = DatatypesFactory().create_datatype_with_storage("test_subject", "RAW_STATE",
                                                             'this is the stored data'.split())
        args = {'length': 101}
        returned_data = self.flow_c.read_datatype_attribute(dt.gid, 'return_test_data', **args)
        self.assertTrue(returned_data == str(range(101)))
        
        
    def test_get_simple_adapter_interface(self):
        adapter = dao.get_algorithm_by_module('tvb.tests.framework.adapters.testadapter1', 'TestAdapter1')
        result = self.flow_c.get_simple_adapter_interface(adapter.id)
        expected_interface = TestAdapter1().get_input_tree()
        self.assertEqual(result['inputList'], expected_interface)
        
    
    def _long_burst_launch(self, is_range=False):
        self.burst_c.index()
        connectivity = DatatypesFactory().create_connectivity()[1]
        launch_params = copy.deepcopy(SIMULATOR_PARAMETERS)
        launch_params['connectivity'] = dao.get_datatype_by_id(connectivity.id).gid
        if not is_range:
            launch_params['simulation_length'] = '10000'
        else:
            launch_params['simulation_length'] = '[10000,10001,10002]'
            launch_params[model.RANGE_PARAMETER_1] = 'simulation_length'
        launch_params = {"simulator_parameters": json.dumps(launch_params)}
        burst_id = json.loads(self.burst_c.launch_burst("new", "test_burst", **launch_params))['id']
        return dao.get_burst_by_id(burst_id)


    def _wait_for_burst_ops(self, burst_config):
        """ sleeps until some operation of the burst is created"""
        waited = 1
        timeout = 50
        operations = dao.get_operations_in_burst(burst_config.id)
        while not len(operations) and waited <= timeout:
            sleep(1)
            waited += 1
            operations = dao.get_operations_in_burst(burst_config.id)
        operations = dao.get_operations_in_burst(burst_config.id)
        return operations


    def test_stop_burst_operation(self):
        burst_config = self._long_burst_launch()
        operation = self._wait_for_burst_ops(burst_config)[0]
        self.assertFalse(operation.has_finished)
        self.flow_c.stop_burst_operation(operation.id, 0, False)
        operation = dao.get_operation_by_id(operation.id)
        self.assertEqual(operation.status, model.STATUS_CANCELED)
        
        
    def test_stop_burst_operation_group(self):
        burst_config = self._long_burst_launch(True)
        operations = self._wait_for_burst_ops(burst_config)
        operations_group_id = 0
        for operation in operations:
            self.assertFalse(operation.has_finished)
            operations_group_id = operation.fk_operation_group
        self.flow_c.stop_burst_operation(operations_group_id, 1, False)
        for operation in operations:
            operation = dao.get_operation_by_id(operation.id)
            self.assertEqual(operation.status, model.STATUS_CANCELED)
        
        
    def test_remove_burst_operation(self):
        burst_config = self._long_burst_launch()
        operation = self._wait_for_burst_ops(burst_config)[0]
        self.assertFalse(operation.has_finished)
        self.flow_c.stop_burst_operation(operation.id, 0, True)
        operation = dao.try_get_operation_by_id(operation.id)
        self.assertTrue(operation is None)
        
        
    def test_remove_burst_operation_group(self):
        burst_config = self._long_burst_launch(True)
        operations = self._wait_for_burst_ops(burst_config)
        operations_group_id = 0
        for operation in operations:
            self.assertFalse(operation.has_finished)
            operations_group_id = operation.fk_operation_group
        self.flow_c.stop_burst_operation(operations_group_id, 1, True)
        for operation in operations:
            operation = dao.try_get_operation_by_id(operation.id)
            self.assertTrue(operation is None)


    def _launch_test_algo_on_cluster(self, **data):
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.testadapter1", "TestAdapter1")
        algo = adapter.stored_adapter
        algo_category = dao.get_category_by_id(algo.fk_category)
        operations, _ = self.operation_service.prepare_operations(self.test_user.id, self.test_project.id, algo,
                                                                  algo_category, {}, **data)
        self.operation_service._send_to_cluster(operations, adapter)
        return operations


    def test_stop_operations(self):
        data = {"test1_val1": 5, 'test1_val2': 5}
        operations = self._launch_test_algo_on_cluster(**data)
        operation = dao.get_operation_by_id(operations[0].id)
        self.assertFalse(operation.has_finished)
        self.flow_c.stop_operation(operation.id, 0, False)
        operation = dao.get_operation_by_id(operation.id)
        self.assertEqual(operation.status, model.STATUS_CANCELED)
        
        
    def test_stop_operations_group(self):
        data = {model.RANGE_PARAMETER_1: "test1_val1", "test1_val1": '5,6,7', 'test1_val2': 5}
        operations = self._launch_test_algo_on_cluster(**data)
        operation_group_id = 0
        for operation in operations:
            operation = dao.get_operation_by_id(operation.id)
            self.assertFalse(operation.has_finished)
            operation_group_id = operation.fk_operation_group
        self.flow_c.stop_operation(operation_group_id, 1, False)
        for operation in operations:
            operation = dao.get_operation_by_id(operation.id)
            self.assertEqual(operation.status, model.STATUS_CANCELED)
        
        

def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(FlowContollerTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)
