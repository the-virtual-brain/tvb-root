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
import json
import unittest
import cherrypy
import tvb.interfaces.web.controllers.common as common
from tvb.interfaces.web.controllers.spatial.region_model_parameters_controller import RegionsModelParametersController
from tvb.interfaces.web.controllers.burst.burst_controller import BurstController
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseControllersTest
from tvb.tests.framework.datatypes.datatypes_factory import DatatypesFactory
from tvb.tests.framework.adapters.simulator.simulator_adapter_test import SIMULATOR_PARAMETERS


class RegionsModelParametersControllerTest(TransactionalTestCase, BaseControllersTest):
    """ Unit tests for RegionsModelParametersController """
    
    def setUp(self):
        """
        Sets up the environment for testing;
        creates a `RegionsModelParametersController` and a connectivity
        """
        BaseControllersTest.init(self)
        self.region_m_p_c = RegionsModelParametersController()
        BurstController().index()
        stored_burst = cherrypy.session[common.KEY_BURST_CONFIG]
        _, self.connectivity = DatatypesFactory().create_connectivity()
        new_params = {}
        for key, val in SIMULATOR_PARAMETERS.iteritems():
            new_params[key] = {'value': val}
        new_params['connectivity'] = {'value': self.connectivity.gid}
        stored_burst.simulator_configuration = new_params
    
    
    def tearDown(self):
        """ Clean the testing environment """
        BaseControllersTest.cleanup(self)
#        if os.path.exists(cfg.TVB_CONFIG_FILE):
#            os.remove(cfg.TVB_CONFIG_FILE)
    
    
    def test_edit_model_parameters(self):
        """
        Verifies that result dictionary has the expected keys / values after call to
        `edit_model_parameters()`
        """
        result_dict = self.region_m_p_c.edit_model_parameters()
        self.assertEqual(self.connectivity.gid, result_dict['connectivity_entity'].gid)
        self.assertTrue(result_dict['displayDefaultSubmitBtn'])
        self.assertEqual(result_dict['mainContent'], 'spatial/model_param_region_main')
        self.assertEqual(result_dict['submit_parameters_url'], 
                         '/spatial/modelparameters/regions/submit_model_parameters')
        self.assertTrue('paramSlidersData' in result_dict)
        self.assertTrue('parametersNames' in result_dict)
        self.assertTrue('pointsLabels' in result_dict)
        self.assertTrue('positions' in result_dict)
        
        
    def test_load_model_for_connectivity_node(self):
        """
        Verifies that result dictionary has the expected keys / values after call to
        `edit_model_parameters()`
        """
        self.region_m_p_c.edit_model_parameters()
        result_dict = self.region_m_p_c.load_model_for_connectivity_node(1)
        self.assertTrue('paramSlidersData' in result_dict)
        self.assertTrue('parametersNames' in result_dict)
        
        
    def test_update_model_parameter_for_nodes(self):
        """
        Verifies that result dictionary has the expected keys / values after call to
        `update_model_parameter_for_nodes(...)`
        """
        self.region_m_p_c.edit_model_parameters()
        result_dict = self.region_m_p_c.load_model_for_connectivity_node(1)
        param_names = result_dict['parametersNames']
        param_values = json.loads(result_dict['paramSlidersData'])
        old_value = param_values[param_names[0]]["default"]
        self.region_m_p_c.update_model_parameter_for_nodes(param_names[0], 
                                                           old_value + 1, json.dumps([1]))
        result_dict = self.region_m_p_c.load_model_for_connectivity_node(1)
        param_names = result_dict['parametersNames']
        param_values = json.loads(result_dict['paramSlidersData'])
        self.assertEqual(param_values[param_names[0]]["default"], old_value + 1)
        
        
    def test_copy_model(self):
        """
        Verifies that result dictionary has the expected keys / values after call to
        `copy_model()`
        """
        self.region_m_p_c.edit_model_parameters()
        result_dict = self.region_m_p_c.load_model_for_connectivity_node(1)
        param_names = result_dict['parametersNames']
        param_values = json.loads(result_dict['paramSlidersData'])
        old_value = param_values[param_names[0]]["default"]
        self.region_m_p_c.update_model_parameter_for_nodes(param_names[0], 
                                                           old_value + 1, json.dumps([1]))
        self.region_m_p_c.copy_model('1', json.dumps([2]))
        result_dict = self.region_m_p_c.load_model_for_connectivity_node(2)
        param_names = result_dict['parametersNames']
        param_values = json.loads(result_dict['paramSlidersData'])
        self.assertEqual(param_values[param_names[0]]["default"], old_value + 1)
        
        
    def test_reset_model_parameters_for_nodes(self):
        """
        Verifies that result dictionary has the expected keys / values after call to
        `reset_model_parameters_for_nodes(...)`
        """
        self.region_m_p_c.edit_model_parameters()
        result_dict = self.region_m_p_c.load_model_for_connectivity_node(1)
        param_names = result_dict['parametersNames']
        param_values = json.loads(result_dict['paramSlidersData'])
        old_value = param_values[param_names[0]]["default"]
        self.region_m_p_c.update_model_parameter_for_nodes(param_names[0], 
                                                           old_value + 1, json.dumps([1]))
        self.region_m_p_c.reset_model_parameters_for_nodes(json.dumps([1]))
        result_dict = self.region_m_p_c.load_model_for_connectivity_node(2)
        param_names = result_dict['parametersNames']
        param_values = json.loads(result_dict['paramSlidersData'])
        self.assertEqual(param_values[param_names[0]]["default"], old_value)
        
        
    def test_submit_model_parameters(self):
        """
        Verifies call to `submit_model_parameters(...)` correctly redirects to '/burst/'
        """
        self.region_m_p_c.edit_model_parameters()
        self._expect_redirect('/burst/', self.region_m_p_c.submit_model_parameters)
        
            
def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(RegionsModelParametersControllerTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)