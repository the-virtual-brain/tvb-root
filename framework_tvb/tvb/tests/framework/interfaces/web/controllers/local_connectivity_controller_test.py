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

import unittest
import cherrypy
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseTransactionalControllerTest
from tvb.core.entities.transient.context_local_connectivity import ContextLocalConnectivity
from tvb.interfaces.web.controllers.spatial.local_connectivity_controller import LocalConnectivityController
from tvb.interfaces.web.controllers.spatial.local_connectivity_controller import KEY_LCONN_CONTEXT


class LocalConnectivityControllerTest(BaseTransactionalControllerTest):
    """ Unit tests for LocalConnectivityController """
    
    def setUp(self):
        """
        Sets up the environment for testing;
        creates a `LolcalConnectivityController`
        """
        self.init()
        self.local_p_c = LocalConnectivityController()


    def tearDown(self):
        """ Cleans the testing environment """
        self.cleanup()

            
    def _default_checks(self, result_dict):
        """
        Check for some info that should be true for all steps from the
        local connectivity controller.
        """
        self.assertEqual(result_dict['data'], {})
        self.assertTrue(isinstance(result_dict['existentEntitiesInputList'], list))
    
    
    def test_step_1(self):
        """
        Test that the dictionary returned by the controller for the LC Workflow first step is correct.
        """
        result_dict = self.local_p_c.step_1(1)
        self._default_checks(result_dict)
        self.assertEqual(result_dict['equationViewerUrl'], 
                         '/spatial/localconnectivity/get_equation_chart')
        self.assertTrue(isinstance(result_dict['inputList'], list))
        self.assertEqual(result_dict['mainContent'], 'spatial/local_connectivity_step1_main')
        self.assertEqual(result_dict['next_step_url'], '/spatial/localconnectivity/step_2')
        self.assertEqual(result_dict['resetToDefaultUrl'], 
                         '/spatial/localconnectivity/reset_local_connectivity')
        self.assertEqual(result_dict['submit_parameters_url'], 
                         '/spatial/localconnectivity/create_local_connectivity')
        self.assertEqual(result_dict['resetToDefaultUrl'], 
                         '/spatial/localconnectivity/reset_local_connectivity')
        
        
    def test_step_2(self):
        """
        Test that the dictionary returned by the controller for the LC Workflow second step is correct.
        """
        context = ContextLocalConnectivity()
        cherrypy.session[KEY_LCONN_CONTEXT] = context
        result_dict = self.local_p_c.step_2()
        self._default_checks(result_dict)
        self.assertEqual(result_dict['loadExistentEntityUrl'], '/spatial/localconnectivity/load_local_connectivity')
        self.assertEqual(result_dict['mainContent'], 'spatial/local_connectivity_step2_main')
        self.assertEqual(result_dict['next_step_url'], '/spatial/localconnectivity/step_1')
        
        
            
def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(LocalConnectivityControllerTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)
    
    
    
    