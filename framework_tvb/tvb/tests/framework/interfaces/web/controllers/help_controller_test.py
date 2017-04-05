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
#
"""
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""
import unittest
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseTransactionalControllerTest
from tvb.interfaces.web.structure import WebStructure
from tvb.interfaces.web.controllers.help.help_controller import HelpController


class HelpControllerTest(BaseTransactionalControllerTest):
    """ Unit tests for HelpController """
    
    def setUp(self):
        """
        Sets up the environment for testing;
        creates a `HelpController`
        """
        self.init()
        self.help_c = HelpController()


    def tearDown(self):
        """ Cleans the testing environment """
        self.cleanup()

            
    def test_show_online_help(self):
        """
        Verifies that result dictionary has the expected keys / values
        """
        result_dict = self.help_c.showOnlineHelp(WebStructure.SECTION_PROJECT, WebStructure.SUB_SECTION_OPERATIONS)
        self.assertTrue('helpURL' in result_dict)
        self.assertEqual(result_dict['helpURL'], '/statichelp/manuals/UserGuide/UserGuide-UI_Project.html#operations')
        self.assertEqual(result_dict['overlay_class'], 'help')
        self.assertEqual(result_dict['overlay_content_template'], 'help/online_help')
        self.assertEqual(result_dict['overlay_description'], 'Online-Help')
        
            
def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(HelpControllerTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)