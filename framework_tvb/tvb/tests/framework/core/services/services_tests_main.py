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
from tvb.tests.framework.core.services import burst_service_test
from tvb.tests.framework.core.services import figure_service_test
from tvb.tests.framework.core.services import flow_service_test
from tvb.tests.framework.core.services import import_service_test
from tvb.tests.framework.core.services import links_test
from tvb.tests.framework.core.services import operation_service_test
from tvb.tests.framework.core.services import project_service_test
from tvb.tests.framework.core.services import project_structure_test
from tvb.tests.framework.core.services import remove_test
from tvb.tests.framework.core.services import serialization_manager_test
from tvb.tests.framework.core.services import settings_service_test
from tvb.tests.framework.core.services import user_service_test
from tvb.tests.framework.core.services import workflow_service_test



def suite():
    """
    Gather all the service tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(burst_service_test.suite())
    test_suite.addTest(figure_service_test.suite())
    test_suite.addTest(flow_service_test.suite())
    test_suite.addTest(import_service_test.suite())
    test_suite.addTest(links_test.suite())
    test_suite.addTest(operation_service_test.suite())
    test_suite.addTest(project_service_test.suite())
    test_suite.addTest(project_structure_test.suite())
    test_suite.addTest(remove_test.suite())
    test_suite.addTest(serialization_manager_test.suite())
    test_suite.addTest(settings_service_test.suite())
    test_suite.addTest(user_service_test.suite())
    test_suite.addTest(workflow_service_test.suite())
    return test_suite


if __name__ == "__main__":
    #So you can run tests individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)
    
    