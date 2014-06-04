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

from tvb.tests.framework.interfaces.web.controllers import burst_controller_test
from tvb.tests.framework.interfaces.web.controllers import exploration_controller_test
from tvb.tests.framework.interfaces.web.controllers import figure_controller_test
from tvb.tests.framework.interfaces.web.controllers import flow_controller_test
from tvb.tests.framework.interfaces.web.controllers import help_controller_test
from tvb.tests.framework.interfaces.web.controllers import local_connectivity_controller_test
from tvb.tests.framework.interfaces.web.controllers import project_controller_test
from tvb.tests.framework.interfaces.web.controllers import region_model_parameters_controller_test
from tvb.tests.framework.interfaces.web.controllers import region_stimulus_controller_test
from tvb.tests.framework.interfaces.web.controllers import settings_controllers_test
from tvb.tests.framework.interfaces.web.controllers import surface_model_parameters_controller_test
from tvb.tests.framework.interfaces.web.controllers import surface_stimulus_controller_test
from tvb.tests.framework.interfaces.web.controllers import users_controller_test
from tvb.tests.framework.interfaces.web.controllers import noise_configuration_controller_test

def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(burst_controller_test.suite())
    test_suite.addTest(exploration_controller_test.suite())
    test_suite.addTest(figure_controller_test.suite())
    test_suite.addTest(flow_controller_test.suite())
    test_suite.addTest(local_connectivity_controller_test.suite())
    test_suite.addTest(help_controller_test.suite())
    test_suite.addTest(project_controller_test.suite())
    test_suite.addTest(region_model_parameters_controller_test.suite())
    test_suite.addTest(region_stimulus_controller_test.suite())
    test_suite.addTest(settings_controllers_test.suite())
    test_suite.addTest(surface_model_parameters_controller_test.suite())
    test_suite.addTest(surface_stimulus_controller_test.suite())
    test_suite.addTest(users_controller_test.suite())
    test_suite.addTest(noise_configuration_controller_test.suite())
    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)
    
    