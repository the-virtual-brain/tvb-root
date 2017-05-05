# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
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
Gather the tests of the simulator module

.. moduleauthor:: Paula Sanz Leon <sanzleon.paula@gmail.com>
"""

# NOTE: for the moment test cases are not relevant (except for the on running all the simulations). 
# They are more like placeholders, but we definitely need to add more exhaustive tests.

if __name__ == "__main__":
    from tvb.tests.library import setup_test_console_env
    setup_test_console_env()


import unittest
from tvb.tests.library.simulator import common_test
from tvb.tests.library.simulator import coupling_test
from tvb.tests.library.simulator import integrators_test
from tvb.tests.library.simulator import models_test
from tvb.tests.library.simulator import monitors_test
from tvb.tests.library.simulator import noise_test
from tvb.tests.library.simulator import simulator_test
from tvb.tests.library.simulator import region_boundaries_test
from tvb.tests.library.simulator import history_test


def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(common_test.suite())
    test_suite.addTest(coupling_test.suite())
    test_suite.addTest(integrators_test.suite())
    test_suite.addTest(history_test.suite())
    test_suite.addTest(models_test.suite())
    test_suite.addTest(monitors_test.suite())
    test_suite.addTest(noise_test.suite())
    test_suite.addTest(region_boundaries_test.suite())
    test_suite.addTest(simulator_test.suite())

    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)