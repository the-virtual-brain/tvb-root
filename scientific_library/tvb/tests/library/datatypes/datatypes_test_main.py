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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

if __name__ == "__main__":
    from tvb.tests.library import setup_test_console_env
    setup_test_console_env()


import unittest
from tvb.tests.library.datatypes import arrays_test
from tvb.tests.library.datatypes import connectivity_test
from tvb.tests.library.datatypes import equations_test
from tvb.tests.library.datatypes import graph_test
from tvb.tests.library.datatypes import mapped_test
from tvb.tests.library.datatypes import mode_decompositions_test
from tvb.tests.library.datatypes import patterns_test
from tvb.tests.library.datatypes import projections_test
from tvb.tests.library.datatypes import sensors_test
from tvb.tests.library.datatypes import spectral_test
from tvb.tests.library.datatypes import surfaces_test
from tvb.tests.library.datatypes import temporal_correlations_test
from tvb.tests.library.datatypes import timeseries_test
from tvb.tests.library.datatypes import volumes_test


def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(arrays_test.suite())
    test_suite.addTest(connectivity_test.suite())
    test_suite.addTest(equations_test.suite())
    test_suite.addTest(graph_test.suite())
    test_suite.addTest(mapped_test.suite())
    test_suite.addTest(mode_decompositions_test.suite())
    test_suite.addTest(patterns_test.suite())
    test_suite.addTest(projections_test.suite())
    test_suite.addTest(sensors_test.suite())
    test_suite.addTest(spectral_test.suite())
    test_suite.addTest(surfaces_test.suite())
    test_suite.addTest(temporal_correlations_test.suite())
    test_suite.addTest(timeseries_test.suite())
    test_suite.addTest(volumes_test.suite())
    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)