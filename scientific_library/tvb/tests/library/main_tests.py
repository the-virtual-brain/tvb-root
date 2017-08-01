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
Entry point for all unit-tests for TVB Scientific Library.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

from tvb.tests.library import setup_test_console_env
from tvb.basic.profile import TvbProfile

setup_test_console_env()

# Make sure we are in library mode and are not influenced by framework
try:
    import tvb.interfaces

    raise Exception("Found framework in library mode testing. Abort....")
except ImportError:
    pass

import os
from sys import argv
from coverage import coverage


KEY_CONSOLE = 'console'
KEY_COVERAGE = 'coverage'
KEY_XML = 'xml'


def generate_excludes(root_folders):
    """
    Specify excludes for Coverage.
    """
    excludes = []
    for root in root_folders:
        for root, _, files in os.walk(root):
            for file_n in files:
                full_path = os.path.join(root, file_n)
                if (full_path.endswith('__init__.py') or
                            os.path.join('simulator', 'demos') in full_path):
                    excludes.append(full_path)
    return excludes


if __name__ == "__main__":
    # Start all TVB tests (if in Coverage mode)
    if KEY_COVERAGE in argv:
        import tvb.simulator as sim

        SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(sim.__file__)))
        COVERAGE = coverage(source=["tvb.basic", "tvb.datatypes", "tvb.simulator"],
                            omit=generate_excludes([SOURCE_DIR]), cover_pylib=False, branch=True)
        COVERAGE.start()
        ## This needs to be executed before any TVB import.

import unittest
import datetime
from tvb.tests.library.basic import basic_test_main
from tvb.tests.library.datatypes import datatypes_test_main
from tvb.tests.library.simulator import simulator_test_main
from tvb.tests.library.xml_runner import XMLTestRunner


def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(basic_test_main.suite())
    test_suite.addTest(datatypes_test_main.suite())
    test_suite.addTest(simulator_test_main.suite())
    return test_suite


if __name__ == "__main__":
    # Start all TVB tests
    START_TIME = datetime.datetime.now()

    if KEY_CONSOLE in argv:
        TEST_RUNNER = unittest.TextTestRunner()
        TEST_SUITE = suite()
        TEST_RUNNER.run(TEST_SUITE)
    if KEY_XML in argv:
        XML_STREAM = open(os.path.join(TvbProfile.current.TVB_LOG_FOLDER, "TEST-LIBRARY-RESULTS.xml"), "w")
        OUT_STREAM = open(os.path.join(TvbProfile.current.TVB_LOG_FOLDER, "TEST-LIBRARY.out"), "w")
        TEST_RUNNER = XMLTestRunner(XML_STREAM, OUT_STREAM)
        TEST_SUITE = suite()
        TEST_RUNNER.run(TEST_SUITE)
        XML_STREAM.close()
        OUT_STREAM.close()

    print('It run tests for %d sec.' % (datetime.datetime.now() - START_TIME).seconds)

    if KEY_COVERAGE in argv:
        COVERAGE.stop()
        COVERAGE.xml_report(outfile=os.path.join(TvbProfile.current.TVB_LOG_FOLDER, 'coverage_library.xml'))
        # COVERAGE.html_report(directory=os.path.join(TvbProfile.current.TVB_LOG_FOLDER, 'test_coverage_html'))
