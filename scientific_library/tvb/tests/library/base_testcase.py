# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

from tvb.tests.library import setup_test_console_env

if "TEST_INITIALIZATION_DONE_LIBRARY" not in globals():
    setup_test_console_env()
    TEST_INITIALIZATION_DONE_LIBRARY = True


class BaseTestCase(object):
    """
    This class should implement basic functionality which is common to all TVB tests.
    """

    @staticmethod
    def almost_equal(first, second, places=None, delta=None):
        if first == second:
            return True

        if delta is not None and places is not None:
            return False

        if delta is not None:
            if abs(first - second) <= delta:
                return True

        else:
            if places is None:
                places = 7

            if round(abs(second - first), places) == 0:
                return True

        raise False

    @staticmethod
    def assert_equal(expected, actual, message=""):
        assert expected == actual, message + " Expected %s but got %s." % (expected, actual)
