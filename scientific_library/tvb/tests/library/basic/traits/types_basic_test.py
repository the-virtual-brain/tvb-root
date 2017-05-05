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

Floating-point range generator (IEEE-754-proof)
Original: https://gist.github.com/diogobaeder/1239977


.. moduleauthor:: Paula Sanz Leon <paula.sanz-leon@unvi-amu.fr>
"""


if __name__ == "__main__":
    from tvb.tests.library import setup_test_console_env
    setup_test_console_env()

import unittest
from tvb.basic.traits.types_basic import Range

 
 
class RangeTest(unittest.TestCase):
    def test_generates_range_with_only_stop_provided(self):
        floats = list(Range(hi=3.0, step=1.0))
        self.assertEqual(floats, [0.0, 1.0, 2.0])
 
    def test_generates_range_with_start_and_stop(self):
        floats = list(Range(lo=1.0, hi=3.0, step=1.0))
        self.assertEqual(floats, [1.0, 2.0])
 
    def test_generates_range_with_start_stop_and_step_smaller_than_one(self):
        floats = list(Range(lo=0.0, hi=0.5, step=0.1))
        self.assertEqual(floats, [0.0, 0.1, 0.2, 0.3, 0.4])

    def test_generates_range_with_start_stop_and_periodic_repeating_step(self):
        floats = list(Range(lo=0.0, hi=1.0, step=1./3))
        self.assertEqual(floats,[0.0, 0.333333333333, 0.666666666666, 0.999999999999])
 
    def test_generates_range_with_start_and_end_included(self):
        floats = list(Range(lo=0.0, hi=3.0, step=1.0, mode=Range.MODE_INCLUDE_BOTH))
        self.assertEqual(floats, [0.0, 1.0, 2.0, 3.0])
 
    def test_generates_range_with_start_and_end_excluded(self):
        floats = list(Range(lo=0.0, hi=3.0, step=1.0, mode=Range.MODE_EXCLUDE_BOTH))
        self.assertEqual(floats, [1.0, 2.0])
 
    def test_generates_range_with_only_end_included(self):
        floats = list(Range(lo=0.0, hi=3.0, step=1.0, mode=Range.MODE_INCLUDE_END))
        self.assertEqual(floats, [1.0, 2.0, 3.0])
 
    def test_generates_range_with_negative_end(self):
        floats = list(Range(lo=1.0, hi=-3.0, step=1.0))
        self.assertEqual(floats, [1.0, 0.0, -1.0, -2.0])
 
    def test_generates_range_with_negative_end_including_both(self):
        floats = list(Range(lo=1.0, hi=-3.0, step=1.0, mode=Range.MODE_INCLUDE_BOTH))
        self.assertEqual(floats, [1.0, 0.0, -1.0, -2.0, -3.0])
 
    def test_generates_range_with_negative_end_but_excluding_start(self):
        floats = list(Range(lo=1.0, hi=-3.0, step=1.0, mode=Range.MODE_INCLUDE_END))
        self.assertEqual(floats, [0.0, -1.0, -2.0, -3.0])
 
    def test_generates_range_with_challenging_float_point_arithmetics(self):
        floats = list(Range(lo=0.0, hi=2.2, step=0.7))
        self.assertEqual(floats, [0.0, 0.7, 1.4, 2.1])
 
def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(RangeTest))
    return test_suite

if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE) 