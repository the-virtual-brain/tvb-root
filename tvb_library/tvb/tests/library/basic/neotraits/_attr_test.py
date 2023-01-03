# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

from tvb.tests.library.base_testcase import BaseTestCase
from tvb.basic.neotraits._attr import Range


class TestRange(BaseTestCase):

    def assert_with_epsilon_error(self, list_one, list_two):
        epsilon = 0.000001
        value_for_return = True
        if len(list_one) != len(list_two):
            return False
        for i in range(0, len(list_one)):
            if (list_one[i] + epsilon) <= list_two[i] <= (list_one[i] + epsilon):
                continue
            elif (list_one[i] - epsilon) <= list_two[i] <= (list_one[i] - epsilon):
                continue
            elif (list_one[i] - epsilon) <= list_two[i] <= (list_one[i] + epsilon):
                continue
            elif (list_one[i] + epsilon) <= list_two[i] <= (list_one[i] - epsilon):
                continue
            else:
                value_for_return = False
        return value_for_return

    def test_generates_range_with_start_and_stop_provided(self):
        floats = list(Range(lo=0.0, hi=3.0, step=1.0).to_array())
        assert TestRange.assert_with_epsilon_error(self, floats, list([0.0, 1.0, 2.0]))

    def test_generates_range_with_another_start_stop_and_step(self):
        floats = list(Range(lo=1.0, hi=8.0, step=2.0).to_array())
        assert TestRange.assert_with_epsilon_error(self, floats, list([1.0, 3.0, 5.0, 7.0]))

    def test_generates_range_with_start_stop_and_step_smaller_than_one(self):
        floats = list(Range(lo=0.0, hi=0.5, step=0.1).to_array())
        assert TestRange.assert_with_epsilon_error(self, floats, list([0.0, 0.1, 0.2, 0.3, 0.4]))

    def test_generates_range_with_start_stop_and_periodic_repeating_step(self):
        floats = list(Range(lo=0.0, hi=1.1, step=1. / 3).to_array())
        assert TestRange.assert_with_epsilon_error(self, floats,
                                                   list([0.0, 0.333333333333, 0.666666666666, 0.999999999999]))

    def test_generates_range_with_negative_end(self):
        floats = list(Range(lo=1.0, hi=-3.0, step=-1.0).to_array())
        assert TestRange.assert_with_epsilon_error(self, floats, list([1.0, 0.0, -1.0, -2.0]))

    def test_generates_range_with_negative_end_including_both(self):
        floats = list(Range(lo=-3.0, hi=1.1, step=1.0).to_array())
        assert TestRange.assert_with_epsilon_error(self, floats, list([-3.0, -2.0, -1.0, 0.0, 1.0]))

    def test_generates_range_with_challenging_float_point_arithmetics(self):
        floats = list(Range(lo=0.0, hi=2.2, step=0.7).to_array())
        assert TestRange.assert_with_epsilon_error(self, floats, list([0.0, 0.7, 1.4, 2.1]))
