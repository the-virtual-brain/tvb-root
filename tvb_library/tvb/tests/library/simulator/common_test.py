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

"""
Test for tvb.simulator.common module

.. moduleauthor:: Paula Sanz Leon <paula@tvb.invalid>

"""

import pytest
import numpy
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.simulator.backend.ref import RefBase
from tvb.simulator import common


class TestCommon(BaseTestCase):
    """
    Define test cases for common:
        - initialise each class
        - check default parameters 
        
    """

    def test_struct(self):
        st = common.Struct(x=42.0, y=33.0)
        assert st.x == 42.0
        assert st.y == 33.0

    def test_linear_interpolation(self):
        t_start = 0.0
        t_end = 1.0
        y_start = 4.0
        y_end = 8.0
        t_mid = 0.5
        val = RefBase.linear_interp1d(t_start, t_end, y_start, y_end, t_mid)
        assert val == 6.0

    @pytest.mark.skipif(not hasattr(numpy.add, 'at'),
                        reason='Cannot test fallback numpy.add.at implementation without '
                               'a version of NumPy which provides this ufunc method (>=1.8).')
    def test_add_at(self):
        ri = numpy.random.randint
        for nd in range(1, 5):
            m, n, rest = ri(3, 50), ri(51, 100), tuple(ri(3, 10, nd - 1))
            source = ri(-100, 100, (n,) + rest)
            map = ri(0, m, n)
            expected, actual = numpy.zeros((2, m) + rest)
            numpy.add.at(expected, map, source)
            RefBase._add_at(actual, map, source)
            assert numpy.allclose(expected, actual)
