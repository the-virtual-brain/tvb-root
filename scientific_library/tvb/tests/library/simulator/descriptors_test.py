# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

"""
Tests for data descriptors for declaring workspace for algorithms and checking usage.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import pytest
import numpy
from tvb.simulator.descriptors import StaticAttr, NDArray, ImmutableAttrError, Final, Dim
import six


class TestStaticAttr(object):
    """Test API of StaticAttr base class."""

    def setup_method(self):
        class TestClass(StaticAttr):
            x = 5
            z = 2

            def __init__(self):
                self.z = 42

            def set_x(self):
                self.x = 6

            def set_y(self):
                self.y = 'hello'

        self.test_class = TestClass()

    def test_set_existing_ok(self):
        assert self.test_class.z == 42
        assert self.test_class.x == 5
        self.test_class.set_x()
        assert self.test_class.x == 6

    def test_set_non_existing_y(self):
        with pytest.raises(AttributeError):
            self.test_class.set_y()


class TestNDArray(object):
    """Test API of NDArray descriptor."""

    def setup_method(self):
        class PointSet(object):
            positions = NDArray(('n_point', 'dim'), 'f')
            counts = NDArray(('n_point',), 'i', read_only=False)

            def __init__(self, n_point, dim=3):
                self.n_point = n_point
                self.dim = dim

        self.ps50 = PointSet(50)
        self.ps25 = PointSet(25)

    def test_shape_dtype(self):
        pos = self.ps50.positions
        cnt = self.ps50.counts
        assert pos.shape == (50, 3)
        assert cnt.shape == (50,)
        assert pos.dtype == numpy.float32
        assert cnt.dtype == numpy.int32
        pos = self.ps25.positions
        cnt = self.ps25.counts
        assert pos.shape == (25, 3)
        assert cnt.shape == (25,)

        assert self.ps50.positions.shape != self.ps25.positions.shape

    def _set_positions(self):
        self.ps50.positions = numpy.random.randn(*self.ps50.positions.shape)

    def _set_counts(self):
        self.ps50.counts = numpy.random.randint(0, 5, size=self.ps50.counts.shape)

    def test_mutability(self):
        self._set_positions()
        with pytest.raises(ImmutableAttrError):
            self.ps50.positions = numpy.random.randn(*self.ps50.positions.shape)

        self._set_counts()
        self._set_counts()

    def _set_incorrect_shape(self):
        self.ps50.counts = numpy.array([2, 3])

    def test_incorrect_shape(self):
        with pytest.raises(ValueError):
            self._set_incorrect_shape()


class TestFinal(object):
    def setup_method(self):
        class Inst(object):
            n = Final()
            m0, m1, m2, m3, m4 = Dim(), Dim(), Dim(), Dim(), Dim()
            x = Final(float)

            def __init__(self):
                self.n = 42

            def change_n(self):
                self.n = 'asdf'

            def set_x_int(self):
                self.x = 32

            def set_x_float(self):
                self.x = 2.3

        self.Inst = Inst
        self.foo = Inst()
        self.bar = Inst()

    def test_immutability(self):
        assert self.foo.n == 42
        with pytest.raises(AttributeError):
            self.foo.change_n()

    def test_class_get_descriptor(self):
        assert isinstance(self.Inst.n, Final)

    def test_count_weakref(self):
        assert len(self.Inst.n.instance_state) == 2
        delattr(self, 'foo')
        assert len(self.Inst.n.instance_state) == 1
        delattr(self, 'bar')
        assert len(self.Inst.n.instance_state) == 0

    def test_type_check(self):
        with pytest.raises(AttributeError):
            self.foo.set_x_int()

        self.foo.set_x_float()
        assert self.foo.x == 2.3

    def test_dim_accepts_many_int_types(self):
        int_types = list(six.integer_types) + [numpy.int32, numpy.uint32, numpy.int64]
        for i, int_type in enumerate(six.integer_types):
            setattr(self.foo, 'm%d' % i, int_type(0))
