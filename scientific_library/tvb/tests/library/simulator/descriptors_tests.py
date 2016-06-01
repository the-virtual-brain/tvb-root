# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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

"""
Tests for data descriptors for declaring workspace for algorithms and checking usage.

.. moduleauthor:: Marmaduke Woodman <mmwoodman@gmail.com>

"""

import unittest
import numpy
from tvb.simulator.descriptors import (StaticAttr, NDArray, ImmutableAttrError,
    IncorrectTypeAttrError, InstanceOf, Dim)


class TestStaticAttr(unittest.TestCase):
    "Test API of StaticAttr base class."

    def setUp(self):
        class TestClass(StaticAttr):
            x = 5
            def __init__(self):
                self.z = 42
            def set_x(self):
                self.x = 6
            def set_y(self):
                self.y = 'hello'
        self.test_class = TestClass()

    def test_set_existing_ok(self):
        self.assertEqual(self.test_class.z, 42)
        self.assertEqual(self.test_class.x, 5)
        self.test_class.set_x()
        self.assertEqual(self.test_class.x, 6)

    def test_set_non_existing_y(self):
        self.assertRaises(AttributeError, self.test_class.set_y)


class TestNDArray(unittest.TestCase):
    "Test API of NDArray descriptor."

    def setUp(self):
        class PointSet(object):
            positions = NDArray(('n_point', 'dim'), 'f')
            counts = NDArray(('n_point',), 'i', mutable=True)
            def __init__(self, n_point, dim=3):
                self.n_point = n_point
                self.dim = dim
        self.ps50 = PointSet(50)
        self.ps25 = PointSet(25)

    def test_shape_dtype(self):
        pos = self.ps50.positions
        cnt = self.ps50.counts
        self.assertEqual(pos.shape, (50, 3))
        self.assertEqual(cnt.shape, (50, ))
        self.assertEqual(pos.dtype, numpy.float32)
        self.assertEqual(cnt.dtype, numpy.int32)
        pos = self.ps25.positions
        cnt = self.ps25.counts
        self.assertEqual(pos.shape, (25, 3))
        self.assertEqual(cnt.shape, (25, ))

        self.assertNotEqual(self.ps50.positions.shape, self.ps25.positions.shape)

    def _set_positions(self):
        self.ps50.positions = numpy.random.randn(*self.ps50.positions.shape)

    def _set_counts(self):
        self.ps50.counts = numpy.random.randint(0, 5, size=self.ps50.counts.shape)

    def test_mutability(self):
        self._set_positions()
        self.assertRaises(ImmutableAttrError, self._set_positions)
        self._set_counts()
        self._set_counts()

    def _set_incorrect_shape(self):
        self.ps50.counts = numpy.array([2, 3])

    def test_incorrect_shape(self):
        self.assertRaises(ValueError, self._set_incorrect_shape)


class TestInstanceOf(unittest.TestCase):
    "Test API of InstanceOf and Dim descriptors."

    def setUp(self):
        class TestClass(object):
            n = Dim()
            dt = InstanceOf(float)
            x = InstanceOf(float, mutable=True)
        self.test_class = TestClass()

    def _set_n(self):
        self.test_class.n = 5

    def _set_dt(self):
        self.test_class.dt = 0.124

    def _set_dt_int(self):
        self.test_class.dt = 100

    def _set_n_float(self):
        self.test_class.n = 1.2

    def _set_x_int(self):
        self.test_class.x = 234

    def _set_x(self):
        self.test_class.x = 2.0

    def test_type_checks(self):
        self._set_n()
        self.assertEqual(self.test_class.n, 5)
        self._set_dt()
        self.assertEqual(self.test_class.dt, 0.124)
        self.test_class.x = 23.0
        self.assertRaises(IncorrectTypeAttrError, self._set_dt_int)
        self.assertRaises(IncorrectTypeAttrError, self._set_n_float)
        self.assertRaises(IncorrectTypeAttrError, self._set_x_int)

    def test_mutability(self):
        self._set_n()
        self.assertRaises(ImmutableAttrError, self._set_n)
        self._set_x()
        self._set_x()

