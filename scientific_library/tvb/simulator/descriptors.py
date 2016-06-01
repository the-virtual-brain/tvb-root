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
Data descriptors for declaring workspace for algorithms and checking usage.

.. moduleauthor:: Marmaduke Woodman <mmwoodman@gmail.com>

"""

import numpy
import traceback
import collections

# TODO turn off checking at run time?

# StaticAttr prevents descriptors from placing instance-owned, descriptor storage

# Not currently used; doesn't (yet)  work well with descriptors below.
class StaticAttr(object):
    "Base class which requires all attributes to be declared at class level or set during __init__."

    def __setattr__(self, name, value, set_it_anyway=False):
        attr_exists = hasattr(self, name)
        attr_init = any(['__init__' == fn for _, _, fn, _ in traceback.extract_stack()])
        if attr_exists or attr_init or set_it_anyway:
            super(StaticAttr, self).__setattr__(name, value)
        else:
            raise AttributeError('%r has no attr %r or not set during __init__.' % (self, name))


class ImmutableAttrError(AttributeError):
    "Error due to modifying an immutable attribute."
    pass


class IncorrectTypeAttrError(AttributeError):
    "Error due to using incorrect type to set attribute value."
    pass


class NDArray(StaticAttr):
    "Data descriptor for a NumPy array, with type, mutability and shape checking."

    # Owner can provide array constructor via _array_ctor attr, i.e. NumPy or PyOpenCL, etc.

    def __init__(self, shape, dtype, mutable=False):
        self.shape = shape # may have strings which eval in owner ns
        self.dtype = dtype
        self.mutable = mutable
        self.key = '__%d' % (id(self), )

    def _make_array(self, instance):
        shape = []
        for dim in self.shape:
            if isinstance(dim, str):
                dim = getattr(instance, dim)
            if isinstance(dim, Dim):
                dim = getattr(instance, dim.key) # type: int
            shape.append(dim)
        array = numpy.empty(shape, self.dtype)
        if not self.mutable and hasattr(array, 'setflags'):
            array.setflags(write=False)
        return array

    def __get__(self, instance, _):
        if instance is None:
            return self
        if not hasattr(instance, self.key):
            setattr(instance, self.key, self._make_array(instance))
        return getattr(instance, self.key)

    def __set__(self, instance, value):
        is_init = getattr(instance, self.key + '_is_init', False)
        array = self.__get__(instance, None)
        if not self.mutable:
            if is_init:
                raise ImmutableAttrError('Cannot modify an immutable ndarray.')
            else:
                array.setflags(write=True)
        # set with [:] to ensure shape compat and safe type coercion
        array[:] = value
        if not self.mutable:
            array.setflags(write=False)
        if not is_init:
            setattr(instance, self.key + '_is_init', True)


class InstanceOf(StaticAttr):
    "Data descriptor for object instance, checking mutability and type."

    def __init__(self, otype, mutable=False):
        self.otype = otype
        self.mutable = mutable
        self.key = '__%d' % (id(self), )

    def __get__(self, instance, _):
        if instance is None:
            return self
        return getattr(instance, self.key)

    def __set__(self, instance, value):
        if not isinstance(value, self.otype):
            raise IncorrectTypeAttrError(
                '%r not an instance of %r.' % (value, self.otype))
        if hasattr(instance, self.key) and not self.mutable:
            raise ImmutableAttrError('Cannot modify an immutable attribute.')
        setattr(instance, self.key, value)


class Dim(InstanceOf):
    "Specialization of InstanceOf to int type."
    def __init__(self):
        super(Dim, self).__init__(int, mutable=False)


# TODO
class Workspace(StaticAttr):
    pass