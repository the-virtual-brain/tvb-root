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

"""
Data descriptors for declaring workspace for algorithms and checking usage.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import numpy
import collections
import weakref
import six
from .common import get_logger

LOG = get_logger(__name__)

# StaticAttr prevents descriptors from placing instance-owned, descriptor storage

class StaticAttr(object):
    "Base class which requires all attributes to be declared at class level."

    def __setattr__(self, name, value):
        attr_exists = hasattr(self, name)
        if not hasattr(self, name):
            raise AttributeError('%r has no attr %r.' % (self, name))
        else:
            super(StaticAttr, self).__setattr__(name, value)


class ImmutableAttrError(AttributeError):
    "Error due to modifying an immutable attribute."
    pass


class IncorrectTypeAttrError(AttributeError):
    "Error due to using incorrect type to set attribute value."
    pass


class NDArray(StaticAttr):
    "Data descriptor for a NumPy array, with type, mutability and shape checking."

    # Owner can provide array constructor via _array_ctor attr, i.e. NumPy or PyOpenCL, etc. ?

    State = collections.namedtuple('State', 'array initialized')

    shape, dtype, read_only, instance_state = (), None, True, {}

    def __init__(self, shape, dtype, read_only=True):
        self.shape = shape # may have strings which eval in owner ns
        self.dtype = dtype
        self.read_only = read_only
        self.instance_state = weakref.WeakKeyDictionary()

    def _make_array(self, instance):
        shape = []
        for dim in self.shape:
            if isinstance(dim, str):
                dim = getattr(instance, dim)
            elif isinstance(dim, Dim):
                if instance in dim.instance_state:
                    dim = dim.instance_state[instance].value
                else:
                    raise AttributeError('Dimension referenced before definition.')
            else:
                raise TypeError('expect int, str but found %r' % (type(dim), ))
            shape.append(dim)
        array = numpy.empty(shape, self.dtype)
        if self.read_only and hasattr(array, 'setflags'):
            array.setflags(write=False)
        return array

    def _get_or_create_state(self, instance):
        if instance not in self.instance_state:
            array = self._make_array(instance)
            self.instance_state[instance] = NDArray.State(array, False)
        return self.instance_state[instance]

    def __get__(self, instance, _):
        if instance is None:
            LOG.debug('NDArray returning self for None instance.')
            return self
        else:
            return self._get_or_create_state(instance).array

    def __set__(self, instance, value):
        state = self._get_or_create_state(instance)
        if self.read_only:
            if state.initialized:
                raise ImmutableAttrError('Cannot modify an immutable ndarray.')
            else:
                state.array.setflags(write=True)
        # set with [:] to ensure shape compat and safe type coercion
        _, value = numpy.broadcast_arrays(state.array, value)
        state.array[:] = value
        if self.read_only:
            state.array.setflags(write=False)
        if not state.initialized:
            self.instance_state[instance] = NDArray.State(state.array, True)


class Final(object):
    "A descriptor for an attribute, possibly type-checked, that once initialized, cannot be changed."

    State = collections.namedtuple('State', 'value initialized')

    def __init__(self, type=None):
        self.instance_state = weakref.WeakKeyDictionary()
        self.type = type

    def _get_or_create_state(self, instance):
        if instance not in self.instance_state:
            self.instance_state[instance] = Final.State(None, False)
        return self.instance_state[instance]

    def _correct_type(self, value):
        return isinstance(value, self.type)

    def __set__(self, instance, value):
        state = self._get_or_create_state(instance) # type: Final.State
        if state.initialized:
            raise AttributeError('final attribute cannot be set.')
        else:
            if self.type and not self._correct_type(value):
                raise AttributeError('value %r does not match expected type %r'
                                      % (value, self.type))
            self.instance_state[instance] = Final.State(value, True)

    def __get__(self, instance, owner):
        if instance is None:
            LOG.debug('Final returning self for None instance.')
            return self
        else:
            return self._get_or_create_state(instance).value


class Dim(Final):
    "Specialization of Final to int/long type."

    def __init__(self):
        super(Dim, self).__init__(int)

    def _correct_type(self, value):
        return isinstance(value, six.integer_types) \
               or numpy.issubdtype(type(value), numpy.integer)


# TODO
class Workspace(StaticAttr):
    pass