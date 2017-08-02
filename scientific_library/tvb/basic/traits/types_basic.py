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
This module describes the simple of traited attributes one might needs on a class.

If your subclass should be mapped to a database table (true for most entities
that will be reused), use MappedType as superclass.

If you subclass is supported natively by SQLAlchemy, subclass Type, otherwise
subclass MappedType.

Important:
- Type - traited, possible mapped to db *col*
- MappedType - traited, mapped to db *table*


.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: marmaduke <duke@eml.cc>
.. moduleauthor:: Paula Sanz-Leon <paula.sanz-leon@univ-amu.fr>
"""

import json
import numpy
from decimal import Decimal
import tvb.basic.traits.core as core
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)


class String(core.Type):
    """
    Traits type that wraps a Python string.
    """
    wraps = (str, unicode)



class Bool(core.Type):
    """
    Traits type wrapping Python boolean primitive. 
    The only instances of Python bool are True and False.
    """
    wraps = bool


class Integer(core.Type):
    """
    Traits type that wraps Numpy's int32.
    """
    wraps = (int, long)



class Float(core.Type):
    """
    Traits type that wraps Numpy's float64.
    """
    wraps = (float, numpy.float32, int)


class Complex(core.Type):
    """
    Traits type that wraps Numpy's complex64.
    """
    wraps = numpy.complex64


class MapAsJson():
    """Add functionality of converting from/to JSON"""


    def __get__(self, inst, cls):
        if inst is not None and self.trait.bound and hasattr(inst, '_' + self.trait.name):

            if hasattr(inst, '__' + self.trait.name):
                return getattr(inst, '__' + self.trait.name)

            string = getattr(inst, '_' + self.trait.name)
            if string is None or (not isinstance(string, (str, unicode))):
                return string
            if len(string) < 1:
                return None
            json_value = self.from_json(string)

            # Cache for future usages (e.g. Stimulus.spatial should be the same instance on multiple accesses)
            setattr(inst, '__' + self.trait.name, json_value)
            return json_value
        return self


    @staticmethod
    def to_json(entity):
        return json.dumps(entity)


    @staticmethod
    def from_json(string):
        return json.loads(string)


    @staticmethod
    def decode_map_as_json(dct):
        """
        Used in the __convert_to_array to get an equation from the UI corresponding string.
        """
        for key, value in dct.items():
            if isinstance(value, unicode) and '__mapped_module' in value:
                dict_value = json.loads(value)
                if '__mapped_module' not in dict_value:
                    dct[key] = MapAsJson.decode_map_as_json(dict_value)
                else:
                    modulename = dict_value['__mapped_module']
                    classname = dict_value['__mapped_class']
                    module_entity = __import__(modulename, globals(), locals(), [classname])
                    class_entity = eval('module_entity.' + classname)
                    loaded_entity = class_entity.from_json(value)
                    dct[key] = loaded_entity
        return dct

    class MapAsJsonEncoder(json.JSONEncoder):
        """
        Used before any save to the database to encode Equation type objects.
        """


        def default(self, obj):
            if isinstance(obj, MapAsJson):
                return obj.to_json(obj)
            else:
                return json.JSONEncoder.default(self, obj)


class Sequence(MapAsJson, String):
    """
    Traits type base class that wraps python sequence 
    python types (containers)
    """
    wraps = (dict, list, tuple, set, slice, numpy.ndarray)


class Enumerate(Sequence):
    """
    Traits type that mimics an enumeration.
    """
    wraps = numpy.ndarray


    def __get__(self, inst, cls):
        if inst is None:
            return self
        if self.trait.bound:
            return numpy.array(super(Enumerate, self).__get__(inst, cls))
        return numpy.array(self.trait.value)


    def __set__(self, inst, value):
        if not isinstance(value, list):
            # So it works for simple selects aswell as multiple selects
            value = [value]
        if self.trait.select_multiple:
            super(Enumerate, self).__set__(inst, value)
        else:
            # Bypass default since that only accepts arrays for multiple selects
            setattr(inst, '_' + self.trait.name, self.to_json(value))
            self.trait.value = value


class Dict(Sequence):
    """
    Traits type that wraps a python dict.
    """
    wraps = dict


class Set(Sequence):
    """
    Traits type that wraps a python set.
    """
    wraps = set


class Tuple(Sequence):
    """
    Traits type that wraps a python tuple.
    """
    wraps = tuple


    def __get__(self, inst, cls):
        list_value = super(Tuple, self).__get__(inst, cls)

        if isinstance(list_value, list):
            return list_value[0], list_value[1]

        return list_value


class List(Sequence):
    """
    Traits type that wraps a Python list.
    """
    wraps = (list, numpy.ndarray)


class Slice(Sequence):
    """
    Useful of for specifying views or slices of containers.
    """
    wraps = slice


class Range(core.Type):
    """
    Range is a range that can have multiplicative or additive step sizes.
    Values generated by Range are by default in the interval [start, stop).
    Different flags/modes can be set to include/exclude one or both bounds.
    See the corresponding unittest for more examples.

    Instances of Range will not generate their discrete values automatically,
    but these values can be obtained by converting to a list

    Multiplicative ranges are not yet supported by the web ui.

        >>> range_values = list(Range(lo=0.0, hi=1.0, step=0.1))

        [0.0,
         0.1,
         0.2,
         0.30000000000000004,
         0.4,
         0.5,
         0.6000000000000001,
         0.7000000000000001,
         0.8,
         0.9]

        or by direct iteration:

        >>> for val in Range(lo=0.0, hi=1.0, step=0.1):
                print val

        0.1
        0.2
        0.3
        0.4
        0.5
        0.6
        0.7
        0.8
        0.9


        >>> for val in Range(lo=0.0, hi=2.0, step=1.0):
                print val
        0.0
        1.0
        2.0

        >>> for val in Range(lo=0.0, hi=3.0, step=1.0):
                print val
        0.0
        1.0
        2.0

    using a fixed multiplier
        >>> for val in Range(lo=0.0, hi=5.0, base=2.0):
                print val
        0.0
        2.0
        4.0

        >>> for val in Range(lo=1.0, hi=8.O, base=2.0):
                print val
        1.0
        2.0
        4.0

    """

    # TODO: Improve Range with a fix multiplier and logarithmic range.
    lo = Float(default=0, doc='start of range')
    hi = Float(default=None, doc='end of range')

    step = Float(default=None, doc='fixed step size between elements (for additive). It has priority over "base"')
    base = Float(default=2.0, doc='fixed multiplier between elements')

    # These modes/flags only work for additive step
    MODE_EXCLUDE_BOTH = 0  # flag to exclude both lower and upper bounds
    MODE_INCLUDE_START = 1  # flag to include lo, exclude hi
    MODE_INCLUDE_END = 2  # flag to exclude lo, include hi
    MODE_INCLUDE_BOTH = 3  # flag to include both lower and upper bounds
    mode = Integer(default=1, doc='default behaviour, equivalent to include lo, exclude hi')


    def __iter__(self):
        """ Get valid values in interval"""

        def gen():
            if self.step:
                start, stop, step = self.args_to_decimal(self.lo, self.hi, self.step)
                current = start
                if not self.mode & self.MODE_INCLUDE_START:
                    current += step

                while True:
                    if self.out_of_range(current, stop, self.mode, step):
                        raise StopIteration
                    yield float(current)
                    current += step
            else:
                if self.base <= 1.0:
                    msg = "Invalid base value: %s"
                    LOG.error(msg % str(self.base))
                else:
                    val = self.lo
                    if val == 0:
                        yield val
                        val += self.base
                    while val < self.hi:
                        yield val
                        val *= self.base

        return gen()

    def out_of_range(self, current, stop, mode, step):
        if mode & self.MODE_INCLUDE_END and step > 0:
            return current > stop
        if mode & self.MODE_INCLUDE_END and step < 0:
            return current < stop
        elif step < 0:
            return current <= stop
        return current >= stop


    def args_to_decimal(self, start, stop, step):
        if start > stop and stop is not None:
            step = -step
        if stop is None:
            stop = Decimal(str(start))
            start = Decimal(str(0.0))
        else:
            stop = Decimal(str(stop))
            start = Decimal(str(start))
        step = Decimal(str(step))
        return start, stop, step


class ValidationRange(core.Type):
    """
    ValidationRange represents a Range used only for validating a number.
    """


class JSONType(String):
    """
    Wrapper over a String which holds a serializable object.
    On set/get JSON load/dump will be called.
    """


    def __get__(self, inst, cls):
        if inst:
            string = super(JSONType, self).__get__(inst, cls)
            if string is None or (not isinstance(string, (str, unicode))):
                return string
            if len(string) < 1:
                return None
            return json.loads(string)
        return super(JSONType, self).__get__(inst, cls)


    def __set__(self, inst, value):
        if not isinstance(value, (str, unicode)):
            value = json.dumps(value)
        super(JSONType, self).__set__(inst, value)


class DType(String):
    """
    Traits type that wraps a Numpy dType specification.
    """

    wraps = (numpy.dtype, str)
    defaults = ((numpy.float64,), {})


    def __get__(self, inst, cls):
        if inst:
            type_ = super(DType, self).__get__(inst, cls)
            return str(type_).replace("<type '", '').replace("'>", '')
        return super(DType, self).__get__(inst, cls)


    def __set__(self, inst, value):
        super(DType, self).__set__(inst, str(value))
