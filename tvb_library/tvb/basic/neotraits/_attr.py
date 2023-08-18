# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
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
This private module implements concrete declarative attributes
"""
import inspect
import numpy
import types
import typing
from collections.abc import Sequence
from ._declarative_base import _Attr, MetaType
from .ex import TraitValueError, TraitTypeError, TraitAttributeError, TraitFinalAttributeError
from tvb.basic.logger.builder import get_logger

if typing.TYPE_CHECKING:
    from ._core import HasTraits

# a logger for the whole traits system
log = get_logger('tvb.traits')


class Attr(_Attr):
    """
    An Attr declares the following about the attribute it describes:
    * the type
    * a default value shared by all instances
    * if the value might be missing
    * documentation
    It will resolve to attributes on the instance.
    """

    # This class is a python data descriptor.
    # For an introduction see https://docs.python.org/2/howto/descriptor.html

    def __init__(
            self, field_type, default=None, doc='', label='', required=True, final=False, choices=None
    ):
        # type: (type, typing.Any, str, str, bool, bool, typing.Optional[tuple]) -> None
        """
        :param field_type: the python type of this attribute
        :param default: A shared default value. Behaves like class level attribute assignment.
                        Take care with mutable defaults.
        :param doc: Documentation for this field.
        :param label: A short description.
        :param required: required fields should not be None.
        :param final: Final fields can only be assigned once.
        :param choices: A tuple of the values that this field is allowed to take.
        """
        super(Attr, self).__init__()
        self.field_type = field_type
        self.default = default
        self.doc = doc
        self.label = label
        self.required = bool(required)
        self.final = bool(final)
        self.choices = choices

    def __validate(self, value):
        """ check field_type and choices """
        if not isinstance(value, self.field_type) and not (
                inspect.isclass(self.default) and issubclass(value, self.field_type)):
            raise TraitTypeError("Attribute can't be set to an instance of {}".format(type(value)), attr=self)
        if self.choices is not None:
            if value not in self.choices and not (value is None and not self.required):
                raise TraitValueError("Value {!r} must be one of {}".format(value, self.choices), attr=self)

    # subclass api

    def _post_bind_validate(self):
        # type: () -> None
        """
        Validates this instance of Attr.
        This is called just after field_name is set, by MetaType.
        We do checks here and not in init in order to give better error messages.
        Attr should be considered initialized only after this has run
        """
        if not isinstance(self.field_type, type):
            msg = 'Field_type must be a type not {!r}. Did you mean to declare a default?'.format(
                self.field_type
            )
            raise TraitTypeError(msg, attr=self)

        skip_default_checks = self.default is None or isinstance(self.default, types.FunctionType)

        if not skip_default_checks:
            self.__validate(self.default)

        # heuristic check for mutability. might be costly. hasattr(__hash__) is fastest but less reliable
        try:
            hash(self.default)
        except TypeError:
            log.warning('Field seems mutable and has a default value. '
                        'Consider using a lambda as a value factory \n   attribute {}'.format(self))
        # we do not check here if we have a value for a required field
        # it is too early for that, owner.__init__ has not run yet

    def _validate_set(self, instance, value):
        # type: ('HasTraits', typing.Any) -> typing.Any
        """
        Called before updating the value of an attribute.
        It checks the type *AND* returns the valid value.
        You can override this for further checks. Call super to retain this check.
        Raise if checks fail.
        You should return a cleaned up value if validation passes
        """
        if value is None:
            if self.required:
                raise TraitValueError("Attribute is required. Can't set to None", attr=self)
            else:
                return value

        self.__validate(value)
        return value

    # descriptor protocol
    def __get__(self, instance, owner):
        # type: (typing.Optional['HasTraits'], 'MetaType') -> typing.Any
        self._assert_have_field_name()
        if instance is None:
            # called from class, not an instance
            return self
        # data is stored on the instance in a field with the same name
        # If field is not on the instance yet, return the class level default
        # (this attr instance is a class field, so the default is for the class)
        # This is consistent with how class fields work before they are assigned and become instance bound
        if self.field_name not in instance.__dict__:
            if (self.field_type != types.FunctionType and isinstance(self.default, types.FunctionType)
                    or inspect.isclass(self.default) and issubclass(self.default, self.field_type)):
                default = self.default()
            else:
                default = self.default

            # Unless we store the default on the instance, this will keep returning self.default()
            # when the default is a function. So if the default is mutable, any changes to it are
            # lost as a new one is created every time.
            instance.__dict__[self.field_name] = default

        value = instance.__dict__[self.field_name]
        if self.required and value is None:
            raise TraitAttributeError('required attribute referenced before assignment. '
                                      'Use a default or assign a value before reading it', attr=self)
        return value

    def __set__(self, instance, value):
        # type: ('HasTraits', typing.Any) -> None
        self._assert_have_field_name()

        if self.final:
            # non-set to set final transition happens when instance stored value becomes not none
            # getattr will call __get__. We want that in order to allow the default to be set.
            # If __set__ is called before a __get__ then no defaults have been assigned.
            # subtlety: if the value of this final field is not set then __get__ will raise
            #           getattr with a default value swallows that exception and returns false
            present_value = getattr(instance, self.field_name, None)
            if present_value is not None:
                raise TraitFinalAttributeError("can't write final attribute")

        value = self._validate_set(instance, value)

        instance.__dict__[self.field_name] = value

    def _defined_on_str_helper(self):
        if self.owner is not None:
            return '{}.{}.{} = {}'.format(
                self.owner.__module__,
                self.owner.__name__,
                self.field_name,
                type(self).__name__
            )
        else:
            return '{}'.format(type(self).__name__)

    def __str__(self):
        return '{}(field_type={}, default={!r}, required={})'.format(
            self._defined_on_str_helper(), self.field_type, self.default, self.required
        )


class Final(Attr):
    """
    An attribute that can only be set once.
    If a default is provided it counts as a set, so it cannot be written to.
    Note that if the default is a mutable type, the value is shared with all instances
    of the owning class.
    We cannot enforce true constancy in python
    """

    def __init__(self, default=None, field_type=None, doc='', label=''):
        """
        :param default: The constant value
        """
        # it would be nice if we could turn the default immutable. But this is unreasonable work in python
        # maybe a deep copy?
        if default is not None:
            field_type = type(default)

        if default is None and field_type is None:
            raise ValueError('Either a default or a field_type is required')

        super(Final, self).__init__(
            field_type=field_type, default=default, doc=doc, label=label, required=True, final=True
        )


class List(Attr):
    """
    The attribute is a list of values.
    Choices and type are reinterpreted as applying not to the list but to the elements of it
    """

    def __init__(self, of=object, default=(), doc='', label='', final=False, choices=None):
        # type: (type, tuple, str, str, bool, typing.Optional[tuple]) -> None
        super(List, self).__init__(
            field_type=Sequence,
            default=default,
            doc=doc,
            label=label,
            required=True,
            final=final,
            choices=None,
        )
        self.element_type = of
        self.element_choices = choices

    def __validate_elements(self, value):
        """ check that all elements are of the declared type and one of the declared choices """
        for i, el in enumerate(value):
            if not isinstance(el, self.element_type):
                raise TraitTypeError("value[{}] can't be of type {}".format(i, type(el)), attr=self)

        if self.element_choices is not None:
            for i, el in enumerate(value):
                if el not in self.element_choices:
                    raise TraitValueError(
                        "value[{}]=={!r} must be one of {}".format(i, el, self.element_choices),
                        attr=self
                    )

    def _post_bind_validate(self):
        super(List, self)._post_bind_validate()
        # check that the default contains elements of the declared type
        self.__validate_elements(self.default)

    def _validate_set(self, instance, value):
        value = super(List, self)._validate_set(instance, value)
        if value is None:
            # value is optional and missing, nothing to do here
            return
        self.__validate_elements(value)
        return value

    # __get__ __set__ here only for typing purposes, for better ide checking and autocomplete

    def __get__(self, instance, owner):
        # type: (typing.Optional[HasTraits], MetaType) -> typing.Sequence
        return super(List, self).__get__(instance, owner)

    def __set__(self, instance, value):
        # type: (HasTraits, typing.Sequence) -> None
        super(List, self).__set__(instance, value)

    def __str__(self):
        return '{}(of={}, default={!r}, required={})'.format(
            self._defined_on_str_helper(), self.element_type, self.default, self.required
        )


class _Number(Attr):

    def __validate(self, value):
        """ value should be safely cast to field type and choices must be enforced """
        if not isinstance(value, (int, float, complex, numpy.number)):
            # we have to check that the value is numeric before the can_cast check
            # as can_cast works with dtype strings as well
            # can_cast('i8', 'i32')
            raise TraitTypeError("can't be set to {!r}. Need a number.".format(value), attr=self)
        if not numpy.can_cast(value, self.field_type, 'safe'):
            raise TraitTypeError("can't be set to {!r}. No safe cast.".format(value), attr=self)
        if self.choices is not None:
            if value not in self.choices:
                raise TraitValueError("value {!r} must be one of {}".format(value, self.choices), attr=self)

    def _post_bind_validate(self):
        if self.default is not None:
            self.__validate(self.default)

    def _validate_set(self, instance, value):
        if value is None:
            if self.required:
                raise TraitValueError("is required. Can't set to None", attr=self)
            else:
                return value

        self.__validate(value)
        return self.field_type(value)


class Int(_Number):
    """
    Declares an integer
    This is different from Attr(field_type=int).
    The former enforces int subtypes
    This allows all integer types, including numpy ones that can be safely cast to the declared type
    according to numpy rules
    """

    def __init__(
            self, field_type=int, default=0, doc='', label='', required=True, final=False, choices=None
    ):
        super(_Number, self).__init__(
            field_type=field_type,
            default=default,
            doc=doc,
            label=label,
            required=required,
            final=final,
            choices=choices,
        )

    def _post_bind_validate(self):
        if not issubclass(self.field_type, (int, numpy.integer)):
            msg = 'field_type must be a python int or a numpy.integer not {!r}.'.format(self.field_type)
            raise TraitTypeError(msg, attr=self)
        # super call after the field_type check above
        super(Int, self)._post_bind_validate()


class Float(_Number):
    """
    Declares a float.
    This is different from Attr(field_type=float).
    The former enforces float subtypes.
    This allows any type that can be safely cast to the declared float type
    according to numpy rules.

    Reading and writing this attribute is slower than a plain python attribute.
    In performance sensitive code you might want to use plain python attributes
    or even better local variables.
    """

    def __init__(
            self, field_type=float, default=0, doc='', label='', required=True, final=False, choices=None
    ):
        super(_Number, self).__init__(
            field_type=field_type,
            default=default,
            doc=doc,
            label=label,
            required=required,
            final=final,
            choices=choices,
        )

    def _post_bind_validate(self):
        if not issubclass(self.field_type, (float, numpy.floating)):
            msg = 'field_type must be a python float or a numpy.floating not {!r}.'.format(self.field_type)
            raise TraitTypeError(msg, attr=self)
        # super call after the field_type check above
        super(Float, self)._post_bind_validate()


class Dim(Final):
    """
    A symbol that defines a dimension in a numpy array shape.
    It can only be set once. It is an int.
    Dimensions have to be set before any NArrays that reference them are used.
    """

    any = object()  # sentinel

    def __init__(self, doc=''):
        super(Dim, self).__init__(field_type=int, doc=doc)


class NArray(Attr):
    """
    Declares a numpy array.
    dtype enforces the dtype. The default dtype is float64.
    An optional symbolic shape can be given, as a tuple of Dim attributes from the owning class.
    The shape will be enforced, but no broadcasting will be done.
    domain declares what values are allowed in this array.
    It can be any object that can be checked for membership
    Defaults are checked if they are in the declared domain.
    For performance reasons this does not happen on every attribute set.
    """

    def __init__(
            self,
            default=None,
            required=True,
            doc='',
            label='',
            dtype=numpy.float64,
            shape=None,
            dim_names=(),
            domain=None,
    ):
        # type: (numpy.ndarray, bool, str, str, typing.Union[numpy.dtype, type, str], typing.Optional[typing.Tuple[Dim, ...]], typing.Tuple[str, ...], typing.Container) -> None
        """
        :param dtype: The numpy datatype. Defaults to float64. This is checked by neotraits.
        :param shape: An optional symbolic shape, tuple of Dim's declared on the owning class
        :param dim_names: Optional names for the names of the dimensions
        :param domain: Any type that can be checked for membership like xrange.
                       Represents the expected domain of the values in the array.
        """

        if numpy.issubdtype(dtype, numpy.integer):
            dtype = numpy.int64
        self.dtype = numpy.dtype(dtype)

        super(NArray, self).__init__(
            field_type=numpy.ndarray, default=default, required=required, doc=doc, label=label
        )
        self.shape = shape
        self.domain = domain  # anything that supports 3.1 in domain
        self.dim_names = dim_names

        if self.shape is not None:  # we have a shape
            if self.dim_names:  # and dim_names
                # ensure that len(shape) == len(dim_names)
                if len(self.shape) != len(self.dim_names):
                    raise TraitValueError('shape contradicts dim_names', attr=self)

            # maybe a over zealous type check
            for d in self.shape:
                if d is not Dim.any and type(d) != Dim:
                    raise TraitValueError("shape elements must be Dim's not {}".format(type(d)), attr=self)
            self.ndim = len(self.shape)
        elif self.dim_names:  # no shape but dim_names
            self.ndim = len(self.dim_names)
        else:
            self.ndim = None

    def __validate(self, value):
        """ check that ndim's and dtypes match"""
        if self.ndim is not None and value.ndim != self.ndim:
            raise TraitValueError("can't be set to an array with ndim {}".format(value.ndim), attr=self)

        if not numpy.can_cast(value.dtype, self.dtype, 'safe'):
            raise TraitTypeError("can't be set to an array of dtype {}".format(value.dtype), attr=self)

    def _post_bind_validate(self):
        if self.default is None:
            return
        if not isinstance(self.default, numpy.ndarray):
            msg = 'default {} should be a numpy.ndarray'.format(self.default)
            raise TraitTypeError(msg, attr=self)

        self.__validate(self.default)

        # we make the default a read only array
        self.default.setflags(write=False)

        # check that the default array values are in the declared domain
        # this may be expensive
        if self.domain is not None and self.default is not None:
            for e in self.default.flat:
                if e not in self.domain:
                    msg = 'default contains values out of the declared domain. Ex {}'.format(e)
                    log.warning('{} \n   attribute  {}'.format(msg, self))

                    break

    def _lookup_expected_shape(self, instance):
        """ look up expected shape on the instance """
        expected_shape = []

        for dim_attr in self.shape:
            if dim_attr is Dim.any:
                expected_dim = Dim.any
            else:
                try:
                    # invoke Dim's __get__(instance)
                    expected_dim = getattr(instance, dim_attr.field_name)
                except TraitAttributeError:
                    # re-raise with a better error message
                    msg = "Narray's shape references undefined dimension <{}>. " \
                          "Set it before accessing this array"
                    raise TraitAttributeError(msg.format(dim_attr.field_name), attr=self)
            expected_shape.append(expected_dim)
        return expected_shape

    def _validate_set(self, instance, value):
        value = super(NArray, self)._validate_set(instance, value)
        if value is None:
            # value is optional and missing, nothing to do here
            return
        self.__validate(value)
        # we should know here the concrete shape
        # check it

        if self.shape is not None:
            expected_shape = self._lookup_expected_shape(instance)

            for expected_dim, value_dim in zip(expected_shape, value.shape):
                if expected_dim is Dim.any:
                    continue
                if value_dim != expected_dim:
                    raise TraitValueError(
                        'Shape mismatch. Expected {}. Given {}. Not broadcasting'.format(
                            expected_shape, value.shape
                        )
                    )

        return value.astype(self.dtype)

    # here only for typing purposes, so ide's can get better suggestions
    def __get__(self, instance, owner):
        # type: (typing.Optional['HasTraits'], 'MetaType') -> typing.Union[numpy.ndarray, 'NArray']
        return super(NArray, self).__get__(instance, owner)

    def __set__(self, instance, value):
        # type: (HasTraits, numpy.ndarray) -> None
        super(NArray, self).__set__(instance, value)

    def __str__(self):
        return '{}(label={!r}, dtype={}, default={!r}, dim_names={}, ndim={}, required={})'.format(
            self._defined_on_str_helper(),
            self.label,
            self.dtype,
            self.default,
            self.dim_names,
            self.ndim,
            self.required,
        )


class EnumAttr(Attr):
    def __init__(self, field_type=None, default=None, doc='', label='', required=True):
        """
        :param default: The default enum value
        """

        if field_type is None and default is not None:
            field_type = type(default)

        if default is None and field_type is None:
            raise ValueError('Either a default or a field_type is required')

        super(EnumAttr, self).__init__(
            field_type=field_type, default=default, doc=doc, label=label, required=required, choices=tuple(field_type)
        )

    def __validate(self, value):
        """
        value either has to be in the enum choices or its tpye has to be in the value list of enum choices
        """
        if value in self.choices:
            return

        if type(value) in [choice.value for choice in self.choices]:
            return

        raise TraitTypeError("Attribute can't be set to an instance of {}".format(type(value)), attr=self)

    def _validate_set(self, instance, value):
        # type: ('HasTraits', typing.Any) -> typing.Any
        """
        Same method as in Attr.
        The reason we override it is to not call the __validate method from the superclass.
        """
        if value is None:
            if self.required:
                raise TraitValueError("Attribute is required. Can't set to None", attr=self)
            else:
                return value

        self.__validate(value)
        return value

    def _post_bind_validate(self):
        if self.default is not None:
            self.__validate(self.default)

    # def __get__(self, instance, owner):
    #     # type: (typing.Optional['TupleEnum'], 'MetaType') -> typing.Union[Enum, 'Enum']
    #     return super(EnumAttr).__get__(instance, owner)
    #
    # def __set__(self, instance, value):
    #     # type: (TupleEnum, Enum) -> None
    #     super(EnumAttr).__set__(instance, value)


class Range(object):
    """
    Defines a domain like the one that numpy.arange generates
    Points are precisely equidistant but the largest point is <= hi
    """

    def __init__(self, lo, hi, step=1.0):
        self.lo = lo
        self.hi = hi
        self.step = step if abs(hi - lo) > abs(step) else abs(hi - lo)

    def __contains__(self, item):
        """ true if item between lo and high. ignores the step"""
        return self.lo <= item < self.hi

    def to_array(self):
        return numpy.arange(self.lo, self.hi, self.step)

    def __repr__(self):
        return 'Range(lo={}, hi={}, step={})'.format(self.lo, self.hi, self.step)


class LinspaceRange(object):
    """
    Defines a domain with precise endpoints but the points are not precisely equidistant
    Similar to numpy.linspace
    """

    def __init__(self, lo, hi, npoints=50):
        self.lo = lo
        self.hi = hi
        self.npoints = npoints

    def __contains__(self, item):
        """ true if item between lo and high. ignores the step"""
        return self.lo <= item < self.hi

    def to_array(self):
        return numpy.linspace(self.lo, self.hi, self.npoints)

    def __repr__(self):
        return 'LinspaceRange(lo={}, hi={}, step={})'.format(self.lo, self.hi, self.npoints)
