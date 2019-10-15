"""
This private module implements concrete declarative attributes
"""
import typing
import types
import collections
import numpy
import logging
from ._declarative_base import _Attr
from .ex import TraitValueError, TraitTypeError, TraitAttributeError

if typing.TYPE_CHECKING:
    from ._core import HasTraits
    from tvb.basic.neotraits._declarative_base import MetaType

# a logger for the whole traits system
log = logging.getLogger('tvb.traits')


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
            self, field_type, default=None, doc='', label='', required=True, readonly=False, choices=None
    ):
        # type: (type, typing.Any, str, str, bool, bool, typing.Optional[tuple]) -> None
        """
        :param field_type: the python type of this attribute
        :param default: A shared default value. Behaves like class level attribute assignment.
                        Take care with mutable defaults.
        :param doc: Documentation for this field.
        :param label: A short description.
        :param required: required fields should not be None. Not strongly enforced.
        :param readonly: If assignment should be prohibited.
        :param choices: A tuple of the values that this field is allowed to take.
        """
        super(Attr, self).__init__()
        self.field_type = field_type
        self.default = default
        self.doc = doc
        self.label = label
        self.required = bool(required)
        self.readonly = bool(readonly)
        self.choices = choices

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

        if not skip_default_checks and not isinstance(self.default, self.field_type):
            msg = 'Attribute should have a default of type {} not {}'.format(
                self.field_type, type(self.default)
            )
            raise TraitTypeError(msg, attr=self)

        if self.choices is not None and not skip_default_checks:
            if self.default not in self.choices:
                msg = 'The default {} must be one of the choices {}'.format(self.default, self.choices)
                raise TraitTypeError(msg, attr=self)

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

        if not isinstance(value, self.field_type):
            raise TraitTypeError("Attribute can't be set to an instance of {}".format(type(value)), attr=self)
        if self.choices is not None:
            if value not in self.choices:
                raise TraitValueError("Value {!r} must be one of {}".format(value, self.choices), attr=self)
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
            if isinstance(self.default, types.FunctionType):
                default = self.default()
            else:
                default = self.default

            # Unless we store the default on the instance, this will keep returning self.default()
            # when the default is a function. So if the default is mutable, any changes to it are
            # lost as a new one is created every time.
            instance.__dict__[self.field_name] = default

        return instance.__dict__[self.field_name]


    def __set__(self, instance, value):
        # type: ('HasTraits', typing.Any) -> None
        self._assert_have_field_name()
        if self.readonly:
            raise TraitAttributeError("can't set readonly attribute")
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



class Const(Attr):
    """
    An attribute that resolves to the given default.
    Note that if it is a mutable type, the value is shared with all instances of the owning class
    We cannot enforce true constancy in python
    """

    def __init__(self, default, doc='', label=''):
        """
        :param default: The constant value
        """
        # it would be nice if we could turn the default immutable. But this is unreasonable work in python
        super(Const, self).__init__(
            field_type=type(default), default=default, doc=doc, label=label, required=True, readonly=True
        )


class List(Attr):
    """
    The attribute is a list of values.
    Choices and type are reinterpreted as applying not to the list but to the elements of it
    """

    def __init__(self, of=object, default=(), doc='', label='', readonly=False, choices=None):
        # type: (type, tuple, str, str, bool, typing.Optional[tuple]) -> None
        super(List, self).__init__(
            field_type=collections.Sequence,
            default=default,
            doc=doc,
            label=label,
            required=True,
            readonly=readonly,
            choices=None,
        )
        self.element_type = of
        self.element_choices = choices


    def _post_bind_validate(self):
        super(List, self)._post_bind_validate()
        # check that the default contains elements of the declared type
        for i, el in enumerate(self.default):
            if not isinstance(el, self.element_type):
                msg = 'default[{}] must have type {} not {}'.format(i, self.element_type, type(el))
                raise TraitTypeError(msg, attr=self)

        if self.element_choices is not None:
            # check that the default respects the declared choices
            for i, el in enumerate(self.default):
                if el not in self.element_choices:
                    msg = 'default[{}]=={} must be one of the choices {}'.format(
                        i, self.default, self.element_choices
                    )
                    raise TraitTypeError(msg, attr=self)


    def _validate_set(self, instance, value):
        value = super(List, self)._validate_set(instance, value)
        if value is None:
            # value is optional and missing, nothing to do here
            return

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
    def _post_bind_validate(self):
        if self.default is not None and not numpy.can_cast(self.default, self.field_type, 'safe'):
            msg = 'can not safely cast default value {} to the declared type {}'.format(
                self.default, self.field_type
            )
            raise TraitTypeError(msg, attr=self)

        if self.choices is not None and self.default is not None:
            if self.default not in self.choices:
                msg = 'the default {} must be one of the choices {}'.format(self.default, self.choices)
                raise TraitTypeError(msg, attr=self)


    def _validate_set(self, instance, value):
        if value is None:
            if self.required:
                raise TraitValueError("is required. Can't set to None", attr=self)
            else:
                return value

        if not isinstance(value, (int, long, float, complex, numpy.number)):
            # we have to check that the value is numeric before the can_cast check
            # as can_cast works with dtype strings as well
            # can_cast('i8', 'i32')
            raise TraitTypeError("can't be set to {!r}. Need a number.".format(value), attr=self)

        if not numpy.can_cast(value, self.field_type, 'safe'):
            raise TraitTypeError("can't be set to {!r}. No safe cast.".format(value), attr=self)
        if self.choices is not None:
            if value not in self.choices:
                raise TraitValueError("value {!r} must be one of {}".format(value, self.choices), attr=self)
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
        self, field_type=int, default=0, doc='', label='', required=True, readonly=False, choices=None
    ):
        super(_Number, self).__init__(
            field_type=field_type,
            default=default,
            doc=doc,
            label=label,
            required=required,
            readonly=readonly,
            choices=choices,
        )

    def _post_bind_validate(self):
        if not issubclass(self.field_type, (int, long, numpy.integer)):
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
        self, field_type=float, default=0, doc='', label='', required=True, readonly=False, choices=None
    ):
        super(_Number, self).__init__(
            field_type=field_type,
            default=default,
            doc=doc,
            label=label,
            required=required,
            readonly=readonly,
            choices=choices,
        )

    def _post_bind_validate(self):
        if not issubclass(self.field_type, (float, numpy.floating)):
            msg = 'field_type must be a python float or a numpy.floating not {!r}.'.format(self.field_type)
            raise TraitTypeError(msg, attr=self)
        # super call after the field_type check above
        super(Float, self)._post_bind_validate()



class NArray(Attr):
    """
    Declares a numpy array.
    If specified ndim enforces the number of dimensions.
    dtype enforces the precise dtype. No implicit conversions. The default dtype is float32.
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
        dtype=numpy.float,
        ndim=None,
        dim_names=(),
        domain=None,
    ):
        # type: (numpy.ndarray, bool, str, str, typing.Union[numpy.dtype, type, str], int, typing.Tuple[str, ...], typing.Container) -> None
        """
        :param dtype: The numpy datatype. Defaults to float64. This is checked by neotraits.
        :param ndim: If given then only arrays of this many dimensions are allowed
        :param dim_names: Optional names for the names of the dimensions
        :param domain: Any type that can be checked for membership like xrange.
                       Represents the expected domain of the values in the array.
        """

        self.dtype = numpy.dtype(dtype)
        # default to zero-dimensional arrays, these behave somewhat curious and similar to numbers
        # this eliminates the is None state. But the empty array is not much better. Shape will be ()
        #
        # todo: review this concept.
        # if default is None:
        #     default = numpy.zeros((), dtype=dtype)

        super(NArray, self).__init__(
            field_type=numpy.ndarray, default=default, required=required, doc=doc, label=label
        )
        self.ndim = int(ndim) if ndim is not None else None
        self.domain = domain  # anything that supports 3.1 in domain
        self.dim_names = tuple(dim_names)

        if dim_names:
            # dimensions are named, infer ndim
            if ndim is not None:
                if ndim != len(dim_names):
                    raise TraitValueError('dim_names contradicts ndim')
                log.warning('If you declare dim_names ndim is not necessary. attr {}'.format(self))
            self.ndim = len(dim_names)


    def _post_bind_validate(self):
        if self.default is None:
            return
        if not isinstance(self.default, numpy.ndarray):
            msg = 'default {} should be a numpy.ndarray'.format(self.default)
            raise TraitTypeError(msg, attr=self)
        if not numpy.can_cast(self.default, self.dtype, 'safe'):
            msg = 'the default={} value can not be safely cast to the declared dtype={}'.format(
                self.default, self.dtype
            )
            raise TraitValueError(msg, attr=self)
        # if ndim is None we allow any ndim
        if self.ndim is not None and self.default.ndim != self.ndim:
            msg = 'default ndim={} is not the declared one={}'.format(self.default.ndim, self.ndim)
            raise TraitValueError(msg, attr=self)

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


    def _validate_set(self, instance, value):
        value = super(NArray, self)._validate_set(instance, value)
        if value is None:
            # value is optional and missing, nothing to do here
            return

        if self.ndim is not None and value.ndim != self.ndim:
            raise TraitTypeError("can't be set to an array with ndim {}".format(value.ndim), attr=self)

        if not numpy.can_cast(value.dtype, self.dtype, 'safe'):
            raise TraitTypeError("can't be set to an array of dtype {}".format(value.dtype), attr=self)

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


class Range(object):
    """
    Defines a domain like the one that numpy.arange generates
    Points are precisely equidistant but the largest point is <= hi
    """

    def __init__(self, lo, hi, step=1.0):
        self.lo = lo
        self.hi = hi
        self.step = step

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
