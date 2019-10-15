import typing
import collections
import numpy
import logging
from .neotraits_impl import Attr


# a logger for the whole traits system
log = logging.getLogger('tvb.traits')


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
        super(Const, self).__init__(field_type=type(default), default=default,
                                    doc=doc, label=label, required=True, readonly=True)



class List(Attr):
    """
    The attribute is a list of values.
    Choices and type are reinterpreted as applying not to the list but to the elements of it
    """
    def __init__(self, of=object, default=(), doc='', label='',
                 readonly=False, choices=None):
        # type: (type, tuple, str, str, bool, typing.Optional[tuple]) -> None
        super(List, self).__init__(field_type=collections.Sequence, default=default,
                                   doc=doc, label=label,
                                   required=True, readonly=readonly, choices=None)
        self.element_type = of
        self.element_choices = choices


    def _post_bind_validate(self):
        super(List, self)._post_bind_validate()
        # check that the default contains elements of the declared type
        for i, el in enumerate(self.default):
            if not isinstance(el, self.element_type):
                msg = 'default[{}] must have type {} not {}'.format(
                    i, self.element_type, type(el))
                raise TypeError(self._err_msg(msg))

        if self.element_choices is not None:
            # check that the default respects the declared choices
            for i, el in enumerate(self.default):
                if el not in self.element_choices:
                    msg = 'default[{}]=={} must be one of the choices {}'.format(
                        i, self.default, self.element_choices)
                    raise TypeError(self._err_msg(msg))


    def _validate_set(self, instance, value):
        super(List, self)._validate_set(instance, value)
        for i, el in enumerate(value):
            if not isinstance(el, self.element_type):
                raise TypeError(self._err_msg("value[{}] can't be of type {}".format(i, type(el))))

        if self.element_choices is not None:
            for i, el in enumerate(value):
                if el not in self.element_choices:
                    raise ValueError(self._err_msg("value[{}]=={} must be one of {}".format(i, el, self.element_choices)))


    # __get__ __set__ here only for typing purposes, for better ide checking and autocomplete


    def __get__(self, instance, owner):
        # type: (typing.Any, type) -> typing.Sequence
        return super(List, self).__get__(instance, owner)


    def __set__(self, instance, value):
        # type: (object, typing.Sequence) -> None
        super(List, self).__set__(instance, value)

    def __str__(self):
        return '{}(of={}, default={!r}, required={})'.format(
            type(self).__name__, self.element_type, self.default, self.required)



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
    def __init__(self, default=None, required=True, doc='', label='',
                 dtype=numpy.float, ndim=None, dim_names=(), domain=None):
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

        super(NArray, self).__init__(field_type=numpy.ndarray, default=default,
                                     required=required, doc=doc, label=label)
        self.ndim = int(ndim) if ndim is not None else None
        self.domain = domain  # anything that supports 3.1 in domain
        self.dim_names = tuple(dim_names)

        if dim_names:
            # dimensions are named, infer ndim
            if ndim is not None:
                if ndim != len(dim_names):
                    raise ValueError('dim_names contradicts ndim')
                log.warn('if you declare dim_names ndim is not necessary')
            self.ndim = len(dim_names)


    def _post_bind_validate(self):
        if self.default is None:
            return
        if not isinstance(self.default, numpy.ndarray):
            msg = 'default {} should be a numpy.ndarray'.format(self.default)
            raise TypeError(self._err_msg(msg))
        if not numpy.can_cast(self.default.dtype, self.dtype, 'safe'):
            msg = 'the dtype of the default={} is not compatible with the declared one={}'.format(self.default.dtype, self.dtype)
            raise ValueError(self._err_msg(msg))
        # if ndim is None we allow any ndim
        if self.ndim is not None and self.default.ndim != self.ndim:
            msg = 'default ndim={} is not the declared one={}'.format(self.default.ndim, self.ndim)
            raise ValueError(self._err_msg(msg))

        # we make the default a read only array
        self.default.setflags(write=False)

        # check that the default array values are in the declared domain
        # this may be expensive
        if self.domain is not None and self.default is not None:
            for e in self.default.flat:
                if e not in self.domain:
                    msg = 'default contains values out of the declared domain. Ex {}'.format(e)
                    log.warning(self._err_msg(msg))
                    break


    def _validate_set(self, instance, value):
        super(NArray, self)._validate_set(instance, value)

        if self.ndim is not None and value.ndim != self.ndim:
            raise TypeError(self._err_msg("can't be set to an array with ndim {}".format(value.ndim)))

        if not numpy.can_cast(value.dtype, self.dtype, 'safe'):
            raise TypeError(self._err_msg("can't be set to an array of dtype {}".format(value.dtype)))

    # here only for typing purposes, so ide's can get better suggestions
    def __get__(self, instance, owner):
        # type: (typing.Optional[object], type) -> typing.Union[numpy.ndarray, 'NArray']
        return super(NArray, self).__get__(instance, owner)

    def __set__(self, instance, value):
        # type: (object, numpy.ndarray) -> None
        super(NArray, self).__set__(instance, value)

    def __str__(self):
        return '{}(label={!r}, dtype={}, default={!r}, dim_names={}, ndim={}, required={})'.format(
            type(self).__name__, self.label, self.dtype, self.default, self.dim_names, self.ndim, self.required)


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

