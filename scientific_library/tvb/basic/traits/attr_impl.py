import typing
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
        super(List, self).__init__(field_type=typing.Sequence, default=default,
                                   doc=doc, label=label,
                                   required=True, readonly=readonly, choices=None)
        self.element_type = of
        self.element_choices = choices


    def _post_bind_validate(self, defined_in_type_name):
        super(List, self)._post_bind_validate(defined_in_type_name)
        # check that the default contains elements of the declared type
        for i, el in enumerate(self.default):
            if not isinstance(el, self.element_type):
                msg = 'default[{}] must have type {} not {}'.format(
                    i, self.element_type, type(self.default))
                raise TypeError(self._err_msg_where(defined_in_type_name) + msg)

        if self.element_choices is not None:
            # check that the default respects the declared choices
            for i, el in enumerate(self.default):
                if el not in self.element_choices:
                    msg = 'default[{}]=={} must be one of the choices {}'.format(
                        i, self.default, self.element_choices)
                    raise TypeError(self._err_msg_where(defined_in_type_name) + msg)


    def _validate_set(self, instance, value):
        super(List, self)._validate_set(instance, value)
        for i, el in enumerate(value):
            if not isinstance(el, self.element_type):
                msg_where = self._err_msg_where(type(instance).__name__)
                raise TypeError(msg_where + "value[{}] can't be of type {}".format(i, type(el)))

        if self.element_choices is not None:
            for i, el in enumerate(value):
                if el not in self.element_choices:
                    msg_where = self._err_msg_where(type(instance).__name__)
                    raise ValueError(msg_where + "value[{}]=={} must be one of {}".format(i, el, self.choices))


    # here only for typing purposes, for better ide checking and autocomplete
    def __get__(self, instance, owner):
        # type: (typing.Any, type) -> typing.Sequence
        return super(List, self).__get__(instance, owner)


    def __set__(self, instance, value):
        # type: (object, typing.Sequence) -> None
        super(List, self).__set__(instance, value)



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
        # type: (numpy.ndarray, bool, str, str, typing.Union[numpy.dtype, type], int, typing.Tuple[str, ...], typing.Container) -> None
        super(NArray, self).__init__(field_type=numpy.ndarray, default=default,
                                     required=required, doc=doc, label=label)
        self.dtype = numpy.dtype(dtype)
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


    def _post_bind_validate(self, defined_in_type_name):
        if self.default is None:
            return
        if not isinstance(self.default, numpy.ndarray):
            msg = 'default {} should be a numpy.ndarray'.format(self.default)
            raise TypeError(self._err_msg_where(defined_in_type_name) + msg)
        # we check strict dtype conformance. Compatible dtypes are not ok
        if self.default.dtype != self.dtype:
            msg = 'default dtype={} is not the declared one={}'.format(self.default.dtype, self.dtype)
            raise ValueError(self._err_msg_where(defined_in_type_name) + msg)
        # if ndim is None we allow any ndim
        if self.ndim is not None and self.default.ndim != self.ndim:
            msg = 'default ndim={} is not the declared one={}'.format(self.default.ndim, self.ndim)
            raise ValueError(self._err_msg_where(defined_in_type_name) + msg)

        # we make the default a read only array
        self.default.setflags(write=False)

        # check that the default array values are in the declared domain
        # this may be expensive
        if self.domain is not None and self.default is not None:
            for e in self.default.flat:
                if e not in self.domain:
                    msg = 'default contains values out of the declared domain. Ex {}'.format(e)
                    log.warning(self._err_msg_where(defined_in_type_name) + msg)
                    break


    def _validate_set(self, instance, value):
        super(NArray, self)._validate_set(instance, value)

        def _msg():
            return 'attribute {}.{} = {}(dtype={}, ndim={}) : '.format(
                type(instance).__name__, self.field_name, type(self).__name__,
                self.dtype, self.ndim)

        if self.ndim is not None and value.ndim != self.ndim:
            raise TypeError(_msg() + "can't be set to an array with ndim {}".format(value.ndim))

        # todo review this special case: string dtypes
        # tvb treats numpy string arrays like python lists
        # this goes bad with their fixed size type and the strict dtype checks that we do here
        if self.dtype.kind == 'S' == value.dtype.kind:
            return
        # endtodo

        if value.dtype != self.dtype:
            raise TypeError(_msg() + "can't be set to an array of dtype {}".format(value.dtype))

    # here only for typing purposes, so ide's can get better suggestions
    def __get__(self, instance, owner):
        # type: (typing.Optional[object], type) -> typing.Union[numpy.ndarray, 'NArray']
        return super(NArray, self).__get__(instance, owner)

    def __set__(self, instance, value):
        # type: (object, numpy.ndarray) -> None
        super(NArray, self).__set__(instance, value)


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

