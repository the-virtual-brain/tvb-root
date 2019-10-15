import typing
import numpy
import logging
from .neotraits import Attr


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
                                    required=True, doc=doc, label=label, readonly=True)


class NArray(Attr):
    """
    Declares a numpy array.
    If specified ndim enforces the number of dimensions.
    dtype enforces the precise dtype. Create one like this np.dtype(np.float32)
    The default dtype is float32.
    Implicit conversions are not supported
    domain declares what values are allowed in this array. It can be any object that can be checked for membership
    """
    def __init__(self, default=None, required=True, doc='', label='',
                 dtype=numpy.dtype(numpy.float), ndim=None, domain=None):
        # type: (numpy.ndarray, bool, str, str, typing.Union[numpy.dtype, type], int, typing.Container[float]) -> None
        super(NArray, self).__init__(field_type=numpy.ndarray, default=default,
                                     required=required, doc=doc, label=label)
        self.dtype = dtype
        self.ndim = ndim
        self.domain = domain  # anything that supports 3.1 in domain


    def _post_bind_validate(self, defined_in_type_name):
        if self.default is None:
            return
        # we check strict dtype conformance. Compatible dtypes are not ok
        if self.default.dtype != self.dtype:
            msg = 'default dtype={} is not the declared one={}'.format(self.default.dtype, self.dtype)
            raise ValueError(self._err_msg_where(defined_in_type_name) + msg)
        # if ndim is None we allow any ndim
        if self.ndim is not None and self.default.ndim != self.ndim:
            msg = 'default ndim={} is not the declared one={}'.format(self.default.ndim, self.ndim)
            raise ValueError(self._err_msg_where(defined_in_type_name) + msg)

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

        if value.dtype != self.dtype:
            raise TypeError(_msg() + "can't be set to an array of dtype {}".format(value.dtype))
        if self.ndim is not None and value.ndim != self.ndim:
            raise TypeError(_msg() + "can't be set to an array with ndim {}".format(value.ndim))
