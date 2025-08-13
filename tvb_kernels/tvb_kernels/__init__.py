import enum
import numpy as np
import ctypes

# TODO mv functionality used to C
import scipy.sparse

from . import _ctg_tvbk


def _to_ct(a):
    val_type = {
        'float32': ctypes.c_float,
        'uint32': ctypes.c_uint32,
        'int32': ctypes.c_int32
    }[a.dtype.name]
    return a.ctypes.data_as(ctypes.POINTER(val_type))


class CxMode(enum.Enum):
    CX_J = 1
    CX_I = 2
    CX_NOP = 3


class Cx:
    def __init__(self, num_node, num_time):
        if num_time & (num_time - 1) != 0:
            raise ValueError('num_time not power of two')
        self.cx1, self.cx2 = np.zeros((2, num_node), 'f')
        self.buf = np.zeros((num_node, num_time), 'f')
        self._cx = _ctg_tvbk.tvbk_cx(
            cx1=_to_ct(self.cx1),
            cx2=_to_ct(self.cx2),
            buf=_to_ct(self.buf),
            num_node=num_node,
            num_time=num_time 
        )


class Conn:

    def __init__(self, cx, dt, cv, weights, lengths, mode: CxMode = CxMode.CX_J):
        self.mode = mode
        assert weights.shape[0] == weights.shape[1]
        assert weights.shape == lengths.shape
        self.num_node = weights.shape[0]
        zero_mask = weights == 0
        if mode == CxMode.CX_J:
            # cx_j requires csc format, cx_i requires csr
            spw = scipy.sparse.csc_matrix(weights)
            # since numpy is row major, we need to transpose zero mask then ravel
            nnz_ravel = np.argwhere(~zero_mask.T.copy().ravel())[:, 0]
            # then we also need to extract them in the transposed order
            idelays = (lengths.T.copy().ravel()[
                       nnz_ravel] / cv / dt).astype('i') + 2
        elif mode == CxMode.CX_I:
            spw = scipy.sparse.csr_matrix(weights)
            nnz_ravel = np.argwhere(~zero_mask.ravel())[:, 0]
            idelays = (lengths.ravel()[nnz_ravel] / cv / dt).astype('i') + 2
        else:
            raise NotImplementedError

        # retain references to arrays so they aren't collected
        self.idelays = idelays.astype(np.uint32)
        self.weights = spw.data.astype(np.float32)
        self.indices = spw.indices.astype(np.uint32)
        self.indptr = spw.indptr.astype(np.uint32)
        self.cx = cx
        self._conn = _ctg_tvbk.tvbk_conn(
            num_node=weights.shape[0],
            num_nonzero=self.indices.size,
            num_cvar=1,
            weights=_to_ct(self.weights),
            indices=_to_ct(self.indices),
            indptr=_to_ct(self.indptr),
            idelays=_to_ct(self.idelays),
        )

    def __call__(self, t):
        if self.mode == CxMode.CX_J:
            _ctg_tvbk.tvbk_cx_j(self.cx._cx, self._conn, t)
        if self.mode == CxMode.CX_I:
            _ctg_tvbk.tvbk_cx_i(self.cx._cx, self._conn, t)
