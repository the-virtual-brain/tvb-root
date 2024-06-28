import numpy as np
import scipy.sparse
import enum

from tvb_kernels import tvb_kernels


class CxMode(enum.Enum):
    CX_J = 1
    CX_I = 2
    CX_NOP = 3


def base_setup(mode: CxMode = CxMode.CX_J):
    np.random.seed(43)
    num_node = 90
    sparsity = 0.4
    dt = 0.1
    horizon = 256
    horizonm1 = horizon - 1
    assert ((horizon) & (horizonm1)) == 0
    weights, lengths = np.random.rand(2, num_node, num_node).astype('f')
    lengths[:] *= 0.8
    lengths *= (horizon*dt*0.8)
    zero_mask = weights < (1-sparsity)
    weights[zero_mask] = 0
    # so far so good

    if mode == CxMode.CX_J:
        # cx_j requires csc format, cx_i requires csr
        spw = scipy.sparse.csc_matrix(weights)
        # since numpy is row major, we need to transpose zero mask then ravel
        nnz_ravel = np.argwhere(~zero_mask.T.copy().ravel())[:, 0]
        # then we also need to extract them in the transposed order
        idelays = (lengths.T.copy().ravel()[nnz_ravel] / dt).astype('i') + 2
    elif mode == CxMode.CX_I:
        spw = scipy.sparse.csr_matrix(weights)
        nnz_ravel = np.argwhere(~zero_mask.ravel())[:, 0]
        idelays = (lengths.ravel()[nnz_ravel] / dt).astype('i') + 2
    else:
        raise NotImplementedError

    # then we can test
    buf = np.r_[:1.0:1j*num_node *
                horizon].reshape(1, num_node, horizon).astype('f')*4.0
    buf = np.random.randn(1, num_node, horizon).astype('f')
    c_buf = buf.copy()
    c_cx1, c_cx2 = c_cx = np.zeros((2, num_node), 'f')

    # create teh C struct w/ our data
    conn = tvb_kernels.Conn(
        weights=spw.data,
        indices=spw.indices,
        indptr=spw.indptr,
        idelays=idelays,
        buf=c_buf,
        cx=c_cx
    )

    # impl simple numpy version
    def make_cfun_np():
        csr_weights = scipy.sparse.csr_matrix(weights)
        idelays = (lengths[~zero_mask]/dt).astype('i')+2
        idelays2 = -horizon + np.c_[idelays, idelays-1].T
        assert idelays2.shape == (2, csr_weights.nnz)
        buffer = buf[0]

        def cfun_np(t):
            cx = buffer[csr_weights.indices, (t-idelays2) % horizon]
            cx *= csr_weights.data
            cx = np.add.reduceat(cx, csr_weights.indptr[:-1], axis=1)
            return cx  # (2, num_node)
        return cfun_np

    return conn, c_cx, locals(), make_cfun_np()


def test_conn_kernels():
    connj, cxj, dataj, cfun_np = base_setup(CxMode.CX_J)
    conni, cxi, datai, _ = base_setup(CxMode.CX_I)

    for t in range(1024):
        cx = cfun_np(t)
        tvb_kernels.cx_j(connj, t)
        tvb_kernels.cx_i(conni, t)
        np.testing.assert_allclose(cx, cxj, 1e-4, 1e-6)
        np.testing.assert_allclose(cx, cxi, 1e-4, 1e-6)

