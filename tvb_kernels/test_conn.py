import numpy as np
import scipy.sparse
import enum

import tvb_kernels


def rand_weights(seed=43, sparsity=0.4, dt=0.1, horizon=256, num_node=90):
    np.random.seed(seed)
    horizonm1 = horizon - 1
    assert ((horizon) & (horizonm1)) == 0
    weights, lengths = np.random.rand(2, num_node, num_node).astype('f')
    lengths[:] *= 0.8
    lengths *= (horizon*dt*0.8)
    zero_mask = weights < (1-sparsity)
    weights[zero_mask] = 0
    return weights, lengths


def base_setup(mode: tvb_kernels.CxMode = tvb_kernels.CxMode.CX_J):
    cv = 1.0
    dt = 0.1
    num_node = 90
    horizon = 256
    weights, lengths = rand_weights(num_node=num_node, horizon=horizon, dt=dt)
    cx = tvb_kernels.Cx(num_node, horizon)
    conn = tvb_kernels.Conn(cx, dt, cv, weights, lengths, mode=mode)

    # then we can test
    cx.buf[:] = np.r_[:1.0:1j*num_node *
                      horizon].reshape(num_node, horizon).astype('f')*4.0

    # impl simple numpy version
    def make_cfun_np():
        zero_mask = weights == 0
        csr_weights = scipy.sparse.csr_matrix(weights)
        idelays = (lengths[~zero_mask]/cv/dt).astype('i')+2
        idelays2 = -horizon + np.c_[idelays, idelays-1].T
        assert idelays2.shape == (2, csr_weights.nnz)

        def cfun_np(t):
            _ = cx.buf[csr_weights.indices, (t-idelays2) % horizon]
            _ *= csr_weights.data
            _ = np.add.reduceat(_, csr_weights.indptr[:-1], axis=1)
            return _  # (2, num_node)
        return cfun_np

    return conn, cx, make_cfun_np()


def test_conn_kernels():
    connj, cxj, cfun_np = base_setup(tvb_kernels.CxMode.CX_J)
    conni, cxi, _ = base_setup(tvb_kernels.CxMode.CX_I)

    for t in range(1024):
        cx = cfun_np(t)
        connj(t)
        conni(t)
        np.testing.assert_allclose(cx[0], cxj.cx1, 1e-4, 1e-6)
        np.testing.assert_allclose(cx[1], cxj.cx2, 1e-4, 1e-6)
        np.testing.assert_allclose(cx[0], cxi.cx1, 1e-4, 1e-6)
        np.testing.assert_allclose(cx[1], cxi.cx2, 1e-4, 1e-6)
