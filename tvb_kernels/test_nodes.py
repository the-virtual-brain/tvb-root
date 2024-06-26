import numpy as np
import scipy.sparse
import nodes
import ctypes as ct

# test connectivity
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

# for the new approach, we need to invert this, so use CSC instead
csc = scipy.sparse.csc_matrix(weights)

# since numpy is row major, we need to transpose zero mask then ravel
nnz_ravel = np.argwhere(~zero_mask.T.copy().ravel())[:, 0]

# then we also need to extract them in the transposed order
idelays = (lengths.T.copy().ravel()[nnz_ravel] / dt).astype('i') + 2

# then we can test
buf = np.r_[:1.0:1j*num_node*horizon].reshape(1, num_node, horizon).astype('f')*4.0
buf = np.random.randn(1, num_node, horizon).astype('f')
c_buf = buf.copy()
c_cx1, c_cx2 = np.zeros((2, num_node), 'f')

to_c = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_float if a.dtype==np.float32 else ct.c_int))

conn = nodes.connectivity(
    num_node=num_node,
    num_nonzero=nnz_ravel.size,
    num_cvar=1,
    horizon=horizon,
    horizon_minus_1=horizonm1,
    weights=to_c(csc.data),
    indices=to_c(csc.indices),
    indptr=to_c(csc.indptr),
    idelays=to_c(idelays),
    buf=to_c(c_buf),
    cx1=to_c(c_cx1),
    cx2=to_c(c_cx2),
)
p_conn = ct.byref(conn)

def setup_cx_all2_csr():
    csr = scipy.sparse.csr_matrix(weights)
    nnz_ravel = np.argwhere(~zero_mask.ravel())[:, 0]
    idelays = (lengths.ravel()[nnz_ravel] / dt).astype('i') + 2
    c_buf = buf.copy()
    c_cx1, c_cx2 = np.zeros((2, num_node), 'f')
    conn = nodes.connectivity(
        num_node=num_node,
        num_nonzero=nnz_ravel.size,
        num_cvar=1,
        horizon=horizon,
        horizon_minus_1=horizonm1,
        weights=to_c(csr.data),
        indices=to_c(csr.indices),
        indptr=to_c(csr.indptr),
        idelays=to_c(idelays),
        buf=to_c(c_buf),
        cx1=to_c(c_cx1),
        cx2=to_c(c_cx2),
    )
    p_conn = ct.byref(conn)
    return p_conn, c_cx1, locals()


def setup_cx_all3_csr():
    csr = scipy.sparse.csr_matrix(weights)
    nnz_ravel = np.argwhere(~zero_mask.ravel())[:, 0]
    idelays = (lengths.ravel()[nnz_ravel] / dt).astype('i') + 2
    c_buf = buf.transpose(0,2,1).copy()
    c_cx1, c_cx2 = np.zeros((2, num_node), 'f')
    conn = nodes.connectivity(
        num_node=num_node,
        num_nonzero=nnz_ravel.size,
        num_cvar=1,
        horizon=horizon,
        horizon_minus_1=horizonm1,
        weights=to_c(csr.data),
        indices=to_c(csr.indices),
        indptr=to_c(csr.indptr),
        idelays=to_c(idelays),
        buf=to_c(c_buf),
        cx1=to_c(c_cx1),
        cx2=to_c(c_cx2),
    )
    p_conn = ct.byref(conn)
    return p_conn, c_cx1, locals()

p_conn2, c_cx12, cxall2work = setup_cx_all2_csr()
p_conn3, c_cx13, cxall3work = setup_cx_all3_csr()

# run them
nodes.cx_all(p_conn, 15)
nodes.cx_all2(p_conn2, 15)
nodes.cx_all3(p_conn3, 15)


# now a numpy version of similar
cx1, cx2 = np.zeros((2, num_node), 'f')
for j in range(num_node):
    j0, j1 = csc.indptr[j:j+2]
    p1 = (15 - idelays[j0:j1] + horizon) % horizon
    cx1[csc.indices[j0:j1]] += csc.data[j0:j1]*buf[0, j, p1]

# ah maybe a dense version
idelays_dense = (lengths / dt).astype('i') + 2
wxij = weights*buf[0,
    np.tile(np.r_[:num_node], (num_node,1)),
    (15 - idelays_dense + horizon) % horizon]
cx13 = wxij.sum(axis=1)

np.testing.assert_allclose(cx1, cx13, 2e-6, 1e-6)  # the two numpy versions agree
np.testing.assert_allclose(cx1, c_cx1, 2e-6, 1e-6)
np.testing.assert_allclose(cx1, c_cx12, 2e-6, 1e-6)
np.testing.assert_allclose(cx1, c_cx13, 2e-6, 1e-6)

import pylab as pl
pl.plot(cx1, c_cx1, '.')
pl.plot(cx1, c_cx12, '.', alpha=0.4)
pl.plot(cx1, c_cx13, '.', alpha=0.4)
pl.savefig('nodes.jpg')

# now benchmarking

def run_sim_np(weights, lengths, zero_mask):
    csr_weights = scipy.sparse.csr_matrix(weights)
    idelays = (lengths[~zero_mask]/dt).astype('i')+2
    
    idelays2 = -horizon + np.c_[idelays, idelays-1].T
    assert idelays2.shape == (2, csr_weights.nnz)
    buffer = np.zeros((num_node, horizon), 'f')

    def cfun(t):
        cx = buffer[csr_weights.indices, (t-idelays2) % horizon]
        cx *= csr_weights.data
        cx = np.add.reduceat(cx, csr_weights.indptr[:-1], axis=1)
        return cx  # (2, num_node)
    return cfun


cfun_np = run_sim_np(weights, lengths, zero_mask)
cx = cfun_np(15)

import time

fs = [
    lambda i: cfun_np(i),
    lambda i: nodes.cx_all(p_conn, i),
    lambda i: nodes.cx_all2(p_conn2, i),
    lambda i: nodes.cx_all3(p_conn3, i),
    lambda i: nodes.cx_all_nop(p_conn3, i),
]
tt = []

for f in fs:
    tik = time.time()
    for i in range(100_000):
        f(i)
    tok = time.time()
    tt.append(tok - tik)

ij_pct = (tt[1]-tt[2])/tt[2]*100
ijT_pct = (tt[1]-tt[3])/tt[3]*100
nop_pct = tt[4]/tt[1]*100
print(f'np {tt[0]:0.3f} cj {tt[1]:0.3f}, ci {tt[2]:0.3f} ciT {tt[3]:0.3f}'
      f' x {tt[0]/tt[1]:0.1f}'
      f' ij% {ij_pct:0.2f} ijT%{ijT_pct:0.2f} overhead {nop_pct:0.2f}% ')
