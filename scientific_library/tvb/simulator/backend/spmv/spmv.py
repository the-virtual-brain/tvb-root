from scipy import io, sparse
import numpy as np
import numba as nb
import time

mat = io.loadmat('conn515k.mat')
data, ir, jc = [mat['Mat'][key][0,0][0] for key in 'data ir jc'.split(' ')]
weights = sparse.csc_matrix((data, ir, jc), shape=(515056, 515056))
b = np.random.randn(weights.shape[0])
out = b.copy()

@nb.njit(parallel=True, boundscheck=False)
def spmatvec(out, b, data, ir, jc):
    chunks = (jc.size - 1) // 512
    for ci in nb.prange(chunks):
        for cj in range(512):
            c = ci*512 + cj
            acc = nb.float32(0.0)
            for r in range(jc[c],jc[c+1]):
                acc += data[r] * b[ir[r]]
            out[c] = acc
    for c in range(chunks*512, jc.size - 1):
        acc = nb.float32(0.0)
        for r in range(jc[c],jc[c+1]):
            acc += data[r] * b[ir[r]]
        out[c] = acc

wf = weights.data.astype('f')
bf = b.astype('f')
outf = out.astype('f')
spmatvec(outf, bf, wf, weights.indices, weights.indptr)

tic = time.time()
for i in range(500):
    spmatvec(outf, bf, wf, weights.indices, weights.indptr)
print((time.time() - tic)/500*1000, 'ms')

