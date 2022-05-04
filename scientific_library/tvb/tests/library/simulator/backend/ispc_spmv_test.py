import subprocess, ctypes as ct
from scipy import io, sparse
import numpy as np
mat = io.loadmat('conn515k.mat')
data, ir, jc = [mat['Mat'][key][0,0][0] for key in 'data ir jc'.split(' ')]
weights = sparse.csc_matrix((data, ir, jc), shape=(515056, 515056))

b = np.random.randn(weights.shape[0])
out = b.copy()

wf = weights.data.astype('f')
bf = b.astype('f')
print(bf.size)
outf = out.astype('f')

subprocess.check_call('/home/duke/Downloads/ispc-v1.15.0-linux/bin/ispc spmv.ispc --target=avx2-i32x8 --pic -O3 -o spmv.o'.split(' '))
subprocess.check_call('g++ -fPIC -c tasksys.cpp'.split(' '))
subprocess.check_call('g++ -shared tasksys.o spmv.o -o spmv.so -lpthread'.split(' '))

lib2 = ct.CDLL('./spmv.so')
lib2.spmv3.restype = None
fvec = np.ctypeslib.ndpointer(dtype=np.float32)
ivec = np.ctypeslib.ndpointer(dtype=np.int32)
lib2.spmv3.argtypes = fvec, ivec, ivec, ct.c_int, ct.c_int, ct.c_int, fvec, fvec, ct.c_int
outf[:] = 0
args = (
    wf, weights.indices, weights.indptr,
    weights.shape[0], weights.shape[1], weights.nnz,
    bf, outf, 1024)
import time
tic = time.time()
for i in range(100):
    lib2.spmv3(*args)
print((time.time() - tic)/100 * 1000, 'ms/iter')
print(np.abs(outf - (bf*weights)).max())
