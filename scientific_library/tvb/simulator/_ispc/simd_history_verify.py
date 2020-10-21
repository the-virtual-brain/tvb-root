import numpy as np
import subprocess
import logging
import ctypes
import tqdm
from numpy.testing import assert_allclose

def ref(aff, rh, wij, ih):
    "Reference algorithm for delay updates."
    aff = aff.copy()
    rh = rh.copy()
    for t in range(16):
        for i in range(wij.shape[0]):
            aff[i] = 0
            for j in range(wij.shape[1]):
                aff[i] += wij[j, i] * rh[j, ih[j, i]]
        rh[:,1:] = rh[:, :-1]
        rh[:, 0] = aff
    return rh

def cmd(str):
    "Call and check a command string."
    subprocess.check_call(str.split(' '))

def buf(shape, dtype):
    "Make zero'd NumPy buffer and ctypes pointer for it."
    np_dtype = {'f': np.float32, 'u32': np.uint32}[dtype]
    z = np.zeros(shape, np_dtype).astype(dtype=np_dtype, order='C', copy=True)
    ct = ctypes.POINTER({'f':ctypes.c_float, 'u32': ctypes.c_uint32}[dtype])
    return z, z.ctypes.data_as(ct)

nn = 128
for target in 'sse4-i8x16 avx1-i32x16 avx2-i16x16 avx2-i32x16 avx2-i8x32 avx512skx-i32x16 avx512skx-i16x32 avx512skx-i8x64'.split():
    nl = int(target.split('x')[-1])
    nc = nn / nl
    print('==== target: ', target, '====')
    defs = f'-Dnn={nn} -Dnc={nc} -Dnl={nl}'
    cmd(f'/usr/local/bin/ispc -g --target={target} --math-lib=fast {defs} --pic simd_history_verify.c -o simd_history_verify.o')
    # cmd(f'clang -g {defs} -c simd_history_verify.c')
    cmd(f'clang -g -shared simd_history_verify.o -o verify.{target}.so')
    lib = ctypes.CDLL(f'./verify.{target}.so')
    for i in (0,1): #,10, 11):
        fn = getattr(lib, f'loop{i}')
        aff, c_aff = buf((nn,), 'f')
        rh, c_rh = buf((nn, nl), 'f')
        rh[:] = np.random.randn(*rh.shape).astype('f')
        wij, c_wij = buf((nn, nn), 'f')
        wij[:] = np.random.randn(*wij.shape).astype('f')
        ih, c_ih = buf((nn, nn), 'u32')
        ih[:] = np.random.randint(0, nl - 1, ih.shape, np.uint32)
        ref_rh = ref(aff, rh, wij, ih)
        fn(c_aff, c_rh, c_wij, c_ih)
        # assert_allclose(ref_rh, rh, 1e-4, 0.02)
        for j in tqdm.trange(1000):
            fn(c_aff, c_rh, c_wij, c_ih)