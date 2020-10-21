import numpy as np
import subprocess
import logging
import ctypes
from numpy.random import SFC64

def cmd(str):# {{{
    subprocess.check_call(str.split(' '))# }}}

def make_kernel():
    cmd('/usr/local/bin/ispc --target=avx512skx-i32x16 --math-lib=fast --pic montbrio.c -o montbrio.o')
    cmd('g++ -shared montbrio.o -o montbrio.so')
    dll = ctypes.CDLL('./montbrio.so')
    fn = getattr(dll, 'loop')
    fn.restype = ctypes.c_void_p
    i32a = ctypes.POINTER(ctypes.c_int)
    f32 = ctypes.c_float
    f32a = ctypes.POINTER(f32)
    fn.argtypes = [f32, f32a, f32a, f32a, f32a, i32a, f32a, f32a, f32a, f32a, f32a, f32a]
    def _(*args):
        args_ = []
        args_ct = []
        for ct, arg_ in zip(fn.argtypes, args):
            if hasattr(arg_, 'shape'):
                arg_ = arg_.astype(dtype='f', order='C', copy=True)
                args_.append(arg_)
                args_ct.append(arg_.ctypes.data_as(ct))
            else:
                args_ct.append(ct(arg_))
                args_.append(arg_)
        def __():
            fn(*args_ct)
        return args_, __
    return _

nn, nl, nc = 96, 16, 6
k = 0.1
aff, r, V, nr, nV = np.zeros((5, nc, nl), 'f')
W  = np.zeros((2, nl, nc, nl), 'f')
V -= 2.0
rh, Vh, wij = np.zeros((3, nn, nl), 'f')
Vh -= 2.0
ih = np.zeros((nn, nl), np.int32)
tavg = np.zeros((2*nn,), 'f')
rng = np.random.default_rng(SFC64(42))                      # create RNG w/ known seed
kerneler = make_kernel()
(_, aff, *_, W, r, V, nr, nV, tavg), kernel = kerneler(k, aff, rh, Vh, wij, ih, W, r, V, nr, nV, tavg)
tavgs = []
import tqdm
for i in tqdm.trange(int(30e3/0.1/16)):
  rng.standard_normal(size=W.shape, dtype='f', out=W) # ~50%
  kernel()
  tavgs.append(r.flat[:].copy())
tavgs = np.array(tavgs)

import pylab as pl
pl.subplot(211); pl.plot(tavgs[:, 0], 'k')
# pl.subplot(212); pl.plot(tavgs[:, 96], 'b')
pl.show()