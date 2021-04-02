import numpy as np
import subprocess
import logging
import ctypes
import tqdm
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
    u32a = ctypes.POINTER(ctypes.c_uint32)
    f32 = ctypes.c_float
    f32a = ctypes.POINTER(f32)
    fn.argtypes = [f32, f32a, f32a, f32a, f32a, u32a, f32a, f32a, f32a, f32a, f32a, f32a]
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
W  = np.zeros((2, 16, nc, nl), 'f')
V -= 2.0
rh, Vh = np.zeros((2, nn, nl), 'f')
wij = np.zeros((nn, nn), 'f')
Vh -= 2.0
ih = np.zeros((nn, nl), np.uint32)
tavg = np.zeros((2*nn,), 'f')
rng = np.random.default_rng(SFC64(42))                      # create RNG w/ known seed
kerneler = make_kernel()
(_, aff, *_, W, r, V, nr, nV, tavg), kernel = kerneler(k, aff, rh, Vh, wij, ih, W, r, V, nr, nV, tavg)
tavgs = []
for i in tqdm.trange(int(100*60e3/1.0/16)):
  rng.standard_normal(size=W.shape, dtype='f', out=W) # ~63% of time
  kernel()
  tavgs.append(tavg.flat[:].copy())
tavgs = np.array(tavgs)

import pylab as pl
t = np.r_[:len(tavgs)] * 0.016
pl.subplot(211); pl.plot(t, tavgs[:, 0], 'k-')
pl.ylabel('tavg r'); pl.xlabel('time (s)')
pl.subplot(212); pl.plot(t, tavgs[:, 0], 'k-'); pl.xlim([t[-1000], t[-1]])
pl.ylabel('tavg r'); pl.xlabel('time (s)')
pl.grid(1)
pl.show()