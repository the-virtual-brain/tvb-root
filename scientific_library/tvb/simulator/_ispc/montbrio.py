import numpy as np
import subprocess
import logging
import ctypes
import tqdm
from numpy.random import SFC64


def cmd(str):# {{{
    subprocess.check_call(str.split(' '))# }}}


def make_kernel():
    import os.path
    here = os.path.dirname(os.path.abspath(__file__))
    cmd(f'/usr/local/bin/ispc --target=avx512skx-i32x16 --math-lib=fast --pic {here}/montbrio.c -o {here}/_montbrio.o')
    cmd(f'g++ -shared {here}/_montbrio.o -o {here}/_montbrio.so')
    dll = ctypes.CDLL(f'{here}/_montbrio.so')
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


def run_ispc_montbrio(w, d, total_time):
    assert w.shape[0] == 96
    nn, nl, nc = 96, 16, 6
    k = 0.1
    aff, r, V, nr, nV = np.zeros((5, nc, nl), 'f')
    W = np.zeros((2, 16, nc, nl), 'f')
    V -= 2.0
    rh, Vh = np.zeros((2, nn, nl), 'f')
    wij = w.copy().astype('f')
    Vh -= 2.0
    # assume dt=1
    ih = d.copy().astype(np.uint32)  # np.zeros((nn, nl), np.uint32)
    tavg = np.zeros((2*nn,), 'f')
    rng = np.random.default_rng(SFC64(42))                      # create RNG w/ known seed
    kerneler = make_kernel()
    (_, aff, *_, W, r, V, nr, nV, tavg), kernel = kerneler(k, aff, rh, Vh, wij, ih, W, r, V, nr, nV, tavg)
    tavgs = []
    for i in tqdm.trange(int(total_time/1.0/16)):
      rng.standard_normal(size=W.shape, dtype='f', out=W) # ~63% of time
      kernel()
      tavgs.append(tavg.flat[:].copy())
    tavgs = np.array(tavgs)
    return tavgs, None


if __name__ == '__main__':
    from ctypesgen.main import main
    main()