import numpy as np
import subprocess
import logging
import ctypes
import tqdm
from numpy.random import SFC64


def cmd(str):# {{{
    subprocess.check_call(str.split(' '))# }}}


# shoudl refactor this into a build and load process
# build requires ISPC, cc & ctypesgen
# load is plain Python
# though to customize which variant is loaded, maybe need ctypesgen at runtime
def make_kernel():
    import os.path
    here = os.path.dirname(os.path.abspath(__file__))
    # compile ISPC to object + header file
    cmd(f'/usr/local/bin/ispc --target=avx512skx-i32x16 --math-lib=fast -h {here}/_montbrio.h --pic {here}/_montbrio.c -o {here}/_montbrio.o')
    # link object into shared lib / dll
    cmd(f'g++ -shared {here}/_montbrio.o -o {here}/_montbrio.so')
    # generate ctypes interface
    import ctypesgen.main
    ctypesgen.main.main(f'-o {here}/_montbrio_ctypes.py -L {here} -l {here}/_montbrio.so {here}/_montbrio.h'.split())
    from tvb.simulator._ispc._montbrio_ctypes import loop as fn
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
    from tvb.simulator._ispc._montbrio_ctypes import Data
    data = Data(k=0.1)
    (_, aff, *_, W, r, V, nr, nV, tavg), kernel = kerneler(data, aff, rh, Vh, wij, ih, W, r, V, nr, nV, tavg)
    tavgs = []
    for i in tqdm.trange(int(total_time/1.0/16)):
      rng.standard_normal(size=W.shape, dtype='f', out=W) # ~63% of time
      kernel()
      tavgs.append(tavg.flat[:].copy())
    tavgs = np.array(tavgs)
    return tavgs, None


if __name__ == '__main__':

    nn = 96
    w = np.random.randn(nn, nn)**2
    d = np.random.rand(nn, nn)**2 * 15
    run_ispc_montbrio(w, d, 60e3)