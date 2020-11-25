import numpy as np
import subprocess
import logging
import ctypes
import tqdm
from numpy.random import SFC64
import os



def needs_built(output, *inputs):
    if not os.path.exists(output):
        return True
    for input in inputs:
        if os.stat(input).st_mtime > os.stat(output).st_mtime:
            return True
    return False


def make(str, output, *inputs):
    if needs_built(output, *inputs):
        subprocess.check_call(str.split(' '))

# assume 16 wide only for now
targets = {
    'sse4': 'sse4-i8x16',
    'avx1': 'avx1-i32x16',
    'avx2': 'avx2-i16x16',
    'avx512': 'avx512skx-i32x16',
}

def find_ispc():
    import sys, subprocess
    which_cmd = 'where' if sys.platform == 'win32' else 'which'
    b_path = subprocess.check_output([which_cmd, 'ispc'])
    return b_path.decode('ascii').strip()


def buildmod(isa='avx2'):  # prolly refactor to generic ISPC builder
    import os.path
    here = os.path.dirname(os.path.abspath(__file__))
    ispc = find_ispc()
    cxx = 'g++'
    target = targets[isa]
    model = 'montbrio'
    name = f'{model}_{target}'
    source = f"{here}/_{name.split('_')[0]}.c"
    object = f'{here}/_{name}.o'
    shared_lib = f'{here}/_{name}.so'
    header = f'{here}/_{name}.h'
    ctypes_mod = f'{here}/_{name.replace("-","_")}_ct.py'
    make(f'{ispc} --target={target} --math-lib=fast --pic -h {header} {source} -o {object}', object, source)
    make(f'{cxx} -shared {object} -o {shared_lib}', shared_lib, object)
    import ctypesgen.main
    if needs_built(ctypes_mod, header):
        ctypesgen.main.main(f'-o {ctypes_mod} -L {here} -l {shared_lib} {header}'.split())
    import importlib
    mod_name = os.path.basename(ctypes_mod).split('.py')[0]
    return importlib.import_module(f'tvb.simulator._ispc.{mod_name}')


# shoudl refactor this into a build and load process
# build requires ISPC, cc & ctypesgen
# load is plain Python
# though to customize which variant is loaded, maybe need ctypesgen at runtime
def make_kernel(isa='avx512'):
    mod = buildmod(isa)
    fn = mod.loop
    # from tvb.simulator._ispc._montbrio_avx512skx_i32x16_ct import loop as fn
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
    return _, mod


def run_ispc_montbrio(
        weights, delays,
        total_time=60e3, bold_tr=1800, coupling_scaling=0.01,
        r_sigma=1e-3, V_sigma=1e-3,
        I=1.0, Delta=1.0, eta=-5.0, tau=100.0, J=15.0, cr=0.01, cv=0.0,
        dt=1.0,
        progress=False,
        isa='avx512',
        ):
    w, d = weights, delays
    assert w.shape[0] == 96
    nn, nl, nc = 96, 16, 6
    aff, r, V, nr, nV = np.zeros((5, nc, nl), 'f')
    W = np.zeros((2, 16, nc, nl), 'f')
    V -= 2.0
    rh, Vh = np.zeros((2, nn, nl), 'f')
    wij = w.copy().astype('f')
    Vh -= 2.0
    # assume dt=1
    ih = (d/dt).astype(np.uint32, order='C', copy=True)
    tavg = np.zeros((2*nn,), 'f')
    rng = np.random.default_rng(SFC64(42))                      # create RNG w/ known seed
    kerneler, mod = make_kernel(isa=isa)
    data = mod.Data(k=coupling_scaling, I=I, Delta=Delta, eta=eta, tau=tau, J=J, cr=cr, cv=cv, dt=dt, r_sigma=r_sigma, V_sigma=V_sigma)
    (_, aff, *_, W, r, V, nr, nV, tavg), kernel = kerneler(data, aff, rh, Vh, wij, ih, W, r, V, nr, nV, tavg)
    tavgs = []
    for i in (tqdm.trange if progress else range)(int(total_time/dt/16)):
        rng.standard_normal(W.shape, dtype='f', out=W)
        kernel()
        tavgs.append(tavg.flat[:].copy())
    return tavgs, None


if __name__ == '__main__':

    nn = 96
    w = np.random.randn(nn, nn)**2
    d = np.random.rand(nn, nn)**2 * 15
    run_ispc_montbrio(w, d, 60e3, progress=True)