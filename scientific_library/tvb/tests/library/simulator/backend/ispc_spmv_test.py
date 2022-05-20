# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

import numpy as np
import subprocess, ctypes as ct
from scipy import io, sparse
import subprocess
import pytest

def run():
        
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

import subprocess, cpuinfo, sys, tempfile, os, ctypes as ct, numpy as np

here = os.path.abspath(os.path.dirname(__file__))

def _build_spmv_lib(fname):
    cxx = 'g++'
    info = cpuinfo.get_cpu_info()
    arch_prefix = ''
    if sys.platform == 'darwin' and info['arch'] == 'ARM_8':
        target = 'neon-i32x4'
        arch_prefix = 'arch -arm64 '
    else: # intel
        if 'avx512f' in info['flags']:
            target = 'avx512skx-i32x16'
        elif 'avx2' in info['flags']:
            target = 'avx2-i32x8'
        elif 'avx' in info['flags']:
            target = 'avx-i32x8'
        elif 'sse4_2' in info['flags']:
            target = 'sse4-i32x4'
        else:
            target = 'sse4-i32x4'
    tasksys_fname = 'tvb/simulator/backend/ispc_tasksys.cpp'
    run = lambda cmd: subprocess.check_call(cmd.split(' '))
    with tempfile.TemporaryDirectory() as dir:
        fnameo = f'{dir}/{os.path.basename(fname)}.o'
        tasksyso = f'{dir}/tasksys.o'
        so = f'{dir}/{os.path.basename(fname)}.so'
        run(f'ispc {fname} --target {target} -o {fnameo}')
        run(f'{arch_prefix}{cxx} -std=c++11 -O3 -c {tasksys_fname} -o {tasksyso}')
        run(f'{arch_prefix}{cxx} -shared {tasksyso} {fnameo} -o {so}')
        lib = ct.CDLL(so)
        lib.spmv3.restype = None
        fvec = np.ctypeslib.ndpointer(dtype=np.float32)
        ivec = np.ctypeslib.ndpointer(dtype=np.int32)
        lib.spmv3.argtypes = fvec, ivec, ivec, ct.c_int, ct.c_int, ct.c_int, fvec, fvec, ct.c_int
        return lib


@pytest.mark.parametrize('use', ['scipy', 'ispc'])
@pytest.mark.parametrize('max_dist', [10.0, 20.0])
def test_spmv_perf(benchmark, max_dist, use):
    from tvb.datatypes.surfaces import Surface
    surf = Surface.from_file()
    surf.configure()
    surf.compute_geodesic_distance_matrix(max_dist=max_dist)
    lcmat = surf.geodesic_distance_matrix.astype('f')
    lcmat.data = 1.0 / lcmat.data
    b = np.zeros((lcmat.shape[0], ), 'f') + 0.1
    if use == 'scipy':
        benchmark(lambda: lcmat*b)
    else:
        lcmat = lcmat.tocsr()
        lib = _build_spmv_lib('tvb/simulator/backend/templates/ispc-spmv.ispc.mako')
        out = np.zeros((lcmat.shape[0], ), 'f')
        args = (
            lcmat.data, lcmat.indices, lcmat.indptr,
            lcmat.shape[0], lcmat.shape[1], lcmat.nnz,
            b, out, 1024)
        lib.spmv3(*args)
        print(os.environ)
        np.testing.assert_allclose(lcmat*b, out, rtol=1e-6, atol=1e-6)
        benchmark(lambda: lib.spmv3(*args))
