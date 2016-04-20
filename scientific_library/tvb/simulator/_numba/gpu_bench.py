
"""
Benchmark GPUs.

"""

import time
import numpy
import numba.cuda
numba.cuda.detect()

from tvb.simulator._numba.models import make_bistable, make_jr
from tvb.simulator._numba.integrators import make_euler, make_rk4
from tvb.simulator._numba.coupling import make_cfun
from tvb.simulator._numba.loops import make_loop


def select_device_and_build_kernel(id, n_step, sch, jr=False):
    numba.cuda.close(); numba.cuda.select_device(id)
    if jr:
        n_svar = 6
        dt = 0.01
        model = make_jr()
        cfun = make_cfun(0.0, 1)
    else:
        n_svar = 1
        dt = 0.1
        model = make_bistable()
        cfun = make_cfun(2.0, 0)

    scheme = sch(dt, model, n_svar, n_step)
    kernel = make_loop(cfun, scheme, n_svar)
    return kernel


def run(kernel, W, G, Sig, n_iter= 1, block_size=128, n_inner=20):
    n_nodes, n_threads = W.shape[0], G.size
    X = numba.cuda.to_device(numpy.random.rand(n_nodes, kernel.n_svar, n_threads).astype('f'))
    Xs = numpy.zeros((n_iter, ) + X.shape, 'f')
    tic = time.time()
    for i in range(n_iter):
        kernel[(n_threads/block_size, ), (block_size,)](n_inner, W, X, G)
        numba.cuda.synchronize()
        Xs[i] = X.copy_to_host()
    toc = time.time() - tic
    return Xs, toc, n_inner*n_threads*n_nodes*(2 + 2*n_nodes), n_inner*n_threads*n_nodes


def benchmark(weights_matrix):
    W = weights_matrix.astype('f')
    W /= W.sum(axis=0).max() # normalize by in-strength
    for ng in (64, ):#(32, 64, 128):
        G, Sig = 10**numpy.mgrid[1.0:2.0:1j*ng, -4.0:-3.0:ng*1j].astype('f')
        msg = 'ng %d, %s %d block, %s, %s, %d, result shape %r required %.3f s'
        for i in range(len(numba.cuda.devices.gpus)):
            for jr in (True, False):
                for bs in (64, ):#(32, 64, 128):
                    for sch in (make_rk4,):#(make_euler, make_rk4):
                        for n_step in (10,):#(1, 10):
                            kernel = select_device_and_build_kernel(i, n_step, sch, jr=jr)
                            Xs, t, nr, nw = run(kernel, W, G.ravel(), Sig.ravel(), block_size=bs)
                            mname = 'jr' if jr else 'bs'
                            print msg % (ng, numba.cuda.gpus.current.name, bs, mname, sch, n_step, Xs.shape, t)


if __name__ == '__main__':
    benchmark(numpy.loadtxt('/home/mw/Work/CEP/data/mats/aa-t02_N.txt'))

"""
x shared

ng 64, GeForce GTX 970 64 block, jr, <function make_rk4 at 0x7fb17f53dc80>, 10, result shape (1, 164, 6, 4096) required 1.365 s
ng 64, GeForce GTX 970 64 block, bs, <function make_rk4 at 0x7fb17f53dc80>, 10, result shape (1, 164, 1, 4096) required 1.032 s
ng 64, Quadro K2200 64 block, jr, <function make_rk4 at 0x7fb17f53dc80>, 10, result shape (1, 164, 6, 4096) required 1.082 s
ng 64, Quadro K2200 64 block, bs, <function make_rk4 at 0x7fb17f53dc80>, 10, result shape (1, 164, 1, 4096) required 0.604 s

w row shared

ng 64, GeForce GTX 970 64 block, jr, <function make_rk4 at 0x7f178fc5ec80>, 10, result shape (1, 164, 6, 4096) required 1.428 s
ng 64, GeForce GTX 970 64 block, bs, <function make_rk4 at 0x7f178fc5ec80>, 10, result shape (1, 164, 1, 4096) required 1.003 s
ng 64, Quadro K2200 64 block, jr, <function make_rk4 at 0x7f178fc5ec80>, 10, result shape (1, 164, 6, 4096) required 1.289 s
ng 64, Quadro K2200 64 block, bs, <function make_rk4 at 0x7f178fc5ec80>, 10, result shape (1, 164, 1, 4096) required 0.676 s

fix use of per thread temp vars (previous just wrong)

ng 64, GeForce GTX 970 64 block, jr, <function make_rk4 at 0x7f29f6137c80>, 10, result shape (1, 164, 6, 4096) required 1.501 s
ng 64, GeForce GTX 970 64 block, bs, <function make_rk4 at 0x7f29f6137c80>, 10, result shape (1, 164, 1, 4096) required 0.984 s
ng 64, Quadro K2200 64 block, jr, <function make_rk4 at 0x7f29f6137c80>, 10, result shape (1, 164, 6, 4096) required 1.346 s
ng 64, Quadro K2200 64 block, bs, <function make_rk4 at 0x7f29f6137c80>, 10, result shape (1, 164, 1, 4096) required 0.725 s

rm w row shared

ng 64, GeForce GTX 970 64 block, jr, <function make_rk4 at 0x7f53119c5c80>, 10, result shape (1, 164, 6, 4096) required 1.400 s
ng 64, GeForce GTX 970 64 block, bs, <function make_rk4 at 0x7f53119c5c80>, 10, result shape (1, 164, 1, 4096) required 0.971 s
ng 64, Quadro K2200 64 block, jr, <function make_rk4 at 0x7f53119c5c80>, 10, result shape (1, 164, 6, 4096) required 1.032 s
ng 64, Quadro K2200 64 block, bs, <function make_rk4 at 0x7f53119c5c80>, 10, result shape (1, 164, 1, 4096) required 0.644 s

"""