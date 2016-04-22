
"""
Benchmark GPUs.

"""

import time
import numpy
import math
from numba import cuda, int32, float32
from tvb.simulator._numba.coupling import cu_delay_cfun, next_pow_of_2
from tvb.simulator._numba.util import cu_expr
from tvb.simulator.async import AsyncResult
from randomstate.prng.xorshift128 import xorshift128


cuda.detect()


class AsyncNoise(object):

    def __init__(self, shape, rng):
        self.shape = shape
        self.rng = rng
        self._set_ar()

    def _set_ar(self):
        self._ar = AsyncResult.do(self.rng.randn, *self.shape)

    def get(self):
        noise = self._ar.result
        self._set_ar()
        return noise


def make_kernel(delays, n_thread_per_block, n_inner):
    horizon = next_pow_of_2(delays.max() + 1)
    cfpre = cu_expr('sin(xj - xi)', ('xi', 'xj'), {})
    cfpost = cu_expr('rcp_n * gx', ('gx', ), {'rcp_n': 1.0 / delays.shape[0]})
    n_thread_per_block = int32(n_thread_per_block)
    n_inner = int32(n_inner)
    dcf = cu_delay_cfun(horizon, cfpre, cfpost, 1, n_thread_per_block)
    @cuda.jit
    def kernel(step, state, buf, dt, omega, cvars, weights, delays, a_values, s_values, Z):
        i_t = cuda.threadIdx.x
        i_thread = cuda.blockIdx.x * cuda.blockDim.x + i_t
        aff = cuda.shared.array((1, 1, 1, n_thread_per_block), float32)
        a = a_values[i_thread]
        s = math.sqrt(dt) * math.sqrt(2.0 * s_values[i_thread])
        sqrt_dt = math.sqrt(dt)
        for i_step in range(n_inner):
            for i_post in range(weights.shape[0]):
                dcf(aff, delays, weights, state, i_post, i_thread, step[0], cvars, buf)
                state[0, i_post, 0, i_thread] += \
                    dt * (omega + a * aff[0, 0, 0, i_t]) + s * Z[i_step, i_post, i_thread]
            if i_thread == 0:
                step[0] += 1
            cuda.syncthreads()
    return horizon, kernel


if __name__ == '__main__':
    cuda.close()
    cuda.select_device(0)
    # load data
    path = '/home/mw/Work/CEP/data/mats/aa-t02_%s.txt'
    weights = numpy.loadtxt(path % 'N').astype('f')
    tract_lengths = numpy.loadtxt(path % 'dist')
    # normalize
    weights = weights / weights.sum(axis=0).max()
    dt, omega = 1.0, 10*2.0*math.pi/1e3
    delays = (tract_lengths / 5.0 / dt).astype(numpy.int32)
    # parameter space
    n_grid, n_iter, n_inner = 64, 10, 10
    a_values, s_values = [ary.reshape((-1, )) for ary in 10**numpy.mgrid[1.0:2.0:1j * n_grid, -4.0:-3.0:n_grid * 1j].astype('f')]
    # workspace
    n_nodes, n_threads = weights.shape[0], n_grid**2
    state = numpy.random.rand(1, n_nodes, 1, n_threads).astype('f')
    time_series = numpy.zeros((n_iter, n_nodes, n_threads), 'f')
    step = numpy.zeros((1, ), numpy.int32)
    cvars = numpy.zeros((0, ), numpy.int32)
    # noise
    xorshift128.seed(42)
    async_noise = AsyncNoise((n_inner, n_nodes, n_threads), xorshift128)
    # kernel
    n_thread_per_block = 128
    n_block = int(n_threads / n_thread_per_block)
    horizon, kernel = make_kernel(delays, n_thread_per_block, n_inner)
    buf = numpy.zeros((n_nodes, horizon, 1, n_threads), numpy.float32)
    print 'buf dims', buf.shape
    # begin
    tic = time.time()
    for i in range(n_iter):
        noise = async_noise.get()
        kernel[(n_thread_per_block, ), (n_block,)](
            step, state, buf, dt, omega, cvars, weights, delays, a_values, s_values, noise)
        print step
    toc = time.time() - tic
    print toc
