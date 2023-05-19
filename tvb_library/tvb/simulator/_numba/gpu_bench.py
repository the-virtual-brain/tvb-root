# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
Benchmark GPUs.

"""

import time
import numpy
import math
from randomstate.prng.xorshift128 import xorshift128
from datetime import datetime, timedelta

from numba import cuda, int32, float32
from tvb.simulator._numba.coupling import cu_delay_cfun, next_pow_of_2
from tvb.simulator._numba.util import cu_expr
from tvb.simulator.async import AsyncResult




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
    def kernel(step, state, update, buf, dt, omega, cvars,
	       weights, delays, a_values, s_values, Z):
        i_t = cuda.threadIdx.x
        i_thread = cuda.blockIdx.x * cuda.blockDim.x + i_t
        aff = cuda.shared.array((1, 1, 1, n_thread_per_block), float32)
        a = a_values[i_thread]
        s = math.sqrt(dt) * math.sqrt(2.0 * s_values[i_thread])
        sqrt_dt = math.sqrt(dt)
        for i_step in range(n_inner):
            for i_post in range(weights.shape[0]):
                dcf(aff, delays, weights, state, i_post, i_thread, step[0], cvars, buf)
                update[i_post, i_thread] = dt * (omega + a * aff[0, 0, 0, i_t]) \
			+ s * Z[i_step, i_post, i_thread]
            for i_post in range(weights.shape[0]):
                state[0, i_post, 0, i_thread] += update[i_post, i_thread]
            if i_thread == 0:
                step[0] += 1
            cuda.syncthreads()
    return horizon, kernel


if __name__ == '__main__':
    import sys
    print sys.executable
    cuda.close()
    cuda.select_device(0)
    # load data
    path = '/home/mw/Work/CEP/data/mats/aa-t02_%s.txt'
    weights = numpy.loadtxt(path % 'N').astype('f')
    tract_lengths = numpy.loadtxt(path % 'dist')
    # normalize
    weights = weights / weights.sum(axis=0).max()
    dt, omega = 1.0, 10*2.0*math.pi/1e3
    delays = (tract_lengths / 2.0 / dt).astype(numpy.int32)
    # parameter space
    n_iter = 5 * 60 * 10
    n_grid, n_inner = 64,  100
    a_values, s_values = [ary.reshape((-1, )) for ary in 10**numpy.mgrid[0.0:4.0:1j * n_grid, -5.0:-1.0:n_grid * 1j].astype('f')]
    # workspace
    n_nodes, n_threads = weights.shape[0], n_grid**2
    state = numpy.random.rand(1, n_nodes, 1, n_threads).astype('f')
    update = numpy.zeros((n_nodes, n_threads), numpy.float32)
    from numpy.lib.format import open_memmap
    time_series = open_memmap('/dat4/mw/tvb-test-gpu-time-series.npy', 'w+', numpy.float32, (n_iter, n_nodes, n_threads))
    step = numpy.zeros((1, ), numpy.int32)
    cvars = numpy.zeros((1, ), numpy.int32)
    # noise
    xorshift128.seed(42)
    async_noise = AsyncNoise((n_inner, n_nodes, n_threads), numpy.random)
    # kernel
    n_thread_per_block = 64
    n_block = int(n_threads / n_thread_per_block)
    horizon, kernel = make_kernel(delays, n_thread_per_block, n_inner)
    buf = numpy.zeros((n_nodes, horizon, 1, n_threads), numpy.float32)
    print 'buf dims', buf.shape
    # begin
    tic = time.time()
    print datetime.now().isoformat(' ')
    for i in range(n_iter):
        noise = async_noise.get().astype('f')
        kernel[(n_thread_per_block, ), (n_block,)](
            step, state, update, buf, dt, omega, cvars, weights, delays, a_values, s_values, noise)
        time_series[i] = state[0, :, 0, :]
        if i%10==1:
            pct = i * 1e2 / n_iter
            tta = (time.time() - tic) / pct * (100 - pct)
            eta = (datetime.now() + timedelta(seconds=tta)).isoformat(' ')
            print 'Step %d of %d, %02.2f %% done, ETA %s' % (i, n_iter, pct, eta)
    toc = time.time() - tic
    print toc
