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
Tests for the experiment Numba simulator component implementations.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import pytest
import numpy
import numpy.testing

try:
    import numba
    from numba import cuda
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    CUDA_SIM = True


from tvb.tests.library import setup_test_console_env
setup_test_console_env()

from tvb.tests.library.base_testcase import BaseTestCase

if HAVE_NUMBA:
    from tvb.simulator._numba.coupling import (cu_simple_cfun, next_pow_of_2, cu_linear_cfe_post, cu_linear_cfe_pre,
                                               cu_delay_cfun, _cu_mod_pow_2, cu_tanh_cfe_pre, cu_sigm_cfe_post,
                                               cu_kura_cfe_pre)
    from tvb.simulator._numba.util import CUDA_SIM, cu_expr

from tvb.simulator import coupling as py_coupling, simulator, models, monitors, integrators
from tvb.datatypes import connectivity


skip_if_no_numba = pytest.mark.skipif(not HAVE_NUMBA, reason="Numba unavailable")

class CudaBaseCase(BaseTestCase):

    def setup_method(self): pass

    # Implementations and tests written for a 1D block & 1D grid layout.
    # Results in 4 threads under simulator.
    block_dim, grid_dim = ((2, ), (2, )) if CUDA_SIM else ((16, ), (2, ))
    n_thread = block_dim[0] * grid_dim[0]

    def jit_and_run(self, *args):
        "Convenience decorator for compiling and running a kernel."
        def _(kernel):
            kernel = cuda.jit(kernel)
            kernel[self.block_dim, self.grid_dim](*args)
        return _


class TestSimpleCfun(CudaBaseCase):

    def _run_n_node(self, n_node):
        weights = numpy.random.randn(n_node, n_node).astype('f')
        state = numpy.random.randn(n_node, 2, self.n_thread).astype('f')
        out = numpy.zeros((n_node, self.n_thread)).astype('f')
        offset = 0.5
        cf = cu_simple_cfun(offset, 1)
        @self.jit_and_run(out, weights, state)
        def kernel(out, weights, state):
            t = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
            for i in range(weights.shape[0]):
                out[i, t] = cf(weights, state, i, t)
        expected = weights.dot(state[:, 1] + offset)
        ok = numpy.allclose(expected, out, 1e-4, 1e-5)
        se = numpy.sqrt((expected - out)**2)
        numpy.testing.assert_allclose(out, expected, 1e-4, 1e-5)

    @skip_if_no_numba
    def test_cf(self):
        for n_node in [3, 5, 10, 50]:
            self._run_n_node(n_node)


class TestUtils(CudaBaseCase):

    @skip_if_no_numba
    def test_pow_2(self):
        for n, e in ((5, 8), (32, 64), (63, 64)):
            assert e == next_pow_of_2(n)

    @skip_if_no_numba
    def test_mod_pow_2(self):
        n = 32
        i = numpy.r_[:self.n_thread].astype(numpy.int32)
        out = (i * 0).astype(numpy.int32)
        @self.jit_and_run(out, i, n)
        def kernel(out, i, n):
            t = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
            out[t] = _cu_mod_pow_2(i[t], n)
        numpy.testing.assert_equal(out, i % n)


class TestCfunExpr(CudaBaseCase):

    @skip_if_no_numba
    def test_linear_post(self):
        slope, intercept = 0.2, 0.5
        out = numpy.zeros((self.n_thread, ), 'f')
        state = numpy.random.rand(self.n_thread).astype('f')
        post = cu_linear_cfe_post(slope, intercept)

        @self.jit_and_run(out, state)
        def kernel(out, state):
            t = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
            out[t] = post(state[t])

        numpy.testing.assert_allclose(out, state*slope + intercept, 1e-4, 1e-5)

    @skip_if_no_numba
    def test_linear_pre(self):
        ai, aj, intercept = -0.2, 0.3, 0.25
        out = numpy.zeros((self.n_thread,), 'f')
        xj, xi = numpy.random.rand(2, self.n_thread).astype('f')
        pre = cu_linear_cfe_pre(ai, aj, intercept)

        @self.jit_and_run(out, xi, xj)
        def kernel(out, xi, xj):
            t = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
            out[t] = pre(xi[t], xj[t])

        numpy.testing.assert_allclose(out, ai * xi + aj * xj + intercept, 1e-4, 1e-5)

    @skip_if_no_numba
    def test_tanh(self):
        py_cf = py_coupling.HyperbolicTangent()
        cu_cf = cu_tanh_cfe_pre(py_cf.a[0], py_cf.b[0], py_cf.midpoint[0], py_cf.sigma[0])
        out = numpy.zeros((self.n_thread,), 'f')
        xj, xi = numpy.random.rand(2, self.n_thread).astype('f')

        @self.jit_and_run(out, xi, xj)
        def kernel(out, xi, xj):
            t = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
            out[t] = cu_cf(xi[t], xj[t])

        numpy.testing.assert_allclose(out, py_cf.pre(xi, xj))

    @skip_if_no_numba
    def test_sigm(self):
        py_cf = py_coupling.Sigmoidal()
        cu_cf = cu_sigm_cfe_post(py_cf.cmin[0], py_cf.cmax[0], py_cf.midpoint[0], py_cf.a[0], py_cf.sigma[0])
        out = numpy.zeros((self.n_thread,), 'f')
        gx = numpy.random.rand(self.n_thread).astype('f')

        @self.jit_and_run(out, gx)
        def kernel(out, gx):
            t = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
            out[t] = cu_cf(gx[t])

        numpy.testing.assert_allclose(out, py_cf.post(gx), 1e-4, 1e-5)

    @skip_if_no_numba
    def test_kura(self):
        py_cf = py_coupling.Kuramoto()
        cu_cf = cu_kura_cfe_pre()
        out = numpy.zeros((self.n_thread,), 'f')
        xj, xi = numpy.random.rand(2, self.n_thread).astype('f')

        @self.jit_and_run(out, xi, xj)
        def kernel(out, xi, xj):
            t = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
            out[t] = cu_cf(xi[t], xj[t])

        numpy.testing.assert_allclose(out, py_cf.pre(xi, xj), 1e-5, 1e-6)


class TestDcfun(CudaBaseCase):

    def _do_for_params(self, horizon):

        # setup test data
        numpy.random.seed(42)
        n_step, n_node, n_cvar, n_svar, n_thread = 100, 5, 2, 4, self.n_thread
        cvars = numpy.random.randint(0, n_svar, n_cvar).astype(numpy.int32)
        out = numpy.zeros((n_step, n_node, n_cvar, n_thread), numpy.float32)
        delays = numpy.random.randint(0, horizon - 2, (n_node, n_node)).astype(numpy.int32)
        weights = numpy.random.randn(n_node, n_node).astype(numpy.float32)
        weights[numpy.random.rand(*weights.shape) < 0.25] = 0.0
        state = numpy.random.randn(n_step, n_node, n_svar, n_thread).astype(numpy.float32)
        buf = numpy.zeros((n_node, horizon, n_cvar, n_thread), numpy.float32)
        # debugging
        delayed_step = numpy.zeros_like(delays)

        # setup cu functions
        pre = cu_linear_cfe_pre(0.0, 1.0, 0.0)
        post = cu_linear_cfe_post(1.0, 0.0)
        dcf = cu_delay_cfun(horizon, pre, post, n_cvar, self.block_dim[0], step_stride=1, aff_node_stride=1)

        # run it
        @self.jit_and_run(out, delays, weights, state, cvars, buf)#,delayed_step)
        def kernel(out, delays, weights, state, cvars, buf):#, delayed_step):
            i_thread = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
            for step in range(state.shape[0]):
                for i_post in range(state.shape[1]):
                    dcf(out, delays, weights, state, i_post, i_thread, step, cvars, buf)#,delayed_step)

        # ensure buffer is updating correctly
        buf_state = numpy.roll(state[:, :, cvars][-horizon:].transpose((1, 0, 2, 3)), n_step, axis=1)
        numpy.testing.assert_allclose(buf, buf_state)

        # ensure buffer time indexing is correct
        # numpy.testing.assert_equal(delayed_step, (n_step - 1 - delays + horizon) % horizon)

        # replay
        nodes = numpy.tile(numpy.r_[:n_node], (n_node, 1))
        for step in range(horizon + 3, n_step):
            delayed_state = state[:, :, cvars][step - delays, nodes] # (n_node, n_node, n_cvar, n_thread)
            afferent = (weights.reshape((n_node, n_node, 1, 1)) * delayed_state).sum(axis=1) # (n_node, n_cvar, n_thread)
            numpy.testing.assert_allclose(afferent, out[step], 1e-5, 1e-6)

    @skip_if_no_numba
    @pytest.mark.skipif(CUDA_SIM, reason="https://github.com/numba/numba/issues/1837")
    def test_dcfun_horizons(self):
        self.assertRaises(ValueError, self._do_for_params, 13)
        # with pytest.raises(ValueError):
        #     self._do_for_params(13)
        for horizon in (16, 32, 64):
            self._do_for_params(horizon)


class TestCuExpr(CudaBaseCase):
    "Test generic generation of single expression device functions."

    @skip_if_no_numba
    def test_linear_full_pars(self):
        expr = 'ai * xi + aj * xj + offset'
        pars = 'ai aj xi xj offset'.split()
        const = {}
        cu_fn, fn = cu_expr(expr, pars, const, return_fn=True)
        pars = numpy.random.randn(5, 10, self.n_thread).astype(numpy.float32)
        out = numpy.zeros((10, self.n_thread), numpy.float32)

        @self.jit_and_run(out, *pars)
        def kernel(out, ai, aj, xi, xj, offset):
            i_thread = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
            for i in range(out.shape[0]):
                out[i, i_thread] = cu_fn(ai[i, i_thread], aj[i, i_thread], xi[i, i_thread],
                                         xj[i, i_thread], offset[i, i_thread])

        numpy.testing.assert_allclose(out, fn(*pars), 1e-5, 1e-6)

    @skip_if_no_numba
    def test_linear_constant_slopes(self):
        expr = 'ai * xi + aj * xj + offset'
        pars = 'xi xj offset'.split()
        const = {'ai': 0.3, 'aj': -0.84}
        cu_fn, fn = cu_expr(expr, pars, const, return_fn=True)
        pars = numpy.random.randn(3, 10, self.n_thread).astype(numpy.float32)
        out = numpy.zeros((10, self.n_thread), numpy.float32)

        @self.jit_and_run(out, *pars)
        def kernel(out, xi, xj, offset):
            i_thread = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
            for i in range(out.shape[0]):
                out[i, i_thread] = cu_fn(xi[i, i_thread], xj[i, i_thread], offset[i, i_thread])

        numpy.testing.assert_allclose(out, fn(*pars), 1e-5, 1e-6)

    @skip_if_no_numba
    def test_math_functions(self):
        cu_fn = cu_expr('exp(x) + sin(y)', ['x', 'y'], {})
        x, y = numpy.random.randn(2, self.n_thread).astype(numpy.float32)
        out = numpy.zeros((self.n_thread,), numpy.float32)

        @self.jit_and_run(out, x, y)
        def kernel(out, x, y):
            i_thread = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
            out[i_thread] = cu_fn(x[i_thread], y[i_thread])

        numpy.testing.assert_allclose(out, numpy.exp(x) + numpy.sin(y), 1e-5, 1e-6)


class TestSim(CudaBaseCase):

    @skip_if_no_numba
    @pytest.mark.skipif(CUDA_SIM, reason="https://github.com/numba/numba/issues/1837")
    def test_kuramoto(self):

        # build & run Python simulations
        numpy.random.seed(42)
        n = 5

        weights = numpy.zeros((n, n), numpy.float32)
        idelays = numpy.zeros((n, n), numpy.int32)
        for i in range(n - 1):
            idelays[i, i + 1] = i + 1
            weights[i, i + 1] = i + 1

        def gen_sim(a):
            dt = 0.1
            conn = connectivity.Connectivity()
            conn.weights = weights
            conn.tract_lengths = idelays * dt
            conn.speed = 1.0
            sim = simulator.Simulator(
                coupling=py_coupling.Kuramoto(a=a),
                connectivity=conn,
                model=models.Kuramoto(omega=100 * 2 * numpy.pi / 1e3),
                monitors=monitors.Raw(),
                integrator=integrators.EulerDeterministic(dt=dt)
            )
            sim.configure()
            sim.history[:] = 0.1
            return sim

        a_values = numpy.r_[:self.n_thread].astype(numpy.float32)
        sims = [gen_sim(a) for a in a_values]

        py_data = []
        py_coupling0 = []
        for sim in sims:
            ys = []
            cs = []
            for (t, y), in sim(simulation_length=10.0):
                ys.append(y[0, :, 0])
                # cs.append(sim.model._coupling_0[:, 0])
            py_data.append(numpy.array(ys))
            # py_coupling0.append(numpy.array(cs))
        py_data = numpy.array(py_data)
        # py_coupling0 = numpy.array(py_coupling0)

        # build CUDA kernels
        cfpre = cu_expr('sin(xj - xi)', ('xi', 'xj'), {})
        cfpost = cu_expr('rcp_n * gx', ('gx', ), {'rcp_n': 1.0 / n})
        horiz2 = next_pow_of_2(sims[0].horizon)
        dcf = cu_delay_cfun(horiz2, cfpre, cfpost, 1, self.block_dim[0], aff_node_stride=1)

        # build kernel
        dt = numba.float32(sims[0].integrator.dt)
        omega = numba.float32(sims[0].model.omega[0])
        cvars = numpy.array([0], numpy.int32)
        weights = sims[0].connectivity.weights.astype(numpy.float32)
        delays = sims[0].connectivity.idelays.astype(numpy.int32)

        @cuda.jit
        def kernel(step, state, coupling, aff, buf, dt, omega, cvars, weights, delays, a_values):
            i_thread = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            a = a_values[i_thread]
            for i_post in range(weights.shape[0]):
                dcf(aff, delays, weights, state, i_post, i_thread, step[0], cvars, buf)
                coupling[i_post, i_thread] = a * aff[0, i_post, 0, i_thread]
                state[0, i_post, 0, i_thread] += dt * (omega + a * aff[0, i_post, 0, i_thread])

        step = numpy.array([0], numpy.int32)
        state = (numpy.zeros((1, n, 1, self.n_thread)) + 0.1).astype(numpy.float32)
        coupling0 = numpy.zeros((n, self.n_thread), numpy.float32)
        aff = numpy.zeros((1, n, 1, self.n_thread), numpy.float32)
        buf = numpy.zeros((n, horiz2, 1, self.n_thread), numpy.float32)
        buf += 0.1

        cu_data = numpy.zeros(py_data.shape, numpy.float32)
        cu_coupling0 = numpy.zeros((cu_data.shape[1], ) + coupling0.shape)
        for step_ in range(cu_data.shape[1]):
            step[0] = step_
            kernel[self.block_dim, self.grid_dim](step, state, coupling0, aff, buf, dt, omega, cvars, weights, delays, a_values)
            cu_data[:, step_] = state[0, :, 0].T
            cu_coupling0[step_] = coupling0

        # Maybe accept higher error because it accumulates over time
        numpy.testing.assert_allclose(cu_data, py_data, 1e-2, 1e-2)
