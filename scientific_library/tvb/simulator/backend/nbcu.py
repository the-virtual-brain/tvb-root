# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
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

"""
A Numba CUDA backend.

... moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""


import numpy
import numba
from numba import cuda, float32, int32
import numba.cuda
from .nb import NbBackend


# http://numba.pydata.org/numba-doc/dev/cuda/simulator.html
try:
    CUDA_SIM = int(os.environ['NUMBA_ENABLE_CUDASIM']) == 1
except:
    CUDA_SIM = False


_cu_expr_type_map = {
    int: numba.int32,
    float: numba.float32
}


def cu_expr(expr, parameters, constants, return_fn=False):
    "Generate CUDA device function for given expression, with parameters and constants."
    ns = {}
    template = "from math import *\ndef fn(%s):\n    return %s"
    for name, value in constants.items():
        value_type = type(value)
        if value_type not in _cu_expr_type_map:
            msg = "unhandled constant type: %r" % (value_type, )
            raise TypeError(msg)
        ns[name] = _cu_expr_type_map[value_type](value)
    template %= ', '.join(parameters), expr
    exec(template, ns)
    fn = ns['fn']
    cu_fn = numba.cuda.jit(device=True)(fn)
    if return_fn:
        return cu_fn, fn
    return cu_fn


def cu_simple_cfun(offset, cvar):
    "Construct CUDA device function for simple summation coupling."

    offset = float32(offset)

    @cuda.jit(device=True)
    def cfun(weights, state, i_post, i_thread): # 2*n reads
        H = float32(0.0)
        for j in range(state.shape[0]):
            H += weights[i_post, j] * (state[j, cvar, i_thread] + offset)
        return H

    return cfun


def next_pow_of_2(i):
    "Compute next power of two for integer."
    return int(2**(numpy.floor(numpy.log2(i)) + 1))

# TODO rework for sweep over cfe pars e.g. variants for const & parametrize cfes

def cu_linear_cfe_pre(ai, aj, offset):
    "Construct CUDA device function for pre-summation linear coupling function."
    ai, aj, offset = float32(ai), float32(aj), float32(offset)
    @cuda.jit(device=True)
    def cfe(xi, xj):
        return ai * xi + aj * xj + offset
    return cfe

# NB Difference handled by linear_pre(ai=-1, aj=1)

def cu_linear_cfe_post(slope, offset):
    "Construct CUDA device function for post-summation linear coupling function."
    slope, offset = float32(slope), float32(offset)
    @cuda.jit(device=True)
    def cfe(gx):
        return slope * gx + offset
    return cfe


def cu_tanh_cfe_pre(a, b, midpoint, sigma):
    "Construct CUDA device function for HyperbolicTangent coupling function."
    a, b, midpoint, sigma = [float32(_) for _ in (a, b, midpoint, sigma)]
    from math import tanh
    @cuda.jit(device=True)
    def cfe(xi, xj):
        return a * (1 +  tanh((b * xj - midpoint) / sigma))
    return cfe


def cu_sigm_cfe_post(cmin, cmax, midpoint, a, sigma):
    "Construct CUDA device function for Sigmoidal coupling function."
    cmin, cmax, midpoint, a, sigma = [float32(_) for _ in (cmin, cmax, midpoint, a, sigma)]
    from math import exp
    @cuda.jit(device=True)
    def cfe(gx):
        return cmin + ((cmax - cmin) / (1.0 + exp(-a *((gx - midpoint) / sigma))))
    return cfe

# TODO Sigmoidal Jansen Rit & PreSigmoidal are model specific hacks

def cu_kura_cfe_pre():
    "Construct CUDA device function for Kuramoto coupling function, pre-summation."
    from math import sin
    @cuda.jit(device=True)
    def cfe(xi, xj):
        # TODO slow for large argument
        return sin(xj - xi)
    return cfe


@cuda.jit(device=True)
def _cu_mod_pow_2(i, n):
    "Integer modulo for base power of 2, from CUDA programming guide."
    return i & (n - 1)


# TODO http://stackoverflow.com/a/30524712
def cu_delay_cfun(horizon, cfpre, cfpost, n_cvar, n_thread_per_block, step_stride=0, aff_node_stride=0):
    "Construct CUDA device function for delayed coupling with given pre & post summation functions."

    if horizon < 2 or (horizon & (horizon - 1)) != 0:
        msg = "cu_delay_cfun argument `horizon` should be a positive power of 2, but received %d"
        msg %= horizon
        raise ValueError(msg)

    # 0 except for testing
    step_stride = int32(step_stride)
    aff_node_stride = int32(aff_node_stride)

    @cuda.jit(device=True)
    def dcfun(aff, delays, weights, state, i_post, i_thread, step, cvars, buf):#, delayed_step):

        # shared mem temporary for summation, indexed by block-local thread index
        aff_i = cuda.shared.array((n_cvar, n_thread_per_block), float32)
        i_t = cuda.threadIdx.x

        # 0 except for testing
        step_ = step_stride * step

        # update buffer with state
        for i_cvar in range(cvars.shape[0]):
            buf[i_post, _cu_mod_pow_2(step, horizon), i_cvar, i_thread] = state[step_, i_post, cvars[i_cvar], i_thread]

        # initialize sums to zero
        for i_cvar in range(cvars.shape[0]):
            aff_i[i_cvar, i_t] = float32(0.0)
            #aff[step_, i_post * aff_node_stride, i_cvar, i_thread*0] = float32(0.0)

        # query buffer, summing over cfpre applied to delayed efferent cvar values
        for i_pre in range(weights.shape[0]):
            weight = weights[i_post, i_pre]
            if weight == 0.0:
                continue
            # delayed_step[i_post, i_pre] = _cu_mod_pow_2(step - delays[i_post, i_pre] + horizon, horizon)
            delayed_step = _cu_mod_pow_2(step - delays[i_post, i_pre] + horizon, horizon)
            for i_cvar in range(cvars.shape[0]):
                cval = buf[i_pre, delayed_step, i_cvar, i_thread]
                #aff[step_, i_post * aff_node_stride, i_cvar, i_thread*0] += \
                aff_i[i_cvar, i_t] += \
                    weight * cfpre(state[step_, i_post, cvars[i_cvar], i_thread], cval)

        # apply cfpost
        for i_cvar in range(cvars.shape[0]):
            # i_t use and i_thread for tests...
            aff[step_, i_post * aff_node_stride, i_cvar, i_t] = cfpost(
                aff_i[i_cvar, i_t]
                #aff[step_, i_post * aff_node_stride, i_cvar, i_thread*0]
            )

    return dcfun

# TODO these should be generated from model dfun attrs

def make_bistable():
    "Construct CUDA device function for a bistable system."

    @cuda.jit(device=True)
    def f(dX, X, I):
        t = cuda.threadIdx.x
        x = X[0, t]
        dX[0, t] = (x - x*x*x - float32(1.0) + I) / float32(50.0)

    return f


def make_jr():
    "Construct CUDA device function for the Jansen-Rit model."

    # parameters
    A   ,    B,   a,    b,   v0, nu_max,    r,     J, a_1, a_2,  a_3,  a_4, p_min, p_max,   mu = list(map(float32, [
    3.25, 22.0, 0.1, 0.05, 5.52, 0.0025, 0.56, 135.0, 1.0, 0.8, 0.25, 0.25,  0.12,  0.32, 0.22]))

    # jit maps this to CUDA exp
    from math import exp

    @cuda.jit(device=True)
    def f(dX, X, I):
        one, two = float32(1.0), float32(2.0)
        t = cuda.threadIdx.x
        dX[0, t] = X[3, t]
        dX[1, t] = X[4, t]
        dX[2, t] = X[5, t]
        dX[3, t] = A * a * two * nu_max / (one + exp(r * (v0 - (X[1, t] - X[2, t])))) - float32(2.0) * a * X[3, t] - a * a * X[0, t]
        dX[4, t] = A * a * (mu + a_2 * J * two * nu_max / (one + exp(r * (v0 - (a_1 * J * X[0, t])))) + I) - two * a * X[4, t] - a * a * X[1, t]
        dX[5, t] = B * b * (a_4 * J * two * nu_max / (one + exp(r * (v0 - (a_3 * J * X[0, t]))))) - two * b * X[5, t] - b * b * X[2, t]

    return f


def make_euler(dt, f, n_svar, n_step):
    "Construct CUDA device function for Euler scheme."

    n_step = int32(n_step)
    dt = float32(dt)

    @cuda.jit(device=True)
    def scheme(X, I):
        dX = cuda.local.array((n_svar,), float32)
        for i in range(n_step):
            f(dX, X, I)
            for j in range(n_svar):
                X[j] += dX[j]

    return scheme

# TODO Heun

def make_rk4(dt, f, n_svar, n_step):
    "Construct CUDA device function for Runge-Kutta 4th order scheme."

    n_step = int32(n_step)
    dt = float32(dt)

    @cuda.jit(device=True)
    def scheme(X, I):
        k = cuda.shared.array((4, n_svar, 64), float32)
        x = cuda.shared.array((n_svar, 64), float32)
        t = cuda.threadIdx.x
        for i in range(n_step):
            f(k[0], X, I)
            for j in range(n_svar):
                x[j, t] = X[j, t] + (dt / float32(2.0)) * k[0, j, t]
            f(k[1], x, I)
            for j in range(n_svar):
                x[j, t] = X[j, t] + (dt / float32(2.0)) * k[1, j, t]
            f(k[2], x, I)
            for j in range(n_svar):
                x[j, t] = X[j, t] + dt * k[2, j, t]
            f(k[3], x, I)
            for j in range(n_svar):
                X[j, t] += (dt/float32(6.0)) * (k[0, j, t] + k[3, j, t] + float32(2.0)*(k[1, j, t] + k[2, j, t]))

    return scheme

def make_loop(cfun, model, n_svar):
    "Construct CUDA device function for integration loop."

    @cuda.jit
    def loop(n_step, W, X, G):
        # TODO only for 1D grid/block dims
        t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        ti = cuda.threadIdx.x
        x = cuda.shared.array((n_svar, 64), float32)
        g = G[t]
        for j in range(n_step):
            for i in range(W.shape[0]):
                x[:, ti] = X[i, :, t]
                model(x, g * cfun(W, X, i, t))
                X[i, :, t] = x[:, ti]

    # TODO hack
    loop.n_svar = n_svar
    return loop


class NbCuBackend(NbBackend):
    pass


if __name__ == '__main__':

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


def run_bench(conn_txt):
    cuda.close()
    cuda.select_device(0)
    # load data
    weights = numpy.loadtxt(conn_txt % 'N').astype('f')
    tract_lengths = numpy.loadtxt(conn_txt % 'dist')
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
    # begin
    tic = time.time()
    for i in range(n_iter):
        noise = async_noise.get().astype('f')
        kernel[(n_thread_per_block, ), (n_block,)](
            step, state, update, buf, dt, omega, cvars, weights, delays, a_values, s_values, noise)
        time_series[i] = state[0, :, 0, :]
        if i%10==1:
            pct = i * 1e2 / n_iter
            tta = (time.time() - tic) / pct * (100 - pct)
            eta = (datetime.now() + timedelta(seconds=tta)).isoformat(' ')
    toc = time.time() - tic
    print(toc)
