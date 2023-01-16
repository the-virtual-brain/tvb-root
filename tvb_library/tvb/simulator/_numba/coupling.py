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

import numpy
from numba import cuda, float32, int32
from .util import CUDA_SIM


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
