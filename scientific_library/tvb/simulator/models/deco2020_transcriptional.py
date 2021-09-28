# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2021, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

"""
Model based on:

Deco, Gustavo & Aquino, Kevin & Arnatkeviciute, Aurina & Oldham, Stuart & Sabaroedin, Kristina & Rogasch,
Nigel & Kringelbach, Morten & Fornito, Alex. (2020). Dynamical consequences of regional heterogeneity in the
brain’s transcriptional landscape. 10.1101/2020.10.28.359943.

.. moduleauthor:: Ignacio Martín <natx.mc@gmail.com>
"""

import numpy
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range
from tvb.simulator.models.base import ModelNumbaDfun

from tvb.simulator.models.wong_wang_exc_inh import ReducedWongWangExcInh


@guvectorize([(float64[:],)*23], '(n),(m)' + ',()'*20 + '->(n)', nopython=True)
def _numba_dfun(S, c, mi, ae, be, de, ge, te, wp, we, jn, ai, bi, di, gi, ti, wi, ji, g, l, io, ie, dx):
    """Gufunc for transcriptional model presented in Deco et Al 2020, Dynamical consequences of regional heterogeneity in the
    brain’s transcriptional landscape"""

    cc = g[0]*jn[0]*c[0]

    jnSe = jn[0]*S[0]

    x = wp[0]*jnSe - ji[0]*S[1] + we[0]*io[0] + cc + ie[0]
    x = (ae[0]*x - be[0]) * mi[0]
    h = x / (1 - numpy.exp(-de[0]*x))
    dx[0] = - (S[0] / te[0]) + (1.0 - S[0]) * h * ge[0]

    x = jnSe - S[1] + wi[0]*io[0] + l[0]*cc
    x = (ai[0]*x - bi[0]) * mi[0]
    h = x / (1 - numpy.exp(-di[0]*x))
    dx[1] = - (S[1] / ti[0]) + h * gi[0]


class Deco2020Transcriptional(ReducedWongWangExcInh):
    r"""
    Deco, Gustavo & Aquino, Kevin & Arnatkeviciute, Aurina & Oldham, Stuart & Sabaroedin, Kristina & Rogasch,
    Nigel & Kringelbach, Morten & Fornito, Alex. (2020). Dynamical consequences of regional heterogeneity in the
    brain’s transcriptional landscape. 10.1101/2020.10.28.359943.

    """

    # Define traited attributes for this model, these represent possible kwargs.

    M_i = NArray(
        label=":math:`ratio`",
        default=numpy.array([1.0, ]),
        domain=Range(lo=1.0, hi=10., step=0.01),
        doc="""Effective gain within a region.""")


    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        Numpy dfun for transcriptional model presented in Deco et Al 2020, Dynamical consequences of regional heterogeneity in the
        brain’s transcriptional landscape
        """

        S = state_variables[:, :]

        S_e = S[:, 0]
        S_i = S[:, 1]

        c_0 = coupling[:, 0]

        # if applicable
        lc_0 = local_coupling * S_e

        coupling = self.G * self.J_N * (c_0 + lc_0)

        J_N_S_e = self.J_N * S_e

        inh = self.J_i * S_i

        I_e = self.W_e * self.I_o + self.w_p * J_N_S_e + coupling - inh + self.I_ext

        x_e = (self.a_e * I_e - self.b_e) * self.M_i
        H_e = x_e / (1 - numpy.exp(-self.d_e * x_e))

        dS_e = - (S_e / self.tau_e) + (1.0 - S_e) * H_e * self.gamma_e

        I_i = self.W_i * self.I_o + J_N_S_e - S_i + self.lamda * coupling

        x_i = (self.a_i * I_i - self.b_i) * self.M_i
        H_i = x_i / (1 - numpy.exp(-self.d_i * x_i))

        dS_i = - (S_i / self.tau_i) + H_i * self.gamma_i

        derivative = numpy.array([dS_e, dS_i])

        return derivative

    def dfun(self, x, c, local_coupling=0.0, **kwargs):
        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T + local_coupling * x[0]
        deriv = _numba_dfun(x_, c_,
                            self.M_i,
                            self.a_e, self.b_e, self.d_e, self.gamma_e, self.tau_e,
                            self.w_p, self.W_e, self.J_N,
                            self.a_i, self.b_i, self.d_i, self.gamma_i, self.tau_i,
                            self.W_i, self.J_i,
                            self.G, self.lamda, self.I_o, self.I_ext)
        return deriv.T[..., numpy.newaxis]
        # Numpy version, for debugging purposes
        # deriv = self._numpy_dfun(x_, c_, local_coupling)
        # return deriv[..., numpy.newaxis]

