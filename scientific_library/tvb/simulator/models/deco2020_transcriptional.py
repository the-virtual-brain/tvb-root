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


@guvectorize([(float64[:],)*24], '(n),(m)' + ',()'*21 + '->(n)', nopython=True)
def _numba_dfun(S, c, alpha, beta, ratio, ae, be, de, ge, te, wp, we, jn, ai, bi, di, gi, ti, wi, ji, g, l, io, dx):
    """Gufunc for transcriptional model equations."""

    cc = g[0]*jn[0]*c[0]

    jnSe = jn[0]*S[0]

    ie = wp[0]*jnSe - ji[0]*S[1] + we[0]*io[0] + cc
    gain = 1.0+alpha[0]+beta[0]*ratio[0]
    x = (ae[0]*ie - be[0]) * gain
    he = x / (1 - numpy.exp(-de[0]*x))
    dx[0] = - (S[0] / te[0]) + (1.0 - S[0]) * he * ge[0]

    i = jnSe - S[1] + wi[0]*io[0] + l[0]*cc
    x = (ai[0]*i - bi[0]) * gain
    h = x / (1 - numpy.exp(-di[0]*x))
    dx[1] = - (S[1]/ ti[0]) + h * gi[0]

    sigma = numpy.exp(de[0]*(be[0] - ae[0]*i))
    sigma_1 = sigma - 1.0

    term1 = ae[0]*(ji[0]*dx[1] - jn[0]*wp[0]*dx[0]) / sigma_1
    term2 = (ae[0]*de[0]*sigma*(be[0] - ae[0]*i))*(ji[0]*dx[1] - jn[0]*wp[0]*dx[0]) / (sigma_1*sigma_1)

    dx[2] = term1 - term2
    dx[2] = he - S[2]
    dx[3] = ie - S[3]


class Deco2020Transcriptional(ModelNumbaDfun):
    r"""
    Deco, Gustavo & Aquino, Kevin & Arnatkeviciute, Aurina & Oldham, Stuart & Sabaroedin, Kristina & Rogasch,
    Nigel & Kringelbach, Morten & Fornito, Alex. (2020). Dynamical consequences of regional heterogeneity in the
    brain’s transcriptional landscape. 10.1101/2020.10.28.359943.

    """

    # Define traited attributes for this model, these represent possible kwargs.

    alpha = NArray(
        label=":math:`alpha`",
        default=numpy.array([0.0, ]),
        domain=Range(lo=0., hi=1., step=0.01),
        doc="""Parameters for the the scaling between regional biological measures of heterogeneity, R_i, and the
            effective gain within a region""")

    beta = NArray(
        label=":math:`beta`",
        default=numpy.array([0.0, ]),
        domain=Range(lo=0., hi=1., step=0.01),
        doc="""Parameters for the the scaling between regional biological measures of heterogeneity, R_i, and the
            effective gain within a region""")

    ratio = NArray(
        label=":math:`ratio`",
        default=numpy.array([0.0, ]),
        domain=Range(lo=0., hi=1., step=0.01),
        doc="""Parameters for the the scaling between regional biological measures of heterogeneity, R_i, and the
            effective gain within a region""")

    a_e = NArray(
        label=":math:`a_e`",
        default=numpy.array([310., ]),
        domain=Range(lo=0., hi=500., step=1.),
        doc="[n/C]. Excitatory population input gain parameter, chosen to fit numerical solutions.")

    b_e = NArray(
        label=":math:`b_e`",
        default=numpy.array([125., ]),
        domain=Range(lo=0., hi=200., step=1.),
        doc="[Hz]. Excitatory population input shift parameter chosen to fit numerical solutions.")

    d_e = NArray(
        label=":math:`d_e`",
        default=numpy.array([0.160, ]),
        domain=Range(lo=0.0, hi=0.2, step=0.001),
        doc="""[s]. Excitatory population input scaling parameter chosen to fit numerical solutions.""")

    a_i = NArray(
        label=":math:`a_i`",
        default=numpy.array([615., ]),
        domain=Range(lo=0., hi=1000., step=1.),
        doc="[n/C]. Inhibitory population input gain parameter, chosen to fit numerical solutions.")

    b_i = NArray(
        label=":math:`b_i`",
        default=numpy.array([177.0, ]),
        domain=Range(lo=0.0, hi=200.0, step=1.0),
        doc="[Hz]. Inhibitory population input shift parameter chosen to fit numerical solutions.")

    d_i = NArray(
        label=":math:`d_i`",
        default=numpy.array([0.087, ]),
        domain=Range(lo=0.01, hi=0.2, step=0.001),
        doc="""[s]. Inhibitory population input scaling parameter chosen to fit numerical solutions.""")


    gamma_e = NArray(
        label=r":math:`\gamma_e`",
        default=numpy.array([0.000641, ]),
        domain=Range(lo=0.0, hi=0.001, step=0.00001),
        doc="""Excitatory population kinetic parameter""")

    tau_e = NArray(
        label=r":math:`\tau_e`",
        default=numpy.array([100., ]),
        domain=Range(lo=50., hi=150., step=1.),
        doc="""[ms]. Excitatory population NMDA decay time constant.""")

    w_p = NArray(
        label=r":math:`w_p`",
        default=numpy.array([1.4, ]),
        domain=Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Excitatory population recurrence weight""")

    J_N = NArray(
        label=r":math:`J_{N}`",
        default=numpy.array([0.15, ]),
        domain=Range(lo=0.001, hi=0.5, step=0.001),
        doc="""[nA] NMDA current""")

    W_e = NArray(
        label=r":math:`W_e`",
        default=numpy.array([1.0, ]),
        domain=Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Excitatory population external input scaling weight""")

    gamma_i = NArray(
        label=r":math:`\gamma_i`",
        default=numpy.array([0.001, ]),
        domain=Range(lo=0.0, hi=0.002, step=0.0001),
        doc="""Inhibitory population kinetic parameter""")

    tau_i = NArray(
        label=r":math:`\tau_i`",
        default=numpy.array([10., ]),
        domain=Range(lo=5., hi=150., step=1.0),
        doc="""[ms]. Inhibitory population NMDA decay time constant.""")

    J_i = NArray(
        label=r":math:`J_{i}`",
        default=numpy.array([1.0, ]),
        domain=Range(lo=0.001, hi=2.0, step=0.001),
        doc="""[nA] Local inhibitory current""")

    W_i = NArray(
        label=r":math:`W_i`",
        default=numpy.array([0.7, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Inhibitory population external input scaling weight""")

    I_o = NArray(
        label=":math:`I_{o}`",
        default=numpy.array([0.382, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.001),
        doc="""[nA]. Effective external input""")

    G = NArray(
        label=":math:`G`",
        default=numpy.array([2.0, ]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Global coupling scaling""")

    lamda = NArray(
        label=":math:`\lambda`",
        default=numpy.array([0.0, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Inhibitory global coupling scaling""")

    state_variable_range = Final(
        default={
            "S_e": numpy.array([0.0, 1.0]),
            "S_i": numpy.array([0.0, 1.0]),
            "H_e": numpy.array([0.0, 100]),
            "I_e": numpy.array([0.0, 30.0])
        },
        label="State variable ranges [lo, hi]",
        doc="Population firing rate")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"S_e": numpy.array([0.0, 1.0]),
                 "S_i": numpy.array([0.0, 1.0]),
                 "H_e": numpy.array([0.0, 150.0]),
                 "I_e": numpy.array([0.0, 30.0])},
        doc="""The values for each state-variable should be set to encompass
            the boundaries of the dynamic range of that state-variable. Set None for one-sided boundaries""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('H_e', 'I_e', 'S_i', 'S_e'),
        default=('H_e', 'I_e', 'S_i', 'S_e'),
        doc="""default state variables to be monitored""")

    state_variables = ['S_e', 'S_i', 'H_e', 'I_e']
    _nvar = 4
    cvar = numpy.array([0], dtype=numpy.int32)

    def configure(self):
        """  """
        super(Deco2020Transcriptional, self).configure()
        self.update_derived_parameters()

    def initial(self, dt, history_shape, rng=numpy.random):
        """Generates uniformly distributed initial conditions,
        bounded by the state variable limits defined by the model.
        """
        ic = numpy.empty(history_shape)

        ic[:, 0] = 0.001
        ic[:, 1] = 0.001
        ic[:, 2] = 0.001
        ic[:, 3] = 0.001

        return ic

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        """

        S = state_variables[:, :]

        S_e = S[:, 0]
        S_i = S[:, 1]

        c_0 = coupling[:, 0]

        # if applicable
        lc_0 = local_coupling * S_e

        coupling = self.G * self.J_N * (c_0 + lc_0)

        J_N_S_e = self.J_N * S_e

        I_ext = 0.0  # Resting state

        inh = self.J_i * S_i

        I_e = self.W_e * self.I_o + self.w_p * J_N_S_e + coupling - inh + I_ext

        gain = 1.0 + self.alpha + self.beta * self.ratio
        x_e = (self.a_e * I_e - self.b_e) * gain
        H_e = x_e / (1 - numpy.exp(-self.d_e * x_e))

        dS_e = - (S_e / self.tau_e) + (1.0 - S_e) * H_e * self.gamma_e

        I_i = self.W_i * self.I_o + J_N_S_e - S_i + self.lamda * coupling

        x_i = (self.a_i * I_i - self.b_i) * gain
        H_i = x_i / (1 - numpy.exp(-self.d_i * x_i))

        dS_i = - (S_i / self.tau_i) + H_i * self.gamma_i

        dH_e = H_e - S[:, 2]
        dI_e = I_e - S[:, 3]

        derivative = numpy.array([dS_e, dS_i, dH_e, dI_e])

        return derivative

    def dfun(self, x, c, local_coupling=0.0, **kwargs):
        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T + local_coupling * x[0]
        deriv = _numba_dfun(x_, c_,
                            self.alpha, self.beta, self.ratio,
                            self.a_e, self.b_e, self.d_e,
                            self.gamma_e, self.tau_e,
                            self.w_p, self.W_e, self.J_N,
                            self.a_i, self.b_i, self.d_i,
                            self.gamma_i, self.tau_i,
                            self.W_i, self.J_i,
                            self.G, self.lamda, self.I_o)
        return deriv.T[..., numpy.newaxis]
        # deriv = self._numpy_dfun(x_, c_, local_coupling)
        # return deriv[..., numpy.newaxis]

