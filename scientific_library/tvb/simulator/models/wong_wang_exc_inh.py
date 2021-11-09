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

"""
Models based on Wong-Wang's work.

First one follows [DPA_2014], with an excitatory and an inhibitory
population, mutually coupled.

Second adds regional heterogeneity (excitation-inhibition balance) as described in [Deco_2021].


.. moduleauthor:: Dionysios Perdikis <dionperd@gmail.com>
.. moduleauthor:: Ignacio Martín <natx.mc@gmail.com>
.. moduleauthor:: Jan Fousek <jan.fousek@univ-amu.fr>
"""

import numpy
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range
from tvb.simulator.models.base import ModelNumbaDfun


@guvectorize([(float64[:],)*22], '(n),(m)' + ',()'*19 + '->(n)', nopython=True)
def _numba_dfun(S, c, ae, be, de, ge, te, wp, we, jn, ai, bi, di, gi, ti, wi, ji, g, l, io, ie, dx):
    "Gufunc for reduced Wong-Wang model equations."

    cc = g[0]*jn[0]*c[0]

    jnSe = jn[0]*S[0]

    x = wp[0]*jnSe - ji[0]*S[1] + we[0]*io[0] + cc + ie[0]
    x = ae[0]*x - be[0]
    h = x / (1 - numpy.exp(-de[0]*x))
    dx[0] = - (S[0] / te[0]) + (1.0 - S[0]) * h * ge[0]

    x = jnSe - S[1] + wi[0]*io[0] + l[0]*cc
    x = ai[0]*x - bi[0]
    h = x / (1 - numpy.exp(-di[0]*x))
    dx[1] = - (S[1] / ti[0]) + h * gi[0]


class ReducedWongWangExcInh(ModelNumbaDfun):
    r"""
    .. [WW_2006] Kong-Fatt Wong and Xiao-Jing Wang,  *A Recurrent Network
                Mechanism of Time Integration in Perceptual Decisions*.
                Journal of Neuroscience 26(4), 1314-1328, 2006.

    .. [DPA_2014] Deco Gustavo, Ponce Alvarez Adrian, Patric Hagmann,
                  Gian Luca Romani, Dante Mantini, and Maurizio Corbetta. *How Local
                  Excitation–Inhibition Ratio Impacts the Whole Brain Dynamics*.
                  The Journal of Neuroscience 34(23), 7886 –7898, 2014.


    Equations taken from [DPA_2013]_ , page 11242

    .. math::
                 x_{ek}       &=   w_p\,J_N \, S_{ek} - J_iS_{ik} + W_eI_o + GJ_N \mathbf\Gamma(S_{ek}, S_{ej}, u_{kj}) \\
                 H(x_{ek})    &=  \dfrac{a_ex_{ek}- b_e}{1 - \exp(-d_e(a_ex_{ek} -b_e))} \\
                 \dot{S}_{ek} &= -\dfrac{S_{ek}}{\tau_e} + (1 - S_{ek}) \, {\gamma}H(x_{ek}) \\

                 x_{ik}       &=   J_N \, S_{ek} - S_{ik} + W_iI_o + {\lambda}GJ_N \mathbf\Gamma(S_{ik}, S_{ej}, u_{kj}) \\
                 H(x_{ik})    &=  \dfrac{a_ix_{ik} - b_i}{1 - \exp(-d_i(a_ix_{ik} -b_i))} \\
                 \dot{S}_{ik} &= -\dfrac{S_{ik}}{\tau_i} + \gamma_iH(x_{ik}) \

    """

    # Define traited attributes for this model, these represent possible kwargs.

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

    gamma_e = NArray(
        label=r":math:`\gamma_e`",
        default=numpy.array([0.641/1000, ]),
        domain=Range(lo=0.0, hi=1.0/1000, step=0.01/1000),
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
        label=r":math:`J_N`",
        default=numpy.array([0.15, ]),
        domain=Range(lo=0.001, hi=0.5, step=0.001),
        doc="""[nA] NMDA current""")

    W_e = NArray(
        label=r":math:`W_e`",
        default=numpy.array([1.0, ]),
        domain=Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Excitatory population external input scaling weight""")

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
        domain=Range(lo=0.0, hi=0.2, step=0.001),
        doc="""[s]. Inhibitory population input scaling parameter chosen to fit numerical solutions.""")

    gamma_i = NArray(
        label=r":math:`\gamma_i`",
        default=numpy.array([1.0/1000, ]),
        domain=Range(lo=0.0, hi=2.0/1000, step=0.01/1000),
        doc="""Inhibitory population kinetic parameter""")

    tau_i = NArray(
        label=r":math:`\tau_i`",
        default=numpy.array([10., ]),
        domain=Range(lo=5., hi=100., step=1.0),
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

    I_ext = NArray(
        label=":math:`I_{ext}`",
        default=numpy.array([0.0, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.001),
        doc="""[nA]. Effective external stimulus input""")

    G = NArray(
        label=":math:`G`",
        default=numpy.array([2.0, ]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Global coupling scaling""")

    lamda = NArray(
        label=r":math:`\lambda`",
        default=numpy.array([0.0, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Inhibitory global coupling scaling""")

    state_variable_range = Final(
        default={
            "S_e": numpy.array([0.0, 1.0]),
            "S_i": numpy.array([0.0, 1.0])
        },
        label="State variable ranges [lo, hi]",
        doc="Population firing rate")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"S_e": numpy.array([0.0, 1.0]), "S_i": numpy.array([0.0, 1.0])},
        doc="""The values for each state-variable should be set to encompass
            the boundaries of the dynamic range of that state-variable. Set None for one-sided boundaries""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('S_e', 'S_i'),
        default=('S_e', 'S_i'),
        doc="""default state variables to be monitored""")

    state_variables = ['S_e', 'S_i']
    _nvar = 2
    cvar = numpy.array([0], dtype=numpy.int32)

    def configure(self):
        """  """
        super(ReducedWongWangExcInh, self).configure()
        self.update_derived_parameters()

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0):
        S = state_variables[:, :]

        c_0 = coupling[0, :]

        # if applicable
        lc_0 = local_coupling * S[0]

        coupling = self.G * self.J_N * (c_0 + lc_0)

        J_N_S_e = self.J_N * S[0]

        x_e = self.w_p * J_N_S_e - self.J_i * S[1] + self.W_e * self.I_o + coupling + self.I_ext

        x_e = self.a_e * x_e - self.b_e
        H_e = x_e / (1 - numpy.exp(-self.d_e * x_e))

        dS_e = - (S[0] / self.tau_e) + (1 - S[0]) * H_e * self.gamma_e

        x_i = J_N_S_e - S[1] + self.W_i * self.I_o + self.lamda * coupling

        x_i = self.a_i * x_i - self.b_i
        H_i = x_i / (1 - numpy.exp(-self.d_i * x_i))

        dS_i = - (S[1] / self.tau_i) + H_i * self.gamma_i

        derivative = numpy.array([dS_e, dS_i])

        return derivative

    def dfun(self, x, c, local_coupling=0.0, **kwargs):
        r"""
        Equations taken from [DPA_2013]_ , page 11242

        .. math::
                 x_{ek}       &=   w_p\,J_N \, S_{ek} - J_iS_{ik} + W_eI_o + GJ_N \mathbf\Gamma(S_{ek}, S_{ej}, u_{kj}) \\
                 H(x_{ek})    &=  \dfrac{a_ex_{ek}- b_e}{1 - \exp(-d_e(a_ex_{ek} -b_e))} \\
                 \dot{S}_{ek} &= -\dfrac{S_{ek}}{\tau_e} + (1 - S_{ek}){\gamma}H(x_{ek}) \\

                 x_{ik}       &=   J_N \, S_{ek} - S_{ik} + W_iI_o + {\lambda}GJ_N \mathbf\Gamma(S_{ik}, S_{ej}, u_{kj}) \\
                 H(x_{ik})    &=  \dfrac{a_ix_{ik} - b_i}{1 - \exp(-d_i(a_ix_{ik} -b_i))} \\
                 \dot{S}_{ik} &= -\dfrac{S_{ik}}{\tau_i} + \gamma_iH(x_{ik}) \

        """
        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T + local_coupling * x[0]
        deriv = _numba_dfun(x_, c_,
                            self.a_e, self.b_e, self.d_e, self.gamma_e, self.tau_e,
                            self.w_p, self.W_e, self.J_N,
                            self.a_i, self.b_i, self.d_i, self.gamma_i, self.tau_i,
                            self.W_i, self.J_i,
                            self.G, self.lamda, self.I_o, self.I_ext)
        return deriv.T[..., numpy.newaxis]





@guvectorize([(float64[:],)*23], '(n),(m)' + ',()'*20 + '->(n)', nopython=True)
def _numba_dfun_bei(S, c, mi, ae, be, de, ge, te, wp, we, jn, ai, bi, di, gi, ti, wi, ji, g, l, io, ie, dx):
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

class DecoBalancedExcInh(ReducedWongWangExcInh):
    r"""
    .. [Deco_2021] Deco, Gustavo, Morten L. Kringelbach, Aurina Arnatkeviciute,
    Stuart Oldham, Kristina Sabaroedin, Nigel C. Rogasch, Kevin M. Aquino, and
    Alex Fornito. "Dynamical consequences of regional heterogeneity in the
    brain’s transcriptional landscape." Science Advances 7, no. 29 (2021):
    eabf4752.

    Equations extend the [DPA_2013] with effective gain parameter M_i to


    .. math::
                 x_{ek}       &=   w_p\,J_N \, S_{ek} - J_iS_{ik} + W_eI_o + GJ_N \mathbf\Gamma(S_{ek}, S_{ej}, u_{kj}) \\
                 H(x_{ek})    &=  \dfrac{M_i(a_ex_{ek}- b_e)}{1 - \exp(-d_e M_i(a_ex_{ek} -b_e))} \\
                 \dot{S}_{ek} &= -\dfrac{S_{ek}}{\tau_e} + (1 - S_{ek}) \, {\gamma}H(x_{ek}) \\

                 x_{ik}       &=   J_N \, S_{ek} - S_{ik} + W_iI_o + {\lambda}GJ_N \mathbf\Gamma(S_{ik}, S_{ej}, u_{kj}) \\
                 H(x_{ik})    &=  \dfrac{M_i(a_ix_{ik} - b_i)}{1 - \exp(-d_i M_i(a_ix_{ik} -b_i))} \\
                 \dot{S}_{ik} &= -\dfrac{S_{ik}}{\tau_i} + \gamma_iH(x_{ik}) \
    """

    # Define traited attributes for this model, these represent possible kwargs.

    M_i = NArray(
        label=":math:`ratio`",
        default=numpy.array([1.0, ]),
        domain=Range(lo=1.0, hi=10., step=0.01),
        doc="""Effective gain within a region.""")


    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        Numpy dfun for transcriptional model presented in [Deco_2020],
        Dynamical consequences of regional heterogeneity in the brain’s
        transcriptional landscape
        """

        S = state_variables[:, :]

        S_e = S[0, :]
        S_i = S[1, :]

        c_0 = coupling[0, :]

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
        deriv = _numba_dfun_bei(x_, c_,
                            self.M_i,
                            self.a_e, self.b_e, self.d_e, self.gamma_e, self.tau_e,
                            self.w_p, self.W_e, self.J_N,
                            self.a_i, self.b_i, self.d_i, self.gamma_i, self.tau_i,
                            self.W_i, self.J_i,
                            self.G, self.lamda, self.I_o, self.I_ext)
        return deriv.T[..., numpy.newaxis]
        # Numpy version, for debugging purposes
        # return self._numpy_dfun(x, c, local_coupling)

