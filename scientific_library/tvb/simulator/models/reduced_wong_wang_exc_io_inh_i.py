# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

"""

from numba import guvectorize, float64
from tvb.simulator.models.base import numpy, ModelNumbaDfun
from tvb.basic.neotraits.api import NArray, Final, List, Range


@guvectorize([(float64[:],)*19], '(n),(m)' + ',()'*16 + '->(n)', nopython=True)
def _numba_update_non_state_variables(S, c, ae, be, de, wp, we, jn, re, ai, bi, di, wi, ji, ri, g, l, io, newS):
    "Gufunc for reduced Wong-Wang model equations."

    cc = g[0]*jn[0]*c[0]

    jnSe = jn[0] * S[0]

    if re[0] < 0.0:
        x = wp[0]*jnSe - ji[0]*S[1] + we[0]*io[0] + cc
        x = ae[0]*x - be[0]
        h = x / (1 - numpy.exp(-de[0]*x))
        S[2] = h

    if ri[0] < 0.0:
        x = jnSe - S[1] + wi[0]*io[0] + l[0]*cc
        x = ai[0]*x - bi[0]
        h = x / (1 - numpy.exp(-di[0]*x))
        S[3] = h

    newS[0] = S[0]
    newS[1] = S[1]
    newS[3] = S[2]
    newS[4] = S[4]


@guvectorize([(float64[:],)*6], '(n)' + ',()'*4 + '->(n)', nopython=True)
def _numba_dfun(S, ge, te, gi, ti, dx):
    "Gufunc for reduced Wong-Wang model equations."

    dx[0] = - (S[0] / te[0]) + (1.0 - S[0]) * S[2] * ge[0]
    dx[2] = 0.0

    dx[1] = - (S[1] / ti[0]) + S[3] * gi[0]
    dx[3] = 0.0


class ReducedWongWangExcIOInhI(ModelNumbaDfun):
    r"""
    .. [WW_2006] Kong-Fatt Wong and Xiao-Jing Wang,  *A Recurrent Network
                Mechanism of Time Integration in Perceptual Decisions*.
                Journal of Neuroscience 26(4), 1314-1328, 2006.

    .. [DPA_2014] Deco Gustavo, Ponce Alvarez Adrian, Patric Hagmann,
                  Gian Luca Romani, Dante Mantini, and Maurizio Corbetta. *How Local
                  Excitation–Inhibition Ratio Impacts the Whole Brain Dynamics*.
                  The Journal of Neuroscience 34(23), 7886 –7898, 2014.



    .. automethod:: ReducedWongWang.__init__

    Equations taken from [DPA_2013]_ , page 11242

    .. math::
                 x_{ek}       &=   w_p\,J_N \, S_{ek} - J_iS_{ik} + W_eI_o + GJ_N \mathbf\Gamma(S_{ek}, S_{ej}, u_{kj}),\\
                 H(x_{ek})    &=  \dfrac{a_ex_{ek}- b_e}{1 - \exp(-d_e(a_ex_{ek} -b_e))},\\
                 \dot{S}_{ek} &= -\dfrac{S_{ek}}{\tau_e} + (1 - S_{ek}) \, \gammaH(x_{ek}) \,

                 x_{ik}       &=   J_N \, S_{ek} - S_{ik} + W_iI_o + \lambdaGJ_N \mathbf\Gamma(S_{ik}, S_{ej}, u_{kj}),\\
                 H(x_{ik})    &=  \dfrac{a_ix_{ik} - b_i}{1 - \exp(-d_i(a_ix_{ik} -b_i))},\\
                 \dot{S}_{ik} &= -\dfrac{S_{ik}}{\tau_i} + \gamma_iH(x_{ik}) \,

    """
    _ui_name = "Reduced Wong-Wang"
    ui_configurable_parameters = ['a_e', 'b_e', 'd_e', 'gamma_e', 'tau_e', 'W_e', 'w_p', 'J_N', "R_e",
                                  'a_i', 'b_i', 'd_i', 'gamma_i', 'tau_i', 'W_i', 'J_i', "R_i",
                                  'I_o', 'G', 'lamda']

    # Define traited attributes for this model, these represent possible kwargs.

    R_e = NArray(
        label=":math:`R_e`",
        default=numpy.array([-1., ]),
        domain=Range(lo=-1., hi=10000., step=1.),
        doc="[Hz]. Excitatory population firing rate.")

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
        domain=Range(lo=10., hi=150., step=1.),
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

    R_i = NArray(
        label=":math:`R_i`",
        default=numpy.array([-1., ]),
        domain=Range(lo=-1., hi=10000., step=1.),
        doc="[Hz]. Inhibitory population firing rate.")

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

    # r_o = NArray(
    #     label=":math:`r_o`",
    #     default=numpy.array([0., ]),
    #     domain=Range(lo=0., hi=10000., step=1.),
    #     doc="[Hz]. Excitatory population output firing rate.")

    G = NArray(
        label=":math:`G`",
        default=numpy.array([2.0, ]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Global coupling scaling""")

    lamda = NArray(
        label=":math:`\lambda`",
        default=numpy.array([1.0, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Inhibitory global coupling scaling""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_boundaries = Final(
        default={"S_e": numpy.array([0.0, 1.0]),
                 "S_i": numpy.array([0.0, 1.0]),
                 "R_e": numpy.array([0.0, None]),
                 "R_i": numpy.array([0.0, None])},
        label="State Variable boundaries [lo, hi]",
        doc="""The values for each state-variable should be set to encompass
            the boundaries of the dynamic range of that state-variable. 
            Set None for one-sided boundaries""")

    state_variable_range = Final(
        default={"S_e": numpy.array([0.0, 1.0]),
                 "S_i": numpy.array([0.0, 1.0]),
                 "R_e": numpy.array([0.0, 1000.0]),
                 "R_i": numpy.array([0.0, 1000.0])
                 },
        label="State variable ranges [lo, hi]",
        doc="Population firing rate")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('S_e', 'S_i', 'R_e', 'R_i'),
        default=('S_e', 'S_i', 'R_e', 'R_i'),
        doc="""default state variables to be monitored""")

    state_variables = ['S_e', 'S_i', 'R_e', 'R_i']
    _nvar = 4
    cvar = numpy.array([0], dtype=numpy.int32)

    def configure(self):
        """  """
        super(ReducedWongWangExcIOInhI, self).configure()
        self.update_derived_parameters()

    def update_non_state_variables(self, state_variables, coupling, local_coupling=0.0, use_numba=True):
        if use_numba:
            _numba_update_non_state_variables(state_variables.reshape(state_variables.shape[:-1]).T,
                                              coupling.reshape(coupling.shape[:-1]).T+local_coupling*state_variables[0],
                                              self.a_e, self.b_e, self.d_e,
                                              self.w_p, self.W_e, self.J_N, self.R_e,
                                              self.a_i, self.b_i, self.d_i,
                                              self.W_i, self.J_i, self.R_i,
                                              self.G, self.lamda, self.I_o)
            return state_variables

        # In this case, rates (H_e, H_i) are non-state variables,
        # i.e., they form part of state_variables but have no dynamics assigned on them
        # Most of the computations of this dfun aim at computing rates, including coupling considerations.
        # Therefore, we compute and update them only once a new state is computed,
        # and we consider them constant for any subsequent possible call to this function,
        # by any integration scheme

        S = state_variables[:2, :]  # synaptic gating dynamics
        R = state_variables[2:, :]  # rates

        c_0 = coupling[0, :]

        # if applicable
        lc_0 = local_coupling * S[0]

        coupling = self.G * self.J_N * (c_0 + lc_0)

        J_N_S_e = self.J_N * S[0]

        # TODO: Confirm that this computation is correct for this model depending on the r_e and r_i values!
        x_e = self.w_p * J_N_S_e - self.J_i * S[1] + self.W_e * self.I_o + coupling

        x_e = self.a_e * x_e - self.b_e
        # Only rates with r_e < 0 will be updated by TVB.
        H_e = numpy.where(self.R_e >= 0, R[0], x_e / (1 - numpy.exp(-self.d_e * x_e)))

        x_i = J_N_S_e - S[1] + self.W_i * self.I_o + self.lamda * coupling

        x_i = self.a_i * x_i - self.b_i
        # Only rates with r_i < 0 will be updated by TVB.
        H_i = numpy.where(self.R_i >= 0, R[1], x_i / (1 - numpy.exp(-self.d_i * x_i)))

        # We now update the state_variable vector with the new rates:
        state_variables[2, :] = H_e
        state_variables[3, :] = H_i

        return state_variables

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0, update_non_state_variables=True):
        r"""
        Equations taken from [DPA_2013]_ , page 11242

        .. math::
                 x_{ek}       &=   w_p\,J_N \, S_{ek} - J_iS_{ik} + W_eI_o + GJ_N \mathbf\Gamma(S_{ek}, S_{ej}, u_{kj}),\\
                 H(x_{ek})    &=  \dfrac{a_ex_{ek}- b_e}{1 - \exp(-d_e(a_ex_{ek} -b_e))},\\
                 \dot{S}_{ek} &= -\dfrac{S_{ek}}{\tau_e} + (1 - S_{ek}) \, \gammaH(x_{ek}) \,

                 x_{ik}       &=   J_N \, S_{ek} - S_{ik} + W_iI_o + \lambdaGJ_N \mathbf\Gamma(S_{ik}, S_{ej}, u_{kj}),\\
                 H(x_{ik})    &=  \dfrac{a_ix_{ik} - b_i}{1 - \exp(-d_i(a_ix_{ik} -b_i))},\\
                 \dot{S}_{ik} &= -\dfrac{S_{ik}}{\tau_i} + \gamma_iH(x_{ik}) \,

        """

        if update_non_state_variables:
            state_variables = \
                self.update_non_state_variables(state_variables, coupling, local_coupling, use_numba=False)

        S = state_variables[:2, :]  # synaptic gating dynamics
        R = state_variables[2:, :]  # rates

        dS_e = - (S[0] / self.tau_e) + (1 - S[0]) * R[0] * self.gamma_e
        dS_i = - (S[1] / self.tau_i) + R[1] * self.gamma_i

        # Rates are non-state variables:
        dummy = 0.0*dS_e
        derivative = numpy.array([dS_e, dS_i, dummy, dummy])

        return derivative

    def dfun(self, x, c, local_coupling=0.0, update_non_state_variables=True):
        if update_non_state_variables:
            self.update_non_state_variables(x, c, local_coupling, use_numba=True)
        deriv = _numba_dfun(x.reshape(x.shape[:-1]).T, self.gamma_e, self.tau_e, self.gamma_i, self.tau_i)
        return deriv.T[..., numpy.newaxis]

