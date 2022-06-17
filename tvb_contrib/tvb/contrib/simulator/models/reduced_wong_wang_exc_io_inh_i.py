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
.. moduleauthor:: Dionysios Perdikis <dionysios.perdikis@charite.de>


"""

from numba import guvectorize, float64
from tvb.simulator.models.wong_wang_exc_inh import ReducedWongWangExcInh as TVBReducedWongWangExcInh
from tvb.simulator.models.base import numpy, ModelNumbaDfun
from tvb.basic.neotraits.api import NArray, Final, List, Range


@guvectorize([(float64[:],)*18], '(n),(m)' + ',()'*15 + '->(n)', nopython=True)
def _numba_update_non_state_variables_before_integration(S, c,
                                                         ae, be, de, wp, we, jn,
                                                         ai, bi, di, wi, ji,
                                                         g, l, io, ie, newS):
    "Gufunc for reduced Wong-Wang model equations."

    newS[0] = S[0]  # S_e
    newS[1] = S[1]  # S_i

    cc = g[0]*jn[0]*c[0]

    jnSe = jn[0] * S[0]

    I_e = wp[0] * jnSe - ji[0] * S[1] + we[0] * io[0] + cc + ie[0]  # I_e
    x = ae[0]*I_e - be[0]
    h = x / (1 - numpy.exp(-de[0]*x))
    newS[2] = h  # R_e

    I_i = jnSe - S[1] + wi[0] * io[0] + l[0] * cc   # I_i
    x = ai[0]*I_i - bi[0]
    h = x / (1 - numpy.exp(-di[0]*x))
    newS[3] = h  # R_i


@guvectorize([(float64[:],)*7], '(n),(m)' + ',()'*4 + '->(n)', nopython=True)
def _numba_dfun(S, R, ge, te, gi, ti, dx):
    "Gufunc for reduced Wong-Wang model equations."
    dx[0] = - (S[0] / te[0]) + (1.0 - S[0]) * R[0] * ge[0]
    dx[1] = - (S[1] / ti[0]) + R[1] * gi[0]


class ReducedWongWangExcIOInhI(TVBReducedWongWangExcInh):

    r"""
    .. [WW_2006] Kong-Fatt Wong and Xiao-Jing Wang,  *A Recurrent Network
                Mechanism of Time Integration in Perceptual Decisions*.
                Journal of Neuroscience 26(4), 1314-1328, 2006.

    .. [DPA_2014] Deco Gustavo, Ponce Alvarez Adrian, Patric Hagmann,
                  Gian Luca Romani, Dante Mantini, and Maurizio Corbetta. *How Local
                  Excitation–Inhibition Ratio Impacts the Whole Brain Dynamics*.
                  The Journal of Neuroscience 34(23), 7886 –7898, 2014.



    .. automethod:: ReducedWongWangExcIOInhI.__init__

    Equations taken from [DPA_2014]_

    .. math::
                 x_{ek}       &=   w_p\,J_N \, S_{ek} - J_iS_{ik} + W_eI_o + GJ_N \mathbf\Gamma(S_{ek}, S_{ej}, u_{kj}),\\
                 H(x_{ek})    &=  \dfrac{a_ex_{ek}- b_e}{1 - \exp(-d_e(a_ex_{ek} -b_e))},\\
                 \dot{S}_{ek} &= -\dfrac{S_{ek}}{\tau_e} + (1 - S_{ek}) \, \gammaH(x_{ek}) \,

                 x_{ik}       &=   J_N \, S_{ek} - S_{ik} + W_iI_o + \lambdaGJ_N \mathbf\Gamma(S_{ik}, S_{ej}, u_{kj}),\\
                 H(x_{ik})    &=  \dfrac{a_ix_{ik} - b_i}{1 - \exp(-d_i(a_ix_{ik} -b_i))},\\
                 \dot{S}_{ik} &= -\dfrac{S_{ik}}{\tau_i} + \gamma_iH(x_{ik}) \,

    """

    # Define traited attributes for this model, these represent possible kwargs.

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
                 "R_i": numpy.array([0.0, 1000.0])},
        label="State variable ranges [lo, hi]",
        doc="Population firing rate")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('S_e', 'S_i', 'R_e', 'R_i'),
        default=('S_e', 'S_i', 'R_e', 'R_i'),
        doc="""default state variables to be monitored""")

    state_variables = ['S_e', 'S_i', 'R_e', 'R_i']
    non_integrated_variables = ['R_e', 'R_i']
    _nvar = 4
    cvar = numpy.array([0, 1, 2, 3], dtype=numpy.int32)
    _R = None
    _stimulus = 0.0
    use_numba = True

    def update_derived_parameters(self):
        """
        When needed, this should be a method for calculating parameters that are
        calculated based on paramaters directly set by the caller. For example,
        see, ReducedSetFitzHughNagumo. When not needed, this pass simplifies
        code that updates an arbitrary models parameters -- ie, this can be
        safely called on any model, whether it's used or not.
        """
        self.n_nonintvar = self.nvar - self.nintvar
        self._R = None
        self._stimulus = 0.0

    def update_state_variables_before_integration(self, state_variables, coupling, local_coupling=0.0, stimulus=0.0):
        self._stimulus = stimulus
        if self.use_numba:
            state_variables = \
                _numba_update_non_state_variables_before_integration(
                    state_variables.reshape(state_variables.shape[:-1]).T,
                    coupling.reshape(coupling.shape[:-1]).T +
                    local_coupling * state_variables[0],
                    self.a_e, self.b_e, self.d_e,  self.w_p, self.W_e, self.J_N,
                    self.a_i, self.b_i, self.d_i, self.W_i, self.J_i,
                    self.G, self.lamda, self.I_o, self.I_ext)
            return state_variables.T[..., numpy.newaxis]

        # In this case, rates (H_e, H_i) are non-state variables,
        # i.e., they form part of state_variables but have no dynamics assigned on them
        # Most of the computations of this dfun aim at computing rates, including coupling considerations.
        # Therefore, we compute and update them only once a new state is computed,
        # and we consider them constant for any subsequent possible call to this function,
        # by any integration scheme

        S = state_variables[:2, :]  # synaptic gating dynamics

        c_0 = coupling[0, :]

        # if applicable
        lc_0 = local_coupling * S[0]

        coupling = self.G * self.J_N * (c_0 + lc_0)

        J_N_S_e = self.J_N * S[0]

        # TODO: Confirm that this computation is correct for this model depending on the r_e and r_i values!
        I_e = self.w_p * J_N_S_e - self.J_i * S[1] + self.W_e * self.I_o + coupling + self.I_ext
        x_e = self.a_e * I_e - self.b_e
        R_e =  x_e / (1 - numpy.exp(-self.d_e * x_e))

        I_i = J_N_S_e - S[1] + self.W_i * self.I_o + self.lamda * coupling
        x_i = self.a_i * I_i - self.b_i
        R_i = x_i / (1 - numpy.exp(-self.d_i * x_i))

        # We now update the state_variable vector with the new rates:
        state_variables[2, :] = R_e
        state_variables[3, :] = R_i

        return state_variables

    def _integration_to_state_variables(self, integration_variables):
        return numpy.array(integration_variables.tolist() + [0.0*integration_variables[0]] * self.n_nonintvar)

    def _numpy_dfun(self, integration_variables, R):
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

        S = integration_variables[:2, :]     # Synaptic gating dynamics

        # Synaptic gating dynamics
        dS_e = - (S[0] / self.tau_e) + (1 - S[0]) * R[0] * self.gamma_e
        dS_i = - (S[1] / self.tau_i) + R[1] * self.gamma_i

        return numpy.array([dS_e, dS_i])

    def dfun(self, x, c, local_coupling=0.0, time=0.0):
        if self._R is None:
            state_variables = self._integration_to_state_variables(x)
            state_variables = \
                self.update_state_variables_before_integration(state_variables, c, local_coupling, self._stimulus)
            R = state_variables[2:4]  # Rates
        else:
            R = self._R
        if self.use_numba:
            deriv = _numba_dfun(x.reshape(x.shape[:-1]).T, R.reshape(x.shape[:-1]).T,
                                self.gamma_e, self.tau_e, self.gamma_i, self.tau_i).T[..., numpy.newaxis]
        else:
            deriv = self._numpy_dfun(x, R)
        #  Set them to None so that they are recomputed on subsequent steps
        #  for multistep integration schemes such as Runge-Kutta:
        self._R = None
        return deriv


@guvectorize([(float64[:],)*17], '(n),(m)' + ',()'*14 + '->(n)', nopython=True)
def _numba_update_non_state_variables_before_integration_fic(S, c,
                                                             ae, be, de, wp, we, jn,
                                                             ai, bi, di, wi,
                                                             g, l, io, ie, newS):
    "Gufunc for reduced Wong-Wang model equations."

    newS[0] = S[0]  # S_e
    newS[1] = S[1]  # S_i
    newS[2] = S[2]  # J_i

    cc = g[0]*jn[0]*c[0]

    jnSe = jn[0] * S[0]

    I_e = wp[0] * jnSe - S[2] * S[1] + we[0] * io[0] + cc + ie[0]  # I_e
    x = ae[0]*I_e - be[0]
    h = x / (1 - numpy.exp(-de[0]*x))
    newS[3] = h  # R_e

    I_i = jnSe - S[1] + wi[0] * io[0] + l[0] * cc   # I_i
    x = ai[0]*I_i - bi[0]
    h = x / (1 - numpy.exp(-di[0]*x))
    newS[4] = h  # R_i


@guvectorize([(float64[:],)*9], '(n),(m)' + ',()'*6 + '->(n)', nopython=True)
def _numba_dfun_fic(X, R, ge, te, gi, ti, eta, r_e_0, dx):
    "Gufunc for reduced Wong-Wang model equations."
    dx[0] = - (X[0] / te[0]) + (1.0 - X[0]) * R[0] * ge[0]
    dx[1] = - (X[1] / ti[0]) + R[1] * gi[0]
    dx[2] = eta[0] * X[2] * X[1] * (R[0] - r_e_0[0])  # X[2] * X[1] *  or R[1]


class ReducedWongWangExcIOInhIFIC(ReducedWongWangExcIOInhI):

    eta = NArray(
        label="eta",
        default=numpy.array([0.001, ]),
        domain=Range(lo=0.0001, hi=0.01, step=0.0001),
        doc="""Feedback Inhibition Control adaptation rate.""")

    R_e_0 = NArray(
        label="R_e_0",
        default=numpy.array([3.06, ]),
        domain=Range(lo=0.01, hi=10., step=0.01),
        doc="""Target rate of excitatory population for Feedback Inhibition Control.""")

    state_variable_range = Final(
        default={
            "S_e": numpy.array([0.0, 1.0]),
            "S_i": numpy.array([0.0, 1.0]),
            "R_e": numpy.array([0.0, 10.0]),
            "R_i": numpy.array([0.0, 10.0]),
            "J_i": numpy.array([0.0, 10.0]),
        },
        label="State variable ranges [lo, hi]",
        doc="Population firing rate")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"S_e": numpy.array([0.0, 1.0]),
                 "S_i": numpy.array([0.0, 1.0]),
                 "R_e": numpy.array([0.0, None]),
                 "R_i": numpy.array([0.0, None]),
                 "J_i": numpy.array([0.0, None])},
        doc="""The values for each state-variable should be set to encompass
                the boundaries of the dynamic range of that state-variable. Set None for one-sided boundaries""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('S_e', 'S_i', 'J_i', 'R_e', 'R_i'),
        default=('S_e', 'S_i', 'J_i', 'R_e', 'R_i'),
        doc="""default state variables to be monitored""")

    state_variables = ['S_e', 'S_i', 'J_i', 'R_e', 'R_i']
    non_integrated_variables = ['R_e', 'R_i']
    _nvar = 5
    cvar = numpy.array([0], dtype=numpy.int32)

    def update_state_variables_before_integration(self, state_variables, coupling, local_coupling=0.0, stimulus=0.0):
        self._stimulus = stimulus
        if self.use_numba:
            state_variables = \
                _numba_update_non_state_variables_before_integration_fic(
                    state_variables.reshape(state_variables.shape[:-1]).T,
                    coupling.reshape(coupling.shape[:-1]).T +
                    local_coupling * state_variables[0],
                    self.a_e, self.b_e, self.d_e,  self.w_p, self.W_e, self.J_N,
                    self.a_i, self.b_i, self.d_i, self.W_i,
                    self.G, self.lamda, self.I_o, self.I_ext)
            return state_variables.T[..., numpy.newaxis]

        # In this case, rates (H_e, H_i) are non-state variables,
        # i.e., they form part of state_variables but have no dynamics assigned on them
        # Most of the computations of this dfun aim at computing rates, including coupling considerations.
        # Therefore, we compute and update them only once a new state is computed,
        # and we consider them constant for any subsequent possible call to this function,
        # by any integration scheme

        S = state_variables[:2, :]  # synaptic gating dynamics
        J_i = state_variables[2, :]  # FIC dynamics

        c_0 = coupling[0, :]

        # if applicable
        lc_0 = local_coupling * S[0]

        coupling = self.G * self.J_N * (c_0 + lc_0)

        J_N_S_e = self.J_N * S[0]

        I_e = self.w_p * J_N_S_e - J_i * S[1] + self.W_e * self.I_o + coupling + self.I_ext
        x_e = self.a_e * I_e - self.b_e
        R_e = x_e / (1 - numpy.exp(-self.d_e * x_e))

        I_i = J_N_S_e - S[1] + self.W_i * self.I_o + self.lamda * coupling
        x_i = self.a_i * I_i - self.b_i
        R_i = x_i / (1 - numpy.exp(-self.d_i * x_i))

        # We now update the state_variable vector with the new rates:
        state_variables[3, :] = R_e
        state_variables[4, :] = R_i

        return state_variables

    def _numpy_dfun(self, integration_variables, R):
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
        S = integration_variables[:2, :]     # Synaptic gating dynamics
        J_i = integration_variables[2, :]  # FIC dynamics

        # Synaptic gating dynamics
        dS_e = - (S[0] / self.tau_e) + (1 - S[0]) * R[0] * self.gamma_e
        dS_i = - (S[1] / self.tau_i) + R[1] * self.gamma_i

        # FIC dynamics
        dJ_i = self.eta * J_i * S[1] * (R[0] - self.R_e_0)  # or R[1] instead of J_i * S[1] *

        return numpy.array([dS_e, dS_i, dJ_i])

    def dfun(self, x, c, local_coupling=0.0, time=0.0):
        if self._R is None:
            state_variables = self._integration_to_state_variables(x)
            state_variables = \
                self.update_state_variables_before_integration(state_variables, c, local_coupling, self._stimulus)
            R = state_variables[3:5]  # Rates
        else:
            R = self._R
        if self.use_numba:
            deriv = _numba_dfun_fic(x.reshape(x.shape[:-1]).T, R.reshape(R.shape[:-1]).T,
                                    self.gamma_e, self.tau_e, self.gamma_i, self.tau_i,
                                    self.eta, self.R_e_0).T[..., numpy.newaxis]
        else:
            deriv = self._numpy_dfun(x, R)
        #  Set them to None so that they are recomputed on subsequent steps
        #  for multistep integration schemes such as Runge-Kutta:
        self._R = None
        return deriv


