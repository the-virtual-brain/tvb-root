# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
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

"""
Models based on Wong-Wang's work.
.. moduleauthor:: Dionysios Perdikis <dionysios.perdikis@charite.de>
"""

from numba import guvectorize, float64
from tvb.simulator.models.wong_wang_exc_inh import ReducedWongWangExcInh as TVBReducedWongWangExcInh
from tvb.simulator.models.base import numpy, ModelNumbaDfun
from tvb.basic.neotraits.api import NArray, Final, List, Range


@guvectorize([(float64[:],)*20], '(n),(m)' + ',()'*17 + '->(n)', nopython=True)
def _numba_update_non_state_variables_before_integration(S, c, ae, be, de, wp, we, jn, re,
                                                         ai, bi, di, wi, ji, ri, g, l, io, ie, newS):
    "Gufunc for reduced Wong-Wang model equations."

    newS[0] = S[0]  # S_e
    newS[1] = S[1]  # S_i

    cc = g[0]*jn[0]*c[0]

    jnSe = jn[0] * S[0]
                                    # S_i
    newS[6] = wp[0] * jnSe - ji[0] * S[1] + we[0] * io[0] + cc + ie[0]  # I_e
    if re[0] <= 0.0:
        # TVB computation
        x = ae[0]*newS[6] - be[0]
        h = x / (1 - numpy.exp(-de[0]*x))
        newS[2] = h     # R_e
        newS[4] = 0.0   # Rin_e
    else:
        # Updating from spiking network
        newS[2] = S[2]  # R_e
        newS[4] = S[4]  # Rin_e

                    # S_i
    newS[7] = jnSe - S[1] + wi[0] * io[0] + l[0] * cc   # I_i
    if ri[0] <= 0.0:
        # TVB computation
        x = ai[0]*newS[7] - bi[0]
        h = x / (1 - numpy.exp(-di[0]*x))
        newS[3] = h     # R_i
        newS[5] = 0.0   # Rin_i
    else:
        newS[3] = S[3]  # R_i
        newS[5] = S[5]  # Rin_i


@guvectorize([(float64[:],)*12], '(n),(m),(k)' + ',()'*8 + '->(n)', nopython=True)
def _numba_dfun(S, rine, rini, ge, te, re_mask, tre, gi, ti, ri_mask,  tri, dx):
    "Gufunc for reduced Wong-Wang model equations."

    dx[0] = - (S[0] / te[0]) + (1.0 - S[0]) * S[2] * ge[0]
    if re_mask[0] > 0.0:
        # Integrating input from spiking network
        dx[2] = (- S[2] + rine[0]) / tre[0]
    else:
        # TVB computation
        dx[2] = 0.0

    dx[1] = - (S[1] / ti[0]) + S[3] * gi[0]
    if ri_mask[0] > 0.0:
        # Integrating input from spiking network
        dx[3] = (- S[3] + rini[0]) / tri[0]
    else:
        # TVB computation
        dx[3] = 0.0


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

    tau_rin_e = NArray(
        label=r":math:`\tau_rin_e`",
        default=numpy.array([1., ]),
        domain=Range(lo=1., hi=100., step=1.0),
        doc="""[ms]. Excitatory population instant spiking rate time constant.""")

    tau_rin_i = NArray(
        label=r":math:`\tau_rin_i`",
        default=numpy.array([1., ]),
        domain=Range(lo=1., hi=100., step=1.0),
        doc="""[ms]. Inhibitory population instant spiking rate time constant.""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_boundaries = Final(
        default={"S_e": numpy.array([0.0, 1.0]),
                 "S_i": numpy.array([0.0, 1.0]),
                 "R_e": numpy.array([0.0, None]),
                 "R_i": numpy.array([0.0, None]),
                 "Rin_e": numpy.array([0.0, None]),
                 "Rin_i": numpy.array([0.0, None]),
                 "I_e": numpy.array([None, None]),
                 "I_i": numpy.array([None, None])},
        label="State Variable boundaries [lo, hi]",
        doc="""The values for each state-variable should be set to encompass
            the boundaries of the dynamic range of that state-variable. 
            Set None for one-sided boundaries""")

    state_variable_range = Final(
        default={"S_e": numpy.array([0.0, 1.0]),
                 "S_i": numpy.array([0.0, 1.0]),
                 "R_e": numpy.array([0.0, 1000.0]),
                 "R_i": numpy.array([0.0, 1000.0]),
                 "Rin_e": numpy.array([0.0, 1000.0]),
                 "Rin_i": numpy.array([0.0, 1000.0]),
                 "I_e": numpy.array([0.0, 2.0]),
                 "I_i": numpy.array([0.0, 1.0])
                 },
        label="State variable ranges [lo, hi]",
        doc="Population firing rate")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('S_e', 'S_i', 'R_e', 'R_i', 'Rin_e', 'Rin_i', 'I_e', 'I_i'),
        default=('S_e', 'S_i', 'R_e', 'R_i', 'Rin_e', 'Rin_i', 'I_e', 'I_i'),
        doc="""default state variables to be monitored""")

    state_variables = ['S_e', 'S_i', 'R_e', 'R_i', 'Rin_e', 'Rin_i', 'I_e', 'I_i']
    non_integrated_variables = ['Rin_e', 'Rin_i', 'I_e', 'I_i']
    _nvar = 8
    cvar = numpy.array([0], dtype=numpy.int32)
    _Rin = None
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
        self._Rin = None
        self._stimulus = 0.0
        for var in ["Rin_e", "Rin_i"]:
            if hasattr(self, var):
                setattr(self, "_%s_mask" % var, getattr(self, var) > 0)
            else:
                setattr(self, var, numpy.array([0.0, ]))
                setattr(self, "_%s_mask" % var, numpy.array([False, ]))

    def update_state_variables_before_integration(self, state_variables, coupling, local_coupling=0.0, stimulus=0.0):
        self._stimulus = stimulus
        if self.use_numba:
            state_variables = \
                _numba_update_non_state_variables_before_integration(
                    state_variables.reshape(state_variables.shape[:-1]).T,
                    coupling.reshape(coupling.shape[:-1]).T +
                    local_coupling * state_variables[0],
                    self.a_e, self.b_e, self.d_e,
                    self.w_p, self.W_e, self.J_N, self.Rin_e,
                    self.a_i, self.b_i, self.d_i,
                    self.W_i, self.J_i, self.Rin_i,
                    self.G, self.lamda, self.I_o, self.I_ext)
            return state_variables.T[..., numpy.newaxis]

        # In this case, rates (H_e, H_i) are non-state variables,
        # i.e., they form part of state_variables but have no dynamics assigned on them
        # Most of the computations of this dfun aim at computing rates, including coupling considerations.
        # Therefore, we compute and update them only once a new state is computed,
        # and we consider them constant for any subsequent possible call to this function,
        # by any integration scheme

        S = state_variables[:2, :]  # synaptic gating dynamics
        R = state_variables[2:4, :]  # Rates
        Rin = state_variables[4:6, :]  # Input rates from spiking network

        c_0 = coupling[0, :]

        # if applicable
        lc_0 = local_coupling * S[0]

        coupling = self.G * self.J_N * (c_0 + lc_0)

        J_N_S_e = self.J_N * S[0]

        # TODO: Confirm that this computation is correct for this model depending on the r_e and r_i values!
        I_e = self.w_p * J_N_S_e - self.J_i * S[1] + self.W_e * self.I_o + coupling + self.I_ext

        x_e = self.a_e * I_e - self.b_e
        # Only rates with R_e <= 0 0 will be updated by TVB.
        R_e = numpy.where(self._Rin_e_mask, R[0], x_e / (1 - numpy.exp(-self.d_e * x_e)))
        # ...and their Rin_e should be zero:
        Rin_e = numpy.where(self._Rin_e_mask, Rin[0], 0.0)

        I_i = J_N_S_e - S[1] + self.W_i * self.I_o + self.lamda * coupling

        x_i = self.a_i * I_i - self.b_i
        # Only rates with R_i < 0 will be updated by TVB.
        R_i = numpy.where(self._Rin_i_mask, R[1], x_i / (1 - numpy.exp(-self.d_i * x_i)))
        # ...and their Rin_i should be zero:
        Rin_i = numpy.where(self._Rin_i_mask, Rin[1], 0.0)

        # We now update the state_variable vector with the new rates:
        state_variables[2, :] = R_e
        state_variables[3, :] = R_i
        state_variables[4, :] = Rin_e
        state_variables[5, :] = Rin_i
        state_variables[6, :] = I_e
        state_variables[7, :] = I_i

        # Keep them here so that they are not recomputed in the dfun
        self._Rin = numpy.copy(state_variables[4:6])

        return state_variables

    def _integration_to_state_variables(self, integration_variables):
        return numpy.array(integration_variables.tolist() + [0.0*integration_variables[0]] * self.n_nonintvar)

    def _numpy_dfun(self, integration_variables, Rin):
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
        R = integration_variables[2:4, :]  # Rates

        # Synaptic gating dynamics
        dS_e = - (S[0] / self.tau_e) + (1 - S[0]) * R[0] * self.gamma_e
        dS_i = - (S[1] / self.tau_i) + R[1] * self.gamma_i

        # Rates
        # Low pass filtering, linear dynamics for variables updated from the spiking network
        # No dynamics in the case of TVB rates
        dR_e = numpy.where(self._Rin_e_mask, (- R[0] + Rin[0]) / self.tau_rin_e, 0.0)
        dR_i = numpy.where(self._Rin_i_mask, (- R[1] + Rin[1]) / self.tau_rin_i, 0.0)

        return numpy.array([dS_e, dS_i, dR_e, dR_i])

    def dfun(self, x, c, local_coupling=0.0):
        if self._Rin is None:
            state_variables = self._integration_to_state_variables(x)
            state_variables = \
                self.update_state_variables_before_integration(state_variables, c, local_coupling, self._stimulus)
            x[2:4] = state_variables[2:4]  # Rates
            Rin = state_variables[4:6]  # Exc input instant spiking rates
        else:
            Rin = self._Rin
        if self.use_numba:
            deriv = _numba_dfun(x.reshape(x.shape[:-1]).T, Rin[0], Rin[1],
                                self.gamma_e, self.tau_e, self.Rin_e,  self.tau_rin_e,
                                self.gamma_i, self.tau_i, self.Rin_i, self.tau_rin_i).T[..., numpy.newaxis]
        else:
            deriv = self._numpy_dfun(x, Rin)
        #  Set them to None so that they are recomputed on subsequent steps
        #  for multistep integration schemes such as Runge-Kutta:
        self._Rin = None
        return deriv

