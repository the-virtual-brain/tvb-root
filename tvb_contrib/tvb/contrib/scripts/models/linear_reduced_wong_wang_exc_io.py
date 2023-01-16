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
from tvb.simulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO
from tvb.simulator.models.base import numpy, ModelNumbaDfun
from tvb.basic.neotraits.api import NArray, Final, List, Range


@guvectorize([(float64[:],)*11], '(n),(m)' + ',()'*8 + '->(n)', nopython=True)
def _numba_update_non_state_variables_before_integration(S, c, a, b, d, w, jn, r, g, io, newS):
    "Gufunc for reduced Wong-Wang model equations."

    newS[0] = S[0]  # S
    newS[1] = S[1]  # Rint

    cc = g[0]*jn[0]*c[0]
    newS[4] = w[0] * jn[0] * S[0] + io[0] + cc  # I
    if r[0] <= 0.0:
        # R computed in TVB
        x = a[0]*newS[4] - b[0]
        h = x / (1 - numpy.exp(-d[0]*x))
        newS[2] = h     # R
        newS[3] = 0.0   # Rin
    else:
        # R updated from Spiking Network model
        # Rate has to be scaled down from the Spiking neural model rate range to the TVB one
        newS[2] = S[1]  # R = Rint
        newS[3] = S[3]  # Rin


@guvectorize([(float64[:],)*6], '(n),(m),(k)' + ',()'*4 + '->(n)', nopython=True)
def _numba_dfun(S, r, rin, g, t, r_mask, tr, dx):
    "Gufunc for reduced Wong-Wang model equations."
    if r_mask[0] > 0.0:
        # Integrate rate from Spiking Network
        # Rint
        dx[1] = (- S[1] + rin[0]) / tr[0]
    else:
        # TVB computation
        dx[1] = 0.0
    dx[0] = - (S[0] / t[0]) + r[0] * g[0]   # S


class LinearReducedWongWangExcIO(ReducedWongWangExcIO):

    d = NArray(
        label=":math:`d`",
        default=numpy.array([0.2, ]),
        domain=Range(lo=0.0, hi=0.200, step=0.001),
        doc="""[s]. Parameter chosen to fit numerical solutions.""")

    non_integrated_variables = ["R", "Rin", "I"]

    def update_state_variables_before_integration(self, state_variables, coupling, local_coupling=0.0, stimulus=0.0):
        if self.use_numba:
            state_variables = \
                _numba_update_non_state_variables(state_variables.reshape(state_variables.shape[:-1]).T,
                                                  coupling.reshape(coupling.shape[:-1]).T +
                                                  local_coupling * state_variables[0],
                                                  self.a, self.b, self.d,
                                                  self.w, self.J_N, self.Rin,
                                                  self.G, self.I_o)
            return state_variables.T[..., numpy.newaxis]

        # In this case, rates (H_e, H_i) are non-state variables,
        # i.e., they form part of state_variables but have no dynamics assigned on them
        # Most of the computations of this dfun aim at computing rates, including coupling considerations.
        # Therefore, we compute and update them only once a new state is computed,
        # and we consider them constant for any subsequent possible call to this function,
        # by any integration scheme

        S = state_variables[0, :]  # synaptic gating dynamics
        Rint = state_variables[1, :]  # Rates from Spiking Network, integrated
        Rin = state_variables[3, :]  # Input rates from Spiking Network

        c_0 = coupling[0, :]

        # if applicable
        lc_0 = local_coupling * S[0]

        coupling = self.G * self.J_N * (c_0 + lc_0)

        # Currents
        I = self.w * self.J_N * S + self.I_o + coupling
        x = self.a * I - self.b

        # Rates
        # Only rates with _Rin <= 0 0 will be updated by TVB.
        # The rest, are updated from the Spiking Network
        R = numpy.where(self._Rin, Rint, x / (1 - numpy.exp(-self.d * x)))

        Rin = numpy.where(self._Rin, Rin, 0.0)  # Reset to 0 the Rin for nodes not updated by Spiking Network

        # We now update the state_variable vector with the new rates and currents:
        state_variables[2, :] = R
        state_variables[3, :] = Rin
        state_variables[4, :] = I

        # Keep them here so that they are not recomputed in the dfun
        self._R = numpy.copy(R)
        self._Rin = numpy.copy(Rin)

        return state_variables

    def _numpy_dfun(self, integration_variables, R, Rin):
        r"""
        Equations taken from [DPA_2013]_ , page 11242

        .. math::
                  x_k       &=   w\,J_N \, S_k + I_o + J_N \mathbf\Gamma(S_k, S_j, u_{kj}),\\
                 H(x_k)    &=  \dfrac{ax_k - b}{1 - \exp(-d(ax_k -b))},\\
                 \dot{S}_k &= -\dfrac{S_k}{\tau_s} + (1 - S_k) \, H(x_k) \, \gamma

        """

        S = integration_variables[0, :]  # Synaptic gating dynamics
        Rint = integration_variables[1, :]  # Rates from Spiking Network, integrated

        # Synaptic gating dynamics
        dS = - (S / self.tau_s) + R * self.gamma

        # Rates
        # Low pass filtering, linear dynamics for rates updated from the spiking network
        # No dynamics in the case of TVB rates
        dRint = numpy.where(self._Rin_mask, (- Rint + Rin) / self.tau_rin, 0.0)

        return numpy.array([dS, dRint])

    def dfun(self, x, c, local_coupling=0.0):
        if self._R is None or self._Rin is None:
            state_variables = self._integration_to_state_variables(x)
            state_variables = \
                self.update_state_variables_before_integration(state_variables, c, local_coupling,
                                                               self._stimulus)
            R = state_variables[2]  # Rates
            Rin = state_variables[3]  # input instant spiking rates
        else:
            R = self._R
            Rin = self._Rin
        if self.use_numba:
            deriv = _numba_dfun(x.reshape(x.shape[:-1]).T, R, Rin,
                                self.gamma, self.tau_s, self.Rin,self.tau_rin).T[..., numpy.newaxis]
        else:
            deriv = self._numpy_dfun(x, R, Rin)
        #  Set them to None so that they are recomputed on subsequent steps
        #  for multistep integration schemes such as Runge-Kutta:
        self._R = None
        self._Rin = None
        return deriv

