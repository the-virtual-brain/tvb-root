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
from tvb.simulator.models.wong_wang import ReducedWongWang as TVBReducedWongWang
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
        if S[1] < 40.0:  # Rint
            # For low activity
            R = numpy.sqrt(S[1])
        else:
            # For high activity
            R = 0.0000050 * g[0] * S[1] ** 2
        newS[2] = R
        newS[3] = S[3]  # Rin


@guvectorize([(float64[:],)*8], '(n),(m),(k)' + ',()'*4 + '->(n)', nopython=True)
def _numba_dfun(S, r, rin, g, t, rmask, tr, dx):
    "Gufunc for reduced Wong-Wang model equations."
    if rmask[0] > 0.0:
        # Integrate rate from Spiking Network
        # Rint
        dx[1] = (- S[1] + rin[0]) / tr[0]
    else:
        # TVB computation
        dx[1] = 0.0
    dx[0] = - (S[0] / t[0]) + (1.0 - S[0]) * r[0] * g[0]   # S


class ReducedWongWangExcIO(TVBReducedWongWang):

    r"""
    .. [WW_2006] Kong-Fatt Wong and Xiao-Jing Wang,  *A Recurrent Network
                Mechanism of Time Integration in Perceptual Decisions*.
                Journal of Neuroscience 26(4), 1314-1328, 2006.

    .. [DPA_2013] Deco Gustavo, Ponce Alvarez Adrian, Dante Mantini,
                  Gian Luca Romani, Patric Hagmann, and Maurizio Corbetta.
                  *Resting-State Functional Connectivity Emerges from
                  Structurally and Dynamically Shaped Slow Linear Fluctuations*.
                  The Journal of Neuroscience 33(27), 11239 –11252, 2013.



    .. automethod:: ReducedWongWangExcIO.__init__

    Equations taken from [DPA_2013]_ , page 11242

    .. math::
                 x_{ek}       &=   w_p\,J_N \, S_{ek} - J_iS_{ik} + W_eI_o + GJ_N \mathbf\Gamma(S_{ek}, S_{ej}, u_{kj}),\\
                 H(x_{ek})    &=  \dfrac{a_ex_{ek}- b_e}{1 - \exp(-d_e(a_ex_{ek} -b_e))},\\
                 \dot{S}_{ek} &= -\dfrac{S_{ek}}{\tau_e} + (1 - S_{ek}) \, \gammaH(x_{ek}) \,

    """

    # Define traited attributes for this model, these represent possible kwargs.

    a = NArray(
        label=":math:`a`",
        default=numpy.array([270., ]),
        domain=Range(lo=0.0, hi=0.270, step=0.01),
        doc="[n/C]. Input gain parameter, chosen to fit numerical solutions.")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([108., ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="[Hz]. Input shift parameter chosen to fit numerical solutions.")

    d = NArray(
        label=":math:`d`",
        default=numpy.array([0.154, ]),
        domain=Range(lo=0.0, hi=0.200, step=0.001),
        doc="""[s]. Parameter chosen to fit numerical solutions.""")

    gamma = NArray(
        label=r":math:`\gamma`",
        default=numpy.array([0.641/1000, ]),
        domain=Range(lo=0.0, hi=1.0/1000, step=0.01/1000),
        doc="""Kinetic parameter""")

    tau_s = NArray(
        label=r":math:`\tau_S`",
        default=numpy.array([100., ]),
        domain=Range(lo=1.0, hi=150.0, step=1.0),
        doc="""[ms]. NMDA decay time constant.""")

    w = NArray(
        label=r":math:`w`",
        default=numpy.array([0.9, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Excitatory recurrence""")

    J_N = NArray(
        label=r":math:`J_{N}`",
        default=numpy.array([0.2609, ]),
        domain=Range(lo=0.0000, hi=0.5, step=0.0001),
        doc="""Excitatory recurrence""")

    I_o = NArray(
        label=":math:`I_{o}`",
        default=numpy.array([0.3, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""[nA] Effective external input""")

    G = NArray(
        label=":math:`G`",
        default=numpy.array([2.0, ]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Global coupling scaling""")

    sigma_noise = NArray(
        label=r":math:`\sigma_{noise}`",
        default=numpy.array([0.000000001, ]),
        domain=Range(lo=0.0, hi=0.005, step=0.0001),
        doc="""[nA] Noise amplitude. Take this value into account for stochatic
            integration schemes.""")

    tau_rin = NArray(
        label=r":math:`\tau_rin_e`",
        default=numpy.array([100., ]),
        domain=Range(lo=1., hi=100., step=1.0),
        doc="""[ms]. Excitatory population instant spiking rate time constant.""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_boundaries = Final(
        default={"S": numpy.array([0.0, 1.0]),
                 "Rint": numpy.array([0.0, None]),
                 "R": numpy.array([0.0, None]),
                 "Rin": numpy.array([0.0, None]),
                 "I": numpy.array([None, None])},
        label="State Variable boundaries [lo, hi]",
        doc="""The values for each state-variable should be set to encompass
            the boundaries of the dynamic range of that state-variable. 
            Set None for one-sided boundaries""")

    state_variable_range = Final(
        default={"S": numpy.array([0.0, 1.0]),
                 "Rint": numpy.array([0.0, 1000.0]),
                 "R": numpy.array([0.0, 1000.0]),
                 "Rin": numpy.array([0.0, 1000.0]),
                 "I": numpy.array([0.0, 2.0])},
        label="State variable ranges [lo, hi]",
        doc="Population firing rate")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('S', 'Rint', 'R', 'Rin', 'I'),
        default=('S', 'Rint', 'R', 'Rin', 'I'),
        doc="""default state variables to be monitored""")

    state_variables = ['S', 'Rint', 'R', 'Rin', 'I']
    non_integrated_variables = ['R', 'Rin', 'I']
    _nvar = 5
    cvar = numpy.array([0], dtype=numpy.int32)
    _R = None
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
        self._R = None
        self._Rin = None
        self._stimulus = 0.0
        if hasattr(self, "Rin"):
            setattr(self, "_Rin_mask", getattr(self, "Rin") > 0)
        else:
            setattr(self, "Rin", numpy.array([0.0, ]))
            setattr(self, "_Rin_mask", numpy.array([False, ]))

    def update_state_variables_before_integration(self, state_variables, coupling, local_coupling=0.0, stimulus=0.0):
        self._stimulus = stimulus
        if self.use_numba:
            state_variables = \
                _numba_update_non_state_variables_before_integration(
                    state_variables.reshape(state_variables.shape[:-1]).T,
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
        R = numpy.where(self._Rin_mask,
                        # Downscale rates coming from the Spiking Network
                        numpy.where(Rint < 40,
                                    numpy.sqrt(Rint),      # Low activity scaling
                                    0.0000050 *self.G * Rint ** 2),  # High activity scaling,
                        x / (1 - numpy.exp(-self.d * x)))

        Rin = numpy.where(self._Rin_mask, Rin, 0.0)  # Reset to 0 the Rin for nodes not updated by Spiking Network

        # We now update the state_variable vector with the new rates and currents:
        state_variables[2, :] = R
        state_variables[3, :] = Rin
        state_variables[4, :] = I

        # Keep them here so that they are not recomputed in the dfun
        self._R = numpy.copy(R)
        self._Rin = numpy.copy(Rin)

        return state_variables

    def _integration_to_state_variables(self, integration_variables):
        return numpy.array(integration_variables.tolist() + [0.0*integration_variables[0]] * self.n_nonintvar)

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
        dS = - (S / self.tau_s) + (1 - S) * R * self.gamma

        # Rates
        # Low pass filtering, linear dynamics for rates updated from the spiking network
        # No dynamics in the case of TVB rates
        dRint = numpy.where(self._Rin_mask, (- Rint + Rin) / self.tau_rin, 0.0)

        return numpy.array([dS, dRint])

    def dfun(self, x, c, local_coupling=0.0):
        if self._R is None or self._Rin is None:
            state_variables = self._integration_to_state_variables(x)
            state_variables = \
                self.update_state_variables_before_integration(state_variables, c, local_coupling, self._stimulus)
            R = state_variables[2]  # Rates
            Rin = state_variables[3]  # input instant spiking rates
        else:
            R = self._R
            Rin = self._Rin
        if self.use_numba:
            deriv = _numba_dfun(x.reshape(x.shape[:-1]).T, R, Rin,
                                self.gamma, self.tau_s, self.Rin, self.tau_rin)
            deriv = deriv.T[..., numpy.newaxis]
        else:
            deriv = self._numpy_dfun(x, R, Rin)
        #  Set them to None so that they are recomputed on subsequent steps
        #  for multistep integration schemes such as Runge-Kutta:
        self._R = None
        self._Rin = None
        return deriv

