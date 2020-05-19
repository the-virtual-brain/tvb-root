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
from tvb.simulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb.simulator.models.spiking_wong_wang_exc_io_inh_i import SpikingWongWangExcIOInhI
from tvb.simulator.models.base import numpy, ModelNumbaDfun, Model
from tvb.basic.neotraits.api import NArray, Final, List, Range


class MultiscaleWongWangExcIOInhI(ReducedWongWangExcIOInhI, SpikingWongWangExcIOInhI):

    _spiking_regions_inds = []

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

    # Define traited attributes for this model, these represent possible kwargs.


    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_boundaries = Final(
        default={"s_AMPA": numpy.array([0., 1.]),               # 0
                 "x_NMDA": numpy.array([0., None]),             # 1
                 "s_NMDA": numpy.array([0., 1.]),               # 2
                 "s_GABA": numpy.array([0., 1.]),               # 3
                 "s_AMPA_ext": numpy.array([0., 800.]),         # 4
                 "V_m": numpy.array([None, None]),              # 5
                 "t_ref": numpy.array([0., None]),              # 6
                 "spikes_ext": numpy.array([0., None]),         # 7
                 "spikes": numpy.array([0., None]),             # 8
                 "rate": numpy.array([0., None]),               # 9
                 "I_syn": numpy.array([None, None]),            # 10
                 "I_L": numpy.array([None, None]),              # 11
                 "I_AMPA": numpy.array([None, None]),           # 12
                 "I_NMDA": numpy.array([None, None]),           # 13
                 "I_GABA": numpy.array([None, None]),           # 14
                 "I_AMPA_ext": numpy.array([None, None])},      # 15
        label="State Variable boundaries [lo, hi]",
        doc="""The values for each state-variable should be set to encompass
            the boundaries of the dynamic range of that state-variable. 
            Set None for one-sided boundaries""")

    # Choosing default initial conditions such as that there are no spikes initialy:
    state_variable_range = Final(
        default={"s_AMPA": numpy.array([0., 1.]),           # 0
                 "x_NMDA": numpy.array([0., 200.]),         # 1
                 "s_NMDA": numpy.array([0., 1.]),           # 2
                 "s_GABA": numpy.array([0., 1.]),           # 3
                 "s_AMPA_ext": numpy.array([0., 1.]),       # 4
                 "V_m": numpy.array([-70., -50.]),          # 5
                 "t_ref": numpy.array([0., 1.]),            # 6
                 "spikes_ext": numpy.array([0., 0.5]),      # 7
                 "spikes": numpy.array([0., 0.5]),          # 8
                 "rate": numpy.array([0., 1000.]),          # 9
                 "I_syn": numpy.array([0., 1000.]),         # 10
                 "I_L": numpy.array([0., 1000.]),           # 11
                 "I_AMPA": numpy.array([-1000., 0.]),       # 12
                 "I_NMDA": numpy.array([-1000., 0.]),       # 13
                 "I_GABA": numpy.array([0., 1000.]),        # 14
                 "I_AMPA_ext": numpy.array([-1000., 0.])}, # 15
        label="State variable ranges [lo, hi]",
        doc="""State variable ranges [lo, hi] for random initial condition construction""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("s_AMPA",  "x_NMDA", "s_NMDA", "s_GABA", "V_m",  "t_ref",
                 "spikes_ext",  "spikes", "rate", "I_syn", "I_L", "I_AMPA", "I_NMDA", "I_GABA", "I_AMPA_ext"),
        default=("s_AMPA",  "x_NMDA", "s_NMDA", "s_GABA", "V_m",  "t_ref",
                 "spikes_ext",  "spikes", "rate", "I_syn", "I_L", "I_AMPA", "I_NMDA", "I_GABA", "I_AMPA_ext"),
        doc="""default state variables to be monitored""")

    state_variables = ["s_AMPA", "x_NMDA", "s_NMDA", "s_GABA", "V_m",  "t_ref",  # state variables
                       "spikes_ext",  "spikes", "rate", "I_syn", "I_L", "I_AMPA", "I_NMDA", "I_GABA", "I_AMPA_ext"]  # non-state variables
    _nvar = 16
    cvar = numpy.array([0], dtype=numpy.int32)
    number_of_modes = 200  # assuming that 0...N_E-1 are excitatory and N_E ... number_of_modes-1 are inhibitory

    # Return number of excitatory neurons/modes per region
    def _n_E(self, i_region):
        if i_region in self._spiking_regions_inds:
            return super(MultiscaleWongWangExcIOInhI, self)._n_E(i_region)
        else:
            return 1

    # Return number of inhibitory neurons/modes per region
    def _n_I(self, i_region):
        if i_region in self._spiking_regions_inds:
            return super(MultiscaleWongWangExcIOInhI, self)._n_I(i_region)
        else:
            return 1

    # Return indices of excitatory neurons/modes per region
    def _E(self, i_region):
        if i_region in self._spiking_regions_inds:
            return super(MultiscaleWongWangExcIOInhI, self)._E(i_region)
        else:
            return numpy.arange(self._N_E_max).astype('i')

    # Return indices of inhibitory neurons/modes per region
    def _I(self, i_region):
        if i_region in self._spiking_regions_inds:
            return super(MultiscaleWongWangExcIOInhI, self)._I(i_region)
        else:
            return numpy.arange(self._N_E_max, self.number_of_modes).astype('i')

    def update_initial_conditions_non_state_variables(self, state_variables, coupling, local_coupling=0.0,
                                                      use_numba=False):
        __n_E = []
        __n_I = []
        __E = {}
        __I = {}
        self.__n_E = None
        self.__n_I = None
        self.__E = None
        self.__I = None
        # Initialize all non-state variables, as well as t_ref, to 0, i.e., assuming no spikes in history.
        state_variables[6:] = 0.0
        for ii in range(state_variables.shape[1]):  # For every region node....
            __n_E.append(self._n_E(ii))
            __n_I.append(self._n_I(ii))
            __E[ii] = numpy.arange(__n_E[-1]).astype('i')  # excitatory neurons' indices
            __I[ii] = numpy.arange(self._N_E_max, self._N_E_max + __n_I[-1]).astype('i')  # inhibitory neurons' indices
            # Make sure that all empty positions are set to 0.0, if any:
            self._zero_empty_positions(state_variables, __E[ii], __I[ii], ii)
            # Set  inhibitory synapses for excitatory neurons & excitatory synapses for inhibitory neurons to 0.0...
            self._zero_cross_synapses(state_variables, __E[ii], __I[ii], ii)
        self.__n_E = __n_E
        self.__n_I = __n_I
        self.__E = __E
        self.__I = __I
        return state_variables

    def update_non_state_variables(self, state_variables, coupling, local_coupling=0.0, use_numba=False):

        for ii in range(state_variables[0].shape[0]):
            _E = self._E(ii)  # excitatory neurons'/modes' indices
            _I = self._I(ii)  # inhibitory neurons'/modes' indices

            # Make sure that all empty positions are set to 0.0, if any:
            self._zero_empty_positions(state_variables, _E, _I, ii)

            # Set  inhibitory synapses for excitatory neurons & excitatory synapses for inhibitory neurons to 0.0...
            self._zero_cross_synapses(state_variables, _E, _I, ii)

            # compute large scale coupling_ij = sum(C_ij * S_e(t-t_ij))
            large_scale_coupling = numpy.sum(coupling[0, ii, :self._N_E_max])

            if ii in self._spiking_regions_inds:

                # -----------------------------------Updates after previous iteration:----------------------------------

                # Refractory neurons from past spikes if 6. t_ref > 0.0
                self._refractory_neurons_E[ii] = state_variables[6, ii, _E] > 0.0
                self._refractory_neurons_I[ii] = state_variables[6, ii, _I] > 0.0

                # set 5. V_m for refractory neurons to V_reset
                state_variables[5, ii, _E] = numpy.where(self._refractory_neurons_E[ii],
                                                         self._x_E(self.V_reset, ii), state_variables[5, ii, _E])

                state_variables[5, ii, _I] = numpy.where(self._refractory_neurons_I[ii],
                                                         self._x_I(self.V_reset, ii), state_variables[5, ii, _I])

                # Compute spikes sent at time t:
                # 8. spikes
                state_variables[8, ii, _E] = numpy.where(state_variables[5, ii, _E] > self._x_E(self.V_thr, ii), 1.0,
                                                         0.0)
                state_variables[8, ii, _I] = numpy.where(state_variables[5, ii, _I] > self._x_I(self.V_thr, ii), 1.0,
                                                         0.0)
                self._spikes_E[ii] = state_variables[8, ii, _E] > 0.0
                self._spikes_I[ii] = state_variables[8, ii, _I] > 0.0

                # set 5. V_m for spiking neurons to V_reset
                state_variables[5, ii, _E] = numpy.where(self._spikes_E[ii],
                                                         self._x_E(self.V_reset, ii), state_variables[5, ii, _E])

                state_variables[5, ii, _I] = numpy.where(self._spikes_I[ii],
                                                         self._x_I(self.V_reset, ii), state_variables[5, ii, _I])

                # set 6. t_ref  to tau_ref for spiking neurons
                state_variables[6, ii, _E] = numpy.where(self._spikes_E[ii],
                                                         self._x_E(self.tau_ref_E, ii), state_variables[6, ii, _E])

                state_variables[6, ii, _I] = numpy.where(self._spikes_I[ii],
                                                         self._x_I(self.tau_ref_I, ii), state_variables[6, ii, _I])

                # Refractory neurons including current spikes sent at time t
                self._refractory_neurons_E[ii] = numpy.logical_or(self._refractory_neurons_E[ii], self._spikes_E[ii])
                self._refractory_neurons_I[ii] = numpy.logical_or(self._refractory_neurons_I[ii], self._spikes_I[ii])

                # 9. rate
                # Compute the average population rate sum_of_population_spikes / number_of_population_neurons
                # separately for excitatory and inhibitory populations,
                # and set it at the first position of each population, similarly to the mean-field region nodes
                state_variables[9, ii, _E] = 0.0
                state_variables[9, ii, _I] = 0.0
                state_variables[9, ii, 0] = numpy.sum(state_variables[8, ii, _E]) / self._n_E(ii)
                state_variables[9, ii, self._N_E_max] = numpy.sum(state_variables[8, ii, _I]) / self._n_I(ii)

                # -------------------------------------Updates before next iteration:---------------------------------------

                # ----------------------------------First deal with inputs at time t:---------------------------------------

                # Collect external spikes received at time t, and update the incoming s_AMPA_ext synapse:

                # 7. spikes_ext
                # get external spike stimulus 7. spike_ext, if any:
                state_variables[7, ii, _E] = self._x_E(self.spikes_ext, ii)
                state_variables[7, ii, _I] = self._x_I(self.spikes_ext, ii)

                # 4. s_AMPA_ext
                # ds_AMPA_ext/dt = -1/tau_AMPA * (s_AMPA_exc + spikes_ext)
                # Add the spike at this point to s_AMPA_ext
                state_variables[4, ii, _E] += state_variables[7, ii, _E]  # spikes_ext
                state_variables[4, ii, _I] += state_variables[7, ii, _I]  # spikes_ext

                # Compute currents based on synaptic gating variables at time t:

                # V_E_E = V_m - V_E
                V_E_E = state_variables[5, ii, _E] - self._x_E(self.V_E, ii)
                V_E_I = state_variables[5, ii, _I] - self._x_I(self.V_E, ii)

                # 10. I_syn = I_L + I_AMPA + I_NMDA + I_GABA + I_AMPA_EXT
                state_variables[10, ii, _E] = numpy.sum(state_variables[10:, ii, _E], axis=0)
                state_variables[10, ii, _I] = numpy.sum(state_variables[10:, ii, _I], axis=0)

                # 11. I_L = g_m * (V_m - V_E)
                state_variables[11, ii, _E] = \
                    self._x_E(self.g_m_E, ii) * (state_variables[5, ii, _E] - self._x_E(self.V_L, ii))
                state_variables[11, ii, _I] = \
                    self._x_I(self.g_m_I, ii) * (state_variables[5, ii, _I] - self._x_I(self.V_L, ii))

                w_EE = self._x_E(self.w_EE, ii)
                w_EI = self._x_E(self.w_EI, ii)

                # 12. I_AMPA = g_AMPA * (V_m - V_E) * sum(w * s_AMPA_k)
                coupling_AMPA_E, coupling_AMPA_I = \
                    self._compute_region_exc_population_coupling(state_variables[0, ii, _E], w_EE, w_EI)
                state_variables[12, ii, _E] = self._x_E(self.g_AMPA_E, ii) * V_E_E * coupling_AMPA_E
                # s_AMPA
                state_variables[12, ii, _I] = self._x_I(self.g_AMPA_I, ii) * V_E_I * coupling_AMPA_I

                # 13. I_NMDA = g_NMDA * (V_m - V_E) / (1 + lamda_NMDA * exp(-beta*V_m)) * sum(w * s_NMDA_k)
                coupling_NMDA_E, coupling_NMDA_I = \
                    self._compute_region_exc_population_coupling(state_variables[2, ii, _E], w_EE, w_EI)
                state_variables[13, ii, _E] = \
                    self._x_E(self.g_NMDA_E, ii) * V_E_E \
                    / (self._x_E(self.lamda_NMDA, ii) * numpy.exp(-self._x_E(self.beta, ii) *
                                                                  state_variables[5, ii, _E])) \
                    * coupling_NMDA_E  # s_NMDA
                state_variables[13, ii, _I] = \
                    self._x_I(self.g_NMDA_I, ii) * V_E_I \
                    / (self._x_I(self.lamda_NMDA, ii) * numpy.exp(-self._x_I(self.beta, ii) *
                                                                  state_variables[5, ii, _I])) \
                    * coupling_NMDA_I  # s_NMDA

                # 14. I_GABA = g_GABA * (V_m - V_I) * sum(w_ij * s_GABA_k)
                w_IE = self._x_I(self.w_IE, ii)
                w_II = self._x_I(self.w_II, ii)
                coupling_GABA_E, coupling_GABA_I = \
                    self._compute_region_inh_population_coupling(state_variables[3, ii, _I], w_IE, w_II)
                state_variables[14, ii, _E] = self._x_E(self.g_GABA_E, ii) * \
                                              (state_variables[5, ii, _E] - self._x_E(self.V_I, ii)) * \
                                              coupling_GABA_E  # s_GABA
                state_variables[14, ii, _I] = self._x_I(self.g_GABA_I, ii) * \
                                              (state_variables[5, ii, _I] - self._x_I(self.V_I, ii)) * \
                                              coupling_GABA_I  # s_GABA

                # 15. I_AMPA_ext = g_AMPA_ext * (V_m - V_E) * ( G*sum{c_ij sum{s_AMPA_j(t-delay_ij)}} + s_AMPA_ext)
                # Compute large scale coupling_ij = sum(c_ij * S_e(t-t_ij))
                large_scale_coupling += numpy.sum(local_coupling * state_variables[0, ii, _E])
                state_variables[15, ii, _E] = self._x_E(self.g_AMPA_ext_E, ii) * V_E_E * \
                                              (self._x_E(self.G, ii) * large_scale_coupling
                                               + state_variables[4, ii, _E])
                #                                          # feedforward inhibition
                state_variables[15, ii, _I] = self._x_I(self.g_AMPA_ext_I, ii) * V_E_I * \
                                              (self._x_I(self.G, ii) * self._x_I(self.lamda, ii) * large_scale_coupling
                                               + state_variables[4, ii, _I])

            else:

                # For mean field modes:
                # Given that the 3rd dimension corresponds to neurons, not modes,
                # we use only the first element of its population, i.e., 0 and _I[0],
                # and consider all the rest to be identical
                # Similarly, we assume that all parameters are of of these shapes:
                # (1, ), (1, 1), (number_of_regions, ), (number_of_regions, 1)

                # S_e = s_AMPA = s_NMDA
                # S_i = s_GABA

                # 1. s_NMDA
                # = s_AMPA for excitatory mean field models
                state_variables[1, ii, _E] = state_variables[0, ii, 0]

                # 1. x_NMDA, 4. s_AMPA_ext, 5. V_m, 6. t_ref, 7. spikes_ext, 8. spikes, 11. I_L
                # are 0 for mean field models:
                state_variables[[1, 4, 5, 6, 7, 8, 11], ii] = 0.0

                # J_N is indexed by the receiver node
                J_N_E = self._region(self.J_N, ii)
                J_N_I = self._region(self.J_N, ii)

                # 12. I_AMPA
                # = w+ * J_N * S_e
                state_variables[12, ii, _E] = self._region(self.w_p, ii) * J_N_E * state_variables[0, ii, 0]
                # = J_N * S_e
                state_variables[12, ii, _I] = J_N_I * state_variables[0, ii, 0]

                # 13. I_NMDA = I_AMPA for mean field models
                state_variables[13, ii] = state_variables[13, ii]

                # 14. I_GABA                                              # 3. s_GABA
                state_variables[14, ii, _E] = - self._x_E(self.J_i, ii) * state_variables[3, ii, _I[0]]  # = -J_i*S_i
                state_variables[14, ii, _I] = - state_variables[3, ii, _I[0]]  # = - S_i

                # 15. I_AMPA_ext
                large_scale_coupling += local_coupling * state_variables[0, ii, 0]
                # = G * J_N * coupling_ij = G * J_N * sum(C_ij * S_e(t-t_ij))
                state_variables[15, ii, _E] = self._x_E(self.G, ii)[0] * J_N_E * large_scale_coupling
                # = lamda * G * J_N * coupling_ij = lamda * G * J_N * sum(C_ij * S_e(t-t_ij))
                state_variables[15, ii, _I] = \
                    self._x_I(self.G, ii)[0] * self._x_I(self.lamda, ii)[0] * J_N_I * large_scale_coupling

                # 8. I_syn = I_E(NMDA) + I_I(GABA) + I_AMPA_ext
                # Note measuring twice I_AMPA and I_NMDA though, as they count as a single excitatory current:
                state_variables[10, ii, _E] = numpy.sum(state_variables[13:, ii, 0], axis=0)
                state_variables[10, ii, _I] = numpy.sum(state_variables[13:, ii, _I[0]], axis=0)

                # 6. rate sigmoidal of total current = I_syn + I_o

                total_current = \
                    state_variables[10, ii, 0] + self._region(self.W_e, ii) * self._region(self.I_o, ii)
                # Sigmoidal activation: (a*I_tot_current - b) / ( 1 - exp(-d*(a*I_tot_current - b)))
                total_current = self._region(self.a_e, ii) * total_current - self._region(self.b_e, ii)
                state_variables[9, ii, _E] = \
                    total_current / (1 - numpy.exp(-self._region(self.d_e, ii) * total_current))

                total_current = \
                    state_variables[10, ii, _I[0]] + self._region(self.W_i, ii) * self._region(self.I_o, ii)
                # Sigmoidal activation:
                total_current = self._region(self.a_i, ii) * total_current - self._region(self.b_i, ii)
                state_variables[9, ii, _I] = \
                    total_current / (1 - numpy.exp(-self._region(self.d_i, ii) * total_current))

        return state_variables

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0, update_non_state_variables=False):
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

        derivative = 0.0 * state_variables

        for ii in range(state_variables[0].shape[0]):

            _E = self._E(ii)  # excitatory neurons/populations indices
            _I = self._I(ii)  # inhibitory neurons/populations indices

            if ii in self._spiking_regions_inds:
                exc_spikes = state_variables[8, ii, _E]  # excitatory spikes
                tau_AMPA_E = self._x_E(self.tau_AMPA, ii)

                # 0. s_AMPA
                # ds_AMPA/dt = -1/tau_AMPA * s_AMPA + exc_spikes
                derivative[0, ii, _E] = -state_variables[0, ii, _E] / tau_AMPA_E + exc_spikes

                # 1. x_NMDA
                # dx_NMDA/dt = -x_NMDA/tau_NMDA_rise + exc_spikes
                derivative[1, ii, _E] = \
                    -state_variables[1, ii, _E] / self._x_E(self.tau_NMDA_rise, ii) + exc_spikes

                # 2. s_NMDA
                # ds_NMDA/dt = -1/tau_NMDA_decay * s_NMDA + alpha*x_NMDA*(1-s_NMDA)
                derivative[2, ii, _E] = \
                    -state_variables[2, ii, _E] / self._x_E(self.tau_NMDA_decay, ii) \
                    + self._x_E(self.alpha, ii) * state_variables[1, ii, _E] * (1 - state_variables[2, ii, _E])

                # 4. s_AMPA_ext
                # ds_AMPA_ext/dt = -1/tau_AMPA * (s_AMPA_exc + spikes_ext)
                derivative[4, ii, _E] = -state_variables[4, ii, _E] / tau_AMPA_E

                # excitatory refractory neurons:
                ref = self._refractory_neurons_E[ii]
                not_ref = numpy.logical_not(ref)

                # 5. Integrate only non-refractory V_m
                # C_m*dV_m/dt = - I_L- I_AMPA - I_NMDA - I_GABA  - I_AMPA_EXT + I_ext
                _E_not_ref = _E[not_ref]
                derivative[5, ii, _E_not_ref] = (
                                                        - state_variables[11, ii, _E_not_ref]  # 11. I_L
                                                        - state_variables[12, ii, _E_not_ref]  # 12. I_AMPA
                                                        - state_variables[13, ii, _E_not_ref]  # 13. I_NMDA
                                                        - state_variables[14, ii, _E_not_ref]  # 14. I_GABA
                                                        - state_variables[15, ii, _E_not_ref]  # 15. I_AMPA_ext
                                                        + self._x_E_ref(self.I_ext, ii, not_ref)  # I_ext
                                                ) \
                                                / self._x_E_ref(self.C_m_E, ii, not_ref)

                # 6...and only refractory t_ref:
                # dt_ref/dt = -1 for t_ref > 0  so that t' = t - dt
                # and 0 otherwise
                derivative[6, ii, _E[ref]] = -1.0

                # Inhibitory neurons:

                _I = self._I(ii)  # inhibitory neurons indices

                # 3. s_GABA/dt = - s_GABA/tau_GABA + inh_spikes
                derivative[3, ii, _I] = \
                    -state_variables[3, ii, _I] / self._x_I(self.tau_GABA, ii) + state_variables[8, ii, _I]

                # 4. s_AMPA_ext
                # ds_AMPA_ext/dt = -1/tau_AMPA * (s_AMPA_exc + spikes_ext)
                derivative[4, ii, _I] = -state_variables[4, ii, _I] / self._x_I(self.tau_AMPA, ii)

                # inhibitory refractory neurons:
                ref = self._refractory_neurons_I[ii]
                not_ref = numpy.logical_not(ref)

                # 5. Integrate only non-refractory V_m
                # C_m*dV_m/dt = - I_L - I_AMPA_EXT - I_AMPA - I_GABA - I_NMDA + I_ext
                # 5. Integrate only non-refractory V_m
                # C_m*dV_m/dt = - I_L- I_AMPA - I_NMDA - I_GABA  - I_AMPA_EXT + I_ext
                _I_not_ref = _I[not_ref]
                derivative[5, ii, _I_not_ref] = (
                                                        - state_variables[11, ii, _I_not_ref]   # 11. I_L
                                                        - state_variables[12, ii, _I_not_ref]  # 12. I_AMPA
                                                        - state_variables[13, ii, _I_not_ref]  # 13. I_NMDA
                                                        - state_variables[14, ii, _I_not_ref]  # 14. I_GABA
                                                        - state_variables[15, ii, _I_not_ref]  # 15. I_AMPA_ext
                                                        + self._x_I_ref(self.I_ext, ii, not_ref)  # I_ext
                                                ) \
                                                / self._x_I_ref(self.C_m_I, ii, not_ref)

                # 6...and only refractory t_ref:
                # dt_ref/dt = -1 for t_ref > 0  so that t' = t - dt
                # and 0 otherwise
                derivative[6, ii, _I[ref]] = -1.0

            else:
                # For mean field modes:
                # Given that the 3rd dimension corresponds to neurons, not modes,
                # we use only the first element of its population, i.e., 0 and _I[0],
                # and consider all the rest to be identical
                # Similarly, we assume that all parameters are of of these shapes:
                # (1, ), (1, 1), (number_of_regions, ), (number_of_regions, 1)

                # S_e = s_AMPA
                derivative[0, ii, _E] = \
                   - (state_variables[0, ii, 0] / self._region(self.tau_e, ii)) \
                   + (1 - state_variables[0, ii, 0]) * state_variables[9, ii, 0] * self._region(self.gamma_e, ii)
                # s_NMDA <= s_AMPA
                derivative[2, ii, _E] = derivative[0, ii, 0]

                # S_i = s_GABA
                derivative[1, ii, _I] = \
                    - (state_variables[1, ii, _I[0]] / self._region(self.tau_i, ii)) \
                    + state_variables[9, ii, _I[0]] * self._region(self.gamma_i, ii)

        return derivative

    def dfun(self, x, c, local_coupling=0.0, update_non_state_variables=True):
        return self._numpy_dfun(x, c, local_coupling, update_non_state_variables)
