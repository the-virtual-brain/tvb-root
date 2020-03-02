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
        default={"s_AMPA": numpy.array([0., 1.]),
                 "s_GABA": numpy.array([0., 1.]),
                 "s_NMDA": numpy.array([0., 1.]),
                 "x_NMDA": numpy.array([0., None]),
                 "V_m": numpy.array([None, None]),
                 "t_ref": numpy.array([0., None]),
                 "spikes": numpy.array([0., None]),
                 "rate": numpy.array([0., None]),
                 "I_syn": numpy.array([None, None]),
                 "I_L": numpy.array([None, None]),
                 "I_AMPA_ext": numpy.array([None, None]),
                 "I_AMPA": numpy.array([None, None]),
                 "I_GABA": numpy.array([None, None]),
                 "I_NMDA": numpy.array([None, None])},
        label="State Variable boundaries [lo, hi]",
        doc="""The values for each state-variable should be set to encompass
            the boundaries of the dynamic range of that state-variable. 
            Set None for one-sided boundaries""")

    # Choosing default initial conditions such as that there are no spikes initialy:
    state_variable_range = Final(
        default={"s_AMPA": numpy.array([0., 1.]),        # 0
                 "s_GABA": numpy.array([0., 1.]),        # 1
                 "s_NMDA": numpy.array([0., 1.]),        # 2
                 "x_NMDA": numpy.array([0., 200.]),      # 3
                 "V_m": numpy.array([-70., -50.]),       # 4
                 "t_ref": numpy.array([0., 1.]),         # 5
                 "spikes": numpy.array([0., 0.5]),       # 6
                 "rate": numpy.array([0., 1000.]),       # 7
                 "I_syn": numpy.array([0., 1000.]),      # 8
                 "I_L": numpy.array([0., 1000.]),        # 9
                 "I_AMPA_ext": numpy.array([-10., 0.]),  # 10
                 "I_AMPA": numpy.array([-1000., 0.]),    # 11
                 "I_GABA": numpy.array([0., 1000.]),     # 12
                 "I_NMDA": numpy.array([-1000., 0.])},   # 13
        label="State variable ranges [lo, hi]",
        doc="""State variable ranges [lo, hi] for random initial condition construction""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("s_AMPA",  "s_GABA", "s_NMDA", "x_NMDA",
                 "V_m",  "t_ref", "spikes", "rate", "I_syn", "I_L", "I_AMPA_ext", "I_AMPA", "I_GABA", "I_NMDA"),
        default=("s_AMPA",  "s_GABA", "s_NMDA", "x_NMDA",
                 "V_m",  "t_ref",  "spikes", "rate", "I_syn", "I_L", "I_AMPA_ext", "I_AMPA", "I_GABA", "I_NMDA"),
        doc="""default state variables to be monitored""")

    state_variables = ["s_AMPA",  "s_GABA", "s_NMDA", "x_NMDA", "V_m", "t_ref",      # state variables
                       "spikes", "rate", "I_syn", "I_L", "I_AMPA_ext", "I_AMPA", "I_GABA", "I_NMDA"]  # non-state variables
    _nvar = 14
    cvar = numpy.array([0], dtype=numpy.int32)
    number_of_modes = 170  # assuming that 0...N_E-1 are excitatory and N_E ... number_of_modes-1 are inhibitory

    def update_derived_parameters(self):
        """
        When needed, this should be a method for calculating parameters that are
        calculated based on paramaters directly set by the caller. For example,
        see, ReducedSetFitzHughNagumo. When not needed, this pass simplifies
        code that updates an arbitrary models parameters -- ie, this can be
        safely called on any model, whether it's used or not.
        """
        self._N_E_max = int(numpy.max(self.N_E))  # maximum number of excitatory neurons/modes
        self.number_of_modes = int(self._N_E_max + numpy.max(self.N_I))
        # # todo: this exclusion list is fragile, consider excluding declarative attrs that are not arrays
        # excluded_params = ("state_variable_range", "state_variable_boundaries", "variables_of_interest",
        #                    "noise", "psi_table", "nerf_table", "gid")
        # for param in type(self).declarative_attrs:
        #     if param in excluded_params:
        #         continue
        #     region_parameters = getattr(self, param)
        #     try:
        #         region_parameters[0, 0]
        #     except:
        #         new_parameters = numpy.reshape(region_parameters, (-1, 1))
        #         setattr(self, param, new_parameters)
        #         region_parameters = getattr(self, param)
        #     assert (region_parameters.shape[1] == self.number_of_modes) or (region_parameters.shape[1] == 1)

    def _region(self, x, i_region):
        try:
            return x[i_region]
        except:
            return x[0]

    # Return number of excitatory neurons/modes per region
    def _n_E(self, i_region):
        if i_region in self._spiking_regions_inds:
            return int(self._region(self.N_E, i_region))
        else:
            return 1

    # Return number of inhibitory neurons/modes per region
    def _n_I(self, i_region):
        if i_region in self._spiking_regions_inds:
            return int(self._region(self.N_I, i_region))
        else:
            return 1

    # Return indices of excitatory neurons/modes per region
    def _E(self, i_region):
        if i_region in self._spiking_regions_inds:
            return numpy.arange(self._n_E(i_region)).astype('i')
        else:
            return numpy.arange(self._N_E_max).astype('i')

    # Return indices of inhibitory neurons/modes per region
    def _I(self, i_region):
        if i_region in self._spiking_regions_inds:
            return numpy.arange(self._N_E_max, self._N_E_max + self._n_I(i_region)).astype('i')
        else:
            return numpy.arange(self._N_E_max, self.number_of_modes).astype('i')

    # Return x variables of excitatory neurons/modes per region
    def _x_E(self, x, i_region):
        x_E = self._region(x, i_region)
        try:
            return x_E[self._E]  # if region parameter shape is (n_neurons,)
        except:
            try:
                return numpy.array([x_E[0], ])   # if region parameter shape is (1,) or (1, 1)
            except:
                return numpy.array([x_E, ])      # if region parameter is float

    # Return  x variables of inhibitory neurons/modes per region
    def _x_I(self, x, i_region):
        x_I = self._region(x, i_region)
        try:
            return x_I[self._I]  # if region parameter shape is (n_neurons,)
        except:
            try:
                return numpy.array([x_I[0], ])   # if region parameter shape is (1,) or (1, 1)
            except:
                return numpy.array([x_I, ])      # if region parameter is float

    # Return x variables of refractory excitatory neurons/modes per region
    def _x_E_ref(self, x, i_region, ref):
        x_E = self._x_E(x, i_region)
        try:
            return x_E[ref]
        except:
            return x_E

    # Return x variables of refractory inhibitory neurons/modes per region
    def _x_I_ref(self, x, i_region, ref):
        x_I = self._x_I(x, i_region)
        try:
            return x_I[ref]
        except:
            return x_I

    def _compute_region_exc_population_coupling(self, s_E, w_EE, w_EI):
        # Scale w_EE, w_EI per region accordingly,
        # to implement coupling schemes that depend on the total number of neurons
        # exc -> exc
        c_ee = w_EE * s_E
        c_ee = numpy.sum(c_ee) \
               - numpy.where(self._auto_connection, 0.0, c_ee)  # optionally remove auto-connections
        #                exc -> inh:
        return c_ee, numpy.sum(w_EI * s_E)

    def _compute_region_inh_population_coupling(self, s_I, w_IE, w_II):
        # Scale w_IE, w_II per region accordingly,
        # to implement coupling schemes that depend on the total number of neurons
        # inh -> inh
        c_ii = w_II * s_I
        c_ii = numpy.sum(w_II * s_I) \
               - numpy.where(self._auto_connection, 0.0, c_ii)  # optionally remove auto-connections
        #           inh -> exc:
        return numpy.sum(w_IE * s_I), c_ii

    def update_initial_conditions_non_state_variables(self, state_variables, coupling, local_coupling=0.0,
                                                      use_numba=False):
        # Initialize all non-state variables, as well as t_ref, to 0, i.e., assuming no spikes in history.
        state_variables[5:] = 0.0
        for ii in range(state_variables.shape[1]):  # For every region node....
            _E = self._E(ii)  # excitatory neurons' indices
            _I = self._I(ii)  # inhibitory neurons' indices

            # Make sure that all empty positions are set to 0.0, if any:
            _empty = numpy.unique(_E.tolist() + _I.tolist())  # all indices occupied by neurons
            if len(_empty) < state_variables[0].shape[1]:
                _empty = numpy.delete(numpy.arange(state_variables[0].shape[1]), _empty)  # all empty indices
                # Set all empty positions to 0
                state_variables[:, ii, _empty] = 0.0

            # Set  inhibitory synapses for excitatory neurons to 0.0...
            # 1. s_GABA
            state_variables[1, ii, _E] = 0.0
            # ...and excitatory synapses for inhibitory neurons to 0.0
            # 0. s_AMPA, 2. s_NMDA and 3. x_NMDA
            state_variables[[0, 2, 3]][:, ii, _I] = 0.0
            
            if ii not in self._spiking_regions_inds:
                # 3. x_NMDA, 4. V_m,  are 0 for mean field models:
                state_variables[[3, 4], ii] = 0.0
                
        return state_variables

    def update_non_state_variables(self, state_variables, coupling, local_coupling=0.0, use_numba=False):

        for ii in range(state_variables[0].shape[0]):
            _E = self._E(ii)  # excitatory neurons'/modes' indices
            _I = self._I(ii)  # inhibitory neurons'/modes' indices

            # Make sure that all empty positions are set to 0.0, if any:
            _EI = numpy.unique(_E.tolist() + _I.tolist())  # all indices occupied by neurons
            if len(_EI) < state_variables[0].shape[1]:
                _empty = numpy.delete(numpy.arange(state_variables[0].shape[1]), _EI)  # all empty indices
                # Set all empty positions to 0
                state_variables[:, ii, _empty] = 0.0

            # Set  inhibitory synapses for excitatory neurons to 0.0...
            # 1. s_GABA
            state_variables[1, ii, _E] = 0.0
            # ...and excitatory synapses for inhibitory neurons to 0.0
            # 0. s_AMPA, 2. s_NMDA and 3. x_NMDA
            state_variables[[0, 2, 3]][:, ii, _I] = 0.0

            # compute large scale coupling_ij = sum(C_ij * S_e(t-t_ij))
            large_scale_coupling = numpy.sum(coupling[0, ii, :self._N_E_max])

            if ii in self._spiking_regions_inds:

                # Refractory neurons from past spikes if t_ref > 0.0
                self._refractory_neurons_E[ii] = state_variables[5, ii, _E] > 0.0
                self._refractory_neurons_I[ii] = state_variables[5, ii, _I] > 0.0

                # set 4. V_m for refractory neurons to V_reset
                state_variables[4, ii, _E] = numpy.where(self._refractory_neurons_E[ii],
                                                         self._x_E(self.V_reset, ii), state_variables[4, ii, _E])

                state_variables[4, ii, _I] = numpy.where(self._refractory_neurons_I[ii],
                                                         self._x_I(self.V_reset, ii), state_variables[4, ii, _I])

                # 6. spikes
                state_variables[6, ii, _E] = numpy.where(state_variables[4, ii, _E] > self._x_E(self.V_thr, ii),
                                                         1.0, 0.0)
                state_variables[6, ii, _I] = numpy.where(state_variables[4, ii, _I] > self._x_I(self.V_thr, ii),
                                                         1.0, 0.0)
                self._spikes_E[ii] = state_variables[6, ii, _E] > 0.0
                self._spikes_I[ii] = state_variables[6, ii, _I] > 0.0

                # set 4. V_m for spiking neurons to V_reset
                state_variables[4, ii, _E] = numpy.where(self._spikes_E[ii],
                                                         self._x_E(self.V_reset, ii), state_variables[4, ii, _E])

                state_variables[4, ii, _I] = numpy.where(self._spikes_I[ii],
                                                         self._x_I(self.V_reset, ii), state_variables[4, ii, _I])

                # set 5. t_ref  to t_ref for spiking neurons
                state_variables[5, ii, _E] = numpy.where(self._spikes_E[ii],
                                                         self._x_E(self.tau_ref_E, ii), state_variables[5, ii, _E])

                state_variables[5, ii, _I] = numpy.where(self._spikes_I[ii],
                                                         self._x_I(self.tau_ref_I, ii), state_variables[5, ii, _I])

                # Refractory neurons including current spikes
                self._refractory_neurons_E[ii] = numpy.logical_or(self._refractory_neurons_E[ii], self._spikes_E[ii])
                self._refractory_neurons_I[ii] = numpy.logical_or(self._refractory_neurons_I[ii], self._spikes_I[ii])

                # 7. rate
                # Compute the average population rate sum_of_population_spikes / number_of_population_neurons
                # separately for excitatory and inhibitory populations,
                # and set it to the first position of each population, similarly to the mean-field region nodes
                state_variables[7, ii, _EI] = 0.0
                state_variables[7, ii, 0] = numpy.sum(state_variables[6, ii, _E]) / self._n_E(ii)
                state_variables[7, ii, self._N_E_max] = numpy.sum(state_variables[6, ii, _I]) / self._n_I(ii)

                V_E_E = state_variables[4, ii, _E] - self._x_E(self.V_E, ii)
                V_E_I = state_variables[4, ii, _I] - self._x_I(self.V_E, ii)

                # 9. I_L = g_m * (V_m - V_E)
                state_variables[9, ii, _E] = \
                    self._x_E(self.g_m_E, ii) * (state_variables[4, ii, _E] - self._x_E(self.V_L, ii))
                state_variables[9, ii, _I] = \
                    self._x_I(self.g_m_I, ii) * (state_variables[4, ii, _I] - self._x_I(self.V_L, ii))

                # 10. I_AMPA_ext = g_AMPA_ext * (V_m - V_E) * G*sum{C_ij sum{s_AMPA_j(t-delay_ij)} }
                large_scale_coupling += numpy.sum(local_coupling * state_variables[0, ii, _E])
                state_variables[10, ii, _E] = self._x_E(self.g_AMPA_ext_E, ii) * V_E_E * \
                                              self._x_E(self.G, ii) * large_scale_coupling
                #                                          # feedforward inhibition
                state_variables[10, ii, _I] = self._x_I(self.g_AMPA_ext_I, ii) * V_E_I * \
                                              self._x_I(self.G, ii) * self._x_I(self.lamda, ii) * large_scale_coupling

                w_EE = self._x_E(self.w_EE, ii)
                w_EI = self._x_E(self.w_EI, ii)

                # 11. I_AMPA = g_AMPA * (V_m - V_E) * sum(w * s_AMPA_k)
                coupling_AMPA_E, coupling_AMPA_I = \
                    self._compute_region_exc_population_coupling(state_variables[0, ii, _E], w_EE, w_EI)
                state_variables[11, ii, _E] = self._x_E(self.g_AMPA_E, ii) * V_E_E * coupling_AMPA_E
                                               # s_AMPA
                state_variables[11, ii, _I] = self._x_I(self.g_AMPA_I, ii) * V_E_I * coupling_AMPA_I

                # 12. I_GABA = g_GABA * (V_m - V_I) * sum(w_ij * s_GABA_k)
                w_EI = self._x_I(self.w_EI, ii)
                w_II = self._x_I(self.w_II, ii)
                coupling_GABA_E, coupling_GABA_I = \
                    self._compute_region_inh_population_coupling(state_variables[1, ii, _I], w_EI, w_II)
                state_variables[12, ii, _E] = self._x_E(self.g_GABA_E, ii) * \
                                              (state_variables[4, ii, _E] - self._x_E(self.V_I, ii)) * \
                                              coupling_GABA_E  # s_GABA
                state_variables[12, ii, _I] = self._x_I(self.g_GABA_I, ii) * \
                                              (state_variables[4, ii, _I] - self._x_I(self.V_I, ii)) * \
                                              coupling_GABA_I  # s_GABA

                # 13. I_NMDA = g_NMDA * (V_m - V_E) / (1 + lamda_NMDA * exp(-beta*V_m)) * sum(w * s_NMDA_k)
                coupling_NMDA_E, coupling_NMDA_I = \
                    self._compute_region_exc_population_coupling(state_variables[2, ii, _E], w_EE, w_EI)
                state_variables[13, ii, _E] = \
                    self._x_E(self.g_NMDA_E, ii) * V_E_E / (self._x_E(self.lamda_NMDA, ii) *
                                                            numpy.exp(-self._x_E(self.beta, ii) *
                                                                      state_variables[4, ii, _E])) * \
                    coupling_NMDA_E  # s_NMDA
                state_variables[13, ii, _I] = \
                    self._x_I(self.g_NMDA_I, ii) * V_E_I / (self._x_I(self.lamda_NMDA, ii) *
                                                            numpy.exp(-self._x_I(self.beta, ii) *
                                                                      state_variables[4, ii, _I])) * \
                    coupling_NMDA_I  # s_NMDA

                # 8. I_syn
                state_variables[8, ii, _EI] = numpy.sum(state_variables[10:, ii, _EI], axis=0)

            else:

                # For mean field modes:
                # Given that the 3rd dimension corresponds to neurons, not modes,
                # we use only the first element of its population, i.e., 0 and _I[0],
                # and consider all the rest to be identical
                # Similarly, we assume that all parameters are of of these shapes:
                # (1, ), (1, 1), (number_of_regions, ), (number_of_regions, 1)

                # S_e = s_AMPA = s_NMDA
                # S_i = s_GABA

                # 2. s_NMDA
                # = s_AMPA for excitatory mean field models
                state_variables[2, ii, _E] = state_variables[0, ii, 0]

                # 3. x_NMDA, 4. V_m, 5. t_ref, 6. spikes, 9. I_L are 0 for mean field models:
                state_variables[[3, 4, 5, 6, 9], ii] = 0.0

                # J_N is indexed by the receiver node
                J_N_E = self._region(self.J_N, ii)
                J_N_I = self._region(self.J_N, ii)

                # 10. I_AMPA_ext
                large_scale_coupling += local_coupling * state_variables[0, ii, 0]
                # = G * J_N * coupling_ij = G * J_N * sum(C_ij * S_e(t-t_ij))
                state_variables[10, ii, _E] = self._x_E(self.G, ii)[0] * J_N_E * large_scale_coupling
                # = lamda * G * J_N * coupling_ij = lamda * G * J_N * sum(C_ij * S_e(t-t_ij))
                state_variables[10, ii, _I] = \
                    self._x_I(self.G, ii)[0] * self._x_I(self.lamda, ii)[0] * J_N_I * large_scale_coupling

                # 11. I_AMPA
                # = w+ * J_N * S_e
                state_variables[11, ii, _E] = self._region(self.w_p, ii) * J_N_E * state_variables[0, ii, 0]
                # = J_N * S_e
                state_variables[11, ii, _I] = J_N_I * state_variables[0, ii, 0]

                # 12. I_GABA                                              # s_GABA
                state_variables[12, ii, _E] = - self._x_E(self.J_i, ii) * state_variables[1, ii, _I[0]]  # = -J_i*S_i
                state_variables[12, ii, _I] = - state_variables[1, ii, _I[0]]                            # = - S_i

                # 13. I_NMDA = I_AMPA for mean field models
                state_variables[13, ii] = state_variables[11, ii]

                # 8. I_syn = I_AMPA_ext + I_AMPA + I_NMDA + I_GABA
                # Note measuring twice I_AMPA and I_NMDA though, as they count as a single excitatory current:
                state_variables[8, ii, _E] = numpy.sum(state_variables[10:13, ii, 0], axis=0)
                state_variables[8, ii, _I] = numpy.sum(state_variables[10:13, ii, _I[0]], axis=0)

                # 6. rate sigmoidal of total current = I_syn + I_o

                total_current = \
                    state_variables[8, ii, 0] + self._region(self.W_e, ii) * self._region(self.I_o, ii)
                # Sigmoidal activation: (a*I_tot_current - b) / ( 1 - exp(-d*(a*I_tot_current - b)))
                total_current = self._region(self.a_e, ii) * total_current - self._region(self.b_e, ii)
                state_variables[7, ii, _E] = \
                    total_current / (1 - numpy.exp(-self._region(self.d_e, ii) * total_current))

                total_current = \
                    state_variables[8, ii, _I[0]] + self._region(self.W_i, ii) * self._region(self.I_o, ii)
                # Sigmoidal activation:
                total_current = self._region(self.a_i, ii) * total_current - self._region(self.b_i, ii)
                state_variables[7, ii, _I] = \
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

                # Excitatory neurons:


                exc_spikes = state_variables[6, ii, _E]  # excitatory spikes
                tau_AMPA_E = self._x_E(self.tau_AMPA, ii)

                # 0. s_AMPA_ext
                # ds_AMPA/dt = -1/tau_AMPA * s_AMPA + exc_spikes
                derivative[0, ii, _E] = -state_variables[0, ii, _E] / tau_AMPA_E + exc_spikes

                # 3. x_NMDA
                # dx_NMDA/dt = -1/tau_NMDA_rise * x_NMDA + exc_spikes
                derivative[3, ii, _E] = -state_variables[3, ii, _E] / self._x_E(self.tau_NMDA_rise, ii) + exc_spikes

                # 2. s_NMDA
                # ds_NMDA/dt = -1/tau_NMDA_decay * s_NMDA + alpha*x_NMDA*(1-s_NMDA)
                derivative[2, ii, _E] = \
                    -state_variables[2, ii, _E] / self._x_E(self.tau_NMDA_decay, ii) \
                    + self._x_E(self.alpha, ii) * state_variables[3, ii, _E] * (1 - state_variables[2, ii, _E])

                # excitatory refractory neurons:
                ref = self._refractory_neurons_E[ii]
                not_ref = numpy.logical_not(ref)

                # 5. Integrate only non-refractory V_m
                # C_m*dV_m/dt = - I_L - I_AMPA_EXT - I_AMPA - I_GABA - I_NMDA + I_ext
                derivative[4, ii, _E][not_ref] = \
                    (- state_variables[7, ii, _E][not_ref]      # I_L
                     - state_variables[8, ii, _E][not_ref]      # I_AMPA_ext
                     - state_variables[9, ii, _E][not_ref]      # I_AMPA
                     - state_variables[10, ii, _E][not_ref]     # I_GABA
                     - state_variables[11, ii, _E][not_ref]     # I_NMDA
                     + self._x_E_ref(self.I_ext, ii, not_ref) / self._x_E_ref(self.C_m_E, ii, not_ref))

                # 6...and only refractory t_ref:
                # dt_ref/dt = -1 for t_ref > 0  so that t' = t - dt
                # and 0 otherwise
                derivative[5, ii, _E][ref] = -1.0

                # Inhibitory neurons:

                # 4. ds_GABA/dt = -s_GABA / tau_GABA + inh_spikes
                derivative[1, ii, _I] = -state_variables[1, ii, _I] / self._x_I(self.tau_GABA, ii) \
                                        + state_variables[6, ii, _I]  # inhibitory spikes

                # inhibitory refractory neurons:
                ref = self._refractory_neurons_I[ii]
                not_ref = numpy.logical_not(ref)

                # 5. Integrate only non-refractory V_m
                # C_m*dV_m/dt = - I_L - I_AMPA_EXT - I_AMPA - I_GABA - I_NMDA + I_ext
                derivative[4, ii, _I][not_ref] = \
                    (- state_variables[7, ii, _I][not_ref]   # I_L
                     - state_variables[8, ii, _I][not_ref]   # I_AMPA_ext
                     - state_variables[9, ii, _I][not_ref]   # I_AMPA
                     - state_variables[10, ii, _I][not_ref]  # I_GABA
                     - state_variables[11, ii, _I][not_ref]  # I_NMDA
                     + self._x_I_ref(self.I_ext, ii, not_ref) /self._x_I_ref(self.C_m_I, ii, not_ref))

                # 6...and only refractory t_ref:
                # dt_ref/dt = -1 for t_ref > 0  so that t' = t - dt
                # and 0 otherwise
                derivative[5, ii, _I][ref] = -1.0

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
                   + (1 - state_variables[0, ii, 0]) * state_variables[7, ii, 0] * self._region(self.gamma_e, ii)
                # s_NMDA <= s_AMPA
                derivative[2, ii, _E] = derivative[0, ii, 0]

                # S_i = s_GABA
                derivative[1, ii, _I] = \
                    - (state_variables[1, ii, _I[0]] / self._region(self.tau_i, ii)) \
                    + state_variables[7, ii, _I[0]] * self._region(self.gamma_i, ii)

        return derivative

    def dfun(self, x, c, local_coupling=0.0, update_non_state_variables=True):
        return self._numpy_dfun(x, c, local_coupling, update_non_state_variables)
