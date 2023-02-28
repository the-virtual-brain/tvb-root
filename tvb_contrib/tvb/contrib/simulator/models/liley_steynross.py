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
#
#

"""
A contributed model: The Liley model as presented in Steyn-Ross et al., 1999

.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import numpy

from tvb.simulator.common import get_logger
from tvb.basic.neotraits.api import NArray, Range, List, Final
import tvb.simulator.models as models

LOG = get_logger(__name__)


class LileySteynRoss(models.Model):
    """
    Liley lumped model as presented in Steyn-Ross et al 1999.

    This  model is to be use for modelling cortical dynamics in which "inputs"
    to neuronal assemblies are treated as random Gaussian fluctuations about a
    mean value. Anesthetic agent effects are modelled as as a modulation of the
    inhibitory neurotransmitter rate constant.

    The main state variable is h_e, the average excitatory soma potential,
    coherent fluctuations of which are believed to be the source of scalp-measured
    electroencephalogram ͑EEG͒ signals.

    Parameters are taken from Table 1 [Steyn-Ross_1999]_


    State variables:

    h_e: exc population mean soma potential [mV]
    h_i: exc population mean soma potential [mV]

    I_ee: total 'exc' current input to 'exc' synapses [mV]
    I_ie: total 'inh' current input to 'exc' synapses [mV]
    I_ei: total 'exc' current input to 'inh' synapses [mV]
    I_ii: total 'inh' current input to 'inh' synapses [mV]

    :math:`\Psi_{jk}`:  weighting factors for the I_jk inputs [dimensionless]

    :math:`phi_e`: long-range (cortico-cortical) spike input to exc population
    :math:`phi_i`: long-range (cortico-cortical) spike input to inh population [ms-1]

    EPSP: exc post-synaptic potential [mV]
    IPSP: inh post-synaptic potential [mV]

    Mean axonal conduction speed: 7 mm/ms

    S_e(h_e): sigmoid function mapping soma potential to firing rate [ms]-1
    S_i(h_i): sigmoid function mapping soma potential to firing rate [ms]-1



    The models (:math:`h_e`, :math:h_i`) phase-plane, including a representation of
    the vector field as well as its nullclines, using default parameters, can be
    seen below:

        .. _phase-plane-LSR:
        .. figure :: img/LileySteynRoss_01_mode_0_pplane.svg
            :alt: LileySteynRoss phase plane (E, I)

            The (:math:`h_e`, :math:`hi`) phase-plane for the LileySteynRoss model.


    References
    ----------

    .. [Steyn-Ross et al 1999] Theoretical electroencephalogram stationary spectrum for a 
       white-noise-driven cortex: Evidence for a general anesthetic-induced phase transition.

    .. [Liley et al 2013] The mesoscopic modelin of burst supression during anesthesia.


    """

    # Define traited attributes for this model, these represent possible kwargs.

    tau_e = NArray(
        label=r":math:`\tau_e`",
        default=numpy.array([40.0]),
        domain=Range(lo=5.0, hi=50.0, step=1.00),
        doc="""Excitatory population, membrane time-constant [ms]""")

    tau_i = NArray(
        label=r":math:`\tau_i`",
        default=numpy.array([40.0]),
        domain=Range(lo=5.0, hi=50.0, step=1.0),
        doc="""Inhibitory population, membrane time-constant [ms]""")

    h_e_rest = NArray(
        label=r":math:`h_e^{rest}`",
        default=numpy.array([-70.0]),
        domain=Range(lo=-90.0, hi=-50.0, step=10.00),
        doc="""Excitatory population, cell resting potential [mV]""")

    h_i_rest = NArray(
        label=r":math:`h_i^{rest}`",
        default=numpy.array([-70.0]),
        domain=Range(lo=-90.0, hi=-50.0, step=10.0),
        doc="""Inhibitory population, cell resting potential [mV]""")

    h_e_rev = NArray(
        label=r":math:`h_e^{rev}`",
        default=numpy.array([45.0]),
        domain=Range(lo=0.0, hi=50.0, step=5.00),
        doc="""Excitatory population, cell reversal potential [mV]""")

    h_i_rev = NArray(
        label=r":math:`h_i^{rev}`",
        default=numpy.array([-90.0]),
        domain=Range(lo=-90.0, hi=-50.0, step=10.0),
        doc="""Inhibitory population, cell reversal potential [mV]""")

    p_ee = NArray(
        label=":math:`p_{ee}`",
        default=numpy.array([1.1]),
        domain=Range(lo=1.0, hi=1.8, step=0.1),
        doc="""Exogenous (subcortical) spike input to exc population [ms]-1 [kHz]. 
               This could be replaced by a noise term""")

    p_ie = NArray(
        label=":math:`p_{ie}`",
        default=numpy.array([1.6]),
        domain=Range(lo=1.0, hi=1.8, step=0.1),
        doc="""Exogenous (subcortical) spike input to exc population [ms]-1 [kHz]. 
               This could be replaced by a noise term""")

    p_ei = NArray(
        label=":math:`p_{ei}`",
        default=numpy.array([1.6]),
        domain=Range(lo=1.0, hi=1.8, step=0.1),
        doc="""Exogenous (subcortical) spike input to inh population [ms]-1 [kHz]. 
               This could be replaced by a noise term""")

    p_ii = NArray(
        label=":math:`p_{ii}`",
        default=numpy.array([1.1]),
        domain=Range(lo=1.0, hi=1.8, step=0.1),
        doc="""Exogenous (subcortical) spike input to inh population [ms]-1 [kHz]. 
               This could be replaced by a noise term""")

    A_ee = NArray(
        label=r":math:`\alpha_{ee}`",
        default=numpy.array([0.04]),
        domain=Range(lo=0.02, hi=0.06, step=0.01),
        doc="""Characteristic cortico-cortical inverse length scale [mm]-1. Original: 0.4 cm-1""")

    A_ei = NArray(
        label=r":math:`\alpha_{ei}`",
        default=numpy.array([0.065]),
        domain=Range(lo=0.02, hi=0.08, step=0.01),
        doc="""Characteristic cortico-cortical inverse length scale [mm]-1. Original: 0.4 cm-1""")

    gamma_e = NArray(
        label=r":math:`\gamma_e`",
        default=numpy.array([0.3]),
        domain=Range(lo=0.1, hi=0.4, step=0.01),
        doc="""Neurotransmitter rate constant"for EPSP [ms]-1""")

    gamma_i = NArray(
        label=r":math:`\gamma_i`",
        default=numpy.array([0.065]),
        domain=Range(lo=0.005, hi=0.1, step=0.005),
        doc="""Neurotransmitter rate constant"for IPSP [ms]-1""")

    G_e = NArray(
        label=":math:`G_e`",
        default=numpy.array([0.18]),
        domain=Range(lo=0.1, hi=0.5, step=0.01),
        doc="""peak ampplitude of EPSP [mV]""")

    G_i = NArray(
        label=":math:`G_i`",
        default=numpy.array([0.37]),
        domain=Range(lo=0.1, hi=0.5, step=0.01),
        doc="""peak ampplitude of IPSP [mV]""")

    N_b_ee = NArray(
        label=":math:`N_{ee}^{\beta}`",
        default=numpy.array([3034.0]),
        domain=Range(lo=3000., hi=3050., step=10.0),
        doc="""Total number of local exc to exc synaptic connections.""")

    N_b_ei = NArray(
        label=r":math:`N_{ei}^{\beta}`",
        default=numpy.array([3034.0]),
        domain=Range(lo=3000., hi=3050., step=10.0),
        doc="""Total number of local exc to inh synaptic connections.""")

    N_b_ie = NArray(
        label=r":math:`N_{ie}^{\beta}`",
        default=numpy.array([536.0]),
        domain=Range(lo=500., hi=550., step=1.0),
        doc="""Total number of local inh to exc synaptic connections.""")

    N_b_ii = NArray(
        label=r":math:`N_{ii}^{\beta}`",
        default=numpy.array([536.0]),
        domain=Range(lo=500., hi=550., step=1.0),
        doc="""Total number of local inh to inh synaptic connections.""")

    N_a_ee = NArray(
        label=r":math:`N_{ee}^{\alpha}`",
        default=numpy.array([4000.0]),
        domain=Range(lo=3000., hi=5000., step=10.0),
        doc="""Total number of synaptic connections from distant exc populations""")

    N_a_ei = NArray(
        label=r":math:`N_{ei}^{\alpha}`",
        default=numpy.array([2000.0]),
        domain=Range(lo=1000., hi=3000., step=1.0),
        doc="""Total number of synaptic connections from distant exc populations""")

    theta_e = NArray(
        label=r":math:`\theta_e`",
        default=numpy.array([-60.0]),
        domain=Range(lo=-90.0, hi=-40.0, step=5.0),
        doc="""inflection point voltage for sigmoid function [mV]""")

    theta_i = NArray(
        label=r":math:`\theta_i`",
        default=numpy.array([-60.0]),
        domain=Range(lo=-90.0, hi=-40.0, step=5.0),
        doc="""inflection point voltage for sigmoid function [mV]""")

    g_e = NArray(
        label=":math:`g_e`",
        default=numpy.array([0.28]),
        domain=Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Sigmoid slope at inflection point exc population [mV]-1""")

    g_i = NArray(
        label=":math:`g_i`",
        default=numpy.array([0.14]),
        domain=Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Sigmoid slope at inflection point inh population [mV]-1""")

    lambd = NArray(
        label=":math:`\lambda`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Anesthetic effects""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        {
            "he": numpy.array([-90.0, 70.0]),
            "hi": numpy.array([-90.0, 70.0])
        },
        label="State Variable ranges [lo, hi]",
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current 
        parameters, it is used as a mechanism for bounding random inital 
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("he", "hi"),
        default=("he",),
        doc="""This represents the default state-variables of this Model to be
                                    monitored. It can be overridden for each Monitor if desired. The 
                                    corresponding state-variable indices for this model are :math:`E = 0`
                                    and :math:`I = 1`.""")

    state_variables = ["he", "hi"]
    _nvar = 2
    cvar = numpy.array([0, 1], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""

        TODO:  include equations here and see how to add the local connectivity or the 
        the laplacian oeprator. This model is the one used in Bojak 2011.

        """

        he = state_variables[0, :]
        hi = state_variables[1, :]

        # long-range coupling - phi_e -> he / phi_e -> h_i
        c_0 = coupling[0, :]

        # short-range (local) coupling - local coupling functions should be different for exc and inh
        # lc_0 = local_coupling * he
        # lc_1 = local_coupling * hi

        # 
        psi_ee = (self.h_e_rev - he) / abs(self.h_e_rev - self.h_e_rest) # usually > 0
        psi_ei = (self.h_e_rev - hi) / abs(self.h_e_rev - self.h_i_rest) # usually > 0
        psi_ie = (self.h_i_rev - he) / abs(self.h_i_rev - self.h_e_rest) # usually < 0
        psi_ii = (self.h_i_rev - hi) / abs(self.h_i_rev - self.h_i_rest) # usually < 0

        S_e = 1.0 / (1.0 + numpy.exp(-self.g_e * (he + c_0 - self.theta_e)))
        S_i = 1.0 / (1.0 + numpy.exp(-self.g_i * (hi + c_0 - self.theta_i)))

        F1 = ((self.h_e_rest - he) + psi_ee * ((self.N_a_ee + self.N_b_ee) * S_e  + self.p_ee) * (self.G_e/self.gamma_e) +\
                                  self.lambd * psi_ie * (self.N_b_ie * S_i + self.p_ie) * (self.G_i/self.gamma_i)) / self.tau_e

        F2 = ((self.h_i_rest - hi) + psi_ei * ((self.N_a_ei + self.N_b_ei) * S_e  + self.p_ei) * (self.G_e/self.gamma_e) +\
                                  self.lambd * psi_ii * (self.N_b_ii * S_i + self.p_ii) * (self.G_i/self.gamma_i)) / self.tau_i

        dhe = F1
        dhi = F2

        derivative = numpy.array([dhe, dhi])

        return derivative
