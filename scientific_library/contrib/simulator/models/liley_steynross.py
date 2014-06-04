# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
A contributed model: The Liley model as presented in Steyn-Ross et al., 1999

.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

# Third party python libraries
import numpy

#The Virtual Brain
from tvb.simulator.common import get_logger
LOG = get_logger(__name__)

import tvb.datatypes.arrays as arrays
import tvb.basic.traits.types_basic as basic 
import tvb.simulator.models as models


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

    _ui_name = "Liley-SteynRoss model"
    #ui_configurable_parameters = []

    #Define traited attributes for this model, these represent possible kwargs.

    tau_e = arrays.FloatArray(
        label=r":math:`\tau_e`",
        default=numpy.array([40.0]),
        range=basic.Range(lo=5.0, hi=50.0, step=1.00),
        doc="""Excitatory population, membrane time-constant [ms]""",
        order=1)

    tau_i = arrays.FloatArray(
        label=r":math:`\tau_i`",
        default=numpy.array([40.0]),
        range=basic.Range(lo=5.0, hi=50.0, step=1.0),
        doc="""Inhibitory population, membrane time-constant [ms]""",
        order=2)

    h_e_rest = arrays.FloatArray(
        label=r":math:`h_e^{rest}`",
        default=numpy.array([-70.0]),
        range=basic.Range(lo=-90.0, hi=-50.0, step=10.00),
        doc="""Excitatory population, cell resting potential [mV]""",
        order=3)

    h_i_rest = arrays.FloatArray(
        label=r":math:`h_i^{rest}`",
        default=numpy.array([-70.0]),
        range=basic.Range(lo=-90.0, hi=-50.0, step=10.0),
        doc="""Inhibitory population, cell resting potential [mV]""",
        order=4)

    h_e_rev = arrays.FloatArray(
        label=r":math:`h_e^{rev}`",
        default=numpy.array([45.0]),
        range=basic.Range(lo=0.0, hi=50.0, step=5.00),
        doc="""Excitatory population, cell reversal potential [mV]""",
        order=42)

    h_i_rev = arrays.FloatArray(
        label=r":math:`h_i^{rev}`",
        default=numpy.array([-90.0]),
        range=basic.Range(lo=-90.0, hi=-50.0, step=10.0),
        doc="""Inhibitory population, cell reversal potential [mV]""",
        order=43)

    p_ee = arrays.FloatArray(
        label=":math:`p_{ee}`",
        default=numpy.array([1.1]),
        range=basic.Range(lo=1.0, hi=1.8, step=0.1),
        doc="""Exogenous (subcortical) spike input to exc population [ms]-1 [kHz]. 
               This could be replaced by a noise term""",
        order=5)

    p_ie = arrays.FloatArray(
        label=":math:`p_{ie}`",
        default=numpy.array([1.6]),
        range=basic.Range(lo=1.0, hi=1.8, step=0.1),
        doc="""Exogenous (subcortical) spike input to exc population [ms]-1 [kHz]. 
               This could be replaced by a noise term""",
        order=6)

    p_ei = arrays.FloatArray(
        label=":math:`p_{ei}`",
        default=numpy.array([1.6]),
        range=basic.Range(lo=1.0, hi=1.8, step=0.1),
        doc="""Exogenous (subcortical) spike input to inh population [ms]-1 [kHz]. 
               This could be replaced by a noise term""",
        order=7)

    p_ii = arrays.FloatArray(
        label=":math:`p_{ii}`",
        default=numpy.array([1.1]),
        range=basic.Range(lo=1.0, hi=1.8, step=0.1),
        doc="""Exogenous (subcortical) spike input to inh population [ms]-1 [kHz]. 
               This could be replaced by a noise term""",
        order=8)

    A_ee = arrays.FloatArray(
        label=r":math:`\alpha_{ee}`",
        default=numpy.array([0.04]),
        range=basic.Range(lo=0.02, hi=0.06, step=0.01),
        doc="""Characteristic cortico-cortical inverse length scale [mm]-1. Original: 0.4 cm-1""",
        order=9)

    A_ei = arrays.FloatArray(
        label=r":math:`\alpha_{ei}`",
        default=numpy.array([0.065]),
        range=basic.Range(lo=0.02, hi=0.08, step=0.01),
        doc="""Characteristic cortico-cortical inverse length scale [mm]-1. Original: 0.4 cm-1""",
        order=10)

    gamma_e = arrays.FloatArray(
        label=r":math:`\gamma_e`",
        default=numpy.array([0.3]),
        range=basic.Range(lo=0.1, hi=0.4, step=0.01),
        doc="""Neurotransmitter rate constant"for EPSP [ms]-1""",
        order=11)

    gamma_i = arrays.FloatArray(
        label=r":math:`\gamma_i`",
        default=numpy.array([0.065]),
        range=basic.Range(lo=0.005, hi=0.1, step=0.005),
        doc="""Neurotransmitter rate constant"for IPSP [ms]-1""",
        order=12)

    G_e = arrays.FloatArray(
        label=":math:`G_e`",
        default=numpy.array([0.18]),
        range=basic.Range(lo=0.1, hi=0.5, step=0.01),
        doc="""peak ampplitude of EPSP [mV]""",
        order=13)

    G_i = arrays.FloatArray(
        label=":math:`G_i`",
        default=numpy.array([0.37]),
        range=basic.Range(lo=0.1, hi=0.5, step=0.01),
        doc="""peak ampplitude of IPSP [mV]""",
        order=14)

    N_b_ee = arrays.FloatArray(
        label=":math:`N_{ee}^{\beta}`",
        default=numpy.array([3034.0]),
        range=basic.Range(lo=3000., hi=3050., step=10.0),
        doc="""Total number of local exc to exc synaptic connections.""",
        order=15)

    N_b_ei = arrays.FloatArray(
        label=r":math:`N_{ei}^{\beta}`",
        default=numpy.array([3034.0]),
        range=basic.Range(lo=3000., hi=3050., step=10.0),
        doc="""Total number of local exc to inh synaptic connections.""",
        order=16)

    N_b_ie = arrays.FloatArray(
        label=r":math:`N_{ie}^{\beta}`",
        default=numpy.array([536.0]),
        range=basic.Range(lo=500., hi=550., step=1.0),
        doc="""Total number of local inh to exc synaptic connections.""",
        order=17)

    N_b_ii = arrays.FloatArray(
        label=r":math:`N_{ii}^{\beta}`",
        default=numpy.array([536.0]),
        range=basic.Range(lo=500., hi=550., step=1.0),
        doc="""Total number of local inh to inh synaptic connections.""",
        order=18)


    N_a_ee = arrays.FloatArray(
        label=r":math:`N_{ee}^{\alpha}`",
        default=numpy.array([4000.0]),
        range=basic.Range(lo=3000., hi=5000., step=10.0),
        doc="""Total number of synaptic connections from distant exc populations""",
        order=19)

    N_a_ei = arrays.FloatArray(
        label=r":math:`N_{ei}^{\alpha}`",
        default=numpy.array([2000.0]),
        range=basic.Range(lo=1000., hi=3000., step=1.0),
        doc="""Total number of synaptic connections from distant exc populations""",
        order=20)

    theta_e = arrays.FloatArray(
        label=r":math:`\theta_e`",
        default=numpy.array([-60.0]),
        range=basic.Range(lo=-90.0, hi=-40.0, step=5.0),
        doc="""inflection point voltage for sigmoid function [mV]""",
        order=21)

    theta_i = arrays.FloatArray(
        label=r":math:`\theta_i`",
        default=numpy.array([-60.0]),
        range=basic.Range(lo=-90.0, hi=-40.0, step=5.0),
        doc="""inflection point voltage for sigmoid function [mV]""",
        order=22)

    g_e = arrays.FloatArray(
        label=":math:`g_e`",
        default=numpy.array([0.28]),
        range=basic.Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Sigmoid slope at inflection point exc population [mV]-1""",
        order=23)

    g_i = arrays.FloatArray(
        label=":math:`g_i`",
        default=numpy.array([0.14]),
        range=basic.Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Sigmoid slope at inflection point inh population [mV]-1""",
        order=24)

    lambd = arrays.FloatArray(
        label=":math:`\lambda`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Anesthetic effects""",
        order=24)

    #Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = basic.Dict(
        label="State Variable ranges [lo, hi]",
        default={"he": numpy.array([-90.0, 70.0]),
                 "hi": numpy.array([-90.0, 70.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current 
        parameters, it is used as a mechanism for bounding random inital 
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""",
        order=25)

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["he", "hi"],
        default=["he"],
        select_multiple=True,
        doc="""This represents the default state-variables of this Model to be
                                    monitored. It can be overridden for each Monitor if desired. The 
                                    corresponding state-variable indices for this model are :math:`E = 0`
                                    and :math:`I = 1`.""",
        order=26)



    def __init__(self, **kwargs):
        """
        Initialize the Liley Steyn-Ross model's traited attributes, any provided as
        keywords will overide their traited default.

        """
        LOG.info('%s: initing...' % str(self))
        super(LileySteynRoss, self).__init__(**kwargs)
        #self._state_variables = ["E", "I"]
        self._nvar = 2
        self.cvar = numpy.array([0, 1], dtype=numpy.int32)
        LOG.debug('%s: inited.' % repr(self))


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
        #lc_0 = local_coupling * he
        #lc_1 = local_coupling * hi 

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