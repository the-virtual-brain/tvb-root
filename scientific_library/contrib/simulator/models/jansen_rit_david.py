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
A contributed model: The Jansen and Rit model as presented in (David et al., 2005)

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


class JansenRitDavid(models.Model):
    """
    The Jansen and Rit models as studied by David et al., 2005
    #TODO: finish this model

    """

    _ui_name = "Jansen-Rit (David et al., 2005)"


    #Define traited attributes for this model, these represent possible kwargs.
    He = arrays.FloatArray(
        label=":math:`He`",
        default=numpy.array([3.25]),
        range=basic.Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""",
        order=1)

    Hi = arrays.FloatArray(
        label=":math:`B`",
        default=numpy.array([29.3]),
        range=basic.Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""",
        order=2)

    tau_e = arrays.FloatArray(
        label=":math:`a`",
        default=numpy.array([0.1]),
        range=basic.Range(lo=0.05, hi=0.15, step=0.01),
        doc="""time constant""",
        order=3)

    tau_i = arrays.FloatArray(
        label=":math:`b`",
        default=numpy.array([0.15]),
        range=basic.Range(lo=0.025, hi=0.075, step=0.005),
        doc="""time constant""",
        order=4)

    eo = arrays.FloatArray(
        label=":math:`v_0`",
        default=numpy.array([0.0025]),
        range=basic.Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV].""",
        order=5)


    r = arrays.FloatArray(
        label=":math:`r`",
        default=numpy.array([0.56]),
        range=basic.Range(lo=0.28, hi=0.84, step=0.01),
        doc="""Steepness of the sigmoidal transformation [mV^-1].""",
        order=7)


    gamma_1 = arrays.FloatArray(
        label=r":math:`\alpha_1`",
        default=numpy.array([50.0]),
        range=basic.Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback
        excitatory loop.""",
        order=9)

    gamma_2 = arrays.FloatArray(
        label=r":math:`\alpha_2`",
        default=numpy.array([40.]),
        range=basic.Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback
        excitatory loop.""",
        order=10)

    gamma_3 = arrays.FloatArray(
        label=r":math:`\alpha_3`",
        default=numpy.array([12.]),
        range=basic.Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback
        excitatory loop.""",
        order=11)

    gamma_4 = arrays.FloatArray(
        label=r":math:`\alpha_4`",
        default=numpy.array([12.]),
        range=basic.Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback
        inhibitory loop.""",
        order=12)

    #Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = basic.Dict(
        label="State Variable ranges [lo, hi]",
        default={"x0": numpy.array([-1.0, 1.0]),
                 "x1": numpy.array([-1.0, 1.0]),
                 "x2": numpy.array([-5.0, 5.0]),
                 "x3": numpy.array([-6.0, 6.0]),
                 "x4": numpy.array([-2.0, 2.0]),
                 "x5": numpy.array([-5.0, 5.0]),
                 "x6": numpy.array([-5.0, 5.0]),
                 "x7": numpy.array([-5.0, 5.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current 
        parameters, it is used as a mechanism for bounding random inital 
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""",
        order=16)

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"],
        default=["x0", "x1", "x2", "x3"],
        select_multiple=True,
        doc="""This represents the default state-variables of this Model to be
                                    monitored. It can be overridden for each Monitor if desired. The 
                                    corresponding state-variable indices for this model are :math:`y0 = 0`,
                                    :math:`y1 = 1`, :math:`y2 = 2`, :math:`y3 = 3`, :math:`y4 = 4`, and
                                    :math:`y5 = 5`""",
        order=17)



    def __init__(self, **kwargs):
        """
        Initialise parameters for the Jansen Rit column, [JR_1995]_.

        """
        LOG.info("%s: initing..." % str(self))
        super(JansenRitDavid, self).__init__(**kwargs)

        #self._state_variables = ["y0", "y1", "y2", "y3", "y4", "y5"]
        self._nvar = 8

        self.cvar = numpy.array([1,2], dtype=numpy.int32)



        LOG.debug('%s: inited.' % repr(self))


    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The dynamic equations were taken from:

        TODO: add equations and finish the model ...
            

        """

        magic_exp_number = 709
        AF = 0.1
        AB = 0.2
        AL = 0.05
        x0 = state_variables[0, :]
        x1 = state_variables[1, :]
        x2 = state_variables[2, :]
        x3 = state_variables[3, :]
        x4 = state_variables[4, :]
        x5 = state_variables[5, :]
        x6 = state_variables[6, :]
        x7 = state_variables[7, :]

        y   = x1 - x2
        #delayed activity x1 - x2
        c_12   = coupling[0, :] -  coupling[1, :] 
        c_12_f = AF * ((2 * self.eo) / (1 + numpy.exp(self.r * c_12)) - self.eo)
        c_12_b = AB * ((2 * self.eo) / (1 + numpy.exp(self.r * c_12)) - self.eo)
        c_12_l = AL * ((2 * self.eo) / (1 + numpy.exp(self.r * c_12)) - self.eo)

        lc_f  = (local_coupling *  y)  * AF
        lc_l  = (local_coupling *  y)  * AL
        lc_b  = (local_coupling *  y)  * AB

        S_y  = (2 * self.eo) / (1 + numpy.exp(self.r * y))  - self.eo
        S_x0 = (2 * self.eo) / (1 + numpy.exp(self.r * x0)) - self.eo
        S_x6 = (2 * self.eo) / (1 + numpy.exp(self.r * x6)) - self.eo

        # NOTE: for local couplings
        # 0:3 pyramidal cells
        # 1:4 excitatory interneurons
        # 2:5 inhibitory interneurons
        # 3:7 

        dx0 = x3
        dx3 = self.He / self.tau_e * (c_12_f + c_12_l + self.gamma_1 * S_y) - (2 * x3) / self.tau_e - (x0 / self.tau_e**2)
        dx1 = x4
        dx4 = self.He / self.tau_e * (c_12_b + c_12_l + self.gamma_2 * S_x0) - (2 * x4) / self.tau_e - (x1 / self.tau_e**2)
        dx2 = x5
        dx5 = self.Hi / self.tau_i * (self.gamma_4 * S_x6) - (2 * x5) / self.tau_i - (x2 / self.tau_i**2)
        dx6 = x7
        dx7 = self.He / self.tau_e * (c_12_b + c_12_l + self.gamma_3 * S_y) - (2 * x7) / self.tau_e - (x6 / self.tau_e**2)

        derivative = numpy.array([dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7])

        return derivative
