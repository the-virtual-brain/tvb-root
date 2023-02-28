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
A contributed model: The Jansen and Rit model as presented in (David et al., 2005)

.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import numpy

from tvb.simulator.common import get_logger
from tvb.basic.neotraits.api import NArray, Range, Final, List
import tvb.simulator.models as models

LOG = get_logger(__name__)


class JansenRitDavid(models.Model):
    """
    The Jansen and Rit models as studied by David et al., 2005
    #TODO: finish this model
    """

    # Define traited attributes for this model, these represent possible kwargs.
    He = NArray(
        label=":math:`He`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""")

    Hi = NArray(
        label=":math:`B`",
        default=numpy.array([29.3]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""")

    tau_e = NArray(
        label=":math:`a`",
        default=numpy.array([0.1]),
        domain=Range(lo=0.05, hi=0.15, step=0.01),
        doc="""time constant""")

    tau_i = NArray(
        label=":math:`b`",
        default=numpy.array([0.15]),
        domain=Range(lo=0.025, hi=0.075, step=0.005),
        doc="""time constant""")

    eo = NArray(
        label=":math:`v_0`",
        default=numpy.array([0.0025]),
        domain=Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV].""")

    r = NArray(
        label=":math:`r`",
        default=numpy.array([0.56]),
        domain=Range(lo=0.28, hi=0.84, step=0.01),
        doc="""Steepness of the sigmoidal transformation [mV^-1].""")

    gamma_1 = NArray(
        label=r":math:`\alpha_1`",
        default=numpy.array([50.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback
        excitatory loop.""")

    gamma_2 = NArray(
        label=r":math:`\alpha_2`",
        default=numpy.array([40.]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback
        excitatory loop.""")

    gamma_3 = NArray(
        label=r":math:`\alpha_3`",
        default=numpy.array([12.]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback
        excitatory loop.""")

    gamma_4 = NArray(
        label=r":math:`\alpha_4`",
        default=numpy.array([12.]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback
        inhibitory loop.""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        {
            "x0": numpy.array([-1.0, 1.0]),
            "x1": numpy.array([-1.0, 1.0]),
            "x2": numpy.array([-5.0, 5.0]),
            "x3": numpy.array([-6.0, 6.0]),
            "x4": numpy.array([-2.0, 2.0]),
            "x5": numpy.array([-5.0, 5.0]),
            "x6": numpy.array([-5.0, 5.0]),
            "x7": numpy.array([-5.0, 5.0])
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
        choices=("x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"),
        default=("x0", "x1", "x2", "x3"),
        doc="""This represents the default state-variables of this Model to be
                                    monitored. It can be overridden for each Monitor if desired. The 
                                    corresponding state-variable indices for this model are :math:`y0 = 0`,
                                    :math:`y1 = 1`, :math:`y2 = 2`, :math:`y3 = 3`, :math:`y4 = 4`, and
                                    :math:`y5 = 5`""")

    state_variables = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    _nvar = 8
    cvar = numpy.array([1, 2], dtype=numpy.int32)

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
        # delayed activity x1 - x2
        c_12   = coupling[0, :] -  coupling[1, :]
        c_12_f = AF * ((2 * self.eo) / (1 + numpy.exp(self.r * c_12)) - self.eo)
        c_12_b = AB * ((2 * self.eo) / (1 + numpy.exp(self.r * c_12)) - self.eo)
        c_12_l = AL * ((2 * self.eo) / (1 + numpy.exp(self.r * c_12)) - self.eo)

        lc_f = (local_coupling * y) * AF
        lc_l = (local_coupling * y) * AL
        lc_b = (local_coupling * y) * AB

        S_y = (2 * self.eo) / (1 + numpy.exp(self.r * y)) - self.eo
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
