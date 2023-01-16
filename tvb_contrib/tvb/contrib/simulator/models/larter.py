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
A contributed model: Larter

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>
.. moduleauthor:: Gaurav Malhotra <Gaurav@tvb.invalid>
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import numpy

from tvb.simulator.common import get_logger
from tvb.basic.neotraits.api import NArray, Range, Final, List
import tvb.simulator.models as models

LOG = get_logger(__name__)


class Larter(models.Model):
    """
    A modified Morris-Lecar model that includes a third equation which simulates
    the effect of a population of inhibitory interneurons synapsing on
    the pyramidal cells.
    
    .. [Larteretal_1999] Larter et.al. *A coupled ordinary differential equation
        lattice model for the simulation of epileptic seizures.* Chaos. 9(3):
        795, 1999.
    
    .. [Breaksetal_2003] M. J. Breakspear et.al. *Modulation of excitatory
        synaptic coupling facilitates synchronization and complex dynamics in a
        biophysical model of neuronal dynamics.* Network: Computation in Neural
        Systems 14: 703-732, 2003.
    
    Equations are taken from [Larteretal_1999]_. Parameter values are taken from
    Table I, page 799.
    
    Regarding the choice of coupling: the three biophysically reasonable
    mechanisms (diffusive; delay line axonal/ synaptic or glial; and
    extracellular current flow) are dependent of K++. This is reflected in the
    potassium equilibrium potential (:math:`V_{K}`). Thus, this variable is
    chosen as the coupling element between nodes.
    
        .. figure :: img/Larter_01_mode_0_pplane.svg
            :alt: Larter phase plane (V, W)
            
            The (:math:`V`, :math:`W`) phase-plane for the Larter model.

    .. automethod:: Larter.dfun
    
    """

    # Define traited attributes for this model, these represent possible kwargs.
    gCa = NArray(
        label=":math:`g_{Ca}`",
        default=numpy.array([1.1]),
        domain=Range(lo=0.9, hi=1.5, step=0.01),
        doc="""Conductance of population of Ca++ channels""")

    gK = NArray(
        label=":math:`g_K`",
        default=numpy.array([2.0]),
        domain=Range(lo=1.5, hi=2.5, step=0.01),
        doc="""Conductance of population of K channels""")

    gL = NArray(
        label=":math:`g_L`",
        default=numpy.array([0.5]),
        domain=Range(lo=0.25, hi=0.75, step=0.01),
        doc="""Conductance of population of leak channels""")

    phi = NArray(
        label=":math:`\\phi`",
        default=numpy.array([0.7]),
        domain=Range(lo=0.35, hi=1.05, step=0.01),
        doc="""Temperature scaling factor""")

    V1 = NArray(
        label=":math:`V_1`",
        default=numpy.array([-0.01]),
        domain=Range(lo=-0.1, hi=0.1, step=0.01),
        doc="""Threshold value for :math:`M_{\\infty}`""")

    V2 = NArray(
        label=":math:`V_2`",
        default=numpy.array([0.15]),
        domain=Range(lo=0.01, hi=1.0, step=0.01),
        doc="""Steepness parameter for :math:`M_{\\infty}`""")

    V3 = NArray(
        label=":math:`V_3`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Threshold value for :math:`W_{\\infty}`""")

    V4 = NArray(
        label=":math:`V_4`",
        default=numpy.array([0.3]),
        domain=Range(lo=0.01, hi=1.0, step=0.01),
        doc="""Steepness parameter for :math:`W_{\\infty}`""")

    V5 = NArray(
        label=":math:`V_5`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Threshold value for :math:`a_{exc}`""")

    V6 = NArray(
        label=":math:`V_6`",
        default=numpy.array([0.6]),
        domain=Range(lo=0.01, hi=1.0, step=0.01),
        doc="""Steepness parameter for a_exc and :math:`a_{inh}`""")

    V7 = NArray(
        label=":math:`V_7`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Threshold value for :math:`a_{inh}`""")

    VK = NArray(
        label=":math:`V_K`",
        default=numpy.array([-0.7]),
        domain=Range(lo=-0.8, hi=1.0, step=0.01),
        doc="""K Nernst potential""")

    VL = NArray(
        label=":math:`V_L`",
        default=numpy.array([-0.5]),
        domain=Range(lo=-0.75, hi=-0.25, step=0.01),
        doc="""Nernst potential leak channels""")

    tau_K = NArray(
        label=":math:`\\tau_K`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.01),
        doc="""Time constant for K relaxation time""")

    a_exc = NArray(
        label=":math:`a_{exc}`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.7, hi=1.3, step=0.01),
        doc="""strength of excitatory synapse""")

    a_inh = NArray(
        label=":math:`a_{ie}`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.7, hi=1.3, step=0.01),
        doc="""strength of inhibitory synapse""")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([0.1]),
        domain=Range(lo=0.05, hi=0.15, step=0.01),
        doc="""Time constant scaling factor""")

    c = NArray(
        label=":math:`c`",
        default=numpy.array([0.165]),
        domain=Range(lo=0.0, hi=0.2, step=0.01),
        doc="""strength of feedforward inhibition""")

    Iext = NArray(
        label=":math:`I_{ext}`",
        default=numpy.array([0.3]),
        domain=Range(lo=0.15, hi=0.45, step=0.01),
        doc="""Subcortical input strength. It represents a non-specific
       excitation of both the excitatory and inhibitory populations.""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        {
            "V": numpy.array([-0.3, 0.1]),
            "W": numpy.array([0.0, 0.6]),
            "Z": numpy.array([-0.02, 0.08])
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
        choices=("V", "W", "Z"),
        default=("V", "W", "Z"),
        doc="""This represents the default state-variables of this Model to be
        monitored. It can be overridden for each Monitor if desired. The 
        corresponding state-variable indices for this model are :math:`V = 0`,
        :math:`W = 1`, and :math:`Z = 2`.""")

    state_variables = ["V", "W", "Z"]
    _nvar = 3
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        """
        .. math::
             \\dot{V} &= - g_L \\, (V - V_L) - g_K\\, Z \\, (V - V_K) -
                 g_{Ca} \\, m_{\\infty} \\, (V - 1) + I - \\alpha_{inh}\\,Z  \\\\
             \\dot{W} &= \\frac{\\phi \\, (w_{\\infty} - W)}{\\tau_w} \\\\
             \\dot{Z}(t) &= b ( c \\, I_{ext} + \\alpha_{exc} \\,V ) \\\\
             m_{\\infty} &= 0.5 \\, \\left(1 + \\tanh\\left(\\frac{V - V_1}{V_2}\\right)\\right) \\\\
             w_{\\infty} &= 0.5 \\, \\left(1 + \\tanh\\left(\\frac{V - V_3}{V_4}\\right)\\right)\\\\
             tau_{w} &= \\left[ \\cosh\\left(\\frac{V - V_3}{2 \\,V_4}\\right) \\right]^{-1} \\\\
             \\alpha_{exc} &= a_{exc} \\,\\left(1 + \\tanh\\left(\\frac{V - V_5}{V_6}\\right)\\right)\\\\
             \\alpha_{inh} &= a_{inh} \\,\\left(1 + \\tanh\\left(\\frac{V - V_7}{V_6}\\right)\\right)
             
        See Eqs (1)-(8) in [Larteretal_1999]_
        """
        ##--------------------- As in Larter 1999 ----------------------------##
        V = state_variables[0, :]
        W = state_variables[1, :]
        Z = state_variables[2, :]

        c_0 = coupling[0, :]

        M_inf = 0.5 * (1 + numpy.tanh((V - self.V1) / self.V2))
        W_inf = 0.5 * (1 + numpy.tanh((V - self.V3) / self.V4))
        tau_Winv = numpy.cosh((V - self.V3) / (2 * self.V4))
        alpha_exc = self.a_exc * (1 + numpy.tanh((V - self.V5) / self.V6))
        alpha_inh = self.a_inh * (1 + numpy.tanh((V - self.V7) / self.V6))

        # import pdb; pdb.set_trace()
        dV = (local_coupling * V - alpha_inh * Z -
              self.gL * (V - self.VL) -
              self.gCa * M_inf * (V - 1) -
              self.gK * W * (V - self.VK + c_0) + self.Iext)

        dW = self.phi * tau_Winv * (W_inf - W)

        dZ = self.b * ((self.c * self.Iext) + (alpha_exc * V))

        derivative = numpy.array([dV, dW, dZ])

        return derivative
