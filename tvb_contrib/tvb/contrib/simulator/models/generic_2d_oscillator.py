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
A abstract 2d oscillator model.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import numpy

from tvb.simulator.common import get_logger
from tvb.basic.neotraits.api import NArray, Range, Final, List
import tvb.simulator.models as models

LOG = get_logger(__name__)


class Generic2dOscillator(models.Model):
    """
    The Generic2dOscillator model is ...
    
    .. [FH_1961] FitzHugh, R., *Impulses and physiological states in theoretical
        models of nerve membrane*, Biophysical Journal 1: 445, 1961. 
    
    .. [Nagumo_1962] Nagumo et.al, *An Active Pulse Transmission Line Simulating
        Nerve Axon*, Proceedings of the IRE 50: 2061, 1962.
    
    See also, http://www.scholarpedia.org/article/FitzHugh-Nagumo_model
    
    The models (:math:`V`, :math:`W`) phase-plane, including a representation of
    the vector field as well as its nullclines, using default parameters, can be
    seen below:
        
        .. _phase-plane-FHN:
        .. figure :: img/Generic2dOscillator_01_mode_0_pplane.svg
            :alt: Fitzhugh-Nagumo phase plane (V, W)
            
            The (:math:`V`, :math:`W`) phase-plane for the Fitzhugh-Nagumo 
            model.
            
    .. #Currently there seems to be a clash betwen traits and autodoc, autodoc
    .. #can't find the methods of the class, the class specific names below get
    .. #us around this...
    .. automethod:: Generic2dOscillator.__init__
    .. automethod:: Generic2dOscillator.dfun
    
    """

    # Define traited attributes for this model, these represent possible kwargs.
    tau = NArray(
        label=":math:`\\tau`",
        default=numpy.array([1.25]),
        domain=Range(lo=0.01, hi=5.0, step=0.01),
        doc="""A time-scale separation between the fast, :math:`V`, and slow,
            :math:`W`, state-variables of the model.""")

    a = NArray(
        label=":math:`a`",
        default=numpy.array([1.05]),
        domain=Range(lo=-1.0, hi=1.5, step=0.01),
        doc="""ratio a/b gives W-nullcline slope""")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([0.2]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""dimensionless parameter""")

    omega = NArray(
        label=":math:`\\omega`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""dimensionless parameter""")

    upsilon = NArray(
        label=":math:`\\upsilon`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""dimensionless parameter""")

    gamma = NArray(
        label=":math:`\\gamma`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""dimensionless parameter""")

    eta = NArray(
        label=":math:`\\eta`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""ratio :math:`\\eta/b` gives W-nullcline intersect(V=0)""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors.",
        choices=("V", "W"),
        default="V",
        doc="""This represents the default state-variables of this Model to be
        monitored. It can be overridden for each Monitor if desired.""")

    # Informational attribute, used for phase-plane and initial()
    state_variable_range = Final(
        {
            "V": numpy.array([-2.0, 4.0]),
            "W": numpy.array([-6.0, 6.0])
        },
        label="State Variable ranges [lo, hi]",
        doc="""The values for each state-variable should be set to encompass
            the expected dynamic range of that state-variable for the current 
            parameters, it is used as a mechanism for bounding random inital 
            conditions when the simulation isn't started from an explicit
            history, it is also provides the default range of phase-plane plots.""")

    state_variables = ["V", "W"]
    _nvar = 2
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        """
        The fast, :math:`V`, and slow, :math:`W`, state variables are typically
        considered to represent a membrane potential and recovery variable,
        respectively. Based on equations 1 and 2 of [FH_1961]_, but relabelling
        c as :math:`\\tau` due to its interpretation as a time-scale separation,
        and adding parameters :math:`\\upsilon`, :math:`\\omega`, :math:`\\eta`,
        and :math:`\\gamma`, for flexibility, here we implement:
            
            .. math::
                \\dot{V} &= \\tau (\\omega \\, W + \\upsilon \\, V - 
                                   \\gamma \\,  \\frac{V^3}{3} + I) \\\\
                \\dot{W} &= -(\\eta \\, V - a + b \\, W) / \\tau
                
        where external currents :math:`I` provide the entry point for local and
        long-range connectivity.
        
        For strict consistency with [FH_1961]_, parameters :math:`\\upsilon`, 
        :math:`\\omega`, :math:`\\eta`, and :math:`\\gamma` should be set to 
        1.0, with :math:`a`, :math:`b`, and :math:`\\tau` set in the range 
        defined by equation 3 of [FH_1961]_:
            
            .. math::
                0 \\le b \\le 1 \\\\
                1 - 2 b / 3 \\le a \\le 1 \\\\
                \\tau^2 \\ge b
            
        The default state of these equations can be seen in the
        :ref:`Fitzhugh-Nagumo phase-plane <phase-plane-FHN>`.
        
        """

        V = state_variables[0, :]
        W = state_variables[1, :]

        # [State_variables, nodes]
        c_0 = coupling[0, :]

        dV = self.tau * (self.omega * W + self.upsilon * V -
                         self.gamma * V**3.0 / 3.0 +
                         c_0 + local_coupling * V)

        dW = (self.a - self.eta * V - self.b * W) / self.tau
        derivative = numpy.array([dV, dW])

        return derivative
