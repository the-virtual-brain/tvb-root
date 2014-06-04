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
A contributed model: Larter

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>
.. moduleauthor:: Gaurav Malhotra <Gaurav@tvb.invalid>
.. moduleauthor:: Marmaduke Woodman <Marmaduke@tvb.invalid>

"""

# Third party python libraries
import numpy
import numexpr

#The Virtual Brain
from tvb.simulator.common import psutil, get_logger
LOG = get_logger(__name__)

import tvb.datatypes.arrays as arrays
import tvb.basic.traits.types_basic as basic 
import tvb.simulator.models as models



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
    
    .. #Currently there seems to be a clash betwen traits and autodoc, autodoc
    .. #can't find the methods of the class, the class specific names below get
    .. #us around this...
    .. automethod:: Larter.__init__
    .. automethod:: Larter.dfun
    
    """
    
    _ui_name = "Larter"
    ui_configurable_parameters = ['gCa', 'gK', 'gL', 'phi', 'V1', 'V2', 'V3',
                                  'V4', 'V5', 'V6', 'V7', 'VK', 'VL', 'tau_K',
                                  'a_exc', 'a_inh', 'b', 'c', 'Iext']
    
    #Define traited attributes for this model, these represent possible kwargs.
    gCa = arrays.FloatArray(
        label = ":math:`g_{Ca}`",
        default = numpy.array([1.1]),
        range = basic.Range(lo = 0.9, hi = 1.5, step = 0.01),
        doc = """Conductance of population of Ca++ channels""",
        order = 1)
    
    gK = arrays.FloatArray(
        label = ":math:`g_K`",
        default = numpy.array([2.0]),
        range = basic.Range(lo = 1.5, hi= 2.5, step = 0.01),
        doc = """Conductance of population of K channels""",
        order = 2)
    
    gL = arrays.FloatArray(
        label = ":math:`g_L`",
        default = numpy.array([0.5]),
        range = basic.Range(lo = 0.25 , hi = 0.75, step = 0.01),
        doc = """Conductance of population of leak channels""",
        order = 3)
    
    phi = arrays.FloatArray(
        label = ":math:`\\phi`",
        default = numpy.array([0.7]),
        range = basic.Range(lo = 0.35, hi = 1.05, step = 0.01),
        doc = """Temperature scaling factor""",
        order = 4)
    
    V1 = arrays.FloatArray(
        label = ":math:`V_1`",
        default = numpy.array([-0.01]),
        range = basic.Range(lo = -0.1, hi = 0.1, step = 0.01),
        doc = """Threshold value for :math:`M_{\\infty}`""",
        order = 5)
    
    V2 = arrays.FloatArray(
        label = ":math:`V_2`",
        default = numpy.array([0.15]),
        range = basic.Range(lo = 0.01, hi = 1.0, step = 0.01),
        doc = """Steepness parameter for :math:`M_{\\infty}`""",
        order = 6)
    
    V3 = arrays.FloatArray(
        label = ":math:`V_3`",
        default = numpy.array([0.0]),
        range = basic.Range(lo = 0.0, hi = 1.0, step = 0.01),
        doc = """Threshold value for :math:`W_{\\infty}`""",
        order = 7)
    
    V4 = arrays.FloatArray(
        label = ":math:`V_4`",
        default = numpy.array([0.3]),
        range = basic.Range(lo = 0.01, hi = 1.0, step = 0.01),
        doc = """Steepness parameter for :math:`W_{\\infty}`""",
        order = 8)
    
    V5 = arrays.FloatArray(
        label = ":math:`V_5`",
        default = numpy.array([0.0]),
        range = basic.Range(lo = 0.0, hi = 1.0, step = 0.01),
        doc = """Threshold value for :math:`a_{exc}`""",
        order = 9)
    
    V6 = arrays.FloatArray(
        label = ":math:`V_6`",
        default = numpy.array([0.6]),
        range = basic.Range(lo = 0.01, hi = 1.0, step = 0.01),
        doc = """Steepness parameter for a_exc and :math:`a_{inh}`""",
        order = 10)
    
    V7 = arrays.FloatArray(
        label = ":math:`V_7`",
        default = numpy.array([0.0]),
        range = basic.Range(lo = 0.0, hi = 1.0, step = 0.01),
        doc = """Threshold value for :math:`a_{inh}`""",
        order = 11)
    
    VK = arrays.FloatArray(
        label = ":math:`V_K`",
        default = numpy.array([-0.7]),
        range = basic.Range(lo = -0.8, hi = 1.0, step = 0.01),
        doc = """K Nernst potential""",
        order = 12)
    
    VL = arrays.FloatArray(
        label = ":math:`V_L`",
        default = numpy.array([-0.5]),
        range = basic.Range(lo = -0.75, hi = -0.25, step = 0.01),
        doc = """Nernst potential leak channels""",
        order = 13)
    
    tau_K = arrays.FloatArray(
        label = ":math:`\\tau_K`",
        default = numpy.array([1.0]),
        range = basic.Range(lo = 0.5, hi = 1.5, step = 0.01),
        doc = """Time constant for K relaxation time""",
        order = 14)
    
    a_exc = arrays.FloatArray(
        label = ":math:`a_{exc}`",
        default = numpy.array([1.0]),
        range = basic.Range(lo = 0.7, hi = 1.3, step = 0.01),
        doc = """strength of excitatory synapse""",
        order = 15)
    
    a_inh = arrays.FloatArray(
        label = ":math:`a_{ie}`",
        default = numpy.array([1.0]),
        range = basic.Range(lo = 0.7, hi = 1.3, step = 0.01),
        doc = """strength of inhibitory synapse""",
        order = 16)
    
    b = arrays.FloatArray(
        label = ":math:`b`",
        default = numpy.array([0.1]),
        range = basic.Range(lo = 0.05, hi = 0.15, step = 0.01),
        doc = """Time constant scaling factor""",
        order = 17)
    
    c = arrays.FloatArray(
        label = ":math:`c`",
        default = numpy.array([0.165]),
        range = basic.Range(lo = 0.0, hi = 0.2, step = 0.01),
        doc = """strength of feedforward inhibition""",
        order = 18)
    
    Iext = arrays.FloatArray(
       label = ":math:`I_{ext}`",
       default = numpy.array([0.3]),
       range = basic.Range(lo = 0.15, hi = 0.45, step = 0.01),
       doc = """Subcortical input strength. It represents a non-specific
       excitation of both the excitatory and inhibitory populations.""",
        order = 19)
    
    #Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = basic.Dict(
        label = "State Variable ranges [lo, hi]",
        default = {"V": numpy.array([-0.3, 0.1]),
                   "W": numpy.array([0.0, 0.6]),
                   "Z": numpy.array([-0.02, 0.08])},
        doc = """The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""",
        order = 20)
    
    # variables_of_interest = arrays.IntegerArray(
    #     label = "Variables watched by Monitors",
    #     range = basic.Range(lo = 0, hi = 3, step=1),
    #     default = numpy.array([0, 2], dtype=numpy.int32),
    #     doc = """This represents the default state-variables of this Model to be
    #     monitored. It can be overridden for each Monitor if desired. The 
    #     corresponding state-variable indices for this model are :math:`V = 0`,
    #     :math:`W = 1`, and :math:`Z = 2`.""",
    #     order = 21)

    variables_of_interest = basic.Enumerate(
        label = "Variables watched by Monitors",
        options = ["V", "W", "Z"],
        default = ["V", "W", "Z"],
        select_multiple = True,
        doc = """This represents the default state-variables of this Model to be
        monitored. It can be overridden for each Monitor if desired. The 
        corresponding state-variable indices for this model are :math:`V = 0`,
        :math:`W = 1`, and :math:`Z = 2`.""",
        order = 21)
    
    
    def __init__(self, **kwargs):
        """
        Initialize the Larter model's traited attributes, any provided as
        keywords will overide their traited default.
        
        """
        LOG.info('%s: initing...' % str(self))
        super(Larter, self).__init__(**kwargs)
        
        #self._state_variables = ["V", "W", "Z"]
        self._nvar = 3
        
        self.cvar = numpy.array([0], dtype=numpy.int32)
        
        LOG.debug('%s: inited.' % repr(self))
    
    
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
        
        #import pdb; pdb.set_trace()  
        dV =  (local_coupling * V - alpha_inh * Z - 
              self.gL * (V - self.VL) -
              self.gCa * M_inf * (V - 1) -
              self.gK * W * (V - self.VK + c_0) + self.Iext)
        
        dW = self.phi * tau_Winv * (W_inf - W)
        
        dZ = self.b * ((self.c * self.Iext) + (alpha_exc * V))
        
        derivative = numpy.array([dV, dW, dZ]) 
        
        return derivative

