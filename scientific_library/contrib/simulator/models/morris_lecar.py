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
A contributed model: Morris-Lecar

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



class MorrisLecar(models.Model):
    """
    The Morris-Lecar model is a mathematically simple excitation model having
    two nonlinear, non-inactivating conductances.

    .. [ML_1981] Morris, C. and Lecar, H. *Voltage oscillations in the Barnacle
        giant muscle fibre*, Biophysical Journal 35: 193, 1981.

    See also, http://www.scholarpedia.org/article/Morris-Lecar_model
    
        .. figure :: img/MorrisLecar_01_mode_0_pplane.svg
            :alt: Morris-Lecar phase plane (V, N)
            
            The (:math:`V`, :math:`N`) phase-plane for the Morris-Lecar model.
    
    .. #Currently there seems to be a clash betwen traits and autodoc, autodoc
    .. #can't find the methods of the class, the class specific names below get
    .. #us around this...
    .. automethod:: MorrisLecar.__init__
    .. automethod:: MorrisLecar.dfun
    
    
    """
    
    _ui_name = "Morris-Lecar"
    ui_configurable_parameters = ['gCa', 'gK', 'gL', 'C', 'lambda_Nbar', 'V1',
                                  'V2', 'V3', 'V4', 'VCa', 'VK', 'VL']
    
    #Define traited attributes for this model, these represent possible kwargs.
    gCa = arrays.FloatArray(
        label = ":math:`g_{Ca}`",
        default = numpy.array([4.0]),
        range = basic.Range(lo = 2.0, hi = 6.0, step = 0.01),
        doc = """Conductance of population of Ca++ channels [mmho/cm2]""",
        order = 1)
    
    gK = arrays.FloatArray(
        label = ":math:`g_K`",
        default = numpy.array([8.0]),
        range = basic.Range(lo = 4.0, hi = 12.0, step = 0.01),
        doc = """Conductance of population of K+ channels [mmho/cm2]""",
        order = 2)
    
    gL = arrays.FloatArray(
        label = ":math:`g_L`",
        default = numpy.array([2.0]),
        range = basic.Range(lo = 1.0, hi = 3.0, step = 0.01),
        doc = """Conductance of population of leak channels [mmho/cm2]""",
        order = 3)
    
    C = arrays.FloatArray(
        label = ":math:`C`",
        default = numpy.array([20.0]),
        range = basic.Range(lo = 10.0, hi = 30.0, step = 0.01),
        doc = """Membrane capacitance [uF/cm2]""",
        order = 4)
    
    lambda_Nbar = arrays.FloatArray(
        label = ":math:`\\lambda_{Nbar}`",
        default = numpy.array([0.06666667]),
        range = basic.Range(lo = 0.0, hi = 1.0, step = 0.00000001),
        doc = """Maximum rate for K+ channel opening [1/s]""",
        order = 5)
    
    V1 = arrays.FloatArray(
        label = ":math:`V_1`",
        default = numpy.array([10.0]),
        range = basic.Range(lo = 5.0, hi = 15.0, step = 0.01),
        doc = """Potential at which half of the Ca++ channels are open at steady
        state [mV]""",
        order = 6)
    
    V2 = arrays.FloatArray(
        label = ":math:`V_2`",
        default = numpy.array([15.0]),
        range = basic.Range(lo = 7.5, hi = 22.5, step = 0.01),
        doc = """1/slope of voltage dependence of the fraction of Ca++ channels
        that are open at steady state [mV].""",
        order = 7)
    
    V3 = arrays.FloatArray(
        label = ":math:`V_3`",
        default = numpy.array([-1.0]),
        range = basic.Range(lo = -1.5, hi = -0.5, step = 0.01),
        doc = """Potential at which half of the K+ channels are open at steady
        state [mV].""",
        order = 8)
    
    V4 = arrays.FloatArray(
        label = ":math:`V_4`",
        default = numpy.array([14.5]),
        range = basic.Range(lo = 7.25, hi = 22.0, step = 0.01),
        doc = """1/slope of voltage dependence of the fraction of K+ channels
        that are open at steady state [mV].""",
        order = 9)
    
    VCa = arrays.FloatArray(
        label = ":math:`V_{Ca}`",
        default = numpy.array([100.0]),
        range = basic.Range(lo = 50.0, hi = 150.0, step = 0.01),
        doc = """Ca++ Nernst potential [mV]""",
        order = 10)
    
    VK = arrays.FloatArray(
        label = ":math:`V_K`",
        default = numpy.array([-70.0]),
        range = basic.Range(lo = -105.0, hi = -35.0, step = 0.01),
        doc = """K+ Nernst potential [mV]""",
        order = 11)
    
    VL = arrays.FloatArray(
        label = ":math:`V_L`",
        default = numpy.array([-50.0]),
        range = basic.Range(lo = -75.0, hi = -25.0, step = 0.01),
        doc = """Nernst potential leak channels [mV]""",
        order = 12)
    
    #Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = basic.Dict(
        label = "State Variable ranges [lo, hi]",
        default = {"V": numpy.array([-70.0, 50.0]),
                   "N": numpy.array([-0.2, 0.8])},
        doc = """The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current 
        parameters, it is used as a mechanism for bounding random inital 
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""",
        order = 13)
    
    variables_of_interest = arrays.IntegerArray(
        label = "Variables watched by Monitors",
        range = basic.Range(lo = 0, hi = 2, step=1),
        default = numpy.array([0], dtype=numpy.int32),
        doc = """This represents the default state-variables of this Model to be
        monitored. It can be overridden for each Monitor if desired. The 
        corresponding state-variable indices for this model are :math:`V = 0`,
        and :math:`N = 1`.""",
        order = 14)
    
#    coupling_variables = arrays.IntegerArray(
#        label = "Variables to couple activity through",
#        default = numpy.array([0], dtype=numpy.int32))
    
#    nsig = arrays.FloatArray(
#        label = "Noise dispersion",
#        default = numpy.array([0.0]),
#        range = basic.Range(lo = 0.0, hi = 1.0))
    
    
    def __init__(self, **kwargs):
        """
        Initialize the MorrisLecar model's traited attributes, any provided
        as keywords will overide their traited default.
        
        """
        LOG.info('%s: initing...' % str(self))
        super(MorrisLecar, self).__init__(**kwargs)
        
        self._state_variables = ["V", "N"]
        self._nvar = 2
        
        self.cvar = numpy.array([0], dtype=numpy.int32)
        
        LOG.debug('%s: inited.' % repr(self))
    
    
    def dfun(self, state_variables, coupling, local_coupling=0.0):
        """
        The dynamics of the membrane potential :math:`V` rely on the fraction
        of Ca++ channels :math:`M` and K+ channels :math:`N` open at a given
        time. In order to have a planar model, we make the simplifying
        assumption (following [ML_1981]_, Equation 9) that Ca++ system is much
        faster than K+ system so that :math:`M = M_{\\infty}` at all times:
        
        .. math::
             C \\, \\dot{V} &= I - g_{L}(V - V_L) - g_{Ca} \\, M_{\\infty}(V)
                              (V - V_{Ca}) - g_{K} \\, N \\, (V - V_{K}) \\\\
             \\dot{N} &= \\lambda_{N}(V) \\, (N_{\\infty}(V) - N) \\\\
             M_{\\infty}(V) &= 1/2 \\, (1 + \\tanh((V - V_{1})/V_{2}))\\\\
             N_{\\infty}(V) &= 1/2 \\, (1 + \\tanh((V - V_{3})/V_{4}))\\\\
             \\lambda_{N}(V) &= \\overline{\\lambda_{N}}
                                \\cosh((V - V_{3})/2V_{4})
        
        where external currents :math:`I` provide the entry point for local and
        long-range connectivity. Default parameters are set as per Figure 9 of
        [ML_1981]_ so that the model shows oscillatory behaviour as :math:`I` is
        varied.
        """
        
        V = state_variables[0, :]
        N = state_variables[1, :]
        
        c_0 = coupling[0, :]
        
        M_inf = 0.5 * (1 + numpy.tanh((V - self.V1) / self.V2))
        N_inf = 0.5 * (1 + numpy.tanh((V - self.V3) / self.V4))
        lambda_N = self.lambda_Nbar * numpy.cosh((V - self.V3) / (2.0 * self.V4))
        
        dV = (1.0 / self.C) * (c_0 + (local_coupling * V) -
                               self.gL * (V - self.VL) -
                               self.gCa * M_inf * (V - self.VCa) -
                               self.gK * N * (V - self.VK))
        
        dN = lambda_N * (N_inf - N)
        
        derivative = numpy.array([dV, dN]) 
        
        return derivative

