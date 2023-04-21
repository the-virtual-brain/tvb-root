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
A contributed model: Larter model revisited by Breaskpear M.

.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import numpy

from tvb.simulator.common import get_logger
from tvb.basic.neotraits.api import NArray, Range, List, Final
import tvb.simulator.models as models

LOG = get_logger(__name__)


class LarterBreakspear(models.Model):
    """
    A modified Morris-Lecar model that includes a third equation which simulates
    the effect of a population of inhibitory interneurons synapsing on
    the pyramidal cells.
    
    .. [Larteretal_1999] Larter et.al. *A coupled ordinary differential equation
        lattice model for the simulation of epileptic seizures.* Chaos. 9(3):
        795, 1999.
    

    .. [Breaksetal_2003_a] Breakspear, M.; Terry, J. R. & Friston, K. J.  *Modulation of excitatory
        synaptic coupling facilitates synchronization and complex dynamics in an
        onlinear model of neuronal dynamics*. Neurocomputing 52–54 (2003).151–158

    .. [Breaksetal_2003_b] M. J. Breakspear et.al. *Modulation of excitatory 
        synaptic coupling facilitates synchronization and complex dynamics in a
        biophysical model of neuronal dynamics.* Network: Computation in Neural
        Systems 14: 703-732, 2003.
    
    Equations and default parameters are taken from [Breaksetal_2003_b]_. 
    All equations and parameters are non-dimensional and normalized.
    For values of d_v  < 0.55, the dynamics of a single column settles onto a 
    solitary fixed point attractor.


    Parameters used for simulations in [Breaksetal_2003_a]_ Table 1. Page 153.
    Two nodes were coupled.

    +---------------------------+
    |          Table 1          | 
    +--------------+------------+
    |Parameter     |  Value     |
    +--------------+------------+
    | I            |      0.3   |
    | a_ee         |      0.4   |
    | a_ei         |      0.1   |
    | a_ie         |      1.0   |
    | a_ne         |      1.0   |
    | a_ni         |      0.4   |
    | r_NMDA       |      0.2   |
    | delta        |      0.001 |
    +---------------------------+



    +---------------------------+
    |          Table 2          | 
    +--------------+------------+
    |Parameter     |  Value     |
    +--------------+------------+
    | gK           |      2.0   |
    | gL           |      0.5   |
    | gNa          |      6.7   |
    | gCa          |      1.0   |
    | a_ne         |      1.0   |
    | a_ni         |      0.4   |
    | a_ee         |      0.36  |
    | a_ei         |      2.0   |
    | a_ie         |      2.0   |
    | VK           |     -0.7   |
    | VL           |     -0.5   |
    | VNa          |      0.53  |
    | VCa          |      1.0   |
    | phi          |      0.7   | 
    | b            |      0.1   |
    | I            |      0.3   |
    | r_NMDA       |      0.25  |
    | C            |      0.1   |
    | TCa          |     -0.01  |
    | d_Ca         |      0.15  |
    | TK           |      0.0   |
    | d_K          |      0.3   |
    | VT           |      0.0   |
    | ZT           |      0.0   |
    | TNa          |      0.3   |
    | d_Na         |      0.15  |
    | d_V          |      0.65  |
    | d_Z          |      d_V   |  # note, this parameter might be spatialized: ones(N,1).*0.65 + modn*(rand(N,1)-0.5);
    | QV_max       |      1.0   |
    | QZ_max       |      1.0   |
    +---------------------------+
    |   Alstott et al. 2009     |
    +---------------------------+


    NOTES about parameters

    d_V
    For d_V < 0.55, uncoupled network, the system exhibits fixed point dynamics; 
    for 55 < lb.d_V < 0.59, limit cycle atractors; 
    and for d_V > 0.59 chaotic attractors (eg, d_V=0.6,aee=0.5,aie=0.5, 
                                               gNa=0, Iext=0.165)

    C
    The long-range coupling 'C' is ‘weak’ in the sense that 
    they investigated parameter values for which C < a_ee and C << a_ie.


    
    .. figure :: img/LarterBreakspear_01_mode_0_pplane.svg
            :alt: Larter-Breaskpear phase plane (V, W)
            
            The (:math:`V`, :math:`W`) phase-plane for the Larter-Breakspear model.
    
    """

    # Define traited attributes for this model, these represent possible kwargs.
    gCa = NArray(
        label=":math:`g_{Ca}`",
        default=numpy.array([1.1]),
        domain=Range(lo=0.9, hi=1.5, step=0.1),
        doc="""Conductance of population of Ca++ channels.""")

    gK = NArray(
        label=":math:`g_{K}`",
        default=numpy.array([2.0]),
        domain=Range(lo=1.95, hi=2.05, step=0.025),
        doc="""Conductance of population of K channels.""")

    gL = NArray(
        label=":math:`g_{L}`",
        default=numpy.array([0.5]),
        domain=Range(lo=0.45, hi=0.55, step=0.05),
        doc="""Conductance of population of leak channels.""")

    phi = NArray(
        label=":math:`\\phi`",
        default=numpy.array([0.7]),
        domain=Range(lo=0.3, hi=0.9, step=0.1),
        doc="""Temperature scaling factor.""")

    gNa = NArray(
        label=":math:`g_{Na}`",
        default=numpy.array([6.7]),
        domain=Range(lo=0.0, hi=10.0, step=0.1),
        doc="""Conductance of population of Na channels.""")

    TK = NArray(
        label=":math:`T_{K}`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=0.0001, step=0.00001),
        doc="""Threshold value for K channels.""")

    TCa = NArray(
        label=":math:`T_{Ca}`",
        default=numpy.array([-0.01]),
        domain=Range(lo=-0.02, hi=-0.01, step=0.0025),
        doc="Threshold value for Ca channels.")

    TNa = NArray(
        label=":math:`T_{Na}`",
        default=numpy.array([0.3]),
        domain=Range(lo=0.25, hi=0.3, step=0.025),
        doc="Threshold value for Na channels.")

    VCa = NArray(
        label=":math:`V_{Ca}`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.9, hi=1.1, step=0.05),
        doc="""Ca Nernst potential.""")

    VK = NArray(
        label=":math:`V_{K}`",
        default=numpy.array([-0.7]),
        domain=Range(lo=-0.8, hi=1., step=0.1),
        doc="""K Nernst potential.""")

    VL = NArray(
        label=":math:`V_{L}`",
        default=numpy.array([-0.5]),
        domain=Range(lo=-0.7, hi=-0.4, step=0.1),
        doc="""Nernst potential leak channels.""")

    VNa = NArray(
        label=":math:`V_{Na}`",
        default=numpy.array([0.53]),
        domain=Range(lo=0.51, hi=0.55, step=0.01),
        doc="""Na Nernst potential.""")

    d_K = NArray(
        label=":math:`\\delta_{K}`",
        default=numpy.array([0.3]),
        domain=Range(lo=0.1, hi=0.4, step=0.1),
        doc="""Variance of K channel threshold.""")

    tau_K = NArray(
        label=":math:`\\tau_{K}`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.01, hi=0.0, step=0.1),
        doc="""Time constant for K relaxation time (ms)""")

    d_Na = NArray(
        label=":math:`\\delta_{Na}`",
        default=numpy.array([0.15]),
        domain=Range(lo=0.1, hi=0.2, step=0.05),
        doc="Variance of Na channel threshold.")

    d_Ca = NArray(
        label=":math:`\\delta_{Ca}`",
        default=numpy.array([0.15]),
        domain=Range(lo=0.1, hi=0.2, step=0.05),
        doc="Variance of Ca channel threshold.")

    aei = NArray(
        label=":math:`a_{ei}`",
        default=numpy.array([2.0]),
        domain=Range(lo=0.1, hi=2.0, step=0.1),
        doc="""Excitatory-to-inhibitory synaptic strength.""")

    aie = NArray(
        label=":math:`a_{ie}`",
        default=numpy.array([2.0]),
        domain=Range(lo=0.5, hi=2.0, step=0.1),
        doc="""Inhibitory-to-excitatory synaptic strength.""")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([0.1]),
        domain=Range(lo=0.0001, hi=1.0, step=0.0001),
        doc="""Time constant scaling factor. The original value is 0.1""")

    C = NArray(
        label=":math:`c`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=0.2, step=0.05),
        doc="""Strength of excitatory coupling. Balance between internal and
        local (and global) coupling strength. C > 0 introduces interdependences between 
        consecutive columns/nodes. C=1 corresponds to maximum coupling.
        This strenght should be set to sensible values when a whole network is connected. """)

    ane = NArray(
        label=":math:`a_{ne}`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.4, hi=1.0, step=0.05),
        doc="""Non-specific-to-excitatory synaptic strength.""")

    ani = NArray(
        label=":math:`a_{ni}`",
        default=numpy.array([0.4]),
        domain=Range(lo=0.3, hi=0.5, step=0.05),
        doc="""Non-specific-to-inhibitory synaptic strength.""")

    aee = NArray(
        label=":math:`a_{ee}`",
        default=numpy.array([0.4]),
        domain=Range(lo=0.4, hi=0.6, step=0.05),
        doc="""Excitatory-to-excitatory synaptic strength.""")

    Iext = NArray(
        label=":math:`I_{ext}`",
        default=numpy.array([0.3]),
        domain=Range(lo=0.165, hi=0.3, step=0.005),
        doc="""Subcortical input strength. It represents a non-specific
       excitation or thalamic inputs.""")

    rNMDA = NArray(
        label=":math:`r_{NMDA}`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.2, hi=0.3, step=0.05),
        doc="""Ratio of NMDA to AMPA receptors.""")

    VT = NArray(
        label=":math:`V_{T}`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=0.7, step=0.01),
        doc="""Threshold potential (mean) for excitatory neurons. 
        In [Breaksetal_2003_b]_ this values is 0.""")

    d_V = NArray(
        label=":math:`\\delta_{V}`",
        default=numpy.array([0.65]),
        domain=Range(lo=0.49, hi=0.7, step=0.01),
        doc="""Variance of the excitatory threshold. It is one of the main
        parameters explored in [Breaksetal_2003_b]_.""")

    ZT = NArray(
        label=":math:`Z_{T}`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=0.1, step=0.005),
        doc="""Threshold potential (mean) for inihibtory neurons.""")

    d_Z = NArray(
        label=":math:`\\delta_{Z}`",
        default=numpy.array([0.7]),
        domain=Range(lo=0.001, hi=0.75, step=0.05),
        doc="""Variance of the inhibitory threshold.""")

    # NOTE: the values were not in the article. 
    # I took these ones from DESTEXHE 2001
    QV_max = NArray(
        label=":math:`Q_{max}`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.1, hi=1., step=0.001),
        doc="""Maximal firing rate for excitatory populations (kHz)""")

    QZ_max = NArray(
        label=":math:`Q_{max}`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.1, hi=1., step=0.001),
        doc="""Maximal firing rate for excitatory populations (kHz)""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("V", "W", "Z"),
        default=("V", "W", "Z"),
        doc="""This represents the default state-variables of this Model to be
        monitored. It can be overridden for each Monitor if desired.""")

    # Informational attribute, used for phase-plane and initial()
    state_variable_range = Final(
        {
            "V": numpy.array([-1.5, 1.5]),
            "W": numpy.array([-1.0, 1.0]),
            "Z": numpy.array([-1.5, 1.5])
        },
        label="State Variable ranges [lo, hi]",
        doc="""The values for each state-variable should be set to encompass
            the expected dynamic range of that state-variable for the current 
            parameters, it is used as a mechanism for bounding random inital 
            conditions when the simulation isn't started from an explicit
            history, it is also provides the default range of phase-plane plots.""")

    state_variables = ["V", "W", "Z"]
    _nvar = 3
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        """
        .. math::
             \\dot{V} &= - (g_{Ca} + (1 - C) \\, r_{NMDA} \\, a_{ee} Q_V^i +
            C \\, r_{NMDA} \\, a_{ee} \\langle Q_V \\rangle) \\, m_{Ca} \\,(V - V_{Ca})
            - g_K\\, W\\, (V - V_K) - g_L\\, (V - V_L)
            - (g_{Na} m_{Na} + (1 - C) \\, a_{ee} Q_V^i + 
            C \\, a_{ee} \\langle Q_V \\rangle) \\, (V - V_{Na})
            - a_{ie}\\, Z \\, Q_Z^i + a_{ne} \\, I_{\\delta}
            
            \\dot{W} &= \\frac{\\phi \\, (m_K - W)}{\\tau_K} \\\\
            \\dot{Z} &= b \\, (a_{ni} \\, I_{\\delta} + a_{ei} \\, V \\, Q_V)\\\\
            
            m_{ion}(X) &= 0.5 \\, (1 + tanh(\\frac{V-T_{ion}}{\\delta_{ion}})
            
        See Equations (7), (3), (6) and (2) respectively in [Breaksetal_2003]_.
        Pag: 705-706

        NOTE: Equation (8) has an error the sign before the term :math:`a_{ie}\\, Z \\, Q_Z^i`
        should be a minus (-) and not a plus (+).
        
        """
        V = state_variables[0, :]
        W = state_variables[1, :]
        Z = state_variables[2, :]

        c_0 = coupling[0, :]
        lc_0 = local_coupling

        # relationship between membrane voltage and channel conductance
        m_Ca = 0.5 * (1 + numpy.tanh((V - self.TCa) / self.d_Ca))
        m_Na = 0.5 * (1 + numpy.tanh((V - self.TNa) / self.d_Na))
        m_K = 0.5 * (1 + numpy.tanh((V - self.TK) / self.d_K))

        # voltage to firing rate
        QV = 0.5 * self.QV_max * (1 + numpy.tanh((V - self.VT) / self.d_V))
        QZ = 0.5 * self.QZ_max * (1 + numpy.tanh((Z - self.ZT) / self.d_Z))

        dV = (- (self.gCa + (
                    1.0 - self.C) * self.rNMDA * self.aee * QV + self.C * self.rNMDA * self.aee * c_0) * m_Ca * (
                          V - self.VCa) - self.gK * W * (V - self.VK) - self.gL * (V - self.VL) - (
                          self.gNa * m_Na + (1.0 - self.C) * self.aee * QV + self.C * self.aee * c_0) * (
                          V - self.VNa) - self.aei * Z * QZ + self.ane * self.Iext)

        dW = (self.phi * (m_K - W) / self.tau_K)

        dZ = (self.b * (self.ani * self.Iext + self.aei * V * QV))

        derivative = numpy.array([dV, dW, dZ])

        return derivative
