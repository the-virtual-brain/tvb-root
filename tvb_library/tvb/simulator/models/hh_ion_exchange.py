# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
   A family of mean field models of infinite populations of all-to-all coupled
   quadratic integrate and fire neurons (theta-neurons).
.. moduleauthor:: Giovanni Rabuffo <giovanni.rabuffo@univ-amu.fr>
"""

from tvb.simulator.models.base import Model
from tvb.basic.neotraits.api import NArray, List, Range, Final

import numpy

class HHIonExchange(Model):
    r"""
    5D model describing the Ott-Antonsen reduction of infinite all-to-all
    coupled Hodgkin-Huxley-type neurons (as in Depannemaker et al 2022).
    The six state variables :math:`x` represents a phenomenological variable connected to the firing rate, :math:`V` represent the average membrane potential.......
    """
    #_ui_name = "Infinite HH"
    #ui_configurable_parameters = ['E', 'K_bath', 'J', 'eta', 'Delta','c_minus','R_minus','c_plus','R_plus','Vstar']

    E = NArray(
        label=r":math:`E`",
        default=numpy.array([0.]),
        domain=Range(lo=-80, hi=0, step=0.5),
        doc="""Reversal Potential""",
    )

    K_bath = NArray(
        label=r":math:`K_bath`",
        default=numpy.array([5.5]),
        domain=Range(lo=3, hi=40.0, step=0.25),
        doc="""Potassium concentration in bath""",
    )

    J = NArray(
        label=r":math:`J`",
        default=numpy.array([0.1]),
        domain=Range(lo=0.001, hi=40.0, step=0.01),
        doc="""Mean Synaptic weight""",
    )

    eta = NArray(
        label=":math:`eta`",
        default=numpy.array([0.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.1),
        doc="""Mean heterogeneous noise""",
    )

    Delta = NArray(
        label=r":math:`\Delta`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""HWHM heterogeneous noise""",
    )

    c_minus = NArray(
        label=":math:`c_minus`",
        default=numpy.array([-40.0]),
        domain=Range(lo=-100.0, hi=-10.0, step=0.5),
        doc="""x-coordinate left parabola""",
    )

    R_minus = NArray(
        label=r":math:`R_minus`",
        default=numpy.array([0.5]),
        domain=Range(lo=0.0001, hi=5.0, step=0.01),
        doc="""curvature left parabola""",
    )

    c_plus = NArray(
        label=":math:`c_plus`",
        default=numpy.array([-20.0]),
        domain=Range(lo=-80.0, hi=0.0, step=0.5),
        doc="""x-coordinate right parabola""",
    )

    R_plus = NArray(
        label=r":math:`R_plus`",
        default=numpy.array([-0.5]),
        domain=Range(lo=-5.0, hi=-0.0001, step=0.01),
        doc="""curvature right parabola""",
    )


    Vstar = NArray(
        label=r":math:`Vstar`",
        default=numpy.array([-31]),
        domain=Range(lo=-55.0, hi=-15, step=0.5),
        doc="""x-coordinate meeting point of parabolas""",
    )

    #'Cm': 1, # membrane capacitance
    Cm = NArray(
        label=r":math:`Cm`",
        default=numpy.array([1]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""membrane capacitance""",
    )

    # 'tau_n': 4, #ms # time constant of gating variable
    tau_n = NArray(
        label=r":math:`tau_n`",
        default=numpy.array([4]),
        domain=Range(lo=2, hi=6, step=0.5),
        doc="""time constant of gating variable""",
    )

    # 'gamma': 0.04,  # mol / C  # conversion factor
    gamma = NArray(
        label=r":math:`gamma`",
        default=numpy.array([0.04]),
        domain=Range(lo=0.02, hi=0.06, step=0.005),
        doc="""conversion factor""",
    )

    #'epsilon': 0.001, # mHz  # diffusion rate
    epsilon = NArray(
        label=r":math:`epsilon`",
        default=numpy.array([0.001]),
        domain=Range(lo=0.0005, hi=0.0015, step=0.0001),
        doc="""diffusion rate""",
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            "x": numpy.array([0.0, numpy.inf]),
            "V": numpy.array([-500.0, numpy.inf]),
            "n": numpy.array([0.0, numpy.inf]),
            "DKi": numpy.array([-100.0, numpy.inf]),
            "Kg": numpy.array([-100.0, numpy.inf])
        },
    )


    # TODO should match cvars below..
    coupling_terms = Final(
        label="Coupling terms",
        # how to unpack coupling array
        default=["Coupling_Term"]
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("x", "V","n","DKi", "Kg"),
        default=("x", "V","n","DKi", "Kg"),
        doc="The quantities of interest for monitoring for the Infinite HH 5D.",
    )

    state_variables = ['x', 'V','n','DKi','Kg']
    _nvar = 5
    # Cvar is the coupling variable. 
    cvar = numpy.array([0], dtype=numpy.int32)
    # Stvar is the variable where stimulus is applied.
    stvar = numpy.array([1], dtype=numpy.int32)
    
    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The equations of the infinite HH 5D population model read

        .. math::
            r &= R_{minus}*x/\pi \\
            Vdot &= (-1.0/Cm)*(I_{Na}+I_K+I_{Cl}+I_{pump})\\
            dx &= 
            \begin{cases}
            \Delta+2*R_minus*(V-c_minus)*x-J*r*x & \text{if } V <= Vstar \\
            \Delta+2*R_plus*(V-c_plus)*x-J*r*x& \text{else} 
            \end{cases} \\
            dV &= 
            \begin{cases}
            Vdot- R_minus*x**2 + eta + (R_minus/numpy.pi)*Coupling_Term & \text{if } V <= Vstar \\
            Vdot- R_plus*x**2 + eta + (R_minus/numpy.pi)*Coupling_Term & \text{else}
            \end{cases} \\
            dn &= (n_inf(V) - n) / Par['tau_n']\\
            dDKi &= -(Par['gamma'] / Par['w_i']) * (I_K - 2.0 * I_pump)\\
            dKg &= Par['epsilon'] * (K_bath - K_o)\\


        The equations of the infinite HH 5D population model read (on either side of a threshold Vstar)

        .. math::
                \dot{x} &= Delta+2*R_minus*(V-c_minus)*x-nu*s*x,\\
                \dot{V} &= V_dot_form(I_Na,I_K,I_Cl,I_pump,R_minus,V,x,s,eta,nu), \\
                \dot{n} &= Delta/pi + 2 V r - k r^2,\\
                \dot{s} &= (V^2 - pi^2 r^2 + eta + (k s + J) r - k V r + gamma I ),\\
                \dot{DKi} &= Delta/pi + 2 V r - k r^2,\\
                \dot{Kg} &= (V^2 - pi^2 r^2 + eta + (k s + J) r - k V r + gamma I ),\\

        """
        x = state_variables[0, :]
        V = state_variables[1, :]
        n = state_variables[2, :]
        DKi = state_variables[3, :]
        Kg = state_variables[4, :]
        
        #[State_variables, nodes]
        E = self.E

        K_bath = self.K_bath
        J = self.J
        eta = self.eta
        Delta = self.Delta

        c_minus = self.c_minus
        R_minus = self.R_minus
        c_plus = self.c_plus
        R_plus = self.R_plus
        Vstar = self.Vstar

        Cm      = self.Cm
        tau_n   = self.tau_n
        gamma   = self.gamma
        epsilon = self.epsilon
     
        Coupling_Term = coupling[0, :] #This zero refers to the first element of cvar (trivial in this case)

        # Constants
        Cnap = 21.0  # mol.m**-3 
        DCnap = 2.0  # mol.m**-3 
        Ckp = 5.5  # mol.m**-3 
        DCkp = 1.0  # mol.m**-3 
        Cmna = -24.0  # mV 
        DCmna = 12.0  # mV 
        Chn = 0.4  # dimensionless 
        DChn = -8.0  # dimensionless 
        Cnk = -19.0  # mV 
        DCnk = 18.0  # mV #Ok in the paper
        g_Cl = 7.5  # nS #Ok in the paper   # chloride conductance
        g_Na = 40.0  # nS   # maximal sodiumconductance
        g_K = 22.0  # nS  # maximal potassium conductance
        g_Nal = 0.02  # nS  # sodium leak conductance
        g_Kl = 0.12  # nS  # potassium leak conductance
        rho = 250.  # 250.,#pA # maximal Na/K pump current
        w_i = 2160.0  # umeter**3  # intracellular volume 
        w_o = 720.0  # umeter**3 # extracellular volume 
        Na_i0 = 16.0  # mMol/m**3 # initial concentration of intracellular Na
        Na_o0 = 138.0 # mMol/m**3 # initial concentration of extracellular Na
        K_i0 = 130.0  # mMol/m**3 # initial concentration of intracellular K
        K_o0 = 4.80   # mMol/m**3 # initial concentration of extracellular K
        Cl_i0 = 5.0   # mMol/m**3 # initial concentration of intracellular Cl
        Cl_o0 = 112.0 # mMol/m**3 # initial concentration of extracellular Cl
        

        # helper functions

        def m_inf(V):
            return 1.0/(1.0+numpy.exp((Cmna-V)/DCmna))

        def n_inf(V):
            return 1.0/(1.0+numpy.exp((Cnk-V)/DCnk))

        def h(n):
            return 1.1 - 1.0 / (1.0 + numpy.exp(-8.0 * (n - 0.4)))

        def I_K_form(V,n,K_o,K_i):
            return (g_Kl+g_K*n)*(V- 26.64*numpy.log(K_o/K_i)) 

        def I_Na_form(V,Na_o,Na_i,n):
            return (g_Nal+g_Na*m_inf(V)*h(n))*(V- 26.64*numpy.log(Na_o/Na_i))

        def I_Cl_form(V):
            return g_Cl*(V+ 26.64*numpy.log(Cl_o0/Cl_i0)) 

        def I_pump_form(Na_i,K_o):
            return rho*(1.0/(1.0+numpy.exp((Cnap - Na_i) / DCnap))*(1.0/(1.0+numpy.exp((Ckp - K_o)/DCkp)))) 

        def V_dot_form(I_Na,I_K,I_Cl,I_pump):
            return (-1.0/Cm)*(I_Na+I_K+I_Cl+I_pump) 

        beta= w_i / w_o 
        DNa_i = -DKi 
        DNa_o = -beta * DNa_i
        DK_o = -beta * DKi
        K_i = K_i0 + DKi 
        Na_i = Na_i0 + DNa_i 
        Na_o = Na_o0 + DNa_o 
        K_o = K_o0 + DK_o + Kg 

        ninf=n_inf(V)
        I_K = I_K_form(V,n,K_o,K_i)
        I_Na = I_Na_form(V,Na_o,Na_i,n)
        I_Cl = I_Cl_form(V)
        I_pump = I_pump_form(Na_i,K_o)
    
        r = R_minus*x/numpy.pi
        Vdot = (-1.0/Cm)*(I_Na+I_K+I_Cl+I_pump) 

        derivative = numpy.empty_like(state_variables)

        if_xdot = Delta+2*R_minus*(V-c_minus)*x-J*r*x 
        else_xdot = Delta+2*R_plus*(V-c_plus)*x-J*r*x
        derivative[0] = numpy.where(V <= Vstar, if_xdot, else_xdot)

        if_Vdot = Vdot - R_minus*x**2 + eta + (R_minus/numpy.pi)*Coupling_Term*(E-V)
        else_Vdot = Vdot - R_plus*x**2 + eta + (R_minus/numpy.pi)*Coupling_Term*(E-V)
        derivative[1] = numpy.where(V <= Vstar, if_Vdot, else_Vdot)

        derivative[2] = (ninf - n) / tau_n
        derivative[3] = -(gamma / w_i) * (I_K - 2.0 * I_pump)
        derivative[4] = epsilon * (K_bath - K_o)
        
        return derivative
