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
   EpileptorRestingState model: modeling resting-state in epilepsy.
   
   .. moduleauthor:: courtiol.julie@gmail.com
"""

import numpy
from .base import ModelNumbaDfun
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, List, Range, Final


class EpileptorRestingState(ModelNumbaDfun):
    r"""
        EpileptorRestingState is an extension of the phenomenological neural mass model of partial seizures 
        Epileptor [Jirsaetal_2014], tuned to express regionally specific physiological oscillations in addition
        to the epileptiform discharges. This extension was made using the Generic 2-dimensional Oscillator model
        (parametrized close to a supercritical Hopf Bifurcation) [SanzLeonetal_2013] to reproduce the spontaneous
        local field potential-like signal.
        
        This model, its motivation and derivation can be found in the published article [Courtioletal_2020].
        
        .. Tutorial: Modeling_Resting-State_in_Epilepsy.ipynb
        
        .. References:
            [Jirsaetal_2014] Jirsa, V. K.; Stacey, W. C.; Quilichini, P. P.; Ivanov, A. I.; Bernard, 
            C. *On the nature of seizure dynamics.* Brain, 2014.
            [SanzLeonetal_2013] Sanz Leon, P.; Knock, S. A.; Woodman, M. M.; Domide, L.; Mersmann, 
            J.; McIntosh, A. R.; Jirsa, V. K. *The Virtual Brain: a simulator of primate brain 
            network dynamics.* Front.Neuroinf., 2013.
            [Courtioletal_2020] Courtiol, J.; Guye, M.; Bartolomei, F.; Petkoski, S.; Jirsa, V. K.
            *Dynamical Mechanisms of Interictal Resting-State Functional Connectivity in Epilepsy.*
            J.Neurosci., 2020.
        
        Variables of interest to be used by monitors: p * (-x_{1} + x_{2}) + (1 - p) * x_{rs}
        
            .. math::
                \dot{x_{1}} &=& y_{1} - f_{1}(x_{1}, x_{2}) - z + I_{ext1} \\
                \dot{y_{1}} &=& c - d x_{1}^{2} - y_{1} \\
                \dot{z} &=&
                    \begin{cases}
                        r(4 (x_{1} - x_{0}) - z -0.1 z^{7}) & \text{if} x<0 \\
                        r(4 (x_{1} - x_{0}) - z) & \text{if} x \geq 0
                    \end{cases} \\
                \dot{x_{2}} &=& -y_{2} + x_{2} - x_{2}^{3} + I_{ext2} + b_{2} g(x_{1}) - 0.3 (z-3.5) \\
                \dot{y_{2}} &=& 1 / \tau (-y_{2} + f_{2}(x_{2}))\\
                \dot{g} &=& -0.01 (g - 0.1 x_{1})\\
                \dot{x_{rs}} &=& d_{rs} \tau_{rs} (-f_{rs} x_{rs}^3 + e_{rs} x_{rs}^2 + \alpha_{rs} y_{rs} +
                \gamma_{rs} I_{rs}) \\
                \dot{y_{rs}} &=& d_{rs} (b_{rs}  x_{rs} - \beta_{rs} y_{rs} + a_{rs}) / \tau_{rs}
        
        where:
            .. math::
                f_{1}(x_{1}, x_{2}) =
                    \begin{cases}
                        a x_{1}^{3} - b x_{1}^2 & \text{if } x_{1} <0\\
                        -(slope - x_{2} + 0.6(z-4)^2) x_{1} &\text{if }x_{1} \geq 0
                    \end{cases}
            
        and:
            .. math::
                f_{2}(x_{2}) =
                    \begin{cases}
                        0 & \text{if } x_{2} <-0.25\\
                        a_{2}(x_{2} + 0.25) & \text{if } x_{2} \geq -0.25
                    \end{cases}


    """

    a = NArray(
        label=":math:`a`",
        default=numpy.array([1.0]),
        doc="Coefficient of the cubic term in the first state-variable x1.")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([3.0]),
        doc="Coefficient of the squared term in the first state-variable x1.")

    c = NArray(
        label=":math:`c`",
        default=numpy.array([1.0]),
        doc="Additive coefficient for the second state-variable y1, \
        called :math:'y_{0}' in Jirsa et al. (2014).")

    d = NArray(
        label=":math:`d`",
        default=numpy.array([5.0]),
        doc="Coefficient of the squared term in the second state-variable y1.")

    r = NArray(
        label=":math:`r`",
        domain=Range(lo=0.0, hi=0.001, step=0.00005),
        default=numpy.array([0.00035]),
        doc="Temporal scaling in the third state-variable z, \
        called :math:'1/\tau_{0}' in Jirsa et al. (2014).")

    s = NArray(
        label=":math:`s`",
        default=numpy.array([4.0]),
        doc="Linear coefficient in the third state-variable z.")

    x0 = NArray(
        label=":math:`x_0`",
        domain=Range(lo=-3.0, hi=-1.0, step=0.1),
        default=numpy.array([-1.6]),
        doc="Epileptogenicity parameter.")

    Iext = NArray(
        label=":math:`I_{ext}`",
        domain=Range(lo=1.5, hi=5.0, step=0.1),
        default=numpy.array([3.1]),
        doc="External input current to the first population (x1, y1).")

    slope = NArray(
        label=":math:`slope`",
        domain=Range(lo=-16.0, hi=6.0, step=0.1),
        default=numpy.array([0.]),
        doc="Linear coefficient in the first state-variable x1.")

    Iext2 = NArray(
        label=":math:`I_{ext2}`",
        domain=Range(lo=0.0, hi=1.0, step=0.05),
        default=numpy.array([0.45]),
        doc="External input current to the second population (x2, y2).")

    tau = NArray(
        label=r":math:`\tau`",
        default=numpy.array([10.0]),
        doc="Temporal scaling coefficient in the fifth state-variable y2.")

    aa = NArray(
        label=":math:`aa`",
        default=numpy.array([6.0]),
        doc="Linear coefficient in the fifth state-variable y2.")
        
    bb = NArray(
        label=":math:`bb`",
        default=numpy.array([2.0]),
        doc="Linear coefficient of lowpass excitatory coupling in the fourth \
        state-variable x2.")

    Kvf = NArray(
        label=":math:`K_{vf}`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=4.0, step=0.5),
        doc="Coupling scaling on a very fast time scale.")

    Kf = NArray(
        label=":math:`K_f`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=4.0, step=0.5),
        doc="Coupling scaling on a fast time scale.")

    Ks = NArray(
        label=":math:`K_s`",
        default=numpy.array([0.0]),
        domain=Range(lo=-4.0, hi=4.0, step=0.1),
        doc="Permittivity coupling, that is from the very fast time scale \
        toward the slow time scale.")

    tt = NArray(
        label=":math:`tt`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.001, hi=10.0, step=0.001),
        doc="Time scaling of the Epileptor.")
        
    # Generic-2D's parameters
    tau_rs = NArray(
        label=r":math:`\tau_{rs}`",
        default=numpy.array([1.0]),
        domain=Range(lo=1.0, hi=5.0, step=0.01),
        doc="Temporal scaling coefficient in the third population (x_rs, y_rs).")
        
    I_rs = NArray(
        label=":math:`I_{rs}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.01),
        doc="External input current to the third population (x_rs, y_rs).")
        
    a_rs = NArray(
        label=":math:`a_{rs}`",
        default=numpy.array([-2.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.01),
        doc="Vertical shift of the configurable nullcline \
        in the state-variable y_rs.")
        
    b_rs = NArray(
        label=":math:`b_{rs}`",
        default=numpy.array([-10.0]),
        domain=Range(lo=-20.0, hi=15.0, step=0.01),
        doc="Linear coefficient of the state-variable y_rs.")
        
    d_rs = NArray(
        label=":math:`d_{rs}`",
        default=numpy.array([0.02]),
        domain=Range(lo=0.0001, hi=1.0, step=0.0001),
        doc="Temporal scaling of the whole third system (x_rs, y_rs).")
        
    e_rs = NArray(
        label=":math:`e_{rs}`",
        default=numpy.array([3.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="Coefficient of the squared term in the sixth state-variable x_rs.")
    
    f_rs = NArray(
        label=":math:`f_{rs}`",
        default=numpy.array([1.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="Coefficient of the cubic term in the sixth state-variable x_rs.")

    alpha_rs = NArray(
        label=r":math:`\alpha_{rs}`",
        default=numpy.array([1.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="Constant parameter to scale the rate of feedback from the \
        slow variable y_rs to the fast variable x_rs.")
        
    beta_rs = NArray(
        label=r":math:`\beta_{rs}`",
        default=numpy.array([1.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="Constant parameter to scale the rate of feedback from the \
        slow variable y_rs to itself.")
        
    gamma_rs = NArray(
        label=r":math:`\gamma_{rs}`",
        default=numpy.array([1.0]),
        domain=Range(lo=-1.0, hi=1.0, step=0.1),
        doc="Constant parameter to reproduce FHN dynamics where \
        excitatory input currents are negative.\
        Note: It scales both I_rs and the long-range coupling term.")
        
    K_rs = NArray(
        label=r":math:`K_{rs}`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.001),
        doc="Coupling scaling on a fast time scale.")

    # Combination 2 models
    p = NArray(
        label=r":math:`p`",
        default=numpy.array([0.]),
        domain=Range(lo=-1.0, hi=1.0, step=0.1),
        doc="Linear coefficient.")

    # Initialization.
    # Epileptor model is set in a fixed point by default.
    state_variable_range = Final(
        label="State variable ranges [lo, hi]",
        default={
            "x1": numpy.array([-1.8, -1.4]),
            "y1": numpy.array([-15, -10]),
            "z": numpy.array([3.6, 4.0]),
            "x2": numpy.array([-1.1, -0.9]),
            "y2": numpy.array([0.001, 0.01]),
            "g": numpy.array([-1., 1.]),
            "x_rs": numpy.array([-2.0, 4.0]),
            "y_rs": numpy.array([-6.0, 6.0])},
        doc="Typical bounds on state-variables in EpileptorRestingState model.")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("x1", "y1", "z", "x2", "y2", "g", "x_rs", "y_rs", "x2 - x1"),
        default=("x2 - x1", "z", "x_rs"),
        doc="Quantities of EpileptorRestingState available to monitor.")

    state_variables = ("x1", "y1", "z", "x2", "y2", "g", "x_rs", "y_rs")

    _nvar = 8                                           # number of state-variables
    cvar = numpy.array([0, 3, 6], dtype=numpy.int32)    # coupling variables

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0,
                    array=numpy.array, where=numpy.where, concat=numpy.concatenate):

        y = state_variables
        ydot = numpy.empty_like(state_variables)

        # long-range coupling
        c_pop1 = coupling[0]
        c_pop2 = coupling[1]
        c_pop3 = coupling[2]

        # short-range (local) coupling
        Iext = self.Iext + local_coupling * y[0]
        lc_1 = local_coupling * y[6]

        # Epileptor's equations:
        # population 1
        if_ydot0 = - self.a * y[0] ** 2 + self.b * y[0]
        else_ydot0 = self.slope - y[3] + 0.6 * (y[2] - 4.0) ** 2
        ydot[0] = self.tt * (y[1] - y[2] + Iext + self.Kvf * c_pop1 + where(y[0] < 0., if_ydot0, else_ydot0) * y[0])
        ydot[1] = self.tt * (self.c - self.d * y[0] ** 2 - y[1])

        # energy
        if_ydot2 = - 0.1 * y[2] ** 7
        else_ydot2 = 0
        ydot[2] = self.tt * (
                    self.r * (4 * (y[0] - self.x0) - y[2] + where(y[2] < 0., if_ydot2, else_ydot2) + self.Ks * c_pop1))

        # population 2
        ydot[3] = self.tt * (
                    -y[4] + y[3] - y[3] ** 3 + self.Iext2 + self.bb * y[5] - 0.3 * (y[2] - 3.5) + self.Kf * c_pop2)
        if_ydot4 = 0
        else_ydot4 = self.aa * (y[3] + 0.25)
        ydot[4] = self.tt * ((-y[4] + where(y[3] < -0.25, if_ydot4, else_ydot4)) / self.tau)

        # filter
        ydot[5] = self.tt * (-0.01 * (y[5] - 0.1 * y[0]))  # 0.01 = \gamma

        # G2D's equations:
        ydot[6] = self.d_rs * self.tau_rs * (self.alpha_rs * y[7] - self.f_rs * y[6] ** 3 + self.e_rs * y[
            6] ** 2 + self.gamma_rs * self.I_rs + self.gamma_rs * self.K_rs * c_pop3 + lc_1)
        ydot[7] = self.d_rs * (self.a_rs + self.b_rs * y[6] - self.beta_rs * y[7]) / self.tau_rs

        # output: LFP
        self.output = self.p * (- y[0] + y[3]) + (1 - self.p) * y[6]

        return ydot

    def dfun(self, x, c, local_coupling=0.0):
        r"""
            Computes the derivatives of the state-variables of EpileptorRestingState
            with respect to time.
        """

        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        Iext = self.Iext + local_coupling * x[0, :, 0]
        lc_1 = local_coupling * x[6, :, 0]
        deriv = _numba_dfun(x_, c_,
                            self.x0, Iext, self.Iext2, self.a, self.b, self.slope, self.tt, self.Kvf,
                            self.c, self.d, self.r, self.Ks, self.Kf, self.aa, self.bb, self.tau,
                            self.tau_rs, self.I_rs, self.a_rs, self.b_rs, self.d_rs, self.e_rs, self.f_rs,
                            self.beta_rs, self.alpha_rs, self.gamma_rs, self.K_rs, lc_1)
        return deriv.T[..., numpy.newaxis]


@guvectorize([(float64[:],) * 31], '(n),(m)' + ',()' * 28 + '->(n)', nopython=True)
def _numba_dfun(y, c_pop,
                x0, Iext, Iext2, a, b, slope, tt, Kvf, c, d, r, Ks, Kf, aa, bb, tau,
                tau_rs, I_rs, a_rs, b_rs, d_rs, e_rs, f_rs, beta_rs, alpha_rs, gamma_rs, K_rs, lc_1,
                ydot):
    "Gufunc for EpileptorRestingState model equations."

    #long-range coupling
    c_pop1 = c_pop[0]
    c_pop2 = c_pop[1]
    c_pop3 = c_pop[2]

    # Epileptor equations
    #population 1
    if y[0] < 0.0:
        ydot[0] = - a[0] * y[0] ** 2 + b[0] * y[0]
    else:
        ydot[0] = slope[0] - y[3] + 0.6 * (y[2] - 4.0) ** 2
    ydot[0] = tt[0] * (y[1] - y[2] + Iext[0] + Kvf[0] * c_pop1 + ydot[0] * y[0])
    ydot[1] = tt[0] * (c[0] - d[0] * y[0] ** 2 - y[1])

    #energy
    if y[2] < 0.0:
        ydot[2] = - 0.1 * y[2] ** 7
    else:
        ydot[2] = 0.0
    ydot[2] = tt[0] * (r[0] * (4 * (y[0] - x0[0]) + ydot[2] - y[2]  + Ks[0] * c_pop1))

    #population 2
    ydot[3] = tt[0] * (-y[4] + y[3] - y[3] ** 3 + Iext2[0] + bb[0] * y[5] - 0.3 * (y[2] - 3.5) + Kf[0] * c_pop2)
    if y[3] < -0.25:
        ydot[4] = 0.0
    else:
        ydot[4] = aa[0] * (y[3] + 0.25)
    ydot[4] = tt[0] * ((-y[4] + ydot[4]) / tau[0])

    #filter
    ydot[5] = tt[0] * (-0.01 * (y[5] - 0.1 * y[0]))

    # Generic 2D equations
    ydot[6] = d_rs[0] * tau_rs[0] * (alpha_rs[0] * y[7] - f_rs[0] * y[6] ** 3 + e_rs[0] * y[6] ** 2 + gamma_rs[0] * I_rs[0] + gamma_rs[0] * K_rs[0] * c_pop3 + lc_1[0])
    ydot[7] = d_rs[0] * (a_rs[0] + b_rs[0] * y[6] - beta_rs[0] * y[7]) / tau_rs[0]

