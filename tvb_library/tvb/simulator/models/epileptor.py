# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
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
Hindmarsh-Rose-Jirsa Epileptor model.
"""
import numpy
from .base import ModelNumbaDfun
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, List, Range, Final


@guvectorize([(float64[:],) * 20], '(n),(m)' + ',()' * 17 + '->(n)', nopython=True)
def _numba_dfun(y, c_pop, x0, Iext, Iext2, a, b, slope, tt, Kvf, c, d, r, Ks, Kf, aa, bb, tau, modification, ydot):
    "Gufunc for Hindmarsh-Rose-Jirsa Epileptor model equations."

    c_pop1 = c_pop[0]
    c_pop2 = c_pop[1]

    # population 1
    if y[0] < 0.0:
        ydot[0] = - a[0] * y[0] ** 2 + b[0] * y[0]
    else:
        ydot[0] = slope[0] - y[3] + 0.6 * (y[2] - 4.0) ** 2
    ydot[0] = tt[0] * (y[1] - y[2] + Iext[0] + Kvf[0] * c_pop1 + ydot[0] * y[0])
    ydot[1] = tt[0] * (c[0] - d[0] * y[0] ** 2 - y[1])

    # energy
    if y[2] < 0.0:
        ydot[2] = - 0.1 * y[2] ** 7
    else:
        ydot[2] = 0.0
    if modification[0]:
        h = x0[0] + 3 / (1 + numpy.exp(-(y[0] + 0.5) / 0.1))
    else:
        h = 4 * (y[0] - x0[0]) + ydot[2]
    ydot[2] = tt[0] * (r[0] * (h - y[2] + Ks[0] * c_pop1))

    # population 2
    ydot[3] = tt[0] * (-y[4] + y[3] - y[3] ** 3 + Iext2[0] + bb[0] * y[5] - 0.3 * (y[2] - 3.5) + Kf[0] * c_pop2)
    if y[3] < -0.25:
        ydot[4] = 0.0
    else:
        ydot[4] = aa[0] * (y[3] + 0.25)
    ydot[4] = tt[0] * ((-y[4] + ydot[4]) / tau[0])

    # filter
    ydot[5] = tt[0] * (-0.01 * (y[5] - 0.1 * y[0]))


class Epileptor(ModelNumbaDfun):
    r"""
    The Epileptor is a composite neural mass model of six dimensions which
    has been crafted to model the phenomenology of epileptic seizures.
    (see [Jirsaetal_2014]_)
    Equations and default parameters are taken from [Jirsaetal_2014]_.
          +------------------------------------------------------+
          |                         Table 1                      |
          +----------------------+-------------------------------+
          |        Parameter     |           Value               |
          +======================+===============================+
          |         I_rest1      |              3.1              |
          +----------------------+-------------------------------+
          |         I_rest2      |              0.45             |
          +----------------------+-------------------------------+
          |         r            |            0.00035            |
          +----------------------+-------------------------------+
          |         x_0          |             -1.6              |
          +----------------------+-------------------------------+
          |         slope        |              0.0              |
          +----------------------+-------------------------------+
          |             Integration parameter                    |
          +----------------------+-------------------------------+
          |           dt         |              0.1              |
          +----------------------+-------------------------------+
          |  simulation_length   |              4000             |
          +----------------------+-------------------------------+
          |                    Noise                             |
          +----------------------+-------------------------------+
          |         nsig         | [0., 0., 0., 1e-3, 1e-3, 0.]  |
          +----------------------+-------------------------------+
          |              Jirsa et al. 2014                       |
          +------------------------------------------------------+
    .. figure :: img/Epileptor_01_mode_0_pplane.svg
        :alt: Epileptor phase plane
    .. [Jirsaetal_2014] Jirsa, V. K.; Stacey, W. C.; Quilichini, P. P.;
        Ivanov, A. I.; Bernard, C. *On the nature of seizure dynamics.* Brain,
        2014.

    Variables of interest to be used by monitors: -y[0] + y[3]

        .. math::
            \dot{x_{1}} &=& y_{1} - f_{1}(x_{1}, x_{2}) - z + I_{ext1} \\
            \dot{y_{1}} &=& c - d x_{1}^{2} - y{1} \\
            \dot{z} &=&
            \begin{cases}
            r(4 (x_{1} - x_{0}) - z-0.1 z^{7}) & \text{if } x<0 \\
            r(4 (x_{1} - x_{0}) - z) & \text{if } x \geq 0
            \end{cases} \\
            \dot{x_{2}} &=& -y_{2} + x_{2} - x_{2}^{3} + I_{ext2} + 0.002 g - 0.3 (z-3.5) \\
            \dot{y_{2}} &=& 1 / \tau (-y_{2} + f_{2}(x_{2}))\\
            \dot{g} &=& -0.01 (g - 0.1 x_{1})
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
    Note Feb. 2017: the slow permittivity variable can be modify to account for the time
    difference between interictal and ictal states (see [Proixetal_2014]).
    .. [Proixetal_2014] Proix, T.; Bartolomei, F; Chauvel, P; Bernard, C; Jirsa, V.K. *
        Permittivity coupling across brain regions determines seizure recruitment in
        partial epilepsy.* J Neurosci 2014, 34:15009-21.
    """

    a = NArray(
        label=":math:`a`",
        default=numpy.array([1.0]),
        doc="Coefficient of the cubic term in the first state variable")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([3.0]),
        doc="Coefficient of the squared term in the first state variabel")

    c = NArray(
        label=":math:`c`",
        default=numpy.array([1.0]),
        doc="Additive coefficient for the second state variable, \
        called :math:`y_{0}` in Jirsa paper")

    d = NArray(
        label=":math:`d`",
        default=numpy.array([5.0]),
        doc="Coefficient of the squared term in the second state variable")

    r = NArray(
        label=":math:`r`",
        domain=Range(lo=0.0, hi=0.001, step=0.00005),
        default=numpy.array([0.00035]),
        doc="Temporal scaling in the third state variable, \
        called :math:`1/\\tau_{0}` in Jirsa paper")

    s = NArray(
        label=":math:`s`",
        default=numpy.array([4.0]),
        doc="Linear coefficient in the third state variable")

    x0 = NArray(
        label=":math:`x_0`",
        domain=Range(lo=-3.0, hi=-1.0, step=0.1),
        default=numpy.array([-1.6]),
        doc="Epileptogenicity parameter")

    Iext = NArray(
        label=":math:`I_{ext}`",
        domain=Range(lo=1.5, hi=5.0, step=0.1),
        default=numpy.array([3.1]),
        doc="External input current to the first population")

    slope = NArray(
        label=":math:`slope`",
        domain=Range(lo=-16.0, hi=6.0, step=0.1),
        default=numpy.array([0.]),
        doc="Linear coefficient in the first state variable")

    Iext2 = NArray(
        label=":math:`I_{ext2}`",
        domain=Range(lo=0.0, hi=1.0, step=0.05),
        default=numpy.array([0.45]),
        doc="External input current to the second population")

    tau = NArray(
        label=r":math:`\tau`",
        default=numpy.array([10.0]),
        doc="Temporal scaling coefficient in fifth state variable")

    aa = NArray(
        label=":math:`aa`",
        default=numpy.array([6.0]),
        doc="Linear coefficient in fifth state variable")

    bb = NArray(
        label=":math:`bb`",
        default=numpy.array([2.0]),
        doc="Linear coefficient of lowpass excitatory coupling in fourth state variable")

    Kvf = NArray(
        label=":math:`K_{vf}`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=4.0, step=0.5),
        doc="Coupling scaling on a very fast time scale.")

    Kf = NArray(
        label=":math:`K_{f}`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=4.0, step=0.5),
        doc="Correspond to the coupling scaling on a fast time scale.")

    Ks = NArray(
        label=":math:`K_{s}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-4.0, hi=4.0, step=0.1),
        doc="Permittivity coupling, that is from the fast time scale toward the slow time scale")

    tt = NArray(
        label=":math:`K_{tt}`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.001, hi=10.0, step=0.001),
        doc="Time scaling of the whole system")

    modification = NArray(
        dtype=bool,
        label=":math:`modification`",
        default=numpy.array([False]),
        doc="When modification is True, then use nonlinear influence on z. \
        The default value is False, i.e., linear influence.")

    state_variable_range = Final(
        default={
            "x1": numpy.array([-2., 1.]),
            "y1": numpy.array([-20., 2.]),
            "z": numpy.array([2.0, 5.0]),
            "x2": numpy.array([-2., 0.]),
            "y2": numpy.array([0., 2.]),
            "g": numpy.array([-1., 1.])
        },
        label="State variable ranges [lo, hi]",
        doc="Typical bounds on state variables in the Epileptor model.")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('x1', 'y1', 'z', 'x2', 'y2', 'g', 'x2 - x1'),
        default=("x2 - x1", 'z'),
        doc="Quantities of the Epileptor available to monitor.",
    )

    state_variables = ('x1', 'y1', 'z', 'x2', 'y2', 'g')

    _nvar = 6
    cvar = numpy.array([0, 3], dtype=numpy.int32)  # should these not be constant Attr's?
    cvar.setflags(write=False)  # todo review this

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0,
                    array=numpy.array, where=numpy.where, concat=numpy.concatenate):

        y = state_variables
        ydot = numpy.empty_like(state_variables)

        Iext = self.Iext + local_coupling * y[0]
        c_pop1 = coupling[0, :]
        c_pop2 = coupling[1, :]

        # population 1
        if_ydot0 = - self.a * y[0] ** 2 + self.b * y[0]
        else_ydot0 = self.slope - y[3] + 0.6 * (y[2] - 4.0) ** 2
        ydot[0] = self.tt * (y[1] - y[2] + Iext + self.Kvf * c_pop1 + where(y[0] < 0., if_ydot0, else_ydot0) * y[0])
        ydot[1] = self.tt * (self.c - self.d * y[0] ** 2 - y[1])

        # energy
        if_ydot2 = - 0.1 * y[2] ** 7
        else_ydot2 = 0
        if self.modification:
            h = self.x0 + 3. / (1. + numpy.exp(- (y[0] + 0.5) / 0.1))
        else:
            h = 4 * (y[0] - self.x0) + where(y[2] < 0., if_ydot2, else_ydot2)
        ydot[2] = self.tt * (self.r * (h - y[2] + self.Ks * c_pop1))

        # population 2
        ydot[3] = self.tt * (
                -y[4] + y[3] - y[3] ** 3 + self.Iext2 + self.bb * y[5] - 0.3 * (y[2] - 3.5) + self.Kf * c_pop2)
        if_ydot4 = 0
        else_ydot4 = self.aa * (y[3] + 0.25)
        ydot[4] = self.tt * ((-y[4] + where(y[3] < -0.25, if_ydot4, else_ydot4)) / self.tau)

        # filter
        ydot[5] = self.tt * (-0.01 * (y[5] - 0.1 * y[0]))

        return ydot

    def dfun(self, x, c, local_coupling=0.0):
        r"""
        Computes the derivatives of the state variables of the Epileptor
        with respect to time.
        Implementation note: we expect this version of the Epileptor to be used
        in a vectorized manner. Concretely, y has a shape of (6, n) where n is
        the number of nodes in the network. An consequence is that
        the original use of if/else is translated by calculated both the true
        and false forms and mixing them using a boolean mask.
        Variables of interest to be used by monitors: -y[0] + y[3]
            .. math::
                \dot{x_{1}} &=& y_{1} - f_{1}(x_{1}, x_{2}) - z + I_{ext1} \\
                \dot{y_{1}} &=& c - d x_{1}^{2} - y{1} \\
                \dot{z} &=&
                \begin{cases}
                r(4 (x_{1} - x_{0}) - z-0.1 z^{7}) & \text{if } x<0 \\
                r(4 (x_{1} - x_{0}) - z) & \text{if } x \geq 0
                \end{cases} \\
                \dot{x_{2}} &=& -y_{2} + x_{2} - x_{2}^{3} + I_{ext2} + 0.002 g - 0.3 (z-3.5) \\
                \dot{y_{2}} &=& 1 / \tau (-y_{2} + f_{2}(x_{2}))\\
                \dot{g} &=& -0.01 (g - 0.1 x_{1})
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
        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        Iext = self.Iext + local_coupling * x[0, :, 0]
        deriv = _numba_dfun(x_, c_,
                            self.x0, Iext, self.Iext2, self.a, self.b, self.slope, self.tt, self.Kvf,
                            self.c, self.d, self.r, self.Ks, self.Kf, self.aa, self.bb, self.tau, self.modification)
        return deriv.T[..., numpy.newaxis]


class Epileptor2D(ModelNumbaDfun):
    r"""
        Two-dimensional reduction of the Epileptor.

        .. moduleauthor:: courtiol.julie@gmail.com

        Taking advantage of time scale separation and focusing on the slower time scale,
        the five-dimensional Epileptor reduces to a two-dimensional system (see [Proixetal_2014,
        Proixetal_2017]).

        Note: the slow permittivity variable can be modify to account for the time
        difference between interictal and ictal states (see [Proixetal_2014]).

        Equations and default parameters are taken from [Proixetal_2014]:

        .. math::
            \dot{x_{1,i}} &=& - x_{1,i}^{3} - 2x_{1,i}^{2}  + 1 - z_{i} + I_{ext1,i} \\
            \dot{z_{i}} &=& r(h - z_{i})

        with
        .. math::
            h =
            \begin{cases}
            x_{0} + 3 / (exp((x_{1} + 0.5)/0.1)) & \text{if } modification\\
            4 (x_{1,i} - x_{0}) & \text{else }
            \end{cases}
        References:
            [Proixetal_2014] Proix, T.; Bartolomei, F; Chauvel, P; Bernard, C; Jirsa, V.K. *
            Permittivity coupling across brain regions determines seizure recruitment in
            partial epilepsy.* J Neurosci 2014, 34:15009-21.

            [Proixetal_2017] Proix, T.; Bartolomei, F; Guye, M.; Jirsa, V.K. *Individual brain
            structure and modelling predict seizure propagation.* Brain 2017, 140; 641–654.
    """

    a = NArray(
        label=":math:`a`",
        default=numpy.array([1.0]),
        doc="Coefficient of the cubic term in the first state-variable.")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([3.0]),
        doc="Coefficient of the squared term in the first state-variable.")

    c = NArray(
        label=":math:`c`",
        default=numpy.array([1.0]),
        doc="Additive coefficient for the second state-variable x_{2}, \
        called :math:`y_{0}` in Jirsa paper.")

    d = NArray(
        label=":math:`d`",
        default=numpy.array([5.0]),
        doc="Coefficient of the squared term in the second state-variable x_{2}.")

    r = NArray(
        label=":math:`r`",
        domain=Range(lo=0.0, hi=0.001, step=0.00005),
        default=numpy.array([0.00035]),
        doc=r"Temporal scaling in the slow state-variable, \
        called :math:`1\tau_{0}` in Jirsa paper (see class Epileptor).")

    x0 = NArray(
        label=":math:`x_0`",
        domain=Range(lo=-3.0, hi=-1.0, step=0.1),
        default=numpy.array([-1.6]),
        doc="Epileptogenicity parameter.")

    Iext = NArray(
        label=":math:`I_{ext}`",
        domain=Range(lo=1.5, hi=5.0, step=0.1),
        default=numpy.array([3.1]),
        doc="External input current to the first state-variable.")

    slope = NArray(
        label=":math:`slope`",
        domain=Range(lo=-16.0, hi=6.0, step=0.1),
        default=numpy.array([0.]),
        doc="Linear coefficient in the first state-variable.")

    Kvf = NArray(
        label=":math:`K_{vf}`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=4.0, step=0.5),
        doc="Coupling scaling on a very fast time scale.")

    Ks = NArray(
        label=":math:`K_{s}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-4.0, hi=4.0, step=0.1),
        doc="Permittivity coupling, that is from the fast time scale toward the slow time scale.")

    tt = NArray(
        label=":math:`tt`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.001, hi=1.0, step=0.001),
        doc="Time scaling of the whole system to the system in real time.")

    modification = NArray(
        dtype=bool,
        label=":math:`modification`",
        default=numpy.array([False]),
        doc="When modification is True, then use nonlinear influence on z. \
        The default value is False, i.e., linear influence.")

    state_variable_range = Final(
        default={"x1": numpy.array([-2., 1.]), "z": numpy.array([2.0, 5.0])},
        label="State variable ranges [lo, hi]",
        doc="Typical bounds on state-variables in the Epileptor 2D model.")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('x1', 'z'),
        default=('x1',),
        doc="Quantities of the Epileptor 2D available to monitor.")

    state_variables = ('x1', 'z')

    _nvar = 2
    cvar = numpy.array([0], dtype=numpy.int32)

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0,
                    array=numpy.array, where=numpy.where, concat=numpy.concatenate):

        y = state_variables
        ydot = numpy.empty_like(state_variables)

        Iext = self.Iext + local_coupling * y[0]
        c_pop = coupling[0, :]

        # population 1
        if_ydot0 = self.a * y[0] ** 2 + (self.d - self.b) * y[0]
        else_ydot0 = - self.slope - 0.6 * (y[1] - 4.0) ** 2 + self.d * y[0]

        ydot[0] = self.tt * (self.c - y[1] + Iext + self.Kvf * c_pop - (where(y[0] < 0., if_ydot0, else_ydot0)) * y[0])

        # energy
        if_ydot1 = - 0.1 * y[1] ** 7
        else_ydot1 = 0

        if self.modification:
            h = self.x0 + 3. / (1. + numpy.exp(- (y[0] + 0.5) / 0.1))
        else:
            h = 4 * (y[0] - self.x0) + where(y[1] < 0., if_ydot1, else_ydot1)

        ydot[1] = self.tt * (self.r * (h - y[1] + self.Ks * c_pop))

        return ydot

    def dfun(self, x, c, local_coupling=0.0):
        r"""
        Computes the derivatives of the state-variables of the Epileptor 2D
        with respect to time.
        Equations and default parameters are taken from [Proixetal_2014]:

        .. math::
            \dot{x_{1,i}} &=& - x_{1,i}^{3} - 2x_{1,i}^{2}  + 1 - z_{i} + I_{ext1,i} \\
            \dot{z_{i}} &=& r(h - z_{i})

        with

        .. math::
            h =
            \begin{cases}
            x_{0} + 3 / (exp((x_{1} + 0.5)/0.1)) & \text{if } modification\\
            4 (x_{1,i} - x_{0}) & \text{else }
            \end{cases}
        """

        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        Iext = self.Iext + local_coupling * x[0, :, 0]
        deriv = _numba_dfun_epi2d(x_, c_,
                                  self.x0, Iext, self.a, self.b, self.slope, self.c,
                                  self.d, self.r, self.Kvf, self.Ks, self.tt, self.modification)
        return deriv.T[..., numpy.newaxis]


@guvectorize([(float64[:],) * 15], '(n),(m)' + ',()' * 12 + '->(n)', nopython=True)
def _numba_dfun_epi2d(y, c_pop, x0, Iext, a, b, slope, c, d, r, Kvf, Ks, tt, modification, ydot):
    "Gufunction for Epileptor 2D model equations."

    c_pop = c_pop[0]

    # population 1
    if y[0] < 0.0:
        ydot[0] = a[0] * y[0] ** 2 + (d[0] - b[0]) * y[0]
    else:
        ydot[0] = - slope[0] - 0.6 * (y[1] - 4.0) ** 2 + d[0] * y[0]
    ydot[0] = tt[0] * (c[0] - y[1] + Iext[0] + Kvf[0] * c_pop - ydot[0] * y[0])

    # energy
    if y[1] < 0.0:
        ydot[1] = - 0.1 * y[1] ** 7
    else:
        ydot[1] = 0.0

    if modification[0]:
        h = x0[0] + 3. / (1. + numpy.exp(- (y[0] + 0.5) / 0.1))
    else:
        h = 4 * (y[0] - x0[0]) + ydot[1]

    ydot[1] = tt[0] * (r[0] * (h - y[1] + Ks[0] * c_pop))
