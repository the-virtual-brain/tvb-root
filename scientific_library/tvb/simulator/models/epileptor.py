# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Hindmarsh-Rose-Jirsa Epileptor model.

"""

from .base import ModelNumbaDfun, LOG, numpy, basic, arrays
from numba import guvectorize, float64

@guvectorize([(float64[:],) * 18], '(n),(m)' + ',()'*15 + '->(n)', nopython=True)
def _numba_dfun(y, c_pop, x0, Iext, Iext2, a, b, slope, tt, Kvf, c, d, r, Ks, Kf, aa, tau, ydot):
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
    ydot[2] = tt[0] * (r[0] * (4 * (y[0] - x0[0]) - y[2] + ydot[2] + Ks[0] * c_pop1))

    # population 2
    ydot[3] = tt[0] * (-y[4] + y[3] - y[3] ** 3 + Iext2[0] + 2 * y[5] - 0.3 * (y[2] - 3.5) + Kf[0] * c_pop2)
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

    .. automethod:: Epileptor.__init__

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

    _ui_name = "Epileptor"
    ui_configurable_parameters = ["Iext", "Iext2", "r", "x0", "slope"]

    a = arrays.FloatArray(
        label="a",
        default=numpy.array([1]),
        doc="Coefficient of the cubic term in the first state variable",
        order=-1)

    b = arrays.FloatArray(
        label="b",
        default=numpy.array([3]),
        doc="Coefficient of the squared term in the first state variabel",
        order=-1)

    c = arrays.FloatArray(
        label="c",
        default=numpy.array([1]),
        doc="Additive coefficient for the second state variable, \
        called :math:`y_{0}` in Jirsa paper",
        order=-1)

    d = arrays.FloatArray(
        label="d",
        default=numpy.array([5]),
        doc="Coefficient of the squared term in the second state variable",
        order=-1)

    r = arrays.FloatArray(
        label="r",
        range=basic.Range(lo=0.0, hi=0.001, step=0.00005),
        default=numpy.array([0.00035]),
        doc="Temporal scaling in the third state variable, \
        called :math:`1/\\tau_{0}` in Jirsa paper",
        order=4)

    s = arrays.FloatArray(
        label="s",
        default=numpy.array([4]),
        doc="Linear coefficient in the third state variable",
        order=-1)

    x0 = arrays.FloatArray(
        label="x0",
        range=basic.Range(lo=-3.0, hi=-1.0, step=0.1),
        default=numpy.array([-1.6]),
        doc="Epileptogenicity parameter",
        order=3)

    Iext = arrays.FloatArray(
        label="Iext",
        range=basic.Range(lo=1.5, hi=5.0, step=0.1),
        default=numpy.array([3.1]),
        doc="External input current to the first population",
        order=1)

    slope = arrays.FloatArray(
        label="slope",
        range=basic.Range(lo=-16.0, hi=6.0, step=0.1),
        default=numpy.array([0.]),
        doc="Linear coefficient in the first state variable",
        order=5)

    Iext2 = arrays.FloatArray(
        label="Iext2",
        range=basic.Range(lo=0.0, hi=1.0, step=0.05),
        default=numpy.array([0.45]),
        doc="External input current to the second population",
        order=2)

    tau = arrays.FloatArray(
        label="tau",
        default=numpy.array([10]),
        doc="Temporal scaling coefficient in fifth state variable",
        order=-1)

    aa = arrays.FloatArray(
        label="aa",
        default=numpy.array([6]),
        doc="Linear coefficient in fifth state variable",
        order=-1)

    Kvf = arrays.FloatArray(
        label="K_vf",
        default=numpy.array([0.0]),
        range=basic.Range(lo=0.0, hi=4.0, step=0.5),
        doc="Coupling scaling on a very fast time scale.",
        order=6)

    Kf = arrays.FloatArray(
        label="K_f",
        default=numpy.array([0.0]),
        range=basic.Range(lo=0.0, hi=4.0, step=0.5),
        doc="Correspond to the coupling scaling on a fast time scale.",
        order=7)

    Ks = arrays.FloatArray(
        label="K_s",
        default=numpy.array([0.0]),
        range=basic.Range(lo=-4.0, hi=4.0, step=0.1),
        doc="Permittivity coupling, that is from the fast time scale toward the slow time scale",
        order=8)

    tt = arrays.FloatArray(
        label="tt",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.001, hi=10.0, step=0.001),
        doc="Time scaling of the whole system",
        order=9)

    state_variable_range = basic.Dict(
        label="State variable ranges [lo, hi]",
        default={"x1": numpy.array([-2., 1.]),
                 "y1": numpy.array([-20., 2.]),
                 "z": numpy.array([2.0, 5.0]),
                 "x2": numpy.array([-2., 0.]),
                 "y2": numpy.array([0., 2.]),
                 "g": numpy.array([-1., 1.])},
        doc="Typical bounds on state variables in the Epileptor model.",
        order=16
        )

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=['x1', 'y1', 'z', 'x2', 'y2', 'g', 'x2 - x1'],
        default=["x2 - x1", 'z'],
        select_multiple=True,
        doc="Quantities of the Epileptor available to monitor.",
        order=100
    )

    state_variables = ['x1', 'y1', 'z', 'x2', 'y2', 'g']

    _nvar = 6
    cvar = numpy.array([0, 3], dtype=numpy.int32)

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0,
             array=numpy.array, where=numpy.where, concat=numpy.concatenate):
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

            .. math::
                f_{2}(x_{2}) =
                \begin{cases}
                0 & \text{if } x_{2} <-0.25\\
                a_{2}(x_{2} + 0.25) & \text{if } x_{2} \geq -0.25
                \end{cases}

        """
        y = state_variables
        ydot = numpy.empty_like(state_variables)

        Iext = self.Iext + local_coupling * y[0]
        c_pop1 = coupling[0, :]
        c_pop2 = coupling[1, :]

        # population 1
        if_ydot0 = - self.a*y[0]**2 + self.b*y[0]
        else_ydot0 = self.slope - y[3] + 0.6*(y[2]-4.0)**2
        ydot[0] = self.tt*(y[1] - y[2] + Iext + self.Kvf*c_pop1 + where(y[0] < 0., if_ydot0, else_ydot0) * y[0])
        ydot[1] = self.tt*(self.c - self.d*y[0]**2 - y[1])

        # energy
        if_ydot2 = - 0.1*y[2]**7
        else_ydot2 = 0
        ydot[2] = self.tt*(self.r * ( 4*(y[0] - self.x0) - y[2] + where(y[2] < 0., if_ydot2, else_ydot2) + self.Ks*c_pop1))

        # population 2
        ydot[3] = self.tt*(-y[4] + y[3] - y[3]**3 + self.Iext2 + 2*y[5] - 0.3*(y[2] - 3.5) + self.Kf*c_pop2)
        if_ydot4 = 0
        else_ydot4 = self.aa*(y[3] + 0.25)
        ydot[4] = self.tt*((-y[4] + where(y[3] < -0.25, if_ydot4, else_ydot4))/self.tau)

        # filter
        ydot[5] = self.tt*(-0.01*(y[5] - 0.1*y[0]))

        return ydot

    def dfun(self, x, c, local_coupling=0.0):
        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        Iext = self.Iext + local_coupling * x[0, :, 0]
        deriv = _numba_dfun(x_, c_,
                         self.x0, Iext, self.Iext2, self.a, self.b, self.slope, self.tt, self.Kvf,
                         self.c, self.d, self.r, self.Ks, self.Kf, self.aa, self.tau)
        return deriv.T[..., numpy.newaxis]