# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details. You should have received a copy of the GNU
#  General Public License along with this program.  If not,
# see <http://www.gnu.org/licenses/>.
#
#
# CITATION: When using The Virtual Brain for scientific publications, please
# cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)

"""
Saggio codimension 3 Epileptor model

.. moduleauthor:: Len Spek

"""

import numpy

from .base import ModelNumbaDfun
from numba import guvectorize, float64, int64
from tvb.basic.neotraits.api import NArray, List, Range, Final


class EpileptorCodim3(ModelNumbaDfun):
    r"""
    .. [Saggioetal_2017] Saggio ML, Spiegler A, Bernard C, Jirsa VK.
    *Fast–Slow Bursters in the Unfolding of a High Codimension Singularity
    and the Ultra-slow Transitions of Classes.* Journal of Mathematical
    Neuroscience. 2017;7:7. doi:10.1186/s13408-017-0050-8.

    .. The Epileptor codim 3 model is a neural mass model which contains two
    subsystems acting at different timescales. For the fast subsystem we use
    the unfolding of a degenerate Takens-Bogdanov bifucation of codimension
    3. The slow subsystem steers the fast one back and forth along paths
    leading to bursting behavior. The model is able to produce almost all the
    classes of bursting predicted for systems with a planar fast subsystem.

    .. In this implementation the model can produce Hysteresis-Loop bursters
    of classes c0, c0', c2s, c3s, c4s, c10s, c11s, c2b, c4b, c8b, c14b and
    c16b as classified by [Saggioetal_2017] Table 2. The default model
    parameters correspond to class c2s.

    """

    mu1_start = NArray(
        label="mu1 start",
        default=numpy.array([-0.02285]),
        domain=Range(lo=-1.0, hi=1.0),
        doc="The bifurcation parameter mu1 at the offset point for the given class, default for class c2s "
            "(Saddle-Node at onset and Saddle-Homoclinic at offset)")

    mu2_start = NArray(
        label="mu2 start",
        default=numpy.array([0.3448]),
        domain=Range(lo=-1.0, hi=1.0),
        doc="The bifurcation mu2 parameter at the offset point for the given class, default for class c2s "
            "(Saddle-Node at onset and Saddle-Homoclinic at offset)")

    nu_start = NArray(
        label="nu start",
        default=numpy.array([0.2014]),
        domain=Range(lo=-1.0, hi=1.0),
        doc="The bifurcation nu parameter at the offset point for the given class, default for class c2s "
            "(Saddle-Node at onset and Saddle-Homoclinic at offset)")

    mu1_stop = NArray(
        label="mu1 stop",
        default=numpy.array([-0.07465]),
        domain=Range(lo=-1.0, hi=1.0),
        doc="The bifurcation mu1 parameter at the onset point for the given class, default for class c2s "
            "(Saddle-Node at onset and Saddle-Homoclinic at offset)")

    mu2_stop = NArray(
        label="mu2 stop",
        default=numpy.array([0.3351]),
        domain=Range(lo=-1.0, hi=1.0),
        doc="The bifurcation mu2 parameter at the onset point for the given class, default for class c2s "
            "(Saddle-Node at onset and Saddle-Homoclinic at offset)")

    nu_stop = NArray(
        label="nu stop",
        default=numpy.array([0.2053]),
        domain=Range(lo=-1.0, hi=1.0),
        doc="The bifurcation nu parameter at the onset point for the given class, default for class c2s "
            "(Saddle-Node at onset and Saddle-Homoclinic at offset)")

    b = NArray(
        label="b",
        default=numpy.array([1.0]),
        doc="Unfolding type of the degenerate Takens-Bogdanov bifurcation, default is a focus type")

    R = NArray(
        label="R",
        default=numpy.array([0.4]),
        domain=Range(lo=0.0, hi=2.5),
        doc="Radius in unfolding")

    c = NArray(
        label="c",
        default=numpy.array([0.001]),
        domain=Range(lo=0.0, hi=0.01),
        doc="Speed of the slow variable")

    dstar = NArray(
        label="dstar",
        default=numpy.array([0.3]),
        domain=Range(lo=-0.1, hi=0.5),
        doc="Threshold for the inversion of the slow variable")

    Ks = NArray(
        label="Ks",
        default=numpy.array([0.0]),
        doc="Slow permittivity coupling strength, the default is no coupling")

    N = NArray(
        dtype=int,
        label="N",
        default=numpy.array([1]),
        doc="The branch of the resting state, default is 1")

    modification = NArray(
        dtype=bool,
        label="modification",
        default=numpy.array([True]),
        doc="When modification is True, then use the modification to stabilise the system for negative values of "
            "dstar. If modification is False, then don't use the modification. The default value is True ")

    state_variable_range = Final(
        label="State variable ranges [lo, hi]",
        default={"x": numpy.array([0.4, 0.6]),
                 "y": numpy.array([-0.1, 0.1]),
                 "z": numpy.array([0.0, 0.15])},
        doc="Typical bounds on state variables.")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('x', 'y', 'z'),
        default=('x', 'z'),
        doc="Quantities available to monitor.")

    # state variables names
    state_variables = ('x', 'y', 'z')

    # number of state variables
    _nvar = 3
    cvar = numpy.array([0], dtype=numpy.int32)

    # If there are derived parameters from the predefined parameters, then initialize them to None
    G = None
    H = None
    L = None
    M = None

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The equations were taken from [Saggioetal_2017]
        cf. Eqns. (4) and (7), page 17

        The state variables x and y correspond to the fast subsystem and the
        state variable z corresponds to the slow subsystem.

            .. math::
                \dot{x} &= -y \\
                \dot{y} &= x^3 - \mu_2 x - \mu_1 - y(\nu + b x + x^2) \\
                \dot{z} &= -c(\sqrt{(x-x_s}^2+y^2} - d^*)

        If the bool modification is True, then the equation for zdot will
        been modified to ensure stability for negative dstar

            .. math::
                    \dot{z} = -c(\sqrt{(x-x_s}^2+y^2} - d^* + 0.1(z-0.5)^7)

        Where :math:`\mu_1, \mu_2` and :math:`\nu` lie on a great arc of a
        sphere of radius R parametrised by the unit vectors E and F.

            .. math::
                \begin{pmatrix}\mu_2 & -\mu_1 & \nu \end{pmatrix} = R(E \cos z + F \sin z)

        And where :math:`x_s` is the x-coordinate of the resting state
        (stable equilibrium). This is computed by finding the solution of

            .. math::
                x_s^3 - mu_2*x_s - mu_1 = 0

        And taking the branch which corresponds to the resting state.
        If :math:`x_s` is complex, we take the real part.

        """

        x = state_variables[0, :]
        y = state_variables[1, :]
        z = state_variables[2, :]

        # Computes the values of mu2,mu1 and nu given the great arc (E,F,R) and the value of the slow variable z
        mu2 = self.R * (self.E[0] * numpy.cos(z) + self.F[0] * numpy.sin(z))
        mu1 = -self.R * (self.E[1] * numpy.cos(z) + self.F[1] * numpy.sin(z))
        nu = self.R * (self.E[2] * numpy.cos(z) + self.F[2] * numpy.sin(z))

        # Computes x_s, which is the solution to x_s^3 - mu2*x_s - mu1 = 0
        if self.N == 1:
            xs = (mu1 / 2.0 + numpy.sqrt(
                mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (1.0 / 3.0) + (mu1 / 2.0 - numpy.sqrt(
                mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (1.0 / 3.0)
        elif self.N == 2:
            xs = -1.0 / 2.0 * (1.0 - 1j * 3 ** (1.0 / 2.0)) * (mu1 / 2.0 + numpy.sqrt(
                mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (1.0 / 3.0) - 1.0 / 2.0 * (
                1.0 + 1j * 3 ** (1.0 / 2.0)) * (mu1 / 2.0 - numpy.sqrt(mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (
                1.0 / 3.0)
        elif self.N == 3:
            xs = -1.0 / 2.0 * (1.0 + 1j * 3 ** (1.0 / 2.0)) * (mu1 / 2.0 + numpy.sqrt(
                mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (1.0 / 3.0) - 1.0 / 2.0 * (
                1.0 - 1j * 3 ** (1.0 / 2.0)) * (mu1 / 2.0 - numpy.sqrt(mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (
                1.0 / 3.0)
        xs = numpy.real(xs)

        xdot = -y
        ydot = x ** 3 - mu2 * x - mu1 - y * (nu + self.b * x + x ** 2)
        if self.modification:
            zdot = -self.c * (
                numpy.sqrt((x - xs) ** 2 + y ** 2) - self.dstar + 0.1 * (z - 0.5) ** 7 + self.Ks * coupling[0, :])
        else:
            zdot = -self.c * (numpy.sqrt(
                (x - xs) ** 2 + y ** 2) - self.dstar + self.Ks * coupling[0, :])

        derivative = numpy.array([xdot, ydot, zdot])
        return derivative

    def update_derived_parameters(self):
        r"""
        The equations were taken from [Saggioetal_2017]
        cf. Eqn. (7), page 17

        Here we parametrize the great arc which lies on a sphere of radius R
        between the points A and B, which are given by:

            .. math::
                A &= \begin{pmatrix}\mu_{2,start} & -\mu_{1,start} & \nu_{start} \end{pmatrix} \\
                B &= \begin{pmatrix}\mu_{2,stop} & -\mu_{1,stop} & \nu_{stop} \end{pmatrix}

        Then we parametrize this great arc with z as parameter by :math:`R(E \cos z + F \sin z)`
            where the unit vectors E and F are given by:

            .. math::
                E &= A/\|A\| \\
                F &= ((A \times B) \times A)/\|(A \times B) \times A\|
        """

        A = numpy.array(
            [self.mu2_start[0], -self.mu1_start[0], self.nu_start[0]])
        B = numpy.array([self.mu2_stop[0], -self.mu1_stop[0], self.nu_stop[0]])

        self.E = A / numpy.linalg.norm(A)
        self.F = numpy.cross(numpy.cross(A, B), A)
        self.F = self.F / numpy.linalg.norm(self.F)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        """"The dfun using numba for speed"""
        state_variables_ = state_variables.reshape(state_variables.shape[:-1]).T
        coupling_ = coupling.reshape(coupling.shape[:-1]).T
        derivative = _numba_dfun(state_variables_, coupling_, self.E[0], self.E[1], self.E[2], self.F[0], self.F[1],
                                 self.F[2], self.b, self.R, self.c, self.dstar, self.Ks, self.modification, self.N)
        return derivative.T[..., numpy.newaxis]


@guvectorize([(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
               float64[:], float64[:], float64[:], float64[:], float64[:], int64[:], int64[:], float64[:])],
             '(n),(m)' + ',()' * 13 + '->(n)', nopython=True)
def _numba_dfun(state_variables, coupling, E0, E1, E2, F0, F1, F2, b, R, c, dstar, Ks, modification, N, derivative):
    """Gufunction for the Epileptor Codim 3 model"""

    x = state_variables[0]
    y = state_variables[1]
    z = state_variables[2]

    # Computes the values of mu2,mu1 and nu given the great arc (E,F,R) and the value of the slow variable z
    mu2 = R[0] * (E0[0] * numpy.cos(z) + F0[0] * numpy.sin(z))
    mu1 = -R[0] * (E1[0] * numpy.cos(z) + F1[0] * numpy.sin(z))
    nu = R[0] * (E2[0] * numpy.cos(z) + F2[0] * numpy.sin(z))

    # Computes x_s, which is the solution to x_s^3 - mu2*x_s - mu1 = 0
    if N[0] == 1:
        xs = (mu1 / 2.0 + numpy.sqrt(
            mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (1.0 / 3.0) + (mu1 / 2.0 - numpy.sqrt(
            mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (1.0 / 3.0)
    elif N[0] == 2:
        xs = -1.0 / 2.0 * (1.0 - 1j * 3 ** (1.0 / 2.0)) * (mu1 / 2.0 + numpy.sqrt(
            mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (1.0 / 3.0) - 1.0 / 2.0 * (
            1.0 + 1j * 3 ** (1.0 / 2.0)) * (mu1 / 2.0 - numpy.sqrt(mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (
            1.0 / 3.0)
    elif N[0] == 3:
        xs = -1.0 / 2.0 * (1.0 + 1j * 3 ** (1.0 / 2.0)) * (mu1 / 2.0 + numpy.sqrt(
            mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (1.0 / 3.0) - 1.0 / 2.0 * (
            1.0 - 1j * 3 ** (1.0 / 2.0)) * (mu1 / 2.0 - numpy.sqrt(mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (
            1.0 / 3.0)
    xs = xs.real

    derivative[0] = -y
    derivative[1] = x ** 3 - mu2 * x - mu1 - y * (nu + b[0] * x + x ** 2)
    derivative[2] = -c[0] * (
        numpy.sqrt((x - xs) ** 2 + y ** 2) - dstar[0] + modification[0] * 0.1 * (z - 0.5) ** 7 + Ks[0] * coupling[0])


class EpileptorCodim3SlowMod(ModelNumbaDfun):
    r"""
    .. [Saggioetal_2017] Saggio ML, Spiegler A, Bernard C, Jirsa VK.
    *Fast–Slow Bursters in the Unfolding of a High Codimension Singularity
    and the Ultra-slow Transitions of Classes.* Journal of Mathematical
    Neuroscience. 2017;7:7. doi:10.1186/s13408-017-0050-8.

    .. The Epileptor codim 3 model is a neural mass model which contains two
    subsystems acting at different timescales. For the fast subsystem we use
    the unfolding of a degenerate Takens-Bogdanov bifucation of codimension
    3. The slow subsystem steers the fast one back and forth along these
    paths leading to bursting behavior. The model is able to produce almost
    all the classes of bursting predicted for systems with a planar fast
    subsystem.

    .. In this implementation the model can produce Hysteresis-Loop bursters
    of classes c0, c0', c2s, c3s, c4s, c10s, c11s, c2b, c4b, c8b, c14b and
    c16b as classified by [Saggioetal_2017] Table 2. Through ultra-slow
    modulation of the path through the parameter space we can switch between
    different classes of bursters.

    """

    mu1_Ain = NArray(
        label="mu1 Ain",
        default=numpy.array([0.05494]),
        domain=Range(lo=-1.0, hi=1.0),
        doc="The bifurcation parameter mu1 at the initial point at bursting offset.")

    mu2_Ain = NArray(
        label="mu2 Ain",
        default=numpy.array([0.2731]),
        domain=Range(lo=-1.0, hi=1.0),
        doc="The bifurcation parameter mu2 at the initial point at bursting offset.")

    nu_Ain = NArray(
        label="nu Ain",
        default=numpy.array([0.287]),
        domain=Range(lo=-1.0, hi=1.0),
        doc="The bifurcation parameter nu at the initial point at bursting offset.")

    mu1_Bin = NArray(
        label="mu1 Bin",
        default=numpy.array([-0.0461]),
        domain=Range(lo=-1.0, hi=1.0),
        doc="The bifurcation parameter mu1 at the initial point at bursting onset.")

    mu2_Bin = NArray(
        label="mu2 Bin",
        default=numpy.array([0.243]),
        domain=Range(lo=-1.0, hi=1.0),
        doc="The bifurcation parameter mu2 at the initial point at bursting onset.")

    nu_Bin = NArray(
        label="nu Bin",
        default=numpy.array([0.3144]),
        domain=Range(lo=-1.0, hi=1.0),
        doc="The bifurcation parameter nu at the initial point at bursting onset.")

    mu1_Aend = NArray(
        label="mu1 Aend",
        default=numpy.array([0.06485]),
        domain=Range(lo=-1.0, hi=1.0),
        doc="The bifurcation parameter mu1 at the initial point at bursting offset.")

    mu2_Aend = NArray(
        label="mu2 Aend",
        default=numpy.array([0.07337]),
        domain=Range(lo=-1.0, hi=1.0),
        doc="The bifurcation parameter mu2 at the initial point at bursting offset.")

    nu_Aend = NArray(
        label="nu Aend",
        default=numpy.array([-0.3878]),
        domain=Range(lo=-1.0, hi=1.0),
        doc="The bifurcation parameter nu at the initial point at bursting offset.")

    mu1_Bend = NArray(
        label="mu1 Bend",
        default=numpy.array([0.03676]),
        domain=Range(lo=-1.0, hi=1.0),
        doc="The bifurcation parameter mu1 at the initial point at bursting onset.")

    mu2_Bend = NArray(
        label="mu2 Bend",
        default=numpy.array([-0.02792]),
        domain=Range(lo=-1.0, hi=1.0),
        doc="The bifurcation parameter mu2 at the initial point at bursting onset.")

    nu_Bend = NArray(
        label="nu Bend",
        default=numpy.array([-0.3973]),
        domain=Range(lo=-1.0, hi=1.0),
        doc="The bifurcation parameter nu at the initial point at bursting onset.")

    b = NArray(
        label="b",
        default=numpy.array([1.0]),
        doc="Unfolding type of the degenerate Takens-Bogdanov bifurcation, default is a focus type")

    R = NArray(
        label="R",
        default=numpy.array([0.4]),
        domain=Range(lo=0.0, hi=2.5),
        doc="Radius in unfolding")

    c = NArray(
        label="c",
        default=numpy.array([0.002]),
        domain=Range(lo=0.0, hi=0.01),
        doc="Speed of the slow variable")

    cA = NArray(
        label="cA",
        default=numpy.array([0.0001]),
        domain=Range(lo=0.0, hi=0.001),
        doc="Speed of the ultra-slow transition of the initial point")

    cB = NArray(
        label="cB",
        default=numpy.array([0.00012]),
        domain=Range(lo=0.0, hi=0.001),
        doc="Speed of the ultra-slow transition of the final point")

    dstar = NArray(
        label="dstar",
        default=numpy.array([0.3]),
        domain=Range(lo=-0.1, hi=0.5),
        doc="Threshold for the inversion of the slow variable")

    Ks = NArray(
        label="Ks",
        default=numpy.array([0.0]),
        doc="Slow permittivity coupling strength, the default is no coupling")

    N = NArray(
        dtype=int,
        label="N",
        default=numpy.array([1]),
        doc="The branch of the resting state, default is 1")

    modification = NArray(
        dtype=bool,
        label="modification",
        default=numpy.array([True]),
        doc="When modification is True, then use the modification to stabilise the system for negative values of "
            "dstar. If modification is False, then don't use the modification. The default value is True ")

    state_variable_range = Final(
        label="State variable ranges [lo, hi]",
        default={"x": numpy.array([0.4, 0.6]),
                 "y": numpy.array([-0.1, 0.1]),
                 "z": numpy.array([0.0, 0.1]),
                 "uA": numpy.array([0.0, 0.0]),
                 "uB": numpy.array([0.0, 0.0])},
        doc="Typical bounds on state variables.")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('x', 'y', 'z'),
        default=('x', 'z'),
        doc="Quantities available to monitor.")

    # state variables names
    state_variables = ('x', 'y', 'z', 'uA', 'uB')

    # number of state variables
    _nvar = 5
    cvar = numpy.array([0], dtype=numpy.int32)

    # If there are derived parameters from the predefined parameters, then initialize them to None
    G = None
    H = None
    L = None
    M = None

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The equations were taken from [Saggioetal_2017]
        cf. Eqns. (4) and (7), page 17 and 21

        The state variables x and y correspond to the fast subsystem and the
        state variable z corresponds to the slow subsystem. The state
        variables uA and uB correspond to the transition of the offset and
        onset bifurcations.

            .. math::
                \dot{x} &= -y \\
                \dot{y} &= x^3 - \mu_2 x - \mu_1 - y(\nu + b x + x^2) \\
                \dot{z} &= -c(\sqrt{(x-x_s}^2+y^2} - d^*)\\
                \dot(uA) &= cA\\
                \dot(uB) &= cB\\

        If the bool modification is True, then the equation for zdot will
        been modified to ensure stability for negative dstar

            .. math::
                    \dot{z} = -c(\sqrt{(x-x_s}^2+y^2} - d^* + 0.1(z-0.5)^7)

        Where :math:`\mu_1, \mu_2` and :math:`\nu` lie on a great arc of a
        sphere of radius R parametrised by the unit vectors E and F.

            .. math::
                \begin{pmatrix}\mu_2 & -\mu_1 & \nu \end{pmatrix} = R(E \cos z + F \sin z)

        Where the unit vectors E and F are given by:

            .. math::
                E &= A/\|A\| \\
                F &= ((A \times B) \times A)/\|(A \times B) \times A\|

        The vectors A and B transition across a great arc of the same sphere
        of radius R parametrised by G, H and L, M respectively.

            .. math::
                A &= R(G \cos(uA) + H \sin(uA))
                B &= R(L \cos(uB) + M \sin(uB))

        Finally :math:`x_s` is the x-coordinate of the resting state
        (stable equilibrium). This is computed by finding the solution of

            .. math::
                x_s^3 - mu_2*x_s - mu_1 = 0

        And taking the branch which corresponds to the resting state.
        If :math:`x_s` is complex, we take the real part.

        """
        x = state_variables[0, :]
        y = state_variables[1, :]
        z = state_variables[2, :]
        uA = state_variables[3, :]
        uB = state_variables[4, :]

        A = self.R * (self.G * numpy.cos(uA) + self.H * numpy.sin(uA))
        B = self.R * (self.L * numpy.cos(uB) + self.M * numpy.sin(uB))

        E = A / (numpy.linalg.norm(A, axis=1)).reshape(-1, 1)
        C = numpy.cross(A,B)
        F = numpy.cross(numpy.cross(A, B), A)
        F = F / (numpy.linalg.norm(F, axis=1)).reshape(-1, 1)

        # Computes the values of mu2,mu1 and nu given the great arc (E,F,R) and the value of the slow variable z
        mu2 = self.R * (numpy.array([E[:, 0]]).T * numpy.cos(z) + numpy.array(
            [F[:, 0]]).T * numpy.sin(z))
        mu1 = -self.R * (numpy.array([E[:, 1]]).T * numpy.cos(z) + numpy.array(
            [F[:, 1]]).T * numpy.sin(z))
        nu = self.R * (numpy.array([E[:, 2]]).T * numpy.cos(z) + numpy.array(
            [F[:, 2]]).T * numpy.sin(z))

        # Computes x_s, which is the solution to x_s^3 - mu2*x_s - mu1 = 0
        if self.N == 1:
            xs = (mu1 / 2.0 + numpy.sqrt(
                mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (1.0 / 3.0) + (mu1 / 2.0 - numpy.sqrt(
                mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (1.0 / 3.0)
        elif self.N == 2:
            xs = -1.0 / 2.0 * (1.0 - 1j * 3 ** (1.0 / 2.0)) * (mu1 / 2.0 + numpy.sqrt(
                mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (1.0 / 3.0) - 1.0 / 2.0 * (
                1.0 + 1j * 3 ** (1.0 / 2.0)) * (mu1 / 2.0 - numpy.sqrt(mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (
                1.0 / 3.0)
        elif self.N == 3:
            xs = -1.0 / 2.0 * (1.0 + 1j * 3 ** (1.0 / 2.0)) * (mu1 / 2.0 + numpy.sqrt(
                mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (1.0 / 3.0) - 1.0 / 2.0 * (
                1.0 - 1j * 3 ** (1.0 / 2.0)) * (mu1 / 2.0 - numpy.sqrt(mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (
                1.0 / 3.0)
        xs = numpy.real(xs)

        # global coupling: To be implemented

        xdot = -y
        ydot = x ** 3 - mu2 * x - mu1 - y * (nu + self.b * x + x ** 2)
        if self.modification:
            zdot = -self.c * (
                numpy.sqrt((x - xs) ** 2 + y ** 2) - self.dstar + 0.1 * (
                    z - 0.5) ** 7 + self.Ks * coupling[0, :])
        else:
            zdot = -self.c * (numpy.sqrt((x - xs) ** 2 + y ** 2) - self.dstar + self.Ks * coupling[0, :])
        uAdot = numpy.full_like(uA, self.cA)
        uBdot = numpy.full_like(uB, self.cB)

        derivative = numpy.array([xdot, ydot, zdot, uAdot, uBdot])
        return derivative

    def update_derived_parameters(self):
        r"""
        The equations were adapted from [Saggioetal_2017]
        cf. Eqn. (7), page 17 and page 21

        We parametrize the great arc on the sphere of radius R between the
        points Ain and Aend with the vectors G and H. This great arc is used
        for the offset point of the burster, given by the vector A.

            .. math::
                G &= Ain/\|Ain\| \\
                H &= ((Ain \times Aend) \times Ain)/\|(Ain \times Aend) \times Ain\|

        We also parametrize the great arc on the sphere of radius R between the
        points Bin and Bend with the vectors L and M. This great arc is used
        for the onset point of the burster, given by the vector B.

            .. math::
                L &= Bin/\|Bin\| \\
                M &= ((Bin \times Bend) \times Bin)/\|(Bin \times Bend) \times Bin\|
        """

        Ain = numpy.array([self.mu2_Ain[0], -self.mu1_Ain[0], self.nu_Ain[0]])
        Bin = numpy.array([self.mu2_Bin[0], -self.mu1_Bin[0], self.nu_Bin[0]])
        Aend = numpy.array(
            [self.mu2_Aend[0], -self.mu1_Aend[0], self.nu_Aend[0]])
        Bend = numpy.array(
            [self.mu2_Bend[0], -self.mu1_Bend[0], self.nu_Bend[0]])

        self.G = Ain / numpy.linalg.norm(Ain)
        self.H = numpy.cross(numpy.cross(Ain, Aend), Ain)
        self.H = self.H / numpy.linalg.norm(self.H)

        self.L = Bin / numpy.linalg.norm(Bin)
        self.M = numpy.cross(numpy.cross(Bin, Bend), Bin)
        self.M = self.M / numpy.linalg.norm(self.M)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        """"The dfun using numba for speed"""
        state_variables_ = state_variables.reshape(state_variables.shape[:-1]).T
        coupling_ = coupling.reshape(coupling.shape[:-1]).T
        derivative = _numba_dfun_slowmod(state_variables_, coupling_, self.G[0], self.G[1], self.G[2], self.H[0],
                                         self.H[1], self.H[2], self.L[0], self.L[1], self.L[2], self.M[0], self.M[1],
                                         self.M[2], self.b, self.R, self.c, self.cA, self.cB, self.dstar, self.Ks,
                                         self.modification, self.N)
        return derivative.T[..., numpy.newaxis]


@guvectorize([(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
               float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
               float64[:], float64[:], float64[:], float64[:], float64[:], int64[:], int64[:], float64[:])],
             '(n),(m)' + ',()' * 21 + '->(n)', nopython=True)
def _numba_dfun_slowmod(state_variables, coupling, G0, G1, G2, H0, H1, H2, L0, L1, L2, M0, M1, M2, b, R, c, cA, cB,
                        dstar, Ks, modification, N, derivative):
    """Gufunction for the Epileptor Codim 3 model with ultra-slow modulation of classes"""

    x = state_variables[0]
    y = state_variables[1]
    z = state_variables[2]
    uA = state_variables[3]
    uB = state_variables[4]

    A = R[0] * (numpy.array([G0[0], G1[0], G2[0]]) * numpy.cos(uA) + numpy.array([H0[0], H1[0], H2[0]]) * numpy.sin(uA))
    B = R[0] * (numpy.array([L0[0], L1[0], L2[0]]) * numpy.cos(uB) + numpy.array([M0[0], M1[0], M2[0]]) * numpy.sin(uB))

    E = A / (numpy.linalg.norm(A))
    # Numba does not support numpy.cross so we compute the cross-product using the standard formula.
    C = numpy.array([A[1] * B[2] - A[2] * B[1], A[2] * B[0] - A[0] * B[2], A[0] * B[1] - A[1] * B[0]])
    F = numpy.array([C[1] * A[2] - C[2] * A[1], C[2] * A[0] - C[0] * A[2], C[0] * A[1] - C[1] * A[0]])
    F = F / (numpy.linalg.norm(F))

    # Computes the values of mu2,mu1 and nu given the great arc (E,F,R) and the value of the slow variable z
    mu2 = R[0] * (E[0] * numpy.cos(z) + F[0] * numpy.sin(z))
    mu1 = -R[0] * (E[1] * numpy.cos(z) + F[1] * numpy.sin(z))
    nu = R[0] * (E[2] * numpy.cos(z) + F[2] * numpy.sin(z))

    # Computes x_s, which is the solution to x_s^3 - mu2*x_s - mu1 = 0
    if N[0] == 1:
        xs = (mu1 / 2.0 + numpy.sqrt(
            mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (1.0 / 3.0) + (mu1 / 2.0 - numpy.sqrt(
            mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (1.0 / 3.0)
    elif N[0] == 2:
        xs = -1.0 / 2.0 * (1.0 - 1j * 3 ** (1.0 / 2.0)) * (mu1 / 2.0 + numpy.sqrt(
            mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (1.0 / 3.0) - 1.0 / 2.0 * (
            1.0 + 1j * 3 ** (1.0 / 2.0)) * (mu1 / 2.0 - numpy.sqrt(mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (
            1.0 / 3.0)
    elif N[0] == 3:
        xs = -1.0 / 2.0 * (1.0 + 1j * 3 ** (1.0 / 2.0)) * (mu1 / 2.0 + numpy.sqrt(
            mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (1.0 / 3.0) - 1.0 / 2.0 * (
            1.0 - 1j * 3 ** (1.0 / 2.0)) * (mu1 / 2.0 - numpy.sqrt(mu1 ** 2 / 4.0 - mu2 ** 3 / 27.0 + 0 * 1j)) ** (
            1.0 / 3.0)
    xs = xs.real

    derivative[0] = -y
    derivative[1] = x ** 3 - mu2 * x - mu1 - y * (nu + b[0] * x + x ** 2)
    derivative[2] = -c[0] * (
        numpy.sqrt((x - xs) ** 2 + y ** 2) - dstar[0] + modification[0] * 0.1 * (z - 0.5) ** 7 + Ks[0] * coupling[0])
    derivative[3] = cA[0]
    derivative[4] = cB[0]
