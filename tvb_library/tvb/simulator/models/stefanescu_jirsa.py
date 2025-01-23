# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2024, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Models developed by Stefanescu-Jirsa, based on reduced-set analyses of infinite populations.

"""
import numpy
from scipy.integrate import trapezoid as scipy_integrate_trapz
from scipy.stats import norm as scipy_stats_norm
from .base import Model
from tvb.basic.neotraits.api import NArray, Final, List, Range
from numba import njit


class ReducedSetBase(Model):
    number_of_modes = 3
    nu = 1500
    nv = 1500

    def configure(self):
        super(ReducedSetBase, self).configure()
        if numpy.mod(self.nv, self.number_of_modes):
            raise ValueError("nv (%d) must be divisible by the number_of_modes (%d), nu mod n_mode = %d",
                             self.nv, self.number_of_modes, self.nv % self.number_of_modes)
        if numpy.mod(self.nu, self.number_of_modes):
            raise ValueError("nu (%d) must be divisible by the number_of_modes (%d), nu mod n_mode = %d",
                             self.nu, self.number_of_modes, self.nu % self.number_of_modes)
        self.update_derived_parameters()


class ReducedSetFitzHughNagumo(ReducedSetBase):
    r"""
    A reduced representation of a set of Fitz-Hugh Nagumo oscillators,
    [SJ_2008]_.

    The models (:math:`\xi`, :math:`\eta`) phase-plane, including a
    representation of the vector field as well as its nullclines, using default
    parameters, can be seen below:

        .. _phase-plane-rFHN_0:
        .. figure :: img/ReducedSetFitzHughNagumo_01_mode_0_pplane.svg
            :alt: Reduced set of FitzHughNagumo phase plane (xi, eta), 1st mode.

            The (:math:`\xi`, :math:`\eta`) phase-plane for the first mode of
            a reduced set of Fitz-Hugh Nagumo oscillators.

        .. _phase-plane-rFHN_1:
        .. figure :: img/ReducedSetFitzHughNagumo_01_mode_1_pplane.svg
            :alt: Reduced set of FitzHughNagumo phase plane (xi, eta), 2nd mode.

            The (:math:`\xi`, :math:`\eta`) phase-plane for the second mode of
            a reduced set of Fitz-Hugh Nagumo oscillators.

        .. _phase-plane-rFHN_2:
        .. figure :: img/ReducedSetFitzHughNagumo_01_mode_2_pplane.svg
            :alt: Reduced set of FitzHughNagumo phase plane (xi, eta), 3rd mode.

            The (:math:`\xi`, :math:`\eta`) phase-plane for the third mode of
            a reduced set of Fitz-Hugh Nagumo oscillators.


    The system's equations for the i-th mode at node q are:

    .. math::
                \dot{\xi}_{i}    &=  c\left(\xi_i-e_i\frac{\xi_{i}^3}{3} -\eta_{i}\right)
                                  + K_{11}\left[\sum_{k=1}^{o} A_{ik}\xi_k-\xi_i\right]
                                  - K_{12}\left[\sum_{k =1}^{o} B_{i k}\alpha_k-\xi_i\right] + cIE_i \\
                                 &\, + \left[\sum_{k=1}^{o} \mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                  +  \left[\sum_{k=1}^{o} W_{\zeta}\cdot\xi_{kr} \right] \\
                \dot{\eta}_i     &= \frac{1}{c}\left(\xi_i-b\eta_i+m_i\right) \\
                & \\
                \dot{\alpha}_i   &= c\left(\alpha_i-f_i\frac{\alpha_i^3}{3}-\beta_i\right)
                                  + K_{21}\left[\sum_{k=1}^{o} C_{ik}\xi_i-\alpha_i\right] + cII_i \\
                                 & \, + \left[\sum_{k=1}^{o} \mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                  + \left[\sum_{k=1}^{o} W_{\zeta}\cdot\xi_{kr}\right] \\
                                 & \\
                \dot{\beta}_i    &= \frac{1}{c}\left(\alpha_i-b\beta_i+n_i\right)

    .. automethod:: ReducedSetFitzHughNagumo.update_derived_parameters

    #NOTE: In the Article this modelis called StefanescuJirsa2D

    """

    # Define traited attributes for this model, these represent possible kwargs.
    tau = NArray(
        label=r":math:`\tau`",
        default=numpy.array([3.0]),
        domain=Range(lo=1.5, hi=4.5, step=0.01),
        doc="""doc...(prob something about timescale seperation)""")

    a = NArray(
        label=":math:`a`",
        default=numpy.array([0.45]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""doc...""")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([0.9]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""doc...""")

    K11 = NArray(
        label=":math:`K_{11}`",
        default=numpy.array([0.5]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Internal coupling, excitatory to excitatory""")

    K12 = NArray(
        label=":math:`K_{12}`",
        default=numpy.array([0.15]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Internal coupling, inhibitory to excitatory""")

    K21 = NArray(
        label=":math:`K_{21}`",
        default=numpy.array([0.15]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Internal coupling, excitatory to inhibitory""")

    sigma = NArray(
        label=r":math:`\sigma`",
        default=numpy.array([0.35]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Standard deviation of Gaussian distribution""")

    mu = NArray(
        label=r":math:`\mu`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Mean of Gaussian distribution""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"xi": numpy.array([-4.0, 4.0]),
                 "eta": numpy.array([-3.0, 3.0]),
                 "alpha": numpy.array([-4.0, 4.0]),
                 "beta": numpy.array([-3.0, 3.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("xi", "eta", "alpha", "beta"),
        default=("xi", "alpha"),
        doc=r"""This represents the default state-variables of this Model to be
                monitored. It can be overridden for each Monitor if desired. The
                corresponding state-variable indices for this model are :math:`\xi = 0`,
                :math:`\eta = 1`, :math:`\alpha = 2`, and :math:`\beta= 3`.""")

    state_variables = tuple('xi eta alpha beta'.split())
    _nvar = 4
    cvar = numpy.array([0, 2], dtype=numpy.int32)
    # Derived parameters
    Aik = None
    Bik = None
    Cik = None
    e_i = None
    f_i = None
    IE_i = None
    II_i = None
    m_i = None
    n_i = None

    @njit
    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""


        The system's equations for the i-th mode at node q are:

        .. math::
                \dot{\xi}_{i}    &=  c\left(\xi_i-e_i\frac{\xi_{i}^3}{3} -\eta_{i}\right)
                                  + K_{11}\left[\sum_{k=1}^{o} A_{ik}\xi_k-\xi_i\right]
                                  - K_{12}\left[\sum_{k =1}^{o} B_{i k}\alpha_k-\xi_i\right] + cIE_i                       \\
                                 &\, + \left[\sum_{k=1}^{o} \mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                  +  \left[\sum_{k=1}^{o} W_{\zeta}\cdot\xi_{kr} \right] \\
                \dot{\eta}_i     &= \frac{1}{c}\left(\xi_i-b\eta_i+m_i\right)                                              \\
                & \\
                \dot{\alpha}_i   &= c\left(\alpha_i-f_i\frac{\alpha_i^3}{3}-\beta_i\right)
                                  + K_{21}\left[\sum_{k=1}^{o} C_{ik}\xi_i-\alpha_i\right] + cII_i                          \\
                                 & \, + \left[\sum_{k=1}^{o} \mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                  + \left[\sum_{k=1}^{o} W_{\zeta}\cdot\xi_{kr}\right] \\
                                 & \\
                \dot{\beta}_i    &= \frac{1}{c}\left(\alpha_i-b\beta_i+n_i\right)

        """

        xi = state_variables[0, :]
        eta = state_variables[1, :]
        alpha = state_variables[2, :]
        beta = state_variables[3, :]
        derivative = numpy.empty_like(state_variables)
        # sum the activity from the modes
        c_0 = coupling[0, :].sum(axis=1)[:, numpy.newaxis]

        # TODO: generalize coupling variables to a matrix form
        # c_1 = coupling[1, :] # this cv represents alpha

        N = len(xi)
        
        # Compute numpy.dot(xi, self.Aik) manually
        xi_Aik = numpy.zeros(N)
        for i in range(N):
            xi_Aik[i] = sum(xi[j] * self.Aik[j, i] for j in range(N))

        # Compute numpy.dot(alpha, self.Bik) manually
        alpha_Bik = numpy.zeros(N)
        for i in range(N):
            alpha_Bik[i] = sum(alpha[j] * self.Bik[j, i] for j in range(N))

        # Compute numpy.dot(xi, self.Cik) manually
        xi_Cik = numpy.zeros(N)
        for i in range(N):
            xi_Cik[i] = sum(xi[j] * self.Cik[j, i] for j in range(N))

        for i in range(N):
            derivative[0][i] = (self.tau * (xi[i] - self.e_i * xi[i] ** 3 / 3.0 - eta[i]) +
                                self.K11 * (xi_Aik[i] - xi[i]) -
                                self.K12 * (alpha_Bik[i] - xi[i]) +
                                self.tau * (self.IE_i + c_0 + local_coupling * xi[i]))

            derivative[1][i] = (xi[i] - self.b * eta[i] + self.m_i) / self.tau

            derivative[2][i] = (self.tau * (alpha[i] - self.f_i * alpha[i] ** 3 / 3.0 - beta[i]) +
                                self.K21 * (xi_Cik[i] - alpha[i]) +
                                self.tau * (self.II_i + c_0 + local_coupling * xi[i]))

            derivative[3][i] = (alpha[i] - self.b * beta[i] + self.n_i) / self.tau
            return derivative

    @njit
    def trapz_integrate(self,y, x):
        """Compute the trapezoidal integral explicitly."""
        result = 0.0
        for i in range(len(x) - 1):
            result += 0.5 * (x[i+1] - x[i]) * (y[i+1] + y[i])
        return result

    @njit
    def update_derived_parameters(self):
        """
        Calculate coefficients for the Reduced FitzHugh-Nagumo oscillator based
        neural field model. Specifically, this method implements equations for
        calculating coefficients found in the supplemental material of
        [SJ_2008]_.

        Include equations here...

        """

        stepu = 1.0 / (self.nu + 2 - 1)
        stepv = 1.0 / (self.nv + 2 - 1)

        norm = scipy_stats_norm(loc=self.mu, scale=self.sigma)
        
        # Generate Zu and Zv explicitly
        Zu = numpy.zeros(self.nu)
        Zv = numpy.zeros(self.nv)
        
        for i in range(self.nu):
            Zu[i] = norm.ppf((i + 1) * stepu)
        
        for i in range(self.nv):
            Zv[i] = norm.ppf((i + 1) * stepv)
        
        # Initialize U and V matrices
        V = numpy.zeros((self.number_of_modes, self.nv))
        U = numpy.zeros((self.number_of_modes, self.nu))

        nv_per_mode = self.nv // self.number_of_modes
        nu_per_mode = self.nu // self.number_of_modes

        # Assign ones in blocks
        for i in range(self.number_of_modes):
            for j in range(i * nv_per_mode, (i + 1) * nv_per_mode):
                V[i, j] = 1.0
            for j in range(i * nu_per_mode, (i + 1) * nu_per_mode):
                U[i, j] = 1.0

        # Normalize the modes using explicit integration
        for i in range(self.number_of_modes):
            V[i, :] /= numpy.sqrt(self.trapz_integrate(V[i, :] * V[i, :], Zv))
            U[i, :] /= numpy.sqrt(self.trapz_integrate(U[i, :] * U[i, :], Zu))

        # Compute normal PDFs
        g1 = numpy.zeros(self.nv)
        g2 = numpy.zeros(self.nu)

        for i in range(self.nv):
            g1[i] = norm.pdf(Zv[i])
        
        for i in range(self.nu):
            g2[i] = norm.pdf(Zu[i])

        # Preallocate matrices
        self.Aik = numpy.zeros((self.number_of_modes, self.number_of_modes))
        self.Bik = numpy.zeros((self.number_of_modes, self.number_of_modes))
        self.Cik = numpy.zeros((self.number_of_modes, self.number_of_modes))

        self.e_i = numpy.zeros((1, self.number_of_modes))
        self.f_i = numpy.zeros((1, self.number_of_modes))
        self.IE_i = numpy.zeros((1, self.number_of_modes))
        self.II_i = numpy.zeros((1, self.number_of_modes))
        self.m_i = numpy.zeros((self.number_of_modes, 1))
        self.n_i = numpy.zeros((self.number_of_modes, 1))

        # Compute conjugates
        cV = numpy.conj(V)
        cU = numpy.conj(U)

        # Compute integrations
        intcVdZ = numpy.zeros((self.number_of_modes, 1))
        intG1VdZ = numpy.zeros((1, self.number_of_modes))
        intcUdZ = numpy.zeros((self.number_of_modes, 1))

        for i in range(self.number_of_modes):
            intcVdZ[i, 0] = self.trapz_integrate(cV[i, :], Zv)
            intG1VdZ[0, i] = self.trapz_integrate(g1 * V[i, :], Zv)
            intcUdZ[i, 0] = self.trapz_integrate(cU[i, :], Zu)

        # Compute Aik, Bik, Cik using explicit loops
        for i in range(self.number_of_modes):
            for j in range(self.number_of_modes):
                self.Aik[i, j] = intcVdZ[i, 0] * intG1VdZ[0, j]
                self.Bik[i, j] = intcVdZ[i, 0] * self.trapz_integrate(g2 * U[j, :], Zu)
                self.Cik[i, j] = intcUdZ[i, 0] * intG1VdZ[0, j]

        # Compute e_i, f_i, IE_i, II_i
        for i in range(self.number_of_modes):
            self.e_i[0, i] = self.trapz_integrate(cV[i, :] * (V[i, :] ** 3), Zv)
            self.f_i[0, i] = self.trapz_integrate(cU[i, :] * (U[i, :] ** 3), Zu)
            self.IE_i[0, i] = self.trapz_integrate(Zv * cV[i, :], Zv)
            self.II_i[0, i] = self.trapz_integrate(Zu * cU[i, :], Zu)

        # Compute m_i and n_i
        for i in range(self.number_of_modes):
            self.m_i[i, 0] = self.a * intcVdZ[i, 0]
            self.n_i[i, 0] = self.a * intcUdZ[i, 0]
        # import pdb; pdb.set_trace()


class ReducedSetHindmarshRose(ReducedSetBase):
    r"""
    .. [SJ_2008] Stefanescu and Jirsa, PLoS Computational Biology, *A Low
        Dimensional Description of Globally Coupled Heterogeneous Neural
        Networks of Excitatory and Inhibitory*  4, 11, 26--36, 2008.

    The models (:math:`\xi`, :math:`\eta`) phase-plane, including a
    representation of the vector field as well as its nullclines, using default
    parameters, can be seen below:

        .. _phase-plane-rHR_0:
        .. figure :: img/ReducedSetHindmarshRose_01_mode_0_pplane.svg
            :alt: Reduced set of FitzHughNagumo phase plane (xi, eta), 1st mode.

            The (:math:`\xi`, :math:`\eta`) phase-plane for the first mode of
            a reduced set of Hindmarsh-Rose oscillators.

        .. _phase-plane-rHR_1:
        .. figure :: img/ReducedSetHindmarshRose_01_mode_1_pplane.svg
            :alt: Reduced set of FitzHughNagumo phase plane (xi, eta), 2nd mode.

            The (:math:`\xi`, :math:`\eta`) phase-plane for the second mode of
            a reduced set of Hindmarsh-Rose oscillators.

        .. _phase-plane-rHR_2:
        .. figure :: img/ReducedSetHindmarshRose_01_mode_2_pplane.svg
            :alt: Reduced set of FitzHughNagumo phase plane (xi, eta), 3rd mode.

            The (:math:`\xi`, :math:`\eta`) phase-plane for the third mode of
            a reduced set of Hindmarsh-Rose oscillators.


    The dynamic equations were orginally taken from [SJ_2008]_.

    The equations of the population model for i-th mode at node q are:

    .. math::
                \dot{\xi}_i     &=  \eta_i-a_i\xi_i^3 + b_i\xi_i^2- \tau_i
                                 + K_{11} \left[\sum_{k=1}^{o} A_{ik} \xi_k - \xi_i \right]
                                 - K_{12} \left[\sum_{k=1}^{o} B_{ik} \alpha_k - \xi_i\right] + IE_i \\
                                &\, + \left[\sum_{k=1}^{o} \mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                 + \left[\sum_{k=1}^{o} W_{\zeta}\cdot\xi_{kr} \right] \\
                & \\
                \dot{\eta}_i    &=  c_i-d_i\xi_i^2 -\tau_i \\
                & \\
                \dot{\tau}_i    &=  rs\xi_i - r\tau_i -m_i \\
                & \\
                \dot{\alpha}_i  &=  \beta_i - e_i \alpha_i^3 + f_i \alpha_i^2 - \gamma_i
                                 + K_{21} \left[\sum_{k=1}^{o} C_{ik} \xi_k - \alpha_i \right] + II_i \\
                                &\, +\left[\sum_{k=1}^{o}\mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                 + \left[\sum_{k=1}^{o}W_{\zeta}\cdot\xi_{kr}\right] \\
                & \\
                \dot{\beta}_i   &= h_i - p_i \alpha_i^2 - \beta_i \\
                \dot{\gamma}_i  &= rs \alpha_i - r \gamma_i - n_i

    .. automethod:: ReducedSetHindmarshRose.update_derived_parameters

    #NOTE: In the Article this modelis called StefanescuJirsa3D

    """

    # Define traited attributes for this model, these represent possible kwargs.
    r = NArray(
        label=":math:`r`",
        default=numpy.array([0.006]),
        domain=Range(lo=0.0, hi=0.1, step=0.0005),
        doc="""Adaptation parameter""")

    a = NArray(
        label=":math:`a`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Dimensionless parameter as in the Hindmarsh-Rose model""")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([3.0]),
        domain=Range(lo=0.0, hi=3.0, step=0.01),
        doc="""Dimensionless parameter as in the Hindmarsh-Rose model""")

    c = NArray(
        label=":math:`c`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Dimensionless parameter as in the Hindmarsh-Rose model""")

    d = NArray(
        label=":math:`d`",
        default=numpy.array([5.0]),
        domain=Range(lo=2.5, hi=7.5, step=0.01),
        doc="""Dimensionless parameter as in the Hindmarsh-Rose model""")

    s = NArray(
        label=":math:`s`",
        default=numpy.array([4.0]),
        domain=Range(lo=2.0, hi=6.0, step=0.01),
        doc="""Adaptation paramters, governs feedback""")

    xo = NArray(
        label=":math:`x_{o}`",
        default=numpy.array([-1.6]),
        domain=Range(lo=-2.4, hi=-0.8, step=0.01),
        doc="""Leftmost equilibrium point of x""")

    K11 = NArray(
        label=":math:`K_{11}`",
        default=numpy.array([0.5]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Internal coupling, excitatory to excitatory""")

    K12 = NArray(
        label=":math:`K_{12}`",
        default=numpy.array([0.1]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Internal coupling, inhibitory to excitatory""")

    K21 = NArray(
        label=":math:`K_{21}`",
        default=numpy.array([0.15]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Internal coupling, excitatory to inhibitory""")

    sigma = NArray(
        label=r":math:`\sigma`",
        default=numpy.array([0.3]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Standard deviation of Gaussian distribution""")

    mu = NArray(
        label=r":math:`\mu`",
        default=numpy.array([3.3]),
        domain=Range(lo=1.1, hi=3.3, step=0.01),
        doc="""Mean of Gaussian distribution""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"xi": numpy.array([-4.0, 4.0]),
                 "eta": numpy.array([-25.0, 20.0]),
                 "tau": numpy.array([2.0, 10.0]),
                 "alpha": numpy.array([-4.0, 4.0]),
                 "beta": numpy.array([-20.0, 20.0]),
                 "gamma": numpy.array([2.0, 10.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("xi", "eta", "tau", "alpha", "beta", "gamma"),
        default=("xi", "eta", "tau"),
        doc=r"""This represents the default state-variables of this Model to be
                monitored. It can be overridden for each Monitor if desired. The
                corresponding state-variable indices for this model are :math:`\xi = 0`,
                :math:`\eta = 1`, :math:`\tau = 2`, :math:`\alpha = 3`,
                :math:`\beta = 4`, and :math:`\gamma = 5`""")

    state_variables = 'xi eta tau alpha beta gamma'.split()
    _nvar = 6
    cvar = numpy.array([0, 3], dtype=numpy.int32)
    # derived parameters
    A_ik = None
    B_ik = None
    C_ik = None
    a_i = None
    b_i = None
    c_i = None
    d_i = None
    e_i = None
    f_i = None
    h_i = None
    p_i = None
    IE_i = None
    II_i = None
    m_i = None
    n_i = None

    @njit
    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The equations of the population model for i-th mode at node q are:

        .. math::
                \dot{\xi}_i     &=  \eta_i-a_i\xi_i^3 + b_i\xi_i^2- \tau_i
                                 + K_{11} \left[\sum_{k=1}^{o} A_{ik} \xi_k - \xi_i \right]
                                 - K_{12} \left[\sum_{k=1}^{o} B_{ik} \alpha_k - \xi_i\right] + IE_i \\
                                &\, + \left[\sum_{k=1}^{o} \mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                 + \left[\sum_{k=1}^{o} W_{\zeta}\cdot\xi_{kr} \right] \\
                & \\
                \dot{\eta}_i    &=  c_i-d_i\xi_i^2 -\tau_i \\
                & \\
                \dot{\tau}_i    &=  rs\xi_i - r\tau_i -m_i \\
                & \\
                \dot{\alpha}_i  &=  \beta_i - e_i \alpha_i^3 + f_i \alpha_i^2 - \gamma_i
                                 + K_{21} \left[\sum_{k=1}^{o} C_{ik} \xi_k - \alpha_i \right] + II_i \\
                                &\, +\left[\sum_{k=1}^{o}\mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                 + \left[\sum_{k=1}^{o}W_{\zeta}\cdot\xi_{kr}\right] \\
                & \\
                \dot{\beta}_i   &= h_i - p_i \alpha_i^2 - \beta_i \\
                \dot{\gamma}_i  &= rs \alpha_i - r \gamma_i - n_i

        """

        xi = state_variables[0, :]
        eta = state_variables[1, :]
        tau = state_variables[2, :]
        alpha = state_variables[3, :]
        beta = state_variables[4, :]
        gamma = state_variables[5, :]
        derivative = numpy.empty_like(state_variables)
        
        c_0 = numpy.sum(coupling[0, :], axis=1)[:, numpy.newaxis]

        for i in range(len(xi)):
            sum_Aik_xi = 0.0
            sum_Bik_alpha = 0.0
            sum_Cik_xi = 0.0
            
            for k in range(len(xi)):
                sum_Aik_xi += xi[k] * self.A_ik[i, k]
                sum_Bik_alpha += alpha[k] * self.B_ik[i, k]
                sum_Cik_xi += xi[k] * self.C_ik[i, k]
            
            derivative[0, i] = (eta[i] - self.a_i[i] * xi[i] ** 3 + self.b_i[i] * xi[i] ** 2 - tau[i] +
                                self.K11 * (sum_Aik_xi - xi[i]) -
                                self.K12 * (sum_Bik_alpha - xi[i]) + self.IE_i[i] +
                                c_0[i, 0] + local_coupling * xi[i])
            
            derivative[1, i] = self.c_i[i] - self.d_i[i] * xi[i] ** 2 - eta[i]
            
            derivative[2, i] = self.r * self.s * xi[i] - self.r * tau[i] - self.m_i[i]
            
            derivative[3, i] = (beta[i] - self.e_i[i] * alpha[i] ** 3 + self.f_i[i] * alpha[i] ** 2 - gamma[i] +
                                self.K21 * (sum_Cik_xi - alpha[i]) + self.II_i[i] +
                                c_0[i, 0] + local_coupling * xi[i])
            
            derivative[4, i] = self.h_i[i] - self.p_i[i] * alpha[i] ** 2 - beta[i]
            
            derivative[5, i] = self.r * self.s * alpha[i] - self.r * gamma[i] - self.n_i[i]
        
        return derivative

    @njit
    def trapezoidal_integral(self,y, x):
        """Manually compute the trapezoidal integral."""
        integral = 0.0
        for i in range(len(x) - 1):
            integral += 0.5 * (x[i + 1] - x[i]) * (y[i + 1] + y[i])
        return integral

    @njit
    def normalize_modes(self,modes, I_values):
        """Normalize the modes manually."""
        num_modes, size = modes.shape
        for i in range(num_modes):
            norm_factor = numpy.sqrt(self.trapezoidal_integral(modes[i] * modes[i], I_values))
            if norm_factor > 0:
                modes[i] /= norm_factor
        return modes

    @njit
    def manual_dot(self,vec1, mat):
        """Manually computes the dot product of a vector with a matrix."""
        result = numpy.zeros(mat.shape[1])
        for j in range(mat.shape[1]):
            for i in range(mat.shape[0]):
                result[j] += vec1[i] * mat[i, j]
        return result

    @njit
    def update_derived_parameters(self, corrected_d_p=True):
        """
        Calculate coefficients for the neural field model based on a Reduced set
        of Hindmarsh-Rose oscillators. Specifically, this method implements
        equations for calculating coefficients found in the supplemental
        material of [SJ_2008]_.

        Include equations here...

        """

        stepu = 1.0 / (self.nu + 2 - 1)
        stepv = 1.0 / (self.nv + 2 - 1)

        norm = self.norm_dist
        Iu = numpy.zeros(self.nu)
        Iv = numpy.zeros(self.nv)
        for i in range(self.nu):
            Iu[i] = norm.ppf((i + 1) * stepu)
        for i in range(self.nv):
            Iv[i] = norm.ppf((i + 1) * stepv)

        # Define the modes
        V = numpy.zeros((self.number_of_modes, self.nv))
        U = numpy.zeros((self.number_of_modes, self.nu))

        nv_per_mode = self.nv // self.number_of_modes
        nu_per_mode = self.nu // self.number_of_modes

        for i in range(self.number_of_modes):
            for j in range(i * nv_per_mode, (i + 1) * nv_per_mode):
                if j < self.nv:
                    V[i, j] = 1.0
            for j in range(i * nu_per_mode, (i + 1) * nu_per_mode):
                if j < self.nu:
                    U[i, j] = 1.0

        # Normalise the modes
        V = self.normalize_modes(V, Iv)
        U = self.normalize_modes(U, Iu)

        # Compute Gaussian PDFs
        g1 = numpy.zeros_like(Iv)
        g2 = numpy.zeros_like(Iu)
        for i in range(len(Iv)):
            g1[i] = norm.pdf(Iv[i])
        for i in range(len(Iu)):
            g2[i] = norm.pdf(Iu[i])

        # Compute conjugates
        cV = numpy.conj(V)
        cU = numpy.conj(U)

        # Compute integrals manually
        intcVdI = numpy.zeros((self.number_of_modes, 1))
        intcUdI = numpy.zeros((self.number_of_modes, 1))
        intG1VdI = numpy.zeros((1, self.number_of_modes))

        for i in range(self.number_of_modes):
            intcVdI[i, 0] = self.trapezoidal_integral(cV[i], Iv)
            intcUdI[i, 0] = self.trapezoidal_integral(cU[i], Iu)
            intG1VdI[0, i] = self.trapezoidal_integral(g1 * V[i], Iv)

        # Compute coefficients
        self.A_ik = self.manual_dot(intcVdI, intG1VdI).T
        self.B_ik = self.manual_dot(intcVdI, self.trapezoidal_integral(g2 * U, Iu)[numpy.newaxis, :])
        self.C_ik = self.manual_dot(intcUdI, intG1VdI).T

        self.a_i = self.a * self.trapezoidal_integral(cV * V ** 3, Iv)[numpy.newaxis, :]
        self.e_i = self.a * self.trapezoidal_integral(cU * U ** 3, Iu)[numpy.newaxis, :]
        self.b_i = self.b * self.trapezoidal_integral(cV * V ** 2, Iv)[numpy.newaxis, :]
        self.f_i = self.b * self.trapezoidal_integral(cU * U ** 2, Iu)[numpy.newaxis, :]
        self.c_i = (self.c * intcVdI).T
        self.h_i = (self.c * intcUdI).T

        self.IE_i = self.trapezoidal_integral(Iv * cV, Iv)[numpy.newaxis, :]
        self.II_i = self.trapezoidal_integral(Iu * cU, Iu)[numpy.newaxis, :]

        if corrected_d_p:
            # correction identified by Shrey Dutta & Arpan Bannerjee, confirmed by RS
            self.d_i = self.d * self.trapezoidal_integral(cV * V ** 2, Iv)[numpy.newaxis, :]
            self.p_i = self.d * self.trapezoidal_integral(cU * U ** 2, Iu)[numpy.newaxis, :]
        else:
            # typo in the original paper by RS & VJ, kept for comparison purposes.
            self.d_i = (self.d * intcVdI).T
            self.p_i = (self.d * intcUdI).T

        self.m_i = (self.r * self.s * self.xo * intcVdI).T
        self.n_i = (self.r * self.s * self.xo * intcUdI).T
