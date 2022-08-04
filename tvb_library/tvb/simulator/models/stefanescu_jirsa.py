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
Models developed by Stefanescu-Jirsa, based on reduced-set analyses of infinite populations.

"""
import numpy
from scipy.integrate import trapz as scipy_integrate_trapz
from scipy.stats import norm as scipy_stats_norm
from .base import Model
from tvb.basic.neotraits.api import NArray, Final, List, Range


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

        derivative[0] = (self.tau * (xi - self.e_i * xi ** 3 / 3.0 - eta) +
               self.K11 * (numpy.dot(xi, self.Aik) - xi) -
               self.K12 * (numpy.dot(alpha, self.Bik) - xi) +
               self.tau * (self.IE_i + c_0 + local_coupling * xi))

        derivative[1] = (xi - self.b * eta + self.m_i) / self.tau

        derivative[2] = (self.tau * (alpha - self.f_i * alpha ** 3 / 3.0 - beta) +
                  self.K21 * (numpy.dot(xi, self.Cik) - alpha) +
                  self.tau * (self.II_i + c_0 + local_coupling * xi))

        derivative[3] = (alpha - self.b * beta + self.n_i) / self.tau

        return derivative

    def update_derived_parameters(self):
        """
        Calculate coefficients for the Reduced FitzHugh-Nagumo oscillator based
        neural field model. Specifically, this method implements equations for
        calculating coefficients found in the supplemental material of
        [SJ_2008]_.

        Include equations here...

        """
        newaxis = numpy.newaxis
        trapz = scipy_integrate_trapz

        stepu = 1.0 / (self.nu + 2 - 1)
        stepv = 1.0 / (self.nv + 2 - 1)

        norm = scipy_stats_norm(loc=self.mu, scale=self.sigma)

        Zu = norm.ppf(numpy.arange(stepu, 1.0, stepu))
        Zv = norm.ppf(numpy.arange(stepv, 1.0, stepv))

        # Define the modes
        V = numpy.zeros((self.number_of_modes, self.nv))
        U = numpy.zeros((self.number_of_modes, self.nu))

        nv_per_mode = self.nv // self.number_of_modes
        nu_per_mode = self.nu // self.number_of_modes

        for i in range(self.number_of_modes):
            V[i, i * nv_per_mode:(i + 1) * nv_per_mode] = numpy.ones(nv_per_mode)
            U[i, i * nu_per_mode:(i + 1) * nu_per_mode] = numpy.ones(nu_per_mode)

        # Normalise the modes
        V = V / numpy.tile(numpy.sqrt(trapz(V * V, Zv, axis=1)), (self.nv, 1)).T
        U = U / numpy.tile(numpy.sqrt(trapz(U * U, Zu, axis=1)), (self.nv, 1)).T

        # Get Normal PDF's evaluated with sampling Zv and Zu
        g1 = norm.pdf(Zv)
        g2 = norm.pdf(Zu)
        G1 = numpy.tile(g1, (self.number_of_modes, 1))
        G2 = numpy.tile(g2, (self.number_of_modes, 1))

        cV = numpy.conj(V)
        cU = numpy.conj(U)

        intcVdZ = trapz(cV, Zv, axis=1)[:, newaxis]
        intG1VdZ = trapz(G1 * V, Zv, axis=1)[newaxis, :]
        intcUdZ = trapz(cU, Zu, axis=1)[:, newaxis]
        # import pdb; pdb.set_trace()
        # Calculate coefficients
        self.Aik = numpy.dot(intcVdZ, intG1VdZ).T
        self.Bik = numpy.dot(intcVdZ, trapz(G2 * U, Zu, axis=1)[newaxis, :])
        self.Cik = numpy.dot(intcUdZ, intG1VdZ).T

        self.e_i = trapz(cV * V ** 3, Zv, axis=1)[newaxis, :]
        self.f_i = trapz(cU * U ** 3, Zu, axis=1)[newaxis, :]

        self.IE_i = trapz(Zv * cV, Zv, axis=1)[newaxis, :]
        self.II_i = trapz(Zu * cU, Zu, axis=1)[newaxis, :]

        self.m_i = (self.a * intcVdZ).T
        self.n_i = (self.a * intcUdZ).T
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

        c_0 = coupling[0, :].sum(axis=1)[:, numpy.newaxis]
        # c_1 = coupling[1, :]

        derivative[0] = (eta - self.a_i * xi ** 3 + self.b_i * xi ** 2 - tau +
               self.K11 * (numpy.dot(xi, self.A_ik) - xi) -
               self.K12 * (numpy.dot(alpha, self.B_ik) - xi) +
               self.IE_i + c_0 + local_coupling * xi)

        derivative[1] = self.c_i - self.d_i * xi ** 2 - eta

        derivative[2] = self.r * self.s * xi - self.r * tau - self.m_i

        derivative[3] = (beta - self.e_i * alpha ** 3 + self.f_i * alpha ** 2 - gamma +
                  self.K21 * (numpy.dot(xi, self.C_ik) - alpha) +
                  self.II_i + c_0 + local_coupling * xi)

        derivative[4] = self.h_i - self.p_i * alpha ** 2 - beta

        derivative[5] = self.r * self.s * alpha - self.r * gamma - self.n_i

        return derivative

    def update_derived_parameters(self, corrected_d_p=True):
        """
        Calculate coefficients for the neural field model based on a Reduced set
        of Hindmarsh-Rose oscillators. Specifically, this method implements
        equations for calculating coefficients found in the supplemental
        material of [SJ_2008]_.

        Include equations here...

        """

        newaxis = numpy.newaxis
        trapz = scipy_integrate_trapz

        stepu = 1.0 / (self.nu + 2 - 1)
        stepv = 1.0 / (self.nv + 2 - 1)

        norm = scipy_stats_norm(loc=self.mu, scale=self.sigma)

        Iu = norm.ppf(numpy.arange(stepu, 1.0, stepu))
        Iv = norm.ppf(numpy.arange(stepv, 1.0, stepv))

        # Define the modes
        V = numpy.zeros((self.number_of_modes, self.nv))
        U = numpy.zeros((self.number_of_modes, self.nu))

        nv_per_mode = self.nv // self.number_of_modes
        nu_per_mode = self.nu // self.number_of_modes

        for i in range(self.number_of_modes):
            V[i, i * nv_per_mode:(i + 1) * nv_per_mode] = numpy.ones(nv_per_mode)
            U[i, i * nu_per_mode:(i + 1) * nu_per_mode] = numpy.ones(nu_per_mode)

        # Normalise the modes
        V = V / numpy.tile(numpy.sqrt(trapz(V * V, Iv, axis=1)), (self.nv, 1)).T
        U = U / numpy.tile(numpy.sqrt(trapz(U * U, Iu, axis=1)), (self.nu, 1)).T

        # Get Normal PDF's evaluated with sampling Zv and Zu
        g1 = norm.pdf(Iv)
        g2 = norm.pdf(Iu)
        G1 = numpy.tile(g1, (self.number_of_modes, 1))
        G2 = numpy.tile(g2, (self.number_of_modes, 1))

        cV = numpy.conj(V)
        cU = numpy.conj(U)

        #import pdb; pdb.set_trace()
        intcVdI = trapz(cV, Iv, axis=1)[:, newaxis]
        intG1VdI = trapz(G1 * V, Iv, axis=1)[newaxis, :]
        intcUdI = trapz(cU, Iu, axis=1)[:, newaxis]

        #Calculate coefficients
        self.A_ik = numpy.dot(intcVdI, intG1VdI).T
        self.B_ik = numpy.dot(intcVdI, trapz(G2 * U, Iu, axis=1)[newaxis, :])
        self.C_ik = numpy.dot(intcUdI, intG1VdI).T

        self.a_i = self.a * trapz(cV * V ** 3, Iv, axis=1)[newaxis, :]
        self.e_i = self.a * trapz(cU * U ** 3, Iu, axis=1)[newaxis, :]
        self.b_i = self.b * trapz(cV * V ** 2, Iv, axis=1)[newaxis, :]
        self.f_i = self.b * trapz(cU * U ** 2, Iu, axis=1)[newaxis, :]
        self.c_i = (self.c * intcVdI).T
        self.h_i = (self.c * intcUdI).T

        self.IE_i = trapz(Iv * cV, Iv, axis=1)[newaxis, :]
        self.II_i = trapz(Iu * cU, Iu, axis=1)[newaxis, :]

        if corrected_d_p:
            # correction identified by Shrey Dutta & Arpan Bannerjee, confirmed by RS
            self.d_i = self.d * trapz(cV * V ** 2, Iv, axis=1)[newaxis, :]
            self.p_i = self.d * trapz(cU * U ** 2, Iu, axis=1)[newaxis, :]
        else:
            # typo in the original paper by RS & VJ, kept for comparison purposes.
            self.d_i = (self.d * intcVdI).T
            self.p_i = (self.d * intcUdI).T

        self.m_i = (self.r * self.s * self.xo * intcVdI).T
        self.n_i = (self.r * self.s * self.xo * intcUdI).T
