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
#
#

"""
Defines a set of integration methods for both deterministic and stochastic
differential equations.

Using an integration step size ``dt`` from the following list:
    [0.244140625, 0.1220703125, 0.06103515625, 0.048828125, 0.0244140625, 0.01220703125, 0.009765625, 0.006103515625, 0.0048828125]

will be consistent with Monitor periods corresponding to any of [4096, 2048, 1024, 512, 256, 128] Hz

# TODO: error analysis

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>
.. moduleauthor:: Noelia Montejo <Noelia@tvb.invalid>

"""
import abc
import functools
import scipy.integrate
from . import noise
from .common import get_logger, simple_gen_astr
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, Float

LOG = get_logger(__name__)


class Integrator(HasTraits):
    """
    The Integrator class is a base class for the integration methods...

    .. [1] Kloeden and Platen, Springer 1995, *Numerical solution of stochastic
        differential equations.*

    .. [2] Riccardo Mannella, *Integration of Stochastic Differential Equations
        on a Computer*, Int J. of Modern Physics C 13(9): 1177--1194, 2002.

    .. [3] R. Mannella and V. Palleschi, *Fast and precise algorithm for 
        computer simulation of stochastic differential equations*, Phys. Rev. A
        40: 3381, 1989.

    """

    dt = Float(
        label="Integration-step size (ms)",
        default=0.01220703125, #0.015625,
        #range = basic.Range(lo= 0.0048828125, hi=0.244140625, step= 0.1, base=2.)  mh: was commented
        required=True,
        doc="""The step size used by the integration routine in ms. This
        should be chosen to be small enough for the integration to be
        numerically stable. It is also necessary to consider the desired sample
        period of the Monitors, as they are restricted to being integral
        multiples of this value. The default value is set such that all built-in
        models are numerically stable with there default parameters and because
        it is consitent with Monitors using sample periods corresponding to
        powers of 2 from 128 to 4096Hz."""
    )

    bounded_state_variable_indices = NArray(
        dtype=int,
        label="indices of the state variables to be bounded by the integrators "
              "within the boundaries in the boundaries' values array",
        required=False)

    state_variable_boundaries = NArray(
        label="The boundary values of the state variables",
        required=False)

    clamped_state_variable_indices = NArray(
        dtype=int,
        label="indices of the state variables to be clamped by the integrators to the values in the clamped_values array",
        required=False)

    clamped_state_variable_values = NArray(
        label="The values of the state variables which are clamped ",
        required=False)


    @abc.abstractmethod
    def scheme(self, X, dfun, coupling, local_coupling, stimulus):
        """
        The scheme of integrator should take a state and provide the next
        state in time, e.g. for a differential equation, scheme should take
        :math:`X` and provide an appropriate :math:`X + dX` (dfun in the code).

        """

    def bound_state(self, X):
        for sv_ind, sv_bounds in \
                zip(self.bounded_state_variable_indices,
                    self.state_variable_boundaries):
            if sv_bounds[0] is not None:
                X[sv_ind][X[sv_ind] < sv_bounds[0]] = sv_bounds[0]
            if sv_bounds[1] is not None:
                X[sv_ind][X[sv_ind] > sv_bounds[1]] = sv_bounds[1]

    def clamp_state(self, X):
        X[self.clamped_state_variable_indices] = self.clamped_state_variable_values

    def __str__(self):
        return simple_gen_astr(self, 'dt')


class IntegratorStochastic(Integrator):
    r"""
    The IntegratorStochastic class is a base class for the stochastic
    integration methods. It derives from the Integrator abstract base class.

    We consider a stochastic differential equation has the generic form:

        .. math::
            \dot{X}_i(t) = dX_i(\vec{X}) + g_i(\vec{X})  \xi(t)

    where we assume that the stochastic process :math:`\xi` is a Gaussian
    forcing. In the deterministic case, :math:`g(X)` would be zero. The full
    algorithm, for one external stochastic forcing which is additive, is:

        .. math::
            X_i(t) = X_i(0) + g_i(X) Z_1(t) + dX_i t + Z_2(t) dX_{i,k} g_k(X) +
                    0.5 dX_{i,jk} g_j(X) g_k(X) Z_3(t) + 0.5 t^2 dX_{i,j} dX_j

    where :math:`Z_1`, :math:`Z_2` and :math:`Z_3` are Gaussian variables,
    assuming the Einstein notation and defining:

        .. math::
            dX_{i,j} = \frac{\partial dX_i}{\partial X_j}

    """

    noise = Attr(
        field_type=noise.Noise,
        label = "Integration Noise",
        default=noise.Additive(),
        required = True,
        doc = """The stochastic integrator's noise source. It incorporates its
        own instance of Numpy's RandomState.""")  # type: noise.Noise

    def __str__(self):
        return simple_gen_astr(self, 'dt noise')


class HeunDeterministic(Integrator):
    """
    It is a simple example of a predictor-corrector method. It is also known as
    modified trapezoidal method, which uses the Euler method as its predictor.
    And it is also a implicit integration scheme.

    """

    _ui_name = "Heun"

    def scheme(self, X, dfun, coupling, local_coupling, stimulus):
        r"""
        From [1]_:

        .. math::
            X_{n+1} &= X_n + dt (dX(t_n, X_n) + 
                                 dX(t_{n+1}, \tilde{X}_{n+1})) / 2 \\
            \tilde{X}_{n+1} &= X_n + dt dX(t_n, X_n)

        cf. Equation 1.11, page 283.

        """
        #import pdb; pdb.set_trace()
        m_dx_tn = dfun(X, coupling, local_coupling)
        inter = X + self.dt * (m_dx_tn + stimulus)
        if self.state_variable_boundaries is not None:
            self.bound_state(inter)
        if self.clamped_state_variable_values is not None:
            self.clamp_state(inter)

        dX = (m_dx_tn + dfun(inter, coupling, local_coupling)) * self.dt / 2.0

        X_next = X + dX + self.dt * stimulus
        if self.state_variable_boundaries is not None:
            self.bound_state(X_next)
        if self.clamped_state_variable_values is not None:
            self.clamp_state(X_next)
        return X_next


class HeunStochastic(IntegratorStochastic):
    """
    It is a simple example of a predictor-corrector method. It is also known as
    modified trapezoidal method, which uses the Euler method as its predictor.

    """

    _ui_name = "Stochastic Heun"

    def scheme(self, X, dfun, coupling, local_coupling, stimulus):
        """
        From [2]_:

        .. math::
            X_i(t) = X_i(t-1) + dX(X_i(t)/2 + dX(X_i(t-1))) dt + g_i(X) Z_1

        in our case, :math:`noise = Z_1`

        See page 1180.

        """
        noise = self.noise.generate(X.shape)
        noise_gfun = self.noise.gfun(X)
        if (noise_gfun.shape != (1,) and noise.shape[0] != noise_gfun.shape[0]):
            msg = str("Got shape %s for noise but require %s."
                      " You need to reconfigure noise after you have changed your model."%(
                       noise_gfun.shape, (noise.shape[0], noise.shape[1])))
            raise Exception(msg)

        m_dx_tn = dfun(X, coupling, local_coupling)

        noise *= noise_gfun

        inter = X + self.dt * m_dx_tn + noise + self.dt * stimulus
        if self.state_variable_boundaries is not None:
            self.bound_state(inter)
        if self.clamped_state_variable_values is not None:
            self.clamp_state(inter)

        dX = (m_dx_tn + dfun(inter, coupling, local_coupling)) * self.dt / 2.0

        X_next = X + dX + noise + self.dt * stimulus
        if self.state_variable_boundaries is not None:
            self.bound_state(X_next)
        if self.clamped_state_variable_values is not None:
            self.clamp_state(X_next)

        return X_next


class EulerDeterministic(Integrator):
    """
    It is the simplest difference methods for the initial value problem. The
    recursive structure of Euler scheme, which evaluates approximate values to
    the Ito process at the discretization instants only, is the key to its
    successful implementation.

    """

    _ui_name = "Euler"

    def scheme(self, X, dfun, coupling, local_coupling, stimulus):
        r"""

        .. math::
            X_{n+1} = X_n + dt \, dX(t_n, X_n)

        cf. Equations 1.3 and 1.13, pages 305 and 306 respectively.

        """

        self.dX = dfun(X, coupling, local_coupling) 

        X_next = X + self.dt * (self.dX + stimulus)
        if self.state_variable_boundaries is not None:
            self.bound_state(X_next)
        if self.clamped_state_variable_values is not None:
            self.clamp_state(X_next)
        return X_next


class EulerStochastic(IntegratorStochastic):
    """
    It is the simplest difference methods for the initial value problem. The
    recursive structure of Euler scheme, which evaluates approximate values to
    the Ito process at the discretization instants only, is the key to its
    successful implementation.

    """

    _ui_name = "Euler-Maruyama"

    def scheme(self, X, dfun, coupling, local_coupling, stimulus):
        r"""
        Ones of the simplest time discrete approximations of an Ito process is
        Euler-Maruyama approximation, that satisfies the scalar stochastic
        differential equation (From [1]_, page 305):

        .. math::
            X_{n+1} = X_n + dX(t_n, X_n) \, dt + g(X_n) Z_1

        in our case, :math:`noise = Z_1`

        cf. Equations 1.3 and 1.13, pages 305 and 306 respectively.

        """

        noise = self.noise.generate(X.shape)
        dX = dfun(X, coupling, local_coupling) * self.dt 
        noise_gfun = self.noise.gfun(X)
        X_next = X + dX + noise_gfun * noise + self.dt * stimulus
        if self.state_variable_boundaries is not None:
            self.bound_state(X_next)
        if self.clamped_state_variable_values is not None:
            self.clamp_state(X_next)
        return X_next


class RungeKutta4thOrderDeterministic(Integrator):
    """
    The Runge-Kutta method is a standard procedure with most one-step methods.

    """

    _ui_name = "Runge-Kutta 4th order"

    def scheme(self, X, dfun, coupling, local_coupling=0.0, stimulus=0.0):
        r"""
        The classical 4th order Runge-Kutta method is an explicit method.
        The 4th order Runge-Kutta methods are the most commonly used,
        representing a good compromise between accuracy and computational
        effort.

        From [1]_, pages 289-290, cf. Equation 2.8

        .. math::
            y_{n+1} &= y_n + 1/6 h (k_1 + 2 k_2 + 2 k_3 + k_4) \\
            t_{n+1} &= t_n + h \\
            k_1 &= f(t_n, y_n) \\
            k_2 &= f(t_n + h/2, y_n + h k_1 / 2) \\
            k_3 &= f(t_n + h/2, y_n + h k_2 / 2) \\
            k_4 &= f(t_n + h, y_n + h k_3)


        """

        dt = self.dt
        dt2 = dt / 2.0
        dt6 = dt / 6.0

        k1 = dfun(X, coupling, local_coupling)
        inter_k1 = X + dt2 * k1
        if self.state_variable_boundaries is not None:
            self.bound_state(inter_k1)
        if self.clamped_state_variable_values is not None:
            self.clamp_state(inter_k1)
        k2 = dfun(inter_k1, coupling, local_coupling)
        inter_k2 = X + dt2 * k2
        if self.state_variable_boundaries is not None:
            self.bound_state(inter_k2)
        if self.clamped_state_variable_values is not None:
            self.clamp_state(inter_k2)
        k3 = dfun(inter_k2, coupling, local_coupling)
        inter_k3 = X + dt * k3
        if self.state_variable_boundaries is not None:
            self.bound_state(inter_k3)
        if self.clamped_state_variable_values is not None:
            self.clamp_state(inter_k3)
        k4 = dfun(inter_k3, coupling, local_coupling)

        dX = dt6 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        X_next = X + dX + self.dt * stimulus
        if self.state_variable_boundaries is not None:
            self.bound_state(X_next)
        if self.clamped_state_variable_values is not None:
            self.clamp_state(X_next)
        return X_next


class Identity(Integrator):
    """
    The Identity integrator does not apply any scheme to the
    provided dfun, only returning its results.

    This allows the model to determine its stepping scheme
    directly, and may be used for difference equations and
    cellular automata.

    """

    _ui_name = "Difference equation"

    def scheme(self, X, dfun, coupling=None, local_coupling=0.0, stimulus=0.0):
        """
        The identity scheme simply returns the results of the dfun and
        stimulus.

        .. math::
            x_{n+1} = f(x_{n})

        """

        return dfun(X, coupling, local_coupling) + stimulus


class SciPyODEBase(object):
    "Provides a base class for integrators using SciPy's ode class."

    def _dfun_wrapper(self, dfun, state_shape):
        @functools.wraps(dfun)
        def wrapper(t, X_, coupling=None, local_coupling=0.0):
            X = X_.reshape(state_shape)
            dXdt = dfun(X, coupling, local_coupling)
            return dXdt.ravel()
        return wrapper

    def _prepare_ode(self, X, dfun):
        ode = scipy.integrate.ode(self._dfun_wrapper(dfun, X.shape))
        ode.set_initial_value(X.ravel())
        ode.set_integrator(self._scipy_ode_integrator_name,
                           first_step=self.dt / 5.0)
        LOG.debug('prepped backing SciPy ODE integrator %r', ode)
        return ode

    _ode = None

    def _apply_ode(self, X, dfun, coupling, local_coupling, stimulus):
        if self._ode is None:
            self._ode = self._prepare_ode(X, dfun)
        self._ode.y[:] = X.ravel()
        self._ode.set_f_params(coupling, local_coupling)
        return self._ode.integrate(self._ode.t + self.dt).reshape(X.shape) + self.dt * stimulus


# TODO: Find a solution for boundary application for intermediate steps of SciPy O/SDE solvers
# Right now they would behave differently than the TVB ones.


class SciPyODE(SciPyODEBase):

    def scheme(self, X, dfun, coupling, local_coupling, stimulus):
        X_next = self._apply_ode(X, dfun, coupling, local_coupling, stimulus)
        if self.state_variable_boundaries is not None:
            self.bound_state(X_next)
        if self.clamped_state_variable_values is not None:
            self.clamp_state(X_next)
        return X_next

class SciPySDE(SciPyODEBase):

    def scheme(self, X, dfun, coupling, local_coupling, stimulus):
        X_next = self._apply_ode(X, dfun, coupling, local_coupling, stimulus)
        X_next += self.noise.gfun(X) * self.noise.generate(X.shape)
        if self.state_variable_boundaries is not None:
            self.bound_state(X_next)
        if self.clamped_state_variable_values is not None:
            self.clamp_state(X_next)
        return X_next


class VODE(SciPyODE, Integrator):
    _scipy_ode_integrator_name = "vode"
    _ui_name = "Variable-order Adams / BDF"


class VODEStochastic(SciPySDE, IntegratorStochastic):
    _scipy_ode_integrator_name = "vode"
    _ui_name = "Stochastic variable-order Adams / BDF"


class Dopri5(SciPyODE, Integrator):
    _scipy_ode_integrator_name = "dopri5"
    _ui_name = "Dormand-Prince, order (4, 5)"


class Dopri5Stochastic(SciPySDE, IntegratorStochastic):
    _scipy_ode_integrator_name = "dopri5"
    _ui_name = "Stochastic Dormand-Prince, order (4, 5)"

class Dop853(SciPyODE, Integrator):
    _scipy_ode_integrator_name = "dop853"
    _ui_name = "Dormand-Prince, order 8 (5, 3)"

class Dop853Stochastic(SciPySDE, IntegratorStochastic):
    _scipy_ode_integrator_name = "dop853"
    _ui_name = "Stochastic Dormand-Prince, order 8 (5, 3)"
