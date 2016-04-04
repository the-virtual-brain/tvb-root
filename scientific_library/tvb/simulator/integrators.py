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

# From standard python libraries
import functools

# Third party python libraries  
import numpy
import scipy.integrate

#The Virtual Brain
import tvb.basic.traits.core as core
import tvb.basic.traits.types_basic as basic
from tvb.datatypes import arrays

# vb simulator
import tvb.simulator.noise

from tvb.simulator.common import get_logger
LOG = get_logger(__name__)



class Integrator(core.Type):
    """
    The Integrator class is a base class for the integration methods...

    .. [1] Kloeden and Platen, Springer 1995, *Numerical solution of stochastic
        differential equations.*

    .. [2] Riccardo Mannella, *Integration of Stochastic Differential Equations
        on a Computer*, Int J. of Modern Physics C 13(9): 1177--1194, 2002.

    .. [3] R. Mannella and V. Palleschi, *Fast and precise algorithm for 
        computer simulation of stochastic differential equations*, Phys. Rev. A
        40: 3381, 1989.

    .. #Currently there seems to be a clash betwen traits and autodoc, autodoc
    .. #can't find the methods of the class, the class specific names below get
    .. #us around this...
    .. automethod:: Integrator.__init__
    .. automethod:: Integrator.scheme

    """
    _base_classes = ['Integrator', 'IntegratorStochastic', 'RungeKutta4thOrderDeterministic']

    dt = basic.Float(
        label = "Integration-step size (ms)", 
        default =  0.01220703125, #0.015625,
        #range = basic.Range(lo= 0.0048828125, hi=0.244140625, step= 0.1, base=2.)
        required = True,
        doc = """The step size used by the integration routine in ms. This
        should be chosen to be small enough for the integration to be
        numerically stable. It is also necessary to consider the desired sample
        period of the Monitors, as they are restricted to being integral
        multiples of this value. The default value is set such that all built-in
        models are numerically stable with there default parameters and because
        it is consitent with Monitors using sample periods corresponding to
        powers of 2 from 128 to 4096Hz.""")

    clamped_state_variable_indices = arrays.IntegerArray(
        label = "indices of the state variables to be clamped by the integrators to the values in the clamped_values array",
        default = None,
        order=-1)

    clamped_state_variable_values = arrays.FloatArray(
        label = "The values of the state variables which are clamped ",
        default = None,
        order=-1)

    def __init__(self, **kwargs):
        """Integrators are intialized using their integration step, dt."""
        super(Integrator, self).__init__(**kwargs) 
        LOG.debug(str(kwargs))


    def __repr__(self):
        """A formal, executable, representation of a Model object."""
        class_name = self.__class__.__name__ 
        traited_kwargs = self.trait.keys()
        formal = class_name + "(" + "=%s, ".join(traited_kwargs) + "=%s)"
        return formal % eval("(self." + ", self.".join(traited_kwargs) + ")")


    def __str__(self):
        """An informal, human readable, representation of a Model object."""
        class_name = self.__class__.__name__ 
        traited_kwargs = self.trait.keys()
        informal = class_name + "(" + ", ".join(traited_kwargs) + ")"
        return informal


    def scheme(self, X, dfun, coupling, local_coupling, stimulus):
        """
        The scheme of integrator should take a state and provide the next
        state in time, e.g. for a differential equation, scheme should take
        :math:`X` and provide an appropriate :math:`X + dX` (dfun in the code).

        """
        msg = "Integrator is a base class; please use a suitable subclass."
        raise NotImplementedError(msg)

    def clamp_state(self, X):
        if self.clamped_state_variable_values is not None:
            X[self.clamped_state_variable_indices] = self.clamped_state_variable_values


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

    .. automethod:: IntegratorStochastic.__init__

    """

    noise = tvb.simulator.noise.Noise(
        label = "Integration Noise",
        default = tvb.simulator.noise.Additive,
        required = True,
        doc = """The stochastic integrator's noise source. It incorporates its
        own instance of Numpy's RandomState.""")


    def __init__(self, **kwargs): # dt, 
        """
        Wisdom... and plagiarism. include :math:`g(X)` (gfun in the code), the 
        noise function.

        """
        LOG.info("%s: initing..." % str(self))
        super(IntegratorStochastic, self).__init__(**kwargs)


    def configure(self):
        """  """
        super(IntegratorStochastic, self).configure()
        self.noise.configure()



class HeunDeterministic(Integrator):
    """
    It is a simple example of a predictor-corrector method. It is also known as
    modified trapezoidal method, which uses the Euler method as its predictor.
    And it is also a implicit integration scheme.

    .. #Currently there seems to be a clash betwen traits and autodoc, autodoc
    .. #can't find the methods of the class, the class specific names below get
    .. #us around this...
    .. automethod:: HeunDeterministic.__init__
    .. automethod:: HeunDeterministic.scheme

    """

    _ui_name = "Heun"

    def __init__(self, **kwargs):
        """
        Wisdom... and plagiarism.

        """

        LOG.info("%s: initing..." % str(self))

        super(HeunDeterministic, self).__init__(**kwargs)

        LOG.debug("%s: inited." % repr(self))


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
        inter = X + self.dt * (m_dx_tn  + stimulus)
        self.clamp_state(inter)

        dX = (m_dx_tn + dfun(inter, coupling, local_coupling)) * self.dt / 2.0

        X_next = X + dX + self.dt * stimulus
        self.clamp_state(X_next)
        return X_next

class HeunStochastic(IntegratorStochastic):
    """
    It is a simple example of a predictor-corrector method. It is also known as
    modified trapezoidal method, which uses the Euler method as its predictor.
    And it is also a implicit integration scheme.

    .. #Currently there seems to be a clash betwen traits and autodoc, autodoc
    .. #can't find the methods of the class, the class specific names below get
    .. #us around this...
    .. automethod:: HeunStochastic.__init__
    .. automethod:: HeunStochastic.scheme

    """

    _ui_name = "Stochastic Heun"

    def __init__(self, **kwargs):
        """
        Wisdom... and plagiarism.

        """

        LOG.info("%s: initing..." % str(self))

        super(HeunStochastic, self).__init__(**kwargs)

        LOG.debug("%s: inited." % repr(self))


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
        self.clamp_state(inter)

        dX = (m_dx_tn + dfun(inter, coupling, local_coupling)) * self.dt / 2.0

        X_next = X + dX + noise + self.dt * stimulus
        self.clamp_state(X_next)
        return X_next

class EulerDeterministic(Integrator):
    """
    It is the simplest difference methods for the initial value problem. The
    recursive structure of Euler scheme, which evaluates approximate values to
    the Ito process at the discretization instants only, is the key to its
    successful implementation.

    .. #Currently there seems to be a clash betwen traits and autodoc, autodoc
    .. #can't find the methods of the class, the class specific names below get
    .. #us around this...
    .. automethod:: EulerDeterministic.__init__
    .. automethod:: EulerDeterministic.scheme

    """

    _ui_name = "Euler"

    def __init__(self, **kwargs):
        """
        Wisdom... and plagiarism.

        """

        LOG.info("%s: initing..." % str(self))

        super(EulerDeterministic, self).__init__(**kwargs)

        LOG.debug("%s: inited." % repr(self))


    def scheme(self, X, dfun, coupling, local_coupling, stimulus):
        r"""

        .. math::
            X_{n+1} = X_n + dt \, dX(t_n, X_n)

        cf. Equations 1.3 and 1.13, pages 305 and 306 respectively.

        """

        self.dX = dfun(X, coupling, local_coupling) 

        X_next = X + self.dt * (self.dX + stimulus)
        self.clamp_state(X_next)
        return X_next


class EulerStochastic(IntegratorStochastic):
    """
    It is the simplest difference methods for the initial value problem. The
    recursive structure of Euler scheme, which evaluates approximate values to
    the Ito process at the discretization instants only, is the key to its
    successful implementation.

    .. #Currently there seems to be a clash betwen traits and autodoc, autodoc
    .. #can't find the methods of the class, the class specific names below get
    .. #us around this...
    .. automethod:: EulerStochastic.__init__
    .. automethod:: EulerStochastic.scheme

    """

    _ui_name = "Euler-Maruyama"

    def __init__(self, **kwargs):
        """
        Wisdom... and plagiarism.

        """

        LOG.info("%s: initing..." % str(self))

        super(EulerStochastic, self).__init__(**kwargs)

        LOG.debug("%s: inited." % repr(self))


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
        if (noise_gfun.shape != (1,) and noise.shape[0] != noise_gfun.shape[0]):
            msg = str("Got shape %s for noise but require %s."
                      " You need to reconfigure noise after you have changed your model."%(
                       noise_gfun.shape, (noise.shape[0], noise.shape[1])))
            raise Exception(msg)

        X_next = X + dX + noise_gfun * noise + self.dt * stimulus
        self.clamp_state(X_next)
        return X_next


class RungeKutta4thOrderDeterministic(Integrator):
    """
    The Runge-Kutta method is a standard procedure with most one-step methods.

    .. #Currently there seems to be a clash betwen traits and autodoc, autodoc
    .. #can't find the methods of the class, the class specific names below get
    .. #us around this...
    .. automethod:: RungeKutta4thOrderDeterministic.__init__
    .. automethod:: RungeKutta4thOrderDeterministic.scheme

    """

    _ui_name = "Runge-Kutta 4th order"

    def __init__(self, **kwargs):
        """
        Wisdom... and plagiarism.

        """

        LOG.info("%s: initing..." % str(self))

        super(RungeKutta4thOrderDeterministic, self).__init__(**kwargs)

        self.you_have_been_warned = False

        LOG.debug("%s: inited." % repr(self))


    def scheme(self, X, dfun, coupling=None, local_coupling=0.0, stimulus=0.0):
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

        if (not self.you_have_been_warned) and (coupling != None):
            msg = "%s: Does NOT support coupling. Just to check local dynamics"
            LOG.warning(msg%str(self))
            self.you_have_been_warned = True

        coupling = numpy.zeros(X.shape)

        dt = self.dt
        dt2 = dt / 2.0
        dt6 = dt / 6.0
        #import pdb; pdb.set_trace()
        #todo clamp these
        k1 = dfun(X, coupling, local_coupling)
        inter_k1 = X + dt2 * k1
        self.clamp_state(inter_k1)
        k2 = dfun(inter_k1, coupling, local_coupling)
        inter_k2 = X + dt2 * k2
        self.clamp_state(inter_k2)
        k3 = dfun(inter_k2, coupling, local_coupling)
        inter_k3 = X + dt * k3
        self.clamp_state(inter_k3)
        k4 = dfun(inter_k3, coupling, local_coupling)

        dX = dt6 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        X_next = X + dX + self.dt * stimulus
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
        return ode

    _ode = None

    def _apply_ode(self, X, dfun, coupling, local_coupling, stimulus):
        if self._ode is None:
            self._ode = self._prepare_ode(X, dfun)
        self._ode.y[:] = X.ravel()
        self._ode.set_f_params(coupling, local_coupling)
        return self._ode.integrate(self._ode.t + self.dt).reshape(X.shape) + self.dt * stimulus

class SciPyODE(SciPyODEBase):

    def scheme(self, X, dfun, coupling, local_coupling, stimulus):
        X_next = self._apply_ode(X, dfun, coupling, local_coupling, stimulus)
        self.clamp_state(X_next)
        return X_next

class SciPySDE(SciPyODEBase):

    def scheme(self, X, dfun, coupling, local_coupling, stimulus):
        X_next = self._apply_ode(X, dfun, coupling, local_coupling, stimulus)
        X_next += self.noise.gfun(X) * self.noise.generate(X.shape)
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
