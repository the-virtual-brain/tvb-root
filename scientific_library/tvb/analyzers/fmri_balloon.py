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
#   The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
Implementation of differet BOLD signal models. Four different models are distinguished: 

+ CBM_N: Classical BOLD Model Non-linear
+ CBM_L: Classical BOLD Model Linear
+ RBM_N: Revised   BOLD Model Non-linear (default)
+ RBM_L: Revised   BOLD Model Linear

``Classical`` means that the coefficients used to compute the BOLD signal are
derived as described in [Obata2004]_ . ``Revised`` coefficients are defined in
[Stephan2007]_

References:

.. [Stephan2007] Stephan KE, Weiskopf N, Drysdale PM, Robinson PA,
                 Friston KJ (2007) Comparing hemodynamic models with 
                 DCM. NeuroImage 38: 387-401.

.. [Obata2004]  Obata, T.; Liu, T. T.; Miller, K. L.; Luh, W. M.; Wong, E. C.; Frank, L. R. &
                Buxton, R. B. (2004) **Discrepancies between BOLD and flow dynamics in primary and
                supplementary motor areas: application of the balloon model to the
                interpretation of BOLD transients.** Neuroimage, 21:144-153

.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import numpy
import tvb.datatypes.time_series as time_series
from tvb.basic.logger.builder import get_logger


log = get_logger(__name__)


class BalloonModel:
    """

    A class for calculating the simulated BOLD signal given a TimeSeries
    object of TVB and returning another TimeSeries object.

    The haemodynamic model parameters based on constants for a 1.5 T scanner.
        
    """

    # NOTE: a potential problem when the input is a TimeSeriesSurface.
    # TODO: add an spatial averaging for TimeSeriesSurface.

    def __init__(self, time_series, dt, integrator, bold_model, RBM, neural_input_transformation,
                                        tau_s, tau_f, tau_o, alpha, TE, V0, E0, epsilon, nu_0, r_0):
        """
        Parameters
        __________

        time_series : TimeSeries
        The timeseries for which the CrossCoherence and ComplexCoherence is to be computed.

        dt : float
        The integration time step size for the balloon model (s).

        integrator : Integrator
        A tvb.simulator.Integrator object which is an integration scheme with supporting attributes such as
        integration step size and noise specification for stochastic methods.
        It is used to compute the time courses of the balloon model state variables.

        bold_model : str
        Select the set of equations for the BOLD model.

        RBM : bool
        Select classical vs revised BOLD model (CBM or RBM).

        neural_input_transformation : str
        This represents the operation to perform on the state-variable(s) of the model used to generate
        the input TimeSeries. ``none`` takes the first state-variable as neural input; `` abs_diff`` is the absolute
        value of the derivative (first order difference) of the first state variable;
        ``sum``: sum all the state-variables of the input TimeSeries.

        tau_s : float
        Balloon model parameter. Time of signal decay (s).

        tau_f : float
        Balloon model parameter. Time of flow-dependent elimination or feedback regulation (s).
        The average  time blood take to traverse the venous compartment. It is the  ratio of resting blood
        volume (V0) to resting blood flow (F0).

        tau_o : float
        Balloon model parameter. Haemodynamic transit time (s). The average time blood take to traverse
        the venous compartment. It is the  ratio of resting blood volume (V0) to resting blood flow (F0).

        alpha : float
        Balloon model parameter. Stiffness parameter. Grubb's exponent.

        TE : float
        BOLD parameter. Echo Time

        V0 : float
        BOLD parameter. Resting blood volume fraction.

        E0 : float
        BOLD parameter. Resting oxygen extraction fraction.

        epsilon : ndarray
        BOLD parameter. Ratio of intra- and extravascular signals. In principle  this
        parameter could be derived from empirical data and spatialized.

        nu_0: float
        BOLD parameter. Frequency offset at the outer surface of magnetized vessels (Hz).

        r_0: float
        BOLD parameter. Slope r0 of intravascular relaxation rate (Hz). Only used for ``revised`` coefficients.
        """

        self.time_series = time_series
        self.dt = dt
        self.integrator = integrator
        self.bold_model = bold_model
        self.RBM = RBM
        self.neural_input_transformation = neural_input_transformation
        self.tau_s = tau_s
        self.tau_f = tau_f
        self.tau_o = tau_o
        self.alpha = alpha
        self.TE = TE
        self.V0 = V0
        self.E0 = E0
        self.epsilon = epsilon
        self.nu_0 = nu_0
        self.r_0 = r_0

    def calculate_simulated_bold_signal(self):
        """
         # type: ( ) -> TimeSeriesRegion
        Calculate simulated BOLD signal
        """

        # NOTE: Just using the first state variable, although in the Bold monitor
        #      input is the sum over the state-variables. Only time-series
        #      from basic monitors should be used as inputs.

        neural_activity, t_int = self.input_transformation(self.time_series, self.neural_input_transformation)
        input_shape = neural_activity.shape

        if self.dt is None:
            self.dt = self.time_series.sample_period / 1000.  # (s) integration time step
            msg = "Integration time step size for the balloon model is %s seconds" % str(self.dt)
            log.debug(msg)

        # NOTE: Avoid upsampling ...
        if self.dt < (self.time_series.sample_period / 1000.):
            msg = "Integration time step shouldn't be smaller than the sampling period of the input signal."
            log.error(msg)

        balloon_nvar = 4

        # NOTE: hard coded initial conditions
        state = numpy.zeros((input_shape[0], balloon_nvar, input_shape[2], input_shape[3]))  # s
        state[0, 1, :] = 1.  # f
        state[0, 2, :] = 1.  # v
        state[0, 3, :] = 1.  # q

        # BOLD model coefficients
        k = self.compute_derived_parameters()
        k1, k2, k3 = k[0], k[1], k[2]

        # prepare integrator
        self.integrator.dt = self.dt
        self.integrator.configure()
        log.debug("Integration time step size will be: %s seconds" % str(self.integrator.dt))

        scheme = self.integrator.scheme

        # NOTE: the following variables are not used in this integration but
        # required due to the way integrators scheme has been defined.

        local_coupling = 0.0
        stimulus = 0.0

        # Do some checks:
        if numpy.isnan(neural_activity).any():
            log.warning("NaNs detected in the neural activity!!")

        # normalise the time-series.
        neural_activity = neural_activity - neural_activity.mean(axis=0)[numpy.newaxis, :]

        # solve equations
        for step in range(1, t_int.shape[0]):
            state[step, :] = scheme(state[step - 1, :], self.balloon_dfun,
                                    neural_activity[step, :], local_coupling, stimulus)
            if numpy.isnan(state[step, :]).any():
                log.warning("NaNs detected...")

        # NOTE: just for the sake of clarity, define the variables used in the BOLD model
        s = state[:, 0, :]
        f = state[:, 1, :]
        v = state[:, 2, :]
        q = state[:, 3, :]

        # import pdb; pdb.set_trace()

        # BOLD models
        if self.bold_model == "nonlinear":
            """
            Non-linear BOLD model equations.
            Page 391. Eq. (13) top in [Stephan2007]_
            """
            y_bold = numpy.array(self.V0 * (k1 * (1. - q) + k2 * (1. - q / v) + k3 * (1. - v)))
            y_b = y_bold[:, numpy.newaxis, :, :]
            log.debug("Max value: %s" % str(y_b.max()))

        else:
            """
            Linear BOLD model equations.
            Page 391. Eq. (13) bottom in [Stephan2007]_ 
            """
            y_bold = numpy.array(self.V0 * ((k1 + k2) * (1. - q) + (k3 - k2) * (1. - v)))
            y_b = y_bold[:, numpy.newaxis, :, :]

        sample_period = 1. / self.dt

        bold_signal = time_series.TimeSeriesRegion(
            data=y_b,
            time=t_int,
            sample_period=sample_period,
            sample_period_unit='s')

        return bold_signal

    def compute_derived_parameters(self):
        """
        Compute derived parameters :math:`k_1`, :math:`k_2` and :math:`k_3`.
        """

        if not self.RBM:
            """
            Classical BOLD Model Coefficients [Obata2004]_
            Page 389 in [Stephan2007]_, Eq. (3)
            """
            k1 = 7. * self.E0
            k2 = 2. * self.E0
            k3 = 1. - self.epsilon
        else:
            """
            Revised BOLD Model Coefficients.
            Generalized BOLD signal model.
            Page 400 in [Stephan2007]_, Eq. (12)
            """
            k1 = 4.3 * self.nu_0 * self.E0 * self.TE
            k2 = self.epsilon * self.r_0 * self.E0 * self.TE
            k3 = 1 - self.epsilon

        return numpy.array([k1, k2, k3])

    def input_transformation(self, time_series, mode):
        """
        Perform an operation on the input time-series.
        """

        log.debug("Computing: %s on the input time series" % str(mode))

        if mode == "none":
            ts = time_series.data[:, 0, :, :]
            ts = ts[:, numpy.newaxis, :, :]
            t_int = time_series.time / 1000.  # (s)

        elif mode == "abs_diff":
            ts = abs(numpy.diff(time_series.data, axis=0))
            t_int = (time_series.time[1:] - time_series.time[0:-1]) / 1000.  # (s)

        elif mode == "sum":
            ts = numpy.sum(time_series.data, axis=1)
            ts = ts[:, numpy.newaxis, :, :]
            t_int = time_series.time / 1000.  # (s)

        else:
            log.error("Bad operation/transformation mode, must be one of:")
            log.error("('abs_diff', 'sum', 'none')")
            raise Exception("Bad transformation mode")

        return ts, t_int

    def balloon_dfun(self, state_variables, neural_input, local_coupling=0.0):
        r"""
        The Balloon model equations. See Eqs. (4-10) in [Stephan2007]_
        .. math::
                \frac{ds}{dt} &= x - \kappa\,s - \gamma \,(f-1) \\
                \frac{df}{dt} &= s \\
                \frac{dv}{dt} &= \frac{1}{\tau_o} \, (f - v^{1/\alpha})\\
                \frac{dq}{dt} &= \frac{1}{\tau_o}(f \, \frac{1-(1-E_0)^{1/\alpha}}{E_0} - v^{&/\alpha} \frac{q}{v})\\
                \kappa &= \frac{1}{\tau_s}\\
                \gamma &= \frac{1}{\tau_f}
        """

        s = state_variables[0, :]
        f = state_variables[1, :]
        v = state_variables[2, :]
        q = state_variables[3, :]

        x = neural_input[0, :]

        ds = x - (1. / self.tau_s) * s - (1. / self.tau_f) * (f - 1)
        df = s
        dv = (1. / self.tau_o) * (f - v ** (1. / self.alpha))
        dq = (1. / self.tau_o) * ((f * (1. - (1. - self.E0) ** (1. / f)) / self.E0) -
                                  (v ** (1. / self.alpha)) * (q / v))

        return numpy.array([ds, df, dv, dq])
