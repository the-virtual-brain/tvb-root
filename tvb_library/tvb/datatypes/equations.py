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

The Equation datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""
import numpy
from scipy.special import gamma as sp_gamma
from tvb.basic.neotraits.api import HasTraits, Attr, Final, TupleEnum, SubformEnum
from tvb.simulator.backend.ref import RefBase


# In how many points should the equation be evaluated for the plot. Increasing this will
# give smoother results at the cost of some performance
DEFAULT_PLOT_GRANULARITY = 1024


class EquationsEnum(SubformEnum):
    """
    Superclass of all enums that have equations as values
    """


# class Equation(basic.MapAsJson, core.Type):
# todo: handle the MapAsJson functionality

class Equation(HasTraits):
    """Base class for Equation data types."""

    equation = Attr(
        field_type=str,
        label="Equation as a string",
        doc=""" the equation as it should be interpreted by numexpr""")

    # todo: transform these parameters into plain declarative attrs
    parameters = Attr(
        field_type=dict,
        label="Parameters in a dictionary.",
        default=lambda: {},
        doc="""Should be a list of the parameters and their meaning, Traits
                should be able to take defaults and sensible ranges from any
                traited information that was provided.""")

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance
        of this datatype.
        """
        return {
            "Equation type": self.__class__.__name__,
            "equation": self.equation,
            "parameters": self.parameters
        }

    def evaluate(self, var):
        """
        Generate a discrete representation of the equation for the space
        represented by ``var``.

        The argument ``var`` can represent a distance, or effective distance,
        for each node in a simulation. Or a time, or in principle any arbitrary
        `` space ``. ``var`` can be a single number, a numpy.ndarray or a
        ?scipy.sparse_matrix? TODO: think this last one is true, need to check
        as we need it for LocalConnectivity...
        """
        ns = {'var': var}
        ns.update(self.parameters)
        return RefBase.evaluate(self.equation, ns)

    def get_series_data(self, min_range=0, max_range=100, step=None):
        """
        NOTE: The symbol from the equation which varies should be named: var
        Returns the series data needed for plotting this equation.
        """
        if step is None:
            step = float(max_range - min_range) / DEFAULT_PLOT_GRANULARITY

        var = numpy.arange(min_range, max_range+step, step)
        var = var[numpy.newaxis, :]

        y = self.evaluate(var)
        result = list(zip(var.flat, y.flat))
        return result, False


class TemporalApplicableEquation(Equation):
    """
    Abstract class introduced just for filtering what equations to be displayed in UI,
    for setting the temporal component in Stimulus on region and surface.
    """


class FiniteSupportEquation(TemporalApplicableEquation):
    """
    Equations that decay to zero as the variable moves away from zero. It is
    necessary to restrict spatial equation evaluated on a surface to this
    class, are . The main purpose of this class is to facilitate filtering in the UI,
    for patters on surface (stimuli surface and localConnectivity).
    """


class SpatialApplicableEquation(Equation):
    """
    Abstract class introduced just for filtering what equations to be displayed in UI,
    for setting model parameters on the Surface level.
    """


class DiscreteEquation(FiniteSupportEquation):
    """
    A special case for 'discrete' spaces, such as the regions, where each point
    in the space is effectively just assigned a value.

    """
    equation = Attr(
        field_type=str,
        label="Discrete Equation",
        default="var",
        # locked=True,
        doc="""The equation defines a function of :math:`x`""")


class Linear(TemporalApplicableEquation):
    """
    A linear equation.

    """
    equation = Final(
        label="Linear Equation",
        default="a * var + b",
        # locked=True,
        doc=""":math:`result = a * x + b`""")

    parameters = Attr(
        field_type=dict,
        label="Linear Parameters",
        default=lambda: {"a": 1.0, "b": 0.0})


class RescaleInterval(Equation):
    """
    Direct rescaling of an interval from [oldMin..oldMax] into [newMin..newMax]
    """
    equation = Final(
        label="Rescale Equation",
        default="(var - oldMin) * (newMax - newMin) / (oldMax - oldMin)  + newMin",
        doc=""":math:`result = (x - oldMin) * (newMax - newMin) / (oldMax - oldMin)  + newMin`""")

    parameters = Attr(
        field_type=dict,
        label="Rescale Interval Parameters",
        default=lambda: {"newMin": 0.0, "newMax": 1.0, "oldMin": 0.0, "oldMax": 1.0})


class Absolute(Equation):
    """
    Absolute value
    """
    equation = Final(
        label="Absolute Equation",
        default="abs(var)",
        doc=""":math:`result = abs(x)`""")


class Identity(Equation):
    """
    Identity value
    """
    equation = Final(
        label="Identity Equation",
        default="var",
        doc=""":math:`result = x`""")


class Logarithm(Equation):
    """
    Logarithm Equation
    """
    equation = Final(
        label="Logarithm Equation",
        default="log(var)",
        doc=""":math:`result = log(x)`""")


class Gaussian(SpatialApplicableEquation, FiniteSupportEquation):
    """
    A Gaussian equation.
    offset: parameter to extend the behaviour of this function
    when spatializing model parameters.

    """

    equation = Final(
        label="Gaussian Equation",
        default="(amp * exp(-((var-midpoint)**2 / (2.0 * sigma**2))))+offset",
        doc=""":math:`(amp \\exp\\left(-\\left(\\left(x-midpoint\\right)^2 /
        \\left(2.0 \\sigma^2\\right)\\right)\\right)) + offset`""")

    parameters = Attr(
        field_type=dict,
        label="Gaussian Parameters",
        default=lambda: {"amp": 1.0, "sigma": 1.0, "midpoint": 0.0, "offset": 0.0})


class DoubleGaussian(FiniteSupportEquation):
    """
    A Mexican-hat function approximated by the difference of Gaussians functions.

    """

    equation = Final(
        label="Double Gaussian Equation",
        default="""(amp_1 * exp(-((var-midpoint_1)**2 / (2.0 * sigma_1**2)))) - """
                """(amp_2 * exp(-((var-midpoint_2)**2 / (2.0 * sigma_2**2))))""",
        doc=""":math:`amp_1 \\exp\\left(-\\left((x-midpoint_1)^2 / \\left(2.0
        \\sigma_1^2\\right)\\right)\\right) -
        amp_2 \\exp\\left(-\\left((x-midpoint_2)^2 / \\left(2.0
        \\sigma_2^2\\right)\\right)\\right)`""")

    parameters = Attr(
        field_type=dict,
        label="Double Gaussian Parameters",
        default=lambda: {"amp_1": 0.5, "sigma_1": 20.0, "midpoint_1": 0.0,
                         "amp_2": 1.0, "sigma_2": 10.0, "midpoint_2": 0.0})


class Sigmoid(SpatialApplicableEquation, FiniteSupportEquation):
    """
    A Sigmoid equation.
    offset: parameter to extend the behaviour of this function
    when spatializing model parameters.
    """

    equation = Final(
        label="Sigmoid Equation",
        default="(amp / (1.0 + exp(-1.8137993642342178 * (radius-var)/sigma))) + offset",
        doc=""":math:`(amp / (1.0 + \\exp(-\\pi/\\sqrt(3.0)
            (radius-x)/\\sigma))) + offset`""")

    parameters = Attr(
        field_type=dict,
        label="Sigmoid Parameters",
        default=lambda: {"amp": 1.0, "radius": 5.0, "sigma": 1.0, "offset": 0.0})


class GeneralizedSigmoid(TemporalApplicableEquation):
    """
    A General Sigmoid equation.
    """

    equation = Final(
        label="Generalized Sigmoid Equation",
        default="low + (high - low) / (1.0 + exp(-1.8137993642342178 * (var-midpoint)/sigma))",
        doc=""":math:`low + (high - low) / (1.0 + \\exp(-\\pi/\\sqrt(3.0)
            (x-midpoint)/\\sigma))`""")

    parameters = Attr(
        field_type=dict,
        label="Sigmoid Parameters",
        default=lambda: {"low": 0.0, "high": 1.0, "midpoint": 1.0, "sigma": 0.3})


class Sinusoid(TemporalApplicableEquation):
    """
    A Sinusoid equation.
    """

    equation = Final(
        label="Sinusoid Equation",
        default="amp * sin(6.283185307179586 * frequency * var)",
        doc=""":math:`amp \\sin(2.0 \\pi frequency x)` """)

    parameters = Attr(
        field_type=dict,
        label="Sinusoid Parameters",
        default=lambda: {"amp": 1.0, "frequency": 0.01})


class Cosine(TemporalApplicableEquation):
    """
    A Cosine equation.
    """

    equation = Final(
        label="Cosine Equation",
        default="amp * cos(6.283185307179586 * frequency * var)",
        doc=""":math:`amp \\cos(2.0 \\pi frequency x)` """)

    parameters = Attr(
        field_type=dict,
        label="Cosine Parameters",
        default=lambda: {"amp": 1.0, "frequency": 0.01})


class Alpha(TemporalApplicableEquation):
    """
    An Alpha function belonging to the Exponential function family.
    """

    equation = Final(
        label="Alpha Equation",
        default="""where((var-onset) > 0, (alpha * beta) / (beta - alpha) * (exp(-alpha * (var-onset)) """
                """ - exp(-beta * (var-onset))), 0.0 * var)""",
        doc=""":math:`(\\alpha * \\beta) / (\\beta - \\alpha) *
            (\\exp(-\\alpha * (x-onset)) - \\exp(-\\beta * (x-onset)))` for :math:`(x-onset) > 0`""")

    parameters = Attr(
        field_type=dict,
        label="Alpha Parameters",
        default=lambda: {"onset": 0.5, "alpha": 13.0, "beta": 42.0})


class PulseTrain(TemporalApplicableEquation):
    """
    A pulse train , offset with respect to the time axis.

    **Parameters**:

    * :math:`\\tau` :  pulse width or pulse duration
    * :math:`T`     :  pulse repetition period
    * onset         :  time of the first pulse
    * amp           :  amplitude of the pulse
    """

    equation = Final(
        label="Pulse Train",
        default="where((var>onset)&(((var-onset) % T) < tau), amp, 0)",
        doc=""":math:`\\left\\{{\\begin{array}{rl}amp,&{\\text{if }} ((var-onset) \\mod T) < \\tau \\space and \\space
         var > onset\\\\0, &{\\text{otherwise }}\\end{array}}\\right.`""")

    # onset is in milliseconds
    # T and tau are in milliseconds as well

    parameters = Attr(
        field_type=dict,
        default=lambda: {"T": 42.0, "tau": 13.0, "amp": 1.0, "onset": 30.0},
        label="Pulse Train Parameters")


class HRFKernelEquation(Equation):
    """Base class for hemodynamic response functions."""


class Gamma(HRFKernelEquation):
    """
    A Gamma function for the bold monitor. It belongs to the family of Exponential functions.

    **Parameters**:


    * :math:`\\tau`      : Exponential time constant of the gamma function [seconds].
    * :math:`n`          : The phase delay of the gamma function.
    * :math: `factorial` : (n-1)!. numexpr does not support factorial yet.
    * :math: `a`         : Amplitude factor after normalization.


    **Reference**:

    .. [B_1996] Geoffrey M. Boynton, Stephen A. Engel, Gary H. Glover and David
        J. Heeger (1996). Linear Systems Analysis of Functional Magnetic Resonance
        Imaging in Human V1. J Neurosci 16: 4207-4221

    .. note:: might be filtered from the equations used in Stimulus and Local Connectivity.

    """

    # TODO: Introduce a time delay in the equation (shifts the hrf onset)
    # """:math:`h(t) = \frac{(\frac{t-\delta}{\tau})^{(n-1)} e^{-(\frac{t-\delta}{\tau})}}{\tau(n-1)!}"""
    # delta = 2.05 seconds -- Additional delay in seconds from the onset of the
    # time-series to the beginning of the gamma hrf.
    # delay cannot be negative or greater than the hrf duration.

    equation = Final(
        label="Gamma Equation",
        default="((var / tau) ** (n - 1) * exp(-(var / tau)) )/ (tau * factorial)",
        doc=""":math:`h(var) = \\frac{(\\frac{var}{\\tau})^{(n-1)}\\exp{-(\\frac{var}{\\tau})}}{\\tau(n-1)!}`.""")

    parameters = Attr(
        field_type=dict,
        label="Gamma Parameters",
        default=lambda: {"tau": 1.08, "n": 3.0, "factorial": 2.0, "a": 0.1})

    def evaluate(self, var):
        """
        Generate a discrete representation of the equation for the space
        represented by ``var``.

        .. note: numexpr doesn't support factorial yet

        """

        # compute the factorial
        n = int(self.parameters["n"])
        product = 1
        for i in range(n - 1):
            product *= i + 1

        self.parameters["factorial"] = product
        _pattern = RefBase.evaluate(self.equation,
                                         global_dict=self.parameters)
        _pattern /= max(_pattern)
        _pattern *= self.parameters["a"]
        return _pattern


class DoubleExponential(HRFKernelEquation):
    """
    A difference of two exponential functions to define a kernel for the bold monitor.

    **Parameters** :

    * :math:`\\tau_1`: Time constant of the second exponential function [s]
    * :math:`\\tau_2`: Time constant of the first exponential function [s].
    * :math:`f_1`  : Frequency of the first sine function [Hz].
    * :math:`f_2`  : Frequency of the second sine function [Hz].
    * :math:`amp_1`: Amplitude of the first exponential function.
    * :math:`amp_2`: Amplitude of the second exponential function.
    * :math:`a`    : Amplitude factor after normalization.


    **Reference**:

    .. [P_2000] Alex Polonsky, Randolph Blake, Jochen Braun and David J. Heeger
        (2000). Neuronal activity in human primary visual cortex correlates with
        perception during binocular rivalry. Nature Neuroscience 3: 1153-1159

    """

    equation = Final(
        label="Double Exponential Equation",
        default="((amp_1 * exp(-var/tau_1) * sin(2.*pi*f_1*var)) - (amp_2 * exp(-var/ tau_2) * sin(2.*pi*f_2*var)))",
        doc=""":math:`h(var) = amp_1\\exp(\\frac{-var}{\tau_1})
        \\sin(2\\cdot\\pi f_1 \\cdot var) - amp_2\\cdot \\exp(-\\frac{var}
        {\\tau_2})*\\sin(2\\pi f_2 var)`.""")

    parameters = Attr(
        field_type=dict,
        label="Double Exponential Parameters",
        default=lambda: {"tau_1": 7.22, "f_1": 0.03, "amp_1": 0.1,
                         "tau_2": 7.4, "f_2": 0.12, "amp_2": 0.1,
                         "a": 0.1, "pi": numpy.pi})

    def evaluate(self, var):
        """
        Generate a discrete representation of the equation for the space
        represented by ``var``.
        """
        _pattern = RefBase.evaluate(self.equation, global_dict=self.parameters)
        _pattern /= max(_pattern)

        _pattern *= self.parameters["a"]
        return _pattern


class FirstOrderVolterra(HRFKernelEquation):
    """
    Integral form of the first Volterra kernel of the three used in the
    Ballon Windekessel model for computing the Bold signal.
    This function describes a damped Oscillator.

    **Parameters** :

    * :math:`\\tau_s`: Dimensionless? exponential decay parameter.
    * :math:`\\tau_f`: Dimensionless? oscillatory parameter.
    * :math:`k_1`    : First Volterra kernel coefficient.
    * :math:`V_0` : Resting blood volume fraction.


    **References** :

    .. [F_2000] Friston, K., Mechelli, A., Turner, R., and Price, C., *Nonlinear
        Responses in fMRI: The Balloon Model, Volterra Kernels, and Other
        Hemodynamics*, NeuroImage, 12, 466 - 477, 2000.

    """

    equation = Final(
        label="First Order Volterra Kernel",
        default="""1/3. * exp(-0.5*(var / tau_s)) * (sin(sqrt(1./tau_f - 1./(4.*tau_s**2)) * var)) / """
                """(sqrt(1./tau_f - 1./(4.*tau_s**2)))""",
        doc=""":math:`G(t - t^{\\prime}) =
             e^{\\frac{1}{2} \\left(\\frac{t - t^{\\prime}}{\\tau_s} \\right)}
             \\frac{\\sin\\left((t - t^{\\prime})
             \\sqrt{\\frac{1}{\\tau_f} - \\frac{1}{4 \\tau_s^2}}\\right)}
             {\\sqrt{\\frac{1}{\\tau_f} - \\frac{1}{4 \\tau_s^2}}}
             \\; \\; \\; \\; \\; \\;  for \\; \\; \\; t \\geq t^{\\prime}
             = 0 \\; \\; \\; \\; \\; \\;  for \\; \\; \\;  t < t^{\\prime}`.""")

    parameters = Attr(
        field_type=dict,
        label="Mixture of Gammas Parameters",
        default=lambda: {"tau_s": 0.8, "tau_f": 0.4, "k_1": 5.6, "V_0": 0.02})


class MixtureOfGammas(HRFKernelEquation):
    """
    A mixture of two gamma distributions to create a kernel similar to the one used in SPM.

    >> import scipy.stats as sp_stats
    >> import numpy
    >> t = numpy.linspace(1,20,100)
    >> a1, a2 = 6., 10.
    >> lambda = 1.
    >> c      = 0.5
    >> hrf    = sp_stats.gamma.pdf(t, a1, lambda) - c * sp_stats.gamma.pdf(t, a2, lambda)

    gamma.pdf(x, a, theta) = (lambda*x)**(a-1) * exp(-lambda*x) / gamma(a)
    a                 : shape parameter
    theta: 1 / lambda : scale parameter


    **References**:

    .. [G_1999] Glover, G. *Deconvolution of Impulse Response in Event-Related BOLD fMRI*.
                NeuroImage 9, 416-429, 1999.


    **Parameters**:


    * :math:`a_{1}`       : shape parameter first gamma pdf.
    * :math:`a_{2}`       : shape parameter second gamma pdf.
    * :math:`\\lambda`    : scale parameter first gamma pdf.


    Default values are based on [G_1999]_:
    * :math:`a_{1} - 1 = n_{1} =  5.0`
    * :math:`a_{2} - 1 = n_{2} = 12.0`
    * :math:`c \\equiv a_{2}   = 0.4`

    Alternative values :math:`a_{2}=10` and :math:`c=0.5`

    NOTE: gamma_a_1 and gamma_a_2 are placeholders, the true values are
    computed before evaluating the expression, because numexpr does not
    support certain functions.

    NOTE: [G_1999]_ used a different analytical function that can be approximated
    by this difference of gamma pdfs

    """

    equation = Final(
        label="Mixture of Gammas",
        default="(l * var)**(a_1-1) * exp(-l*var) / gamma_a_1 - c * (l*var)**(a_2-1) * exp(-l*var) / gamma_a_2",
        doc=""":math:`\\frac{\\lambda \\,t^{a_{1} - 1} \\,\\, \\exp^{-\\lambda \\,t}}{\\Gamma(a_{1})}
        - 0.5 \\frac{\\lambda \\,t^{a_{2} - 1} \\,\\, \\exp^{-\\lambda \\,t}}{\\Gamma(a_{2})}`.""")

    parameters = Attr(
        field_type=dict,
        label="Double Exponential Parameters",
        default=lambda: {"a_1": 6.0, "a_2": 13.0, "l": 1.0, "c": 0.4, "gamma_a_1": 1.0, "gamma_a_2": 1.0})

    def evaluate(self, var):
        """
        Generate a discrete representation of the equation for the space
        represented by ``var``.

        .. note: numexpr doesn't support gamma function
        """
        # get gamma functions
        self.parameters["gamma_a_1"] = sp_gamma(self.parameters["a_1"])
        self.parameters["gamma_a_2"] = sp_gamma(self.parameters["a_2"])

        return RefBase.evaluate(self.equation, global_dict=self.parameters)
