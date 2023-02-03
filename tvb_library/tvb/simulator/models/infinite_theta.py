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
   A family of mean field models of infinite populations of all-to-all coupled
   quadratic integrate and fire neurons (theta-neurons).

.. moduleauthor:: Jan Fousek <jan.fousek@univ-amu.fr>, Giovanni Rabuffo <giovanni.rabuffo@univ-amu.fr>
"""

from tvb.simulator.models.base import Model
from tvb.basic.neotraits.api import NArray, List, Range, Final

import numpy


class MontbrioPazoRoxin(Model):
    r"""
    2D model describing the Ott-Antonsen reduction of infinite all-to-all
    coupled QIF neurons (Theta-neurons) as in [Montbrio_Pazo_Roxin_2015]_.

    The two state variables :math:`r` and :math:`V` represent the average
    firing rate and the average membrane potential of our QIF neurons.

    The equations of the infinite QIF 2D population model read

    .. math::
            \dot{r} &= 1/\tau (\Delta/(\pi \tau) + 2 V r)\\
            \dot{V} &= 1/\tau (V^2 - \tau^2 \pi^2 r^2 + \eta + J \tau r + I)
    
    Input from the network enters in the :math:`V` variable as 
    :math:`1/\tau(c_r C_r + c_v C_V)` where C is the incomming coupling. In 
    other words, depending on the parameters :math:`c_r`, :math:`c_v` we couple
    the neural masses via the firing rate and/or the membrane potential.
    
    .. [Montbrio_Pazo_Roxin_2015] Montbrió, E., Pazó, D., & Roxin, A. (2015). Macroscopic description for networks of spiking neurons. *Physical Review X*, 5(2), 021028.
    """

    # Define traited attributes for this model, these represent possible kwargs.

    tau = NArray(
        label=r":math:`\tau`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.001, hi=15.0, step=0.01),
        doc="""Characteristic time""",
    )

    I = NArray(
        label=":math:`I_{ext}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.01),
        doc="""External Current""",
    )

    Delta = NArray(
        label=r":math:`\Delta`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Mean heterogeneous noise""",
    )

    J = NArray(
        label=":math:`J`",
        default=numpy.array([15.0]),
        domain=Range(lo=-25.0, hi=25.0, step=0.0001),
        doc="""Mean Synaptic weight.""",
    )

    eta = NArray(
        label=r":math:`\eta`",
        default=numpy.array([-5.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the
            firing rate variable to itself""",
    )

    Gamma = NArray(
        label=r":math:`\Gamma`",
        default=numpy.array([0.]),
        domain=Range(lo=0., hi=10.0, step=0.01),
        doc="""Half-width of synaptic weight distribution""",
    )

    cr = NArray(
        label=":math:`cr`",
        default=numpy.array([1.]),
        domain=Range(lo=0., hi=1, step=0.1),
        doc="""It is the weight on Coupling through variable r.""",
    )

    cv = NArray(
        label=":math:`cv`",
        default=numpy.array([0.]),
        domain=Range(lo=0., hi=1, step=0.1),
        doc="""It is the weight on Coupling through variable V.""",
    )

    # Informational attribute, used for phase-plane and initial()
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"r": numpy.array([0., 2.0]),
                 "V": numpy.array([-2.0, 1.5])},
        doc="""Expected ranges of the state variables for initial condition generation and phase plane setup.""",
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            "r": numpy.array([0.0, numpy.inf])
        },
    )

    # TODO should match cvars below..
    coupling_terms = Final(
        label="Coupling terms",
        # how to unpack coupling array
        default=["Coupling_Term_r", "Coupling_Term_V"]
    )

    state_variable_dfuns = Final(
        label="Drift functions",
        default={
            "r": "1/tau * ( Delta / (pi * tau) + 2 * V * r)",
            "V": "1/tau * ( V*V - pi*pi*tau*tau*r*r + eta + J * tau * r + I + cr * Coupling_Term_r + cv * Coupling_Term_V)"
        }
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("r", "V"),
        default=("r", "V"),
        doc="The quantities of interest for monitoring for the Infinite QIF 2D oscillator.",
    )

    parameter_names = List(
        of=str,
        label="List of parameters for this model",
        default='tau Delta eta J I cr cv'.split())

    state_variables = ('r', 'V')
    _nvar = 2
    # Cvar is the coupling variable. 
    cvar = numpy.array([0, 1], dtype=numpy.int32)
    # Stvar is the variable where stimulus is applied.
    stvar = numpy.array([1], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
            2D model describing the Ott-Antonsen reduction of infinite all-to-all
            coupled QIF neurons (Theta-neurons) as in [Montbrio_Pazo_Roxin_2015]_.

            The two state variables :math:`r` and :math:`V` represent the average
            firing rate and the average membrane potential of our QIF neurons.

            The equations of the infinite QIF 2D population model read

            .. math::
                    \dot{r} &= 1/\tau (\Delta/(\pi \tau) + 2 V r)\\
                    \dot{V} &= 1/\tau (V^2 - \tau^2 \pi^2 r^2 + \eta + J \tau r + I)
        """

        r, V = state_variables

        # [State_variables, nodes]
        I = self.I
        Delta = self.Delta
        Gamma = self.Gamma
        eta = self.eta
        tau = self.tau
        J = self.J
        cr = self.cr
        cv = self.cv

        Coupling_Term_r = coupling[0, :]  # This zero refers to the first element of cvar (r in this case)
        Coupling_Term_V = coupling[1, :]  # This zero refers to the second element of cvar (V in this case)

        derivative = numpy.empty_like(state_variables)

        derivative[0] = 1 / tau * (Delta / (numpy.pi * tau) + 2 * V * r)
        derivative[1] = 1 / tau * (
                    V ** 2 - numpy.pi ** 2 * tau ** 2 * r ** 2 + eta + J * tau * r + I + cr * Coupling_Term_r + cv * Coupling_Term_V)

        return derivative


class CoombesByrne(Model):
    r"""
    4D model describing the Ott-Antonsen reduction of infinite all-to-all
    coupled QIF neurons (Theta-neurons) as in [Coombes_Byrne_2019]_.
    
    Note: the original equations describe the dynamics of the Kuramoto parameter 
    :math:`Z`. Using the conformal transformation 
    :math:`Z=(1-W^\star)/(1+W^\star)` and :math:`W= \pi r + i V`, 
    we express the system dynamics in terms of two state variables :math:`r` 
    and :math:`V` representing the average firing rate and the average membrane 
    potential of our QIF neurons. The conductance variable and its derivative 
    are :math:`g` and :math:`q`.

    The equations of the model read
    
    .. math::
            \dot{r} &= \Delta/\pi + 2 V r - g r^2 \\
            \dot{V} &= V^2 - \pi^2 r^2 + \eta + (v_{syn} - V) g \\
            \dot{g} &= \alpha q  \\
            \dot{q} &= \alpha (\kappa \pi r - g - 2 q)
            
    .. [Coombes_Byrne_2019] Coombes, S., & Byrne, Á. (2019). Next generation neural mass models. In *Nonlinear Dynamics in Computational Neuroscience* (pp. 1-16). Springer, Cham.
            
    """

    # Define traited attributes for this model, these represent possible kwargs.

    Delta = NArray(
        label=r":math:`\Delta`",
        default=numpy.array([0.5]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Half-width of heterogeneous noise distribution""",
    )

    alpha = NArray(
        label=r":math:`\alpha`",
        default=numpy.array([0.95]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Parameter of the alpha-function""",
    )

    v_syn = NArray(
        label=":math:`v_{syn}`",
        default=numpy.array([-10.0]),
        domain=Range(lo=-20.0, hi=0.0, step=0.01),
        doc="""QIF membrane reversal potential""",
    )

    k = NArray(
        label=":math:`k`",
        default=numpy.array([1.]),
        domain=Range(lo=0., hi=5.0, step=0.01),
        doc="""Local coupling strength""",
    )

    eta = NArray(
        label=r":math:`\eta`",
        default=numpy.array([20.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the
            firing rate variable to itself""",
    )

    # Informational attribute, used for phase-plane and initial()
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={
            "r": numpy.array([0., 6.]),
            "V": numpy.array([-10., 10.]),
            "g": numpy.array([1., 2.]),
            "q": numpy.array([-0.5, 0.7])
        },
        doc="""Expected ranges of the state variables for initial condition generation and phase plane setup.""",
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            "r": numpy.array([0.0, numpy.inf])
        },
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("r", "V", "g", "q"),
        default=("r", "V"),
        doc="The quantities of interest for monitoring for the Infinite QIF 2D oscillator.",
    )

    state_variables = ('r', 'V', 'g', 'q')
    _nvar = 4
    # Cvar is the coupling variable. 
    cvar = numpy.array([0, 1, 2, 3], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
            4D model describing the Ott-Antonsen reduction of infinite all-to-all
            coupled QIF neurons (Theta-neurons) as in [Coombes_Byrne_2019]_.

            The equations of the model read

            .. math::
                    \dot{r} &= \Delta/\pi + 2 V r - g r^2 \\
                    \dot{V} &= V^2 - \pi^2 r^2 + \eta + (v_{syn} - V) g \\
                    \dot{g} &= \alpha q  \\
                    \dot{q} &= \alpha (\kappa \pi r - g - 2 q)
        """
        r, V, g, q = state_variables

        # [State_variables, nodes]
        Delta = self.Delta
        k = self.k
        v_syn = self.v_syn
        eta = self.eta
        alpha = self.alpha

        Coupling_Term_r = coupling[0, :]  # This zero refers to the first element of cvar (r in this case)

        derivative = numpy.empty_like(state_variables)

        derivative[0] = Delta / numpy.pi + 2 * V * r - g * r
        derivative[1] = V ** 2 - numpy.pi ** 2 * r ** 2 + eta + (v_syn - V) * g + Coupling_Term_r
        derivative[2] = alpha * (q)
        derivative[3] = alpha * (k * numpy.pi * r - g - 2 * q)

        return derivative


class CoombesByrne2D(Model):
    r"""
    2D model describing the Ott-Antonsen reduction of infinite all-to-all coupled 
    QIF neurons (Theta-neurons) as in [Coombes_Byrne_2019]_.

    The two state variables :math:`r` and :math:`V` represent the average firing 
    rate and the average membrane potential of our QIF neurons. The conductance 
    :math:`g` is not dynamical and proportional to :math:`r`.

    The equations of the model read
    
    .. math::
            \dot{r} &= \Delta/\pi + 2 V r - g r^2\\
            \dot{V} &= V^2 - \pi^2 r^2 + \eta + (v_{syn} - V) g \\
            g &= \kappa \pi r
    .. [Coombes_Byrne_2019] Coombes, S., & Byrne, Á. (2019). Next generation neural mass models. In *Nonlinear Dynamics in Computational Neuroscience* (pp. 1-16). Springer, Cham.

    """

    # Define traited attributes for this model, these represent possible kwargs.

    Delta = NArray(
        label=r":math:`\Delta`",
        default=numpy.array([1.]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Half-width of heterogeneous noise distribution""",
    )

    v_syn = NArray(
        label=":math:`v_syn`",
        default=numpy.array([-4.0]),
        domain=Range(lo=-20.0, hi=0.0, step=0.01),
        doc="""QIF membrane reversal potential""",
    )

    k = NArray(
        label=":math:`k`",
        default=numpy.array([1.]),
        domain=Range(lo=0., hi=5.0, step=0.01),
        doc="""Local coupling strength""",
    )

    eta = NArray(
        label=r":math:`\eta`",
        default=numpy.array([2.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the
            firing rate variable to itself""",
    )

    # Informational attribute, used for phase-plane and initial()
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"r": numpy.array([0., 2.0]),
                 "V": numpy.array([-2.0, 1.5])},
        doc="""Expected ranges of the state variables for initial condition generation and phase plane setup.""",
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            "r": numpy.array([0.0, numpy.inf])
        },
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("r", "V"),
        default=("r", "V"),
        doc="The quantities of interest for monitoring for the Infinite QIF 2D oscillator.",
    )

    state_variables = ('r', 'V')
    _nvar = 2
    # Cvar is the coupling variable. 
    cvar = numpy.array([0, 1], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
           2D model describing the Ott-Antonsen reduction of infinite all-to-all coupled
           QIF neurons (Theta-neurons) as in [Coombes_Byrne_2019]_.

           The two state variables :math:`r` and :math:`V` represent the average firing
           rate and the average membrane potential of our QIF neurons. The conductance
           :math:`g` is not dynamical and proportional to :math:`r`.

           The equations of the model read

           .. math::
                   \dot{r} &= \Delta/\pi + 2 V r - g r^2\\
                   \dot{V} &= V^2 - \pi^2 r^2 + \eta + (v_{syn} - V) g \\
                   g &= \kappa \pi r
        """
        r, V = state_variables

        # [State_variables, nodes]
        Delta = self.Delta
        k = self.k
        v_syn = self.v_syn
        eta = self.eta

        Coupling_Term_r = coupling[0, :]  # This zero refers to the first element of cvar (r in this case)

        derivative = numpy.empty_like(state_variables)

        derivative[0] = Delta / numpy.pi + 2 * V * r - k * numpy.pi * r ** 2
        derivative[1] = V ** 2 - numpy.pi ** 2 * r ** 2 + eta + (v_syn - V) * k * numpy.pi * r + Coupling_Term_r

        return derivative


class GastSchmidtKnosche_SD(Model):
    r"""
    4D model describing the Ott-Antonsen reduction of infinite all-to-all 
    coupled QIF neurons (Theta-neurons) with Synaptic Depression adaptation 
    mechanisms [Gastetal_2020]_.

    The two state variables :math:`r` and :math:`V` represent the average firing rate and 
    the average membrane potential of our QIF neurons.
    :math:`A` and :math:`B` are respectively the adaptation variable and its derivative.

    The equations of the infinite QIF 2D population model read
    
    .. math::
            \dot{r} &= 1/\tau (\Delta/(\pi \tau) + 2 V r)\\
            \dot{V} &= 1/\tau (V^2 - \tau^2 \pi^2 r^2 + \eta + J \tau r (1 - A) + I)\\ 
            \dot{A} &= 1/\tau_A (B)\\
            \dot{B} &= 1/\tau_A (-2 B - A + \alpha  r) \\

    .. [Gastetal_2020] Gast, R., Schmidt, H., & Knösche, T. R. (2020). A mean-field description of bursting dynamics in spiking neural networks with short-term adaptation. *Neural Computation*, 32(9), 1615-1634.
    """

    # Define traited attributes for this model, these represent possible kwargs.

    tau = NArray(
        label=r":math:`\tau`",
        default=numpy.array([1.0]),
        domain=Range(lo=0., hi=15.0, step=0.01),
        doc="""Characteristic time""",
    )

    tau_A = NArray(
        label=r":math:`\tau_A`",
        default=numpy.array([10.0]),
        domain=Range(lo=0., hi=15.0, step=0.01),
        doc="""Adaptation time scale""",
    )

    alpha = NArray(
        label=r":math:`\alpha`",
        default=numpy.array([0.5]),
        domain=Range(lo=0.0, hi=1.0, step=0.1),
        doc="""adaptation rate""",
    )

    I = NArray(
        label=":math:`I_{ext}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.01),
        doc="""External homogeneous current""",
    )

    Delta = NArray(
        label=r":math:`\Delta`",
        default=numpy.array([2.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Half-width of heterogeneous noise distribution""",
    )

    J = NArray(
        label=":math:`J`",
        default=numpy.array([21.2132]),
        domain=Range(lo=-25.0, hi=25.0, step=0.01),
        doc="""Synaptic weight""",
    )

    eta = NArray(
        label=r":math:`\eta`",
        default=numpy.array([-6.]),
        domain=Range(lo=-10.0, hi=10.0, step=0.0001),
        doc="""Mean of heterogeneous noise distribution""",
    )

    cr = NArray(
        label=":math:`cr`",
        default=numpy.array([1.]),
        domain=Range(lo=0., hi=1, step=0.1),
        doc="""It is the weight on Coupling through variable r.""",
    )

    cv = NArray(
        label=":math:`cv`",
        default=numpy.array([0.]),
        domain=Range(lo=0., hi=1, step=0.1),
        doc="""It is the weight on Coupling through variable V.""",
    )

    # Informational attribute, used for phase-plane and initial()
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"r": numpy.array([0.0, 4]),
                 "V": numpy.array([-3.0, 0.3]),
                 "A": numpy.array([0.0, 0.4]),
                 "B": numpy.array([-0.2, 0.3])},
        doc="""Expected ranges of the state variables for initial condition generation and phase plane setup.""",
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            "r": numpy.array([0.0, numpy.inf])
        },
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("r", "V", "A", "B"),
        default=("r", "V"),
        doc="The quantities of interest for monitoring for the Infinite QIF 2D oscillator.",
    )

    state_variables = ('r', 'V', 'A', 'B')
    _nvar = 4
    # Cvar is the coupling variable. 
    cvar = numpy.array([0, 1, 2, 3], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
            4D model describing the Ott-Antonsen reduction of infinite all-to-all
            coupled QIF neurons (Theta-neurons) with Synaptic Depression adaptation
            mechanisms [Gastetal_2020]_.

            The two state variables :math:`r` and :math:`V` represent the average firing rate and
            the average membrane potential of our QIF neurons.
            :math:`A` and :math:`B` are respectively the adaptation variable and its derivative.

            The equations of the infinite QIF 2D population model read

            .. math::
                    \dot{r} &= 1/\tau (\Delta/(\pi \tau) + 2 V r)\\
                    \dot{V} &= 1/\tau (V^2 - \tau^2 \pi^2 r^2 + \eta + J \tau r (1 - A) + I)\\
                    \dot{A} &= 1/\tau_A (B)\\
                    \dot{B} &= 1/\tau_A (-2 B - A + \alpha  r) \\
        """
        r, V, A, B = state_variables

        # [State_variables, nodes]
        I = self.I
        Delta = self.Delta
        eta = self.eta
        J = self.J
        alpha = self.alpha
        cr = self.cr
        cv = self.cv
        alpha = self.alpha
        tau_A = self.tau_A
        tau = self.tau

        Coupling_Term_r = coupling[0, :]  # This zero refers to the first element of cvar (r in this case)
        Coupling_Term_V = coupling[1, :]  # This one refers to the second element of cvar (V in this case)

        derivative = numpy.empty_like(state_variables)

        derivative[0] = 1 / tau * (Delta / (numpy.pi * tau) + 2 * V * r)
        derivative[1] = 1 / tau * (V ** 2 - numpy.pi ** 2 * tau ** 2 * r ** 2 + eta + J * tau * r * (
                    1 - A) + I + cr * Coupling_Term_r + cv * Coupling_Term_V)
        derivative[2] = 1 / tau_A * (B)
        derivative[3] = 1 / tau_A * (- 2 * B - A + alpha * r)

        return derivative


class GastSchmidtKnosche_SF(Model):
    r"""
    4D model describing the Ott-Antonsen reduction of infinite all-to-all coupled QIF neurons (Theta-neurons) with Spike-Frequency adaptation mechanisms [Gastetal_2020]_.

    The two state variables :math:`r` and :math:`V` represent the average firing rate and 
    the average membrane potential of our QIF neurons.
    :math:`A` and :math:`B` are respectively the adaptation variable and its derivative.

    The equations of the infinite QIF 2D population model read
    
    .. math::
            \dot{r} &= 1/\tau (\Delta/(\pi \tau) + 2 V r)\\
            \dot{V} &= 1/\tau (V^2 - \tau^2 \pi^2 r^2 + \eta + J \tau r - A + I)\\ 
            \dot{A} &= 1/\tau_A (B)\\
            \dot{B} &= 1/\tau_A (-2 B - A + \alpha r) \\
    .. [Gastetal_2020]	Gast, R., Schmidt, H., & Knösche, T. R. (2020). A mean-field description of bursting dynamics in spiking neural networks with short-term adaptation. Neural Computation, 32(9), 1615-1634.
    """

    # Define traited attributes for this model, these represent possible kwargs.

    tau = NArray(
        label=r":math:`\tau`",
        default=numpy.array([1.0]),
        domain=Range(lo=0., hi=15.0, step=0.01),
        doc="""Characteristic time""",
    )

    tau_A = NArray(
        label=r":math:`\tau_A`",
        default=numpy.array([10.0]),
        domain=Range(lo=0., hi=15.0, step=0.01),
        doc="""Adaptation time scale""",
    )

    alpha = NArray(
        label=r":math:`\alpha`",
        default=numpy.array([10.]),
        domain=Range(lo=0.0, hi=1.0, step=0.1),
        doc="""adaptation rate""",
    )

    I = NArray(
        label=":math:`I_{ext}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.01),
        doc="""External homogeneous current""",
    )

    Delta = NArray(
        label=r":math:`\Delta`",
        default=numpy.array([2.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Half-width of heterogeneous noise distribution""",
    )

    J = NArray(
        label=":math:`J`",
        default=numpy.array([21.2132]),
        domain=Range(lo=-25.0, hi=25.0, step=0.01),
        doc="""Synaptic weight""",
    )

    eta = NArray(
        label=r":math:`\eta`",
        default=numpy.array([1.]),
        domain=Range(lo=-10.0, hi=10.0, step=0.0001),
        doc="""Mean of heterogeneous noise distribution""",
    )

    cr = NArray(
        label=":math:`cr`",
        default=numpy.array([1.]),
        domain=Range(lo=0., hi=1, step=0.1),
        doc="""It is the weight on Coupling through variable r.""",
    )

    cv = NArray(
        label=":math:`cv`",
        default=numpy.array([0.]),
        domain=Range(lo=0., hi=1, step=0.1),
        doc="""It is the weight on Coupling through variable V.""",
    )

    # Informational attribute, used for phase-plane and initial()
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"r": numpy.array([0., 2.0]),
                 "V": numpy.array([-2.0, 1.5]),
                 "A": numpy.array([-1., 1.0]),
                 "B": numpy.array([-1.0, 1.0])},
        doc="""Expected ranges of the state variables for initial condition generation and phase plane setup.""",
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            "r": numpy.array([0.0, numpy.inf])
        },
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("r", "V", "A", "B"),
        default=("r", "V"),
        doc="The quantities of interest for monitoring for the Infinite QIF 2D oscillator.",
    )

    state_variables = ('r', 'V', 'A', 'B')
    _nvar = 4
    # Cvar is the coupling variable. 
    cvar = numpy.array([0, 1, 2, 3], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
            4D model describing the Ott-Antonsen reduction of infinite all-to-all coupled QIF neurons (Theta-neurons) with Spike-Frequency adaptation mechanisms [Gastetal_2020]_.

            The two state variables :math:`r` and :math:`V` represent the average firing rate and
            the average membrane potential of our QIF neurons.
            :math:`A` and :math:`B` are respectively the adaptation variable and its derivative.

            The equations of the infinite QIF 2D population model read

            .. math::
                    \dot{r} &= 1/\tau (\Delta/(\pi \tau) + 2 V r)\\
                    \dot{V} &= 1/\tau (V^2 - \tau^2 \pi^2 r^2 + \eta + J \tau r - A + I)\\
                    \dot{A} &= 1/\tau_A (B)\\
                    \dot{B} &= 1/\tau_A (-2 B - A + \alpha r) \\
        """
        r, V, A, B = state_variables

        # [State_variables, nodes]
        I = self.I
        Delta = self.Delta
        eta = self.eta
        J = self.J
        alpha = self.alpha
        cr = self.cr
        cv = self.cv
        alpha = self.alpha
        tau_A = self.tau_A
        tau = self.tau

        Coupling_Term_r = coupling[0, :]  # This zero refers to the first element of cvar (r in this case)
        Coupling_Term_V = coupling[1, :]  # This one refers to the second element of cvar (V in this case)

        derivative = numpy.empty_like(state_variables)

        derivative[0] = 1 / tau * (Delta / (numpy.pi * tau) + 2 * V * r)
        derivative[1] = 1 / tau * (
                    V ** 2 - numpy.pi ** 2 * tau ** 2 * r ** 2 + eta + J * tau * r + I - A + cr * Coupling_Term_r + cv * Coupling_Term_V)
        derivative[2] = 1 / tau_A * (B)
        derivative[3] = 1 / tau_A * (- 2 * B - A + alpha * r)

        return derivative


class DumontGutkin(Model):
    r"""
    8D model describing the Ott-Antonsen reduction of infinite all-to-all 
    coupled QIF Excitatory E and Inhibitory I Theta-neurons with local synaptic 
    dynamics [DumontGutkin2019]_.

    State variables :math:`r` and :math:`V` represent the average firing rate and 
    the average membrane potential of our QIF neurons. 
    The neural masses are coupled through the firing rate of :math:`E_i` population from node i-th into :math:`E_j` and :math:`I_j` subpopulations in node j-th.

    The equations of the excitatory infinite QIF 4D population model read (similar for inhibitory):
    
    .. math::
            \dot{r} &= 1/\tau (\Delta/(\pi \tau) + 2 V r)\\
            \dot{V} &= 1/\tau (V^2 + \eta + \gamma I - \tau^2 \pi^2 r^2 + \tau g - \tau s)\\
            \dot{g} &= 1/\tau_s (-g + J_ r)\\
            \dot{s} &= 1/\tau_s (-s) \\

    .. [DumontGutkin2019] Dumont, G., & Gutkin, B. (2019). Macroscopic phase resetting-curves determine oscillatory coherence and signal transfer in inter-coupled neural circuits. PLoS computational biology, 15(5), e1007019.
    """

    # Define traited attributes for this model, these represent possible kwargs.
    I_e = NArray(
        label=":math:`I_{ext_e}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.01),
        doc="""External homogeneous current on excitatory population""",
    )

    Delta_e = NArray(
        label=r":math:`\Delta_e`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Half-width of heterogeneous noise distribution over excitatory population""",
    )

    eta_e = NArray(
        label=r":math:`\eta_e`",
        default=numpy.array([-5.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.0001),
        doc="""Mean heterogeneous current on excitatory population""",
    )

    tau_e = NArray(
        label=r":math:`\tau_e`",
        default=numpy.array([10.0]),
        domain=Range(lo=0., hi=15.0, step=0.01),
        doc="""Characteristic time of excitatory population""",
    )

    I_i = NArray(
        label=":math:`I_{ext_i}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.01),
        doc="""External current on inhibitory population""",
    )

    Delta_i = NArray(
        label=r":math:`\Delta_i`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Half-width of heterogeneous noise distribution over inhibitory population""",
    )

    eta_i = NArray(
        label=r":math:`\eta_i`",
        default=numpy.array([-5.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.0001),
        doc="""Mean heterogeneous current on inhibitory population""",
    )
    tau_i = NArray(
        label=r":math:`\tau_i`",
        default=numpy.array([10.0]),
        domain=Range(lo=0., hi=15.0, step=0.01),
        doc="""Characteristic time of inhibitory population""",
    )

    tau_s = NArray(
        label=r":math:`\tau_s`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=15.0, step=0.01),
        doc="""Synaptic time constant""",
    )

    J_ee = NArray(
        label=":math:`J_{ee}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-25.0, hi=25.0, step=0.0001),
        doc="""Synaptic weight e-->e""",
    )

    J_ei = NArray(
        label=":math:`J_{ei}`",
        default=numpy.array([10.0]),
        domain=Range(lo=-25.0, hi=25.0, step=0.0001),
        doc="""Synaptic weight i-->e""",
    )

    J_ie = NArray(
        label=":math:`J_{ie}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-25.0, hi=25.0, step=0.0001),
        doc="""Synaptic weight e-->i""",
    )

    J_ii = NArray(
        label=":math:`J_{ii}`",
        default=numpy.array([15.0]),
        domain=Range(lo=-25.0, hi=25.0, step=0.0001),
        doc="""Synaptic weight i-->i""",
    )

    Gamma = NArray(
        label=r":math:`\Gamma`",
        default=numpy.array([5.0]),
        domain=Range(lo=0., hi=10., step=0.1),
        doc="""Ratio of excitatory VS inhibitory global couplings G_ie/G_ee .""",
    )

    # Informational attribute, used for phase-plane and initial()
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"r_e": numpy.array([0., 2.0]),
                 "V_e": numpy.array([-2.0, 1.5]),
                 "s_ee": numpy.array([-1.0, 1.0]),
                 "s_ei": numpy.array([-1.0, 1.0]),
                 "r_i": numpy.array([0., 2.0]),
                 "V_i": numpy.array([-2.0, 1.5]),
                 "s_ie": numpy.array([-1.0, 1.0]),
                 "s_ii": numpy.array([-1.0, 1.0])},
        doc="""Expected ranges of the state variables for initial condition generation and phase plane setup.""",
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            "r_e": numpy.array([0.0, numpy.inf]),
            "r_i": numpy.array([0.0, numpy.inf])
        },
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("r_e", "V_e", "s_ee", "s_ei", "r_i", "V_i", "s_ie", "s_ii"),
        default=("r_e", "V_e", "s_ee", "s_ei", "r_i", "V_i", "s_ie", "s_ii"),
        doc="The quantities of interest for monitoring for the Infinite QIF 2D oscillator.",
    )

    state_variables = ('r_e', 'V_e', 's_ee', 's_ei', 'r_i', 'V_i', 's_ie', 's_ii')
    _nvar = 8
    # Cvar is the coupling variable. 
    cvar = numpy.array([0, 1, 4, 5], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
            8D model describing the Ott-Antonsen reduction of infinite all-to-all
            coupled QIF Excitatory E and Inhibitory I Theta-neurons with local synaptic
            dynamics [DumontGutkin2019]_.

            State variables :math:`r` and :math:`V` represent the average firing rate and
            the average membrane potential of our QIF neurons.
            The neural masses are coupled through the firing rate of :math:`E_i` population from node i-th into :math:`E_j` and :math:`I_j` subpopulations in node j-th.

            The equations of the excitatory infinite QIF 4D population model read (similar for inhibitory):

            .. math::
                    \dot{r} &= 1/\tau (\Delta/(\pi \tau) + 2 V r)\\
                    \dot{V} &= 1/\tau (V^2 + \eta + \gamma I - \tau^2 \pi^2 r^2 + \tau g - \tau s)\\
                    \dot{g} &= 1/\tau_s (-g + J_ r)\\
                    \dot{s} &= 1/\tau_s (-s) \\
        """
        r_e, V_e, s_ee, s_ei, r_i, V_i, s_ie, s_ii = state_variables

        # [State_variables, nodes]
        Delta_e = self.Delta_e
        Delta_i = self.Delta_i
        tau_e = self.tau_e
        tau_i = self.tau_i
        tau_s = self.tau_s
        eta_e = self.eta_e
        eta_i = self.eta_i
        J_ee = self.J_ee
        J_ii = self.J_ii
        J_ei = self.J_ei
        J_ie = self.J_ie
        I_e = self.I_e
        I_i = self.I_i
        Gamma = self.Gamma

        Coupling_Term = coupling[0, :]  # This zero refers to the first element of cvar (r_e in this case)

        derivative = numpy.empty_like(state_variables)

        derivative[0] = 1 / tau_e * (Delta_e / (numpy.pi * tau_e) + 2 * V_e * r_e)
        derivative[1] = 1 / tau_e * (
                    V_e ** 2 + eta_e - tau_e ** 2 * numpy.pi ** 2 * r_e ** 2 + tau_e * s_ee - tau_e * s_ei + I_e)
        derivative[2] = 1 / tau_s * (- s_ee + J_ee * r_e + Coupling_Term)
        derivative[3] = 1 / tau_s * (- s_ei + J_ei * r_i)
        derivative[4] = 1 / tau_i * (Delta_i / (numpy.pi * tau_i) + 2 * V_i * r_i)
        derivative[5] = 1 / tau_i * (
                    V_i ** 2 + eta_i - tau_i ** 2 * numpy.pi ** 2 * r_i ** 2 + tau_i * s_ie - tau_i * s_ii + I_i)
        derivative[6] = 1 / tau_s * (- s_ie + J_ie * r_e + Gamma * Coupling_Term)
        derivative[7] = 1 / tau_s * (- s_ii + J_ii * r_i)

        return derivative
