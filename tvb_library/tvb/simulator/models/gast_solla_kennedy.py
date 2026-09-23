# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
   Adaptive-Izhikevich mean-field model of a population of all-to-all coupled
   Izhikevich neurons with spike-frequency adaptation, reduced via the
   Ott-Antonsen / Lorentzian ansatz.

.. moduleauthor:: TVB contributors
"""

from tvb.simulator.models.base import Model
from tvb.basic.neotraits.api import NArray, List, Range, Final

import numpy


class GastSollaKennedy(Model):
    r"""
    4D mean-field model of an infinite population of all-to-all coupled
    adaptive (Izhikevich) neurons with Lorentzian heterogeneity, as derived by
    [Gast_Solla_Kennedy_2024]_.

    The state variables :math:`r` (mean firing rate, in kHz), :math:`v`, and
    :math:`u` are the mean-field counterparts of the Izhikevich variables
    (membrane potential, recovery/adaptation current); :math:`s` is a synaptic
    gating variable driven by the population rate.

    The equations of the model read

    .. math::
            \dot{r} &= \frac{1}{C}\left(\frac{\Delta k^2\,\mathrm{sign}(v-v_r)(v-v_r)}{\pi C}
                       + r\left(k(2v - v_r - v_\theta) - g s\right)\right)\\
            \dot{v} &= \frac{1}{C}\left(k v (v - v_r - v_\theta) + k v_r v_\theta
                       - \pi C r\left(\Delta\,\mathrm{sign}(v-v_r) + \frac{\pi C r}{k}\right)
                       - u + I + g s (E_r - v)
                       + c_r C_r + c_v C_V\right)\\
            \dot{u} &= \frac{b (v - v_r)}{\tau_u} - \frac{u}{\tau_u} + \kappa r\\
            \dot{s} &= -\frac{s}{\tau_s} + J r

    Default parameter values correspond to the regular-spiking (RS) column of
    Table 1 in [Gast_Solla_Kennedy_2024]_.  With these defaults the bistable
    (fold) region cusp apex sits near :math:`\Delta \sim 3.4\text{-}3.9` mV at
    :math:`I \sim 30` pA.

    .. [Gast_Solla_Kennedy_2024] Gast, R., Solla, S. A., & Kennedy, A. (2024).
        Neural heterogeneity controls computations in spiking neural networks.
        *PNAS*, 121(22), e2311885121.
    """

    # Define traited attributes for this model, these represent possible kwargs.

    C = NArray(
        label=r":math:`C`",
        default=numpy.array([100.0]),
        domain=Range(lo=0.001, hi=1000.0, step=0.01),
        doc="""Membrane capacitance.""",
    )

    k = NArray(
        label=r":math:`k`",
        default=numpy.array([0.7]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Parameter scaling the quadratic membrane-potential dynamics.""",
    )

    v_r = NArray(
        label=r":math:`v_r`",
        default=numpy.array([-60.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.01),
        doc="""Resting membrane potential.""",
    )

    v_theta = NArray(
        label=r":math:`v_\theta`",
        default=numpy.array([-40.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.01),
        doc="""Threshold membrane potential.""",
    )

    b = NArray(
        label=r":math:`b`",
        default=numpy.array([-2.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.01),
        doc="""Sensitivity of the recovery/adaptation variable to subthreshold
            fluctuations of the membrane potential.""",
    )

    tau_u = NArray(
        label=r":math:`\tau_u`",
        default=numpy.array([33.33]),
        domain=Range(lo=0.001, hi=1000.0, step=0.01),
        doc="""Time constant of the recovery/adaptation variable.""",
    )

    tau_s = NArray(
        label=r":math:`\tau_s`",
        default=numpy.array([6.0]),
        domain=Range(lo=0.001, hi=1000.0, step=0.01),
        doc="""Synaptic time constant.""",
    )

    J = NArray(
        label=":math:`J`",
        default=numpy.array([15.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.0001),
        doc="""Synaptic weight drive on the gating variable.""",
    )

    kappa = NArray(
        label=r":math:`\kappa`",
        default=numpy.array([10.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.0001),
        doc="""Firing-rate drive on the recovery/adaptation variable.""",
    )

    Delta = NArray(
        label=r":math:`\Delta`",
        default=numpy.array([0.5]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Half-width-at-half-maximum of the Lorentzian distribution of spike thresholds (within-population heterogeneity).""",
    )

    g = NArray(
        label=":math:`g`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Synaptic conductance scale.""",
    )

    E_r = NArray(
        label=":math:`E_r`",
        default=numpy.array([0.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.01),
        doc="""Synaptic reversal potential.""",
    )

    I = NArray(
        label=":math:`I_{ext}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.01),
        doc="""External homogeneous current.""",
    )

    cr = NArray(
        label=":math:`c_r`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.1),
        doc="""Weight on the coupling through the firing-rate variable r.""",
    )

    cv = NArray(
        label=":math:`c_v`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.1),
        doc="""Weight on the coupling through the membrane-potential variable v.""",
    )

    coupling_terms = Final(
        label="Coupling terms",
        default=["Coupling_Term_r", "Coupling_Term_V"],
    )

    parameter_names = List(
        of=str,
        label="List of parameters for this model",
        default="C k v_r v_theta b tau_u tau_s J kappa Delta g E_r I cr cv".split(),
    )

    state_variable_dfuns = Final(
        label="Drift functions for numba codegen",
        default={
            "r": "(Delta * k**2 * np.sign(v - v_r) * (v - v_r) / (np.pi * C) + r * (k * (2 * v - v_r - v_theta) - g * s)) / C",
            "v": "(k * v * (v - v_r - v_theta) + k * v_r * v_theta - np.pi * C * r * (Delta * np.sign(v - v_r) + np.pi * C * r / k) - u + I + g * s * (E_r - v) + cr * Coupling_Term_r + cv * Coupling_Term_V) / C",
            "u": "b * (v - v_r) / tau_u - u / tau_u + kappa * r",
            "s": "-s / tau_s + J * r",
        },
    )

    # Informational attribute, used for phase-plane and initial()
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={
            "r": numpy.array([0.0, 0.5]),
            "v": numpy.array([-70.0, -40.0]),
            "u": numpy.array([0.0, 20.0]),
            "s": numpy.array([0.0, 5.0]),
        },
        doc="""Expected ranges of the state variables for initial condition generation and phase plane setup.""",
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"r": numpy.array([0.0, numpy.inf])},
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("r", "v", "u", "s"),
        default=("r",),
        doc="The quantities of interest for monitoring the adaptive-Izhikevich mean-field population.",
    )

    state_variables = ("r", "v", "u", "s")
    _nvar = 4
    # Cvar is the coupling variable.
    cvar = numpy.array([0, 1, 2, 3], dtype=numpy.int32)
    # Stvar is the variable where stimulus is applied.
    stvar = numpy.array([1], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
            4D mean-field model of an infinite population of all-to-all coupled
            adaptive (Izhikevich) neurons with Lorentzian heterogeneity, as
            derived by [Gast_Solla_Kennedy_2024]_.

            The equations of the model read

            .. math::
                    \dot{r} &= \frac{1}{C}\left(\frac{\Delta k^2\,\mathrm{sign}(v-v_r)(v-v_r)}{\pi C}
                               + r\left(k(2v - v_r - v_\theta) - g s\right)\right)\\
                    \dot{v} &= \frac{1}{C}\left(k v (v - v_r - v_\theta) + k v_r v_\theta
                               - \pi C r\left(\Delta\,\mathrm{sign}(v-v_r) + \frac{\pi C r}{k}\right)
                               - u + I + g s (E_r - v)
                               + c_r C_r + c_v C_V\right)\\
                    \dot{u} &= \frac{b (v - v_r)}{\tau_u} - \frac{u}{\tau_u} + \kappa r\\
                    \dot{s} &= -\frac{s}{\tau_s} + J r
        """
        r, v, u, s = state_variables

        # [State_variables, nodes]
        C = self.C
        k = self.k
        v_r = self.v_r
        v_theta = self.v_theta
        b = self.b
        tau_u = self.tau_u
        tau_s = self.tau_s
        J = self.J
        kappa = self.kappa
        Delta = self.Delta
        g = self.g
        E_r = self.E_r
        I = self.I
        cr = self.cr
        cv = self.cv

        Coupling_Term_r = coupling[
            0, :
        ]  # This zero refers to the first element of cvar (r in this case)
        Coupling_Term_V = coupling[
            1, :
        ]  # This one refers to the second element of cvar (v in this case)

        derivative = numpy.empty_like(state_variables)

        derivative[0] = (
            Delta * k**2 * numpy.sign(v - v_r) * (v - v_r) / (numpy.pi * C)
            + r * (k * (2 * v - v_r - v_theta) - g * s)
        ) / C
        derivative[1] = (
            k * v * (v - v_r - v_theta)
            + k * v_r * v_theta
            - numpy.pi * C * r * (Delta * numpy.sign(v - v_r) + numpy.pi * C * r / k)
            - u
            + I
            + g * s * (E_r - v)
            + cr * Coupling_Term_r
            + cv * Coupling_Term_V
        ) / C
        derivative[2] = b * (v - v_r) / tau_u - u / tau_u + kappa * r
        derivative[3] = -s / tau_s + J * r

        return derivative
