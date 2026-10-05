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
   Two-population adaptive-Izhikevich mean-field model (RS excitatory + FS
   inhibitory) of all-to-all coupled Izhikevich neurons with spike-frequency
   adaptation, reduced via the Ott-Antonsen / Lorentzian ansatz.

.. moduleauthor:: TVB contributors
"""

from tvb.simulator.models.base import Model
from tvb.basic.neotraits.api import NArray, List, Range, Final

import numpy


class GastSollaKennedy(Model):
    r"""
    8D mean-field model of a coupled **regular-spiking excitatory (RS)** and
    **fast-spiking inhibitory (FS)** population of infinite, all-to-all coupled
    adaptive (Izhikevich) neurons with Lorentzian heterogeneity, as derived in
    [Gast_Solla_Kennedy_2024]_.

    The state variables :math:`(r, v, u, s)` describe the excitatory (RS)
    population - mean firing rate :math:`r` (in kHz), mean membrane potential
    :math:`v`, recovery/adaptation current :math:`u` and synaptic gating
    :math:`s` - and :math:`(r_i, v_i, u_i, s_i)` the inhibitory (FS)
    population. The two populations are coupled *within* each node through
    four synaptic channels with strengths :math:`J_{rr}, J_{rf}, J_{fr},
    J_{ff}` and synaptic reversal potentials :math:`E_{\mathrm{AMPA}}` (onto
    the membrane-potential equations) and :math:`E_{\mathrm{GABA}}`.

    The equations read

    .. math::
            \dot{r}   &= \frac{1}{C}\left(r\left[k(2v - v_r - v_\theta) - J_{rr}s - J_{rf}s_i\right]
                          + \frac{\Delta k^2 |v-v_r|}{\pi C}\right)\\
            \dot{v}   &= \frac{1}{C}\left(k(v-v_r)(v-v_\theta)
                          - \pi C r\left[\Delta\,\mathrm{sign}(v-v_r) + \frac{\pi C r}{k}\right]
                          - u + I + J_{rr}s(E_{\mathrm{AMPA}}-v) + J_{rf}s_i(E_{\mathrm{GABA}}-v)
                          + c_r C_r + c_v C_V\right)\\
            \dot{u}   &= \frac{b(v-v_r) - u}{\tau_u} + \kappa r\\
            \dot{s}   &= -\frac{s}{\tau_s} + r\\
            \dot{r}_i &= \frac{1}{C_i}\left(r_i\left[k_i(2v_i - v_{r,i} - v_{\theta,i})
                          - J_{fr}s - J_{ff}s_i\right]
                          + \frac{\Delta_i k_i^2 |v_i-v_{r,i}|}{\pi C_i}\right)\\
            \dot{v}_i &= \frac{1}{C_i}\left(k_i(v_i-v_{r,i})(v_i-v_{\theta,i})
                          - \pi C_i r_i\left[\Delta_i\,\mathrm{sign}(v_i-v_{r,i}) + \frac{\pi C_i r_i}{k_i}\right]
                          - u_i + I_i + J_{fr}s(E_{\mathrm{AMPA}}-v_i) + J_{ff}s_i(E_{\mathrm{GABA}}-v_i)\right)\\
            \dot{u}_i &= \frac{b_i(v_i-v_{r,i}) - u_i}{\tau_{u,i}} + \kappa_i r_i\\
            \dot{s}_i &= -\frac{s_i}{\tau_{s,i}} + r_i

    Default parameter values follow the RS and FS columns of Tables 1 and 2,
    with the two-population coupling strengths :math:`J` of Table 4 of
    [Gast_Solla_Kennedy_2024]_.

    The width of the FS spike-threshold distribution :math:`\Delta_i` acts as
    the paper's control knob on the computations the coupled network can
    perform: with *heterogeneous* FS interneurons (large :math:`\Delta_i`) the
    E-I network preserves the bifurcation structure of the excitatory
    population, whereas with *homogeneous* FS interneurons (small
    :math:`\Delta_i`) inhibition overwrites it.

    .. [Gast_Solla_Kennedy_2024] Gast, R., Solla, S. A., & Kennedy, A. (2024).
        Neural heterogeneity controls computations in spiking neural networks.
        *PNAS*, 121(22), e2311885121.
    """

    # ---------------------------------------------------------------------- #
    # Excitatory (RS) population - Table 1
    # ---------------------------------------------------------------------- #
    C = NArray(
        label=r":math:`C`",
        default=numpy.array([100.0]),
        domain=Range(lo=0.001, hi=1000.0, step=0.01),
        doc="""Membrane capacitance of the RS population.""",
    )

    k = NArray(
        label=r":math:`k`",
        default=numpy.array([0.7]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Parameter scaling the quadratic membrane-potential dynamics of the RS population.""",
    )

    v_r = NArray(
        label=r":math:`v_r`",
        default=numpy.array([-60.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.01),
        doc="""Resting membrane potential of the RS population.""",
    )

    v_theta = NArray(
        label=r":math:`v_\theta`",
        default=numpy.array([-40.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.01),
        doc="""Centre of the spike-threshold Lorentzian of the RS population.""",
    )

    b = NArray(
        label=r":math:`b`",
        default=numpy.array([-2.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.01),
        doc="""Sensitivity of the RS recovery/adaptation variable to subthreshold
            fluctuations of the membrane potential.""",
    )

    tau_u = NArray(
        label=r":math:`\tau_u`",
        default=numpy.array([33.33]),
        domain=Range(lo=0.001, hi=1000.0, step=0.01),
        doc="""Recovery time constant of the RS population.""",
    )

    tau_s = NArray(
        label=r":math:`\tau_s`",
        default=numpy.array([6.0]),
        domain=Range(lo=0.001, hi=1000.0, step=0.01),
        doc="""Synaptic time constant of the RS population.""",
    )

    kappa = NArray(
        label=r":math:`\kappa`",
        default=numpy.array([10.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.0001),
        doc="""Firing-rate drive on the RS recovery/adaptation variable (spike-frequency adaptation strength).""",
    )

    Delta = NArray(
        label=r":math:`\Delta`",
        default=numpy.array([0.5]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Half-width-at-half-maximum of the Lorentzian distribution of RS spike thresholds.""",
    )

    I = NArray(
        label=r":math:`I_{ext}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.01),
        doc="""External homogeneous current on the RS population.""",
    )

    # ---------------------------------------------------------------------- #
    # Inhibitory (FS) population - Table 2
    # ---------------------------------------------------------------------- #
    C_i = NArray(
        label=r":math:`C_i`",
        default=numpy.array([20.0]),
        domain=Range(lo=0.001, hi=1000.0, step=0.01),
        doc="""Membrane capacitance of the FS (interneuron) population.""",
    )

    k_i = NArray(
        label=r":math:`k_i`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Parameter scaling the quadratic membrane-potential dynamics of the FS population.""",
    )

    v_r_i = NArray(
        label=r":math:`v_{r,i}`",
        default=numpy.array([-55.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.01),
        doc="""Resting membrane potential of the FS population.""",
    )

    v_theta_i = NArray(
        label=r":math:`v_{\theta,i}`",
        default=numpy.array([-40.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.01),
        doc="""Centre of the spike-threshold Lorentzian of the FS population.""",
    )

    b_i = NArray(
        label=r":math:`b_i`",
        default=numpy.array([0.025]),
        domain=Range(lo=-100.0, hi=100.0, step=0.01),
        doc="""Sensitivity of the FS recovery/adaptation variable to subthreshold voltage.""",
    )

    tau_u_i = NArray(
        label=r":math:`\tau_{u,i}`",
        default=numpy.array([5.0]),
        domain=Range(lo=0.001, hi=1000.0, step=0.01),
        doc="""Recovery time constant of the FS population.""",
    )

    tau_s_i = NArray(
        label=r":math:`\tau_{s,i}`",
        default=numpy.array([8.0]),
        domain=Range(lo=0.001, hi=1000.0, step=0.01),
        doc=r"""Synaptic time constant (GABA\ :sub:`A`) of the FS population.""",
    )

    kappa_i = NArray(
        label=r":math:`\kappa_i`",
        default=numpy.array([0.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.0001),
        doc="""Firing-rate drive on the FS recovery/adaptation variable (fast-spiking cells have no adaptation).""",
    )

    Delta_i = NArray(
        label=r":math:`\Delta_i`",
        default=numpy.array([2.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Half-width-at-half-maximum of the Lorentzian distribution of FS spike thresholds
            (the paper's heterogeneity control knob).""",
    )

    I_i = NArray(
        label=r":math:`I_{ext,i}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.01),
        doc="""External homogeneous current on the FS population.""",
    )

    # ---------------------------------------------------------------------- #
    # Two-population coupling - Table 4
    # ---------------------------------------------------------------------- #
    J_rr = NArray(
        label=r":math:`J_{rr}`",
        default=numpy.array([16.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.0001),
        doc="""Synaptic strength RS -> RS (AMPA onto the RS population, from its own rate).""",
    )

    J_rf = NArray(
        label=r":math:`J_{rf}`",
        default=numpy.array([16.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.0001),
        doc=r"""Synaptic strength FS -> RS (GABA\ :sub:`A` onto the RS population).""",
    )

    J_fr = NArray(
        label=r":math:`J_{fr}`",
        default=numpy.array([4.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.0001),
        doc="""Synaptic strength RS -> FS (AMPA onto the FS population, from the RS rate).""",
    )

    J_ff = NArray(
        label=r":math:`J_{ff}`",
        default=numpy.array([4.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.0001),
        doc=r"""Synaptic strength FS -> FS (GABA\ :sub:`A` onto the FS population).""",
    )

    E_AMPA = NArray(
        label=r":math:`E_{\mathrm{AMPA}}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.01),
        doc="""Synaptic reversal potential of the excitatory (AMPA-like) channels.""",
    )

    E_GABA = NArray(
        label=r":math:`E_{\mathrm{GABA}}`",
        default=numpy.array([-65.0]),
        domain=Range(lo=-100.0, hi=100.0, step=0.01),
        doc=r"""Synaptic reversal potential of the inhibitory (GABA\ :sub:`A`-like) channels.""",
    )

    # ---------------------------------------------------------------------- #
    # External (inter-node) coupling weights
    # ---------------------------------------------------------------------- #
    cr = NArray(
        label=":math:`c_r`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.1),
        doc="""Weight on the external coupling through the firing-rate variable r.""",
    )

    cv = NArray(
        label=":math:`c_v`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.1),
        doc="""Weight on the external coupling through the membrane-potential variable v.""",
    )

    coupling_terms = Final(
        label="Coupling terms",
        default=["Coupling_Term_r", "Coupling_Term_V"],
    )

    parameter_names = List(
        of=str,
        label="List of parameters for this model",
        default="C k v_r v_theta b tau_u tau_s kappa Delta I "
                "C_i k_i v_r_i v_theta_i b_i tau_u_i tau_s_i kappa_i Delta_i I_i "
                "J_rr J_rf J_fr J_ff E_AMPA E_GABA cr cv".split(),
    )

    state_variable_dfuns = Final(
        label="Drift functions for numba codegen",
        default={
            "r": "(r * (k * (2 * v - v_r - v_theta) - s * J_rr - s_i * J_rf) + Delta * k**2 * np.sign(v - v_r) * (v - v_r) / (np.pi * C)) / C",
            "v": "(- np.pi * C * r * (Delta * np.sign(v - v_r) + np.pi * C * r / k) + I + s * J_rr * (E_AMPA - v) + s_i * J_rf * (E_GABA - v) + k * (v - v_r) * (v - v_theta) - u + cr * Coupling_Term_r + cv * Coupling_Term_V) / C",
            "u": "b * (v - v_r) / tau_u - u / tau_u + kappa * r",
            "s": "-s / tau_s + r",
            "r_i": "(r_i * (k_i * (2 * v_i - v_r_i - v_theta_i) - s * J_fr - s_i * J_ff) + Delta_i * k_i**2 * np.sign(v_i - v_r_i) * (v_i - v_r_i) / (np.pi * C_i)) / C_i",
            "v_i": "(- np.pi * C_i * r_i * (Delta_i * np.sign(v_i - v_r_i) + np.pi * C_i * r_i / k_i) + I_i + s * J_fr * (E_AMPA - v_i) + s_i * J_ff * (E_GABA - v_i) + k_i * (v_i - v_r_i) * (v_i - v_theta_i) - u_i) / C_i",
            "u_i": "b_i * (v_i - v_r_i) / tau_u_i - u_i / tau_u_i + kappa_i * r_i",
            "s_i": "-s_i / tau_s_i + r_i",
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
            "r_i": numpy.array([0.0, 0.5]),
            "v_i": numpy.array([-70.0, -40.0]),
            "u_i": numpy.array([0.0, 20.0]),
            "s_i": numpy.array([0.0, 5.0]),
        },
        doc="""Expected ranges of the state variables for initial condition generation and phase plane setup.""",
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            "r": numpy.array([0.0, numpy.inf]),
            "r_i": numpy.array([0.0, numpy.inf]),
        },
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("r", "v", "u", "s", "r_i", "v_i", "u_i", "s_i"),
        default=("r",),
        doc="The quantities of interest for monitoring the coupled RS-FS mean-field populations.",
    )

    state_variables = ("r", "v", "u", "s", "r_i", "v_i", "u_i", "s_i")
    _nvar = 8
    # Cvar is the coupling variable: only r and v couple externally.
    cvar = numpy.array([0, 1], dtype=numpy.int32)
    # Stvar is the variable where stimulus is applied (RS membrane potential).
    stvar = numpy.array([1], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
            8D mean-field model of a coupled RS (excitatory) and FS (inhibitory)
            population of adaptive (Izhikevich) neurons with Lorentzian
            heterogeneity, as derived by [Gast_Solla_Kennedy_2024]_.

            Coupling within each node is handled here through the four synaptic
            channels :math:`J_{rr}, J_{rf}, J_{fr}, J_{ff}` and the reversal
            potentials :math:`E_{\mathrm{AMPA}}, E_{\mathrm{GABA}}`.  The
            external (inter-node) coupling :math:`C_r, C_V` enters the RS
            membrane-potential equation weighted by :math:`c_r, c_v`.
        """
        r, v, u, s, r_i, v_i, u_i, s_i = state_variables

        # [State_variables, nodes]
        C = self.C
        k = self.k
        v_r = self.v_r
        v_theta = self.v_theta
        b = self.b
        tau_u = self.tau_u
        tau_s = self.tau_s
        kappa = self.kappa
        Delta = self.Delta
        I = self.I

        C_i = self.C_i
        k_i = self.k_i
        v_r_i = self.v_r_i
        v_theta_i = self.v_theta_i
        b_i = self.b_i
        tau_u_i = self.tau_u_i
        tau_s_i = self.tau_s_i
        kappa_i = self.kappa_i
        Delta_i = self.Delta_i
        I_i = self.I_i

        J_rr = self.J_rr
        J_rf = self.J_rf
        J_fr = self.J_fr
        J_ff = self.J_ff
        E_AMPA = self.E_AMPA
        E_GABA = self.E_GABA
        cr = self.cr
        cv = self.cv

        # Internal synaptic activations, pre-scaled by the coupling strengths.
        s_ee = s * J_rr          # AMPA onto the RS population (from its own rate)
        s_ei = s_i * J_rf        # GABA onto the RS population (from the FS rate)
        s_ie = s * J_fr          # AMPA onto the FS population (from the RS rate)
        s_ii = s_i * J_ff        # GABA onto the FS population (from its own rate)

        Coupling_Term_r = coupling[0, :]  # external input through the firing rate r
        Coupling_Term_V = coupling[1, :]  # external input through the membrane potential v

        derivative = numpy.empty_like(state_variables)

        derivative[0] = (
            r * (k * (2 * v - v_r - v_theta) - s_ee - s_ei)
            + Delta * k**2 * numpy.sign(v - v_r) * (v - v_r) / (numpy.pi * C)
        ) / C
        derivative[1] = (
            k * (v - v_r) * (v - v_theta)
            - numpy.pi * C * r * (Delta * numpy.sign(v - v_r) + numpy.pi * C * r / k)
            - u + I
            + s_ee * (E_AMPA - v) + s_ei * (E_GABA - v)
            + cr * Coupling_Term_r + cv * Coupling_Term_V
        ) / C
        derivative[2] = b * (v - v_r) / tau_u - u / tau_u + kappa * r
        derivative[3] = -s / tau_s + r

        derivative[4] = (
            r_i * (k_i * (2 * v_i - v_r_i - v_theta_i) - s_ie - s_ii)
            + Delta_i * k_i**2 * numpy.sign(v_i - v_r_i) * (v_i - v_r_i) / (numpy.pi * C_i)
        ) / C_i
        derivative[5] = (
            k_i * (v_i - v_r_i) * (v_i - v_theta_i)
            - numpy.pi * C_i * r_i * (Delta_i * numpy.sign(v_i - v_r_i) + numpy.pi * C_i * r_i / k_i)
            - u_i + I_i
            + s_ie * (E_AMPA - v_i) + s_ii * (E_GABA - v_i)
        ) / C_i
        derivative[6] = b_i * (v_i - v_r_i) / tau_u_i - u_i / tau_u_i + kappa_i * r_i
        derivative[7] = -s_i / tau_s_i + r_i

        return derivative
