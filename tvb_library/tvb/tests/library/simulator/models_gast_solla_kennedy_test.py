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
Parity tests for the two-population ``GastSollaKennedy`` adaptive-Izhikevich
mean-field model driven by the traditional ``tvb.simulator.simulator.Simulator``.

The 2024 model couples a regular-spiking excitatory (RS) population with a
fast-spiking inhibitory (FS) population *within* each node through four
synaptic channels (``J_rr/J_rf/J_fr/J_ff``) and two reversal potentials
(``E_AMPA/E_GABA``).  Two properties are tested against independent plain-NumPy
integrations of the *same* equations:

1. **Full parity**: an 8-state-variable Simulator trajectory (all internal
   channels active, zero external coupling) matches a plain-NumPy Heun
   integration of the full coupled equations for both populations' firing
   rates.
2. **Decoupling**: with the cross-population channels disabled
   (``J_rf = J_fr = 0``) the RS and FS populations are two independent masses;
   each matches its own plain-NumPy single-population integration.

The Simulator's ``HeunDeterministic`` scheme is a predictor-corrector
(explicit trapezoidal) method, so the NumPy references reproduce that same
scheme.  A plain forward-Euler reference at ``dt=0.01`` leaves an ``O(1e-3)``
truncation difference against the Simulator's trajectory; mirroring the
integrator's scheme isolates the model's equations, which is what these tests
verify.

.. moduleauthor:: TVB contributors
"""

import numpy

from tvb.datatypes import connectivity
from tvb.simulator import simulator, coupling, integrators, monitors, models
from tvb.tests.library.base_testcase import BaseTestCase


DT = 0.01          # integration step (ms)
PERIOD = 1.0       # TemporalAverage period (ms)
SIM_LENGTH = 100.0  # ms -> 10000 integration steps -> 100 monitored samples
I_EXT = 2.0        # fixed external homogeneous current on the RS population (pA)
I_EXT_I = 1.0      # external current on the FS population (pA)
RTOL = 1e-3


class TestGastSollaKennedy(BaseTestCase):
    """Traditional-simulator parity for the two-population GastSollaKennedy model."""

    @staticmethod
    def _model_parameters(model):
        """Scalars for the full 8-variable equations, read off a configured model."""
        return dict(
            C=float(model.C[0]), k=float(model.k[0]),
            v_r=float(model.v_r[0]), v_theta=float(model.v_theta[0]),
            b=float(model.b[0]), tau_u=float(model.tau_u[0]),
            tau_s=float(model.tau_s[0]), kappa=float(model.kappa[0]),
            Delta=float(model.Delta[0]), I=float(model.I[0]),
            C_i=float(model.C_i[0]), k_i=float(model.k_i[0]),
            v_r_i=float(model.v_r_i[0]), v_theta_i=float(model.v_theta_i[0]),
            b_i=float(model.b_i[0]), tau_u_i=float(model.tau_u_i[0]),
            tau_s_i=float(model.tau_s_i[0]), kappa_i=float(model.kappa_i[0]),
            Delta_i=float(model.Delta_i[0]), I_i=float(model.I_i[0]),
            J_rr=float(model.J_rr[0]), J_rf=float(model.J_rf[0]),
            J_fr=float(model.J_fr[0]), J_ff=float(model.J_ff[0]),
            E_AMPA=float(model.E_AMPA[0]), E_GABA=float(model.E_GABA[0]),
        )

    @classmethod
    def _dfun(cls, state, p):
        """RHS of the coupled two-population mean-field equations
        (zero external coupling).  State order: r, v, u, s, r_i, v_i, u_i, s_i."""
        r, v, u, s, r_i, v_i, u_i, s_i = state
        pi = numpy.pi
        sg = numpy.sign(v - p["v_r"])
        dr = (r * (p["k"] * (2 * v - p["v_r"] - p["v_theta"])
                   - s * p["J_rr"] - s_i * p["J_rf"])
              + p["Delta"] * p["k"] ** 2 * (v - p["v_r"]) * sg / (pi * p["C"])) / p["C"]
        dv = (p["k"] * (v - p["v_r"]) * (v - p["v_theta"])
              - pi * p["C"] * r * (p["Delta"] * sg + pi * p["C"] * r / p["k"])
              - u + p["I"]
              + s * p["J_rr"] * (p["E_AMPA"] - v) + s_i * p["J_rf"] * (p["E_GABA"] - v)) / p["C"]
        du = p["b"] * (v - p["v_r"]) / p["tau_u"] - u / p["tau_u"] + p["kappa"] * r
        ds = -s / p["tau_s"] + r
        sg_i = numpy.sign(v_i - p["v_r_i"])
        dr_i = (r_i * (p["k_i"] * (2 * v_i - p["v_r_i"] - p["v_theta_i"])
                       - s * p["J_fr"] - s_i * p["J_ff"])
                + p["Delta_i"] * p["k_i"] ** 2 * (v_i - p["v_r_i"]) * sg_i / (pi * p["C_i"])) / p["C_i"]
        dv_i = (p["k_i"] * (v_i - p["v_r_i"]) * (v_i - p["v_theta_i"])
                - pi * p["C_i"] * r_i * (p["Delta_i"] * sg_i + pi * p["C_i"] * r_i / p["k_i"])
                - u_i + p["I_i"]
                + s * p["J_fr"] * (p["E_AMPA"] - v_i) + s_i * p["J_ff"] * (p["E_GABA"] - v_i)) / p["C_i"]
        du_i = p["b_i"] * (v_i - p["v_r_i"]) / p["tau_u_i"] - u_i / p["tau_u_i"] + p["kappa_i"] * r_i
        ds_i = -s_i / p["tau_s_i"] + r_i
        return numpy.array([dr, dv, du, ds, dr_i, dv_i, du_i, ds_i])

    @classmethod
    def _reference_rate(cls, x0, p, dt, n_steps, period):
        """Integrate the full equations with a Heun scheme, returning the
        windowed-mean firing rate at state indices (0, 4) = (r, r_i)."""
        state = numpy.array(x0, dtype=float)
        rec = numpy.zeros((n_steps, 2))
        for i in range(n_steps):
            k1 = cls._dfun(state, p)
            k2 = cls._dfun(state + dt * k1, p)
            state = state + dt * (k1 + k2) / 2.0
            state[0] = max(state[0], 0.0)   # model r >= 0 boundary
            state[4] = max(state[4], 0.0)   # model r_i >= 0 boundary
            rec[i] = (state[0], state[4])
        istep = int(round(period / dt))
        return rec.reshape(-1, istep, 2).mean(axis=1)

    @classmethod
    def _build_sim(cls, model):
        """A trivial zero-delay connectivity with all regions identical and
        uncoupled (external coupling disabled), running the given model."""
        conn = connectivity.Connectivity.from_file()
        conn.tract_lengths[:] = 0.0
        conn.configure()
        n_nodes = conn.number_of_regions

        x0 = numpy.array([0.05, -55.0, 10.0, 1.0, 0.02, -50.0, 2.0, 0.5])
        ic = numpy.repeat(x0.reshape(8, 1, 1), n_nodes, axis=1)
        initial_conditions = ic.reshape((1, 8, n_nodes, 1))

        sim = simulator.Simulator()
        sim.model = model
        sim.connectivity = conn
        sim.coupling = coupling.Linear(a=numpy.array([0.0]))
        sim.integrator = integrators.HeunDeterministic(dt=DT)
        sim.monitors = [monitors.TemporalAverage(period=PERIOD)]
        sim.initial_conditions = initial_conditions
        sim.simulation_length = SIM_LENGTH
        return sim, x0

    def _run_and_check_full_parity(self, model):
        """Shared assertion body: Simulator trajectory vs plain-NumPy Heun."""
        model.variables_of_interest = ("r", "r_i")
        model.configure()
        sim, x0 = self._build_sim(model)
        sim.configure()
        monitored = numpy.array(
            [step[0][1][:, 0, 0] for step in sim(simulation_length=SIM_LENGTH)]
        )
        # monitored: (time, 2) with columns (r, r_i)
        assert monitored.shape == (int(SIM_LENGTH / PERIOD), 2)
        assert numpy.all(numpy.isfinite(monitored))

        p = self._model_parameters(model)
        n_steps = int(round(SIM_LENGTH / DT))
        reference = self._reference_rate(x0, p, DT, n_steps, PERIOD)

        numpy.testing.assert_allclose(monitored[:, 0], reference[:, 0], rtol=RTOL)
        numpy.testing.assert_allclose(monitored[:, 1], reference[:, 1], rtol=RTOL)

    def test_traditional_simulator_parity_coupled(self):
        """Full coupled two-population trajectory matches plain-NumPy Heun."""
        model = models.GastSollaKennedy(I=numpy.array([I_EXT]),
                                        I_i=numpy.array([I_EXT_I]))
        self._run_and_check_full_parity(model)

    def test_traditional_simulator_parity_decoupled(self):
        """With J_rf = J_fr = 0 the RS and FS masses decouple and each matches
        its own single-population plain-NumPy reference."""
        model = models.GastSollaKennedy(I=numpy.array([I_EXT]),
                                        I_i=numpy.array([I_EXT_I]),
                                        J_rf=numpy.array([0.0]),
                                        J_fr=numpy.array([0.0]))
        self._run_and_check_full_parity(model)

    def test_zero_cross_coupling_decouples_into_independent_masses(self):
        """Equation-level check that disabling cross coupling makes the full
        RHS blockwise-identical to the RS and FS single-population equations."""
        model = models.GastSollaKennedy(I=numpy.array([I_EXT]),
                                        I_i=numpy.array([I_EXT_I]),
                                        J_rf=numpy.array([0.0]),
                                        J_fr=numpy.array([0.0]))
        model.configure()
        p = self._model_parameters(model)

        rng = numpy.random.RandomState(0)
        states = [rng.uniform(-1, 1, 8) for _ in range(5)]
        for y in states:
            full = self._dfun(y, p)
            # RS block: indices 0-3 of the full RHS with FS drive zeroed (J_rf=0)
            rs_y = numpy.array([y[0], y[1], y[2], y[3]])
            rs = self._dfun_single(rs_y, p, rs=True)
            # FS block: indices 4-7 with RS drive zeroed (J_fr=0)
            fs_y = numpy.array([y[4], y[5], y[6], y[7]])
            fs = self._dfun_single(fs_y, p, rs=False)
            expected = numpy.r_[rs, fs]
            numpy.testing.assert_allclose(full, expected, atol=1e-12,
                                          err_msg="cross-coupling disabled should decouple the masses")

    @classmethod
    def _dfun_single(cls, y4, p, rs):
        """Plain-NumPy single-population RHS (RS if rs else FS)."""
        r, v, u, s = y4
        pi = numpy.pi
        if rs:
            C, k, vr, vt = p["C"], p["k"], p["v_r"], p["v_theta"]
            b_, tau_u, tau_s = p["b"], p["tau_u"], p["tau_s"]
            kappa, Delta, I = p["kappa"], p["Delta"], p["I"]
            Jsel, Esel = p["J_rr"], p["E_AMPA"]
        else:
            C, k, vr, vt = p["C_i"], p["k_i"], p["v_r_i"], p["v_theta_i"]
            b_, tau_u, tau_s = p["b_i"], p["tau_u_i"], p["tau_s_i"]
            kappa, Delta, I = p["kappa_i"], p["Delta_i"], p["I_i"]
            Jsel, Esel = p["J_ff"], p["E_GABA"]
        sg = numpy.sign(v - vr)
        dr = (r * (k * (2 * v - vr - vt) - s * Jsel)
              + Delta * k ** 2 * (v - vr) * sg / (pi * C)) / C
        dv = (k * (v - vr) * (v - vt) - pi * C * r * (Delta * sg + pi * C * r / k)
              - u + I + s * Jsel * (Esel - v)) / C
        du = b_ * (v - vr) / tau_u - u / tau_u + kappa * r
        ds = -s / tau_s + r
        return numpy.array([dr, dv, du, ds])
