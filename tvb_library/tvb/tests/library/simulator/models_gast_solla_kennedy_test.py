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
Parity test for the ``GastSollaKennedy`` adaptive-Izhikevich mean-field model run
through the traditional ``tvb.simulator.simulator.Simulator``.

The test drives the Simulator with a single homogeneous population (zero
coupling, fixed external current ``I``), monitors the mean firing rate ``r`` via
``TemporalAverage``, and checks it against an independent plain-NumPy
integration of the *same* mean-field equations.

The Simulator's ``HeunDeterministic`` scheme is a predictor-corrector
(explicit trapezoidal) method, so the NumPy reference reproduces that same
scheme.  A plain forward-Euler reference at ``dt=0.01`` leaves an ``O(1e-3)``
truncation difference against the Simulator's trajectory, which would not pass
the ``rtol=1e-3`` parity assertion; mirroring the integrator's scheme isolates
the model's equations, which is what this test verifies.

.. moduleauthor:: TVB contributors
"""

import numpy

from tvb.datatypes import connectivity
from tvb.simulator import simulator, coupling, integrators, monitors, models
from tvb.tests.library.base_testcase import BaseTestCase


DT = 0.01          # integration step (ms)
PERIOD = 1.0       # TemporalAverage period (ms)
SIM_LENGTH = 100.0  # ms -> 10000 integration steps -> 100 monitored samples
I_EXT = 2.0        # fixed external homogeneous current (pA)


class TestGastSollaKennedy(BaseTestCase):
    """Traditional-simulator parity for the GastSollaKennedy model."""

    @staticmethod
    def _model_parameters(model):
        """Scalars for the mean-field equations, read off a configured model."""
        return dict(
            C=float(model.C[0]), k=float(model.k[0]),
            v_r=float(model.v_r[0]), v_theta=float(model.v_theta[0]),
            b=float(model.b[0]), tau_u=float(model.tau_u[0]),
            tau_s=float(model.tau_s[0]), J=float(model.J[0]),
            kappa=float(model.kappa[0]), Delta=float(model.Delta[0]),
            g=float(model.g[0]), E_r=float(model.E_r[0]),
            I=float(model.I[0]),
        )

    @classmethod
    def _dfun(cls, state, p):
        """RHS of the GastSollaKennedy mean-field equations (zero coupling)."""
        r, v, u, s = state
        C, k = p["C"], p["k"]
        v_r, v_theta = p["v_r"], p["v_theta"]
        tau_u, tau_s = p["tau_u"], p["tau_s"]
        J, kappa, Delta = p["J"], p["kappa"], p["Delta"]
        g, E_r, I = p["g"], p["E_r"], p["I"]
        step = numpy.sign(v - v_r)

        dr = (Delta * k ** 2 * step * (v - v_r) / (numpy.pi * C)
              + r * (k * (2.0 * v - v_r - v_theta) - g * s)) / C
        dv = (k * v * (v - v_r - v_theta) + k * v_r * v_theta
              - numpy.pi * C * r * (Delta * step + numpy.pi * C * r / k)
              - u + I + g * s * (E_r - v)) / C
        du = p["b"] * (v - v_r) / tau_u - u / tau_u + kappa * r
        ds = -s / tau_s + J * r
        return numpy.array([dr, dv, du, ds])

    @classmethod
    def _reference_rate(cls, x0, p, dt, n_steps, period):
        """Integrate the equations with the Simulator's Heun scheme, mirroring
        the TemporalAverage windowing of the monitored series."""
        state = numpy.array(x0, dtype=float)
        rec = numpy.zeros(n_steps)
        for i in range(n_steps):
            k1 = cls._dfun(state, p)
            k2 = cls._dfun(state + dt * k1, p)
            state = state + dt * (k1 + k2) / 2.0
            state[0] = max(state[0], 0.0)  # model r >= 0 boundary
            rec[i] = state[0]
        istep = int(round(period / dt))
        return rec.reshape(-1, istep).mean(axis=1)

    def test_traditional_simulator_parity(self):
        # A trivial structural connectivity with zero delays: the model runs on
        # `number_of_regions` identical, uncoupled populations.
        conn = connectivity.Connectivity.from_file()
        conn.tract_lengths[:] = 0.0
        conn.configure()
        n_nodes = conn.number_of_regions

        model = models.GastSollaKennedy(I=numpy.array([I_EXT]))

        x0 = numpy.array([0.05, -55.0, 10.0, 1.0])          # per-node IC
        ic = numpy.repeat(x0.reshape(4, 1, 1), n_nodes, axis=1)
        initial_conditions = ic.reshape((1, 4, n_nodes, 1))

        sim = simulator.Simulator()
        sim.model = model
        sim.connectivity = conn
        sim.coupling = coupling.Linear(a=numpy.array([0.0]))
        sim.integrator = integrators.HeunDeterministic(dt=DT)
        sim.monitors = [monitors.TemporalAverage(period=PERIOD)]
        sim.initial_conditions = initial_conditions
        sim.simulation_length = SIM_LENGTH
        sim.configure()

        monitored = numpy.array(
            [step[0][1][0, 0, 0] for step in sim(simulation_length=SIM_LENGTH)]
        )
        assert monitored.shape == (int(SIM_LENGTH / PERIOD),)
        assert numpy.all(numpy.isfinite(monitored))

        n_steps = int(round(SIM_LENGTH / DT))
        reference = self._reference_rate(x0, self._model_parameters(model),
                                         DT, n_steps, PERIOD)

        numpy.testing.assert_allclose(monitored, reference, rtol=1e-3)
