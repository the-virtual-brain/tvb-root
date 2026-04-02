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
Validate steady-state convergence for hybrid simulator models.

Each test initialises a single-node subnetwork at the centre of the model's
state-variable range and integrates without any inter-node coupling or
external stimuli.  The models are parameterised to be in a stable fixed-point
regime so they should converge toward a steady state.

**Convergence criterion** (used in ``_test_model_steady_state``): let
``rfp[k] = ||y[-1] - y[k]||`` be the Euclidean distance from the final state.
Compute the ratio ``dr[k] = rfp[k] / (rfp[k-1] + ε)`` for the last 20
steps.  Assert that all ratios are < 1.0, meaning the trajectory is
monotonically approaching the attractor (not diverging) in the tail of the
run.
"""

import numpy as np
from tvb.tests.library.simulator.hybrid.test_validation_base import ValidationTestBase
from tvb.simulator.models import Generic2dOscillator, JansenRit, ReducedWongWang
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.monitors import Raw
from tvb.simulator.hybrid import Subnetwork, NetworkSet, Simulator


class TestValidationSteadyState(ValidationTestBase):
    """Steady-state convergence tests for hybrid simulator models.

    Models are parameterised to be in a stable fixed-point regime.  The
    shared helper :meth:`_test_model_steady_state` integrates each model for
    a specified number of steps and checks the convergence criterion described
    in the module docstring.
    """

    def _test_model_steady_state(self, model_instance, dt=0.01, n_step=2000):
        """Integrate *model_instance* and assert convergence to a fixed point.

        The model is initialised at the midpoint of each state variable's
        ``state_variable_range`` and integrated for ``n_step`` steps with no
        coupling.

        **Convergence criterion**: let ``rfp[k] = ||y[-1] - y[k]||`` be the
        Euclidean distance from the final state.  Compute the ratio
        ``dr[k] = rfp[k] / (rfp[k-1] + ε)`` for the last 20 steps.  Assert
        that all ratios are < 1.0, meaning the trajectory is monotonically
        approaching the attractor (not diverging) in the tail of the run.

        Parameters
        ----------
        model_instance : Model
            Configured TVB model instance set to a stable fixed-point regime.
        dt : float, default=0.01
            Integration time step in milliseconds.
        n_step : int, default=2000
            Total number of integration steps.
        """
        scheme = HeunDeterministic(dt=dt)
        subnet = Subnetwork(
            name="test_model", model=model_instance, scheme=scheme, nnodes=1
        ).configure()

        model = subnet.model
        svr = model.state_variable_range
        sv_names = list(svr.keys())

        init = np.zeros((model._nvar, 1, model.number_of_modes))
        for i, sv_name in enumerate(sv_names):
            lo, hi = svr[sv_name]
            init[i, :, :] = (hi + lo) / 2.0

        nets = NetworkSet(subnets=[subnet], projections=[])
        sim = Simulator(nets=nets, simulation_length=n_step * dt, monitors=[Raw()])
        sim.configure()
        ((t, y),) = sim.run(initial_conditions=[init])

        rfp = np.sqrt(np.sum(np.sum((y[-1] - y) ** 2, axis=1), axis=1))
        dr = rfp[-20:] / (rfp[-21:-1] + 1e-9)
        assert (dr < 1.0).all(), "Model did not converge to steady state"

    def test_generic_2d_oscillator_steady_state(self):
        """Generic2dOscillator with ``a=-1`` must converge to a stable fixed point.

        Negative ``a`` places the model in a fixed-point (non-oscillatory)
        regime.  ``b=0`` removes the slow-variable offset.
        """
        model = Generic2dOscillator(a=np.array([-1.0]), b=np.array([0.0]))
        self._test_model_steady_state(model)

    def test_generic_2d_oscillator_excitable(self):
        """Generic2dOscillator in excitable regime must converge to a stable fixed point.

        Parameters place the model near but not past the Hopf bifurcation
        (``a=1.05``, cubic nullcline), so the system is excitable rather than
        oscillatory.  The trajectory should settle to a fixed point over
        2000 steps at dt=0.01 ms.
        """
        model = Generic2dOscillator(
            a=np.array([1.05]),
            b=np.array([-1.00]),
            c=np.array([0.0]),
            d=np.array([0.1]),
            I=np.array([0.0]),
            alpha=np.array([1.0]),
            beta=np.array([0.2]),
            gamma=np.array([-1.0]),
            e=np.array([0.0]),
            g=np.array([1.0]),
            f=np.array([1.0 / 3.0]),
            tau=np.array([1.25]),
        )
        self._test_model_steady_state(model)

    def test_jansen_rit_steady_state(self):
        """JansenRit must reach a physiological steady state.

        Uses a longer run (3000 steps at dt=0.05 ms) because the JansenRit
        model has slower dynamics and requires more integration time to reach
        its fixed point from the midpoint initialisation.
        """
        model = JansenRit(
            A=np.array([3.25]),
            B=np.array([22.0]),
            a=np.array([0.1]),
            b=np.array([0.05]),
            v0=np.array([6.0]),
            r=np.array([0.56]),
        )
        self._test_model_steady_state(model, dt=0.05, n_step=3000)

    def test_reduced_wong_wang_steady_state(self):
        """ReducedWongWang must converge to a low-activity steady state.

        Parameters set the single-node model to a monostable low-firing-rate
        fixed point.  Uses 5000 steps at dt=0.1 ms due to the slow synaptic
        time constant (``tau_s=100 ms``).
        """
        model = ReducedWongWang(
            a=np.array([0.270]),
            b=np.array([0.108]),
            d=np.array([154.0]),
            gamma=np.array([0.641]),
            tau_s=np.array([100.0]),
            w=np.array([0.9]),
            J_N=np.array([0.2609]),
            I_o=np.array([0.33]),
        )
        self._test_model_steady_state(model, dt=0.1, n_step=5000)

    def test_stability_small_perturbation(self):
        """Small perturbations around the fixed point must return toward steady state.

        Two nodes are initialised with equal-and-opposite small perturbations
        (±0.01).  After 500 ms they must converge to the same final state to
        within ``rtol=1e-4, atol=1e-4``, confirming asymptotic stability.
        """
        scheme = HeunDeterministic(dt=0.01)
        model = Generic2dOscillator(a=np.array([-1.0]), b=np.array([0.0]))
        subnet = Subnetwork(
            name="stable_model", model=model, scheme=scheme, nnodes=2
        ).configure()

        svr = model.state_variable_range
        init = np.zeros((model._nvar, 2, model.number_of_modes))
        init[0, 0, :] = 0.01
        init[1, 0, :] = 0.01
        init[0, 1, :] = -0.01
        init[1, 1, :] = -0.01

        nets = NetworkSet(subnets=[subnet], projections=[])
        sim = Simulator(nets=nets, simulation_length=500.0, monitors=[Raw()])
        sim.configure()
        ((t, y),) = sim.run(initial_conditions=[init])

        n_vois = y.shape[1]
        final_states = y[-1, :, :, 0]
        # Check both nodes converge to the same steady state
        np.testing.assert_allclose(
            final_states[:, 0],
            final_states[:, 1],
            rtol=1e-4,
            atol=1e-4,
            err_msg="Nodes did not converge to the same steady state",
        )
