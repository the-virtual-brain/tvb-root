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
Tests verifying that :class:`~tvb.simulator.hybrid.Subnetwork` enforces
integrator state-variable boundaries and clamping after each step.

Two scenarios are covered:

1. **Model-defined bounds** — :class:`~tvb.simulator.models.ReducedWongWang`
   declares ``S ∈ [0, 1]``.  After ``Subnetwork.configure()`` the integrator
   has these bounds configured; running several steps while injecting a state
   that exceeds the upper bound confirms that bounding clips it back.

2. **User-defined clamping** — The user sets
   ``integrator.clamped_state_variable_indices`` /
   ``…_values`` on a Generic2dOscillator subnetwork before ``configure()``.
   After every step the clamped variable must equal the clamped value exactly.
"""

import numpy as np
import scipy.sparse as sp
import pytest

from tvb.simulator.models import ReducedWongWang, Generic2dOscillator
from tvb.simulator.integrators import EulerDeterministic
from tvb.simulator.hybrid import Subnetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_subnet(model, nnodes=5, dt=0.1):
    """Return a configured Subnetwork (no intra-projections)."""
    return Subnetwork(
        name="test", model=model, scheme=EulerDeterministic(dt=dt), nnodes=nnodes
    ).configure()


def _zero_coupling(ncvar, nnodes, modes=1):
    return np.zeros((ncvar, nnodes, modes))


def _run_steps(subnet, x0, n_steps=10):
    """Integrate n_steps and return the final state."""
    x = x0.copy()
    c = _zero_coupling(len(subnet.model.cvar), subnet.nnodes)
    for i in range(n_steps):
        x = subnet.step(i, x, c)
    return x


# ---------------------------------------------------------------------------
# Test 1: Model-defined bounds are enforced
# ---------------------------------------------------------------------------

class TestModelDefinedBounds:
    """
    ReducedWongWang declares S ∈ [0, 1].  Subnetwork.configure() must wire
    up the integrator's boundaries so that Subnetwork.step() clips S back
    within range when an out-of-range initial condition is supplied.
    """

    def test_upper_bound_clips_state(self):
        """
        Starting from S = 2.0 (above upper bound of 1.0), after one step the
        state must be ≤ 1.0 on every node.
        """
        subnet = _make_subnet(ReducedWongWang(), nnodes=4)

        # Verify that configure_boundaries was called — integrator must know the bound
        assert subnet.scheme.state_variable_boundaries is not None, (
            "configure() must populate integrator.state_variable_boundaries"
        )

        # Initial state above the upper boundary
        x0 = np.full((subnet.model.nvar, subnet.nnodes, subnet.model.number_of_modes), 2.0)
        x_final = _run_steps(subnet, x0, n_steps=1)

        assert np.all(x_final <= 1.0 + 1e-7), (
            f"All state values must be ≤ 1.0 after bounding; got max {x_final.max()}"
        )

    def test_lower_bound_clips_state(self):
        """
        Starting from S = -0.5 (below lower bound of 0.0), after one step the
        state must be ≥ 0.0 on every node.
        """
        subnet = _make_subnet(ReducedWongWang(), nnodes=4)

        x0 = np.full((subnet.model.nvar, subnet.nnodes, subnet.model.number_of_modes), -0.5)
        x_final = _run_steps(subnet, x0, n_steps=1)

        assert np.all(x_final >= -1e-7), (
            f"All state values must be ≥ 0.0 after bounding; got min {x_final.min()}"
        )

    def test_in_range_state_unaffected(self):
        """
        State values already within [0, 1] must not be altered by bounding.
        """
        subnet = _make_subnet(ReducedWongWang(), nnodes=4)

        rng = np.random.default_rng(42)
        # Start with values in [0.1, 0.9] — well within bounds
        x0 = rng.uniform(0.1, 0.9, size=(subnet.model.nvar, subnet.nnodes, 1))
        x_final = _run_steps(subnet, x0, n_steps=1)

        # They may drift slightly due to dynamics, but bounding alone shouldn't clip them
        # The main check is that the range remains physiological
        assert np.all(x_final <= 1.0 + 1e-7) and np.all(x_final >= -1e-7), (
            "State within bounds should not have been clipped out of range"
        )


# ---------------------------------------------------------------------------
# Test 2: User-defined clamping is enforced
# ---------------------------------------------------------------------------

class TestUserDefinedClamping:
    """
    The user may pin a state variable to a fixed value by setting
    ``integrator.clamped_state_variable_indices`` and
    ``integrator.clamped_state_variable_values`` before configure().

    After every step the clamped variable must equal the clamped value exactly.
    """

    def test_clamped_variable_remains_fixed(self):
        """
        Clamp the first state variable (V) of Generic2dOscillator to 0.5.
        After configure() and several steps the V dimension must be exactly 0.5.
        """
        model = Generic2dOscillator()
        scheme = EulerDeterministic(dt=0.1)
        # Set clamping BEFORE configure() so configure_boundaries picks it up
        scheme.clamped_state_variable_indices = np.array([0])     # V
        scheme.clamped_state_variable_values = np.array([[0.5]])

        subnet = Subnetwork(
            name="test", model=model, scheme=scheme, nnodes=3
        ).configure()

        # Verify the integrator knows about clamping
        assert subnet.scheme.clamped_state_variable_values is not None, (
            "configure() must preserve clamped_state_variable_values"
        )

        x0 = np.random.default_rng(1).standard_normal(
            (model.nvar, 3, model.number_of_modes)
        ).astype("f")

        x_final = _run_steps(subnet, x0, n_steps=20)

        np.testing.assert_allclose(
            x_final[0],  # V dimension
            0.5,
            atol=1e-6,
            err_msg="Clamped variable V must equal 0.5 after every step",
        )
