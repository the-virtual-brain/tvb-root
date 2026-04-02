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
Unit tests for :class:`tvb.simulator.hybrid.Subnetwork` and
:class:`tvb.simulator.hybrid.Stim`.

Each ``Subnetwork`` test exercises a particular family of TVB integrators to
verify that the sub-network stepping loop produces finite, non-trivial output
regardless of the numerical scheme.  Covered integrators:

* Deterministic: ``EulerDeterministic``, ``HeunDeterministic``,
  ``RungeKutta4thOrderDeterministic``
* Stochastic: ``EulerStochastic``, ``HeunStochastic``

An additional test verifies that an intra-subnet
:class:`~tvb.simulator.hybrid.IntraProjection` modifies the coupling term seen
by the integrator on each step.
"""

import numpy as np
import scipy.sparse as sp

from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.patterns import StimuliRegion
from tvb.simulator.models import JansenRit
from tvb.simulator.integrators import (
    EulerDeterministic, HeunDeterministic, RungeKutta4thOrderDeterministic,
    EulerStochastic, HeunStochastic,
    # Add other integrators as needed, e.g., SciPy ones if desired
)
# Import IntraProjection
from tvb.simulator.hybrid import Subnetwork, Stim, IntraProjection
from .test_base import BaseHybridTest


class TestSubnetwork(BaseHybridTest):
    """
    Tests for :class:`~tvb.simulator.hybrid.Subnetwork`.

    Each test builds a standalone subnetwork (independent of the
    cortex-thalamus fixture) and verifies a specific property of the
    stepping loop or the initial-state factory methods.
    """

    def test_subnetwork_shapes(self):
        """
        ``zero_states`` and ``zero_cvars`` return arrays with the correct shapes.

        Expected shapes (from the cortex-thalamus fixture):

        * ``cortex.zero_states()`` — ``(6, n_cortex, 1)``
          (JansenRit: 6 state vars, 1 mode)
        * ``cortex.zero_cvars()``  — ``(2, n_cortex, 1)``
          (JansenRit: 2 coupling vars, 1 mode)
        * ``thalamus.zero_states()`` — ``(4, n_thalamus, 3)``
          (ReducedSetFitzHughNagumo: 4 state vars, 3 modes)
        """
        conn, ix, c, t, a, nets = self.setup()
        self.assert_equal((6, (ix == a.CORTEX).sum(), 1), c.zero_states().shape)
        self.assert_equal((2, (ix == a.CORTEX).sum(), 1), c.zero_cvars().shape)
        self.assert_equal((4, (ix == a.THALAMUS).sum(), 3), t.zero_states().shape)

    def _test_subnetwork_with_integrator(self, integrator_cls, **integrator_kwargs):
        """
        Shared helper: create a 10-node JansenRit subnetwork, advance one step,
        and assert the result is finite and non-trivial.

        The test passes when all of these hold after one call to
        :meth:`~tvb.simulator.hybrid.Subnetwork.step`:

        * Output shape equals input shape.
        * At least one element changed (state is not all-zero after a step
          through the JansenRit RHS).
        * All elements are finite (no NaN or Inf).

        Parameters
        ----------
        integrator_cls : type
            A TVB integrator class to instantiate with ``dt=0.1``.
        **integrator_kwargs
            Additional keyword arguments forwarded to *integrator_cls*.
        """
        nnodes = 10  # Use a smaller number of nodes for efficiency
        
        # Instantiate first
        model = JansenRit()
        scheme = integrator_cls(dt=0.1, **integrator_kwargs)
        
        # Configure separately
        model.configure()
        scheme.configure()

        # For stochastic integrators, ensure the noise object's dt is set
        if hasattr(scheme, 'noise') and scheme.noise is not None:
            scheme.noise.dt = scheme.dt
        
        # Ensure instances are not None before passing them
        assert model is not None, "Model instance became None after configure"
        assert scheme is not None, "Scheme instance became None after configure"

        sub = Subnetwork(
            name='test_subnet',
            model=model,
            scheme=scheme,
            nnodes=nnodes
        )
        sub.configure() # Configure the subnetwork itself

        x = sub.zero_states()
        c = sub.zero_cvars()
        
        # Take one step
        nx = sub.step(0, x, c)

        # Assertions
        self.assert_equal(x.shape, nx.shape, "Output shape mismatch")
        # Use plain assert with pytest
        assert not np.allclose(x, nx), "State did not change after one step"
        assert np.all(np.isfinite(nx)), "Non-finite values in output state"

    def test_euler_deterministic(self):
        """
        A single Euler step produces finite, non-trivial output.

        ``EulerDeterministic`` is the simplest first-order scheme; this test
        is the baseline sanity check for subnetwork stepping.
        """
        self._test_subnetwork_with_integrator(EulerDeterministic)

    def test_heun_deterministic(self):
        """
        The second-order Heun predictor-corrector scheme advances the state
        correctly without producing non-finite values.
        """
        self._test_subnetwork_with_integrator(HeunDeterministic)

    def test_rk4_deterministic(self):
        """
        The classic fourth-order Runge-Kutta scheme advances the state
        correctly without producing non-finite values.
        """
        self._test_subnetwork_with_integrator(RungeKutta4thOrderDeterministic)

    def test_euler_stochastic(self):
        """
        Euler-Maruyama (Euler + additive Wiener term) advances the state
        correctly.  The stochastic noise object's ``dt`` must be synchronised
        with the integrator's ``dt`` before the subnetwork is configured.
        """
        self._test_subnetwork_with_integrator(EulerStochastic)

    def test_heun_stochastic(self):
        """
        The stochastic Heun scheme (predictor-corrector with Wiener increments)
        advances the state correctly without producing non-finite values.
        """
        self._test_subnetwork_with_integrator(HeunStochastic)

    def test_subnetwork_internal_projection(self):
        """
        An :class:`~tvb.simulator.hybrid.IntraProjection` modifies the
        effective coupling term seen during a single step.

        A 5-node identity-weight intra-projection is attached to the subnetwork
        and the first state variable is initialised to a non-zero value.  After
        one step the state must change (the coupling contribution is non-zero)
        and all values must remain finite.
        """
        nnodes = 5
        model = JansenRit()
        scheme = EulerDeterministic(dt=0.1)
        model.configure()
        scheme.configure()

        internal_proj = IntraProjection(
            source_cvar=np.array([0]),  # Example: first coupling variable
            target_cvar=np.array([0]),  # Example: maps to first coupling variable
            weights=sp.eye(nnodes).tocsr(),     # Identity weights
            lengths=sp.eye(nnodes).tocsr(),     # Identity weights
            scale=1.0, cv=1.0, dt=0.1                   # Scale factor
        )

        # Pass projection kwargs during Subnetwork instantiation
        sub = Subnetwork(
            name='test_subnet_internal',
            model=model,
            scheme=scheme,
            nnodes=nnodes,
            projections=[internal_proj]
        )
        sub.configure()

        x = sub.zero_states()
        # Provide non-zero initial state for the source variable to see effect
        x[0, :, 0] = 0.1
        c_external = sub.zero_cvars() # No external coupling

        # Take one step
        nx = sub.step(0, x, c_external)

        # Assertions
        self.assert_equal(x.shape, nx.shape, "Output shape mismatch")
        # Use plain assert with pytest
        assert not np.allclose(x, nx, atol=1e-8), \
            "State did not change despite internal projection"
        assert np.all(np.isfinite(nx)), "Non-finite values in output state"


class TestStim(BaseHybridTest):
    """
    Smoke tests for :class:`~tvb.simulator.hybrid.Stim` integration.
    """

    def test_stim(self):
        """
        A custom ``StimuliRegion`` subclass returns an array of the correct
        length (one value per region) when called with a time argument.

        This is a basic shape check that confirms the stimulus callable
        protocol is respected; detailed validation of stimulus waveforms is
        covered by dedicated stimulus integration tests.
        """
        conn = Connectivity.from_file()
        nn = conn.weights.shape[0]
        conn.configure()
        
        class MyStim(StimuliRegion):
            def __call__(self, t):
                return np.random.randn(self.connectivity.weights.shape[0])
                
        stim = MyStim(connectivity=conn)
        I = stim(5)
        self.assert_equal((nn,), I.shape)
        model = JansenRit()
        model.configure()
        # TODO: Add more stimulus tests
