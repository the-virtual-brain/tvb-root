"""
Tests for the Subnetwork and Stim classes.
"""

import numpy as np
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
    """Tests for the Subnetwork class."""

    def test_subnetwork_shapes(self):
        """Test subnetwork state and coupling variable shapes"""
        conn, ix, c, t, a, nets = self.setup()
        self.assert_equal((6, (ix == a.CORTEX).sum(), 1), c.zero_states().shape)
        self.assert_equal((2, (ix == a.CORTEX).sum(), 1), c.zero_cvars().shape)
        self.assert_equal((4, (ix == a.THALAMUS).sum(), 3), t.zero_states().shape)

    def _test_subnetwork_with_integrator(self, integrator_cls, **integrator_kwargs):
        """Helper method to test a subnetwork with a given integrator."""
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
        """Test subnetwork stepping with EulerDeterministic."""
        self._test_subnetwork_with_integrator(EulerDeterministic)

    def test_heun_deterministic(self):
        """Test subnetwork stepping with HeunDeterministic."""
        self._test_subnetwork_with_integrator(HeunDeterministic)

    def test_rk4_deterministic(self):
        """Test subnetwork stepping with RungeKutta4thOrderDeterministic."""
        self._test_subnetwork_with_integrator(RungeKutta4thOrderDeterministic)

    def test_euler_stochastic(self):
        """Test subnetwork stepping with EulerStochastic."""
        self._test_subnetwork_with_integrator(EulerStochastic)

    def test_heun_stochastic(self):
        """Test subnetwork stepping with HeunStochastic."""
        self._test_subnetwork_with_integrator(HeunStochastic)

    def test_subnetwork_internal_projection(self):
        """Test subnetwork stepping with an internal projection."""
        nnodes = 5
        model = JansenRit()
        scheme = EulerDeterministic(dt=0.1)
        model.configure()
        scheme.configure()

        internal_proj = IntraProjection(
            source_cvar=np.array([0]),  # Example: first coupling variable
            target_cvar=np.array([0]),  # Example: maps to first coupling variable
            weights=np.eye(nnodes),     # Identity weights
            scale=1.0                   # Scale factor
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
    """Tests for the Stim class."""

    def test_stim(self):
        """Test stimulus application to network"""
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
