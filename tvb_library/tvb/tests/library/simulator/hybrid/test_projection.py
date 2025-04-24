"""
Tests for the Projection class.
"""

import numpy as np
import pytest
from tvb.simulator.hybrid import Projection
from .test_base import BaseHybridTest


class TestProjection(BaseHybridTest):
    """Tests for the Projection class."""

    def test_projection_broadcasting(self):
        """Test different coupling variable broadcasting configurations"""
        conn, ix, cortex, thalamus, a, nets = self.setup()
        
        # Get indices for test setup
        cortex_indices = np.where(ix == a.CORTEX)[0]
        thalamus_indices = np.where(ix == a.THALAMUS)[0]
        test_weights = conn.weights[thalamus_indices][:, cortex_indices]
        
        # Case a: One source to many targets (broadcasting)
        proj_broadcast = Projection(
            source=cortex,
            target=thalamus,
            source_cvar=np.r_[0],  # Single source
            target_cvar=np.r_[0:2],  # Multiple targets [0,1]
            weights=test_weights
        )
        
        # Case b: Many sources to one target (reduction)
        proj_reduce = Projection(
            source=cortex,
            target=thalamus,
            source_cvar=np.r_[0, 1],  # Multiple sources
            target_cvar=np.r_[1],  # Single target
            weights=test_weights
        )
        
        # Case c: Equal number of sources to targets (element-wise)
        proj_elementwise = Projection(
            source=cortex,
            target=thalamus,
            source_cvar=np.r_[0, 1],  # Two sources
            target_cvar=np.r_[0, 1],  # Two targets
            weights=test_weights
        )
        
        # Create random test states
        x = self._randn_like_states(nets.zero_states())
        
        # Test case a: broadcasting
        c_broadcast = nets.zero_cvars()
        proj_broadcast.apply(c_broadcast.thalamus, x.cortex)
        expected_val = test_weights @ x.cortex[0] @ proj_broadcast.mode_map
        assert c_broadcast.thalamus[0].shape == expected_val.shape
        np.testing.assert_allclose(c_broadcast.thalamus[0], expected_val)
        np.testing.assert_allclose(c_broadcast.thalamus[1], expected_val)
        
        # Test case b: reduction
        c_reduce = nets.zero_cvars()
        proj_reduce.apply(c_reduce.thalamus, x.cortex)
        expected_val = (test_weights @ x.cortex[0] + test_weights @ x.cortex[1]) @ proj_reduce.mode_map
        np.testing.assert_allclose(c_reduce.thalamus[1], expected_val)
        
        # Test case c: element-wise
        c_elementwise = nets.zero_cvars()
        proj_elementwise.apply(c_elementwise.thalamus, x.cortex)
        expected_val0 = test_weights @ x.cortex[0] @ proj_elementwise.mode_map
        expected_val1 = test_weights @ x.cortex[1] @ proj_elementwise.mode_map
        np.testing.assert_allclose(c_elementwise.thalamus[0], expected_val0)
        np.testing.assert_allclose(c_elementwise.thalamus[1], expected_val1)
        
        # Test invalid configuration
        with pytest.raises(ValueError):
            Projection(
                source=cortex,
                target=thalamus,
                source_cvar=np.r_[0, 1],
                target_cvar=np.r_[0, 1, 2],
                weights=test_weights
            )

    def test_projection_validation(self):
        """Test projection validation for coupling variables"""
        conn, ix, cortex, thalamus, a, nets = self.setup()
        
        # Setup test weights
        cortex_indices = np.where(ix == a.CORTEX)[0]
        thalamus_indices = np.where(ix == a.THALAMUS)[0]
        test_weights = conn.weights[thalamus_indices][:, cortex_indices]
        
        # Test source index out of bounds
        with pytest.raises(ValueError):
            Projection(
                source=cortex,
                target=thalamus,
                source_cvar=np.r_[999],  # Invalid index
                target_cvar=np.r_[0],
                weights=test_weights
            )
        
        # Test target index out of bounds
        with pytest.raises(ValueError):
            Projection(
                source=cortex,
                target=thalamus,
                source_cvar=np.r_[0],
                target_cvar=np.r_[999],  # Invalid index
                weights=test_weights
            )
        
        # Test array of source indices out of bounds
        with pytest.raises(ValueError):
            Projection(
                source=cortex,
                target=thalamus,
                source_cvar=np.r_[0, 999],  # Second index invalid
                target_cvar=np.r_[0, 1],
                weights=test_weights
            )
        
        # Test array of target indices out of bounds
        with pytest.raises(ValueError):
            Projection(
                source=cortex,
                target=thalamus,
                source_cvar=np.r_[0, 1],
                target_cvar=np.r_[0, 999],  # Second index invalid
                weights=test_weights
            )
        
        # Test invalid broadcasting configuration - mismatched sizes
        with pytest.raises(ValueError):
            Projection(
                source=cortex,
                target=thalamus,
                source_cvar=np.r_[0, 1],  # Size 2
                target_cvar=np.r_[0:3],  # Size 3
                weights=test_weights
            ) 