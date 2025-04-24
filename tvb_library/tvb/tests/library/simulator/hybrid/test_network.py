"""
Tests for the NetworkSet class.
"""

import numpy as np
from .test_base import BaseHybridTest


class TestNetwork(BaseHybridTest):
    """Tests for the NetworkSet class."""

    def test_networkset(self):
        """Test network coupling computation"""
        conn, ix, cortex, thalamus, a, nets = self.setup()
        
        # Create random test states and coupling variables
        x = self._randn_like_states(nets.zero_states())
        c = self._randn_like_cvars(nets.zero_cvars())  # For testing initial values
        
        # Test coupling computation
        c_new = nets.cfun(x)
        
        # Verify coupling for each projection
        for proj in nets.projections:
            src_state = getattr(x, proj.source.name)
            tgt_coupling = getattr(c_new, proj.target.name)
            
            if len(proj.source_cvar) == 1:  # Broadcasting case
                expected_val = proj.weights @ src_state[proj.source_cvar[0]] @ proj.mode_map
                for tgt_idx in proj.target_cvar:
                    np.testing.assert_allclose(tgt_coupling[tgt_idx], proj.scale * expected_val)
            
            elif len(proj.target_cvar) == 1:  # Reduction case
                expected_val = sum(proj.weights @ src_state[i] @ proj.mode_map 
                                 for i in proj.source_cvar)
                np.testing.assert_allclose(tgt_coupling[proj.target_cvar[0]], 
                                        proj.scale * expected_val)
            
            else:  # Element-wise case
                for src_idx, tgt_idx in zip(proj.source_cvar, proj.target_cvar):
                    expected_val = proj.weights @ src_state[src_idx] @ proj.mode_map
                    np.testing.assert_allclose(tgt_coupling[tgt_idx], 
                                            proj.scale * expected_val)

    def test_netset_step(self):
        """Test network time stepping"""
        conn, ix, cortex, thalamus, a, nets = self.setup()
        x = nets.zero_states()
        nx = nets.step(0, x)
        self.assert_equal(
            [(6, 37, 1), (4, 39, 3)], nx.shape
        ) 