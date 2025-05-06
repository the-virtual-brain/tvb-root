"""
Tests for the InterProjection class.
"""

import numpy as np
import pytest
# Import the specific projection class being tested from its new file
from tvb.simulator.hybrid.inter_projection import InterProjection
from .test_base import BaseHybridTest


import scipy.sparse

class TestInterProjection(BaseHybridTest): # Keep class name as it tests InterProjection
    """Tests for the InterProjection class."""

    def test_interprojection_apply(self):
        """Test InterProjection.apply method with delays."""
        _, _, cortex_subn, thalamus_subn, _, nets = self.setup()

        # --- Setup ---
        # Get subnetworks and connectivity details from setup
        # Assuming nets.projections[0] is an InterProjection instance after setup
        proj_c_t: InterProjection = nets.projections[0]

        # Ensure weights are sparse CSR (handled by BaseProjection init)
        assert isinstance(proj_c_t.weights, scipy.sparse.csr_matrix)
        # Ensure lengths are sparse CSR if they exist (handled by BaseProjection init)
        if proj_c_t.lengths is not None:
            assert isinstance(proj_c_t.lengths, scipy.sparse.csr_matrix)

        # Define source state and history buffer parameters
        # Use a time step >= max_delay to ensure history buffer indexing is valid
        t = proj_c_t.max_delay + 5
        horizon = proj_c_t.max_delay + 10 # Buffer size must accommodate max_delay

        # Create a realistic history buffer
        # Shape: (n_vars_src, n_nodes_src, n_modes_src, horizon)
        n_vars_src = proj_c_t.source.model.nvar
        n_nodes_src = proj_c_t.source.nnodes
        n_modes_src = proj_c_t.source.model.number_of_modes

        # Create a current source state array instead of a full history buffer
        current_src_state_shape = (n_vars_src, n_nodes_src, n_modes_src)
        current_src_state = np.random.randn(*current_src_state_shape) * 0.1


        # --- Test Case 1: Multiple source cvars to single target cvar ---
        proj_multi_source = InterProjection(
            source=cortex_subn,
            target=thalamus_subn,
            source_cvar=np.r_[0, 1],    # Two source cvars
            target_cvar=np.r_[1],      # Single target cvar
            weights=proj_c_t.weights,  # Reuse weights from setup proj
            lengths=proj_c_t.lengths,  # Reuse lengths
            cv=proj_c_t.cv,            # Reuse cv
            dt=proj_c_t.dt,            # Reuse dt
            scale=1.0
        )
        # Configure the projection's internal buffer
        proj_multi_source.configure_buffer(n_vars_src, n_nodes_src, n_modes_src)
        # Update the buffer with the current state before applying
        # (Simulate multiple updates for a more realistic buffer state)
        for i in range(t + 1):
             # Use slightly different states for each step for testing
             proj_multi_source.update_buffer(current_src_state + i*0.01, i)


        # Target array to modify (using thalamus part of zero_cvars)
        # Shape: (n_vars_tgt, n_nodes_tgt, n_modes_tgt)
        c_target_1 = nets.zero_cvars()
        target_thalamus_1 = c_target_1.thalamus # Get the relevant part

        # Apply the projection (uses internal buffer now)
        proj_multi_source.apply(target_thalamus_1, t)

        # Assertions based on target_cvar=[1]
        assert np.any(target_thalamus_1[1, :, :] != 0), "Target cvar [1] should have received input"
        assert np.all(target_thalamus_1[0, :, :] == 0), "Target cvar [0] should NOT have received input"
        # Check remaining cvars if they exist
        if target_thalamus_1.shape[0] > 2:
             assert np.all(target_thalamus_1[2:, :, :] == 0), "Target cvars [2:] should NOT have received input"
        # Ensure other subnetworks weren't touched
        assert np.all(c_target_1.cortex == 0)


        # --- Test Case 2: Single source cvar to multiple target cvars ---
        proj_one_to_many = InterProjection(
            source=cortex_subn,
            target=thalamus_subn,
            source_cvar=np.r_[0],      # Single source cvar index
            target_cvar=np.r_[0, 1],   # Multiple target cvar indices
            weights=proj_c_t.weights,  # Reuse weights from setup proj
            lengths=proj_c_t.lengths,  # Reuse lengths
            cv=proj_c_t.cv,            # Reuse cv
            dt=proj_c_t.dt,            # Reuse dt
            scale=0.5
        )
        # Configure and update buffer for the second test projection
        proj_one_to_many.configure_buffer(n_vars_src, n_nodes_src, n_modes_src)
        for i in range(t + 1):
             proj_one_to_many.update_buffer(current_src_state + i*0.01, i)


        # Target array to modify
        c_target_2 = nets.zero_cvars()
        target_thalamus_2 = c_target_2.thalamus

        # Apply the projection (uses internal buffer now)
        proj_one_to_many.apply(target_thalamus_2, t)

        # Assertions based on target_cvar=[0, 1]
        assert np.any(target_thalamus_2[0, :, :] != 0), "Target cvar [0] should have received input"
        assert np.any(target_thalamus_2[1, :, :] != 0), "Target cvar [1] should have received input"
        # Check remaining cvars if they exist
        if target_thalamus_2.shape[0] > 2:
            assert np.all(target_thalamus_2[2:, :, :] == 0), "Target cvars [2:] should NOT have received input"
        # Ensure other subnetworks weren't touched
        assert np.all(c_target_2.cortex == 0)

    # TODO: Add tests for InternalProjection if needed, likely requiring
    #       instantiation within a Subnetwork context to test its apply method.
    #       Also need tests for edge cases like zero weights, zero lengths, etc.
