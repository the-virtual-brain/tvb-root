"""
Tests for the NetworkSet class.
"""

import numpy as np
from .test_base import BaseHybridTest


class TestNetwork(BaseHybridTest):
    """Tests for the NetworkSet class."""

    def test_networkset(self):
        """Test network coupling computation"""
        # Setup with zero lengths for projections to ensure minimal delay (idelays=2)
        # This makes assertions simpler as 'delayed' state is the current state.
        conn, ix, cortex, thalamus, a, nets = self.setup()
        
        # Create random test states and coupling variables
        x = self._randn_like_states(nets.zero_states())
        c = self._randn_like_cvars(nets.zero_cvars())  # For testing initial values

        # --- Pre-populate history buffers with known "past" states ---
        # Store mock past states: mock_past_source_states[projection_index][past_step_index_in_buffer]
        mock_past_source_states_for_projections = [{} for _ in nets.projections]

        for proj_idx, proj in enumerate(nets.projections):
            # For each projection, fill its history buffer with distinct "past" states.
            # These steps are relative to the buffer's circular indexing.
            for s_buffer_idx in range(1, proj._horizon): # Iterate through all "past" buffer slots
                # Create a distinct random state for this projection's source type at this "past" slot
                past_state_shape = (
                    proj.source.model.nvar,
                    proj.source.nnodes,
                    proj.source.model.number_of_modes
                )
                mock_state_for_past_slot = np.random.randn(*past_state_shape)
                mock_past_source_states_for_projections[proj_idx][s_buffer_idx] = mock_state_for_past_slot
                proj.update_buffer(mock_state_for_past_slot, s_buffer_idx) # Use s_buffer_idx as step to fill specific slot

        # Test coupling computation, using current_step_idx=0
        current_step_idx = 0
        # NetworkSet.cfun will call proj.update_buffer(src_state_current, current_step_idx)
        # then proj.apply(tgt, current_step_idx)
        c_new = nets.cfun(current_step_idx, x)
        
        # Verify coupling for each projection
        for proj_idx, proj in enumerate(nets.projections):
            src_state_current = getattr(x, proj.source.name) # Current state of source subnetwork
            tgt_coupling_actual = getattr(c_new, proj.target.name) # Actual coupling calculated by cfun

            # --- Replicate BaseProjection.apply logic to calculate expected coupling ---
            # Calculate time_indices based on delays, specific to this projection
            time_indices_for_proj_connections = (current_step_idx - proj.idelays + proj._horizon) % proj._horizon

            # Construct the 'delayed_states' array that BaseProjection.apply would gather.
            # Shape: (n_source_cvar_proj, nnz_proj_weights, n_source_modes)
            delayed_input_values = np.empty((
                proj.source_cvar.size, 
                proj.weights.data.size, 
                proj.source.model.number_of_modes 
            ))

            mock_history_for_this_proj = mock_past_source_states_for_projections[proj_idx]

            for i_scv_proj, scv_model_idx in enumerate(proj.source_cvar):
                for k_conn in range(proj.weights.data.size): 
                    src_node_for_conn = proj.weights.indices[k_conn] 
                    buffer_idx_for_conn = time_indices_for_proj_connections[k_conn]

                    if buffer_idx_for_conn == (current_step_idx % proj._horizon): # Should be 0
                        # Value comes from the current state 'x' which was put into buffer at current_step_idx
                        delayed_input_values[i_scv_proj, k_conn, :] = \
                            src_state_current[scv_model_idx, src_node_for_conn, :]
                    else:
                        # Value comes from the mock past states we pre-filled into the buffer
                        # The mock_history_for_this_proj stores the full state array for the source subnetwork
                        # for that buffer_idx_for_conn.
                        past_source_state_array = mock_history_for_this_proj[buffer_idx_for_conn]
                        delayed_input_values[i_scv_proj, k_conn, :] = \
                            past_source_state_array[scv_model_idx, src_node_for_conn, :]
            
            # Apply weights (element-wise to the gathered delayed_input_values)
            # proj.weights.data has shape (nnz,)
            weighted_delayed_expected = proj.weights.data[np.newaxis, :, np.newaxis] * delayed_input_values
            
            # Sum inputs per target node using reduceat (CSR format specific)
            # summed_input_expected shape: (n_source_cvar_proj, n_target_nodes, n_source_modes)
            summed_input_expected = np.add.reduceat(
                weighted_delayed_expected,
                proj.weights.indptr[:-1], # Defines segments for summation over axis 1
                axis=1 # Sum along the nnz/connections dimension
            )
            
            # Apply scaling factor
            scaled_input_expected = proj.scale * summed_input_expected
            
            # Apply mode mapping
            # aff_expected shape: (n_source_cvar_proj, n_target_nodes, n_target_modes)
            aff_expected = scaled_input_expected @ proj.mode_map
            # --- End of replicating BaseProjection.apply logic ---

            # Now, compare tgt_coupling_actual with aff_expected based on cvar mapping rules
            # tgt_coupling_actual has shape (n_target_model_cvars, n_target_nodes, n_target_modes)
            
            if proj.target_cvar.size == 1:  # M source cvars to 1 target cvar (Reduction)
                # Expected value is sum over source_cvar dimension of aff_expected
                expected_val_for_single_target_cvar = aff_expected.sum(axis=0)
                np.testing.assert_allclose(
                    tgt_coupling_actual[proj.target_cvar[0], :, :],
                    expected_val_for_single_target_cvar,
                    rtol=1e-5, atol=1e-8 
                )
            elif proj.source_cvar.size == 1: # 1 source cvar to N target cvars (Broadcasting)
                # aff_expected has shape (1, n_target_nodes, n_target_modes)
                # Squeeze it for comparison with each of the N target cvars
                squeezed_aff_expected = aff_expected.squeeze(axis=0)
                for tcv_model_idx in proj.target_cvar:
                    np.testing.assert_allclose(
                        tgt_coupling_actual[tcv_model_idx, :, :],
                        squeezed_aff_expected,
                        rtol=1e-5, atol=1e-8
                    )
            elif proj.source_cvar.size == proj.target_cvar.size: # M source cvars to M target cvars (Element-wise)
                # Compare each corresponding source_cvar contribution in aff_expected
                # to the relevant target_cvar slice in tgt_coupling_actual.
                for i_map_idx in range(proj.source_cvar.size):
                    tcv_model_idx = proj.target_cvar[i_map_idx]
                    np.testing.assert_allclose(
                        tgt_coupling_actual[tcv_model_idx, :, :],
                        aff_expected[i_map_idx, :, :], # aff_expected is indexed by projection's source_cvar order
                        rtol=1e-5, atol=1e-8
                    )
            else:
                # This case should be caught by BaseProjection.apply's ValueError
                raise NotImplementedError(
                    f"Test logic for projection cvar mapping "
                    f"({proj.source_cvar.size} to {proj.target_cvar.size}) not implemented."
                )

    def test_netset_step(self):
        """Test network time stepping"""
        conn, ix, cortex, thalamus, a, nets = self.setup()
        x = nets.zero_states()
        nx = nets.step(0, x)
        self.assert_equal(
            [(6, 37, 1), (4, 39, 3)], nx.shape
        ) 
