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
Tests for :class:`~tvb.simulator.hybrid.NetworkSet`.

The primary test (``test_networkset``) validates the coupling computation
produced by :meth:`~tvb.simulator.hybrid.NetworkSet.cfun` against a manually
replicated implementation of :meth:`~tvb.simulator.hybrid.BaseProjection.apply`.
The approach is:

1. Pre-populate each projection's history buffer with known random states.
2. Call :meth:`~tvb.simulator.hybrid.NetworkSet.cfun`.
3. Independently reconstruct the expected coupling by following the same
   CSR-based gather → weight → reduce → scale → mode-map pipeline.
4. Assert element-wise equality for every supported cvar-mapping topology
   (many-to-one reduction, one-to-many broadcast, element-wise M→M).
"""

import numpy as np
from .test_base import BaseHybridTest


class TestNetwork(BaseHybridTest):
    """Unit and integration tests for :class:`~tvb.simulator.hybrid.NetworkSet`.

    ``test_networkset`` is the primary white-box test: it manually replicates
    the logic of :meth:`~tvb.simulator.hybrid.BaseProjection.apply` and
    asserts that :meth:`~tvb.simulator.hybrid.NetworkSet.cfun` returns the
    same values.  This makes the test sensitive to changes in the
    buffer-indexing formula and the CSR reduction scheme.
    """

    def test_networkset(self):
        """Validate cfun() output by manually replicating BaseProjection.apply().

        Procedure
        ---------
        1. Pre-populate every slot of each projection's history buffer with
           distinct random states via ``update_buffer()``.
        2. Call :meth:`~tvb.simulator.hybrid.NetworkSet.cfun` at ``t=0``.
        3. For each projection, independently reconstruct the expected coupling:
           gather delayed states using the ``(t-1-idelays+horizon) % horizon``
           index formula, apply CSR weights, reduce per target node, scale,
           and mode-map.
        4. Compare actual vs expected with ``np.testing.assert_allclose``
           for all supported cvar-mapping topologies (many-to-one, one-to-many,
           element-wise M→M).
        """
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
            for s_buffer_idx in range(proj._horizon): # Fill ALL buffer slots
                # Create a distinct random state for this projection's source type at this "past" slot
                past_state_shape = (
                    proj.source.model.nvar,
                    proj.source.nnodes,
                    proj.source.model.number_of_modes
                )
                mock_state_for_slot = np.random.randn(*past_state_shape)
                mock_past_source_states_for_projections[proj_idx][s_buffer_idx] = mock_state_for_slot
                proj.update_buffer(mock_state_for_slot, s_buffer_idx)

        # Test coupling computation, using current_step_idx=0
        # cfun only calls apply (buffer updates happen post-integration in step())
        current_step_idx = 0
        c_new = nets.cfun(current_step_idx, x)
        
        # Verify coupling for each projection
        for proj_idx, proj in enumerate(nets.projections):
            src_state_current = getattr(x, proj.source.name) # Current state of source subnetwork
            tgt_coupling_actual = getattr(c_new, proj.target.name) # Actual coupling calculated by cfun

            # --- Replicate BaseProjection.apply logic to calculate expected coupling ---
            # Calculate time_indices using t-1 formula matching updated apply() method
            time_indices_for_proj_connections = (current_step_idx - 1 - proj.idelays + proj._horizon) % proj._horizon

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

                    # Read from the mock state pre-filled into the buffer at this slot
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
        """Smoke test for a single :meth:`~tvb.simulator.hybrid.NetworkSet.step` call.

        Asserts that the returned state object has per-subnetwork shapes
        matching the expected ``(nvar, nnodes, modes)`` tuples
        ``[(6, 37, 1), (4, 39, 3)]``.
        """
        conn, ix, cortex, thalamus, a, nets = self.setup()
        x = nets.zero_states()
        nx = nets.step(0, x)
        self.assert_equal(
            [(6, 37, 1), (4, 39, 3)], nx.shape
        ) 
