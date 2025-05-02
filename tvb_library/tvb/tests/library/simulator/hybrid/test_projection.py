"""
Tests for the Projection class.
"""

import numpy as np
import pytest
from tvb.simulator.hybrid import Projection
from .test_base import BaseHybridTest


import scipy.sparse

class TestProjection(BaseHybridTest):
    """Tests for the Projection class."""

    def test_projection_apply(self):
        """Test Projection.apply method with default delays (instantaneous)."""
        conn, ix, cortex, thalamus, a, nets = self.setup()

        # --- Setup ---
        # Get indices and create sparse weights for test setup
        cortex_indices = np.where(ix == a.CORTEX)[0]
        thalamus_indices = np.where(ix == a.THALAMUS)[0]
        # Get subnetworks and connectivity details from setup
        cortex = nets.subnets[0]
        thalamus = nets.subnets[1]
        proj_c_t = nets.projections[0] # Use one of the projections from setup

        # Ensure weights are sparse CSR (should be handled by setup and Projection init)
        test_weights = proj_c_t.weights
        assert isinstance(test_weights, scipy.sparse.csr_matrix)

        # Define source state and history buffer parameters
        x = self._randn_like_states(nets.zero_states()) # Random initial states
        t = 15  # Arbitrary current time step (needs to be >= max_delay)
        horizon = proj_c_t.max_delay + 10 # Buffer size must be > max_delay

        n_modes_src = proj_c_t.source.model.number_of_modes
        buffer_shape = (x.cortex.shape[0], x.cortex.shape[1], n_modes_src, horizon)
        src_history_buffer = np.arange(np.prod(buffer_shape)).reshape(buffer_shape) * 0.1

        proj_multi_source = Projection(
            source=cortex,
            target=thalamus,
            source_cvar=np.r_[0, 1],    # Two source cvars
            target_cvar=np.r_[1],      # Single target cvar
            weights=proj_c_t.weights,  # Reuse weights from setup proj
            lengths=proj_c_t.lengths,  # Reuse lengths
            cv=proj_c_t.cv,            # Reuse cv
            dt=proj_c_t.dt,            # Reuse dt
            scale=1.0                  # Default scale
        )
        # Ensure mode_map is initialized if it wasn't (it should be in __init__)
        if proj_multi_source.mode_map is None:
             proj_multi_source.mode_map = np.ones(
                 (proj_multi_source.source.model.number_of_modes,
                  proj_multi_source.target.model.number_of_modes), dtype=np.int_)


        c_target_1 = nets.zero_cvars() # Target array to modify
        proj_multi_source.apply(c_target_1.thalamus, src_history_buffer, t, horizon)

        time_indices = (t - proj_multi_source.idelays + 2) % horizon
        delayed_states = src_history_buffer[proj_multi_source.source_cvar[:, None, None], proj_multi_source.weights.indices[None, :, None], :, time_indices[None, :, None]]
        weighted_delayed = proj_multi_source.weights.data[None, :, None] * delayed_states
        summed_input_nodes = np.add.reduceat(weighted_delayed, proj_multi_source.weights.indptr[:-1], axis=1)
        scaled_input = proj_multi_source.scale * summed_input_nodes
        assert np.any(c_target_1.thalamus[1] != 0), "Target cvar [1] should have received input"
        assert np.all(c_target_1.thalamus[0] == 0), "Target cvar [0] should NOT have received input"
        assert np.all(c_target_1.thalamus[2:] == 0), "Target cvars [2:] should NOT have received input"
        assert np.all(c_target_1.cortex == 0)

        proj_one_to_many = Projection(
            source=cortex,
            target=thalamus,
            source_cvar=np.r_[0],      # Single source cvar index
            target_cvar=np.r_[0, 1],   # Multiple target cvar indices
            weights=proj_c_t.weights,  # Reuse weights from setup proj
            lengths=proj_c_t.lengths,  # Reuse lengths
            cv=proj_c_t.cv,            # Reuse cv
            dt=proj_c_t.dt,            # Reuse dt
            scale=0.5                  # Use a non-default scale
        )
        # Ensure mode_map is initialized if it wasn't
        if proj_one_to_many.mode_map is None:
             proj_one_to_many.mode_map = np.ones(
                 (proj_one_to_many.source.model.number_of_modes,
                  proj_one_to_many.target.model.number_of_modes), dtype=np.int_)


        c_target_2 = nets.zero_cvars() # Target array to modify
        proj_one_to_many.apply(c_target_2.thalamus, src_history_buffer, t, horizon)

        assert np.any(c_target_2.thalamus[0] != 0), "Target cvar [0] should have received input"
        assert np.any(c_target_2.thalamus[1] != 0), "Target cvar [1] should have received input"
        assert np.all(c_target_2.thalamus[2:] == 0)
        assert np.all(c_target_2.cortex == 0)
