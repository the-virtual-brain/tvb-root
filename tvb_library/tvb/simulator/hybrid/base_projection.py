import numpy as np
from scipy import sparse as sp
import tvb.basic.neotraits.api as t

class BaseProjection(t.HasTraits):
    """Base class for projections defining coupling.

    Handles common attributes and the core logic for applying coupling,
    including time delays and sparse weights. Assumes weights and lengths
    are provided in CSR sparse format.

    Attributes
    ----------
    source_cvar : ndarray
        Array of coupling variable indices in source.
    target_cvar : ndarray
        Array of coupling variable indices in target.
    scale : float, default=1.0
        Scaling factor for the projection.
    weights : scipy.sparse.csr_matrix
        Connectivity weights matrix in CSR sparse format.
    lengths : scipy.sparse.csr_matrix, optional
        Connection lengths matrix in CSR sparse format, corresponding to non-zero weights.
        Required if delays are needed. Shape must match weights.
    cv : float, optional
        Conduction velocity. Required if lengths are provided.
    dt : float, optional
        Simulation time step. Required if lengths are provided.
    """
    source_cvar = t.NArray(dtype=np.int_)
    target_cvar = t.NArray(dtype=np.int_)
    scale: float = t.Float(default=1.0)
    weights = t.Attr(sp.csr_matrix)
    lengths = t.Attr(sp.csr_matrix)
    cv = t.Float(required=False, default=None)
    dt = t.Float(required=False, default=None)

    # Internal attributes derived in __init__
    idelays = t.NArray(dtype=np.int_, required=False, default=None)
    max_delay = t.Int(required=False, default=2) # Default minimum delay steps

    # Internal attributes for history buffer management
    _history_buffer: np.ndarray | None = None
    _horizon: int = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize buffer attributes
        self._history_buffer = None
        self._horizon = 0

        # Validation: Ensure weights and lengths (if provided) are CSR
        if not isinstance(self.weights, sp.csr_matrix):
             raise TypeError(f"Weights must be provided as a scipy.sparse.csr_matrix, got {type(self.weights)}")
        if self.lengths is not None and not isinstance(self.lengths, sp.csr_matrix):
             raise TypeError(f"Lengths must be provided as a scipy.sparse.csr_matrix, got {type(self.lengths)}")
        if self.lengths is not None and self.lengths.shape != self.weights.shape:
             raise ValueError(f"Lengths shape {self.lengths.shape} must match weights shape {self.weights.shape}")

        r, c = self.weights.shape
        eps_val = 2 * np.finfo(self.weights.dtype).eps
        eps_data = np.full(r, eps_val, dtype=self.weights.dtype)
        row_indices = np.arange(r)
        col_indices = np.zeros(r, dtype=int)
        eps_matrix = sp.csr_matrix(
            (eps_data, (row_indices, col_indices)), shape=(r, c), dtype=self.weights.dtype)
        self.weights = self.weights + eps_matrix
        # Apply epsilon to lengths only if lengths are not all zero, to maintain sparsity if L=0
        if np.any(self.lengths.data != 0):
            self.lengths = self.lengths + eps_matrix  # Add sparse matrices
        else: # If lengths are all zero, adding eps_matrix would make them non-zero, changing idelays.
              # Instead, ensure it has the same nnz as weights if it was all zero.
            if self.weights.nnz > 0 and self.lengths.nnz == 0:
                 # Create a zero-valued sparse matrix with same sparsity as weights for nnz consistency
                 self.lengths = sp.csr_matrix( (np.zeros(self.weights.nnz, dtype=self.lengths.dtype), self.weights.indices, self.weights.indptr), shape=self.weights.shape)

        assert self.weights.nnz == self.lengths.nnz, \
            f"Mismatch nnz: weights {self.weights.nnz}, lengths {self.lengths.nnz}"

        # Modify idelays calculation to match original TVB simulator (round, no +2)
        if self.cv is not None and self.dt is not None and self.lengths.nnz > 0:
            # Ensure lengths.data is not empty before division
            if self.lengths.data.size > 0:
                self.idelays = np.round(self.lengths.data / self.cv / self.dt).astype(np.int_)
            else: # This case implies lengths.nnz > 0 but lengths.data is empty, which is inconsistent for CSR.
                  # However, to be safe, if lengths.data is empty, idelays should be empty or zero.
                  # Given lengths.nnz > 0, this path should ideally not be taken if CSR is valid.
                  # If lengths.nnz > 0 but lengths.data is empty, it implies an issue with lengths matrix construction.
                  # For robustness, treat as zero delays if data is unexpectedly empty despite nnz > 0.
                  self.idelays = np.zeros(self.weights.nnz, dtype=np.int_)
        elif self.lengths.nnz > 0 : # lengths provided but cv or dt missing
            raise ValueError("cv and dt must be provided if lengths are non-zero and non-empty.")
        else: # No lengths or all lengths are zero (lengths.nnz == 0)
            self.idelays = np.zeros(self.weights.nnz, dtype=np.int_)

        # now that the sparse matrices are formed, replace the 2*eps values by zeros
        self.weights.data[self.weights.data == eps_val] = 0

        self.max_delay = np.max(self.idelays) if self.idelays.size > 0 else 0
        # Ensure cvars are arrays
        self.source_cvar = np.atleast_1d(self.source_cvar)
        self.target_cvar = np.atleast_1d(self.target_cvar)

    def configure_buffer(self, n_vars_src: int, n_nodes_src: int, n_modes_src: int):
        """Configure and initialize the internal history buffer."""
        # Horizon matches original TVB n_time = max_delay + 1
        # (or at least 1 if max_delay is 0, to store current state)
        self._horizon = self.max_delay + 1
        if self._horizon == 0: # Should not happen if max_delay is at least 0, but defensive
            self._horizon = 1

        buffer_shape = (n_vars_src, n_nodes_src, n_modes_src, self._horizon)
        self._history_buffer = np.zeros(buffer_shape, 'f')

    def update_buffer(self, current_src_state: np.ndarray, t: int):
        """Update the history buffer with the current source state."""
        buffer_idx = t % self._horizon
        state_to_write = current_src_state[self.source_cvar]
        # print('dbg(update_buffer):', buffer_idx, state_to_write.ravel())
        self._history_buffer[..., buffer_idx] = state_to_write

    def apply(self, tgt: np.ndarray, t: int, mode_map: np.ndarray):
        """Apply the projection to compute coupling using its internal history buffer.

        Handles explicitly delayed connections via a unified mechanism.
        Assumes minimal delay (2 steps) if lengths/cv/dt are not provided.
        Parameters
        ----------
        tgt : ndarray
            Target state array to modify (shape [n_vars_tgt, n_nodes_tgt, n_modes_tgt]).
        t : int
            Current time step index relative to the start of the simulation.
        mode_map : ndarray
             Mode mapping matrix (source_modes x target_modes).

        """

        # Calculate time indices into the circular history buffer
        # Shape: (nnz,) where nnz is number of non-zero weights/lengths
        # Here, t is the 0-indexed current step.
        time_indices = (t - self.idelays + self._horizon) % self._horizon

        # Gather delayed states from the internal history buffer
        # Indexing: [source_cvar_indices, node_indices, mode_indices, time_indices]
        # source_cvar needs shape (n_source_cvar, 1) for broadcasting
        # weights.indices gives the source node indices for each non-zero weight
        # We need all modes from the source, hence ':' for mode dimension
        delayed_states = self._history_buffer[
            self.source_cvar[:, np.newaxis], # Shape (n_src_cvar, 1)
            self.weights.indices,            # Shape (nnz,) -> Source node indices
            :,                               # All source modes
            time_indices                     # Shape (nnz,) -> Time index for each connection
        ]
        # Result shape: (n_src_cvar, nnz, n_src_modes)
        # print('dbg(apply delayed_states):', t, delayed_states.ravel())
        # if t == 2:
        #     foo = np.c_[self.weights.indices, self.idelays, time_indices, self.weights.data]
        #     # import pdb; pdb.set_trace()

        # Apply weights element-wise
        # weights.data has shape (nnz,)
        # Need to reshape weights.data for broadcasting: (1, nnz, 1)
        weighted_delayed = self.weights.data[np.newaxis, :, np.newaxis] * delayed_states
        # Result shape: (n_src_cvar, nnz, n_src_modes)

        # Sum inputs per target node using reduceat
        # self.weights.indptr defines the start/end points for summation over nnz axis
        summed_input = np.add.reduceat(
            weighted_delayed,
            self.weights.indptr[:-1],
            axis=1 # Sum along the nnz dimension
        )
        # Result shape: (n_src_cvar, n_target_nodes, n_src_modes)
        # Note: n_target_nodes comes from the structure of the CSR matrix (number of rows)

        # Apply scaling factor
        scaled_input = self.scale * summed_input
        # Result shape: (n_src_cvar, n_target_nodes, n_src_modes)

        # Apply mode mapping
        # mode_map shape: (n_src_modes, n_target_modes)
        # Need matrix multiplication: (n_src_cvar, n_target_nodes, n_src_modes) @ (n_src_modes, n_target_modes)
        # Result shape: (n_src_cvar, n_target_nodes, n_target_modes)
        aff = scaled_input @ mode_map

        # Apply to target coupling variables
        # tgt shape: (n_target_vars, n_target_nodes, n_target_modes)
        # target_cvar shape: (n_target_cvar,)
        if self.target_cvar.size == 1:
            # If only one target cvar, sum contributions from all source cvars
            # Sum along axis 0 (source_cvar dimension)
            # Result shape: (n_target_nodes, n_target_modes)
            summed_aff = aff.sum(axis=0)
            # Add to the single target cvar across all nodes/modes
            tgt[self.target_cvar[0], :, :] += summed_aff
        elif self.source_cvar.size == 1:
            # If only one source cvar, apply its contribution to all target cvars directly
            # aff has shape (1, n_target_nodes, n_target_modes)
            # Squeeze axis 0 to get shape (n_target_nodes, n_target_modes)
            squeezed_aff = aff.squeeze(axis=0)
            # Add to all target cvars
            tgt[self.target_cvar, :, :] += squeezed_aff
        elif self.source_cvar.size == self.target_cvar.size:
            # If source and target cvars match in number, apply element-wise mapping
            # aff shape: (n_cvar, n_target_nodes, n_target_modes)
            # tgt[target_cvar] shape: (n_cvar, n_target_nodes, n_target_modes)
            tgt[self.target_cvar, :, :] += aff
        else:
            # Ambiguous case: M source cvars to N target cvars (M != N, M!=1, N!=1)
            # Raise error or define specific reduction/broadcasting rule
            raise ValueError(f"Unsupported projection: {self.source_cvar.size} source cvars "
                             f"to {self.target_cvar.size} target cvars. "
                             f"Requires M=1, N=1, or M=N.")
