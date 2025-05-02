import numpy as np
from scipy import sparse as sp
import tvb.basic.neotraits.api as t
from .subnetwork import Subnetwork


class Projection(t.HasTraits):
    """A projection from one subnetwork to another.
    
    A projection defines how one subnetwork influences another through
    coupling variables and connectivity weights.
    
    Attributes
    ----------
    source : Subnetwork
        Source subnetwork
    target : Subnetwork
        Target subnetwork
    source_cvar : ndarray
        Array of coupling variable indices in source.
    target_cvar : ndarray
        Array of coupling variable indices in target.
    scale : float, default=1.0
        Scaling factor for the projection.
    weights : scipy.sparse.csr_matrix or ndarray
        Connectivity weights matrix. Will be converted to CSR sparse format.
    lengths : ndarray, optional
        Connection lengths matrix/array corresponding to non-zero weights.
        Required if delays are needed. Shape should match weights.data.
    cv : float, optional
        Conduction velocity. Required if lengths are provided.
    dt : float, optional
        Simulation time step. Required if lengths are provided.
    mode_map : ndarray, optional
        Mapping between source and target modes. Defaults to uniform mapping.
        NOTE: Currently unused in the delayed coupling calculation.
    """

    source: Subnetwork = t.Attr(Subnetwork)
    target: Subnetwork = t.Attr(Subnetwork)
    source_cvar = t.NArray(dtype=np.int_)
    target_cvar = t.NArray(dtype=np.int_)
    scale: float = t.Float(default=1.0)
    weights = t.Attr(sp.csr_matrix)
    lengths = t.Attr(sp.csr_matrix, required=False, default=None) # Changed type to csr_matrix
    cv = t.Float(required=False, default=None)
    dt = t.Float(required=False, default=None)
    mode_map = t.NArray(dtype=np.int_, required=False, default=None) # Expects int

    # Internal attributes, not meant for direct setting
    idelays = t.NArray(dtype=np.int_, required=False)
    max_delay = t.Int(required=False)

    def __init__(self, **kwargs):
        # Convert weights to CSR if needed before super().__init__ validation
        if 'weights' in kwargs and not isinstance(kwargs['weights'], sp.csr_matrix):
            try:
                kwargs['weights'] = sp.csr_matrix(kwargs['weights'])
            except Exception as e:
                raise ValueError(f"Failed to convert weights to csr_matrix: {e}")
        super().__init__(**kwargs)
        r, c = self.weights.shape
        # Add an epsilon column vector to the first column to ensure it's non-empty.
        # This can be important for certain sparse operations or assumptions.
        eps_val = 2 * np.finfo(self.weights.dtype).eps
        # Create data and indices for the first column vector
        eps_data = np.full(r, eps_val, dtype=self.weights.dtype)
        row_indices = np.arange(r)
        col_indices = np.zeros(r, dtype=int)
        # Create the sparse column matrix
        eps_matrix = sp.csr_matrix((eps_data, (row_indices, col_indices)), shape=(r, c), dtype=self.weights.dtype)
        self.weights = self.weights + eps_matrix
        self.lengths = self.lengths + eps_matrix
        # Ensure indptr remains valid (non-decreasing) after potential addition.
        # Note: Adding a full column might significantly change indptr, but it should still be non-decreasing.
        assert np.all(np.diff(self.weights.indptr) >= 0), "CSR indptr must be non-decreasing"
        self.idelays = (self.lengths.data / self.cv / self.dt).astype(np.int_) + 2
        self.max_delay = np.max(self.idelays) if self.idelays.size > 0 else 2
        if self.mode_map is None:
            self.mode_map = np.ones(
                (self.source.model.number_of_modes,
                 self.target.model.number_of_modes), dtype=np.int_)
        self.source_cvar = np.atleast_1d(self.source_cvar)
        self.target_cvar = np.atleast_1d(self.target_cvar)

    def apply(self, tgt, src_history_buffer, t, horizon):
        """Apply the projection to compute coupling using history buffer.

        Handles both instantaneous (approximated by minimum delay) and
        explicitly delayed connections via a unified mechanism.

        Parameters
        ----------
        tgt : ndarray
            Target state array to modify (shape [n_vars, n_nodes]).
        src_history_buffer : ndarray
            Source history buffer (shape [n_vars, n_nodes, horizon]).
        t : int
            Current time step index.
        horizon : int
            Size of the history buffer time dimension.

        """
        # TODO self.src_history_buffer[...,t] = src[self.source_cvar]
        # Shape: (nnz,)
        time_indices = (t - self.idelays + 2) % horizon
        delayed_states = src_history_buffer[self.source_cvar.reshape(
            -1, 1), self.weights.indices, :, time_indices]
        delayed_states = delayed_states.reshape(
            self.source_cvar.size, time_indices.size, -1)
        weighted_delayed = self.weights.data.reshape(-1, 1) * delayed_states
        summed_input = np.add.reduceat(
            weighted_delayed, self.weights.indptr[:-1], axis=1)
        assert summed_input.shape == (
            self.source_cvar.size, tgt.shape[1], self.mode_map.shape[0])
        scaled_input = self.scale * summed_input
        aff = scaled_input @ self.mode_map
        if self.target_cvar.size == 1:
            aff = aff.sum(axis=0)
        tgt[self.target_cvar, :] += aff
