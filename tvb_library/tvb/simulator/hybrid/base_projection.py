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
Base projection infrastructure for the hybrid simulator.

A projection couples state variables from a source set of nodes to coupling
variables of a target set of nodes, with optional conduction-velocity delays
and a coupling function.  The core mechanism is a circular history buffer
that stores source states and is indexed with per-connection delay offsets.

Mathematical Framework
-----------------------
At each simulation step *t*, the delayed afferent signal contributed by
source node *j* to target node *i* through coupling variable *k* is::

    x_j^k(t - tau_ij)    where tau_ij = round(L_ij / cv / dt)

The total weighted coupling at target node *i* is:

.. math::

    C_i = s \\cdot \\mathrm{post}\\!\\left(
          \\sum_{j} w_{ij}\\,\\mathrm{pre}\\!\\bigl(x_j(t - \\tau_{ij}),
          x_i(t)\\bigr)
          \\right)

where *s* is ``scale``, *w_ij* are the sparse weights, and ``pre`` /
``post`` are optional coupling transformations from the attached
``cfun``.  When ``pre`` is a two-argument function (e.g.  ``Difference``
or ``Kuramoto``), the current target state *x_i* at the coupling
variables is provided as the second argument so that expressions like
``x_j - x_i`` or ``sin(x_j - x_i)`` can be evaluated accurately.

Delays are resolved once at construction time from the sparse ``lengths``
matrix, the conduction velocity ``cv``, and the time step ``dt``.  A small
epsilon is temporarily added to the weight and length matrices to guarantee
a uniform CSR sparsity structure (required for ``np.add.reduceat``), then
removed once the structure is established.
"""

import inspect

import numpy as np
from scipy import sparse as sp
import tvb.basic.neotraits.api as t


class BaseProjection(t.HasTraits):
    """Base class for delayed sparse coupling between node sets.

    Encapsulates the core coupling logic shared by both intra- and
    inter-subnetwork projections: CSR-format weight/length validation,
    integer delay computation, a circular history buffer for delayed state
    access, and the weighted-sum-plus-coupling-function pipeline.

    Subclasses supply a ``mode_map`` matrix and optionally store references
    to source/target ``Subnetwork`` objects; this class only manages the
    buffer and the numerical computation.

    Attributes
    ----------
    source_cvar : ndarray of int
        Indices of the coupling variable(s) read from the source state.
    target_cvar : ndarray of int
        Indices of the coupling variable(s) written into the target
        coupling array.
    scale : float, default 1.0
        Global scaling factor applied after the weighted sum.
    weights : scipy.sparse.csr_matrix
        Connectivity weight matrix.  Rows correspond to target nodes,
        columns to source nodes.
    lengths : scipy.sparse.csr_matrix
        Connection-length matrix with the same sparsity structure as
        ``weights``.  Required when delays are desired.
    cv : float, optional
        Conduction velocity (same units as ``lengths``).  Must be
        provided together with ``dt`` when ``lengths`` is non-zero.
    dt : float, optional
        Integration time step.  Must be provided together with ``cv``
        when ``lengths`` is non-zero.
    cfun : Coupling, optional
        Coupling function whose ``pre()`` and ``post()`` methods bracket
        the weighted sum.  When *None* the identity transformation is
        used.

    Notes
    -----
    **Epsilon trick for CSR consistency** — To guarantee that every row
    of the weight matrix has at least one explicit entry (required so
    that ``np.add.reduceat`` with ``weights.indptr[:-1]`` covers all
    target nodes), ``2 * eps`` is temporarily added to column 0 of every
    row.  The matching structural entry is added to ``lengths`` to keep
    their sparsity patterns identical.  Once the sparsity structure is
    established and ``idelays`` has been computed, these epsilon values
    in ``weights.data`` are zeroed, leaving structural zeros in place.

    See Also
    --------
    IntraProjection : Intra-subnetwork projection (identity mode map).
    InterProjection : Inter-subnetwork projection (explicit mode map).
    """

    source_cvar = t.NArray(dtype=np.int_)
    target_cvar = t.NArray(dtype=np.int_)
    scale: float = t.Float(default=1.0)
    target_scales = t.NArray(
        required=False,
        label="Per-target-cvar scaling",
        doc="""Optional per-target-cvar scale factors.  When set, element *i*
        multiplies the contribution delivered to ``target_cvar[i]`` after mode
        mapping, on top of the global ``scale``.  Must have the same length as
        ``target_cvar``.  When *None* the global ``scale`` alone is used.""",
    )
    weights = t.Attr(sp.csr_matrix)
    lengths = t.Attr(sp.csr_matrix)
    cv = t.Float(required=False, default=None)
    dt = t.Float(required=False, default=None)
    cfun = t.Attr(object, required=False, default=None)

    # Internal attributes derived in __init__
    idelays = t.NArray(dtype=np.int_, required=False, default=None)
    max_delay = t.Int(required=False, default=2)  # Default minimum delay steps

    # Internal attributes for history buffer management
    _history_buffer: np.ndarray | None = None
    _horizon: int = 0

    def __init__(self, **kwargs):
        """Validate inputs, compute integer delays, and prime the sparsity structure.

        Checks that ``weights`` (and ``lengths``, if given) are CSR matrices
        with matching shapes, then applies the epsilon trick to guarantee
        uniform row coverage, harmonises the sparsity patterns, computes
        ``idelays`` from ``lengths / cv / dt``, and finally zeroes the
        epsilon sentinel values in ``weights.data``.

        Parameters
        ----------
        **kwargs
            Trait values forwarded to ``HasTraits.__init__``.  Expected keys
            include ``weights``, ``lengths``, ``source_cvar``,
            ``target_cvar``, and optionally ``scale``, ``cv``, ``dt``,
            ``cfun``.

        Raises
        ------
        TypeError
            If ``weights`` or ``lengths`` is not a
            ``scipy.sparse.csr_matrix``.
        ValueError
            If ``lengths.shape != weights.shape``, or if ``lengths``
            contains non-zero entries but ``cv`` or ``dt`` is not
            provided.
        """
        super().__init__(**kwargs)
        # Initialize buffer attributes
        self._history_buffer = None
        self._horizon = 0

        # Validation: Ensure weights and lengths (if provided) are CSR
        if not isinstance(self.weights, sp.csr_matrix):
            raise TypeError(
                f"Weights must be provided as a scipy.sparse.csr_matrix, got {type(self.weights)}"
            )
        if self.lengths is not None and not isinstance(self.lengths, sp.csr_matrix):
            raise TypeError(
                f"Lengths must be provided as a scipy.sparse.csr_matrix, got {type(self.lengths)}"
            )
        if self.lengths is not None and self.lengths.shape != self.weights.shape:
            raise ValueError(
                f"Lengths shape {self.lengths.shape} must match weights shape {self.weights.shape}"
            )

        # If lengths is None, default to zero-lengths matrix (no conduction delays)
        if self.lengths is None:
            self.lengths = sp.csr_matrix(self.weights.shape, dtype=np.float64)

        r, c = self.weights.shape
        eps_val = 2 * np.finfo(self.weights.dtype).eps
        eps_data = np.full(r, eps_val, dtype=self.weights.dtype)
        row_indices = np.arange(r)
        col_indices = np.zeros(r, dtype=int)
        eps_matrix = sp.csr_matrix(
            (eps_data, (row_indices, col_indices)),
            shape=(r, c),
            dtype=self.weights.dtype,
        )
        self.weights = self.weights + eps_matrix
        # Apply epsilon to lengths only if lengths are not all zero, to maintain sparsity if L=0
        if np.any(self.lengths.data != 0):
            self.lengths = self.lengths + eps_matrix  # Add sparse matrices
        else:  # If lengths are all zero, adding eps_matrix would make them non-zero, changing idelays.
            # Instead, ensure it has the same nnz as weights if it was all zero.
            if self.weights.nnz > 0 and self.lengths.nnz == 0:
                # Create a zero-valued sparse matrix with same sparsity as weights for nnz consistency
                self.lengths = sp.csr_matrix(
                    (
                        np.zeros(self.weights.nnz, dtype=self.lengths.dtype),
                        self.weights.indices,
                        self.weights.indptr,
                    ),
                    shape=self.weights.shape,
                )

        # Harmonize sparsity patterns: reconstruct lengths to match weights' pattern
        if self.weights.nnz != self.lengths.nnz:
            w_coo = self.weights.tocoo()
            lengths_full = self.lengths.toarray()
            new_data = lengths_full[w_coo.row, w_coo.col].astype(self.lengths.dtype)
            self.lengths = sp.csr_matrix(
                (new_data, (w_coo.row, w_coo.col)),
                shape=self.weights.shape,
            )

        if self.weights.nnz != self.lengths.nnz:
            raise ValueError(
                f"Mismatch nnz: weights {self.weights.nnz}, lengths {self.lengths.nnz}"
            )

        # Modify idelays calculation to match original TVB simulator (round, no +2)
        if self.cv is not None and self.dt is not None and self.lengths.nnz > 0:
            # Ensure lengths.data is not empty before division
            if self.lengths.data.size > 0:
                self.idelays = np.round(self.lengths.data / self.cv / self.dt).astype(
                    np.int_
                )
            else:  # This case implies lengths.nnz > 0 but lengths.data is empty, which is inconsistent for CSR.
                # However, to be safe, if lengths.data is empty, idelays should be empty or zero.
                # Given lengths.nnz > 0, this path should ideally not be taken if CSR is valid.
                # If lengths.nnz > 0 but lengths.data is empty, it implies an issue with lengths matrix construction.
                # For robustness, treat as zero delays if data is unexpectedly empty despite nnz > 0.
                self.idelays = np.zeros(self.weights.nnz, dtype=np.int_)
        elif self.lengths.nnz > 0:  # lengths provided but cv or dt missing
            raise ValueError(
                "cv and dt must be provided if lengths are non-zero and non-empty."
            )
        else:  # No lengths or all lengths are zero (lengths.nnz == 0)
            self.idelays = np.zeros(self.weights.nnz, dtype=np.int_)

        # now that the sparse matrices are formed, replace the 2*eps values by zeros
        self.weights.data[self.weights.data == eps_val] = 0

        self.max_delay = np.max(self.idelays) if self.idelays.size > 0 else 0
        # Ensure cvars are arrays
        self.source_cvar = np.atleast_1d(self.source_cvar)
        self.target_cvar = np.atleast_1d(self.target_cvar)

    def configure_buffer(self, n_vars_src: int, n_nodes_src: int, n_modes_src: int):
        """Allocate and zero-initialise the circular history buffer.

        The buffer shape is
        ``(n_vars_src, n_nodes_src, n_modes_src, horizon)`` where
        ``horizon = max_delay + 1``.  A horizon of at least 1 is
        guaranteed so that even a zero-delay projection can store the
        previous step's state.

        Parameters
        ----------
        n_vars_src : int
            Number of state variables in the source (``model.nvar``).
        n_nodes_src : int
            Number of nodes in the source subnetwork.
        n_modes_src : int
            Number of modes in the source model
            (``model.number_of_modes``).
        """
        # Validate target_scales length when provided
        if self.target_scales is not None:
            if len(self.target_scales) != len(self.target_cvar):
                raise ValueError(
                    f"target_scales length ({len(self.target_scales)}) must equal "
                    f"target_cvar length ({len(self.target_cvar)})."
                )

        # Horizon = max_delay + 1, matching classic TVB n_time
        self._horizon = self.max_delay + 1
        if (
            self._horizon == 0
        ):  # Should not happen if max_delay is at least 0, but defensive
            self._horizon = 1

        buffer_shape = (n_vars_src, n_nodes_src, n_modes_src, self._horizon)
        self._history_buffer = np.zeros(buffer_shape, "f")

    def update_buffer(self, current_src_state: np.ndarray, t: int):
        """Write the current source state into the circular history buffer.

        The slot written is ``t % horizon``, overwriting the state that is
        exactly ``horizon`` steps old.

        Parameters
        ----------
        current_src_state : ndarray, shape (n_vars_src, n_nodes_src, n_modes_src)
            Full source state at the current time step.
        t : int
            Current time step index (used to select the buffer slot).
        """
        buffer_idx = t % self._horizon
        # Store the full state, not just the source_cvar subset
        # Extraction happens in apply() when reading from buffer
        self._history_buffer[..., buffer_idx] = current_src_state

    def apply(self, tgt: np.ndarray, t: int, mode_map: np.ndarray, x_i=None):
        """Compute delayed weighted coupling and accumulate into the target array.

        Reads delayed source states from the internal history buffer,
        applies the sparse weights and optional coupling function, then
        adds the result to the appropriate coupling variables of ``tgt``.

        For each non-zero weight entry *k* connecting source node *j* to
        target node *i*, the delayed state is read from buffer slot
        ``(t - 1 - idelays[k] + horizon) % horizon``.  Using ``t - 1``
        ensures that coupling at step *t* uses the state from step
        ``t - 1 - idelays``, matching the classic TVB delay convention.

        Parameters
        ----------
        tgt : ndarray, shape (n_vars_tgt, n_nodes_tgt, n_modes_tgt)
            Target coupling-variable array.  The slices indexed by
            ``target_cvar`` are incremented in-place.
        t : int
            Current time step index.
        mode_map : ndarray, shape (n_modes_src, n_modes_tgt)
            Linear map from source modes to target modes applied after
            the weighted sum.  For intra-subnetwork coupling this is the
            identity matrix; for inter-subnetwork coupling it may
            re-weight or collapse mode contributions.
        x_i : ndarray, shape (n_vars_tgt, n_nodes_tgt, n_modes_tgt), optional
            Current target subnetwork state.  When provided and the
            coupling function's ``pre()`` method accepts two arguments,
            the target state at the coupling variables is expanded to
            per-edge shape and passed as the second argument to
            ``cfun.pre()``.  This enables coupling functions like
            ``Difference`` and ``Kuramoto`` to compute expressions that
            depend on both source and target activity (e.g.
            ``x_j - x_i`` or ``sin(x_j - x_i)``).

        Raises
        ------
        ValueError
            If ``source_cvar.size`` and ``target_cvar.size`` are both
            greater than 1 and unequal (ambiguous cvar mapping).
        """

        # Calculate time indices into the circular history buffer
        # Shape: (nnz,) where nnz is number of non-zero weights/lengths
        # Use t - 1 to match classic TVB simulator delay model:
        # coupling at step t uses state from step t-1-idelays
        time_indices = (t - 1 - self.idelays + self._horizon) % self._horizon

        # Gather delayed states from the internal history buffer
        # Indexing: [source_cvar_indices, node_indices, mode_indices, time_indices]
        # source_cvar needs shape (n_source_cvar, 1) for broadcasting
        # weights.indices gives the source node indices for each non-zero weight
        # We need all modes from the source, hence ':' for mode dimension
        delayed_states = self._history_buffer[
            self.source_cvar[:, np.newaxis],  # Shape (n_src_cvar, 1)
            self.weights.indices,  # Shape (nnz,) -> Source node indices
            :,  # All source modes
            time_indices,  # Shape (nnz,) -> Time index for each connection
        ]
        # Result shape: (n_src_cvar, nnz, n_src_modes)

        # Apply pre-scaling coupling function per-edge before weighting
        if self.cfun is not None:
            if x_i is not None:
                if "x_i" in inspect.signature(self.cfun.pre).parameters:
                    # Derive per-edge target state from x_i.
                    # In CSR format, for entry ptr the target node (row) can be
                    # recovered from indptr: row_indices[ptr] = row containing ptr.
                    row_indices = np.repeat(
                        np.arange(self.weights.shape[0]),
                        np.diff(self.weights.indptr),
                    )
                    # x_i has shape (n_vars_tgt, n_nodes_tgt, n_modes_tgt).
                    # Extract target state at the coupling variables for each
                    # target node, giving per-edge x_i with shape
                    # (n_tgt_cvar, nnz, n_modes_tgt).
                    x_i_per_edge = x_i[
                        self.target_cvar[:, np.newaxis], row_indices, :
                    ]
                    # Handle shape mismatch between source and target cvar dims.
                    # When cvar counts match, expand x_i_per_edge to align with
                    # delayed_states along axis 0 for element-wise pre().
                    n_sc = self.source_cvar.size
                    n_tc = self.target_cvar.size
                    if n_sc == n_tc:
                        delayed_states = self.cfun.pre(delayed_states, x_i_per_edge)
                    elif n_sc == 1 and n_tc > 1:
                        # Broadcast single source cvar across multiple target cvars
                        delayed_states = self.cfun.pre(delayed_states, x_i_per_edge)
                    elif n_tc == 1 and n_sc > 1:
                        # Broadcast single target cvar across multiple source cvars
                        delayed_states = self.cfun.pre(delayed_states, x_i_per_edge)
                    else:
                        raise ValueError(
                            f"Cannot pass x_i to pre(): source_cvar size ({n_sc}) "
                            f"and target_cvar size ({n_tc}) are both > 1 and "
                            f"unequal. The shape semantics are ambiguous."
                        )
                else:
                    delayed_states = self.cfun.pre(delayed_states)
            else:
                delayed_states = self.cfun.pre(delayed_states)

        # Detect cvar collapse: when a multi-cvar pre() (e.g. SigmoidalJansenRit
        # or PreSigmoidal in dynamic mode) collapses N source cvars into 1
        # output, the first dimension of delayed_states shrinks.  We record
        # the effective cvar count for the downstream accumulation logic.
        effective_n_src_cvar = delayed_states.shape[0]

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
            axis=1,  # Sum along the nnz dimension
        )
        # Result shape: (n_src_cvar, n_target_nodes, n_src_modes)
        # Note: n_target_nodes comes from the structure of the CSR matrix (number of rows)

        # Apply post-scaling coupling function if provided
        # Order: post is applied BEFORE scale, giving C_i = scale * post(sum).
        # This matches the docstring convention and the classic TVB order
        # where coupling functions are applied before any projection-specific
        # scaling factor.  For nonlinear post() (e.g., Sigmoidal), the
        # order matters: post(sum) * scale != post(scale * sum).
        if self.cfun is not None:
            summed_input = self.cfun.post(summed_input)

        # Apply scaling factor after post()
        scaled_input = self.scale * summed_input

        # Apply mode mapping
        # mode_map shape: (n_src_modes, n_target_modes)
        # Need matrix multiplication: (n_src_cvar, n_target_nodes, n_src_modes) @ (n_src_modes, n_target_modes)
        # Result shape: (n_src_cvar, n_target_nodes, n_target_modes)
        aff = scaled_input @ mode_map

        # Apply to target coupling variables
        # tgt shape: (n_target_vars, n_target_nodes, n_target_modes)
        # target_cvar shape: (n_target_cvar,)
        # Use effective_n_src_cvar instead of source_cvar.size to handle
        # cvar-collapsing coupling functions (e.g. SigmoidalJansenRit 2→1).
        if self.target_cvar.size == 1:
            # If only one target cvar, sum contributions from all source cvars
            # Sum along axis 0 (source_cvar dimension)
            # Result shape: (n_target_nodes, n_target_modes)
            summed_aff = aff.sum(axis=0)
            if self.target_scales is not None:
                summed_aff = summed_aff * self.target_scales[0]
            # Add to the single target cvar across all nodes/modes
            tgt[self.target_cvar[0], :, :] += summed_aff
        elif effective_n_src_cvar == 1:
            # If only one effective source cvar (possibly collapsed from N),
            # apply its contribution to all target cvars directly.
            # aff has shape (1, n_target_nodes, n_target_modes)
            # Squeeze axis 0 to get shape (n_target_nodes, n_target_modes)
            squeezed_aff = aff.squeeze(axis=0)
            if self.target_scales is not None:
                # target_scales shape (n_target_cvar,) → broadcast over nodes and modes
                squeezed_aff = squeezed_aff * self.target_scales[:, None, None]
            # Add to all target cvars
            tgt[self.target_cvar, :, :] += squeezed_aff
        elif effective_n_src_cvar == self.target_cvar.size:
            # If effective source and target cvars match in number, apply element-wise mapping
            # aff shape: (n_cvar, n_target_nodes, n_target_modes)
            # tgt[target_cvar] shape: (n_cvar, n_target_nodes, n_target_modes)
            if self.target_scales is not None:
                aff = aff * self.target_scales[:, None, None]
            tgt[self.target_cvar, :, :] += aff
        else:
            # Ambiguous case: M effective source cvars to N target cvars
            # Raise error or define specific reduction/broadcasting rule
            raise ValueError(
                f"Unsupported projection: {effective_n_src_cvar} effective source cvars "
                f"to {self.target_cvar.size} target cvars. "
                f"Requires M=1, N=1, or M=N."
            )
