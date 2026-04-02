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
Utility functions for creating projections.

This module provides factory functions to simplify projection creation,
automating common boilerplate code like extracting connectivity subsets
and converting to sparse matrices.
"""

import numpy as np
from scipy import sparse as sp
from typing import Optional

from .inter_projection import InterProjection
from .intra_projection import IntraProjection
from .cvar_utils import resolve_cvar_names, validate_cvar_indices


def extract_connectivity_subset(
    connectivity, source_indices, target_indices, use_weights=True, use_lengths=True
):
    """Extract a subset of connectivity for a projection.

    Given global connectivity and source/target node indices, extract
    the corresponding weights and lengths matrices as sparse CSR matrices.

    Parameters
    ----------
    connectivity : Connectivity
        Global connectivity matrix to extract from.
    source_indices : ndarray
        Array of source node indices.
    target_indices : ndarray
        Array of target node indices.
    use_weights : bool, default=True
        Whether to extract weights (default: True).
    use_lengths : bool, default=True
        Whether to extract lengths (default: True).

    Returns
    -------
    dict
        Dictionary with 'weights' and 'lengths' keys containing
        sparse CSR matrices of shape (len(target_indices), len(source_indices)).

    Examples
    --------
    >>> extract_connectivity_subset(conn, [0, 1], [2, 3])
    {'weights': <4x2 sparse matrix>, 'lengths': <4x2 sparse matrix>}
    """
    # Extract weights if requested
    if use_weights:
        if hasattr(connectivity, "weights"):
            weights_subset = connectivity.weights[
                np.ix_(target_indices, source_indices)
            ]
        else:
            raise ValueError("Connectivity has no weights attribute")
    else:
        weights_subset = None

    # Extract lengths if requested
    if use_lengths:
        if hasattr(connectivity, "tract_lengths"):
            lengths_subset = connectivity.tract_lengths[
                np.ix_(target_indices, source_indices)
            ]
        elif hasattr(connectivity, "lengths"):
            lengths_subset = connectivity.lengths[
                np.ix_(target_indices, source_indices)
            ]
        else:
            raise ValueError("Connectivity has no tract_lengths or lengths attribute")
    else:
        lengths_subset = None

    # Convert to sparse CSR format
    result = {}

    if use_weights:
        result["weights"] = sp.csr_matrix(weights_subset)

    if use_lengths:
        result["lengths"] = sp.csr_matrix(lengths_subset)

    return result


def create_inter_projection(
    source_subnet,
    target_subnet,
    source_cvar,
    target_cvar,
    connectivity=None,
    source_indices=None,
    target_indices=None,
    weights=None,
    lengths=None,
    cv=None,
    dt=None,
    scale=1.0,
    mode_map=None,
    coupling=None,
):
    """Factory function to create an InterProjection.

    Simplifies projection creation by handling common boilerplate like
    extracting connectivity subsets and auto-calculating cv/dt from subnetworks.

    Parameters
    ----------
    source_subnet : Subnetwork
        Source subnetwork for the projection.
    target_subnet : Subnetwork
        Target subnetwork for the projection.
    source_cvar : str or list of str or ndarray of int
        Coupling variables in source (can use names now!).
    target_cvar : str or list of str or ndarray of int
        Coupling variables in target (can use names now!).
    connectivity : Connectivity, optional
        Global connectivity to extract weights/lengths from.
        If None, must provide weights explicitly.
    source_indices : ndarray, optional
        Source node indices from connectivity.
        If None and connectivity provided, uses all source subnet nodes.
    target_indices : ndarray, optional
        Target node indices from connectivity.
        If None and connectivity provided, uses all target subnet nodes.
    weights : sparse.csr_matrix, optional
        Explicit weights matrix. Takes precedence over connectivity.
    lengths : sparse.csr_matrix, optional
        Explicit lengths matrix. Takes precedence over connectivity.
    cv : float, optional
        Conduction velocity. Defaults to 3.0.
    dt : float, optional
        Time step. Defaults to source subnet's scheme.dt.
    scale : float, default=1.0
        Scaling factor for projection.
    mode_map : ndarray, optional
        Mode mapping matrix. Defaults to uniform mapping.
    coupling : Coupling, optional
        Coupling function to transform afferent activity. If None, uses identity.

    Returns
    -------
    InterProjection
        Configured inter-subnetwork projection.

    Examples
    --------
    >>> # Using connectivity with named cvars
    >>> proj = create_inter_projection(
    ...     source_subnet=cortex,
    ...     target_subnet=thalamus,
    ...     source_cvar='y0',      # Named cvar!
    ...     target_cvar='V1',      # Named cvar!
    ...     connectivity=conn,
    ...     source_indices=[0, 1, 2],
    ...     target_indices=[3, 4, 5],
    ... )
    >>>
    >>> # Using explicit weights
    >>> proj = create_inter_projection(
    ...     source_subnet=cortex,
    ...     target_subnet=thalamus,
    ...     source_cvar=['y0', 'y1'],  # Multiple source cvars!
    ...     target_cvar='V2',          # Single target cvar!
    ...     weights=custom_weights,
    ...     scale=1e-4,
    ... )
    >>>
    >>> # Using coupling function
    >>> from tvb.simulator.hybrid.coupling import Linear
    >>> proj = create_inter_projection(
    ...     source_subnet=cortex,
    ...     target_subnet=thalamus,
    ...     source_cvar='y0',
    ...     target_cvar='V',
    ...     weights=weights,
    ...     coupling=Linear(a=0.5, b=0.1)
    ... )
    """
    # Determine cv and dt defaults
    if dt is None:
        dt = source_subnet.scheme.dt

    if cv is None:
        cv = 3.0  # Default conduction velocity

    # Extract weights/lengths from connectivity if needed
    if connectivity is not None:
        # Determine source and target indices
        # This assumes subnets are ordered and we need the mapping
        # For now, raise error if not provided
        if source_indices is None or target_indices is None:
            raise ValueError(
                "source_indices and target_indices must be provided when using connectivity"
            )

        conn_subset = extract_connectivity_subset(
            connectivity,
            source_indices,
            target_indices,
            use_weights=(weights is None),
            use_lengths=(lengths is None),
        )

        if weights is None:
            weights = conn_subset["weights"]
        if lengths is None:
            lengths = conn_subset["lengths"]
    elif weights is None:
        raise ValueError("Must provide either connectivity or explicit weights/lengths")

    # Validate weights and lengths
    if not isinstance(weights, sp.csr_matrix):
        raise TypeError(f"Weights must be scipy.sparse.csr_matrix, got {type(weights)}")
    if lengths is not None and not isinstance(lengths, sp.csr_matrix):
        raise TypeError(f"Lengths must be scipy.sparse.csr_matrix, got {type(lengths)}")

    # If lengths not provided, create zero-lengths matrix (no delays)
    if lengths is None:
        r, c = weights.shape
        lengths = sp.csr_matrix((r, c), dtype=np.float64)

    # Resolve cvar names to indices
    from .cvar_utils import resolve_cvar_names

    source_cvar_indices = resolve_cvar_names(source_subnet.model, source_cvar)
    target_cvar_indices = resolve_cvar_names(target_subnet.model, target_cvar)

    # Create projection
    proj = InterProjection(
        source=source_subnet,
        target=target_subnet,
        source_cvar=source_cvar_indices,
        target_cvar=target_cvar_indices,
        weights=weights,
        lengths=lengths,
        cv=cv,
        dt=dt,
        scale=scale,
        mode_map=mode_map,
        cfun=coupling,
    )

    return proj


def create_intra_projection(
    subnet,
    source_cvar,
    target_cvar,
    connectivity=None,
    weights=None,
    lengths=None,
    cv=None,
    dt=None,
    scale=1.0,
    coupling=None,
):
    """Factory function to create an IntraProjection.

    Creates internal projections within a subnetwork, with similar
    auto-configuration as create_inter_projection.

    Parameters
    ----------
    subnet : Subnetwork
        The subnetwork containing the internal projection.
    source_cvar : str or list of str or ndarray of int
        Coupling variables in source (can use names now!).
    target_cvar : str or list of str or ndarray of int
        Coupling variables in target (can use names now!).
    connectivity : Connectivity, optional
        Global connectivity for internal structure.
        If None, must provide weights explicitly.
    weights : sparse.csr_matrix, optional
        Explicit weights matrix.
    lengths : sparse.csr_matrix, optional
        Explicit lengths matrix.
    cv : float, optional
        Conduction velocity. Defaults to 3.0.
    dt : float, optional
        Time step. Defaults to subnet's scheme.dt.
    scale : float, default=1.0
        Scaling factor for projection.
    coupling : Coupling, optional
        Coupling function to transform afferent activity. If None, uses identity.

    Returns
    -------
    IntraProjection
        Configured intra-subnetwork projection.
    """
    # Determine cv and dt defaults
    if dt is None:
        dt = subnet.scheme.dt

    if cv is None:
        cv = 3.0  # Default conduction velocity

    # Validate inputs
    if weights is None and connectivity is None:
        raise ValueError("Must provide either connectivity or explicit weights")

    # Validate sparse matrices if provided
    if weights is not None and not isinstance(weights, sp.csr_matrix):
        raise TypeError(f"Weights must be scipy.sparse.csr_matrix, got {type(weights)}")
    if lengths is not None and not isinstance(lengths, sp.csr_matrix):
        raise TypeError(f"Lengths must be scipy.sparse.csr_matrix, got {type(lengths)}")

    # If lengths not provided, create zero-lengths matrix (no delays)
    if lengths is None and weights is not None:
        r, c = weights.shape
        lengths = sp.csr_matrix((r, c), dtype=np.float64)

    # Resolve cvar names to indices
    from .cvar_utils import resolve_cvar_names

    source_cvar_indices = resolve_cvar_names(subnet.model, source_cvar)
    target_cvar_indices = resolve_cvar_names(subnet.model, target_cvar)

    # Create projection
    proj = IntraProjection(
        source_cvar=source_cvar_indices,
        target_cvar=target_cvar_indices,
        weights=weights,
        lengths=lengths,
        cv=cv,
        dt=dt,
        scale=scale,
        cfun=coupling,
    )

    return proj


def create_all_to_all_weights(n_nodes):
    """Create an all-to-all identity weights matrix.

    Useful as a default for stimulus projections or when you want
    fully connected subnets without a global connectivity.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        Square identity matrix as sparse CSR format.

    Examples
    --------
    >>> weights = create_all_to_all_weights(5)
    >>> weights.toarray()
    array([[1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1.]])
    """
    eye_matrix = np.eye(n_nodes, dtype=np.float64)
    return sp.csr_matrix(eye_matrix)
