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
Intra-subnetwork projection for the hybrid simulator.

An ``IntraProjection`` couples state variables within a single subnetwork —
source and target are the same set of nodes.  Because source and target share
the same mode dimension, the mode mapping is always the identity matrix,
created lazily in ``_get_identity_mode_map()`` and cached between calls.

The coupling-variable (cvar) specification may be given as integer index
arrays or as string names taken from the model's ``cvar`` attribute.  String
names are resolved to integer indices lazily: if the model is not yet
available at construction time (a common pattern when projections are
assembled before the owning ``Subnetwork`` is fully configured), resolution is
deferred and completed when ``set_model_for_cvar_resolution()`` is called
during the subnetwork's ``configure()`` phase.
"""

import numpy as np

from .base_projection import BaseProjection
from .cvar_utils import resolve_cvar_names, validate_cvar_indices


class IntraProjection(BaseProjection):
    """Intra-subnetwork coupling: nodes coupled to themselves within one subnetwork.

    Wraps ``BaseProjection`` for the case where the source and target are the
    same set of nodes in a single ``Subnetwork``.  Because source and target
    share the same mode dimension, ``mode_map`` is always the identity matrix,
    created lazily by ``_get_identity_mode_map()`` and cached between calls.

    Coupling-variable (cvar) indices may be passed as integer arrays or as
    string names matching entries in the model's ``cvar`` list.  String names
    are resolved lazily: if the model is not available at construction time,
    resolution is deferred until ``set_model_for_cvar_resolution()`` is
    called by the owning ``Subnetwork`` during its ``configure()`` phase.

    Attributes
    ----------
    (all inherited from BaseProjection)

    See Also
    --------
    BaseProjection : Parent class providing the buffer and delay logic.
    InterProjection : Coupling between different subnetworks with an
        explicit mode map.
    """

    # Inherits source_cvar, target_cvar, scale, weights, lengths, cv, dt, etc.
    # from BaseProjection.
    # NOTE: Weights and lengths must be provided as sparse CSR matrices.
    # NOTE: Time delays (lengths, cv, dt) can optionally be specified. If not,
    #       minimal delay (2 steps) is assumed by BaseProjection.

    # _identity_mode_map is initialized in __init__ as an internal cache

    def __init__(self, **kwargs):
        """Initialise and attempt eager cvar name resolution.

        Calls ``BaseProjection.__init__`` for weight/length validation and
        delay computation, then attempts to resolve any string cvar names
        immediately.  If ``_source_model`` / ``_target_model`` attributes are
        not yet available (the typical case when projections are created before
        ``Subnetwork.configure()`` runs), resolution is deferred and flagged
        via ``_source_cvar_resolved`` / ``_target_cvar_resolved``.

        Parameters
        ----------
        **kwargs
            Passed directly to ``BaseProjection.__init__``.
        """
        # BaseProjection.__init__ handles validation (CSR format), epsilon, delays etc.
        super().__init__(**kwargs)

        # Resolve cvar names to indices if they are strings
        # Note: This requires model to be configured, which may not be true yet
        # We'll attempt resolution and store for later use
        self._source_cvar_resolved = False
        self._target_cvar_resolved = False

        # Try to resolve source cvar if we can access it
        # (may not be possible if called outside Subnetwork context)
        cvar_source = getattr(self, "source_cvar", None)
        if cvar_source is not None and hasattr(self, "_source_model"):
            try:
                self.source_cvar = resolve_cvar_names(self._source_model, cvar_source)
                validate_cvar_indices(self._source_model, self.source_cvar)
                self._source_cvar_resolved = True
            except (AttributeError, TypeError, ValueError):
                # Model not yet configured, defer resolution
                pass

        # Try to resolve target cvar similarly
        cvar_target = getattr(self, "target_cvar", None)
        if cvar_target is not None and hasattr(self, "_target_model"):
            try:
                self.target_cvar = resolve_cvar_names(self._target_model, cvar_target)
                validate_cvar_indices(self._target_model, self.target_cvar)
                self._target_cvar_resolved = True
            except (AttributeError, TypeError, ValueError):
                # Model not yet configured, defer resolution
                pass

        # Determine the number of modes from the associated model (if possible)
        # This requires the projection to be associated with a subnetwork *after* init,
        # or the number of modes passed explicitly. We'll create it lazily in apply.
        self._identity_mode_map = None

    def set_model_for_cvar_resolution(self, model):
        """Set model reference for cvar name resolution.

        This is called by Subnetwork after the model is configured,
        allowing IntraProjection to resolve string cvar names to indices.

        Parameters
        ----------
        model : Model
            The model to use for resolving cvar names.
        """
        self._source_model = model
        self._target_model = model

        # Resolve cvars if not already done
        if not self._source_cvar_resolved:
            cvar_source = getattr(self, "source_cvar", None)
            if cvar_source is not None:
                self.source_cvar = resolve_cvar_names(model, cvar_source)
                validate_cvar_indices(model, self.source_cvar)
                self._source_cvar_resolved = True

        if not self._target_cvar_resolved:
            cvar_target = getattr(self, "target_cvar", None)
            if cvar_target is not None:
                self.target_cvar = resolve_cvar_names(model, cvar_target)
                validate_cvar_indices(model, self.target_cvar)
                self._target_cvar_resolved = True

    def initialize_history_buffer(self, initial_state: np.ndarray):
        """Initialize the history buffer with a given state.

        This is useful for setting up initial conditions for the projection,
        particularly when the projection is part of a Subnetwork and needs
        its history buffer pre-filled before simulation starts.

        Parameters
        ----------
        initial_state : ndarray
            Initial state to use for initializing the history buffer.
            Shape should be (n_vars_src, n_nodes_src, n_modes_src).
        """
        if self._history_buffer is None:
            raise RuntimeError(
                "History buffer not configured. Call configure_buffer() first."
            )

        # Fill all time slots with the initial state
        horizon = self._horizon
        for t in range(horizon):
            self.update_buffer(initial_state, t)

    def _get_identity_mode_map(self, n_modes: int) -> np.ndarray:
        """Creates or retrieves the identity mode map."""
        if (
            self._identity_mode_map is None
            or self._identity_mode_map.shape[0] != n_modes
        ):
            self._identity_mode_map = np.eye(n_modes, dtype=np.int_)
        return self._identity_mode_map

    def apply(self, tgt: np.ndarray, t: int, n_modes: int):
        """Apply the intra-subnetwork projection using an identity mode map.

        Constructs (or retrieves from cache) an ``n_modes × n_modes`` identity
        matrix and delegates to ``BaseProjection.apply()``.  Both
        ``configure_buffer()`` and at least one prior call to
        ``update_buffer()`` must have been made before this method is invoked.

        Parameters
        ----------
        tgt : ndarray, shape (n_vars, n_nodes, n_modes)
            Target coupling-variable array.  The slices indexed by
            ``target_cvar`` are incremented in-place.
        t : int
            Current time step index.
        n_modes : int
            Number of modes in the subnetwork, used to build the identity
            mode map.
        """
        identity_map = self._get_identity_mode_map(n_modes)
        # Call BaseProjection.apply with the identity map
        # BaseProjection.apply now uses its internal buffer
        super().apply(tgt, t, identity_map)
