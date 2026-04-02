import numpy as np

from .base_projection import BaseProjection
from .cvar_utils import resolve_cvar_names, validate_cvar_indices


class IntraProjection(BaseProjection):
    """
    Defines internal coupling within a Subnetwork using the BaseProjection mechanism.

    Inherits time delay handling and sparse weight requirements from BaseProjection.
    Assumes source and target modes are the same, using an identity mode map.
    """

    # Inherits source_cvar, target_cvar, scale, weights, lengths, cv, dt, etc.
    # from BaseProjection.
    # NOTE: Weights and lengths must be provided as sparse CSR matrices.
    # NOTE: Time delays (lengths, cv, dt) can optionally be specified. If not,
    #       minimal delay (2 steps) is assumed by BaseProjection.

    # _identity_mode_map is initialized in __init__ as an internal cache

    def __init__(self, **kwargs):
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
        """Apply the internal projection using the base class logic.

        Uses an identity matrix for mode mapping. Requires configure_buffer and
        update_buffer to have been called appropriately before this method.

        Parameters
        ----------
        tgt : ndarray
            Target state array (internal coupling) to modify.
        t : int
            Current time step index.
        n_modes : int
            Number of modes in the subnetwork.
        """
        identity_map = self._get_identity_mode_map(n_modes)
        # Call BaseProjection.apply with the identity map
        # BaseProjection.apply now uses its internal buffer
        super().apply(tgt, t, identity_map)
