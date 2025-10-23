import numpy as np

from .base_projection import BaseProjection


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

        # Determine the number of modes from the associated model (if possible)
        # This requires the projection to be associated with a subnetwork *after* init,
        # or the number of modes passed explicitly. We'll create it lazily in apply.
        self._identity_mode_map = None


    def _get_identity_mode_map(self, n_modes: int) -> np.ndarray:
        """Creates or retrieves the identity mode map."""
        if self._identity_mode_map is None or self._identity_mode_map.shape[0] != n_modes:
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
