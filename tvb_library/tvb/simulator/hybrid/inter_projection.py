import numpy as np
import tvb.basic.neotraits.api as t

from .base_projection import BaseProjection
from .subnetwork import Subnetwork


class InterProjection(BaseProjection):
    """A projection from one subnetwork to another (inter-subnetwork).

    Extends BaseProjection by adding source/target subnetwork references
    and mode mapping capabilities.

    Attributes
    ----------
    source : Subnetwork
        Source subnetwork instance.
    target : Subnetwork
        Target subnetwork instance.
    mode_map : ndarray, optional
        Mapping between source and target modes (source_modes x target_modes).
        Defaults to uniform mapping if not provided.
    """
    source: Subnetwork = t.Attr(Subnetwork)
    target: Subnetwork = t.Attr(Subnetwork)
    mode_map = t.NArray(dtype=np.int_, required=False, default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Default mode map if not provided
        if self.mode_map is None:
            self.mode_map = np.ones(
                (self.source.model.number_of_modes,
                 self.target.model.number_of_modes), dtype=np.int_)
        elif self.mode_map.shape != (self.source.model.number_of_modes, self.target.model.number_of_modes):
             raise ValueError(f"Provided mode_map shape {self.mode_map.shape} does not match "
                              f"source modes ({self.source.model.number_of_modes}) x "
                              f"target modes ({self.target.model.number_of_modes})")

    def apply(self, tgt: np.ndarray, step: int):
        """Apply the inter-subnetwork projection.

        Uses the mode_map defined for this projection and the internal history buffer.
        Requires configure_buffer and update_buffer to have been called appropriately.

        Parameters
        ----------
        tgt : ndarray
            Target state array to modify.
        step : int
            Current time step index.
        """
        # Call the base class apply method, passing the specific mode_map
        # BaseProjection.apply now uses its internal buffer
        super().apply(tgt, step, self.mode_map)
