import numpy as np
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
    scale : float
        Scaling factor for the projection
    weights : ndarray
        Connectivity weights matrix
    mode_map : ndarray
        Mapping between source and target modes
    """

    source: Subnetwork = t.Attr(Subnetwork)
    target: Subnetwork = t.Attr(Subnetwork)
    source_cvar = t.NArray(dtype=np.int_)  # Array of source coupling variable indices
    target_cvar = t.NArray(dtype=np.int_)  # Array of target coupling variable indices
    scale: float = t.Float(default=1.0)
    weights: np.ndarray = t.NArray(dtype=np.float_)
    mode_map: np.ndarray = t.NArray(required=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.mode_map is None:
            self.mode_map = np.ones(
                (self.source.model.number_of_modes,
                 self.target.model.number_of_modes, ))
            self.mode_map /= self.source.model.number_of_modes
            
        # Convert scalar indices to arrays for unified handling
        self.source_cvar = np.atleast_1d(self.source_cvar)
        self.target_cvar = np.atleast_1d(self.target_cvar)
        self._validate_cvars()

    def _validate_cvars(self):
        """Validate coupling variable configurations."""
        src_size = len(self.source_cvar)
        tgt_size = len(self.target_cvar)
        
        # Validate indices are within bounds
        if np.any(self.source_cvar >= self.source.model.cvar.size):
            raise ValueError(f"Source coupling variable index {self.source_cvar} out of bounds")
        if np.any(self.target_cvar >= self.target.model.cvar.size):
            raise ValueError(f"Target coupling variable index {self.target_cvar} out of bounds")
            
        # Validate broadcasting configurations
        if not (src_size == 1 or tgt_size == 1 or src_size == tgt_size):
            raise ValueError(
                f"Invalid coupling variable configuration: source size {src_size} "
                f"and target size {tgt_size} must be either equal or one must be 1"
            )

    def apply(self, tgt, src):
        """Apply the projection to compute coupling.
        
        Handles three cases:
        a) One source to many targets (broadcasting)
        b) Many sources to one target (reduction/summation)
        c) Equal number of sources to targets (element-wise)
        
        Parameters
        ----------
        tgt : ndarray
            Target state array to modify
        src : ndarray
            Source state array
        """
        # Compute base coupling for each source variable
        gx_array = self.scale * self.weights @ src[self.source_cvar] @ self.mode_map
        # TODO add time delays
        if self.source_cvar.size == 1:
            # Case a: broadcast single source to all targets
            tgt[self.target_cvar] += gx_array[0]
        elif self.target_cvar.size == 1:
            # Case b: sum all sources to single target
            tgt[self.target_cvar] += gx_array.sum(axis=0)
        else:
            # Case c: element-wise mapping
            tgt[self.target_cvar] += gx_array 