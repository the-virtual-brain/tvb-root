import numpy as np
from scipy import sparse as sp
import tvb.basic.neotraits.api as t
from tvb.datatypes.patterns import SpatioTemporalPattern

from .cvar_utils import resolve_cvar_names
from .projection_utils import create_all_to_all_weights

# Import here to avoid circular dependency
import tvb.simulator.hybrid.subnetwork as sn_module


class Stim(t.HasTraits):
    """External stimulus for hybrid networks.

    Stim provides external input to a target subnetwork, similar to projections
    but with a fixed spatiotemporal pattern rather than coupling from another subnetwork.

    Attributes
    ----------
    stimulus : SpatioTemporalPattern
        Spatiotemporal pattern defining the stimulus (e.g., StimuliRegion)
    target : Subnetwork
        Target subnetwork that receives the stimulus
    target_cvar : str or ndarray of int
        Name of target state variable that receives stimulus (uses cvar resolution)
    projection_scale : float
        Scaling factor for the stimulus
    weights : scipy.sparse.csr_matrix
        Spatial weights matrix in CSR sparse format. Default: all-to-all identity.
    dt : float
        Time step for temporal evaluation (set from target's scheme.dt during configure)
    time : ndarray
        Configured time vector for temporal evaluation

    Methods
    -------
    configure(simulation_length)
        Configure stimulus with time vector and spatial pattern
    get_coupling(step)
        Return coupling contribution at given step
    """

    stimulus = t.Attr(field_type=SpatioTemporalPattern)
    target = t.Attr(field_type=sn_module.Subnetwork)
    target_cvar = t.NArray(dtype=np.int_, required=False)
    projection_scale = t.Float(default=1.0)
    weights = t.Attr(field_type=sp.csr_matrix, required=False)
    dt = t.Float(required=False, default=None)
    time = t.NArray(required=False, default=None)

    def __init__(self, **kwargs):
        # Resolve cvar names to indices BEFORE calling super().__init__()
        # to avoid type validation errors with string cvar names
        target_cvar_input = kwargs.pop("target_cvar", None)
        target_subnet = kwargs.get("target")

        if target_cvar_input is not None:
            if isinstance(target_cvar_input, str) or (
                isinstance(target_cvar_input, (list, tuple, np.ndarray))
                and len(target_cvar_input) > 0
                and isinstance(target_cvar_input[0], str)
            ):
                # Resolve names to indices
                if target_subnet is None:
                    raise ValueError(
                        "target must be provided to resolve cvar names"
                    )
                target_cvar_resolved = resolve_cvar_names(
                    target_subnet.model, target_cvar_input
                )
            else:
                # Already indices
                target_cvar_resolved = np.atleast_1d(target_cvar_input)

            kwargs["target_cvar"] = target_cvar_resolved

        super().__init__(**kwargs)

        # Ensure target_cvar is an array
        self.target_cvar = np.atleast_1d(self.target_cvar)

        # Validate weights is CSR and correct shape
        if self.weights is not None:
            if not isinstance(self.weights, sp.csr_matrix):
                raise TypeError(
                    f"Weights must be provided as a scipy.sparse.csr_matrix, got {type(self.weights)}"
                )
            if self.weights.shape != (self.target.nnodes, self.target.nnodes):
                raise ValueError(
                    f"Weights shape {self.weights.shape} must match target nodes "
                    f"({self.target.nnodes}, {self.target.nnodes})"
                )

        # Validate weights shape
        if self.weights is not None and self.weights.shape != (
            self.target.nnodes,
            self.target.nnodes,
        ):
            raise ValueError(
                f"Weights shape {self.weights.shape} must match target nodes "
                f"({self.target.nnodes}, {self.target.nnodes})"
            )

    def configure(self, simulation_length: float):
        """Configure stimulus with time vector and spatial pattern.

        Parameters
        ----------
        simulation_length : float
            Total simulation length in milliseconds
        """
        # Set dt from target's scheme
        self.dt = self.target.scheme.dt

        # Create time vector
        self.time = np.arange(0.0, simulation_length + self.dt, self.dt)

        # Configure temporal pattern
        if hasattr(self.stimulus, "configure_time"):
            self.stimulus.configure_time(self.time.reshape((1, -1)))

        # Configure spatial pattern if needed (e.g., StimuliRegion)
        if hasattr(self.stimulus, "configure_space"):
            # StimuliRegion needs configure_space called to set up spatial pattern
            self.stimulus.configure_space()

    def get_coupling(self, step: int) -> np.ndarray:
        """Return coupling contribution at given step.

        Parameters
        ----------
        step : int
            Current simulation step index

        Returns
        -------
        ndarray
            Coupling contribution array with shape (n_cvar, n_nodes, n_modes)
        """
        # Get current time
        t = step * self.dt

        # Evaluate temporal pattern at current time
        # Stimulus(temporal_indices=step) returns spatial pattern at that time
        temporal_value = self.stimulus(temporal_indices=step)

        # temporal_value shape from StimuliRegion: (n_stim_regions, 1)
        # But target subnet may have fewer nodes
        # Extract only the relevant nodes for the target subnet

        # Apply spatial weights
        # If weights is None, create identity (no spatial weighting)
        if self.weights is None:
            temporal_value = self.stimulus(temporal_indices=step)
            scaled_value = self.projection_scale * temporal_value
        else:
            # weights: (n_nodes, n_nodes) @ temporal_value: (n_nodes, 1) -> (n_nodes, 1)
            weighted_value = self.weights @ temporal_value

            # Apply projection scale
            scaled_value = self.projection_scale * weighted_value

        # Broadcast to shape (n_cvar, n_nodes, n_modes)
        # target_cvar defines which coupling variables receive stimulus
        # Get number of modes from target
        n_modes = self.target.nmodes if hasattr(self.target, "nmodes") else 1

        # Create coupling array: (n_cvar, n_nodes, n_modes)
        n_cvar = len(self.target_cvar)
        n_nodes = self.target.nnodes

        # scaled_value may have shape (n_stim_regions, 1) from StimuliRegion
        # where n_stim_regions could be larger than target.nnodes
        # Extract only the first n_nodes if necessary
        if scaled_value.shape[0] > n_nodes:
            scaled_value = scaled_value[:n_nodes, :]

        # scaled_value is now (n_nodes, 1)
        # We need to broadcast to (n_cvar, n_nodes, n_modes)
        # Tile scaled_value to match target shape
        coupling = np.tile(scaled_value, (n_cvar, 1, 1))

        return coupling
