import numpy as np 
from typing import Dict, Tuple, Any, Optional, Union
from tvb.simulator.models.base import Model
from tvb.simulator.models.zerlaut import ZerlautAdaptationSecondOrder
from tvb.basic.neotraits.api import Final, List

class CortexUniversalModel(Model):
    """
    A universal cortical wrapper for the Zerlaut model that implements the ModelWrapper
    protocol for hybrid orchestration while maintaining TVB compatibility.

    In this hybrid setup:
      - get_state_shape() returns the native state shape (a column vector: (8, 1))
      - get_coupling_shape() returns the shape of the coupling output (here (2, 1))
      - reshape_state() converts a flattened state to the native shape.
      - flatten_state() converts a native state into a flat vector.
      
    The methods coupling_function(state) and process_channels(state, incoming_channels)
    remain as the primary entry points to extract outputs and compute derivatives.
    """
    _nvar = 8
    num_coupling_vars = 2  # Produces two coupling variables (E and I)

    # Define state variable ranges and boundaries for monitors.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={
            "E": np.array([1e-3, 250.e-3]),
            "I": np.array([1e-3, 250.e-3]),
            "C_ee": np.array([0.0e-3, 0.5e-3]),
            "C_ei": np.array([-0.5e-3, 0.5e-3]),
            "C_ii": np.array([0.0e-3, 0.5e-3]),
            "W_e": np.array([0.0, 200.0]),
            "W_i": np.array([0.0, 0.0]),
            "ou_drift": np.array([0.0, 0.0])
        },
        doc="Expected dynamic range of state variables."
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            "E": np.array([0.0, None]),
            "I": np.array([0.0, None])
        },
        doc="Boundaries on state variables. Use None for one-sided limits."
    )

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("E", "I", "C_ee", "C_ei", "C_ii", "W_e", "W_i", "ou_drift"),
        default=("E", "I", "C_ee", "C_ei", "C_ii", "W_e", "W_i", "ou_drift"),
        doc="Default state variables to monitor."
    )

    state_variables = 'E I C_ee C_ei C_ii W_e W_i ou_drift'.split()

    def __init__(self, **kwargs):
        """Initialize the CortexUniversalModel with an underlying Zerlaut model."""
        super(CortexUniversalModel, self).__init__(**kwargs)
        self.zerlaut_model = ZerlautAdaptationSecondOrder()
        # Set a parameter for the Zerlaut model.
        self.zerlaut_model.K_ext_i = np.array([0])
        
        # Required TVB configuration attributes.
        self.cvar = np.array([0, 1], dtype=int)  # Coupling variables: indices for E and I.
        self.gvar = np.array([0], dtype=int)
        self.stvar = np.array([0, 1], dtype=int)

        self._nvar = 8
        # The initial state is defined as a column vector.
        self.initial_state = np.zeros((self._nvar, 1), dtype=np.float64)
        # Channel mapping for output extraction.
        self._channel_map = {
            "E": {"index": 0, "description": "Excitatory population rate"},
            "I": {"index": 1, "description": "Inhibitory population rate"}
        }
        
        # If the underlying Zerlaut model has an initial_state, use it.
        if hasattr(self.zerlaut_model, "initial_state"):
            self.initial_state = self.zerlaut_model.initial_state.reshape((-1, 1))

        # Define shape information:
        # We now use a column vector representation.
        self.state_shape = (self._nvar, 1)  # Native state shape.
        self.coupling_shape = (2, 1)         # Expected coupling output shape.
        
        # Define region-to-input mapping configuration
        # Default mapping - can be customized for specific use cases
        self._region_mapping = {}

    def get_state_shape(self) -> Tuple[int, ...]:
        """Return the native state shape for this model."""
        return self.state_shape

    def get_coupling_shape(self) -> Tuple[int, ...]:
        """Return the expected coupling output shape for this model."""
        return self.coupling_shape

    def get_variable_metadata(self) -> Dict[str, Dict]:
        """
        Get metadata about variables.
        
        Returns
        -------
        Dict[str, Dict]
            Dictionary mapping variable names to metadata dictionaries
            with keys like 'description', 'unit', etc.
        """
        metadata = {}
        # Add channel metadata
        for name, info in self._channel_map.items():
            metadata[name] = {
                "description": info.get("description", ""),
                "index": info.get("index", 0)
            }
        
        # Add metadata for all state variables
        for i, var_name in enumerate(self.state_variables):
            if var_name not in metadata:
                metadata[var_name] = {
                    "description": f"{var_name} state variable",
                    "index": i
                }
                
        return metadata

    def validate_state(self, state: np.ndarray) -> np.ndarray:
        """
        Validate that the state is a numpy array of the correct shape,
        converting it if necessary.
        
        Parameters
        ----------
        state : numpy.ndarray
            State array to validate
            
        Returns
        -------
        numpy.ndarray
            Validated state in native shape (8,1)
            
        Raises
        ------
        TypeError
            If state is not a numpy array
        ValueError
            If state cannot be reshaped to native shape or contains invalid values
        """
        if not isinstance(state, np.ndarray):
            raise TypeError("State must be a numpy array")
            
        # Get total size to validate
        total_elements = np.prod(self.state_shape)
        
        # Handle various input shapes
        if state.size != total_elements:
            raise ValueError(f"State has {state.size} elements, expected {total_elements}")
        
        # Reshape to native shape
        reshaped_state = state.reshape(self.state_shape)
        
        if not np.all(np.isfinite(reshaped_state)):
            raise ValueError("State contains non-finite values")
            
        return reshaped_state

    def reshape_state(self, state: np.ndarray) -> np.ndarray:
        """
        Reshape a state vector into the native state shape (8,1).
        Handles any input shape as long as total elements match.
        
        Parameters
        ----------
        state : numpy.ndarray
            State array to reshape
            
        Returns
        -------
        numpy.ndarray
            Reshaped state in native shape
        """
        # Use the validation method to ensure consistent handling
        return self.validate_state(state)

    def flatten_state(self, state: np.ndarray) -> np.ndarray:
        """
        Flatten the state into a one-dimensional vector.
        Handles any input shape.
        
        Parameters
        ----------
        state : numpy.ndarray
            State array to flatten
            
        Returns
        -------
        numpy.ndarray
            Flattened 1D state array
        """
        # First ensure we have the correct shape
        state = self.validate_state(state)
        # Then flatten
        return state.flatten()

    def coupling_function(self, state: np.ndarray) -> np.ndarray:
        """
        Extract the coupling variables (E and I firing rates) from the state.
        Expects state to be in shape (8, 1).
        Returns a (2, 1) array.
        """
        state = self.validate_state(state)
        coupling = np.zeros((2, 1), dtype=np.float64)
        coupling[0] = state[0]  # Excitatory rate.
        coupling[1] = state[1]  # Inhibitory rate.
        return coupling

    def get_channel_outputs(self, state: np.ndarray) -> Dict[str, float]:
        """
        Return a dictionary with keys 'E' and 'I' corresponding to the firing rates.
        """
        state = self.validate_state(state)
        return {"E": float(state[0, 0]), "I": float(state[1, 0])}

    def set_region_mapping(self, region: str, channel_to_input_map: Dict[str, str]) -> None:
        """
        Configure how channels from a specific region map to external inputs.
        
        Parameters
        ----------
        region : str
            Name of the source region
        channel_to_input_map : Dict[str, str]
            Dictionary mapping channel names (e.g. 'E', 'I') to input parameters
            Valid input parameters: 'ex_ex', 'ex_in', 'in_ex', 'in_in'
            
        Example
        -------
        model.set_region_mapping('V1', {'E': 'ex_ex', 'I': 'in_ex'})
        # Maps V1's excitatory output to excitatory->excitatory input
        # and V1's inhibitory output to inhibitory->excitatory input
        """
        if not isinstance(region, str):
            raise TypeError("Region name must be a string")
        if not isinstance(channel_to_input_map, dict):
            raise TypeError("Channel mapping must be a dictionary")
            
        # Validate input parameter names
        valid_inputs = {'ex_ex', 'ex_in', 'in_ex', 'in_in'}
        for channel, input_param in channel_to_input_map.items():
            if input_param not in valid_inputs:
                raise ValueError(f"Invalid input parameter '{input_param}'. Must be one of: {valid_inputs}")
        
        self._region_mapping[region] = channel_to_input_map

    def _apply_region_channels(self, region_channels: Dict[str, Dict[str, float]]) -> None:
        """
        Apply region-specific channel inputs to the appropriate external input parameters.
        
        Parameters
        ----------
        region_channels : Dict[str, Dict[str, float]]
            Dictionary mapping region names to channel values
        """
        # Initialize all external inputs to zero
        self.zerlaut_model.external_input_ex_ex = np.array([0.0])
        self.zerlaut_model.external_input_ex_in = np.array([0.0])
        self.zerlaut_model.external_input_in_ex = np.array([0.0])
        self.zerlaut_model.external_input_in_in = np.array([0.0])
        
        # Apply region-specific mapping
        for region, channels in region_channels.items():
            # Skip if no mapping exists for this region
            if region not in self._region_mapping:
                continue
                
            mapping = self._region_mapping[region]
            for channel, value in channels.items():
                # Skip if no mapping exists for this channel
                if channel not in mapping:
                    continue
                    
                input_param = mapping[channel]
                if input_param == 'ex_ex':
                    self.zerlaut_model.external_input_ex_ex = np.array([value])
                elif input_param == 'ex_in':
                    self.zerlaut_model.external_input_ex_in = np.array([value])
                elif input_param == 'in_ex':
                    self.zerlaut_model.external_input_in_ex = np.array([value])
                elif input_param == 'in_in':
                    self.zerlaut_model.external_input_in_in = np.array([value])

    def process_channels(self, state: np.ndarray, incoming_channels: Union[Dict[str, float], Dict[str, Any]]) -> np.ndarray:
        """
        Process the incoming channels by injecting external inputs into the Zerlaut model
        and then compute the state derivatives. Supports both flat channel dictionaries and
        region-structured dictionaries.

        Parameters
        ----------
        state : numpy.ndarray
            The current state (expected shape: (8, 1))
        incoming_channels : Union[Dict[str, float], Dict[str, Any]]
            Either:
            - Traditional flat dictionary: {"E": value, "I": value}
            - Region-structured: {"flat": {"E": value, "I": value}, 
                                "regions": {"region1": {"E": value, "I": value}, ...}}

        Returns
        -------
        numpy.ndarray
            The state derivative, as computed by the Zerlaut model's dfun (shape (8, 1)).
        """
        state = self.validate_state(state)
        
        # Detect if we have region-structured channels
        has_region_structure = isinstance(incoming_channels, dict) and \
                              "flat" in incoming_channels and \
                              "regions" in incoming_channels
        
        # Handle region-structured channels if available and region mapping is configured
        if has_region_structure and self._region_mapping:
            # Apply region-specific mapping
            self._apply_region_channels(incoming_channels["regions"])
        else:
            # Fallback to traditional flat channel handling for backward compatibility
            channels_to_use = incoming_channels["flat"] if has_region_structure else incoming_channels
            
            # Explicitly use keys instead of relying on order
            exc_input = float(channels_to_use.get("E", 0.0))
            inh_input = float(channels_to_use.get("I", 0.0))
            
            # Additional validation for robustness
            if not (np.isfinite(exc_input) and np.isfinite(inh_input)):
                raise ValueError("Channel inputs must be finite")

            # Default routing: feed I input into in_ex (inhibitory to excitatory) 
            self.zerlaut_model.external_input_ex_ex = np.array([exc_input])
            self.zerlaut_model.external_input_in_ex = np.array([0.0])
            self.zerlaut_model.external_input_ex_in = np.array([0.0])
            self.zerlaut_model.external_input_in_in = np.array([0.0])

        # Use a dummy coupling since external inputs are directly injected.
        dummy_coupling = np.zeros((self.num_coupling_vars, 1))
        derivatives = self.zerlaut_model.dfun(state, dummy_coupling, local_coupling=0.0)
        return derivatives

    def dfun(self, state, coupling, local_coupling=0.0):
        """
        Dummy dfun for backward compatibility.
        Delegates to process_channels() using the coupling array as a fallback.
        """
        incoming = {"E": float(coupling[0, 0]) if coupling.shape[0] > 0 else 0.0,
                    "I": float(coupling[1, 0]) if coupling.shape[0] > 1 else 0.0}
        return self.process_channels(state, incoming)