import numpy as np 
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable, Union, Any

# --- TVB Imports ---
from tvb.basic.neotraits.api import HasTraits, Attr
from tvb.simulator.integrators import IntegratorStochastic
from tvb.simulator.models.base import Model
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator import coupling, simulator, monitors

# =============================================================================
# 1. Connectivity and Node Configuration
# =============================================================================

def load_connectivity_file(file_path: str) -> Connectivity:
    """
    Load and configure a TVB connectivity file.
    
    Parameters
    ---------- 
    file_path : str
        Path to the TVB connectivity file (should be in a format TVB can read)
        
    Returns
    ------- 
    Connectivity
        Configured TVB connectivity object ready for simulation
    """
    conn = Connectivity.from_file(file_path)
    conn.configure()
    return conn

@dataclass
class NodeConnection:
    """Defines routing connections between nodes in the hybrid simulator network."""
    source_node: int
    target_node: int
    weight: float = 1.0

    def __post_init__(self):
        if not isinstance(self.source_node, int) or not isinstance(self.target_node, int):
            raise TypeError("Node indices must be integers")
        if not isinstance(self.weight, (int, float)):
            raise TypeError("Weight must be numeric")


class NodeConnector:
    """
    Manages connectivity between nodes in the hybrid network.
    Handles both manual connection definitions and connectivity loaded from TVB files.
    """
    
    def __init__(self, num_nodes: Optional[int] = None, connectivity: Optional[Connectivity] = None):
        """
        Initialize connector with either number of nodes or TVB connectivity.
        
        Parameters
        ---------- 
        num_nodes : int, optional
            Number of nodes if manually creating connections
        connectivity : Connectivity, optional
            TVB connectivity object with pre-defined weights and tract lengths
            
        Raises
        ------ 
        ValueError
            If neither num_nodes nor connectivity is provided
        """
        if connectivity is not None:
            self.connectivity = connectivity
            self.num_nodes = connectivity.number_of_regions
            # Use the connectivity's weights as our connectivity matrix
            self._connectivity_matrix = connectivity.weights.copy()
        elif num_nodes is not None:
            if num_nodes < 1:
                raise ValueError("Number of nodes must be positive")
            self.num_nodes = num_nodes
            self.routes: Dict[int, Dict[int, float]] = {}
            self._connectivity_matrix = None
        else:
            raise ValueError("Either num_nodes or connectivity must be provided")

    def add_connection(self, source: int, target: int, weight: float = 1.0) -> None:
        """
        Add a directional connection between nodes with specified weight.
        Only works with manual configuration (not with loaded connectivity).
        
        Parameters
        ---------- 
        source : int
            Source node index (0-based)
        target : int
            Target node index (0-based)
        weight : float, optional
            Connection weight/strength (default: 1.0)
            
        Raises
        ------ 
        ValueError
            If connectivity was loaded from file or invalid indices/weights are provided
        """
        if hasattr(self, "connectivity"):
            raise ValueError("Connectivity is loaded from file; manual connections are disabled")
            
        if not (0 <= source < self.num_nodes and 0 <= target < self.num_nodes):
            raise ValueError(f"Node indices must be between 0 and {self.num_nodes - 1}")
        if not np.isfinite(weight):
            raise ValueError("Connection weight must be finite")
            
        if source not in self.routes:
            self.routes[source] = {}
        self.routes[source][target] = float(weight)
        self._connectivity_matrix = None  # Force recomputation

    def get_connectivity_matrix(self) -> np.ndarray:
        """
        Get the connectivity matrix representing weights between all nodes.
        
        Returns
        ------- 
        numpy.ndarray
            Matrix of connection weights, shape (num_nodes, num_nodes)
            where matrix[i,j] represents connection strength from j to i
        """
        if self._connectivity_matrix is None:
            if hasattr(self, "connectivity"):
                self._connectivity_matrix = self.connectivity.weights.copy()
            else:
                matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)
                for source, targets in self.routes.items():
                    for target, weight in targets.items():
                        # Use (target, source) ordering so that for node i we sum outputs from all nodes j.
                        matrix[target, source] = weight
                self._connectivity_matrix = matrix
        return self._connectivity_matrix.copy()

@runtime_checkable
class ModelWrapper(Protocol):
    """
    Protocol for model wrappers with standardized shape handling and improved 
    variable semantics. This interface ensures consistent handling of models with
    different state shapes and coupling mechanisms.
    """
    _nvar: int  # Number of state variables
    initial_state: np.ndarray  # Initial state values
    state_shape: Tuple[int, ...]  # Shape of state variables 
    coupling_shape: Tuple[int, ...]  # Shape of coupling variables
    
    # Optional but recommended attributes for better semantics
    state_variables: List[str]  # Names of state variables
    variables_of_interest: List[str]  # Variables to monitor by default
    state_variable_range: Dict[str, np.ndarray]  # Expected ranges for variables
    state_variable_boundaries: Dict[str, np.ndarray]  # Bounds for variables
    
    def get_state_shape(self) -> Tuple[int, ...]:
        """Return the native state shape for this model (e.g., (nvar,) or (nvar, 1)).""" 
        ...
        
    def validate_state(self, state: np.ndarray) -> np.ndarray:
        """
        Validate state shape and content, returning properly shaped state.
        
        Parameters
        ----------
        state : numpy.ndarray
            State to validate
            
        Returns
        -------
        numpy.ndarray
            Validated state with correct shape
        """
        ...
        
    def flatten_state(self, state: np.ndarray) -> np.ndarray:
        """
        Convert state from native shape to flat 1D array.
        
        Parameters
        ----------
        state : numpy.ndarray
            State in native shape
            
        Returns
        -------
        numpy.ndarray
            Flattened 1D array
        """
        ...
        
    def reshape_state(self, state: np.ndarray) -> np.ndarray:
        """
        Convert flat 1D array to native state shape.
        
        Parameters
        ----------
        state : numpy.ndarray
            Flattened state array
            
        Returns
        -------
        numpy.ndarray
            State in native model shape
        """
        ...
        
    def coupling_function(self, state: np.ndarray) -> np.ndarray:
        """
        Extract coupling outputs from state, returning standardized shape.
        
        Parameters
        ----------
        state : numpy.ndarray
            Current state in native shape
            
        Returns
        -------
        numpy.ndarray
            Coupling variables with shape matching coupling_shape
        """
        ...
        
    def process_channels(self, state: np.ndarray,
                         incoming_channels: Dict[str, float]) -> np.ndarray:
        """
        Process inputs and compute state derivatives.
        
        Parameters
        ----------
        state : numpy.ndarray
            Current state in native shape
        incoming_channels : Dict[str, float] or Dict[str, Dict]
            Input values for each channel, either as flat dictionary or structured by regions
            
        Returns
        -------
        numpy.ndarray
            State derivatives in native shape
        """
        ...
        
    def get_variable_metadata(self) -> Dict[str, Dict]:
        """
        Get metadata about variables (optional method).
        
        Returns
        ------- 
        Dict[str, Dict]
            Dictionary mapping variable names to metadata dictionaries.
            Each metadata dict may contain keys like 'description', 'unit', 'range', etc.
        """
        ...

@dataclass
class NodeConfig:
    """
    Configuration for a single node in the hybrid network.
    Defines the node's model wrapper and how its channels map to indices.
    """
    id: int  # Unique identifier for the node
    wrapper: ModelWrapper  # The model implementation
    channel_mapping: Dict[str, int]  # Maps channel names to indices, e.g., {"E": 0, "I": 1}

    def __post_init__(self):
        if not isinstance(self.id, int) or self.id < 0:
            raise ValueError("Node ID must be a non-negative integer")
        if not isinstance(self.channel_mapping, dict):
            raise TypeError("Channel mapping must be a dictionary")
        if not all(isinstance(k, str) and isinstance(v, int) for k, v in self.channel_mapping.items()):
            raise TypeError("Channel mapping must map strings to integers")

# =============================================================================
# 2. Hybrid TVB Model (Meta‑Model)
# =============================================================================

class HybridTvbModel(Model):
    """
    A TVB meta-model that aggregates multiple heterogeneous node models into a unified model.
    
    This model routes signals between nodes according to connectivity weights and handles
    the conversion between TVB's expected state format and each node's native format.
    
    Its dfun() method uses the node_index_ranges from the orchestrator and delegates to
    each wrapper's methods to reshape/flatten states. This design ensures the orchestrator
    never needs modification when adding models with new state shapes or coupling mechanisms.
    """
    @property
    def nvar(self):
        return self.number_of_state_variables

    def __init__(self, orchestrator, local_coupling: float = 0.0):
        super().__init__()
        self.orchestrator = orchestrator
        
        # Build maps to preserve variable semantics and proper ranges
        self.variable_names = []
        self.variable_map = {}  # Maps global index to (node_id, var_name)
        self.state_variable_range = {}
        self.state_variable_boundaries = {}
        self.variables_of_interest = []
        
        # Build coupling variable information from each node's wrapper.
        coupling_vars = []
        
        # Track current position in global state vector
        global_idx = 0
        
        # Process each node
        for i, node in enumerate(orchestrator.nodes):
            node_id = node.id
            start_idx, end_idx = orchestrator.node_index_ranges[i]
            
            # Get variable information from wrapper if available
            if hasattr(node.wrapper, 'state_variables'):
                node_var_names = node.wrapper.state_variables
            else:
                # Create generic names if unavailable
                node_var_names = [f"var_{j}" for j in range(node.wrapper._nvar)]
                
            # Get ranges information
            node_ranges = getattr(node.wrapper, 'state_variable_range', {})
            node_boundaries = getattr(node.wrapper, 'state_variable_boundaries', {})
            
            # Get variables of interest if available
            node_voi = getattr(node.wrapper, 'variables_of_interest', node_var_names)
            
            # Map local variable indices to global state indices
            var_idx = start_idx
            for var_name in node_var_names:
                # Create qualified name with node id for uniqueness
                qualified_name = f"node{node_id}_{var_name}"
                self.variable_names.append(qualified_name)
                self.variable_map[global_idx] = (node_id, var_name)
                
                # Get proper range or use default
                if var_name in node_ranges:
                    self.state_variable_range[qualified_name] = node_ranges[var_name].copy()
                else:
                    self.state_variable_range[qualified_name] = np.array([-100.0, 100.0])
                    
                # Set boundaries (keeping one-sided boundaries)
                if var_name in node_boundaries:
                    self.state_variable_boundaries[qualified_name] = node_boundaries[var_name].copy()
                else:
                    self.state_variable_boundaries[qualified_name] = self.state_variable_range[qualified_name].copy()
                    
                # Add to variables of interest if in node's voi
                if var_name in node_voi:
                    self.variables_of_interest.append(qualified_name)
                    
                global_idx += 1
                var_idx += 1
                
            # Process coupling vars
            if hasattr(node.wrapper, 'coupling_variables'):
                coupling_vars.extend(node.wrapper.coupling_variables)
            else:
                # Default: assume first two state indices for E and I
                coupling_vars.extend([0, 1])
                
        self.cvar = np.array(coupling_vars, dtype=int)
        self.gvar = np.array([0], dtype=int)
        self.stvar = np.array(coupling_vars, dtype=int)
        
        self.number_of_state_variables = orchestrator.total_nvar
        self.num_coupling_vars = sum(
            node.wrapper.num_coupling_vars if hasattr(node.wrapper, 'num_coupling_vars') else 2
            for node in orchestrator.nodes
        )
        
        # TVB-required attributes
        self._nvar = self.number_of_state_variables
        self._nintvar = self._nvar
        self.non_integrated_variables = []
        
        # Use the constructed maps for TVB variables
        self.state_variables = self.variable_names

    def dfun(self, state, coupling, local_coupling=0.0):
        """
        Compute the derivative of the entire network's state with improved shape handling.
        Supports both global coupling and region-based routing of neural activity.
        
        Parameters
        ---------- 
        state : numpy.ndarray
            State variables. Shape can be any of:
            - (state_variables, nodes, modes) for regular TVB calls
            - (state_variables, nodes) for simplified calls
            - (state_variables,) for single node, no modes
        coupling : numpy.ndarray
            Array of coupling variables from TVB's coupling functions
        local_coupling : float
            Local coupling strength (within-node coupling)
        
        Returns
        -------
        numpy.ndarray
            State derivatives with shape matching input state array
        """
        # Store original shape for output
        orig_shape = state.shape
        
        # Ensure consistent 2D shape for processing
        if state.ndim == 3:  # (variables, nodes, modes)
            state_2d = state.reshape(self.number_of_state_variables, -1)
        elif state.ndim == 1:  # (variables,)
            state_2d = state.reshape(-1, 1)
        else:
            state_2d = state  # Already 2D
            
        derivatives = np.zeros_like(state_2d)
        node_outputs = {}

        # First pass: compute outputs for coupling with shape validation
        for i, node_cfg in enumerate(self.orchestrator.nodes):
            start_idx, end_idx = self.orchestrator.node_index_ranges[i]
            
            # Extract node state slice
            node_flat_state = state_2d[start_idx:end_idx]
            
            # Validate slice size
            if node_flat_state.size != (end_idx - start_idx):
                raise ValueError(f"Node {node_cfg.id} state slice size {node_flat_state.size} "
                                f"doesn't match expected {end_idx - start_idx}")
            
            # Use wrapper's methods with validation
            try:
                node_state_native = node_cfg.wrapper.reshape_state(node_flat_state)
                out = node_cfg.wrapper.coupling_function(node_state_native)
                
                # Validate coupling output shape
                expected_shape = node_cfg.wrapper.coupling_shape
                if out.shape != expected_shape:
                    raise ValueError(f"Node {node_cfg.id} coupling output shape {out.shape} "
                                    f"doesn't match expected {expected_shape}")
                                    
                # Extract channel outputs
                node_outputs[i] = {
                    channel: float(out[idx]) if idx < out.size else 0.0
                    for channel, idx in node_cfg.channel_mapping.items()
                }
            except Exception as e:
                raise ValueError(f"Error processing node {node_cfg.id}: {str(e)}")

        # Second pass: compute derivatives with shape validation and region-based routing
        conn_matrix = self.orchestrator.connector.get_connectivity_matrix()
        
        # Check if region labels are available (from TVB connectivity)
        has_region_labels = hasattr(self.orchestrator.connector, "connectivity") and \
                           hasattr(self.orchestrator.connector.connectivity, "region_labels")
        
        for i, node_cfg in enumerate(self.orchestrator.nodes):
            start_idx, end_idx = self.orchestrator.node_index_ranges[i]
            
            # Initialize structure for incoming channels
            if has_region_labels:
                # Region-based structure: {region_name: {channel_name: value}}
                region_channels = {}
                # Also create flat channels for backward compatibility
                flat_channels = {channel: 0.0 for channel in node_cfg.channel_mapping}
                
                # Get region labels
                region_labels = self.orchestrator.connector.connectivity.region_labels
                
                # Group channels by region
                for j in range(self.orchestrator.num_nodes):
                    # Skip if no connection from j to i
                    weight = conn_matrix[i, j]
                    if weight == 0:
                        continue
                        
                    # Get region label for source node
                    src_region = region_labels[j]
                    
                    # Initialize region entry if needed
                    if src_region not in region_channels:
                        region_channels[src_region] = {
                            channel: 0.0 for channel in node_cfg.channel_mapping
                        }
                    
                    # Add weighted contributions from this source node
                    for channel, value in node_outputs[j].items():
                        weighted_val = value * weight
                        region_channels[src_region][channel] += weighted_val
                        flat_channels[channel] += weighted_val  # Also update flat channels
                
                # Create combined structure with both region-grouped and flat channels
                incoming_channels = {
                    "flat": flat_channels,
                    "regions": region_channels
                }
            else:
                # Traditional flat channel structure (backward compatibility)
                incoming_channels = {}
                for channel in node_cfg.channel_mapping:
                    source_values = np.array([node_outputs[j].get(channel, 0.0) 
                                             for j in range(self.orchestrator.num_nodes)])
                    weighted_sum = np.dot(conn_matrix[i, :], source_values)
                    incoming_channels[channel] = float(weighted_sum)

            # Process incoming signals
            node_flat_state = state_2d[start_idx:end_idx]
            node_state_native = node_cfg.wrapper.reshape_state(node_flat_state)
            
            try:
                dstate = node_cfg.wrapper.process_channels(node_state_native, incoming_channels)
                flat_dstate = node_cfg.wrapper.flatten_state(dstate)
                
                # Validate derivative size
                if flat_dstate.size != (end_idx - start_idx):
                    raise ValueError(f"Node {node_cfg.id} derivative size {flat_dstate.size} "
                                   f"doesn't match expected {end_idx - start_idx}")
                                   
                derivatives[start_idx:end_idx] = flat_dstate.reshape(-1, 1)
            except Exception as e:
                raise ValueError(f"Error computing derivatives for node {node_cfg.id}: {str(e)}")

        # Return derivatives in same shape as input state
        return derivatives.reshape(orig_shape)

# =============================================================================
# 3. Hybrid Orchestrator
# =============================================================================

class HybridOrchestrator(HasTraits):
    """
    Orchestrator for heterogeneous neural model networks in TVB.
    
    Manages the coordination between different neural model implementations,
    handles state conversions, and provides a unified interface to TVB's simulation framework.
    
    It relies on each wrapper to supply state shape, initialization, and reshaping methods.
    Node_index_ranges and the global state are computed dynamically—ensuring that new models 
    with different state shapes require no changes in the orchestrator code.
    """
    dt = Attr(
        field_type=float,
        default=0.1,
        doc="Integration time step (ms)",
    )
    local_coupling = Attr(
        field_type=float,
        default=0.0,
        doc="Local coupling strength"
    )

    def __init__(self, nodes: List[NodeConfig],
                 connector: Optional[NodeConnector] = None,
                 connectivity_file: Optional[str] = None,
                 **kwargs):
        """
        Initialize the orchestrator with nodes and connectivity information.
        
        Parameters
        ---------- 
        nodes : List[NodeConfig]
            List of node configurations
        connector : NodeConnector, optional
            Manually configured connector (ignored if connectivity_file is provided)
        connectivity_file : str, optional
            Path to TVB connectivity file to load
        **kwargs
            Additional parameters passed to parent class
            
        Raises
        ------ 
        ValueError
            If no nodes are provided or connectivity configuration is invalid
        """
        if not nodes:
            raise ValueError("At least one node must be provided")
            
        super().__init__(**kwargs)
        self.nodes = nodes
        self.num_nodes = len(nodes)
        
        # Load connectivity from file if provided
        if connectivity_file is not None:
            try:
                connectivity = load_connectivity_file(connectivity_file)
                if connectivity.number_of_regions != self.num_nodes:
                    raise ValueError(
                        f"Mismatch between connectivity regions ({connectivity.number_of_regions}) "
                        f"and provided nodes ({self.num_nodes})"
                    )
                self.connector = NodeConnector(connectivity=connectivity)
                print(f"Loaded connectivity from {connectivity_file} with {connectivity.number_of_regions} regions")
            except Exception as e:
                raise ValueError(f"Failed to load connectivity file: {str(e)}")
        elif connector is not None:
            if connector.num_nodes != self.num_nodes:
                raise ValueError(
                    f"Mismatch between connector nodes ({connector.num_nodes}) "
                    f"and provided nodes ({self.num_nodes})"
                )
            self.connector = connector
        else:
            self.connector = self._create_default_connector()
            
        self._init_node_indices()
        self._init_global_state()
        self._validate_configuration()

    def _create_default_connector(self) -> NodeConnector:
        connector = NodeConnector(self.num_nodes)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    connector.add_connection(i, j, weight=0.05)
        return connector

    def _validate_configuration(self) -> None:
        seen_ids = set()
        for node in self.nodes:
            if not hasattr(node.wrapper, '_nvar'):
                raise TypeError(f"Node {node.id}: wrapper must have '_nvar' attribute")
            if not hasattr(node.wrapper, 'initial_state'):
                raise TypeError(f"Node {node.id}: wrapper must have 'initial_state' attribute")
            if not hasattr(node.wrapper, 'process_channels'):
                raise TypeError(f"Node {node.id}: wrapper must implement 'process_channels'")
            if not hasattr(node.wrapper, 'coupling_function'):
                raise TypeError(f"Node {node.id}: wrapper must implement 'coupling_function'")
            if not hasattr(node.wrapper, 'get_state_shape'):
                raise TypeError(f"Node {node.id}: wrapper must implement 'get_state_shape'")
            if not hasattr(node.wrapper, 'reshape_state'):
                raise TypeError(f"Node {node.id}: wrapper must implement 'reshape_state'")
            if not hasattr(node.wrapper, 'flatten_state'):
                raise TypeError(f"Node {node.id}: wrapper must implement 'flatten_state'")
            if not hasattr(node.wrapper, 'coupling_shape'):
                raise TypeError(f"Node {node.id}: wrapper must have 'coupling_shape' attribute")
            if not node.channel_mapping:
                raise ValueError(f"Node {node.id}: channel mapping cannot be empty")
            if node.id in seen_ids:
                raise ValueError(f"Duplicate node ID: {node.id}")
            seen_ids.add(node.id)

    def _init_node_indices(self) -> None:
        """
        Dynamically compute node_index_ranges from each wrapper's state shape.
        
        This allows the orchestrator to work with any model without modification,
        regardless of the model's internal state representation.
        """
        self.node_index_ranges = []
        current_index = 0
        for node in self.nodes:
            state_shape = node.wrapper.get_state_shape()  # e.g., (nvar, 1)
            nvar = int(np.prod(state_shape))
            if nvar < 1:
                raise ValueError(f"Node {node.id}: Invalid state shape {state_shape}")
            self.node_index_ranges.append((current_index, current_index + nvar))
            current_index += nvar
        self.total_nvar = current_index

    def _init_global_state(self) -> None:
        """
        Build the global initial state vector by combining all node states.
        
        Queries each wrapper's initial_state and flattens it using the wrapper's 
        flatten_state() method with validation to ensure correct dimensionality.
        """
        self.global_state = np.zeros((self.total_nvar,), dtype=np.float64)
        
        # Fix: Use enumerate to get the index and node
        for i, node in enumerate(self.nodes):
            start, end = self.node_index_ranges[i]
            
            # Skip if no initial state provided
            if not hasattr(node.wrapper, 'initial_state') or node.wrapper.initial_state is None:
                continue
                
            try:
                # Use wrapper's own flattening method
                flat = node.wrapper.flatten_state(node.wrapper.initial_state)
                
                # Validate size
                expected_size = end - start
                if flat.size != expected_size:
                    raise ValueError(
                        f"Node {node.id} flattened state size {flat.size} "
                        f"does not match expected size {expected_size}"
                    )
                    
                # Assign to global state
                self.global_state[start:end] = flat
                
            except Exception as e:
                raise ValueError(f"Failed to initialize state for node {node.id}: {str(e)}")

    def run(self, simulation_length: float,
            integrator: Optional[IntegratorStochastic] = None,
            record_timeseries: bool = False,
            use_tvb_connectivity: bool = False) -> Optional[Dict]:
        """
        Run the hybrid simulation with properly handled monitor data.
        
        This method configures TVB's simulator with the hybrid model and runs
        the simulation, handling state conversions and data collection.
        
        Parameters
        ---------- 
        simulation_length : float
            Length of simulation in milliseconds
        integrator : IntegratorStochastic, optional
            TVB integrator to use for the simulation
        record_timeseries : bool, optional
            Whether to record and return timeseries data (may use more memory)
        use_tvb_connectivity : bool, optional
            Whether to use TVB connectivity for simulation instead of dummy connectivity
            
        Returns
        ------- 
        Optional[Dict]
            Dictionary with simulation results if record_timeseries is True, containing:
            - 'data': numpy.ndarray with recorded timeseries
            - 'variables': list of variable names
            - 'time': numpy.ndarray with time points
            
        Raises
        ------ 
        ValueError
            If simulation parameters are invalid
        RuntimeError
            If no data is collected during simulation
        """
        if simulation_length <= 0:
            raise ValueError("Simulation length must be positive")
        if integrator is None:
            raise ValueError("A TVB integrator must be provided")
            
        print("\nSimulation parameters:")
        print(f"- Length: {simulation_length}ms")
        print(f"- dt: {integrator.dt}ms")
        print(f"- Expected steps: {int(simulation_length/integrator.dt)}")
            
        # Instantiate the meta-model
        hybrid_model = HybridTvbModel(self, local_coupling=self.local_coupling)
        self._hybrid_model = hybrid_model
        print(f"\nModel configuration:")
        print(f"- Number of state variables: {hybrid_model.number_of_state_variables}")
        print(f"- Variable names: {hybrid_model.state_variables}")

        # Use shape (1, total_nvar, 1, 1) to match single aggregated node
        init_conditions = np.zeros((1, self.total_nvar, 1, 1))
        init_conditions[0, :, 0, 0] = self.global_state
        print(f"- Initial conditions shape: {init_conditions.shape}")
        print("- Using single aggregated node configuration")
        
        # Configure monitoring
        monitor_indices = np.arange(self.total_nvar, dtype=np.int64)
        # Set period to match integrator dt to get data at every timestep
        mon = monitors.Raw(variables_of_interest=monitor_indices, period=integrator.dt)
        print("\nMonitor configuration:")
        print(f"- Variables monitored: {self.total_nvar}")
        print(f"- Monitor period: {mon.period}ms")

        # Configure simulation with appropriate connectivity
        connectivity = None
        if use_tvb_connectivity and hasattr(self.connector, "connectivity"):
            connectivity = self.connector.connectivity
            print("\nUsing loaded TVB connectivity for simulation")
        else:
            connectivity = self._setup_dummy_connectivity()
            print("\nUsing dummy connectivity for simulation")
            
        sim = simulator.Simulator(
            model=hybrid_model,
            connectivity=connectivity,
            coupling=coupling.Linear(a=np.zeros((2, 1))),
            integrator=integrator,
            monitors=(mon,),
            simulation_length=simulation_length,
            initial_conditions=init_conditions
        ).configure()

        # Improved data collection with proper handling of monitor output
        raw_data = []
        time_points = []
        step_count = 0
        
        print("\nStarting simulation loop...")
        try:
            for raw_tuple in sim.run():
                if raw_tuple is not None:
                    # Proper unpacking of monitor data tuple
                    time_point, data_point = raw_tuple
                    
                    step_count += 1
                    
                    # Debug the first step and periodic steps
                    if step_count == 1 or step_count % 1000 == 0:
                        print(f"Step {step_count}: raw_tuple[0]={time_point}, data_point.shape={data_point.shape}")
                    
                    # Ensure data has proper shape
                    if data_point.size == 0:
                        print(f"Warning: Empty data at step {step_count}")
                        continue
                        
                    # If data is scalar or 1D, reshape to expected dimensions
                    if np.isscalar(data_point) or data_point.ndim == 0:
                        print(f"Warning: Scalar data at step {step_count}, reshaping")
                        data_point = np.array([[data_point]])
                    elif data_point.ndim == 1:
                        data_point = data_point.reshape(-1, 1, 1)
                        
                    # Collect data and time
                    raw_data.append(data_point)
                    time_points.append(time_point)

            if not raw_data:
                raise RuntimeError("No data collected from simulation")

            print(f"\nSimulation complete:")
            print(f"- Steps processed: {step_count}")
            print(f"- Data points collected: {len(raw_data)}")

            # Process the collected data
            try:
                # Check shapes before stacking
                data_shapes = [d.shape for d in raw_data]
                print(f"- First few data shapes: {data_shapes[:5]}")
                
                # Stack data with explicit shape handling
                timeseries = np.concatenate(raw_data, axis=0)
                print(f"- Stacked data shape: {timeseries.shape}")
                
                # Update global state from last time step
                if timeseries.shape[0] > 0:
                    # Extract the last state, properly handling dimensions
                    last_state = timeseries[-1]
                    if last_state.ndim > 1:
                        last_state = last_state.reshape(-1)
                    self.global_state = last_state
                    
            except Exception as e:
                print(f"Error processing timeseries: {str(e)}")
                import traceback
                print(traceback.format_exc())
                raise

            # Return results if requested
            if record_timeseries:
                return {
                    'data': timeseries,
                    'variables': hybrid_model.state_variables,
                    'time': np.array(time_points) if time_points else np.arange(0, simulation_length, integrator.dt)[:len(raw_data)]
                }
                
        except Exception as e:
            print(f"\nERROR during simulation: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
        return None

    def _setup_dummy_connectivity(self) -> Connectivity:
        """
        Create a single aggregated node TVB connectivity for compatibility.
        
        Since TVB requires a connectivity object but our hybrid orchestrator handles
        the actual node connections internally, we create a minimal dummy connectivity
        with a single aggregated node to satisfy TVB's requirements.
        
        Returns
        -------
        Connectivity
            TVB connectivity object with a single aggregated node
        """
        # Use a single aggregated node instead of self.num_nodes
        n = 1  # Single aggregated node
        dummy_weights = np.zeros((n, n))
        dummy_conn = Connectivity(
            weights=dummy_weights,
            tract_lengths=dummy_weights,
            region_labels=np.array(["Aggregated"], dtype='<U128'),
            centres=np.zeros((n, 3)),
            areas=np.ones((n,)),
            orientations=np.zeros((n, 3)),
            cortical=np.array([True] * n),
            hemispheres=np.array([False] * n),
            number_of_regions=n
        )
        dummy_conn.configure()
        print(f"\nTVB Dummy Connectivity:")
        print(f"- Using single aggregated node")
        print(f"- Region labels: {dummy_conn.region_labels}")
        return dummy_conn