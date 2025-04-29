import numpy as np
from typing import List
import tvb.basic.neotraits.api as t
from tvb.simulator.models import Model
from tvb.simulator.integrators import Integrator
from tvb.simulator.monitors import Monitor
from .recorder import Recorder
# Removed: from .projection import Projection - breaks circular import


class InternalProjection(t.HasTraits):
    """
    Defines internal coupling within a Subnetwork.

    Maps coupling variables back onto the same subnetwork, potentially
    transforming them via weights and scaling. Does not involve mode mapping
    as source and target modes are inherently the same.
    """
    source_cvar = t.NArray(dtype=np.int_)  # Array of source coupling variable indices
    target_cvar = t.NArray(dtype=np.int_)  # Array of target coupling variable indices
    scale: float = t.Float(default=1.0)
    weights: np.ndarray = t.NArray(dtype=np.float_)
    # NOTE: No mode_map needed as source/target modes are the same

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Convert scalar indices to arrays for unified handling
        self.source_cvar = np.atleast_1d(self.source_cvar)
        self.target_cvar = np.atleast_1d(self.target_cvar)

    def apply(self, tgt, src):
        """Apply the internal projection to compute coupling.

        Handles three cases based on source/target cvar sizes:
        a) One source to many targets (broadcasting)
        b) Many sources to one target (reduction/summation)
        c) Equal number of sources to targets (element-wise)

        Parameters
        ----------
        tgt : ndarray
            Target state array (internal coupling) to modify
        src : ndarray
            Source state array (subnetwork state)
        """
        # Compute base coupling for each source variable
        # No mode_map needed for internal projection
        gx_array = self.scale * self.weights @ src[self.source_cvar]
        # TODO: Add time delays if relevant for internal projections

        if self.source_cvar.size == 1:
            # Case a: broadcast single source to all targets
            tgt[self.target_cvar] += gx_array[0]
        elif self.target_cvar.size == 1:
            # Case b: sum all sources to single target
            tgt[self.target_cvar] += gx_array.sum(axis=0)
        else:
            # Case c: element-wise mapping
            tgt[self.target_cvar] += gx_array


class Subnetwork(t.HasTraits):
    """A subnetwork that can be reused across different models.
    
    A subnetwork represents a group of brain regions that share the same
    dynamical model and integration scheme. It is designed to be reusable
    across different models by separating the dynamics from the connectivity.
    
    Attributes
    ----------
    name : str
        Name of the subnetwork
    model : Model
        Dynamical model for the subnetwork
    scheme : Integrator
        Integration scheme for the subnetwork
    nnodes : int
        Number of nodes in the subnetwork
    monitors : list
        List of recorders for monitoring simulation data
    """

    name: str = t.Attr(str)
    model: Model = t.Attr(Model)
    scheme: Integrator = t.Attr(Integrator)
    monitors: List[Recorder] # NOTE: Consider making this t.List(of=Recorder) too
    projections: List[InternalProjection] = t.List(of=InternalProjection)
    nnodes: int = t.Int()

    def __init__(self, **kwargs):
        self.monitors = []
        super().__init__(**kwargs)

    def configure(self):
        """Configure the subnetwork's model.
        
        Returns
        -------
        self
            The configured subnetwork
        """
        self.model.configure()
        return self

    def add_monitor(self, monitor: Monitor):
        """Add a monitor to record simulation data.
        
        Parameters
        ----------
        monitor : Monitor
            The monitor to add
        """
        monitor._config_dt(self.scheme.dt)
        if hasattr(monitor, '_config_stock'):
            monitor._config_stock(len(self.model.variables_of_interest),
                                self.nnodes,
                                self.model.number_of_modes)
        self.monitors.append(Recorder(monitor=monitor))

    @property
    def var_shape(self) -> tuple[int]:
        """Shape of the state variables.
        
        Returns
        -------
        tuple
            (number of nodes, number of modes)
        """
        return self.nnodes, self.model.number_of_modes

    def zero_states(self) -> np.ndarray:
        """Create an array of zeros for the state variables.
        
        Returns
        -------
        ndarray
            Array of zeros with shape (nvar, nnodes, modes)
        """
        return np.zeros((self.model.nvar, ) + self.var_shape)

    def zero_cvars(self) -> np.ndarray:
        """Create an array of zeros for the coupling variables.
        
        Returns
        -------
        ndarray
            Array of zeros with shape (ncvar, nnodes, modes)
        """
        return np.zeros((self.model.cvar.size, ) + self.var_shape)

    def cfun(self, x: np.ndarray) -> np.ndarray:
        """Compute internal coupling within the subnetwork.

        Parameters
        ----------
        x : ndarray
            Current state of the subnetwork

        Returns
        -------
        ndarray
            Internal coupling variables for the subnetwork
        """
        internal_c = self.zero_cvars()
        # Iterate through InternalProjection instances
        for p in self.projections:
            # Apply the internal projection
            p.apply(internal_c, x)
        return internal_c

    def step(self, step, x, c):
        """Take a single integration step.
        Parameters
        ----------
        step : int
            Current simulation step
        x : ndarray
            Current state
        c : ndarray
            Current coupling variables
            
        Returns
        -------
        ndarray
            Next state after integration
        """
        # Calculate internal coupling first
        internal_c = self.cfun(x)
        # Add internal coupling to external coupling
        total_c = c + internal_c
        # Integrate
        nx = self.scheme.scheme(x, self.model.dfun, total_c, 0, 0)
        # Record monitored variables
        for monitor in self.monitors:
            monitor.record(step, self.model.observe(nx))
        return nx


class Stim(Subnetwork):
    """Stimulator adapted for hybrid cases"""
    # classic use is non-modal:
    # stimulus[self.model.stvar, :, :] = \
    #   self.stimulus(stim_step).reshape((1, -1, 1))
    pass 
