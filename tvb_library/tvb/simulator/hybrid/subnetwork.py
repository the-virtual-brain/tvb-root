import numpy as np
from typing import List
import tvb.basic.neotraits.api as t
from tvb.simulator.models import Model
from tvb.simulator.integrators import Integrator
from tvb.simulator.monitors import Monitor
from .recorder import Recorder


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
    monitors: List[Recorder]
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
        nx = self.scheme.scheme(x, self.model.dfun, c, 0, 0)
        for monitor in self.monitors:
            monitor.record(step, self.model.observe(nx))
        return nx


class Stim(Subnetwork):
    """Stimulator adapted for hybrid cases"""
    # classic use is non-modal:
    # stimulus[self.model.stvar, :, :] = \
    #   self.stimulus(stim_step).reshape((1, -1, 1))
    pass 