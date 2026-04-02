import numpy as np
from typing import List
import tvb.basic.neotraits.api as t
from tvb.simulator.models import Model
from tvb.simulator.integrators import Integrator
from tvb.simulator.monitors import Monitor, AfferentCoupling, AfferentCouplingTemporalAverage

from .recorder import Recorder
from .intra_projection import IntraProjection


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
    monitors: List[Recorder] = t.List(of=Recorder)
    projections: List[IntraProjection] = t.List(of=IntraProjection)
    nnodes: int = t.Int()

    def configure(self):
        """Configure the subnetwork's model and history buffer.

        Returns
        -------
        self
            The configured subnetwork
        """
        self.model.configure()
        for p in self.projections:
            p: IntraProjection
            # Set model reference for cvar name resolution (if using named cvars)
            if hasattr(p, "set_model_for_cvar_resolution"):
                p.set_model_for_cvar_resolution(self.model)
            p.configure_buffer(
                self.model.cvar.size, self.nnodes, self.model.number_of_modes
            )
        return self

    def add_monitor(self, monitor: Monitor):
        """Add a monitor to record simulation data.

        Parameters
        ----------
        monitor : Monitor
            The monitor to add
        """
        # NOTE default for list is a tuple
        if isinstance(self.monitors, tuple):
            self.monitors = []
        monitor._config_dt(self.scheme.dt)
        if hasattr(monitor, "_config_stock"):
            # AfferentCoupling monitors track coupling variables, not state variables
            if isinstance(monitor, (AfferentCoupling, AfferentCouplingTemporalAverage)):
                num_vars = len(self.model.cvar)
            else:
                num_vars = len(self.model.variables_of_interest)
            monitor._config_stock(
                num_vars,
                self.nnodes,
                self.model.number_of_modes,
            )
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
        return np.zeros((self.model.nvar,) + self.var_shape)

    def zero_cvars(self) -> np.ndarray:
        """Create an array of zeros for the coupling variables.

        Returns
        -------
        ndarray
            Array of zeros with shape (ncvar, nnodes, modes)
        """
        return np.zeros((self.model.cvar.size,) + self.var_shape)

    def cfun(self, step: int, x: np.ndarray) -> np.ndarray:
        """Compute internal coupling within the subnetwork.

        Parameters
        ----------
        step : int
            Current simulation step index.
        x : ndarray
            Current state of the subnetwork (shape [nvar, nnodes, modes]).

        Returns
        -------
        ndarray
            Internal coupling variables for the subnetwork
        """
        internal_c = self.zero_cvars()  # Shape (ncvar, nnodes, nmodes)

        if not self.projections:
            return internal_c  # No internal projections to apply

        # Update history buffers for all internal projections first
        for p in self.projections:
            p.update_buffer(x, step)

        # Apply internal projections using their own buffers
        for p in self.projections:
            p.apply(internal_c, step, self.model.number_of_modes)

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
        # Calculate internal coupling first, passing the current step
        internal_c = self.cfun(step, x)
        # Add internal coupling to external coupling
        total_c = c + internal_c
        nx = self.scheme.scheme(x, self.model.dfun, total_c, 0.0, 0.0)
        # Record monitored variables
        for monitor in self.monitors:
            # AfferentCoupling monitors need coupling data, not state
            if isinstance(monitor.monitor, (AfferentCoupling, AfferentCouplingTemporalAverage)):
                # Use coupling variables for AfferentCoupling monitors
                # total_c is (ncvar, nnodes, modes), matching node_coupling shape
                monitor.record(step, total_c)
            else:
                # Use observed state for other monitors
                monitor.record(step, self.model.observe(nx))
        return nx


class Stim(Subnetwork):
    """Stimulator adapted for hybrid cases"""

    # classic use is non-modal:
    # stimulus[self.model.stvar, :, :] = \
    #   self.stimulus(stim_step).reshape((1, -1, 1))
    pass
