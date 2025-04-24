import numpy as np
import tvb.basic.neotraits.api as t
from tvb.simulator.monitors import Monitor


class Recorder(t.HasTraits):
    """Records simulation data from a monitor.
    
    This class acts as a wrapper around a Monitor to store simulation data
    in memory rather than directly to disk.
    
    Attributes
    ----------
    monitor : Monitor
        The monitor to record data from
    times : ndarray
        Array of time points where data was recorded
    samples : ndarray
        Array of recorded state samples
    """

    monitor: Monitor = t.Attr(Monitor)

    def __init__(self, **kwargs):
        self.times = None
        self.samples = None
        self._current_idx = 0
        super().__init__(**kwargs)

    def configure(self, simulation_length):
        """Configure recorder arrays based on simulation parameters.
        
        Parameters
        ----------
        simulation_length : float
            Total simulation time in milliseconds
        """
        # Calculate number of samples based on simulation length and monitor period
        self.num_samples = int(simulation_length / self.monitor.period)
        
        # Get sample shape from monitor configuration
        if hasattr(self.monitor, 'voi') and self.monitor.voi is not None:
            if isinstance(self.monitor.voi, slice):
                # For slice, calculate length based on stock shape
                start = self.monitor.voi.start or 0
                stop = self.monitor.voi.stop or self.monitor._stock.shape[0]
                step = self.monitor.voi.step or 1
                num_vars = len(range(start, stop, step))
            else:
                num_vars = len(self.monitor.voi)
        else:
            # Default to using stock shape if voi not configured
            num_vars = self.monitor._stock.shape[0]
            
        num_nodes = self.monitor._stock.shape[1] 
        num_modes = self.monitor._stock.shape[2]
        
        # Preallocate arrays
        self.times = np.zeros(self.num_samples, dtype=np.float32)
        self.samples = np.zeros((self.num_samples, num_vars, num_nodes, num_modes), dtype=np.float32)
        self._current_idx = 0

    def record(self, step, state):
        """Record a state sample if the monitor indicates it should be recorded.
        
        Parameters
        ----------
        step : int
            Current simulation step
        state : array_like
            Current state of the system
        """
        ty = self.monitor.record(step, state)
        if ty is not None:
            t, y = ty
            self.times[self._current_idx] = t
            self.samples[self._current_idx] = y
            self._current_idx += 1

    @property
    def shape(self):
        """Shape of the recorded data.
        
        Returns
        -------
        tuple
            Shape of the recorded data array
        """
        if self.samples is None:
            return None
        return self.samples.shape

    def to_arrays(self):
        """Get recorded data as numpy arrays.
        
        Returns
        -------
        tuple
            (times, samples) as numpy arrays
        """
        return self.times, self.samples