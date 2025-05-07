import math
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
        self.num_samples = None
        super().__init__(**kwargs)

    def configure(self, simulation_length):
        """Configure recorder based on simulation parameters.
        
        Parameters
        ----------
        simulation_length : float
            Total simulation time in milliseconds
        """
        # Calculate expected number of samples
        self.num_samples = int(math.ceil(simulation_length / self.monitor.period))
        self._current_idx = 0

    def _allocate_arrays(self, sample_shape):
        """Allocate arrays based on first sample shape.
        
        Parameters
        ----------
        sample_shape : tuple
            Shape of the first sample from monitor
        """
        if len(sample_shape) == 4 and sample_shape[0] == 1:
            sample_shape = sample_shape[1:]
        self.samples = np.zeros((self.num_samples,) + sample_shape, dtype=np.float32)
        self.times = np.zeros(self.num_samples, dtype=np.float32)

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
            print(id(self), self._current_idx, t, y[0,0,0])
            # Lazy allocation on first sample
            if self.samples is None:
                self._allocate_arrays(y.shape)
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
