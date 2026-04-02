# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
In-memory recorder for hybrid simulator monitors.

Wraps a TVB :class:`~tvb.simulator.monitors.Monitor` and accumulates its
output into pre-allocated NumPy arrays during a simulation run.  After the
simulation, call :meth:`Recorder.to_arrays` to retrieve the collected
``(times, samples)`` pair.
"""

import math
import numpy as np

import tvb.basic.neotraits.api as t
from tvb.simulator.monitors import Monitor


class Recorder(t.HasTraits):
    """In-memory accumulator for a TVB monitor during a hybrid simulation.

    :class:`Recorder` wraps any TVB :class:`~tvb.simulator.monitors.Monitor`
    and stores every sample it emits into pre-allocated NumPy arrays.
    Arrays are allocated lazily on the first recorded sample so that the
    sample shape does not need to be known in advance.

    After the simulation, use :meth:`to_arrays` to retrieve the collected
    data as a ``(times, samples)`` tuple, trimmed to the number of samples
    actually recorded.

    Attributes
    ----------
    monitor : Monitor
        The TVB monitor whose output is accumulated.
    times : ndarray of float32 or None
        Recorded simulation times, shape ``(num_samples,)``.  ``None`` until
        the first sample is received.
    samples : ndarray of float32 or None
        Recorded state samples, shape ``(num_samples, ...)``.  ``None`` until
        the first sample is received.

    Examples
    --------
    >>> from tvb.simulator.monitors import TemporalAverage
    >>> rec = Recorder(monitor=TemporalAverage(period=1.0))
    >>> rec.configure(simulation_length=100.0)
    >>> times, samples = rec.to_arrays()
    """

    monitor: Monitor = t.Attr(Monitor)

    def __init__(self, **kwargs):
        self.times = None
        self.samples = None
        self._current_idx = 0
        self.num_samples = None
        super().__init__(**kwargs)

    def configure(self, simulation_length):
        """Pre-allocate index tracking for the expected number of samples.

        Computes ``num_samples`` from ``simulation_length`` and
        ``monitor.period``, then resets the internal write index.
        Array memory is not allocated here; it is deferred until the first
        call to :meth:`record`.

        Parameters
        ----------
        simulation_length : float
            Total simulation duration in milliseconds.
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
        """Offer a simulation step to the monitor and store any emitted sample.

        Delegates to ``monitor.record(step, state)``.  If the monitor returns
        a ``(time, data)`` pair, the values are appended to :attr:`times` and
        :attr:`samples`.  Arrays are allocated lazily on the first sample.

        Parameters
        ----------
        step : int
            Current simulation step index (0-based).
        state : array_like
            Full state array of the subnetwork at the current step.
        """
        ty = self.monitor.record(step, state)
        if ty is not None:
            t, y = ty
            # Lazy allocation on first sample
            if self.samples is None:
                self._allocate_arrays(y.shape)
            self.times[self._current_idx] = t
            self.samples[self._current_idx] = y
            self._current_idx += 1

    @property
    def shape(self):
        """Shape of the full pre-allocated samples array.

        Returns
        -------
        tuple or None
            Shape of :attr:`samples`, or ``None`` if no data has been
            recorded yet.
        """
        if self.samples is None:
            return None
        return self.samples.shape

    def to_arrays(self):
        """Return the recorded data trimmed to the samples actually collected.

        Returns
        -------
        times : ndarray of float32, shape (n,)
            Simulation times at which samples were recorded.
        samples : ndarray of float32, shape (n, ...)
            Corresponding state samples from the monitor.
        """
        return self.times[: self._current_idx], self.samples[: self._current_idx]
