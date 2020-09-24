# -*- coding: utf-8 -*-
import numpy
from tvb.simulator.descriptors import Dim, NDArray
from tvb.simulator.history import DenseHistory


class CosimHistory(DenseHistory):

    """Class for cosimulation history implementation.
       It stores the whole TVB state for the co-simulation synchronization time.
       The synchronization time has to be shorter than the maximum delay
       (it usually is equal to the minimum delay of coupling between the co-simulators).
       It is a DenseHistory since the whole state has to be stored for all delays."""

    n_time, n_node, n_var, n_mode = Dim(), Dim(), Dim(), Dim()

    buffer = NDArray(('n_time', 'n_var', 'n_node', 'n_mode'), 'f', read_only=False)

    def __init__(self, weights, delays, cvars, n_mode, bound_and_clamp, n_var=None):
        super(CosimHistory, self).__init__(weights, delays, cvars, n_mode)
        self.bound_and_clamp = bound_and_clamp
        if n_var is None:
            # Assuming all state variables are also coupling variables:
            self.n_var = self.n_cvar
        else:
            # Not all state variables are coupling variables in the general case.
            self.n_var = n_var
            # We need to correct the cvar indices:
            na = numpy.newaxis
            self.es_icvar = numpy.array(self.cvars)[na, :, na]

    def initialize(self, init):
        """Initialize CosimHistory from the initial condition."""
        self.buffer = init[:self.n_time]

    def query_state(self, step):
        """This method returns the whole TVB current_state
           by querying the CosimHistory buffer for a time step."""
        return self.buffer[(step - 1) % self.n_time]

    def update(self, step, new_state):
        """This method will update the CosimHistory buffer with the whole TVB state for a specific time step."""
        self.buffer[step % self.n_time] = new_state

    def update_from_cosim(self, steps, new_states, vois, proxy_inds):
        """This method will update the CosimHistory buffer from input from the other co-simulator, for
           - the state variables with indices vois,
           - the region nodes with indices proxy_inds,
           - and for the specified time steps."""
        for step, new_state in zip(steps, new_states):
            self.buffer[step % self.n_time, vois, proxy_inds] = new_state
            self.bound_and_clamp(self.buffer[step % self.n_time])
