# -*- coding: utf-8 -*-

from tvb.simulator.descriptors import StaticAttr, Dim, NDArray


class CosimHistory(StaticAttr):

    """Class for cosimulation history implementation.
       It stores the whole TVB state for the co-simulation synchronization time.
       The synchronization time has to be shorter than the maximum delay
       (it usually is equal to the minimum delay of coupling between the co-simulators).
       It is a DenseHistory since the whole state has to be stored for all delays."""

    n_time, n_node, n_var, n_cvar, n_mode = Dim(), Dim(), Dim(), Dim(), Dim()

    state_buffer = NDArray(('n_time', 'n_var', 'n_node', 'n_mode'), 'f', read_only=False)

    def __init__(self,  n_time, n_var, n_node, n_mode):
        self.n_time = n_time + 1  # state buffer has n past steps and the current one
        self.n_ctime = 2*self.n_time  # coupling buffer has n-1 past steps, the current one, and n future steps
        self.n_var = n_var
        self.n_node = n_node
        self.n_mode = n_mode

    def initialize(self, init_state):
        """Initialize CosimHistory from the initial condition."""
        self.state_buffer = init_state[:self.n_time]

    def update_state(self, step, new_state):
        """This method will update the CosimHistory state buffer
           with the whole TVB state for a specific time step."""
        self.state_buffer[step % self.n_time] = new_state

    def update_state_from_cosim(self, steps, new_states, vois, proxy_inds):
        """This method will update the CosimHistory state buffer from input from the other co-simulator, for
           - the state variables with indices vois,
           - the region nodes with indices proxy_inds,
           - and for the specified time steps."""
        for step, new_state in zip(steps, new_states):
            self.state_buffer[step % self.n_time, vois, proxy_inds] = new_state

    def query_state(self, step):
        """This method returns the whole TVB current_state
           by querying the CosimHistory state buffer for a time step."""
        return self.state_buffer[(step - 1) % self.n_time]
