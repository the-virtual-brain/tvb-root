# -*- coding: utf-8 -*-

from tvb.simulator.descriptors import StaticAttr, Dim, NDArray
import numpy

class CosimHistory(StaticAttr):

    """Class for cosimulation history implementation.
       It stores the whole TVB state for the co-simulation synchronization time.
       The synchronization time has to be shorter than the maximum delay
       (it usually is equal to the minimum delay of coupling between the co-simulators).
       It is a DenseHistory since the whole state has to be stored for all delays."""

    n_time, n_node, n_var, n_mode = Dim(), Dim(), Dim(), Dim()

    state_buffer = NDArray(('n_time', 'n_var', 'n_node', 'n_mode'), float, read_only=False)

    def __init__(self,  n_time, n_var, n_node, n_mode):
        self.n_time = n_time  # state buffer has n past steps and the current one
        self.n_var = n_var
        self.n_node = n_node
        self.n_mode = n_mode
        self.state_buffer[:]=numpy.NAN

    def initialize(self, init_state):
        """Initialize CosimHistory from the initial condition."""
        self.state_buffer[:] = init_state[:self.n_time]

    def update_state(self, step, new_state):
        """This method will update the CosimHistory state buffer
           with the whole TVB state for a specific time step."""
        self.state_buffer[step % self.n_time] = new_state

    def update_state_from_cosim(self, steps, new_states, vois, proxy_inds):
        """This method will update the CosimHistory state buffer from input from the other co-simulator, for
           - the state variables with indices vois,
           - the region nodes with indices proxy_inds,
           - and for the specified time steps."""
        index = [[] for i in range(4)]
        # get the index to update #TODO need to be optimize
        for i in steps %self.n_time:
            for j in vois:
                for k in proxy_inds:
                    for l in range(self.state_buffer.shape[3]):
                        index[0].append(i)
                        index[1].append(j)
                        index[2].append(k)
                        index[3].append(l)
        shape = self.state_buffer[tuple(index)].shape
        self.state_buffer[tuple(index)] = new_states.reshape(shape)


    def query_state(self, step):
        """This method returns the whole TVB current_state
           by querying the CosimHistory state buffer for a time step."""
        return self.state_buffer[step % self.n_time]
