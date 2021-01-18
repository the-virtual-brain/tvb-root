# -*- coding: utf-8 -*-

import numpy

from tvb.simulator.history import BaseHistory
from tvb.simulator.descriptors import Dim, NDArray


class CosimHistory(BaseHistory):

    """Class for cosimulation history implementation.
       It stores the whole TVB state for the co-simulation synchronization time.
       The synchronization time has to be shorter than the maximum delay
       (it usually is equal to the minimum delay of coupling between the co-simulators).
       It is a DenseHistory since the whole state has to be stored for all delays."""

    n_time, n_var, n_node, n_mode = Dim(), Dim(), Dim(), Dim()

    # Overwritting attributes not needed here:
    weights = None
    delays = None
    cvars = None

    buffer = NDArray(('n_time', 'n_var', 'n_node', 'n_mode'), float, read_only=False)

    @property
    def nbytes(self):
        return self.buffer.nbytes

    def __init__(self,  n_time, n_var, n_node, n_mode):
        self.n_time = n_time  # state buffer has n past steps and the current one
        self.n_var = n_var
        self.n_node = n_node
        self.n_mode = n_mode
        self.buffer[:] = numpy.NAN

    def initialize(self, history, current_step=0):
        """Initialize CosimHistory from the TVB history which is assumed already configured.
        """
        for i_step, step in enumerate(range(current_step, current_step+self.n_time)):
            self.buffer[i_step] = history.query(step)[0]

    def update(self, step, new_state):
        """This method will update the CosimHistory state buffer
           with the whole TVB state for a specific time step."""
        self.buffer[step % self.n_time] = new_state

    def query(self, step):
        """This method returns the whole TVB current_state
           by querying the CosimHistory state buffer for a time step."""
        return self.buffer[step % self.n_time]

    @classmethod
    def from_simulator(cls, sim):
        inst = cls(sim.synchronization_n_step,
                   sim.model.nvar,
                   sim.number_of_nodes,
                   sim.model.number_of_modes)
        inst.initialize(sim.history, sim.current_step)
        return inst

    def update_state_from_cosim(self, steps, new_states, vois, proxy_inds):
        """This method will update the CosimHistory state buffer from input from the other co-simulator, for
           - the state variables with indices vois,
           - the region nodes with indices proxy_inds,
           - and for the specified time steps."""
        self.buffer[(steps % self.n_time)[:, None, None, None],
                    vois[None, :, None, None],
                    proxy_inds[None, None, :, None],
                    numpy.arange(self.n_mode)[None, None, None, :]] = new_states
