# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

"""
.. moduleauthor:: Lionel Kusch <lkusch@thevirtualbrain.org>
.. moduleauthor:: Dionysios Perdikis <dionperd@gmail.com>
"""

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
        self.n_time = n_time  # state buffer has n past steps including the current one
        self.n_var = n_var
        self.n_node = n_node
        self.n_mode = n_mode
        self.buffer[:] = numpy.NAN

    def initialize(self, history, voi, current_step=0):
        """Initialize CosimHistory from the TVB history which is assumed already configured.
        """
        for i_step, step in enumerate(range(current_step, current_step+self.n_time)):
            self.buffer[i_step, voi[None, :,]] = history.query(step)[0]

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
        inst.initialize(sim.history, sim.voi, sim.current_step)
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
