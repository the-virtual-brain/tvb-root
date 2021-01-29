# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)

"""
This is the module responsible for co-simulation of TVB with spiking simulators.
It inherits the Simulator class.
.. moduleauthor:: Lionel Kusch <lkusch@thevirtualbrain.org>
.. moduleauthor:: Dionysios Perdikis <dionperd@gmail.com>
"""

import numpy

from tvb.basic.neotraits.api import Attr, NArray, Float, List
from tvb.simulator.common import iround
from tvb.simulator.simulator import Simulator, math

from tvb.contrib.cosimulation.cosim_history import CosimHistory
from tvb.contrib.cosimulation.cosim_monitors import CosimMonitor


class CoSimulator(Simulator):

    exclusive = Attr( #Todo need to test it
        field_type=bool,
        default=False, required=False,
        doc="1, when the proxy nodes substitute TVB nodes and their mutual connections should be removed.")

    voi = NArray(
        dtype=int,
        label="Cosimulation model state variables' indices",
        doc=("Indices of model's variables of interest (VOI) that"
             "should be updated (i.e., overwriten) during cosimulation."),
        default=numpy.asarray([], dtype=numpy.int),
        required=True)

    proxy_inds = NArray(
        dtype=numpy.int,
        label="Indices of TVB proxy nodes",
        default=numpy.asarray([], dtype=numpy.int),
        required=True)

    cosim_monitors = List(
                    of=CosimMonitor,
                    label="TVB monitor")

    synchronization_time = Float(
        label="Cosimulation synchronization time (ms)",
        default=0.0,
        required=True,
        doc="""Cosimulation synchronization time for exchanging data 
               in milliseconds, must be an integral multiple
               of integration-step size. It defaults to simulator.integrator.dt""")

    synchronization_n_step = 0
    good_cosim_update_values_shape = (0, 0, 0, 0)
    cosim_history = None  # type: CosimHistory
    _cosimulation_flag = False
    _compute_requirements = True

    def _configure_cosimulation(self):
        """This method will
           - set the synchronization time and number of steps,
           - check the time and the variable of interest are correct
           - create and initialize CosimHistory,
           - configure the cosimulation monitor
           - zero connectivity weights to/from nodes modelled exclusively by the other cosimulator
           """
        # the synchronization time should be at least equal to integrator.dt:
        self.synchronization_time = numpy.maximum(self.synchronization_time, self.integrator.dt)
        # Compute the number of synchronization time steps:
        self.synchronization_n_step = iround(self.synchronization_time / self.integrator.dt)
        # Check if the synchronization time is smaller than the delay of the connectivity
        # the condition is probably not correct. It will change with usage.
        if self.synchronization_n_step > numpy.min(self.connectivity.idelays[numpy.nonzero(self.connectivity.idelays)]):
            raise ValueError('the synchronization time is too long')

        # Check if the couplings variables are in the cosimulation variables of interest
        for cvar in self.model.cvar:
            if cvar not in self.voi:
                raise ValueError('The variables of interest need to contain the coupling variables')

        self.good_cosim_update_values_shape = (self.synchronization_n_step, self.voi.shape[0],
                                               self.proxy_inds.shape[0], self.model.number_of_modes)
        # We create a CosimHistory,
        # for delayed state [synchronization_step+1, n_var, n_node, n_mode],
        # including, initialization of the delayed state from the simulator's history,
        # which must be already configured.
        self.cosim_history = CosimHistory.from_simulator(self)

        # Reconfigure the connectivity for regions modelled by the other cosimulator exclusively:
        if self.exclusive:
            self.connectivity.weights[self.proxy_inds][:, self.proxy_inds] = 0.0
            self.connectivity.configure()

        # Configure the cosimulator monitor
        for monitor in self.cosim_monitors:
            monitor.configure()
            monitor.config_for_sim(self)

    def configure(self, full_configure=True):
        """Configure simulator and its components.

        The first step of configuration is to run the configure methods of all
        the Simulator's components, ie its traited attributes.

        Configuration of a Simulator primarily consists of calculating the
        attributes, etc, which depend on the combinations of the Simulator's
        traited attributes (keyword args).

        Converts delays from physical time units into integration steps
        and updates attributes that depend on combinations of the 6 inputs.

        Returns
        -------
        sim: Simulator
            The configured Simulator instance.

        """
        super(CoSimulator, self).configure(full_configure=full_configure)
        # (Re)Set his flag after every configuration, so that runtime and storage requirements are recomputed,
        # just in case the simulator has been modified (connectivity size, synchronization time, dt etc)
        self._compute_requirements = True
        if self.voi.shape[0] * self.proxy_inds.shape[0] != 0:
            self._cosimulation_flag = True
            self._configure_cosimulation()
        elif self.voi.shape[0] + self.proxy_inds.shape[0] > 0:
            raise ValueError("One of CoSimulator.voi (size=%i) or simulator.proxy_inds (size=%i) are empty!"
                             % (self.voi.shape[0], self.proxy_inds.shape[0]))
        else:
            self._cosimulation_flag = False
            self.synchronization_n_step = 0

    def _loop_update_cosim_history(self, step, state):
        """
        update the history :
            - copy the state and put the state in the cosim_history and the history
            - copy the delayed state and pass it to the monitor
        :param step: the actual step
        :param state: the current state
        :return:
        """
        state_copy = numpy.copy(state)
        if self._cosimulation_flag:
            state_copy[:, self.proxy_inds] = numpy.NAN
            state_output = numpy.copy(self.cosim_history.query(step - self.synchronization_n_step))
            # Update the cosimulation history for the delayed monitor and the next update of history
            self.cosim_history.update(step, state_copy)
            state_copy[:, self.proxy_inds] = 0.0
        else:
            state_output = state
        # Update TVB history to allow for all types of coupling
        super(CoSimulator,self)._loop_update_history(step, state_copy)
        return state_output

    def _update_cosim_history(self, current_steps, cosim_updates):
        """
        Update cosim history and history with the external data
        :param current_steps: the steps to update
        :param cosim_updates: the value of the update step
        :return:
        """
        # Update the proxy nodes in the cosimulation history for synchronization_n_step past steps
        self.cosim_history.update_state_from_cosim(current_steps, cosim_updates, self.voi, self.proxy_inds)
        # Update TVB history with the proxy nodes values
        # TODO optimize step : update all the steps in one
        for step in current_steps:
            state = numpy.copy(self.cosim_history.query(step))
            super(CoSimulator,self)._loop_update_history(step, state)

    def _prepare_stimulus(self, synchronization_time):
        simulation_length = float(self.simulation_length)
        self.simulation_length = float(synchronization_time)
        super(CoSimulator, self)._prepare_stimulus()
        self.simulation_length = simulation_length

    def __call__(self, simulation_length=None, random_state=None, n_steps=None,
                 cosim_updates=None, recompute_requirements=False):
        """
        Return an iterator which steps through simulation time, generating monitor outputs.

        See the run method for a convenient way to collect all output in one call.

        :param cosim_updates: data from the other co-simulator to update TVB state and history
        :param random_state:  State of NumPy RNG to use for stochastic integration,
        :return: Iterator over monitor outputs.
        """

        self.calls += 1

        if simulation_length is not None:
            self.simulation_length = float(simulation_length)

        # Check if the cosimulation update inputs (if any) are correct and update cosimulation history:
        if self._cosimulation_flag:
            if n_steps is not None:
                raise ValueError("n_steps is not used in cosimulation!")
            if cosim_updates is None:
                n_steps = self.synchronization_n_step
            elif len(cosim_updates) != 2:
                raise ValueError("Incorrect cosimulation updates input length %i, expected 2 (i.e., time steps, values)"
                                 % len(cosim_updates))
            elif len(cosim_updates[1].shape) != 4 \
                     or self.good_cosim_update_values_shape[0] < cosim_updates[1].shape[0] \
                     or numpy.any(self.good_cosim_update_values_shape[1:] != cosim_updates[1].shape[1:]):
                raise ValueError("Incorrect cosimulation updates values shape %s, \nexpected %s "
                                 "(i.e., (<=synchronization_n_step, n_voi, n_proxy_nodes, number_of_modes))" %
                                 (str(cosim_updates[1].shape), str(self.good_cosim_update_values_shape)))
            else:
                n_steps = cosim_updates[0].shape[0]
                # Now update cosimulation history with the cosimulation inputs:
                self._update_cosim_history(numpy.array(numpy.around(cosim_updates[0] / self.integrator.dt),
                                                       dtype=numpy.int),
                                           cosim_updates[1])

            # Effective time to run for this __call__
            synchronization_time = n_steps * self.integrator.dt

            if self.simulation_length is None:
                self.simulation_length = float(synchronization_time)

            # Stimulus initialization...
            if self.simulation_length != synchronization_time:
                # ...for synchronization_time = simulation_length
                stimulus = super(CoSimulator, self)._prepare_stimulus()
            else:
                # ...for synchronization_time != simulation_length
                stimulus = self._prepare_stimulus()
        else:
            # Normal TVB simulation - no cosimulation:
            if cosim_updates is not None:
                raise ValueError("cosim_update is not used in normal simulation")

            if n_steps is None:
                n_steps = int(math.ceil(self.simulation_length / self.integrator.dt))
            else:
                if not numpy.issubdtype(type(n_steps), numpy.integer):
                    raise TypeError("Incorrect type for n_steps: %s, expected integer" % type(n_steps))
                self.simulation_length = n_steps * self.integrator.dt

            # Stimulus initialization for simulation_length
            stimulus = super(CoSimulator, self)._prepare_stimulus()

        # Initialization
        if self._compute_requirements or recompute_requirements:
            # Compute requirements for CoSimulation.simulation_length, not for synchronization time
            self._guesstimate_runtime()
            self._calculate_storage_requirement()
            self._compute_requirements = False
        self.integrator.set_random_state(random_state)

        local_coupling = self._prepare_local_coupling()
        state = self.current_state
        start_step = self.current_step + 1
        node_coupling = self._loop_compute_node_coupling(start_step)

        # integration loop
        for step in range(start_step, start_step + n_steps):
            self._loop_update_stimulus(step, stimulus)
            state = self.integrate_next_step(state, self.model, node_coupling, local_coupling, stimulus)
            state_output = self._loop_update_cosim_history(step, state)
            node_coupling = self._loop_compute_node_coupling(step + 1)
            output = self._loop_monitor_output(step-self.synchronization_n_step, state_output, node_coupling)
            if output is not None:
                yield output

        self.current_state = state
        self.current_step = self.current_step + n_steps

    def loop_cosim_monitor_output(self, start_step, n_steps):
        """
        return the value of the cosimulator monitors
        :param start_step: the first step of the values
        :param n_steps: the number of step
        :return:
        """
        if self._cosimulation_flag:
            # check if it's valid input
            if self.good_cosim_update_values_shape[0] < n_steps:
               ValueError("Incorrect n_step, for a number of steps %i, the value should be under %i".format(
                          n_steps, self.good_cosim_update_values_shape[0]))
            if start_step + n_steps > self.good_cosim_update_values_shape[0] + self.current_step:
               ValueError("Incorrect start_step, too early step %i, the value should between %i and %i".format(
                          start_step,self.current_step, self.good_cosim_update_values_shape[0] + self.current_step))
            return [monitor.sample(start_step, n_steps, self.cosim_history, self.history)
                    for monitor in self.cosim_monitors]
        else:
            return []