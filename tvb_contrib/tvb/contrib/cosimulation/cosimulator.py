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
This is the module responsible for co-simulation of TVB with spiking simulators.
It inherits the Simulator class.
.. moduleauthor:: Lionel Kusch <lkusch@thevirtualbrain.org>
.. moduleauthor:: Dionysios Perdikis <dionperd@gmail.com>
"""

import numpy
from tvb.basic.neotraits.api import Attr, NArray, Float, List, TupleEnum, EnumAttr
from tvb.simulator.common import iround
from tvb.simulator.simulator import Simulator, math
from tvb.contrib.cosimulation.cosim_history import CosimHistory
from tvb.contrib.cosimulation.cosim_monitors import CosimMonitor, CosimMonitorFromCoupling
from tvb.contrib.cosimulation.exception import NumericalInstability


class CoSimulator(Simulator):

    exclusive = Attr(
        field_type=bool,
        default=False, required=False,
        doc="1, when the proxy nodes substitute TVB nodes and their mutual connections should be removed.")

    voi = NArray(
        dtype=int,
        label="Cosimulation model state variables' indices",
        doc=("Indices of model's variables of interest (VOI) that"
             "should be updated (i.e., overwriten) during cosimulation."),
        default=numpy.asarray([], dtype=numpy.int_),
        required=True)

    proxy_inds = NArray(
        dtype=numpy.int_,
        label="Indices of TVB proxy nodes",
        default=numpy.asarray([], dtype=numpy.int_),
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
    number_of_cosim_monitors = 0
    _cosim_monitors_noncoupling_indices = []
    _cosim_monitors_coupling_indices = []

    def _configure_synchronization_time(self):
        """This method will set the synchronization time and number of steps,
           and certainly longer or equal to the integration time step.
           Moreover, the synchronization time must be equal or shorter
           than the minimum delay of all existing connections.
           Existing connections are considered those with nonzero weights.
        """
        # The synchronization time should be at least equal to integrator.dt:
        self.synchronization_time = numpy.maximum(self.synchronization_time, self.integrator.dt)
        # Compute the number of synchronization time steps:
        self.synchronization_n_step = iround(self.synchronization_time / self.integrator.dt)
        # Check if the synchronization time is smaller than the minimum delay of the connectivity:
        existing_connections = self.connectivity.weights != 0
        if numpy.any(existing_connections):
            min_idelay = self.connectivity.idelays[existing_connections].min()
            if self.synchronization_n_step > min_idelay:
                raise ValueError('The synchronization time %g is longer than '
                                 'the minimum delay time %g '
                                 'of all existing connections (i.e., of nonzero weight)!'
                                 % (self.synchronization_time, min_idelay * self.integrator.dt))

    def _configure_cosimulation(self):
        """This method will
           - set the synchronization time and number of steps,
           - check the time and the variable of interest are correct
           - create and initialize CosimHistory,
           - configure the cosimulation monitor
           - zero connectivity weights to/from nodes modelled exclusively by the other cosimulator
           """
        # Configure the synchronizatin time and number of steps
        self._configure_synchronization_time()

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
        self.number_of_cosim_monitors = len(self.cosim_monitors)
        self._cosim_monitors_noncoupling_indices = list(range(self.number_of_cosim_monitors))
        self._cosim_monitors_coupling_indices = []
        for iM, monitor in enumerate(self.cosim_monitors):
            monitor.configure()
            monitor.config_for_sim(self)
            if isinstance(monitor, CosimMonitorFromCoupling):
                self._cosim_monitors_noncoupling_indices.remove(iM)
                self._cosim_monitors_coupling_indices.append(iM)

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
            if numpy.any(numpy.isnan(state[self.model.cvar,:,:])):
                raise NumericalInstability("There are missing values for continue the simulation")
            super(CoSimulator,self)._loop_update_history(step, state)

    def __call__(self, simulation_length=None, random_state=None, n_steps=None,
                 cosim_updates=None, recompute_requirements=False):
        """
        Return an iterator which steps through simulation time, generating monitor outputs.

        See the run method for a convenient way to collect all output in one call.

        :param simulation_length: Length of the simulation to perform in ms.
        :param random_state:  State of NumPy RNG to use for stochastic integration.
        :param n_steps: Length of the simulation to perform in integration steps. Overrides simulation_length.
        :param cosim_updates: data from the other co-simulator to update TVB state and history
        :param recompute_requirements: check if the requirement of the simulation
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
                                                       dtype=numpy.int_),
                                           cosim_updates[1])

            self.simulation_length = n_steps * self.integrator.dt
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

        # Initialization
        if self._compute_requirements or recompute_requirements:
            # Compute requirements for CoSimulation.simulation_length, not for synchronization time
            self._guesstimate_runtime()
            self._calculate_storage_requirement()
            self._compute_requirements = False
        self.integrator.set_random_state(random_state)

        local_coupling = self._prepare_local_coupling()
        stimulus = self._prepare_stimulus()
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

    def loop_cosim_monitor_output(self, n_steps=None, relative_start_step=0):
        """
        return the value of the cosimulator monitors
        :param n_steps=None: the number of steps, it defaults to CoSimulator.synchronization_n_step
        :param relative_start_step=0: the first step of the values,
                                      the default value 0 corresponds to
                                      start_step = CoSimulator.current_step - CoSimulator.synchronization_n_step + 1
                                      for non-coupling CosimMonitor,
                                      and to start_step = CoSimulator.current_step + 1
                                      for coupling CosimMonitor, instances

        :return: list of monitor outputs
        """
        if self._cosimulation_flag:
            if n_steps is None:
                n_steps = self.synchronization_n_step
            elif self.good_cosim_update_values_shape[0] < n_steps:
                # check if it's valid input
                ValueError("Incorrect n_steps = %i; it should be <= %i".format(
                           n_steps, self.good_cosim_update_values_shape[0]))
            if relative_start_step < 0 or \
                    relative_start_step + n_steps > self.good_cosim_update_values_shape[0]:
               ValueError("Incorrect relative_start_step %i; it should be in the interval [0, %i]".format(
                          relative_start_step, self.good_cosim_update_values_shape[0] - n_steps))
            coupling_start_step = self.current_step + relative_start_step + 1  # it has to be in the future
            start_step = coupling_start_step - self.synchronization_n_step  # it has to be in the past
            outputs = [[]] * self.number_of_cosim_monitors
            for iM in self._cosim_monitors_noncoupling_indices:
                # Loop over all non coupling cosimulation monitors:
                outputs[iM] = self.cosim_monitors[iM].sample(
                                    self.current_step, start_step, n_steps, self.cosim_history, self.history)
            for iM in self._cosim_monitors_coupling_indices:
                # Loop over all coupling cosimulation monitors:
                outputs[iM] = self.cosim_monitors[iM].sample(
                                    self.current_step, coupling_start_step, n_steps, self.cosim_history, self.history)
            return outputs
        else:
            return []
