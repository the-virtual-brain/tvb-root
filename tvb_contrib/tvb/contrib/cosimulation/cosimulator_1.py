# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
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
#
#

"""
This is the module responsible for co-simulation of TVB with spiking simulators.
It inherits the Simulator class.

.. moduleauthor:: Dionysios Perdikis <dionysios.perdikis@charite.de>


"""
import numpy

from tvb.simulator.common import iround
from tvb.simulator.simulator import Simulator
from tvb.simulator.coupling import Difference,Kuramoto
from tvb.contrib.cosimulation.history import CosimHistory
from tvb.contrib.cosimulation import co_sim_monitor
from tvb.basic.neotraits.api import Attr, NArray, Float


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
        required=False)

    proxy_inds = NArray(
        dtype=numpy.int,
        label="Indices of TVB proxy nodes",
        required=True)

    co_monitor = Attr(field_type=co_sim_monitor.Co_sim_Monitor,
                   required=True,
                   label="TVB monitor")

    synchronization_time = Float(
        label="Cosimulation synchronization time (ms)",
        default=None,
        required=True,
        doc="""Cosimulation synchronization time for exchanging data 
               in milliseconds, must be an integral multiple
               of integration-step size. It defaults to simulator.integrator.dt""")

    cosim_history = None  # type: CosimHistory

    def _configure_cosimulation(self):
        """This method will
           - set the synchronization time and number of steps,
           - create CosimHistory,
           - run all the configuration methods of all TVB <-> Cosimulator interfaces,
           If there are any Cosimulator -> TVB update interfaces:
            - remove connectivity among region nodes modelled exclusively in the other co-simulator.
           If there is any current state cosim_to_tvb interface update:
            - generate a CosimModel class from the original Model class,
            - set the cosim_vars and cosim_vars_proxy_inds properties of the CosimModel class,
              based on the respective vois and proxy_inds of all cosim_to_tvb state interfaces.
           If there is any history cosim_to_tvb interface update:
            - throw a warning if we are not using a coupling function that requires the current state.
           """
        if self.synchronization_time is None:
            # Default synchronization time to dt:
            self.synchronization_time = self.integrator.dt
        self.simulation_length = self.synchronization_time
        # Compute the number of synchronization time steps:
        self.synchronization_n_step = iround(self.synchronization_time / self.integrator.dt)
        # TODO check if the not less than the min of the delay
        for cvar in self.model.cvar:
            if cvar not in self.voi:
                raise ValueError('variable of interest need to contain the coupling variable')

        if self.coupling.__class__ == Difference or self.coupling.__class__ == Kuramoto:
            raise Warning('the co-simulation coupling can have some error')

        # We create a CosimHistory,
        # for delayed state [synchronization_step+1, n_var, n_node, n_mode]
        self.good_update_value_shape = [self.synchronization_n_step,self.voi.shape[0],self.number_of_nodes,self.model.number_of_modes]
        self.cosim_history = CosimHistory(self.synchronization_n_step,
                                          self.model.nvar,
                                          self.number_of_nodes,
                                          self.model.number_of_modes)
        # Initialization of the delayed state from the initial condition,
        # which must be already configured during history configuration
        self.cosim_history.initialize(self.initial_conditions)
        self.dt = self.integrator.dt
        self.co_monitor.configure()
        self.co_monitor.config_for_sim(self)
        # A flag to know if the connectivity needs to be reconfigured:
        if self.exclusive:
            self.connectivity.weights[self.proxy_inds][:, self.proxy_inds] = 0.0
            self.connectivity.configure()

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
        self._configure_cosimulation()

    def _loop_update_state(self, cosim_update):
        """Method to update the current state"""
        new_state = numpy.array(cosim_update[1])
        # ...first bound and clamp the new state
        self.integrator.bound_and_clamp(new_state)
        # ...if we have to further update some of the state variables...
        if self.cosim_to_tvb_interfaces.update_variables_fun:
            # ...we do so...
            new_state = self.cosim_to_tvb_interfaces.update_variables_fun(new_state)
            # ...and we bound and clamp again:
            self.integrator.bound_and_clamp(new_state)
        self.cosim_history.update_state_from_cosim(cosim_update[0],new_state,self.voi,self.proxy_inds)

    def _loop_update_cohistory(self, step, n_reg, state):
        """This method
            - first updates TVB history buffer with the current state,
            - and then (optionally) updates (TVB and cosim) history buffers from cosimulation for past states only,
            - and finally, updates the cosim coupling buffer for future steps, accordingly."""
        state_copy = numpy.copy(state)
        state_copy[:,self.proxy_inds] = numpy.NAN
        # Also, put the current state as it stands
        # to the CosimHistory state buffer:
        # If there are any history cosim to TVB interfaces
        # and if it is time to synchronize...
        delayed_state = numpy.copy(self.cosim_history.query_state(step-self.synchronization_n_step))
        self.cosim_history.update_state(step, state_copy) # update the cosim history for delayed monitor and next update of history
        state_copy[:,self.proxy_inds] = 0.0
        super(CoSimulator,self)._loop_update_history(step,n_reg,state_copy) # update the history for allow all type of coupling
        return delayed_state

    def _update_cohistory(self,current_steps,cosim_updates,n_reg,):
        self.cosim_history.update_state_from_cosim(current_steps,cosim_updates,self.voi,self.proxy_inds) # update the proxy node in the cosim history
        # update the history with the proxy nodes values #TODO optimize step ?
        for step in current_steps:
            state = numpy.copy(self.cosim_history.query_state(step))
            super(CoSimulator,self)._loop_update_history(step,n_reg,state)

    def __call__(self, cosim_updates=None, random_state=None):
        """
        Return an iterator which steps through simulation time, generating monitor outputs.

        See the run method for a convenient way to collect all output in one call.

        :param cosim_updates: data from the other co-simulator to update TVB state and history
        :param random_state:  State of NumPy RNG to use for stochastic integration,
        :return: Iterator over monitor outputs.
        """
        self.calls += 1
        if cosim_updates is not None\
            and len(cosim_updates) == 2\
            and len(cosim_updates[1].shape) == 4\
            and self.good_update_value_shape[0] <  cosim_updates[1].shape[0]\
            and self.good_update_value_shape[1] != cosim_updates[1].shape[1]\
            and self.good_update_value_shape[2] != cosim_updates[1].shape[2]\
            and self.good_update_value_shape[3] != cosim_updates[1].shape[3]:
            raise ValueError("Incorrect update value shape %s, expected %s"%
            cosim_updates[1].shape, self.good_update_value_shape )
        if cosim_updates is None:
            self.simulation_length = self.synchronization_n_step*self.dt
        else:
            self.simulation_length = cosim_updates[1].shape[0]*self.dt

        # Initialization
        self._guesstimate_runtime()
        self._calculate_storage_requirement()
        self._handle_random_state(random_state)
        n_reg = self.connectivity.number_of_regions
        local_coupling = self._prepare_local_coupling()
        stimulus = self._prepare_stimulus()
        state = self.current_state
        start_step = self.current_step + 1
        node_coupling = self._loop_compute_node_coupling(start_step)
        if cosim_updates is not None:
            self._update_cohistory(numpy.array(numpy.around(cosim_updates[0]/self.dt),dtype=numpy.int),cosim_updates[1],n_reg)

        # integration loop
        if cosim_updates is None:
            n_steps = self.synchronization_n_step
        else:
            n_steps = cosim_updates[0].shape[0]
        for step in range(start_step, start_step + n_steps):
            # needs implementing by history + coupling?
            self._loop_update_stimulus(step, stimulus)
            state = self.integrate_next_step(state, self.model, node_coupling, local_coupling, stimulus)
            state_delay = self._loop_update_cohistory(step, n_reg, state)
            node_coupling = self._loop_compute_node_coupling(step + 1)
            output = self._loop_monitor_output(step-self.synchronization_n_step,state_delay, node_coupling)
            if output is not None:
                yield output

        self.current_state = state
        self.current_step = self.current_step + n_steps

    def output_co_sim_monitor(self,start_step,n_steps):
        if self.good_update_value_shape[0] < n_steps:
           ValueError("Incorrect n_step, too number of step %i, the value should be under %i".format(
                      n_steps,self.good_update_value_shape[0]))
        if start_step + n_steps > self.good_update_value_shape[0] + self.current_step:
           ValueError("Incorrect start_step, too early step %i, the value should between %i and %i".format(
                      start_step,self.current_step,self.good_update_value_shape[0] + self.current_step))
        return self.co_monitor.sample(start_step, n_steps,self.cosim_history,self.history)
