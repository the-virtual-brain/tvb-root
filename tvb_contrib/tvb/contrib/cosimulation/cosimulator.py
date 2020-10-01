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
import time
import math
import numpy

from tvb.basic.neotraits.api import Attr, Float
from tvb.simulator.common import iround
from tvb.simulator.simulator import Simulator
from tvb.simulator.coupling import Coupling, CouplingWithCurrentState
from tvb.simulator.monitors import Monitor
from tvb.simulator.models.base import ModelNumbaDfun
from tvb.contrib.cosimulation.history import CosimHistory
from tvb.contrib.cosimulation.tvb_to_cosim_interfaces import TVBtoCosimInterfaces
from tvb.contrib.cosimulation.cosim_to_tvb_interfaces import CosimToTVBInterfaces
from tvb.contrib.cosimulation.models.base import CosimModel, CosimModelNumbaDfun


class CoSimulator(Simulator):

    tvb_to_cosim_interfaces = Attr(
        field_type=TVBtoCosimInterfaces,
        label="TVB to cosimulation outlet interfaces",
        default=None,
        required=False,
        doc="""Interfaces to couple from TVB to a 
               cosimulation outlet (i.e., translator level or another (co-)simulator""")

    cosim_to_tvb_interfaces = Attr(
        field_type=CosimToTVBInterfaces,
        label="Cosimulation to TVB interfaces",
        default=None,
        required=False,
        doc="""Interfaces for updating from a cosimulation outlet 
               (i.e., translator level or another (co-)simulator to TVB.""")

    synchronization_time = Float(
        label="Cosimulation synchronization time (ms)",
        default=None,
        required=True,
        doc="""Cosimulation synchronization time for exchanging data 
               in milliseconds, must be an integral multiple
               of integration-step size. It defaults to simulator.integrator.dt""")

    cosim_history = None  # type: CosimHistory

    use_numba = True

    PRINT_PROGRESSION_MESSAGE = True

    def _configure_cosimulation(self, synchronization_time=None):
        """This method will run all the configuration methods of all TVB <-> Cosimulator interfaces,
           If there are any Cosimulator -> TVB update interfaces:
            - remove connectivity among region nodes modelled exclusively in the other co-simulator.
            - generate a CosimModel class from the original Model class
            - set the cosim_vars and cosim_vars_proxy_inds properties of the CosimModel class,
              based on the respective vois and proxy_inds of all cosim_to_tvb state interfaces.
           """
        if synchronization_time is not None:
            self.synchronization_time = synchronization_time
        if self.synchronization_time is None:
            # Default synchronization time to dt:
            self.synchronization_time = self.integrator.dt
        # Compute the number of synchronization time steps:
        self.synchronization_n_step = iround(self.synchronization_time / self.integrator.dt)
        if self.synchronization_n_step > 1:
            if self.coupling in CouplingWithCurrentState:
                raise ValueError("You cannot have delayed synchronization with %s,"
                                 "which requires the current state of its computation!"
                                 % self.coupling.__class__.__name__)
        # We create a CosimHistory,
        # for delayed state [synchronization_istep + 1, n_var, n_node, n_mode]
        # and node_coupling [2 * synchronization_istep, n_cvar, n_node, n_mode]
        n_cvar = len(self.model.cvar)
        self.cosim_history = CosimHistory(self.synchronization_n_step,
                                          self.model.nvar,
                                          n_cvar,
                                          self.number_of_nodes,
                                          self.model.number_of_modes)
        # Initialization of the delayed state from the initial condition,
        # which must be already configured during history configuration
        # Instead, CosimHistory coupling buffer is initialized
        # for synchronization_nsteps
        # both in the past (for possible TVB monitors),
        # and in the future (for possible Coupling TVB to Cosim interfaces).
        coupling_buffer = numpy.empty((self.cosim_history.n_ctime, self.cosim_history.n_cvar,
                                       self.cosim_history.n_node, self.cosim_history.n_mode)).astype("f")
        start_step = self.current_step + 1
        for istep in range(start_step-self.synchronization_n_step, start_step + self.synchronization_n_step):
            coupling_buffer[istep % self.cosim_history.n_ctime] = self._loop_compute_node_coupling(istep)
        self.cosim_history.initialize(self.initial_conditions, coupling_buffer)
        if self.tvb_to_cosim_interfaces:
            # Configure any TVB to Cosim interfaces:
            self.tvb_to_cosim_interfaces.configure(self)
        # A flag to identify if there is any update of the current TVB state:
        self._update_state = self.cosim_to_tvb_interfaces.number_of_state_interfacese
        if self.cosim_to_tvb_interfaces:
            # Configure any Cosim to TVB interfaces:
            self.cosim_to_tvb_interfaces.configure(self)
            # Create a CosimModel out of the TVB model,
            # in order to add cosim_vars and cosim_vars_proxy_inds properties
            # that might be used in the model's code to
            # identify the states that get updated from the Cosimulator:
            if isinstance(self.model, ModelNumbaDfun):
                self.model = CosimModelNumbaDfun.from_model(self.model)
            else:
                self.model = CosimModel.from_model(self.model)
            self.model.cosim_vars = self.cosim_to_tvb_interfaces.state_vois
            self.model.cosim_vars_proxy_inds = self.cosim_to_tvb_interfaces.state_proxy_inds
            self.model.configure()
            self.model.update_derived_parameters()
            # A flag to know if the connectivity needs to be reconfigured:
            reconfigure_connectivity = False
            for interface in self.cosim_to_tvb_interfaces.interfaces:
                # Interfaces marked as "exclusive" by the user
                # should eliminate the connectivity weights among the proxy nodes,
                # since those nodes are mutually coupled within the other (co-)simulator network model.
                if interface.exclusive:
                    reconfigure_connectivity = True
                    self.connectivity.weights[interface.proxy_inds][:, interface.proxy_inds] = 0.0
            if reconfigure_connectivity:
                self.connectivity.configure()

    def configure(self, full_configure=True, synchronization_time=None):
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
        if not self.use_numba or self.model.number_of_modes > 1:
            self.use_numba = False
            if hasattr(self.model, "_numpy_dfun"):
                self._dfun = self.model._numpy_dfun
            self._spatial_param_reshape = (-1, 1)

        super(CoSimulator, self).configure(full_configure=full_configure)
        self._configure_cosimulation(synchronization_time)
        return self

    def _loop_update_state(self, state):
        """Method to update the current state"""
        new_state = numpy.array(state)
        # First update using each one of the cosim_to_tvb_interfaces state interfaces
        for state_interface in self.cosim_to_tvb_interfaces.state_interfaces:
            new_state = state_interface.update(new_state)
        # If indeed the state has changed:
        if numpy.any(new_state != state):
            # ...first bound and clamp the new state
            self.bound_and_clamp(new_state)
            # ...if we have to further update some of the state variables...
            if self.cosim_to_tvb_interfaces.update_variables_fun:
                # ...we do so...
                new_state = self.cosim_to_tvb_interfaces.update_variables_fun(new_state)
                # ...and we bound and clamp again:
                self.bound_and_clamp(new_state)
        return new_state

    def _loop_update_history(self, step, n_reg, state, cosim_updates=[]):
        """This method
            - first updates TVB history buffer with the current state,
            - and then (optionally) updates (TVB and cosim) history buffers from cosimulation for past states only,
            - and finally, updates the cosim coupling buffer for future steps, accordingly."""
        # NOTE that if there is no history interface this state is already final!!
        # Update TVB history with the current state as usual:
        super(CoSimulator, self)._loop_update_history(step, n_reg, state)
        # Also, put the current state as it stands
        # to the CosimHistory state buffer:
        self.cosim_history.update_state(step, state)
        # Compute the node_coupling for next step:
        next_step = step + 1
        node_coupling = self._loop_compute_node_coupling(next_step)
        # ...and update the CosimHistory coupling buffer:
        self.cosim_history.update_coupling(next_step, node_coupling)
        # If there are any history cosim to TVB interfaces
        # and if it is time to synchronize...
        if self.number_of_history_interfaces and step % self.synchronization_n_step == 0:
            # For every step of synchronization_time,
            # starting from the past towards the present,
            start_step = step - self.synchronization_n_step
            synchronization_steps = list(range(start_step, step))
            # until the current step (not included):
            for interface in self.cosim_to_tvb_interfaces.history_interfaces:
                try:
                    # To get update from input, if any:
                    update = cosim_updates.pop(0)
                except:
                    # ...this is the case that the interfaces reads from a file or MPI port
                    update = None
                interface.update(synchronization_steps, update)
                for istep in range(start_step, step):
                    # For every step...
                    # ...get the assumed updated state from the CosimHistory...
                    istate = self.cosim_history.query_state(istep)
                    # ...bound and clamp it...
                    self.bound_and_clamp(istate)
                    # ...if we have to further update some of the state variables...
                    if self.cosim_to_tvb_interfaces.update_variables_fun:
                        # ...we do so...
                        istate = self.cosim_to_tvb_interfaces.update_variables_fun(istate)
                        # ...bound and clamp it again...
                        self.bound_and_clamp(istate)
                    # ...put it back to CosimHistory...
                    self.cosim_history.update_state(istep, self.bound_and_clamp(istate))
                    # ...and finally update the TVB history as well...
                    self.history.update(istep, self.cosim_history.query_state(istep))
                # Now that all history is updated,
                # we need to also recompute the node_coupling for the next synchronization_n_step time steps:
                start_step = step + 1
                for istep in range(start_step, start_step+self.synchronization_n_step):
                    self.cosim_history.update_coupling(istep, self._loop_compute_node_coupling(istep))
                # Return the current state and next time step node coupling from CosimHistory:
                return self.cosim_history.query_state(step), self.cosim_history.query_state(step+1)
            else:
                return state, node_coupling

    @staticmethod
    def _get_tvb_monitor(monitor):
        return monitor

    @staticmethod
    def _get_tvb_monitor_from_tvb_to_cosim_interface(interface):
        return interface.monitor

    def _loop_record_monitors(self, step, monitors, get_monitor_fun=None, return_flag=False):
        """This method records from a sequence of monitors,
           returned outputs, if any,
           and sets a return_flag to True, if there is at least one output returned."""
        if get_monitor_fun is None:
            get_monitor_fun = self._get_tvb_monitor
        outputs = None
        # If it is time to synchronize...
        if step % self.synchronization_n_step == 0:
            # ...compute the first step of the sampling period...
            start_step = step - self.synchronization_n_step
            outputs = []
            for monitor in monitors:
                # ...loop through the sampling period...
                time = []
                data = []
                for _step in range(start_step, step):
                    # Get the observable, based either on state variables or on node coupling:
                    observable = numpy.where(isinstance(get_monitor_fun(monitor), Coupling),
                                             self.cosim_history.query_coupling(_step),
                                             self.model.observe(self.cosim_history.query_state(_step)())).item()
                    # ...to provide states to the sample method of the TVB monitor...
                    _output = self.monitor.record(_step, observable)
                    if _output is not None:
                        time.append(_output[0])
                        data.append(_output[1])

                if len(time) > 0:
                    # ...and form the final output of record, if any...
                    outputs.append([time, data])
            return_flag = True
        return outputs, return_flag

    def _loop_cosim_monitor_output(self, step, return_outputs=[], return_flag=False):
        """This method records from all cosimulation monitors"""
        cosim_state_outputs, return_flag = \
            self._loop_record_monitors(step, self.tvb_to_cosim_interfaces.state_interfaces,
                                       self._get_tvb_monitor_from_tvb_to_cosim_interface,
                                       return_flag)
        return_outputs.append(cosim_state_outputs)
        cosim_coupling_outputs, return_flag = \
            self._loop_record_monitors(step + self.synchronization_n_step + 1,
                                       self.tvb_to_cosim_interfaces.coupling_interfaces, return_flag)
        return_outputs.append(cosim_coupling_outputs)
        return return_outputs, return_flag

    def _loop_monitor_output(self, step):
        """This method computes the observed state,
           records from all TVB monitors, and then, all cosimulation monitors,
           and returns:
            - either None if none of the above monitors returns an ouput,
            - (monitor_outputs, cosim_state_outputs, cosim_coupling_outputs),
               with None for outputs that do not exist."""
        return_outputs, return_flag = self._loop_record_monitors(step, self.monitors)
        if self.tvb_to_cosim_interfaces:
            cosim_state_outputs, cosim_coupling_outputs, return_flag = \
                self._loop_cosim_monitor_output(step, return_outputs, return_flag)
            if return_flag:
                # return a tuple of (monitors, cosim_state_outputs, cosim_coupling_outputs) outputs
                return tuple([return_outputs, cosim_state_outputs, cosim_coupling_outputs])
        else:
            if return_flag:
                # return only (monitors,) outputs
                return tuple([return_outputs])

    def _print_progression_message(self, step, n_steps):
        if step - self.current_step >= self._tic_point:
            toc = time.time() - self._tic
            if toc > 600:
                if toc > 7200:
                    time_string = "%0.1f hours" % (toc / 3600)
                else:
                    time_string = "%0.1f min" % (toc / 60)
            else:
                time_string = "%0.1f sec" % toc
            print_this = "\r...%0.1f%% done in %s" % \
                         (100.0 * (step - self.current_step) / n_steps, time_string)
            self.log.info(print_this)
            self._tic_point += self._tic_ratio * n_steps

    def __call__(self, simulation_length=None, random_state=None, cosim_updates=[]):
        """
        Return an iterator which steps through simulation time, generating monitor outputs.

        See the run method for a convenient way to collect all output in one call.

        :param simulation_length: Length of the simulation to perform in ms,
        :param random_state:  State of NumPy RNG to use for stochastic integration,
        :param sync_time: co-simulation synchronization time in ms,
        :param cosim_updates: data from the other co-simulator to update TVB state and history
        :return: Iterator over monitor outputs.
        """
        self.calls += 1
        if simulation_length is not None:
            self.simulation_length = simulation_length

        # Initialization
        self._guesstimate_runtime()
        self._calculate_storage_requirement()
        self._handle_random_state(random_state)
        n_reg = self.connectivity.number_of_regions
        local_coupling = self._prepare_local_coupling()
        stimulus = self._prepare_stimulus()
        state = self.current_state
        start_step = self.current_step + 1
        node_coupling = self.cosim_history.query_coupling(start_step)

        # integration loop
        n_steps = int(math.ceil(self.simulation_length / self.integrator.dt))
        if self.PRINT_PROGRESSION_MESSAGE:
            self._tic = time.time()
            self._tic_ratio = 0.1
            self._tic_point = self._tic_ratio * n_steps
        for step in range(start_step, start_step + n_steps):
            # needs implementing by history + coupling?
            self._loop_update_stimulus(step, stimulus)
            state = self.integrate_next_step(state, self.model, node_coupling, local_coupling, stimulus)
            if self._update_state:
                state = self._loop_update_state(state)
            state, node_coupling = self._loop_update_history(step, n_reg, state, cosim_updates)
            output = self._loop_monitor_output(step)
            if output is not None:
                yield output
            if self.PRINT_PROGRESSION_MESSAGE:
                self._print_progression_message(step, n_steps)

        self.current_state = state
        self.current_step = self.current_step + n_steps - 1  # -1 : don't repeat last point

# TODO: adjust function to compute fine scale resources' requirements as well, ...if you can! :)

    def send_initial_condition_to_cosimulator(self):
        """This method sends the initial condition to the co-simulator."""
        data, return_flag = self._loop_cosim_monitor_output(self.current_step)
        if return_flag:
            self.send_data_to_cosimulator(data)

    def receive_data_from_cosimulator(self):
        """This method receives data from the other simulator in the format,
           where cosim_updates are lists of data
           that correspond to the cosim_to_tvb_interfaces.history_interfaces,
           in the same order."""
        return []

    def send_data_to_cosimulator(self, data):
        """This method sends TVB output data to the other simulator."""
        pass

    def prepare_run(self, **kwds):
        """This method sets
            - sync_time (default = self.integrator.dt),
            - current_time = self.current_step * self.integrator.dt,
            - and end_time = current_time + simulation_length,
            from optional user kwds,
            as well as updates kwds by setting kwds["simulation_length"] = sync_time."""
        sync_time = kwds.get("sync_time", self.integrator.dt)
        simulation_length = kwds.pop("simulation_length", self.integrator.dt)
        current_time = self.current_step * self.integrator.dt
        end_time = current_time + simulation_length
        kwds["simulation_length"] = sync_time
        return sync_time, current_time, end_time, kwds


class SequentialCosimulator(CoSimulator):

    def configure_cosimulator(self, *args, **kwargs):
        """This method configures the other simulator."""
        pass

    def configure(self, full_configure=True, *args, **kwargs):
        """This method (optionally) configures the other simulator, besides TVB."""
        configure_cosimulator = kwargs.pop("configure_cosimulator", False)
        super(SequentialCosimulator, self).configure(full_configure=True)
        if configure_cosimulator:
            self.configure_cosimulator(*args, **kwargs)

    def run_cosimulator(self, sync_time):
        """This method runs the other simulator for sync_time."""
        pass

    def cleanup_cosimulator(self):
        """This method cleans up the other simulator."""
        pass

    def run(self, sync_time=None,
            send_initial_condition_to_cosimulator=False, cleanup_cosimulator=False, **kwds):
        """Convenience method to call the simulator with **kwds and collect output data.
            The method is modified to
            (a) (optionally) send initial condition TVB data to the other simulator,
            (b) run the other simulator,
            (c) (optionally) send TVB data to the other simulator,
            (d) (optionally) receive data from the other simulator,
            (e) cleanup the other simulator.
            """
        sync_time, current_time, end_time, kwds = self.prepare_run(**kwds)
        ts, xs = [], []
        for _ in self.monitors:
            ts.append([])
            xs.append([])
        if kwds.pop("send_initial_condition_to_cosimulator", False) and self.tvb_to_cosim_interfaces:
            self.send_initial_condition_to_cosimulator()
        cleanup_cosimulator = kwds.pop("cleanup_cosimulator", False)
        self.PRINT_PROGRESSION_MESSAGE = kwds.pop("print_progression_message", True)
        wall_time_start = time.time()
        while current_time < end_time:
            for data in self(**kwds):
                current_time += sync_time
                if data is not None:
                    for tl, xl, t_x in zip(ts, xs, data[0]):
                        if t_x is not None:
                            t, x = t_x
                            tl.append(t)
                            xl.append(x)
                    if len(data) > 1:
                        self.send_data_to_cosimulator(data[1:])
                self.run_cosimulator()
                if self.cosim_to_tvb_interfaces:
                    kwds["cosim_updates"] = self.receive_data_from_cosimulator()
        elapsed_wall_time = time.time() - wall_time_start
        self.log.info("%.3f s elapsed, %.3fx real time", elapsed_wall_time,
                      elapsed_wall_time * 1e3 / self.simulation_length)
        for i in range(len(ts)):
            ts[i] = numpy.array(ts[i])
            xs[i] = numpy.array(xs[i])
        if cleanup_cosimulator:
            self.cleanup_cosimulator()
        return list(zip(ts, xs))


class ParallelCosimulator(CoSimulator):

    def receive_data_from_cosimulator(self):
        """This method receives data from the other simulator in the format,
           where cosim_updates are lists of data
           that correspond to the cosim_to_tvb_interfaces.history_interfaces,
           in the same order."""
        return []

    def run(self, **kwds):
        """Convenience method to call the simulator with **kwds and collect output data.
            The method is modified to
            (a) (optionally) send initial condition TVB data to the other simulator,
            (b) (optionally) send TVB data to the other simulator,
            (c) (optionally) receive data from the other simulator.
            """
        sync_time, current_time, end_time, kwds = self.prepare_run(**kwds)
        ts, xs = [], []
        for _ in self.monitors:
            ts.append([])
            xs.append([])
        if kwds.pop("send_initial_condition_to_cosimulator", False) and self.tvb_to_cosim_interfaces:
            self.send_initial_condition_to_cosimulator()
        self.PRINT_PROGRESSION_MESSAGE = kwds.pop("print_progression_message", True)
        wall_time_start = time.time()
        while current_time < end_time:
            for data in self(**kwds):
                current_time += sync_time
                if data is not None:
                    for tl, xl, t_x in zip(ts, xs, data[0]):
                        if t_x is not None:
                            t, x = t_x
                            tl.append(t)
                            xl.append(x)
                    if len(data) > 1:
                        self.send_data_to_cosimulator(data[1:])
                if self.cosim_to_tvb_interfaces:
                    kwds["cosim_updates"] = self.receive_data_from_cosimulator()
        elapsed_wall_time = time.time() - wall_time_start
        self.log.info("%.3f s elapsed, %.3fx real time", elapsed_wall_time,
                      elapsed_wall_time * 1e3 / self.simulation_length)
        for i in range(len(ts)):
            ts[i] = numpy.array(ts[i])
            xs[i] = numpy.array(xs[i])
        return list(zip(ts, xs))
