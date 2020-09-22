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

from tvb.basic.neotraits.api import Attr
from tvb.simulator.simulator import Simulator
from tvb.simulator import models
from tvb.simulator.models.base import ModelNumbaDfun
from tvb.contrib.cosimulation.tvb_to_cosim_interfaces import TVBtoCosimInterfaces
from tvb.contrib.cosimulation.cosim_to_tvb_interfaces import CosimToTVBInterfaces
from tvb.contrib.cosimulation.models.base import CosimModel, CosimModelNumbaDfun
from tvb.contrib.cosimulation.models.wilson_cowan_constraint import WilsonCowan


class CoSimulator(Simulator):

    tvb_to_cosim_interfaces = Attr(
        field_type=TVBtoCosimInterfaces,
        label="TVB to cosimulation outlet interfaces",
        default=None,
        required=False,
        doc="""Interfaces from TVB to a cosimulation outlet (i.e., translator level or another simulator""")

    cosim_to_tvb_interfaces = Attr(
        field_type=CosimToTVBInterfaces,
        label="Cosimulation to TVB interfaces",
        default=None,
        required=False,
        doc="""Interfaces from TVB to a cosimulation outlet (i.e., translator level or another simulator""")

    model = Attr(
        field_type=models.Model,
        label="Local dynamic model",
        default=WilsonCowan(),
        required=True,
        doc="""A tvb.simulator.Model object which describe the local dynamic
            equations, their parameters, and, to some extent, where connectivity
            (local and long-range) enters and which state-variables the Monitors
            monitor. By default the 'WilsonCowan' model with constraints is used. Read the
            Scientific documentation to learn more about this model.""")

    use_numba = True

    PRINT_PROGRESSION_MESSAGE = True

    def _eliminate_exclusive_proxy_nodes_connectivity(self, proxy_inds):
        """This method will set the proxy nodes mutual connectivity to zero"""
        self.connectivity.weights[proxy_inds][:, proxy_inds] = 0.0

    def _configure_cosim_interfaces(self):
        """This method will run all the configuration methods of all TVB <-> Cosimulator interfaces,
           as well as generated a CosimModel class from the original Model class,
           and set the cosim_(c)vars and cosim_(c)vars_proxy_inds properties of the CosimModel class,
           based on the respective vois and proxy_inds of all cosim_to_tvb_interfaces."""
        if self.tvb_to_cosim_interfaces:
            for state_interface in self.tvb_to_cosim_interfaces.state_interfaces:
                state_interface.config_for_sim(self)
            for history_interface in self.tvb_to_cosim_interfaces.history_interfaces:
                history_interface.config_for_sim(self)
        if self.cosim_to_tvb_interfaces:
            if isinstance(self.model, ModelNumbaDfun):
                self.model = CosimModelNumbaDfun.from_model(self.model)
            else:
                self.model = CosimModel.from_model(self.model)
            for state_interface in self.cosim_to_tvb_interfaces.state_interfaces:
                state_interface.config_for_sim(self)
                self.model.cosim_vars.append(state_interface.voi)
                self.model.cosim_vars_proxy_inds.append(state_interface.proxy_inds)
                if state_interface.exclusive:
                    self._eliminate_exclusive_proxy_nodes_connectivity(state_interface.proxy_inds)
            self.connectivity.configure()
            for history_interface in self.cosim_to_tvb_interfaces.history_interfaces:
                history_interface.config_for_sim(self)
                self.model.cosim_cvars.append(history_interface.voi)
                self.model.cosim_cvars_proxy_inds.append(history_interface.proxy_inds)
            self.model.configure()
            self.model.update_derived_parameters()

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
        if not self.use_numba or self.model.number_of_modes > 1:
            self.use_numba = False
            if hasattr(self.model, "_numpy_dfun"):
                self._dfun = self.model._numpy_dfun
            self._spatial_param_reshape = (-1, 1)

        super(CoSimulator, self).configure(full_configure=full_configure)
        self._configure_cosim_interfaces()
        return self

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

    def _loop_record_monitors(self, step, observed, monitors, return_flag=False):
        """This method recordd from a sequence of monitors,
           returnd outputs, if any,
           and sets a return_flag to True, if there is at least one output returned."""
        outputs = []
        for monitor in monitors:
            output = monitor.record(step, observed)
            if output is not None:
                outputs.append(output)
                return_flag = True
        return outputs, return_flag

    def _loop_cosim_monitor_ouput(self, step, observed, return_outputs=[], return_flag=False):
        """This method records from all cosimulation monitors"""
        cosim_state_output, return_flag = self._loop_record_monitors(step, observed,
                                                                     self.tvb_to_cosim_interfaces.state_interfaces,
                                                                     return_flag)
        if len(cosim_state_output):
            return_outputs += cosim_state_output
        cosim_history_output, return_flag = self._loop_record_monitors(step, observed,
                                                                       self.tvb_to_cosim_interfaces.history_interfaces,
                                                                       return_flag)
        if len(cosim_history_output):
            return_outputs += cosim_history_output
        cosim_coupling_output, return_flag = self._loop_record_monitors(step, observed,
                                                                        self.tvb_to_cosim_interfaces.coupling_interfaces,
                                                                        return_flag)
        if len(cosim_coupling_output):
            return_outputs += cosim_coupling_output
        return return_outputs, return_flag

    def _loop_monitor_output(self, step, state):
        """This method computes the observed state,
           records from all TVB monitors, and then, all cosimulation monitors,
           and returns:
            - either None if none of the above monitors returns an ouput,
            - (monitor_outputs, ) if only TVB monitors return an output,
            - (None, cosim_outputs) if only cosimulation monitors return an ouput,
            - (monitor_outputs, cosim_outputs) if all types of monitors returns an output"""
        observed = self.model.observe(state)
        return_outputs, return_flag = self._loop_record_monitors(step, observed, self.monitors)
        if self.tvb_to_cosim_interfaces:
            cosim_outputs, return_flag = self._loop_cosim_monitor_ouput(step, observed, [], return_flag)
            if return_flag:
                # return a tuple of (monitors, cosim_monitors) outputs
                return tuple([return_outputs, cosim_outputs])
        else:
            if return_flag:
                # return only (monitors,) outputs
                return tuple([return_outputs])

    def _loop_cosim_update_state(self, state, cosim_state_updates=[]):
        """This method updates the state from the cosimulator, if there are any state cosim to tvb interfaces.
            If there is an update, the updated state gets again through the bound_and_clamp method.
        """
        new_state = numpy.array(state)
        for ii, state_interface in enumerate(self.cosim_to_tvb_interfaces.state_interfaces):
            try:
                update_state = cosim_state_updates.pop(0)
            except:
                update_state = None
            new_state = state_interface.update(state, update_state)
        if any(new_state != state):
            self.bound_and_clamp(new_state)
        return new_state

    def _loop_update_history(self, step, n_reg, state, cosim_state_updates=[], cosim_history_updates=[]):
        """This method
            - first (optionally) updates the state from cosimulation,
            - then updates TVB history buffer with the current state,
            - and finally (optionally) updates history from cosimulation."""
        if self.cosim_to_tvb_interfaces:
            state = self._loop_cosim_update_state(state, cosim_state_updates)
        super(CoSimulator, self)._loop_update_history(step, n_reg, state)
        if self.cosim_to_tvb_interfaces:
            for history_interface in self.cosim_to_tvb_interfaces.history_interfaces:
                try:
                    history_update = cosim_history_updates.pop(0)
                except:
                    history_update = None
                history_interface.update(step, history_update)

    def __call__(self, simulation_length=None, random_state=None,
                 cosim_state_updates=[], cosim_history_updates=[]):
        """
        Return an iterator which steps through simulation time, generating monitor outputs.

        See the run method for a convenient way to collect all output in one call.

        :param simulation_length: Length of the simulation to perform in ms.
        :param random_state:  State of NumPy RNG to use for stochastic integration.
        :return: Iterator over monitor outputs.
        """
        self.calls += 1
        if simulation_length is not None:
            self.simulation_length = simulation_length

        # Intialization
        self._guesstimate_runtime()
        self._calculate_storage_requirement()
        self._handle_random_state(random_state)
        n_reg = self.connectivity.number_of_regions
        local_coupling = self._prepare_local_coupling()
        stimulus = self._prepare_stimulus()
        state = self.current_state

        # integration loop
        n_steps = int(math.ceil(self.simulation_length / self.integrator.dt))
        if self.PRINT_PROGRESSION_MESSAGE:
            self._tic = time.time()
            self._tic_ratio = 0.1
            self._tic_point = self._tic_ratio * n_steps
        for step in range(self.current_step + 1, self.current_step + n_steps + 1):
            # needs implementing by history + coupling?
            node_coupling = self._loop_compute_node_coupling(step)
            self._loop_update_stimulus(step, stimulus)
            state = self.integrate_next_step(state, self.model, node_coupling, local_coupling, stimulus)
            self._loop_update_history(step, n_reg, state, cosim_state_updates, cosim_history_updates)
            output = self._loop_monitor_output(step, state)
            if output is not None:
                yield output
            if self.PRINT_PROGRESSION_MESSAGE:
                self._print_progression_message(step, n_steps)

        self.current_state = state
        self.current_step = self.current_step + n_steps - 1  # -1 : don't repeat last point

# TODO: adjust function to compute fine scale resources' requirements as well, ...if you can! :)

    def send_initial_condition_to_cosimulator(self):
        """This method sends the initial condition to the co-simulator."""
        data, return_flag = self._loop_cosim_monitor_ouput(self.current_step, self.current_state)
        if return_flag:
            self.send_data_to_cosimulator(data)

    def receive_data_from_cosimulator(self):
        """This method receives data from the other simulator in the format,
           (cosim_state_updates, cosim_history_updates),
           where cosim_state_updates (cosim_history_updates) are lists of data
           that correspond to the cosim_to_tvb_interfaces.state_interfaces(history_interfaces), respectively,
           in the same order."""
        return [], []

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
        sync_time = kwds.pop("sync_time", self.integrator.dt)
        simulation_length = kwds.pop("simulation_length", self.integrator.dt)
        current_time = self.current_step * self.integrator.dt
        end_time = current_time + simulation_length
        kwds["simulation_length"] = sync_time
        return sync_time, current_time, end_time, kwds

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
                    cosim_state_updates, cosim_history_updates = self.receive_data_from_cosimulator()
                    kwds["cosim_state_updates"] = cosim_state_updates
                    kwds["cosim_history_updates"] = cosim_history_updates
        elapsed_wall_time = time.time() - wall_time_start
        self.log.info("%.3f s elapsed, %.3fx real time", elapsed_wall_time,
                      elapsed_wall_time * 1e3 / self.simulation_length)
        for i in range(len(ts)):
            ts[i] = numpy.array(ts[i])
            xs[i] = numpy.array(xs[i])
        return list(zip(ts, xs))


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
            (b) (optionally) send TVB data to the other simulator,
            (c) run the other simulator,
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
                    cosim_state_updates, cosim_history_updates = self.receive_data_from_cosimulator()
                    kwds["cosim_state_updates"] = cosim_state_updates
                    kwds["cosim_history_updates"] = cosim_history_updates
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
    pass
