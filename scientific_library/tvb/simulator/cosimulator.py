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

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>
.. moduleauthor:: Dionysios Perdikis <dionysios.perdikis@charite.de>

"""
import time
import math
import numpy
from tvb.simulator.simulator import Simulator
from tvb.simulator import models
from tvb.basic.neotraits.api import Attr

from tvb.simulator.models.wilson_cowan_constraint import WilsonCowan


class CoSimulator(Simulator):
    tvb_spikeNet_interface = None
    configure_spiking_simulator = None
    run_spiking_simulator = None

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

    def _configure_integrator_noise(self):
        """
        This enables having noise to be state variable specific and/or to enter
        only via specific brain structures, for example it we only want to
        consider noise as an external input entering the brain via appropriate
        thalamic nuclei.

        Support 3 possible shapes:
            1) number_of_nodes;

            2) number_of_state_variables; and

            3) (number_of_state_variables, number_of_nodes).

        """

        if self.tvb_spikeNet_interface is not None and self.integrator.noise.ntau > 0.0:
            # TODO: find out if this is really a problem
            self.log.warning("Colored noise is currently not supported for tvb-multiscale co-simulations!\n" +
                             "Setting integrator.noise.ntau = 0.0 and configuring white noise!")
            self.integrator.noise.ntau = 0.0

        super(CoSimulator, self)._configure_integrator_noise()

    def preconfigure(self):
        if self.tvb_spikeNet_interface is not None:
            # TODO: decide if this is really necessary...
            if self.integrator.dt >= 2 * self.tvb_spikeNet_interface.spikeNet_min_delay:
                self.integrator.dt = int(numpy.round(self.integrator.dt /
                                                     self.tvb_spikeNet_interface.spikeNet_min_delay)) * \
                                     self.tvb_spikeNet_interface.spikeNet_min_delay
            else:
                raise ValueError("TVB integration time step dt=%f "
                                 "is not equal or greater than twice the Spiking Network minimum delay min_delay=%f!" %
                                 (self.integrator.dt, self.tvb_spikeNet_interface.spikeNet_min_delay))
        super(CoSimulator, self).preconfigure()

    def _configure_spikeNet_interface(self):
        # TODO: Shall we implement a parallel implentation for multiple modes for SpikeNet as well?!
        if self.current_state.shape[2] > 1:
            raise ValueError("Multiple modes' simulation not supported for TVB multiscale simulations!\n"
                             "Current modes number is %d." % self.initial_conditions.shape[3])
        # Setup Spiking Simulator configure() and Run() method
        self.configure_spiking_simulator = self.tvb_spikeNet_interface.spiking_network.configure
        self.run_spiking_simulator = self.tvb_spikeNet_interface.spiking_network.Run
        if len(self.tvb_spikeNet_interface.spikeNet_to_tvb_params) > 0:
            # Create TVB model parameter for SpikeNet to target
            dummy = numpy.zeros((self.connectivity.number_of_regions,)).reshape(self._spatial_param_reshape)
            dummy[self.tvb_spikeNet_interface.spiking_nodes_ids] = 1.0
            for param in self.tvb_spikeNet_interface.spikeNet_to_tvb_params:
                setattr(self.model, param, dummy)
            self.model.update_derived_parameters()
        # Configure tvb-spikeNet interface
        self.tvb_spikeNet_interface.configure(self.model)
        # If there are Spiking nodes and are represented exclusively in Spiking Network...
        if self.tvb_spikeNet_interface.exclusive_nodes and len(self.tvb_spikeNet_interface.spiking_nodes_ids) > 0:
            # ...set the respective connectivity weights among them to zero:
            self.connectivity.weights[self.tvb_spikeNet_interface.spiking_nodes_ids] \
                [:, self.tvb_spikeNet_interface.spiking_nodes_ids] = 0.0
            self.connectivity.delays[self.tvb_spikeNet_interface.spiking_nodes_ids] \
                [:, self.tvb_spikeNet_interface.spiking_nodes_ids] = 0.0
            self.connectivity.configure()

    def configure(self, tvb_spikeNet_interface=None, full_configure=True):
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
        # Set TVB - spikeNet interface:
        self.tvb_spikeNet_interface = tvb_spikeNet_interface
        super(CoSimulator, self).configure(full_configure=full_configure)
        if self.tvb_spikeNet_interface is not None:
            self._configure_spikeNet_interface()
        return self

    def __call__(self, simulation_length=None, random_state=None):
        """
        Return an iterator which steps through simulation time, generating monitor outputs.

        See the run method for a convenient way to collect all output in one call.

        :param simulation_length: Length of the simulation to perform in ms.
        :param random_state:  State of NumPy RNG to use for stochastic integration.
        :return: Iterator over monitor outputs.
        """
        if self.tvb_spikeNet_interface is None:
            for data in super(CoSimulator, self).__call__(simulation_length, random_state):
                yield data
        else:
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

            # Do for initial condition, i.e., prepare step t0 -> t1:
            init_step = self.current_step + 1  # the first step in the loop
            node_coupling = self._loop_compute_node_coupling(init_step)
            self._loop_update_stimulus(init_step, stimulus)
            if self._spike_stimulus_fun:
                self._apply_spike_stimulus(init_step)
            state = self.update_state(state, node_coupling, local_coupling)
            # NOTE!!!: we don't update TVB from spikeNet initial condition,
            # since there is no output yet from spikeNet

            # spikeNet simulation preparation:
            self.configure_spiking_simulator()
            # A flag to skip unnecessary steps when TVB does NOT couple to Spiking Simulator
            coupleTVBstateToSpikeNet = len(self.tvb_spikeNet_interface.tvb_to_spikeNet_interfaces) > 0
            # A flag to skip unnecessary steps when Spiking Simulator does NOT update TVB state
            updateTVBstateFromSpikeNet = len(self.tvb_spikeNet_interface.spikeNet_to_tvb_interfaces) > 0

            # integration loop
            n_steps = int(math.ceil(self.simulation_length / self.integrator.dt))
            if self.PRINT_PROGRESSION_MESSAGE:
                self._tic = time.time()
                self._tic_ratio = 0.1
                self._tic_point = self._tic_ratio * n_steps
            end_step = init_step + n_steps
            for step in range(init_step, end_step):

                # 1. Update spikeNet with TVB state t_(step-1)
                #    ...and integrate it for one time step
                if coupleTVBstateToSpikeNet:
                    # TVB state t_(step-1) -> SpikeNet (state or parameter)
                    # Communicate TVB state to some SpikeNet device (TVB proxy) or TVB coupling to SpikeNet nodes,
                    # including any necessary conversions from TVB state to SpikeNet variables,
                    # in a model specific manner
                    # TODO: find what is the general treatment of local coupling, if any!
                    #  Is this addition correct in all cases?
                    self.tvb_spikeNet_interface.tvb_state_to_spikeNet(state, node_coupling + local_coupling, stimulus)

                # 2. Integrate Spiking Network to get the new Spiking Network state t_step
                self.run_spiking_simulator(self.integrator.dt)

                # 3. Integrate TVB to get the new TVB state t_step
                state = self.integrator.scheme(state, self._dfun, node_coupling, local_coupling, stimulus)

                if numpy.any(numpy.isnan(state)) or numpy.any(numpy.isinf(state)):
                    raise ValueError("NaN or Inf values detected in simulator state!:\n%s" % str(state))

                # 4. Update the new TVB state t_step with the new spikeNet state t_step
                if updateTVBstateFromSpikeNet:
                    # SpikeNet state t_(step) -> TVB state t_(step)
                    # Update the new TVB state variable with the new SpikeNet state,
                    # including any necessary conversions from SpikeNet variables to TVB state,
                    # in a model specific manner
                    state = self.tvb_spikeNet_interface.spikeNet_state_to_tvb_state(state)
                    # TODO: Deprecate this since we have introduced TVB non-state variables
                    # SpikeNet state t_(step)-> TVB model parameter at time t_(step)
                    # Couple the SpikeNet state to some TVB model parameter,
                    # including any necessary conversions in a model specific manner
                    # !!! Deprecate it since we have introduced dynamic non-state variables !!!
                    # self.model = self.tvb_spikeNet_interface.spikeNet_state_to_tvb_parameter(self.model)
                    self.bound_and_clamp(state)

                # 5. Prepare next TVB time step integration

                # Prepare coupling and stimulus at time t_step, i.e., for next time iteration
                # and, therefore, for the new TVB state t_step+1, if any:
                node_coupling = self._loop_compute_node_coupling(step)
                self._loop_update_stimulus(step, stimulus)
                if self._spike_stimulus_fun:
                    self._apply_spike_stimulus(step)

                # Update any non-state variables and apply any boundaries again to the new state t_step:
                if self.model._update_non_state_variables:
                    state = self.update_state(state, node_coupling, local_coupling)

                # Now direct the new state t_step to history buffer and monitors
                self._loop_update_history(step, n_reg, state)
                output = self._loop_monitor_output(step, state)
                if output is not None:
                    yield output
                if self.PRINT_PROGRESSION_MESSAGE:
                    self._print_progression_message(step, n_steps)

            self.current_state = state
            self.current_step = self.current_step + n_steps - 1  # -1 : don't repeat last point

    # TODO: adjust function to compute fine scale resources' requirements as well, ...if you can! :)
