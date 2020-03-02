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
This is the main module of the simulator. It defines the Simulator class which
brings together all the structural and dynamic components necessary to define a
simulation and the method for running the simulation.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import time
import math
import numpy
import scipy.sparse
from tvb.basic.profile import TvbProfile
from tvb.datatypes import cortex, connectivity, patterns
from tvb.simulator import models, integrators, monitors, coupling
from .common import psutil, numpy_add_at
from .history import SparseHistory
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, List, Float


# TODO with refactor, this becomes more of a builder, since iterator will account for
# most of the runtime associated with a simulation.
class Simulator(HasTraits):
    """A Simulator assembles components required to perform simulations."""

    connectivity = Attr(
        field_type=connectivity.Connectivity,
        label="Long-range connectivity",
        default=None,
        required=True,
        doc="""A tvb.datatypes.Connectivity object which contains the
         structural long-range connectivity data (i.e., white-matter tracts). In
         combination with the ``Long-range coupling function`` it defines the inter-regional
         connections. These couplings undergo a time delay via signal propagation
         with a propagation speed of ``Conduction Speed``""")

    conduction_speed = Float(
        label="Conduction Speed",
        default=3.0,
        required=False,
        # range=basic.Range(lo=0.01, hi=100.0, step=1.0),
        doc="""Conduction speed for ``Long-range connectivity`` (mm/ms)""")

    coupling = Attr(
        field_type=coupling.Coupling,
        label="Long-range coupling function",
        default=coupling.Linear(),
        required=True,
        doc="""The coupling function is applied to the activity propagated
        between regions by the ``Long-range connectivity`` before it enters the local
        dynamic equations of the Model. Its primary purpose is to 'rescale' the
        incoming activity to a level appropriate to Model.""")

    surface = Attr(
        field_type=cortex.Cortex,
        label="Cortical surface",
        default=None,
        required=False,
        doc="""By default, a Cortex object which represents the
        cortical surface defined by points in the 3D physical space and their
        neighborhood relationship. In the current TVB version, when setting up a
        surface-based simulation, the option to configure the spatial spread of
        the ``Local Connectivity`` is available.""")

    stimulus = Attr(
        field_type=patterns.SpatioTemporalPattern,
        label="Spatiotemporal stimulus",
        default=None,
        required=False,
        doc="""A ``Spatiotemporal stimulus`` can be defined at the region or surface level.
        It's composed of spatial and temporal components. For region defined stimuli
        the spatial component is just the strength with which the temporal
        component is applied to each region. For surface defined stimuli,  a
        (spatial) function, with finite-support, is used to define the strength
        of the stimuli on the surface centred around one or more focal points.
        In the current version of TVB, stimuli are applied to the first state
        variable of the ``Local dynamic model``.""")

    model = Attr(
        field_type=models.Model,
        label="Local dynamic model",
        default=models.Generic2dOscillator(),
        required=True,
        doc="""A tvb.simulator.Model object which describe the local dynamic
        equations, their parameters, and, to some extent, where connectivity
        (local and long-range) enters and which state-variables the Monitors
        monitor. By default the 'Generic2dOscillator' model is used. Read the
        Scientific documentation to learn more about this model.""")

    integrator = Attr(
        field_type=integrators.Integrator,
        label="Integration scheme",
        default=integrators.HeunDeterministic(),
        required=True,
        doc="""A tvb.simulator.Integrator object which is
            an integration scheme with supporting attributes such as
            integration step size and noise specification for stochastic
            methods. It is used to compute the time courses of the model state
            variables.""")

    initial_conditions = NArray(
        label="Initial Conditions",
        required=False,
        doc="""Initial conditions from which the simulation will begin. By
        default, random initial conditions are provided. Needs to be the same shape
        as simulator 'history', ie, initial history function which defines the 
        minimal initial state of the network with time delays before time t=0. 
        If the number of time points in the provided array is insufficient the 
        array will be padded with random values based on the 'state_variables_range'
        attribute.""")

    monitors = List(
        of=monitors.Monitor,
        label="Monitor(s)",
        default=(monitors.TemporalAverage(),),
        doc="""A tvb.simulator.Monitor or a list of tvb.simulator.Monitor
        objects that 'know' how to record relevant data from the simulation. Two
        main types exist: 1) simple, spatial and temporal, reductions (subsets
        or averages); 2) physiological measurements, such as EEG, MEG and fMRI.
        By default the Model's specified variables_of_interest are returned,
        temporally downsampled from the raw integration rate to a sample rate of
        1024Hz.""")

    simulation_length = Float(
        label="Simulation Length (ms, s, m, h)",
        default=1000.0,  # ie 1 second
        required=True,
        doc="""The length of a simulation (default in milliseconds).""")

    history = None  # type: SparseHistory

    @property
    def good_history_shape(self):
        """Returns expected history shape."""
        n_reg = self.connectivity.number_of_regions
        shape = self.horizon, len(self.model.state_variables), n_reg, self.model.number_of_modes
        return shape

    calls = 0
    current_step = 0
    number_of_nodes = None
    _memory_requirement_guess = None
    _memory_requirement_census = None
    _storage_requirement = None
    _runtime = None

    # methods consist of
    # 1) generic configure
    # 2) component specific configure
    # 3) loop preparation
    # 4) loop step
    # 5) estimations

    @property
    def is_surface_simulation(self):
        if self.surface:
            return True
        return False

    def _configure_integrator_boundaries(self):
        if self.model.state_variable_boundaries is not None:
            indices = []
            boundaries = []
            for sv, sv_bounds in self.model.state_variable_boundaries.items():
                indices.append(self.model.state_variables.index(sv))
                boundaries.append(sv_bounds)
            sort_inds = numpy.argsort(indices)
            self.integrator.bounded_state_variable_indices = numpy.array(indices)[sort_inds]
            self.integrator.state_variable_boundaries = numpy.array(boundaries).astype("float64")[sort_inds]
        else:
            self.integrator.bounded_state_variable_indices = None
            self.integrator.state_variable_boundaries = None

    def preconfigure(self):
        """Configure just the basic fields, so that memory can be estimated."""
        self.connectivity.configure()
        if self.surface:
            self.surface.configure()
        if self.stimulus:
            self.stimulus.configure()
        self.coupling.configure()
        self.model.configure()
        self.integrator.configure()
        self._configure_integrator_boundaries()
        # monitors needs to be a list or tuple, even if there is only one...
        if not isinstance(self.monitors, (list, tuple)):
            self.monitors = [self.monitors]
        # Configure monitors
        for monitor in self.monitors:
            monitor.configure()
        # "Nodes" refers to either regions or vertices + non-cortical regions.
        if self.surface is None:
            self.number_of_nodes = self.connectivity.number_of_regions
            self.log.info('Region simulation with %d ROI nodes', self.number_of_nodes)
        else:
            rm = self.surface.region_mapping
            unmapped = self.connectivity.unmapped_indices(rm)
            self._regmap = numpy.r_[rm, unmapped]
            self.number_of_nodes = self._regmap.shape[0]
            self.log.info('Surface simulation with %d vertices + %d non-cortical, %d total nodes',
                     rm.size, unmapped.size, self.number_of_nodes)
        self._guesstimate_memory_requirement()

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
        if full_configure:
            # When run from GUI, preconfigure is run separately, and we want to avoid running that part twice
            self.preconfigure()
        # Make sure spatialised model parameters have the right shape (number_of_nodes, 1)
        # todo: this exclusion list is fragile, consider excluding declarative attrs that are not arrays
        excluded_params = ("state_variable_range", "state_variable_boundaries", "variables_of_interest",
                           "noise", "psi_table", "nerf_table", "gid")
        spatial_reshape = self.model.spatial_param_reshape
        for param in type(self.model).declarative_attrs:
            if param in excluded_params:
                continue
            # If it's a surface sim and model parameters were provided at the region level
            region_parameters = getattr(self.model, param)
            if self.surface is not None:
                if region_parameters.size == self.connectivity.number_of_regions:
                    new_parameters = region_parameters[self.surface.region_mapping].reshape(spatial_reshape)
                    setattr(self.model, param, new_parameters)
            region_parameters = getattr(self.model, param)
            if region_parameters.size == self.number_of_nodes:
                new_parameters = region_parameters.reshape(spatial_reshape)
                setattr(self.model, param, new_parameters)
        # Configure spatial component of any stimuli
        self._configure_stimuli()
        # Set delays, provided in physical units, in integration steps.
        self.connectivity.set_idelays(self.integrator.dt)
        self.horizon = self.connectivity.idelays.max() + 1
        # Reshape integrator.noise.nsig, if necessary.
        if isinstance(self.integrator, integrators.IntegratorStochastic):
            self._configure_integrator_noise()
        # Setup history
        self._configure_history(self.initial_conditions)
        # Configure Monitors to work with selected Model, etc...
        self._configure_monitors()
        # Estimate of memory usage.
        self._census_memory_requirement()
        # Allow user to chain configure to another call or assignment.
        return self

    def _handle_random_state(self, random_state):
        if random_state is not None:
            if isinstance(self.integrator, integrators.IntegratorStochastic):
                self.integrator.noise.random_stream.set_state(random_state)
                msg = "random_state supplied with seed %s"
                self.log.info(msg, self.integrator.noise.random_stream.get_state()[1][0])
            else:
                self.log.warn("random_state supplied for non-stochastic integration")

    def _prepare_local_coupling(self):
        if self.surface is None:
            local_coupling = 0.0
        else:
            if self.surface.coupling_strength.size == 1:
                local_coupling = (self.surface.coupling_strength[0] *
                                  self.surface.local_connectivity.matrix)
            elif self.surface.coupling_strength.size == self.surface.number_of_vertices:
                ind = numpy.arange(self.number_of_nodes, dtype=numpy.intc)
                vec_cs = numpy.zeros((self.number_of_nodes,))
                vec_cs[:self.surface.number_of_vertices] = self.surface.coupling_strength
                sp_cs = scipy.sparse.csc_matrix((vec_cs, (ind, ind)),
                                                shape=(self.number_of_nodes, self.number_of_nodes))
                local_coupling = sp_cs * self.surface.local_connectivity.matrix
            if local_coupling.shape[1] < self.number_of_nodes:
                # must match unmapped indices handling in preconfigure
                from scipy.sparse import csr_matrix, vstack, hstack
                nn = self.number_of_nodes
                npad = nn - local_coupling.shape[0]
                rpad = csr_matrix((local_coupling.shape[0], npad))
                bpad = csr_matrix((npad, nn))
                local_coupling = vstack([hstack([local_coupling, rpad]), bpad])
        return local_coupling

    def _prepare_stimulus(self):
        if self.stimulus is None:
            stimulus = 0.0
        else:
            time = numpy.r_[0.0 : self.simulation_length : self.integrator.dt]
            self.stimulus.configure_time(time.reshape((1, -1)))
            stimulus = numpy.zeros((self.model.nvar, self.number_of_nodes, 1))
            self.log.debug("stimulus shape is: %s", stimulus.shape)
        return stimulus

    def _loop_compute_node_coupling(self, step):
        """Compute delayed node coupling values."""
        coupling = self.coupling(step, self.history)
        if self.surface is not None:
            coupling = coupling[:, self._regmap]
        return coupling

    def _loop_update_stimulus(self, step, stimulus):
        """Update stimulus values for current time step."""
        if self.stimulus is not None:
            # TODO stim_step != current step
            stim_step = step - (self.current_step + 1)
            stimulus[self.model.stvar, :, :] = self.stimulus(stim_step).reshape((1, -1, 1))

    def _loop_update_history(self, step, n_reg, state):
        """Update history."""
        if self.surface is not None and state.shape[1] > self.connectivity.number_of_regions:
            region_state = numpy.zeros((n_reg, state.shape[0], state.shape[2]))         # temp (node, cvar, mode)
            numpy_add_at(region_state, self._regmap, state.transpose((1, 0, 2)))        # sum within region
            region_state /= numpy.bincount(self._regmap).reshape((-1, 1, 1))            # div by n node in region
            state = region_state.transpose((1, 0, 2))                                   # (cvar, node, mode)
        self.history.update(step, state)

    def _loop_monitor_output(self, step, state):
        observed = self.model.observe(state)
        output = [monitor.record(step, observed) for monitor in self.monitors]
        if any(outputi is not None for outputi in output):
            return output

    def __call__(self, simulation_length=None, random_state=None):
        """
        Return an iterator which steps through simulation time, generating monitor outputs.

        See the run method for a convenient way to collect all output in one call.

        :param simulation_length: Length of the simulation to perform in ms.
        :param random_state:  State of NumPy RNG to use for stochastic integration.
        :return: Iterator over monitor outputs.
        """

        self.calls += 1
        if simulation_length is not None:
            self.simulation_length = float(simulation_length)

        # intialization
        self._guesstimate_runtime()
        self._calculate_storage_requirement()
        self._handle_random_state(random_state)
        n_reg = self.connectivity.number_of_regions
        local_coupling = self._prepare_local_coupling()
        stimulus = self._prepare_stimulus()
        state = self.current_state

        # integration loop
        n_steps = int(math.ceil(self.simulation_length / self.integrator.dt))
        for step in range(self.current_step + 1, self.current_step + n_steps + 1):
            # needs implementing by hsitory + coupling?
            node_coupling = self._loop_compute_node_coupling(step)
            self._loop_update_stimulus(step, stimulus)
            state = self.integrator.scheme(state, self.model.dfun, node_coupling, local_coupling, stimulus)
            self._loop_update_history(step, n_reg, state)
            output = self._loop_monitor_output(step, state)
            if output is not None:
                yield output

        self.current_state = state
        self.current_step = self.current_step + n_steps

    def _configure_history(self, initial_conditions):
        """
        Set initial conditions for the simulation using either the provided
        initial_conditions or, if none are provided, the model's initial()
        method. This method is called durin the Simulator's __init__().

        Any initial_conditions that are provided as an argument are expected
        to have dimensions 1, 2, and 3 with shapse corresponding to the number
        of state_variables, nodes and modes, respectively. If the provided
        inital_conditions are shorter in time (dim=0) than the required history
        the model's initial() method is called to make up the difference.

        """
        rng = numpy.random
        if hasattr(self.integrator, 'noise'):
            rng = self.integrator.noise.random_stream
        # Default initial conditions
        if initial_conditions is None:
            n_time, n_svar, n_node, n_mode = self.good_history_shape
            self.log.info('Preparing initial history of shape %r using model.initial()', self.good_history_shape)
            if self.surface is not None:
                n_node = self.number_of_nodes
            history = self.model.initial(self.integrator.dt, (n_time, n_svar, n_node, n_mode), rng)
        # ICs provided
        else:
            # history should be [timepoints, state_variables, nodes, modes]
            self.log.info('Using provided initial history of shape %r', initial_conditions.shape)
            n_time, n_svar, n_node, n_mode = ic_shape = initial_conditions.shape
            nr = self.connectivity.number_of_regions
            if self.surface is not None and n_node == nr:
                initial_conditions = initial_conditions[:, :, self._regmap]
                return self._configure_history(initial_conditions)
            elif ic_shape[1:] != self.good_history_shape[1:]:
                raise ValueError("Incorrect history sample shape %s, expected %s"
                                 % (ic_shape[1:], self.good_history_shape[1:]))
            else:
                if ic_shape[0] >= self.horizon:
                    self.log.debug("Using last %d time-steps for history.", self.horizon)
                    history = initial_conditions[-self.horizon:, :, :, :].copy()
                else:
                    self.log.debug('Padding initial conditions with model.initial')
                    history = self.model.initial(self.integrator.dt, self.good_history_shape, rng)
                    shift = self.current_step % self.horizon
                    history = numpy.roll(history, -shift, axis=0)
                    history[:ic_shape[0], :, :, :] = initial_conditions
                    history = numpy.roll(history, shift, axis=0)
                self.current_step += ic_shape[0] - 1

        if self.integrator.state_variable_boundaries is not None:
            self.integrator.bound_state(numpy.swapaxes(history, 0, 1))
        self.log.info('Final initial history shape is %r', history.shape)

        # create initial state from history
        self.current_state = history[self.current_step % self.horizon].copy()
        self.log.debug('initial state has shape %r' % (self.current_state.shape, ))
        if self.surface is not None and history.shape[2] > self.connectivity.number_of_regions:
            n_reg = self.connectivity.number_of_regions
            (nt, ns, _, nm), ax = history.shape, (2, 0, 1, 3)
            region_history = numpy.zeros((nt, ns, n_reg, nm))
            numpy_add_at(region_history.transpose(ax), self._regmap, history.transpose(ax))
            region_history /= numpy.bincount(self._regmap).reshape((-1, 1))
            history = region_history
        # create history query implementation
        self.history = SparseHistory(
            self.connectivity.weights,
            self.connectivity.idelays,
            self.model.cvar,
            self.model.number_of_modes
        )
        # initialize its buffer
        self.history.initialize(history)

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

        noise = self.integrator.noise        

        if self.integrator.noise.ntau > 0.0:
            self.integrator.noise.configure_coloured(self.integrator.dt,
                                                     self.good_history_shape[1:])
        else:
            self.integrator.noise.configure_white(self.integrator.dt,
                                                  self.good_history_shape[1:])

        if self.surface is not None:
            if self.integrator.noise.nsig.size == self.connectivity.number_of_regions:
                self.integrator.noise.nsig = self.integrator.noise.nsig[self.surface.region_mapping]
            elif self.integrator.noise.nsig.size == self.model.nvar * self.connectivity.number_of_regions:
                self.integrator.noise.nsig = self.integrator.noise.nsig[:, self.surface.region_mapping]

        good_nsig_shape = (self.model.nvar, self.number_of_nodes,
                           self.model.number_of_modes)
        nsig = self.integrator.noise.nsig
        self.log.debug("Given noise shape is %s", nsig.shape)
        if nsig.shape in (good_nsig_shape, (1,)):
            return
        elif nsig.shape == (self.model.nvar, ):
            nsig = nsig.reshape((self.model.nvar, 1, 1))
        elif nsig.shape == (self.number_of_nodes, ):
            nsig = nsig.reshape((1, self.number_of_nodes, 1))
        elif nsig.shape == (self.model.nvar, self.number_of_nodes):
            nsig = nsig.reshape((self.model.nvar, self.number_of_nodes, 1))
        else:
            msg = "Bad Simulator.integrator.noise.nsig shape: %s"
            self.log.error(msg % str(nsig.shape))

        self.log.debug("Corrected noise shape is %s", nsig.shape)
        self.integrator.noise.nsig = nsig

    def _configure_monitors(self):
        """ Configure the requested Monitors for this Simulator """
        # Coerce to list if required
        if not isinstance(self.monitors, (list, tuple)):
            self.monitors = [self.monitors]
        # Configure monitors
        for monitor in self.monitors:
            monitor.config_for_sim(self)

    def _configure_stimuli(self):
        """ Configure the defined Stimuli for this Simulator """
        if self.stimulus is not None:
            if self.surface:
                self.stimulus.configure_space(self.surface.region_mapping)
            else:
                self.stimulus.configure_space()

    # used by simulator adaptor
    def memory_requirement(self):
        """
        Return an estimated of the memory requirements (Bytes) for this
        simulator's current configuration.
        """
        self._guesstimate_memory_requirement()
        return self._memory_requirement_guess

    # appears to be unused
    def runtime(self, simulation_length):
        """
        Return an estimated run time (seconds) for the simulator's current 
        configuration and a specified simulation length.

        """
        self.simulation_length = simulation_length
        self._guesstimate_runtime()
        return self._runtime

    # used by simulator adaptor
    def storage_requirement(self):
        """
        Return an estimated storage requirement (Bytes) for the simulator's
        current configuration and a specified simulation length.

        """
        self._calculate_storage_requirement()
        return self._storage_requirement

    def _guesstimate_memory_requirement(self):
        """
        guesstimate the memory required for this simulator.

        Guesstimate is based on the shape of the dominant arrays, and as such 
        can operate before configuration.

        NOTE: Assumes returned/yeilded data is in some sense "taken care of" in
            the world outside the simulator, and so doesn't consider it, making
            the simulator's history, and surface if present, the dominant 
            memory pigs...

        """
        if self.surface:
            number_of_nodes = self.surface.number_of_vertices
        else:
            number_of_nodes = self.connectivity.number_of_regions

        number_of_regions = self.connectivity.number_of_regions

        magic_number = 2.42  # Current guesstimate is low by about a factor of 2, seems safer to over estimate...
        bits_64 = 8.0  # Bytes
        bits_32 = 4.0  # Bytes
        # NOTE: The speed hack for getting the first element of hist shape should
        #      partially resolves calling of this method with a non-configured
        #     connectivity, there remains the less common issue if no tract_lengths...
        hist_shape = (self.connectivity.tract_lengths.max() / (self.conduction_speed or
                                                               self.connectivity.speed or 3.0) / self.integrator.dt,
                      self.model.nvar, number_of_nodes, 
                      self.model.number_of_modes)
        self.log.debug("Estimated history shape is %r", hist_shape)

        memreq = numpy.prod(hist_shape) * bits_64
        if self.surface:
            memreq += self.surface.number_of_triangles * 3 * bits_32 * 2  # normals
            memreq += self.surface.number_of_vertices * 3 * bits_64 * 2   # normals
            memreq += number_of_nodes * number_of_regions * bits_64 * 4   # region_mapping, region_average, region_sum
            # ???memreq += self.surface.local_connectivity.matrix.nnz * 8

        if not hasattr(self.monitors, '__len__'):
            self.monitors = [self.monitors]

        for monitor in self.monitors:
            if not isinstance(monitor, monitors.Bold):
                stock_shape = (monitor.period / self.integrator.dt, 
                               len(self.model.variables_of_interest),
                               number_of_nodes,
                               self.model.number_of_modes)
                memreq += numpy.prod(stock_shape) * bits_64
                if hasattr(monitor, "sensors"):
                    try:
                        memreq += number_of_nodes * monitor.sensors.number_of_sensors * bits_64  # projection_matrix
                    except AttributeError:
                        self.log.debug("No sensors specified, guessing memory based on default EEG.")
                        memreq += number_of_nodes * 62.0 * bits_64

            else:
                stock_shape = (monitor.hrf_length * monitor._stock_sample_rate,
                               len(self.model.variables_of_interest),
                               number_of_nodes,
                               self.model.number_of_modes)
                interim_stock_shape = (1.0 / (2.0 ** -2 * self.integrator.dt),
                                       len(self.model.variables_of_interest),
                                       number_of_nodes,
                                       self.model.number_of_modes)
                memreq += numpy.prod(stock_shape) * bits_64
                memreq += numpy.prod(interim_stock_shape) * bits_64

        if psutil and memreq > psutil.virtual_memory().total:
            self.log.warning("There may be insufficient memory for this simulation.")

        self._memory_requirement_guess = magic_number * memreq
        msg = "Memory requirement estimate: simulation will need about %.1f MB"
        self.log.info(msg, self._memory_requirement_guess / 2**20)

    def _census_memory_requirement(self):
        """
        Guesstimate the memory required for this simulator. 

        Guesstimate is based on a census of the dominant arrays after the
        simulator has been configured.

        NOTE: Assumes returned/yeilded data is in some sense "taken care of" in
            the world outside the simulator, and so doesn't consider it, making
            the simulator's history, and surface if present, the dominant 
            memory pigs...

        """
        magic_number = 2.42  # Current guesstimate is low by about a factor of 2, seems safer to over estimate...
        memreq = self.history.nbytes
        try:
            memreq += self.surface.triangles.nbytes * 2
            memreq += self.surface.vertices.nbytes * 2
            memreq += self.surface.region_mapping.nbytes * self.number_of_nodes * 8. * 4  # region_average, region_sum
            memreq += self.surface.local_connectivity.matrix.nnz * 8
        except AttributeError:
            pass

        for monitor in self.monitors:
            memreq += monitor._stock.nbytes
            if isinstance(monitor, monitors.Bold):
                memreq += monitor._interim_stock.nbytes

        if psutil and memreq > psutil.virtual_memory().total:
            self.log.warning("Memory estimate exceeds total available RAM.")

        self._memory_requirement_census = magic_number * memreq
        # import pdb; pdb.set_trace()
        msg = "Memory requirement census: simulation will need about %.1f MB"
        self.log.info(msg % (self._memory_requirement_census / 1048576.0))

    def _guesstimate_runtime(self):
        """
        Estimate the runtime for this simulator.

        Spread in parallel executions of larger arrays means this will be an over-estimation,
        or rather a single threaded estimation...
        Different choice of integrators and monitors has an additional effect,
        on the magic number though relatively minor

        """
        magic_number = 6.57e-06  # seconds
        self._runtime = (magic_number * self.number_of_nodes * self.model.nvar * self.model.number_of_modes *
                         self.simulation_length / self.integrator.dt)
        msg = "Simulation runtime should be about %0.3f seconds"
        self.log.info(msg, self._runtime)

    def _calculate_storage_requirement(self):
        """
        Calculate the storage requirement for the simulator, configured with
        models, monitors, etc being run for a particular simulation length. 
        While this is only approximate, it is far more reliable/accurate than
        the memory and runtime guesstimates.
        """
        self.log.info("Calculating storage requirement for ...")
        strgreq = 0
        for monitor in self.monitors:
            # Avoid division by zero for monitor not yet configured
            # (in framework this is executed, when only preconfigure has been called):
            current_period = monitor.period or self.integrator.dt
            strgreq += (TvbProfile.current.MAGIC_NUMBER * self.simulation_length *
                        self.number_of_nodes * self.model.nvar *
                        self.model.number_of_modes / current_period)
        self.log.info("Calculated storage requirement for simulation: %d " % int(strgreq))
        self._storage_requirement = int(strgreq)

    def run(self, **kwds):
        """Convenience method to call the simulator with **kwds and collect output data."""
        ts, xs = [], []
        for _ in self.monitors:
            ts.append([])
            xs.append([])
        wall_time_start = time.time()
        for data in self(**kwds):
            for tl, xl, t_x in zip(ts, xs, data):
                if t_x is not None:
                    t, x = t_x
                    tl.append(t)
                    xl.append(x)
        elapsed_wall_time = time.time() - wall_time_start
        self.log.info("%.3f s elapsed, %.3fx real time", elapsed_wall_time,
                      elapsed_wall_time * 1e3 / self.simulation_length)
        for i in range(len(ts)):
            ts[i] = numpy.array(ts[i])
            xs[i] = numpy.array(xs[i])
        return list(zip(ts, xs))
