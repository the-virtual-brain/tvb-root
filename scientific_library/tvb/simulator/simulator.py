# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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

import numpy
import scipy.sparse as sparse
from tvb.basic.profile import TvbProfile
import tvb.basic.traits.core as core
import tvb.basic.traits.types_basic as basic
from tvb.basic.filters.chain import UIFilter, FilterChain
from tvb.datatypes.cortex import Cortex

import tvb.simulator.models as models_module
import tvb.simulator.integrators as integrators_module
import tvb.simulator.monitors as monitors_module
import tvb.simulator.coupling as coupling_module

import tvb.datatypes.arrays as arrays_dtype
import tvb.datatypes.surfaces as surfaces_dtype
import tvb.datatypes.connectivity as connectivity_dtype
#import tvb.datatypes.coupling as coupling_dtype
import tvb.datatypes.patterns as patterns_dtype

from tvb.simulator.common import psutil, get_logger, numpy_add_at
from tvb.simulator.history import get_state

LOG = get_logger(__name__)


#from tvb.simulator.common import iround

class Simulator(core.Type):
    """
    The Simulator class coordinates classes from all other modules in the
    simulator package in order to perform simulations. 

    In general, it is necessary to initialiaze a simulator with the desired
    components and then call the simulator in a loop to obtain simulation
    data:
    
    >>> sim = Simulator(...)
    >>> for output in sim(simulation_length=1000):
            ...
    
    Please refer to the user guide and the demos for more detail.


    .. #Currently there seems to be a clash betwen traits and autodoc, autodoc
    .. #can't find the methods of the class, the class specific names below get
    .. #us around this...
    .. automethod:: Simulator.__init__
    .. automethod:: Simulator.configure
    .. automethod:: Simulator.__call__
    .. automethod:: Simulator.configure_history
    .. automethod:: Simulator.configure_integrator_noise
    .. automethod:: Simulator.memory_requirement
    .. automethod:: Simulator.runtime
    .. automethod:: Simulator.storage_requirement


    """

    connectivity = connectivity_dtype.Connectivity(
        label="Long-range connectivity",
        default=None,
        order=1,
        required=True,
        filters_ui=[UIFilter(linked_elem_name="region_mapping_data",
                             linked_elem_field=FilterChain.datatype + "._connectivity",
                             linked_elem_parent_name="surface",
                             linked_elem_parent_option=None),
                    UIFilter(linked_elem_name="region_mapping",
                             linked_elem_field=FilterChain.datatype + "._connectivity",
                             linked_elem_parent_name="monitors",
                             linked_elem_parent_option="EEG"),
                    UIFilter(linked_elem_name="region_mapping",
                             linked_elem_field=FilterChain.datatype + "._connectivity",
                             linked_elem_parent_name="monitors",
                             linked_elem_parent_option="MEG"),
                    UIFilter(linked_elem_name="region_mapping",
                             linked_elem_field=FilterChain.datatype + "._connectivity",
                             linked_elem_parent_name="monitors",
                             linked_elem_parent_option="iEEG")],
        doc="""A tvb.datatypes.Connectivity object which contains the
        structural long-range connectivity data (i.e., white-matter tracts). In
        combination with the ``Long-range coupling function`` it defines the inter-regional
        connections. These couplings undergo a time delay via signal propagation 
        with a propagation speed of ``Conduction Speed``""")

    conduction_speed = basic.Float(
        label="Conduction Speed",
        default=3.0,
        order=2,
        required=False,
        range=basic.Range(lo=0.01, hi=100.0, step=1.0),
        doc="""Conduction speed for ``Long-range connectivity`` (mm/ms)""")

    coupling = coupling_module.Coupling(
        label="Long-range coupling function",
        default=coupling_module.Linear(),
        required=True,
        order=2,
        doc="""The coupling function is applied to the activity propagated
        between regions by the ``Long-range connectivity`` before it enters the local
        dynamic equations of the Model. Its primary purpose is to 'rescale' the
        incoming activity to a level appropriate to Model.""")

    surface = Cortex(
        label="Cortical surface",
        default=None,
        order=3,
        required=False,
        filters_backend=FilterChain(fields=[FilterChain.datatype + '._valid_for_simulations'],
                                    operations=["=="], values=[True]),
        filters_ui=[UIFilter(linked_elem_name="projection_matrix_data",
                             linked_elem_field=FilterChain.datatype + "._sources",
                             linked_elem_parent_name="monitors",
                             linked_elem_parent_option="EEG"),
                    UIFilter(linked_elem_name="local_connectivity",
                             linked_elem_field=FilterChain.datatype + "._surface",
                             linked_elem_parent_name="surface",
                             linked_elem_parent_option=None)],
        doc="""By default, a tvb.datatypes.Cortex object which represents the
        cortical surface defined by points in the 3D physical space and their 
        neighborhood relationship. In the current TVB version, when setting up a 
        surface-based simulation, the option to configure the spatial spread of 
        the ``Local Connectivity`` is available.""")

    stimulus = patterns_dtype.SpatioTemporalPattern(
        label="Spatiotemporal stimulus",
        default=None,
        order=4,
        required=False,
        doc="""A ``Spatiotemporal stimulus`` can be defined at the region or surface level.
        It's composed of spatial and temporal components. For region defined stimuli
        the spatial component is just the strength with which the temporal
        component is applied to each region. For surface defined stimuli,  a
        (spatial) function, with finite-support, is used to define the strength 
        of the stimuli on the surface centred around one or more focal points. 
        In the current version of TVB, stimuli are applied to the first state 
        variable of the ``Local dynamic model``.""")

    model = models_module.Model(
        label="Local dynamic model",
        default=models_module.Generic2dOscillator,
        required=True,
        order=5,
        doc="""A tvb.simulator.Model object which describe the local dynamic
        equations, their parameters, and, to some extent, where connectivity
        (local and long-range) enters and which state-variables the Monitors
        monitor. By default the 'Generic2dOscillator' model is used. Read the 
        Scientific documentation to learn more about this model.""")

    integrator = integrators_module.Integrator(
        label="Integration scheme",
        default=integrators_module.HeunDeterministic,
        required=True,
        order=6,
        doc="""A tvb.simulator.Integrator object which is
            an integration scheme with supporting attributes such as 
            integration step size and noise specification for stochastic 
            methods. It is used to compute the time courses of the model state 
            variables.""")

    initial_conditions = arrays_dtype.FloatArray(
        label="Initial Conditions",
        default=None,
        order=-1,
        required=False,
        doc="""Initial conditions from which the simulation will begin. By
        default, random initial conditions are provided. Needs to be the same shape
        as simulator 'history', ie, initial history function which defines the 
        minimal initial state of the network with time delays before time t=0. 
        If the number of time points in the provided array is insufficient the 
        array will be padded with random values based on the 'state_variables_range'
        attribute.""")

    monitors = monitors_module.Monitor(
        label="Monitor(s)",
        default=monitors_module.TemporalAverage,
        required=True,
        order=8,
        select_multiple=True,
        doc="""A tvb.simulator.Monitor or a list of tvb.simulator.Monitor
        objects that 'know' how to record relevant data from the simulation. Two
        main types exist: 1) simple, spatial and temporal, reductions (subsets
        or averages); 2) physiological measurements, such as EEG, MEG and fMRI.
        By default the Model's specified variables_of_interest are returned,
        temporally downsampled from the raw integration rate to a sample rate of
        1024Hz.""")

    simulation_length = basic.Float(
        label="Simulation Length (ms)",
        default=1000.0,     # ie 1 second
        required=True,
        order=9,
        doc="""The length of a simulation in milliseconds (ms).""")


    def __init__(self, **kwargs): 
        """
        Use the base class' mechanisms to initialise the traited attributes 
        declared above, overriding defaults with any provided keywords. Then
        declare any non-traited attributes.

        """
        super(Simulator, self).__init__(**kwargs) 
        LOG.debug(str(kwargs))

        self.calls = 0
        self.current_step = 0

        self.number_of_nodes = None
        self.horizon = None
        self.good_history_shape = None
        self.history = None
        self._memory_requirement_guess = None
        self._memory_requirement_census = None
        self._storage_requirement = None
        self._runtime = None

    def __str__(self):
        return "Simulator(**kwargs)"

    def preconfigure(self):
        """
        Configure just the basic fields, so that memory can be estimated
        """
        self.connectivity.configure()

        if self.surface:
            self.surface.configure()

        if self.stimulus:
            self.stimulus.configure()

        self.coupling.configure()
        self.model.configure()
        self.integrator.configure()

        # monitors needs to be a list or tuple, even if there is only one...
        if not isinstance(self.monitors, (list, tuple)):
            self.monitors = [self.monitors]

        # Configure monitors
        for monitor in self.monitors:
            monitor.configure()

        ##------------- Now the the interdependant configuration -------------##

        #"Nodes" refers to either regions or vertices + non-cortical regions.
        if self.surface is None:
            self.number_of_nodes = self.connectivity.number_of_regions
        else:
            rm = self.surface.region_mapping
            self._regmap = numpy.r_[rm, self.connectivity.unmapped_indices(rm)]
            self.number_of_nodes = self._regmap.shape[0]

        self._guesstimate_memory_requirement()

    def configure(self, full_configure=True):
        """
        The first step of configuration is to run the configure methods of all
        the Simulator's components, ie its traited attributes.

        Configuration of a Simulator primarily consists of calculating the
        attributes, etc, which depend on the combinations of the Simulator's
        traited attributes (keyword args).

        Converts delays from physical time units into integration steps
        and updates attributes that depend on combinations of the 6 inputs.
        """
        if full_configure:
            # When run from GUI, preconfigure is run separately, and we want to avoid running that part twice
            self.preconfigure()

        #Make sure spatialised model parameters have the right shape (number_of_nodes, 1)
        excluded_params = ("state_variable_range", "variables_of_interest", "noise", "psi_table", "nerf_table")

        for param in self.model.trait.keys():
            if param in excluded_params:
                continue
            #If it's a surface sim and model parameters were provided at the region level
            region_parameters = getattr(self.model, param)
            if self.surface is not None:
                if region_parameters.size == self.connectivity.number_of_regions:
                    new_parameters = region_parameters[self.surface.region_mapping].reshape((-1, 1))
                    setattr(self.model, param, new_parameters)
            region_parameters = getattr(self.model, param)
            if region_parameters.size == self.number_of_nodes:
                new_parameters = region_parameters.reshape((-1, 1))
                setattr(self.model, param, new_parameters)

        #Configure spatial component of any stimuli
        self.configure_stimuli()

        #Set delays, provided in physical units, in integration steps.
        self.connectivity.set_idelays(self.integrator.dt)

        self.horizon = numpy.max(self.connectivity.idelays) + 1
        LOG.info("horizon is %d steps" % self.horizon)

        # workspace -- minimal state of network with delays
        self.good_history_shape = (self.horizon, self.model.nvar,
                                   self.number_of_nodes,
                                   self.model.number_of_modes)
        msg = "%s: History shape will be: %s"
        LOG.debug(msg % (repr(self), str(self.good_history_shape)))

        #Reshape integrator.noise.nsig, if necessary.
        if isinstance(self.integrator, integrators_module.IntegratorStochastic):
            self.configure_integrator_noise()

        self.configure_history(self.initial_conditions)

        #Configure Monitors to work with selected Model, etc...
        self.configure_monitors()

        #Estimate of memory usage. 
        self._census_memory_requirement()

        return self


    def __call__(self, simulation_length=None, random_state=None):
        """
        When a Simulator is called it returns an iterator.

        kwargs:

        ``simulation_length``:
           total time of simulation

        ``random_state``: 
           a state for the NumPy random number generator, saved from a previous 
           call to permit consistent continuation of a simulation.

        """

        self.calls += 1
        self.simulation_length = simulation_length or self.simulation_length

        # Estimate run time and storage requirements, with logging.
        self._guesstimate_runtime()
        self._calculate_storage_requirement()

        if random_state is not None:
            if isinstance(self.integrator, integrators_module.IntegratorStochastic):
                self.integrator.noise.random_stream.set_state(random_state)
                msg = "random_state supplied with seed %s"
                LOG.info(msg, self.integrator.noise.random_stream.get_state()[1][0])
            else:
                LOG.warn("random_state supplied for non-stochastic integration")

        # number of steps to perform integrqtion
        int_steps = int(simulation_length / self.integrator.dt)
        msg = 'sim length %f ms requires %d steps'
        LOG.info(msg, simulation_length, int_steps)

        # locals for cleaner code.
        ncvar = len(self.model.cvar)
        number_of_regions = self.connectivity.number_of_regions

        # Exact dtypes and alignment are required by c speedups. Once we have history objects these will be encapsulated
        # cvar index array broadcastable to nodes, cvars, nodes
        cvar = numpy.array(self.model.cvar[numpy.newaxis, :, numpy.newaxis], dtype=numpy.intc)
        LOG.debug("%s: cvar is: %s" % (str(self), str(cvar)))

        # idelays array broadcastable to nodes, cvars, nodes
        idelays = numpy.array(self.connectivity.idelays[:, numpy.newaxis, :], dtype=numpy.intc, order='c')
        LOG.debug("%s: idelays shape is: %s" % (str(self), str(idelays.shape)))

        # weights array broadcastable to nodes, cva, nodes, modes
        weights = self.connectivity.weights[:, numpy.newaxis, :, numpy.newaxis]
        LOG.debug("%s: weights shape is: %s" % (str(self), str(weights.shape)))

        # node_ids broadcastable to nodes, cvars, nodes
        node_ids = numpy.array(numpy.arange(number_of_regions)[numpy.newaxis, numpy.newaxis, :], dtype=numpy.intc)
        LOG.debug("%s: node_ids shape is: %s"%(str(self), str(node_ids.shape)))

        if self.surface is None:
            local_coupling = 0.0
        else:
            (nt, ns, _, nm), ax = self.history.shape, (2, 0, 1, 3)
            region_history = numpy.zeros((nt, ns, number_of_regions, nm))
            numpy_add_at(region_history.transpose(ax), self._regmap, self.history.transpose(ax))
            region_history /= numpy.bincount(self._regmap).reshape((-1, 1))
            if self.surface.coupling_strength.size == 1:
                local_coupling = (self.surface.coupling_strength[0] *
                                  self.surface.local_connectivity.matrix)
            elif self.surface.coupling_strength.size == self.surface.number_of_vertices:
                ind = numpy.arange(self.number_of_nodes, dtype=int)
                vec_cs = numpy.zeros((self.number_of_nodes,))
                vec_cs[:self.surface.number_of_vertices] = self.surface.coupling_strength
                sp_cs = sparse.csc_matrix((vec_cs, (ind, ind)),
                                          shape=(self.number_of_nodes, self.number_of_nodes))
                local_coupling = sp_cs * self.surface.local_connectivity.matrix

        if self.stimulus is None:
            stimulus = 0.0
        else:
            self.stimulus.configure_time(numpy.r_[:simulation_length:self.integrator.dt].reshape((1, -1)))
            stimulus = numpy.zeros((self.model.nvar, self.number_of_nodes, 1))
            LOG.debug("stimulus shape is: %s", stimulus.shape)

        # initial state, history[timepoint[0], state_variables, nodes, modes]
        state = self.history[self.current_step % self.horizon, :]
        LOG.debug("state shape is: %s", state.shape)

        delayed_state = numpy.zeros((number_of_regions, ncvar, number_of_regions, self.model.number_of_modes))

        # integration loop
        for step in xrange(self.current_step + 1, self.current_step+int_steps+1):

            # compute afferent coupling
            time_indices = (step - 1 - idelays) % self.horizon
            if self.surface is None:
                get_state(self.history, time_indices, cvar, node_ids, out=delayed_state)
                node_coupling = self.coupling(weights, state[self.model.cvar], delayed_state)
            else:
                get_state(region_history, time_indices, cvar, node_ids, out=delayed_state)
                region_coupling = self.coupling(weights, region_history[(step - 1) % self.horizon, self.model.cvar], delayed_state)
                node_coupling = region_coupling[:, self._regmap]

            # stimulus pattern at this time point
            if self.stimulus is not None:
                stim_step = step - (self.current_step + 1) # TODO stim_step != current step ??
                stimulus[self.model.cvar, :, :] = self.stimulus(stim_step).reshape((1, -1, 1))

            # apply integration scheme
            state = self.integrator.scheme(state, self.model.dfun, node_coupling, local_coupling, stimulus)

            # update full history & region history if applicable
            self.history[step % self.horizon, :] = state
            if self.surface is not None:
                region_state = numpy.zeros((number_of_regions, state.shape[0], state.shape[2]))
                numpy_add_at(region_state, self._regmap, state.transpose((1, 0, 2)))
                region_state /= numpy.bincount(self._regmap).reshape((-1, 1, 1))
                region_history[step % self.horizon, :] = region_state.transpose((1, 0, 2))

            # record monitor output & forward to caller
            output = [monitor.record(step, state) for monitor in self.monitors]
            if any(outputi is not None for outputi in output):
                yield output

        # This -1 is here for not repeating the point on resume
        self.current_step = self.current_step + int_steps - 1


    def configure_history(self, initial_conditions=None):
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

        history = self.history
        if initial_conditions is None:
            msg = "%s: Setting default history using model's initial() method."
            LOG.info(msg % str(self))
            history = self.model.initial(self.integrator.dt, self.good_history_shape)
        else:
            # history should be [timepoints, state_variables, nodes, modes]
            LOG.info("%s: Received initial conditions as arg." % str(self))
            ic_shape = initial_conditions.shape
            if ic_shape[1:] != self.good_history_shape[1:]:
                msg = "%s: bad initial_conditions[1:] shape %s, should be %s"
                msg %= self, ic_shape[1:], self.good_history_shape[1:]
                raise ValueError(msg)
            else:
                if ic_shape[0] >= self.horizon:
                    msg = "%s: Using last %s time-steps for history."
                    LOG.info(msg % (str(self), self.horizon))
                    history = initial_conditions[-self.horizon:, :, :, :].copy()
                else:
                    msg = "%s: initial_conditions shorter than required."
                    LOG.info(msg % str(self))
                    msg = "%s: Using model's initial() method for difference."
                    LOG.info(msg % str(self))
                    history = self.model.initial(self.integrator.dt, self.good_history_shape)
                    csmh = self.current_step % self.horizon
                    history = numpy.roll(history, -csmh, axis=0)
                    history[:ic_shape[0], :, :, :] = initial_conditions
                    history = numpy.roll(history, csmh, axis=0)
                self.current_step += ic_shape[0] - 1
            msg = "%s: history shape is: %s"
            LOG.debug(msg % (str(self), str(history.shape)))
        self.history = history

    def configure_integrator_noise(self):
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
        LOG.debug("Simulator.integrator.noise.nsig shape: %s" % str(nsig.shape))
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
            LOG.error(msg % str(nsig.shape))

        LOG.debug("Simulator.integrator.noise.nsig shape: %s" % str(nsig.shape))
        self.integrator.noise.nsig = nsig


    def configure_monitors(self):
        """ Configure the requested Monitors for this Simulator """
        if not isinstance(self.monitors, (list, tuple)):
            self.monitors = [self.monitors]

        # Configure monitors 
        for monitor in self.monitors:
            monitor.config_for_sim(self)


    def configure_stimuli(self):
        """ Configure the defined Stimuli for this Simulator """
        if self.stimulus is not None:
            if self.surface:
                self.stimulus.configure_space(self.surface.region_mapping)
            else:
                self.stimulus.configure_space()


    def memory_requirement(self):
        """
        Return an estimated of the memory requirements (Bytes) for this
        simulator's current configuration.
        """
        self._guesstimate_memory_requirement()
        return self._memory_requirement_guess


    def runtime(self, simulation_length):
        """
        Return an estimated run time (seconds) for the simulator's current 
        configuration and a specified simulation length.

        """
        self.simulation_length = simulation_length
        self._guesstimate_runtime()
        return self._runtime


    def storage_requirement(self, simulation_length):
        """
        Return an estimated storage requirement (Bytes) for the simulator's
        current configuration and a specified simulation length.

        """
        self.simulation_length = simulation_length
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
        #NOTE: The speed hack for getting the first element of hist shape should
        #      partially resolves calling of this method with a non-configured
        #     connectivity, there remains the less common issue if no tract_lengths...
        hist_shape = (self.connectivity.tract_lengths.max() / (self.conduction_speed or
                                                               self.connectivity.speed or 3.0) / self.integrator.dt,
                      self.model.nvar, number_of_nodes, 
                      self.model.number_of_modes)
        memreq = numpy.prod(hist_shape) * bits_64
        if self.surface:
            memreq += self.surface.number_of_triangles * 3 * bits_32 * 2  # normals
            memreq += self.surface.number_of_vertices * 3 * bits_64 * 2   # normals
            memreq += number_of_nodes * number_of_regions * bits_64 * 4   # region_mapping, region_average, region_sum
            #???memreq += self.surface.local_connectivity.matrix.nnz * 8

        if not isinstance(self.monitors, (list, tuple)):
            monitors = [self.monitors]
        else:
            monitors = self.monitors
        for monitor in monitors:
            if not isinstance(monitor, monitors_module.Bold):
                stock_shape = (monitor.period / self.integrator.dt, 
                               self.model.variables_of_interest.shape[0], 
                               number_of_nodes,
                               self.model.number_of_modes)
                memreq += numpy.prod(stock_shape) * bits_64
                if hasattr(monitor, "sensors"):
                    try:
                        memreq += number_of_nodes * monitor.sensors.number_of_sensors * bits_64  # projection_matrix
                    except AttributeError:
                        LOG.debug("No sensors specified, guessing memory based on default EEG.")
                        memreq += number_of_nodes * 62.0 * bits_64

            else:
                stock_shape = (monitor.hrf_length * monitor._stock_sample_rate,
                               self.model.variables_of_interest.shape[0],
                               number_of_nodes,
                               self.model.number_of_modes)
                interim_stock_shape = (1.0 / (2.0 ** -2 * self.integrator.dt),
                                       self.model.variables_of_interest.shape[0],
                                       number_of_nodes,
                                       self.model.number_of_modes)
                memreq += numpy.prod(stock_shape) * bits_64
                memreq += numpy.prod(interim_stock_shape) * bits_64

        if psutil and memreq > psutil.virtual_memory().total:
            LOG.error("This is gonna get ugly...")

        self._memory_requirement_guess = magic_number * memreq
        msg = "Memory requirement guesstimate: simulation will need about %.1f MB"
        LOG.info(msg % (self._memory_requirement_guess / 1048576.0))


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
            memreq += self.surface.eeg_projection.nbytes
            memreq += self.surface.local_connectivity.matrix.nnz * 8
        except AttributeError:
            pass

        for monitor in self.monitors:
            memreq += monitor._stock.nbytes
            if isinstance(monitor, monitors_module.Bold):
                memreq += monitor._interim_stock.nbytes

        if psutil and memreq > psutil.virtual_memory().total:
            LOG.error("This is gonna get ugly...")

        self._memory_requirement_census = magic_number * memreq
        #import pdb; pdb.set_trace()
        msg = "Memory requirement census: simulation will need about %.1f MB"
        LOG.info(msg % (self._memory_requirement_census / 1048576.0))


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
        msg = "Simulation single-threaded runtime should be about %s seconds!"
        LOG.info(msg % str(int(self._runtime)))


    def _calculate_storage_requirement(self):
        """
        Calculate the storage requirement for the simulator, configured with
        models, monitors, etc being run for a particular simulation length. 
        While this is only approximate, it is far more reliable/accurate than
        the memory and runtime guesstimates.
        """
        LOG.info("Calculating storage requirement for ...")
        strgreq = 0
        for monitor in self.monitors:
            # Avoid division by zero for monitor not yet configured
            # (in framework this is executed, when only preconfigure has been called):
            current_period = monitor.period or self.integrator.dt
            strgreq += (TvbProfile.current.MAGIC_NUMBER * self.simulation_length *
                        self.number_of_nodes * self.model.nvar *
                        self.model.number_of_modes / current_period)
        LOG.info("Calculated storage requirement for simulation: %d " % int(strgreq))
        self._storage_requirement = int(strgreq)

    def run(self, **kwds):
        "Convenience method to call the simulator with **kwds and collect output data."
        ts, xs = [], []
        for _ in self.monitors:
            ts.append([])
            xs.append([])
        for data in self(**kwds):
            for tl, xl, t_x in zip(ts, xs, data):
                if t_x is not None:
                    t, x = t_x
                    tl.append(t)
                    xl.append(x)
        for i in range(len(ts)):
            ts[i] = numpy.array(ts[i])
            xs[i] = numpy.array(xs[i])
        return list(zip(ts, xs))
