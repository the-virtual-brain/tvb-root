# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2024, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
#
#

"""
Hybrid model simulation framework for The Virtual Brain.

This module provides a framework for simulating hybrid brain models where different
brain regions can be modeled using different dynamical systems. The framework allows:

1. Definition of subnetworks with their own dynamics (models and integrators)
2. Creation of projections between subnetworks with custom connectivity weights
3. Simulation of the coupled system with monitoring capabilities

The design emphasizes reusability of subnetworks across different models and
separation of subnetwork dynamics from their connectivity patterns.

Example
-------
>>> from tvb.simulator.models import JansenRit, ReducedSetFitzHughNagumo
>>> from tvb.simulator.integrators import HeunDeterministic
>>> from tvb.simulator.hybrid import Subnetwork, Projection, NetworkSet, Simulator
>>> from tvb.simulator.monitors import TemporalAverage
>>> 
>>> # Create subnetworks with different models
>>> # Specify the same number of variables of interest for both models
>>> jrkwargs = {'variables_of_interest': JansenRit.variables_of_interest.default[:2]}
>>> fhnkwargs = {'variables_of_interest': ReducedSetFitzHughNagumo.variables_of_interest.default[:2]}
>>> 
>>> cortex = Subnetwork(
...     name='cortex',
...     model=JansenRit(**jrkwargs),
...     scheme=HeunDeterministic(dt=0.1),
...     nnodes=76
... ).configure()  # Configure the model
>>> 
>>> thalamus = Subnetwork(
...     name='thalamus',
...     model=ReducedSetFitzHughNagumo(**fhnkwargs),
...     scheme=HeunDeterministic(dt=0.1),
...     nnodes=76
... ).configure()  # Configure the model
>>> 
>>> # Define projections between subnetworks
>>> nets = NetworkSet(
...     subnets=[cortex, thalamus],
...     projections=[
...         Projection(
...             source=cortex, target=thalamus,
...             source_cvar=np.r_[0], target_cvar=np.r_[1],
...             weights=np.random.randn(76, 76)
...         )
...     ]
... )
>>> 
>>> # Simulate the coupled system
>>> tavg = TemporalAverage(period=1.0)  # Add a monitor
>>> sim = Simulator(
...     nets=nets, 
...     simulation_length=100,
...     monitors=[tavg]  # Include the monitor
... )
>>> sim.configure()
>>> (t, y), = sim.run()  # Unpack the first (and only) monitor result
"""

import collections
import numpy as np
from typing import List
import tvb.basic.neotraits.api as t
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.models import Model
from tvb.simulator.integrators import Integrator
from tvb.simulator.monitors import Monitor


class Recorder(t.HasTraits):
    """Records simulation data from a monitor.
    
    This class acts as a wrapper around a Monitor to store simulation data
    in memory rather than directly to disk.
    
    Attributes
    ----------
    monitor : Monitor
        The monitor to record data from
    times : list
        List of time points where data was recorded
    samples : list
        List of recorded state samples
    """

    monitor: Monitor = t.Attr(Monitor)

    def __init__(self, **kwargs):
        self.times = []
        self.samples = []
        super().__init__(**kwargs)

    def record(self, step, state):
        """Record a state sample if the monitor indicates it should be recorded.
        
        Parameters
        ----------
        step : int
            Current simulation step
        state : array_like
            Current state of the system
        """
        ty = self.monitor.record(step, state)
        if ty is not None:
            t, y = ty
            self.times.append(t)
            self.samples.append(y)

    @property
    def shape(self):
        """Shape of the recorded data.
        
        Returns
        -------
        tuple
            Shape of the recorded data array
        """
        return (len(self.samples), ) + self.samples[0].shape

    def to_arrays(self):
        """Convert recorded data to numpy arrays.
        
        Returns
        -------
        tuple
            (times, samples) as numpy arrays
        """
        return np.array(self.times), np.array(self.samples)


class Subnetwork(t.HasTraits):
    """A subnetwork that can be reused across different models.
    
    A subnetwork represents a group of brain regions that share the same
    dynamical model and integration scheme. It is designed to be reusable
    across different models by separating the dynamics from the connectivity.
    
    Attributes
    ----------
    name : str
        Name of the subnetwork
    model : Model
        Dynamical model for the subnetwork
    scheme : Integrator
        Integration scheme for the subnetwork
    nnodes : int
        Number of nodes in the subnetwork
    monitors : list
        List of recorders for monitoring simulation data
    """

    name: str = t.Attr(str)
    model: Model = t.Attr(Model)
    scheme: Integrator = t.Attr(Integrator)
    monitors: List[Recorder]
    nnodes: int = t.Int()

    def __init__(self, **kwargs):
        self.monitors = []
        super().__init__(**kwargs)

    def configure(self):
        """Configure the subnetwork's model.
        
        Returns
        -------
        self
            The configured subnetwork
        """
        self.model.configure()
        return self

    def add_monitor(self, monitor: Monitor):
        """Add a monitor to record simulation data.
        
        Parameters
        ----------
        monitor : Monitor
            The monitor to add
        """
        monitor._config_dt(self.scheme.dt)
        monitor._config_stock(len(self.model.variables_of_interest),
                            self.nnodes,
                            self.model.number_of_modes)
        self.monitors.append(Recorder(monitor=monitor))

    @property
    def var_shape(self) -> tuple[int]:
        """Shape of the state variables.
        
        Returns
        -------
        tuple
            (number of nodes, number of modes)
        """
        return self.nnodes, self.model.number_of_modes

    def zero_states(self) -> np.ndarray:
        """Create an array of zeros for the state variables.
        
        Returns
        -------
        ndarray
            Array of zeros with shape (nvar, nnodes, modes)
        """
        return np.zeros((self.model.nvar, ) + self.var_shape)

    def zero_cvars(self) -> np.ndarray:
        """Create an array of zeros for the coupling variables.
        
        Returns
        -------
        ndarray
            Array of zeros with shape (ncvar, nnodes, modes)
        """
        return np.zeros((self.model.cvar.size, ) + self.var_shape)

    def step(self, step, x, c):
        """Take a single integration step.
        
        Parameters
        ----------
        step : int
            Current simulation step
        x : ndarray
            Current state
        c : ndarray
            Current coupling variables
            
        Returns
        -------
        ndarray
            Next state after integration
        """
        nx = self.scheme.scheme(x, self.model.dfun, c, 0, 0)
        for monitor in self.monitors:
            monitor.record(step, self.model.observe(nx))
        return nx


class Stim(Subnetwork):
    "Stimulator adapted for hybrid cases"
    # classic use is non-modal:
    # stimulus[self.model.stvar, :, :] = \
    #   self.stimulus(stim_step).reshape((1, -1, 1))
    pass


class Projection(t.HasTraits):
    """A projection from one subnetwork to another.
    
    A projection defines how one subnetwork influences another through
    coupling variables and connectivity weights.
    
    Attributes
    ----------
    source : Subnetwork
        Source subnetwork
    target : Subnetwork
        Target subnetwork
    source_cvar : ndarray
        Array of coupling variable indices in source.
    target_cvar : ndarray
        Array of coupling variable indices in target.
    scale : float
        Scaling factor for the projection
    weights : ndarray
        Connectivity weights matrix
    mode_map : ndarray
        Mapping between source and target modes
    """

    source: Subnetwork = t.Attr(Subnetwork)
    target: Subnetwork = t.Attr(Subnetwork)
    source_cvar = t.NArray(dtype=np.int_)  # Array of source coupling variable indices
    target_cvar = t.NArray(dtype=np.int_)  # Array of target coupling variable indices
    scale: float = t.Float(default=1.0)
    weights: np.ndarray = t.NArray(dtype=np.float_)
    mode_map: np.ndarray = t.NArray(required=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.mode_map is None:
            self.mode_map = np.ones(
                (self.source.model.number_of_modes,
                 self.target.model.number_of_modes, ))
            self.mode_map /= self.source.model.number_of_modes
            
        # Convert scalar indices to arrays for unified handling
        self.source_cvar = np.atleast_1d(self.source_cvar)
        self.target_cvar = np.atleast_1d(self.target_cvar)
        self._validate_cvars()

    def _validate_cvars(self):
        """Validate coupling variable configurations."""
        src_size = len(self.source_cvar)
        tgt_size = len(self.target_cvar)
        
        # Validate indices are within bounds
        if np.any(self.source_cvar >= self.source.model.cvar.size):
            raise ValueError(f"Source coupling variable index {self.source_cvar} out of bounds")
        if np.any(self.target_cvar >= self.target.model.cvar.size):
            raise ValueError(f"Target coupling variable index {self.target_cvar} out of bounds")
            
        # Validate broadcasting configurations
        if not (src_size == 1 or tgt_size == 1 or src_size == tgt_size):
            raise ValueError(
                f"Invalid coupling variable configuration: source size {src_size} "
                f"and target size {tgt_size} must be either equal or one must be 1"
            )

    def apply(self, tgt, src):
        """Apply the projection to compute coupling.
        
        Handles three cases:
        a) One source to many targets (broadcasting)
        b) Many sources to one target (reduction/summation)
        c) Equal number of sources to targets (element-wise)
        
        Parameters
        ----------
        tgt : ndarray
            Target state array to modify
        src : ndarray
            Source state array
        """
        # Compute base coupling for each source variable
        gx_array = self.scale * self.weights @ src[self.source_cvar] @ self.mode_map
        # TODO add time delays
        if self.source_cvar.size == 1:
            # Case a: broadcast single source to all targets
            tgt[self.target_cvar] += gx_array[0]
        elif self.target_cvar.size == 1:
            # Case b: sum all sources to single target
            tgt[self.target_cvar] += gx_array.sum(axis=0)
        else:
            # Case c: element-wise mapping
            tgt[self.target_cvar] += gx_array


class NetworkSet(t.HasTraits):
    """A collection of subnetworks and their projections.
    
    A NetworkSet represents a complete hybrid model by collecting subnetworks
    and defining how they interact through projections.
    
    Attributes
    ----------
    subnets : list
        List of subnetworks
    projections : list
        List of projections between subnetworks
    States : namedtuple
        Named tuple class for accessing subnetwork states
    """

    subnets: [Subnetwork] = t.List(of=Subnetwork)
    projections: [Subnetwork] = t.List(of=Projection)

    # NOTE dynamically generated namedtuple based on subnetworks
    States: collections.namedtuple = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.States = collections.namedtuple(
            'States',
            ' '.join([_.name for _ in self.subnets]))
        self.States.shape = property(lambda self: [_.shape for _ in self])

    def zero_states(self) -> States:
        """Create zero states for all subnetworks.
        
        Returns
        -------
        States
            Named tuple containing zero states for each subnetwork
        """
        return self.States(*[_.zero_states() for _ in self.subnets])

    def zero_cvars(self) -> States:
        """Create zero coupling variables for all subnetworks.
        
        Returns
        -------
        States
            Named tuple containing zero coupling variables for each subnetwork
        """
        return self.States(*[_.zero_cvars() for _ in self.subnets])

    def observe(self, states: States, flat=False) -> np.ndarray:
        """Compute observations across all subnetworks.
        
        Parameters
        ----------
        states : States
            Current states of all subnetworks
        flat : bool
            If True, flatten observations into a single array
            
        Returns
        -------
        ndarray
            Array of observations
        """
        obs = self.States(*[sn.model.observe(x).sum(axis=-1)[..., None]
                          for sn, x in zip(self.subnets, states)])
        if flat:
            obs = np.hstack(obs)
        return obs

    def cfun(self, eff: States) -> States:
        """Compute coupling between subnetworks.
        
        Parameters
        ----------
        eff : States
            Current states of all subnetworks
            
        Returns
        -------
        States
            Coupling variables for each subnetwork
        """
        aff = self.zero_cvars()
        for p in self.projections:
            tgt = getattr(aff, p.target.name)
            src = getattr(eff, p.source.name)
            p.apply(tgt, src)
        return aff

    def step(self, step, xs: States) -> States:
        """Take a single integration step for all subnetworks.
        
        Parameters
        ----------
        step : int
            Current simulation step
        xs : States
            Current states of all subnetworks
            
        Returns
        -------
        States
            Next states after integration
        """
        cs = self.cfun(xs)
        nxs = self.zero_states()
        for sn, nx, x, c in zip(self.subnets, nxs, xs, cs):
            nx[:] = sn.step(step, x, c)
        return nxs


class Simulator(t.HasTraits):
    """Simulator for hybrid network models.
    
    The Simulator class manages the simulation of a hybrid network model,
    including configuration of monitors and time stepping.
    
    Attributes
    ----------
    nets : NetworkSet
        The network model to simulate
    monitors : list
        List of monitors for recording simulation data
    simulation_length : float
        Total simulation time in milliseconds
    """

    nets: NetworkSet = t.Attr(NetworkSet)
    monitors: List[Monitor] = t.List(of=Monitor)
    simulation_length: float = t.Float()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.validate_dts()
        self.validate_vois()

    def validate_vois(self):
        """Validate variables of interest across subnetworks."""
        if len(self.monitors) == 0:
            return
        nv0 = self.nets.subnets[0].model.variables_of_interest
        for sn in self.nets.subnets[1:]:
            msg = 'Variables of interest must have same size on all models.'
            assert len(nv0) == len(sn.model.variables_of_interest), msg
        for monitor in self.monitors:
            num_nodes = sum([sn.nnodes for sn in self.nets.subnets])
            monitor._config_stock(len(nv0), num_nodes, 1)

    def validate_dts(self):
        """Validate integration time steps across subnetworks."""
        self._dt0 = self.nets.subnets[0].scheme.dt
        for sn in self.nets.subnets[1:]:
            assert self._dt0 == sn.scheme.dt
        for monitor in self.monitors:
            monitor: Monitor
            monitor._config_dt(self._dt0)
            monitor.voi = slice(None)  # all vars

    def run(self):
        """Run the simulation.
        
        Returns
        -------
        list
            List of (time, data) tuples for each monitor, if any
        """
        x = self.nets.zero_states()
        mts = [[] for _ in self.monitors]
        mxs = [[] for _ in self.monitors]
        for step in range(int(self.simulation_length / self._dt0)):
            x = self.nets.step(step, x)
            if self.monitors:
                ox = self.nets.observe(x, flat=True)
                for mt, mx, mon in zip(mts, mxs, self.monitors):
                    maybe_tx = mon.record(step, ox)
                    if maybe_tx is not None:
                        mt.append(maybe_tx[0])
                        mx.append(maybe_tx[1])
        return [(np.array(t), np.array(x)) for t, x in zip(mts, mxs)]
