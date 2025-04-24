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

from .recorder import Recorder
from .subnetwork import Subnetwork, Stim
from .projection import Projection
from .network import NetworkSet
from .simulator import Simulator

__all__ = [
    'Recorder',
    'Subnetwork',
    'Stim',
    'Projection', 
    'NetworkSet',
    'Simulator'
] 