"""
Hybrid model simulation framework for The Virtual Brain.

This module provides a framework for simulating hybrid brain models where different
brain regions can be modeled using different dynamical systems. The framework allows:

1. Definition of subnetworks with their own dynamics (models and integrators)
2. Creation of projections between subnetworks with custom connectivity weights
3. Application of external stimuli to subnetworks with spatiotemporal patterns
4. Simulation of the coupled system with monitoring capabilities

The design emphasizes reusability of subnetworks across different models and
separation of subnetwork dynamics from their connectivity patterns.

Coupling Variable Names
-----------------------
Instead of using integer indices, you can now use descriptive names for state
variables. This makes your code more readable and self-documenting.

Example: Hybrid Simulation with Stimulus
----------------------------------------
>>> from tvb.simulator.models import JansenRit, ReducedSetFitzHughNagumo
>>> from tvb.simulator.integrators import HeunDeterministic
>>> from tvb.simulator.monitors import TemporalAverage
>>> from tvb.datatypes import patterns, equations
>>>
>>> # Create subnetworks with different models
>>> cortex = Subnetwork(
...     name='cortex',
...     model=JansenRit(),
...     scheme=HeunDeterministic(dt=0.1),
...     nnodes=76
... )
>>>
>>> thalamus = Subnetwork(
...     name='thalamus',
...     model=ReducedSetFitzHughNagumo(),
...     scheme=HeunDeterministic(dt=0.1),
...     nnodes=76
... )
>>>
>>> # Define projections between subnetworks using named cvars
>>> nets = NetworkSet(
...     subnets=[cortex, thalamus],
...     projections=[
...         InterProjection(
...             source=cortex, target=thalamus,
...             source_cvar='y0',
...             target_cvar='V1',
...             weights=np.random.randn(76, 76) * 0.1
...         ),
...     ]
... )
>>>
>>> # Add external stimulus to cortex
>>> stim_weights = np.zeros(76)
>>> stim_weights[0] = 1.0
>>>
>>> temporal = equations.PulseTrain()
>>> temporal.parameters['onset'] = 10.0
>>> temporal.parameters['T'] = 20.0
>>> temporal.parameters['tau'] = 5.0
>>>
>>> stimulus = patterns.StimuliRegion(
...     temporal=temporal,
...     connectivity=cortex.connectivity,
...     weight=stim_weights
... )
>>>
>>> nets.add_stimulus(
...     target_name='cortex',
...     stimulus=stimulus,
...     stimulus_cvar='y0',
...     projection_scale=2.0
... )
>>>
>>> # Simulate coupled system
>>> tavg = TemporalAverage(period=1.0)
>>> sim = Simulator(
...     nets=nets,
...     simulation_length=100,
...     monitors=[tavg]
... )
>>> sim.configure()
>>> (t, y), = sim.run()

Advanced Features
----------------
- Multiple source cvars: source_cvar=['y0', 'y1'] maps multiple state variables to target
- Multiple target cvars: target_cvar=['V1', 'V2'] broadcasts to multiple targets
- Repeated cvars: source_cvar=['y0', 'y0', 'y0'] allows more cvars than svars
- External stimuli: Add Stim objects to NetworkSet for controlled external input
- Named cvars in stimuli: Use model.state_variables names instead of indices

Coupling Functions
------------------
Projections can use coupling functions to transform afferent activity:

>>> from tvb.simulator.hybrid.coupling import Linear, Sigmoidal
>>>
>>> # Linear coupling: a * x + b
>>> proj = create_inter_projection(
...     source_subnet=cortex,
...     target_subnet=thalamus,
...     source_cvar='y0',
...     target_cvar='V',
...     weights=weights,
...     coupling=Linear(a=0.5, b=0.1)
... )
>>>
>>> # Sigmoidal coupling: saturating sigmoid
>>> proj = create_inter_projection(
...     source_subnet=cortex,
...     target_subnet=thalamus,
...     source_cvar='y0',
...     target_cvar='V',
...     weights=weights,
...     coupling=Sigmoidal(cmin=0.0, cmax=1.0)
... )

Available coupling functions:
- Linear: a * x + b (linear scaling with offset)
- Scaling: a * x (simple scaling)
- Sigmoidal: cmin + (cmax-cmin) / (1 + exp(-a*(x-midpoint)/sigma)) (post-summation)
- Kuramoto: (a/N) * sin(x) (phase coupling, post-summation)
- Difference: a * x (diffusive coupling, post-summation)
- HyperbolicTangent: a * (1 + tanh((x-midpoint)/sigma)) (pre-summation)
- SigmoidalJansenRit: a * (2*e0) / (1 + exp(r*(v0-x))) (pre-summation, JansenRit-specific)
- PreSigmoidal: H * (Q + tanh(G*(P*x-theta))) (pre-summation with dynamic threshold)
"""

import warnings

msg = "Hybrid simulation is experimental: please report bugs or suggestions."
warnings.warn(msg)

from .base_projection import BaseProjection
from .intra_projection import IntraProjection
from .inter_projection import InterProjection
from .recorder import Recorder
from .subnetwork import Subnetwork
from .network import NetworkSet
from .simulator import Simulator
from .stimulus import Stim
from . import cvar_utils
from . import projection_utils
from . import stimulus_utils
from . import coupling
from .coupling import (
    Coupling,
    Linear,
    Scaling,
    Sigmoidal,
    Kuramoto,
    Difference,
    HyperbolicTangent,
    SigmoidalJansenRit,
    PreSigmoidal,
)
