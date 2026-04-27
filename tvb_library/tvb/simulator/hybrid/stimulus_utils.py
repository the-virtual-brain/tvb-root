# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Factory helpers for creating :class:`~tvb.simulator.hybrid.stimulus.Stim` objects.

The single exported function, :func:`create_stimulus`, handles the common
boilerplate of resolving coupling variable names, validating sparse weight
matrices, and constructing a ready-to-configure :class:`Stim`.
"""

import numpy as np
from scipy import sparse as sp
from tvb.datatypes.patterns import SpatioTemporalPattern

from .stimulus import Stim
from .projection_utils import create_all_to_all_weights


def create_stimulus(
    target_subnet,
    stimulus,
    stimulus_cvar,
    projection_scale=1.0,
    weights=None,
):
    """Create a Stim object with auto-configuration.

    This factory function simplifies Stim creation by auto-setting:
    - dt from target's scheme
    - Resolving named cvar to indices
    - Creating default all-to-all weights if not provided

    Parameters
    ----------
    target_subnet : Subnetwork
        Target subnetwork that will receive the stimulus
    stimulus : SpatioTemporalPattern
        Spatiotemporal pattern defining the stimulus (e.g., StimuliRegion)
    stimulus_cvar : str or list of str or ndarray of int
        Name(s) of target state variable(s) that receive stimulus.
        Can be a single string name, list of names, or integer indices.
    projection_scale : float, optional (default=1.0)
        Scaling factor for the stimulus
    weights : scipy.sparse.csr_matrix, optional
        Spatial weights matrix in CSR sparse format.
        Shape must be (n_nodes, n_nodes) where n_nodes = target_subnet.nnodes.
        If None, creates an all-to-all identity matrix.

    Returns
    -------
    Stim
        Configured Stim object (configure() must still be called before use)

    Examples
    --------
    >>> from tvb.datatypes import patterns, equations
    >>> from tvb.simulator.lab import *
    >>>
    >>> # Create spatial pattern (stimulate region 0)
    >>> stim_weights = np.zeros((n_nodes,))
    >>> stim_weights[0] = 1.0
    >>>
    >>> # Create temporal pattern (pulse train)
    >>> temporal = equations.PulseTrain()
    >>> temporal.parameters['onset'] = 500.0
    >>> temporal.parameters['T'] = 1000.0
    >>> temporal.parameters['tau'] = 100.0
    >>>
    >>> # Create stimulus pattern
    >>> stimulus = patterns.StimuliRegion(
    ...     temporal=temporal,
    ...     connectivity=conn,
    ...     weight=stim_weights
    ... )
    >>>
    >>> # Create Stim with named cvar
    >>> stim = create_stimulus(
    ...     target_subnet=cortex,
    ...     stimulus=stimulus,
    ...     stimulus_cvar='y0',
    ...     projection_scale=1.0
    ... )
    >>>
    >>> # Configure and use
    >>> stim.configure(simulation_length=1000.0)
    >>> coupling = stim.get_coupling(step=100)
    """
    # Validate inputs
    if target_subnet is None:
        raise ValueError("target_subnet must be provided")

    if stimulus is None:
        raise ValueError("stimulus must be provided")

    if not isinstance(stimulus, SpatioTemporalPattern):
        raise TypeError(
            f"stimulus must be a SpatioTemporalPattern, got {type(stimulus)}"
        )

    # For StimuliRegion, don't create default weights since the pattern already has spatial weights
    # Only create default weights if not provided and stimulus is not StimuliRegion
    from tvb.datatypes.patterns import StimuliRegion

    if weights is None and not isinstance(stimulus, StimuliRegion):
        weights = create_all_to_all_weights(target_subnet.nnodes)

    # Validate weights if provided
    if weights is not None:
        if not isinstance(weights, sp.csr_matrix):
            raise TypeError(
                f"weights must be a scipy.sparse.csr_matrix, got {type(weights)}"
            )
        n_nodes = target_subnet.nnodes
        if weights.shape != (n_nodes, n_nodes):
            raise ValueError(
                f"weights shape {weights.shape} must match target nodes "
                f"({n_nodes}, {n_nodes})"
            )

    # Create Stim object
    stim = Stim(
        target=target_subnet,
        stimulus=stimulus,
        target_cvar=stimulus_cvar,
        projection_scale=projection_scale,
        weights=weights,
    )

    return stim


def _make_weight_vector(n_nodes, target_node, amplitude=1.0):
    """Build a per-node weight vector for StimuliRegion."""
    if target_node is None:
        return np.full(n_nodes, amplitude)
    w = np.zeros(n_nodes)
    w[target_node] = amplitude
    return w


def constant_stim(subnet, amplitude, target_node=0, target_cvar=0,
                  projection_scale=1.0, simulation_length=None):
    """Create a constant-amplitude stimulus in one call.

    Builds a Linear(a=0, b=amplitude) temporal equation, a minimal
    Connectivity, a StimuliRegion, and a configured Stim.

    Parameters
    ----------
    subnet : Subnetwork
        Target subnetwork.
    amplitude : float
        Constant stimulus amplitude.
    target_node : int or None
        Which node to stimulate. None → all nodes.
    target_cvar : int
        Target coupling-variable index (default 0).
    projection_scale : float
        Scaling factor for the stimulus.
    simulation_length : float or None
        If provided, ``stim.configure(simulation_length)`` is called.

    Returns
    -------
    Stim
        Ready-to-use stimulus.
    """
    from tvb.datatypes import equations as eqs
    from tvb.datatypes.patterns import StimuliRegion

    temporal = eqs.Linear()
    temporal.parameters["a"] = 0.0
    temporal.parameters["b"] = float(amplitude)
    weight = _make_weight_vector(subnet.nnodes, target_node)
    pattern = StimuliRegion.from_weights(weight=weight, temporal=temporal)
    target_cvar = np.array([target_cvar], dtype=np.int_)
    stim = Stim(target=subnet, stimulus=pattern,
                target_cvar=target_cvar, projection_scale=projection_scale)
    if simulation_length is not None:
        stim.configure(simulation_length=simulation_length)
    return stim


def pulse_stim(subnet, amplitude, onset, period, pulse_width,
               target_node=0, target_cvar=0,
              projection_scale=1.0, simulation_length=None):
    """Create a pulse-train stimulus in one call.

    Parameters
    ----------
    subnet : Subnetwork
        Target subnetwork.
    amplitude : float
        Pulse amplitude.
    onset : float
        Onset time (ms).
    period : float
        Pulse period T (ms).
    pulse_width : float
        Pulse width tau (ms).
    target_node : int or None
        Which node to stimulate. None → all nodes.
    target_cvar : int
        Target coupling-variable index (default 0).
    projection_scale : float
        Scaling factor for the stimulus.
    simulation_length : float or None
        If provided, ``stim.configure(simulation_length)`` is called.

    Returns
    -------
    Stim
        Ready-to-use stimulus.
    """
    from tvb.datatypes import equations as eqs
    from tvb.datatypes.patterns import StimuliRegion

    temporal = eqs.PulseTrain()
    temporal.parameters["onset"] = float(onset)
    temporal.parameters["T"] = float(period)
    temporal.parameters["tau"] = float(pulse_width)
    temporal.parameters["amp"] = float(amplitude)
    weight = _make_weight_vector(subnet.nnodes, target_node)
    pattern = StimuliRegion.from_weights(
        weight=weight, temporal=temporal,
    )
    target_cvar = np.array([target_cvar], dtype=np.int_)
    stim = Stim(target=subnet, stimulus=pattern,
                target_cvar=target_cvar, projection_scale=projection_scale)
    if simulation_length is not None:
        stim.configure(simulation_length=simulation_length)
    return stim


def sinusoid_stim(subnet, amplitude, frequency, target_node=0,
                  target_cvar=0, projection_scale=1.0,
                  simulation_length=None):
    """Create a sinusoidal stimulus in one call.

    Parameters
    ----------
    subnet : Subnetwork
        Target subnetwork.
    amplitude : float
        Sinusoid amplitude.
    frequency : float
        Sinusoid frequency.
    target_node : int or None
        Which node to stimulate. None → all nodes.
    target_cvar : int
        Target coupling-variable index (default 0).
    projection_scale : float
        Scaling factor for the stimulus.
    simulation_length : float or None
        If provided, ``stim.configure(simulation_length)`` is called.

    Returns
    -------
    Stim
        Ready-to-use stimulus.
    """
    from tvb.datatypes import equations as eqs
    from tvb.datatypes.patterns import StimuliRegion

    temporal = eqs.Sinusoid()
    temporal.parameters["amp"] = float(amplitude)
    temporal.parameters["frequency"] = float(frequency)
    weight = _make_weight_vector(subnet.nnodes, target_node)
    pattern = StimuliRegion.from_weights(
        weight=weight, temporal=temporal,
    )
    target_cvar = np.array([target_cvar], dtype=np.int_)
    stim = Stim(target=subnet, stimulus=pattern,
                target_cvar=target_cvar, projection_scale=projection_scale)
    if simulation_length is not None:
        stim.configure(simulation_length=simulation_length)
    return stim
