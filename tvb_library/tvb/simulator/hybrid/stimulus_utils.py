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
