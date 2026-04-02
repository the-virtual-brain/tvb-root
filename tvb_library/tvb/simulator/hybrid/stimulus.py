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
External stimulus for hybrid subnetworks.

Provides :class:`Stim`, which wraps a :class:`~tvb.datatypes.patterns.SpatioTemporalPattern`
and injects it into a target :class:`~tvb.simulator.hybrid.subnetwork.Subnetwork` at each
simulation step via the same coupling interface used by projections.
"""

import numpy as np
from scipy import sparse as sp
import tvb.basic.neotraits.api as t
from tvb.datatypes.patterns import SpatioTemporalPattern

from .cvar_utils import resolve_cvar_names
from .projection_utils import create_all_to_all_weights

# Import here to avoid circular dependency
import tvb.simulator.hybrid.subnetwork as sn_module


class Stim(t.HasTraits):
    """External stimulus injected into a hybrid subnetwork.

    :class:`Stim` wraps a :class:`~tvb.datatypes.patterns.SpatioTemporalPattern` and
    delivers it to a target subnetwork through the same interface used by projections.
    It must be configured before use by calling :meth:`configure`.

    Attributes
    ----------
    stimulus : SpatioTemporalPattern
        Spatiotemporal pattern defining the waveform (e.g., ``StimuliRegion``).
    target : Subnetwork
        Target subnetwork that receives the stimulus.
    target_cvar : ndarray of int
        Integer indices of the target coupling variables that receive the
        stimulus.  String names are resolved to indices in ``__init__``.
    projection_scale : float
        Global scaling factor applied to the stimulus signal.
    weights : scipy.sparse.csr_matrix or None
        Spatial weights matrix of shape ``(n_nodes, n_nodes)``.  When
        ``None`` the raw stimulus output is used without spatial weighting.
        Default: ``None`` (``StimuliRegion`` carries its own spatial pattern).
    dt : float
        Integration time step in milliseconds.  Set from ``target.scheme.dt``
        during :meth:`configure`.
    time : ndarray
        Time vector used by the stimulus pattern; populated by :meth:`configure`.

    See Also
    --------
    stimulus_utils.create_stimulus : Convenience factory for building a :class:`Stim`.
    """

    stimulus = t.Attr(field_type=SpatioTemporalPattern)
    target = t.Attr(field_type=sn_module.Subnetwork)
    target_cvar = t.NArray(dtype=np.int_, required=False)
    projection_scale = t.Float(default=1.0)
    weights = t.Attr(field_type=sp.csr_matrix, required=False)
    dt = t.Float(required=False, default=None)
    time = t.NArray(required=False, default=None)

    def __init__(self, **kwargs):
        # Resolve cvar names to indices BEFORE calling super().__init__()
        # to avoid type validation errors with string cvar names
        target_cvar_input = kwargs.pop("target_cvar", None)
        target_subnet = kwargs.get("target")

        if target_cvar_input is not None:
            if isinstance(target_cvar_input, str) or (
                isinstance(target_cvar_input, (list, tuple, np.ndarray))
                and len(target_cvar_input) > 0
                and isinstance(target_cvar_input[0], str)
            ):
                # Resolve names to indices
                if target_subnet is None:
                    raise ValueError(
                        "target must be provided to resolve cvar names"
                    )
                target_cvar_resolved = resolve_cvar_names(
                    target_subnet.model, target_cvar_input
                )
            else:
                # Already indices
                target_cvar_resolved = np.atleast_1d(target_cvar_input)

            kwargs["target_cvar"] = target_cvar_resolved

        super().__init__(**kwargs)

        # Ensure target_cvar is an array
        self.target_cvar = np.atleast_1d(self.target_cvar)

        # Validate weights is CSR and correct shape
        if self.weights is not None:
            if not isinstance(self.weights, sp.csr_matrix):
                raise TypeError(
                    f"Weights must be provided as a scipy.sparse.csr_matrix, got {type(self.weights)}"
                )
            if self.weights.shape != (self.target.nnodes, self.target.nnodes):
                raise ValueError(
                    f"Weights shape {self.weights.shape} must match target nodes "
                    f"({self.target.nnodes}, {self.target.nnodes})"
                )

        # Validate weights shape
        if self.weights is not None and self.weights.shape != (
            self.target.nnodes,
            self.target.nnodes,
        ):
            raise ValueError(
                f"Weights shape {self.weights.shape} must match target nodes "
                f"({self.target.nnodes}, {self.target.nnodes})"
            )

    def configure(self, simulation_length: float):
        """Configure the time vector and spatial pattern for the given simulation length.

        Sets ``self.dt`` from the target subnetwork's integrator, builds the
        time vector ``self.time``, and calls ``configure_time`` (and, when
        present, ``configure_space``) on the underlying :attr:`stimulus` pattern
        so it is ready for evaluation.

        Parameters
        ----------
        simulation_length : float
            Total simulation duration in milliseconds.
        """
        # Set dt from target's scheme
        self.dt = self.target.scheme.dt

        # Create time vector
        self.time = np.arange(0.0, simulation_length + self.dt, self.dt)

        # Configure temporal pattern
        if hasattr(self.stimulus, "configure_time"):
            self.stimulus.configure_time(self.time.reshape((1, -1)))

        # Configure spatial pattern if needed (e.g., StimuliRegion)
        if hasattr(self.stimulus, "configure_space"):
            # StimuliRegion needs configure_space called to set up spatial pattern
            self.stimulus.configure_space()

    def get_coupling(self, step: int) -> np.ndarray:
        """Evaluate the stimulus at the given step and return the coupling contribution.

        Calls ``stimulus(temporal_indices=step)`` to obtain the spatial pattern
        at the current simulation step, optionally applies the sparse
        :attr:`weights` matrix, scales by :attr:`projection_scale`, and
        broadcasts the result to the expected coupling shape.

        Parameters
        ----------
        step : int
            Current simulation step index (0-based).

        Returns
        -------
        ndarray of shape (n_cvar, n_nodes, n_modes)
            Coupling contribution to add to the target subnetwork's coupling
            buffer.  ``n_cvar`` equals ``len(self.target_cvar)``.
        """
        # Get current time
        t = step * self.dt

        # Evaluate temporal pattern at current time
        # Stimulus(temporal_indices=step) returns spatial pattern at that time
        temporal_value = self.stimulus(temporal_indices=step)

        # temporal_value shape from StimuliRegion: (n_stim_regions, 1)
        # But target subnet may have fewer nodes
        # Extract only the relevant nodes for the target subnet

        # Apply spatial weights
        # If weights is None, create identity (no spatial weighting)
        if self.weights is None:
            temporal_value = self.stimulus(temporal_indices=step)
            scaled_value = self.projection_scale * temporal_value
        else:
            # weights: (n_nodes, n_nodes) @ temporal_value: (n_nodes, 1) -> (n_nodes, 1)
            weighted_value = self.weights @ temporal_value

            # Apply projection scale
            scaled_value = self.projection_scale * weighted_value

        # Broadcast to shape (n_cvar, n_nodes, n_modes)
        # target_cvar defines which coupling variables receive stimulus
        # Get number of modes from target
        n_modes = self.target.nmodes if hasattr(self.target, "nmodes") else 1

        # Create coupling array: (n_cvar, n_nodes, n_modes)
        n_cvar = len(self.target_cvar)
        n_nodes = self.target.nnodes

        # scaled_value may have shape (n_stim_regions, 1) from StimuliRegion
        # where n_stim_regions could be larger than target.nnodes
        # Extract only the first n_nodes if necessary
        if scaled_value.shape[0] > n_nodes:
            scaled_value = scaled_value[:n_nodes, :]

        # scaled_value is now (n_nodes, 1)
        # We need to broadcast to (n_cvar, n_nodes, n_modes)
        # Tile scaled_value to match target shape
        coupling = np.tile(scaled_value, (n_cvar, 1, 1))

        return coupling
