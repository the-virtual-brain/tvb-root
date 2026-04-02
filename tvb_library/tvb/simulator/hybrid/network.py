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
Network container for hybrid model

A ``NetworkSet`` collects subnetworks and the projections that couple them,
forming a complete hybrid model.  At each simulation step the network:

1. Reads delayed afferent states from projection history buffers (``cfun``).
2. Applies external stimuli to coupling variables.
3. Delegates integration to each subnetwork (``Subnetwork.step``).
4. Writes the post-integration states back to the projection buffers.

The ``States`` namedtuple provides attribute-style access to per-subnetwork
state arrays so that code such as ``xs.cortex`` is readable throughout the
framework.
"""

import collections
import numpy as np
import tvb.basic.neotraits.api as t
from .subnetwork import Subnetwork
from .inter_projection import InterProjection
from .intra_projection import IntraProjection
from .base_projection import BaseProjection
from .stimulus import Stim
from . import projection_utils
from . import stimulus_utils


class NetworkSet(t.HasTraits):
    """A collection of subnetworks and their projections.

    A NetworkSet represents a complete hybrid model by collecting subnetworks
    and defining how they interact through projections and stimuli.

    Attributes
    ----------
    subnets : list
        List of subnetworks
    projections : list
        List of projections between subnetworks
    stimuli : list
        List of external stimuli
    States : namedtuple
        Named tuple class for accessing subnetwork states
    """

    subnets: [Subnetwork] = t.List(of=Subnetwork)
    projections: [BaseProjection] = t.List(of=BaseProjection)
    stimuli: [Stim] = t.List(of=Stim)
    # NOTE dynamically generated namedtuple based on subnetworks
    States: collections.namedtuple = None
    # TODO consider typing this as a tuple[ndarray[float]]?

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.States = collections.namedtuple(
            "States", " ".join([_.name for _ in self.subnets])
        )
        self.States.shape = property(lambda self: [_.shape for _ in self])

    def configure(self):
        """Configure all inter-subnetwork projections within the network set."""
        for p in self.projections:  # These are InterProjection instances
            p.configure()
        return self

    def zero_states(self, initial_states: list[np.ndarray] = None) -> States:
        """Create zero states or use provided initial states for all subnetworks.

        Parameters
        ----------
        initial_states : list[np.ndarray], optional
            A list of initial state arrays, one for each subnetwork.
            If None, zero states are created.

        Returns
        -------
        States
            Named tuple containing initial states for each subnetwork
        """
        if initial_states is not None:
            if len(initial_states) != len(self.subnets):
                raise ValueError(
                    f"Number of initial_states ({len(initial_states)}) "
                    f"must match number of subnetworks ({len(self.subnets)})."
                )
            # Ensure each initial_state has the correct shape for its subnetwork
            for i, sn in enumerate(self.subnets):
                expected_shape = (sn.model.nvar,) + sn.var_shape
                if initial_states[i].shape != expected_shape:
                    raise ValueError(
                        f"Initial state for subnetwork '{sn.name}' has shape {initial_states[i].shape}, "
                        f"expected {expected_shape}."
                    )
            return self.States(*initial_states)
        else:
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
            Array of observations. When flat=True, returns shape (total_vois, total_nodes, total_modes)
            where subnetwork observations are properly positioned.
        """
        obs = self.States(
            *[
                sn.model.observe(x).sum(axis=-1)[..., None]
                for sn, x in zip(self.subnets, states)
            ]
        )
        if flat:
            total_vois = sum(
                [len(sn.model.variables_of_interest) for sn in self.subnets]
            )
            total_nodes = sum([sn.nnodes for sn in self.subnets])
            total_modes = self.subnets[0].model.number_of_modes
            result = np.zeros((total_vois, total_nodes, total_modes))
            voi_offset = 0
            node_offset = 0
            for sn, ob in zip(self.subnets, obs):
                n_vois = ob.shape[0]
                n_nodes = ob.shape[1]
                n_modes = ob.shape[2]
                result[
                    voi_offset : voi_offset + n_vois,
                    node_offset : node_offset + n_nodes,
                    :n_modes,
                ] = ob
                voi_offset += n_vois
                node_offset += n_nodes
            obs = result
        return obs

    def init_projection_buffers(self, xs: States):
        """Initialize all projection history buffers with initial state.

        Fills the history buffers so that early delay lookups return the
        initial state rather than zeros, matching classic TVB behavior.

        Parameters
        ----------
        xs : States
            Initial states of all subnetworks.
        """
        # Initialize inter-projection buffers
        for p in self.projections:
            if isinstance(p, IntraProjection):
                continue
            src = getattr(xs, p.source.name)
            for t in range(p._horizon):
                p.update_buffer(src, t)

        # Initialize intra-projection buffers
        for sn, x in zip(self.subnets, xs):
            for p in sn.projections:
                for t in range(p._horizon):
                    p.update_buffer(x, t)

    def cfun(self, step: int, eff: States) -> States:
        """Compute coupling inputs for all subnetworks from projection buffers.

        Reads delayed afferent states from each inter-projection's history
        buffer and accumulates them into per-subnetwork coupling arrays.  Any
        external stimuli registered with this network are added on top.

        Note that ``eff`` (the current states) is accepted for API symmetry but
        is not used here: coupling is read exclusively from the pre-filled
        history buffers so that delays are honoured correctly.

        Parameters
        ----------
        step : int
            Current simulation step index.  Used to look up the correct
            position in each projection's delay buffer.
        eff : States
            Current states of all subnetworks.  Not used directly; coupling
            comes from the buffered history rather than the instantaneous state.

        Returns
        -------
        States
            Named tuple of coupling variable arrays, one per subnetwork.
            Each array has shape ``(ncvar, nnodes, modes)`` and includes
            contributions from inter-projections and stimuli.
        """
        aff = self.zero_cvars()

        # Process inter-projections (skip intra-projections)
        # Apply projections using buffers from previous state.
        # Buffer updates happen AFTER integration in step() to match classic TVB.
        for p in self.projections:
            if isinstance(p, IntraProjection):
                continue
            tgt = getattr(aff, p.target.name)
            p.apply(tgt, step)

        # Process stimuli
        for stim in self.stimuli:
            tgt = getattr(aff, stim.target.name)
            stim_coupling = stim.get_coupling(step)
            tgt += stim_coupling

        return aff

    def step(self, step, xs: States) -> States:
        """Advance the entire network by one integration time step.

        Computes inter-subnetwork coupling via ``cfun``, delegates integration
        to each ``Subnetwork.step``, and then updates all projection history
        buffers with the freshly computed states so that subsequent delay
        lookups return correct values.

        Parameters
        ----------
        step : int
            Current simulation step index (1-based, as passed by
            ``Simulator.run``).
        xs : States
            Named tuple of current state arrays, one per subnetwork.
            Each array has shape ``(nvar, nnodes, modes)``.

        Returns
        -------
        States
            Named tuple of next-step state arrays with the same structure
            as ``xs``.
        """
        cs = self.cfun(step, xs)
        nxs = self.zero_states()
        for sn, nx, x, c in zip(self.subnets, nxs, xs, cs):
            nx[:] = sn.step(step, x, c)

        # Update all projection buffers with the NEW (post-integration) states,
        # matching classic TVB which stores state after integration.
        self._update_projection_buffers(step, nxs)

        return nxs

    def _update_projection_buffers(self, step: int, nxs: States):
        """Update all projection history buffers with post-integration states."""
        # Update inter-projection buffers
        for p in self.projections:
            if isinstance(p, IntraProjection):
                continue
            src = getattr(nxs, p.source.name)
            p.update_buffer(src, step)

        # Update intra-projection buffers
        for sn, nx in zip(self.subnets, nxs):
            for p in sn.projections:
                p.update_buffer(nx, step)

    def add_projection(
        self, source_name, target_name, source_cvar, target_cvar, **kwargs
    ):
        """Add a projection between subnetworks by name.

        Convenience method to avoid looking up subnetwork objects manually.

        Parameters
        ----------
        source_name : str
            Name of source subnetwork.
        target_name : str
            Name of target subnetwork.
        source_cvar : str or list of str or ndarray of int
            Coupling variables in source (can use names now!).
        target_cvar : str or list of str or ndarray of int
            Coupling variables in target (can use names now!).
        **kwargs : dict
            Additional arguments passed to create_inter_projection.
            Can include connectivity, weights, lengths, cv, dt, scale, etc.

        Returns
        -------
        InterProjection
            The created and configured projection.

        Examples
        --------
        >>> nets.add_projection(
        ...     source_name='cortex',
        ...     target_name='thalamus',
        ...     source_cvar='y0',      # Named cvar!
        ...     target_cvar='V1',      # Named cvar!
        ...     connectivity=conn,
        ...     source_indices=[0, 1, 2],
        ...     target_indices=[3, 4, 5],
        ... )
        """
        # Look up source and target subnetworks by name
        source_subnets = [sn for sn in self.subnets if sn.name == source_name]
        target_subnets = [sn for sn in self.subnets if sn.name == target_name]

        if not source_subnets:
            raise ValueError(
                f"Source subnetwork '{source_name}' not found. "
                f"Available: {[sn.name for sn in self.subnets]}"
            )
        if not target_subnets:
            raise ValueError(
                f"Target subnetwork '{target_name}' not found. "
                f"Available: {[sn.name for sn in self.subnets]}"
            )

        if len(source_subnets) > 1:
            raise ValueError(f"Multiple subnetworks named '{source_name}' found")
        if len(target_subnets) > 1:
            raise ValueError(f"Multiple subnetworks named '{target_name}' found")

        source = source_subnets[0]
        target = target_subnets[0]

        # Create projection using factory function
        proj = projection_utils.create_inter_projection(
            source_subnet=source,
            target_subnet=target,
            source_cvar=source_cvar,
            target_cvar=target_cvar,
            **kwargs,
        )

        # Add to projections list
        if isinstance(self.projections, tuple):
            self.projections = list(self.projections)
        self.projections.append(proj)

        # Auto-configure the projection
        proj.configure()

        return proj

    def add_projection_from_connectivity(
        self, source_name, target_name, connectivity, source_cvar, target_cvar, **kwargs
    ):
        """Add a projection using global connectivity data.

        Similar to add_projection, but explicitly uses a connectivity object.
        Useful when you have a full brain connectivity and want to
        create projections between specific regions.

        Parameters
        ----------
        source_name : str
            Name of source subnetwork.
        target_name : str
            Name of target subnetwork.
        connectivity : Connectivity
            Global connectivity matrix containing region information.
        source_cvar : str or list of str or ndarray of int
            Coupling variables in source (can use names now!).
        target_cvar : str or list of str or ndarray of int
            Coupling variables in target (can use names now!).
        **kwargs : dict
            Additional arguments including source_indices and target_indices.

        Returns
        -------
        InterProjection
            The created and configured projection.

        Examples
        --------
        >>> # Cortex to thalamus using global connectivity
        >>> nets.add_projection_from_connectivity(
        ...     source_name='cortex',
        ...     target_name='thalamus',
        ...     connectivity=global_conn,
        ...     source_cvar='y0',
        ...     target_cvar='V1',
        ...     source_indices=[0, 1, 2, 3],
        ...     target_indices=[4, 5, 6, 7],
        ...     scale=1e-4,
        ... )
        """
        # Add connectivity to kwargs for the factory function
        kwargs["connectivity"] = connectivity

        # Use the regular add_projection method
        return self.add_projection(
            source_name=source_name,
            target_name=target_name,
            source_cvar=source_cvar,
            target_cvar=target_cvar,
            **kwargs,
        )

    def add_stimulus(self, target_name, stimulus, stimulus_cvar, **kwargs):
        """Add an external stimulus to a subnetwork by name.

        Convenience method to avoid looking up subnetwork objects manually.

        Parameters
        ----------
        target_name : str
            Name of target subnetwork.
        stimulus : SpatioTemporalPattern
            Spatiotemporal pattern defining the stimulus (e.g., StimuliRegion)
        stimulus_cvar : str or list of str or ndarray of int
            Name(s) of target state variable(s) that receive stimulus.
            Can be a single string name, list of names, or integer indices.
        **kwargs : dict
            Additional arguments passed to create_stimulus.
            Can include projection_scale, weights, etc.

        Returns
        -------
        Stim
            The created stimulus.

        Examples
        --------
        >>> from tvb.datatypes import patterns, equations
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
        >>> # Add to network
        >>> nets.add_stimulus(
        ...     target_name='cortex',
        ...     stimulus=stimulus,
        ...     stimulus_cvar='y0',
        ...     projection_scale=1.0
        ... )
        """
        # Look up target subnetwork by name
        target_subnets = [sn for sn in self.subnets if sn.name == target_name]

        if not target_subnets:
            raise ValueError(
                f"Target subnetwork '{target_name}' not found. "
                f"Available: {[sn.name for sn in self.subnets]}"
            )

        if len(target_subnets) > 1:
            raise ValueError(f"Multiple subnetworks named '{target_name}' found")

        target = target_subnets[0]

        # Create stimulus using factory function
        stim = stimulus_utils.create_stimulus(
            target_subnet=target,
            stimulus=stimulus,
            stimulus_cvar=stimulus_cvar,
            **kwargs,
        )

        # Add to stimuli list
        if isinstance(self.stimuli, tuple):
            self.stimuli = list(self.stimuli)
        self.stimuli.append(stim)

        return stim
