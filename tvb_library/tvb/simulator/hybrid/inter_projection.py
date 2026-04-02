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
Inter-subnetwork projection for the hybrid simulator.

An ``InterProjection`` couples state variables from one ``Subnetwork`` to
coupling variables of a *different* ``Subnetwork``.  Unlike
``IntraProjection``, the source and target may have different numbers of
modes, so an explicit ``mode_map`` matrix is supported.

Coupling-variable (cvar) names are resolved to integer indices *eagerly* in
``__init__``, because both the source and target ``Subnetwork`` instances
(and their models) must be fully configured before the projection can be
constructed.
"""

import numpy as np
import tvb.basic.neotraits.api as t

from .base_projection import BaseProjection
from .subnetwork import Subnetwork
from .cvar_utils import resolve_cvar_names, validate_cvar_indices


class InterProjection(BaseProjection):
    """Inter-subnetwork projection: delayed sparse coupling across subnetworks.

    Extends ``BaseProjection`` by holding explicit references to source and
    target ``Subnetwork`` instances and by resolving coupling-variable names
    at construction time (when both subnetworks and their models are already
    available).

    An optional ``mode_map`` matrix scales the contribution of each source
    mode to each target mode after the weighted sum.  When omitted, an
    all-ones matrix of shape ``(n_src_modes, n_tgt_modes)`` is used so that
    every source mode contributes equally to every target mode.

    Attributes
    ----------
    source : Subnetwork
        Source subnetwork whose states are read via the history buffer.
    target : Subnetwork
        Target subnetwork whose coupling-variable array receives the signal.
    mode_map : ndarray of int, shape (n_src_modes, n_tgt_modes), optional
        Linear map from source modes to target modes applied after the
        weighted sum.  Must have shape
        ``(source.model.number_of_modes, target.model.number_of_modes)``
        if provided explicitly.  Defaults to ``np.ones(...)`` (uniform
        contribution across all mode pairs).

    Notes
    -----
    Cvar resolution is *eager*: string cvar names are converted to integer
    indices in ``__init__`` using the model attached to each subnetwork.
    Both subnetworks must therefore be fully initialised before constructing
    an ``InterProjection``.

    See Also
    --------
    BaseProjection : Parent class providing the buffer and delay logic.
    IntraProjection : Coupling within a single subnetwork (identity mode map).
    """

    source: Subnetwork = t.Attr(Subnetwork)
    target: Subnetwork = t.Attr(Subnetwork)
    mode_map = t.NArray(dtype=np.int_, required=False, default=None)

    def __init__(self, **kwargs):
        """Resolve cvar names eagerly and validate the mode map.

        Cvar name resolution is performed *before* the ``BaseProjection``
        constructor so that integer arrays (not strings) reach the trait
        validation machinery.

        Parameters
        ----------
        **kwargs
            Must include ``source``, ``target``, ``weights``,
            ``source_cvar``, and ``target_cvar``.  Optionally ``lengths``,
            ``cv``, ``dt``, ``scale``, ``cfun``, and ``mode_map``.

        Raises
        ------
        ValueError
            If ``mode_map`` is provided but its shape does not equal
            ``(source.model.number_of_modes, target.model.number_of_modes)``.
        """
        # Resolve cvar names to indices BEFORE calling super().__init__()
        # to avoid trait validation errors with string cvar specs
        source_cvar = kwargs.pop('source_cvar', None)
        target_cvar = kwargs.pop('target_cvar', None)

        # Resolve source cvar names if source model is available
        if source_cvar is not None and 'source' in kwargs and hasattr(kwargs['source'], 'model'):
            source_cvar = resolve_cvar_names(kwargs['source'].model, source_cvar)
            validate_cvar_indices(kwargs['source'].model, source_cvar)

        # Resolve target cvar names if target model is available
        if target_cvar is not None and 'target' in kwargs and hasattr(kwargs['target'], 'model'):
            target_cvar = resolve_cvar_names(kwargs['target'].model, target_cvar)
            validate_cvar_indices(kwargs['target'].model, target_cvar)

        # Pass resolved cvars to parent
        if source_cvar is not None:
            kwargs['source_cvar'] = source_cvar
        if target_cvar is not None:
            kwargs['target_cvar'] = target_cvar

        super().__init__(**kwargs)

        # Default mode map if not provided
        if self.mode_map is None:
            self.mode_map = np.ones(
                (self.source.model.number_of_modes, self.target.model.number_of_modes),
                dtype=np.int_,
            )
        elif self.mode_map.shape != (
            self.source.model.number_of_modes,
            self.target.model.number_of_modes,
        ):
            raise ValueError(
                f"Provided mode_map shape {self.mode_map.shape} does not match "
                f"source modes ({self.source.model.number_of_modes}) x "
                f"target modes ({self.target.model.number_of_modes})"
            )

    def configure(self):
        """Allocate the history buffer from the source subnetwork's dimensions.

        Reads ``nvar``, ``nnodes``, and ``number_of_modes`` from the source
        subnetwork and its model, then delegates to
        ``BaseProjection.configure_buffer()``.

        Returns
        -------
        InterProjection
            Returns *self* to allow chained configuration calls.

        Raises
        ------
        ValueError
            If ``source`` or ``source.model`` is not set.
        """
        if not self.source:
            raise ValueError(
                "Source subnetwork must be set before configuring InterProjection."
            )
        if not self.source.model:
            # This check might be redundant if Subnetwork.configure ensures model is configured
            raise ValueError(
                "Source subnetwork's model must be configured before configuring InterProjection."
            )

        n_vars_src = self.source.model.nvar
        n_nodes_src = self.source.nnodes
        n_modes_src = self.source.model.number_of_modes
        self.configure_buffer(n_vars_src, n_nodes_src, n_modes_src)
        return self

    def apply(self, tgt: np.ndarray, step: int):
        """Apply the inter-subnetwork projection to the target coupling array.

        Delegates to ``BaseProjection.apply()`` with this projection's
        ``mode_map``.  ``configure()`` and at least one prior call to
        ``update_buffer()`` must precede this method.

        Parameters
        ----------
        tgt : ndarray, shape (n_vars_tgt, n_nodes_tgt, n_modes_tgt)
            Target coupling-variable array.  The slices indexed by
            ``target_cvar`` are incremented in-place.
        step : int
            Current time step index passed to ``BaseProjection.apply()``.
        """
        # Call the base class apply method, passing the specific mode_map
        # BaseProjection.apply now uses its internal buffer
        super().apply(tgt, step, self.mode_map)
