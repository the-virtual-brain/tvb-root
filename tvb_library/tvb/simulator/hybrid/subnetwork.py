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
Subnetwork building block for hybrid models

A ``Subnetwork`` groups a set of brain regions that share the same dynamical
``Model`` and integration ``Integrator`` (scheme).  Subnetworks are the
primary reusable unit in the hybrid framework: the same subnetwork definition
can be embedded in different ``NetworkSet`` configurations.

Intra-subnetwork connectivity is expressed as ``IntraProjection`` objects
attached to the subnetwork.  Inter-subnetwork coupling is injected by the
parent ``NetworkSet`` via the ``c`` argument of ``step``.

See Also
--------
tvb.simulator.hybrid.network : NetworkSet container
tvb.simulator.hybrid.intra_projection : IntraProjection
"""

import numpy as np
from typing import List
import tvb.basic.neotraits.api as t
from tvb.simulator.models import Model
from tvb.simulator.integrators import Integrator
from tvb.simulator.monitors import (
    Monitor,
    AfferentCoupling,
    AfferentCouplingTemporalAverage,
)

from .recorder import Recorder
from .intra_projection import IntraProjection


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
    monitors: List[Recorder] = t.List(of=Recorder)
    projections: List[IntraProjection] = t.List(of=IntraProjection)
    nnodes: int = t.Int()

    def configure(self):
        """Configure the subnetwork's model and intra-projection buffers.

        Calls ``model.configure()`` and then configures the history buffer of
        every ``IntraProjection`` attached to this subnetwork.  Must be called
        before the first integration step.

        Returns
        -------
        Subnetwork
            ``self``, to allow chained calls.
        """
        self.model.configure()
        for p in self.projections:
            p: IntraProjection
            # Set model reference for cvar name resolution (if using named cvars)
            if hasattr(p, "set_model_for_cvar_resolution"):
                p.set_model_for_cvar_resolution(self.model)
            p.configure_buffer(self.model.nvar, self.nnodes, self.model.number_of_modes)
        return self

    def add_monitor(self, monitor: Monitor):
        """Attach a monitor and configure it for this subnetwork.

        Converts the ``monitors`` tuple to a list on first call, configures
        the monitor's time step and stock buffer size, and wraps it in a
        ``Recorder`` before appending.

        ``AfferentCoupling`` and ``AfferentCouplingTemporalAverage`` monitors
        are sized according to ``len(model.cvar)`` (coupling variables)
        rather than ``len(model.variables_of_interest)``.

        Parameters
        ----------
        monitor : Monitor
            A TVB ``Monitor`` instance to attach.  It must not yet have been
            configured for a different subnetwork.
        """
        # NOTE default for list is a tuple
        if isinstance(self.monitors, tuple):
            self.monitors = []
        monitor._config_dt(self.scheme.dt)
        if hasattr(monitor, "_config_stock"):
            # AfferentCoupling monitors track coupling variables, not state variables
            if isinstance(monitor, (AfferentCoupling, AfferentCouplingTemporalAverage)):
                num_vars = len(self.model.cvar)
            else:
                num_vars = len(self.model.variables_of_interest)
            monitor._config_stock(
                num_vars,
                self.nnodes,
                self.model.number_of_modes,
            )
        self.monitors.append(Recorder(monitor=monitor))

    @property
    def var_shape(self) -> tuple[int]:
        """Shape of a single state or coupling array (excluding the variable axis).

        Returns
        -------
        tuple of int
            ``(nnodes, number_of_modes)`` — the trailing dimensions of arrays
            with shape ``(nvar, nnodes, modes)``.

        Examples
        --------
        >>> sn = Subnetwork(name='ctx', model=Generic2dOscillator(), ..., nnodes=76)
        >>> sn.var_shape
        (76, 1)
        """
        return self.nnodes, self.model.number_of_modes

    def zero_states(self) -> np.ndarray:
        """Allocate a zeroed state array for this subnetwork.

        Returns
        -------
        ndarray
            Zero array with shape ``(nvar, nnodes, modes)``.
        """
        return np.zeros((self.model.nvar,) + self.var_shape)

    def random_states(self, rng=None) -> np.ndarray:
        """Draw random initial conditions from each state variable's range.

        Calls :meth:`~tvb.simulator.models.base.Model.initial` with the
        subnetwork's integration time step and uses ``rng`` as the random
        source, matching the classic TVB simulator's default initialisation.

        Parameters
        ----------
        rng : numpy.random.RandomState or None
            Random number generator.  When *None* the module-level
            ``numpy.random`` is used.

        Returns
        -------
        ndarray
            Array with shape ``(nvar, nnodes, modes)``, each state variable
            drawn uniformly from ``model.state_variable_range``.
        """
        if rng is None:
            rng = np.random
        history_shape = (1, self.model.nvar) + self.var_shape
        return self.model.initial(self.scheme.dt, history_shape, rng)[0]

    def zero_cvars(self) -> np.ndarray:
        """Allocate a zeroed coupling-variable array for this subnetwork.

        Returns
        -------
        ndarray
            Zero array with shape ``(ncvar, nnodes, modes)``.
        """
        return np.zeros((self.model.cvar.size,) + self.var_shape)

    def cfun(self, step: int, x: np.ndarray) -> np.ndarray:
        """Compute internal coupling within the subnetwork.

        Parameters
        ----------
        step : int
            Current simulation step index.
        x : ndarray
            Current state of the subnetwork (shape [nvar, nnodes, modes]).

        Returns
        -------
        ndarray
            Internal coupling variables for the subnetwork
        """
        internal_c = self.zero_cvars()  # Shape (ncvar, nnodes, nmodes)

        if not self.projections:
            return internal_c  # No internal projections to apply

        # Apply projections using buffers from previous state.
        # Buffer updates happen AFTER integration in NetworkSet.step()
        # to match classic TVB which stores post-integration state.
        for p in self.projections:
            p.apply(internal_c, step, self.model.number_of_modes)

        return internal_c

    def step(self, step, x, c):
        """Advance the subnetwork by one integration time step.

        Computes intra-subnetwork coupling from the projection history buffers
        (``cfun``), adds the externally supplied inter-subnetwork coupling
        ``c``, invokes the integration scheme, notifies all attached monitors,
        and returns the new state.

        Parameters
        ----------
        step : int
            Current simulation step index.  Passed to projection buffers for
            delay lookups and to monitors for time-stamping.
        x : ndarray
            Current state array, shape ``(nvar, nnodes, modes)``.
        c : ndarray
            External (inter-subnetwork) coupling array supplied by
            ``NetworkSet.cfun``, shape ``(ncvar, nnodes, modes)``.

        Returns
        -------
        ndarray
            Next state array with the same shape as ``x``.

        Notes
        -----
        History buffer updates are performed by ``NetworkSet._update_projection_buffers``
        *after* this method returns, so intra-projection buffers still hold
        the pre-integration state during the ``cfun`` call inside this method.
        """
        # Calculate internal coupling first, passing the current step
        internal_c = self.cfun(step, x)
        # Add internal coupling to external coupling
        total_c = c + internal_c
        nx = self.scheme.scheme(x, self.model.dfun, total_c, 0.0, 0.0)
        # Record monitored variables
        for monitor in self.monitors:
            # AfferentCoupling monitors need coupling data, not state
            if isinstance(
                monitor.monitor, (AfferentCoupling, AfferentCouplingTemporalAverage)
            ):
                # Use coupling variables for AfferentCoupling monitors
                # total_c is (ncvar, nnodes, modes), matching node_coupling shape
                monitor.record(step, total_c)
            else:
                # Use observed state for other monitors
                monitor.record(step, self.model.observe(nx))
        return nx


class Stim(Subnetwork):
    """Stimulator adapted for hybrid cases"""

    # classic use is non-modal:
    # stimulus[self.model.stvar, :, :] = \
    #   self.stimulus(stim_step).reshape((1, -1, 1))
    pass
