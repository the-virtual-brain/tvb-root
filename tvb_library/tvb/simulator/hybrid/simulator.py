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
Top-level simulator for hybrid network models

``Simulator`` owns a ``NetworkSet`` and a list of TVB ``Monitor`` objects.  It
validates that all subnetworks share the same integration time step, wires up
the monitors, and drives the time loop that calls ``NetworkSet.step`` at every
iteration.

Typical usage::

    sim = Simulator(nets=my_nets, monitors=[Raw()], simulation_length=1000.0)
    sim.configure()
    [(times, data)] = sim.run()

See Also
--------
tvb.simulator.simulator : Classic TVB simulator for comparison
"""

import math
import numpy as np
from typing import List

import tvb.basic.neotraits.api as t
from tvb.simulator.monitors import Monitor
from .network import NetworkSet


class Simulator(t.HasTraits):
    """Simulator for hybrid network models.

    The Simulator class manages the simulation of a hybrid network model,
    including configuration of monitors and time stepping.

    Attributes
    ----------
    nets : NetworkSet
        The network model to simulate
    monitors : list
        List of monitors for recording data
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
        """Configure monitor stock buffers based on total variables of interest.

        Counts the combined number of variables of interest across all
        subnetworks and the total node count, then calls ``_config_stock`` on
        any monitor that exposes it so the monitor allocates a correctly-sized
        internal ring buffer before the simulation starts.

        This method is called automatically during ``__init__``.
        """
        if len(self.monitors) == 0:
            return
        total_vois = sum([len(sn.model.variables_of_interest) for sn in self.nets.subnets])
        for monitor in self.monitors:
            num_nodes = sum([sn.nnodes for sn in self.nets.subnets])
            if hasattr(monitor, "_config_stock"):
                monitor._config_stock(total_vois, num_nodes, 1)

    def validate_dts(self):
        """Assert uniform integration time step and configure monitors.

        Reads the time step ``dt`` from the first subnetwork's integration
        scheme and asserts that every other subnetwork uses the same value.
        All monitors are then configured with this shared ``dt`` and their
        ``voi`` attribute is set to ``slice(None)`` so they record all
        variables of interest.

        Raises
        ------
        AssertionError
            If any subnetwork has a different ``dt`` from the first.

        Notes
        -----
        Called automatically during ``__init__``; the validated step size is
        stored as ``self._dt0`` for use by ``configure`` and ``run``.
        """
        self._dt0 = self.nets.subnets[0].scheme.dt
        for sn in self.nets.subnets[1:]:
            assert self._dt0 == sn.scheme.dt
        for monitor in self.monitors:
            monitor: Monitor
            monitor._config_dt(self._dt0)
            monitor.voi = slice(None)  # all vars

    def configure(self):
        """Configure the simulator and its monitors."""
        for subnet in self.nets.subnets:
            # Subnetwork.configure() also configures its IntraProjections
            subnet.configure()
            # Configure recorders in each subnetwork
            for recorder in subnet.monitors:
                recorder.configure(self.simulation_length)

        # Configure the NetworkSet, which configures InterProjections
        self.nets.configure()

        # Configure all stimuli
        for stim in self.nets.stimuli:
            stim.configure(self.simulation_length)

    def run(self, **kwargs):
        """Run the simulation and return recorded monitor data.

        If the simulator has not yet been configured (i.e. ``configure`` has
        not been called), it is configured automatically before the time loop
        begins.

        Parameters
        ----------
        initial_conditions : list of ndarray, optional
            One state array per subnetwork, each with shape
            ``(nvar, nnodes, modes)``.  When provided, these arrays are used
            directly as the initial state and to pre-fill projection history
            buffers.  Takes precedence over ``random_state``.
        random_state : None, int, or numpy.random.RandomState, optional
            Seed or generator used to draw random initial conditions from each
            model's ``state_variable_range`` (matching classic TVB default
            behaviour).  Ignored when ``initial_conditions`` is supplied.

            * ``None`` (default) — use the global ``numpy.random`` module.
            * ``int`` — create a ``numpy.random.RandomState`` seeded with
              this value, so results are reproducible.
            * ``numpy.random.RandomState`` — use the provided instance
              directly.

        Returns
        -------
        list of tuple
            One ``(times, data)`` tuple per monitor, where ``times`` is a
            1-D ``ndarray`` of recording time points and ``data`` is an
            ``ndarray`` of the corresponding monitor output.  Returns an
            empty list when no monitors are attached.

        Examples
        --------
        >>> [(t, d)] = sim.run()
        >>> [(t, d)] = sim.run(random_state=42)          # reproducible ICs
        >>> [(t, d)] = sim.run(initial_conditions=[ic])  # explicit ICs
        """
        # Configure if not already done
        if not hasattr(self, "_dt0"):
            self.configure()

        # Accept initial_conditions and random_state as parameters to run()
        initial_conditions = kwargs.pop("initial_conditions", None)
        random_state = kwargs.pop("random_state", None)

        mts = [[] for _ in self.monitors]
        mxs = [[] for _ in self.monitors]

        if initial_conditions is not None:
            x = self.nets.zero_states(initial_states=initial_conditions)
        else:
            # Default: draw ICs from state variable ranges, matching classic TVB.
            # Resolve the random number generator from random_state.
            if random_state is None:
                rng = np.random
            elif isinstance(random_state, int):
                rng = np.random.RandomState(random_state)
            else:
                rng = random_state
            x = self.nets.random_states(rng)

        # Initialize projection history buffers with initial state
        # to match classic TVB simulator behavior (history filled with ICs)
        self.nets.init_projection_buffers(x)

        stop = int(math.ceil(self.simulation_length / self._dt0))
        for step in range(1, stop + 1):
            x = self.nets.step(step, x)
            if self.monitors:
                ox = self.nets.observe(x, flat=True)
                for mt, mx, mon in zip(mts, mxs, self.monitors):
                    maybe_tx = mon.record(step, ox)
                    if maybe_tx is not None:
                        mt.append(maybe_tx[0])
                        mx.append(maybe_tx[1])
        return [(np.array(t), np.array(x)) for t, x in zip(mts, mxs)]
