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
Tests for the hybrid :class:`~tvb.simulator.hybrid.Simulator` class.

Covers full end-to-end simulation runs, including:

- :class:`~tvb.simulator.monitors.TemporalAverage` monitors operating over the
  whole network;
- per-subnetwork monitors via :class:`~tvb.simulator.hybrid.Recorder`;
- mixed-model networks with :class:`~tvb.simulator.models.JansenRit` and
  :class:`~tvb.simulator.models.ReducedSetFitzHughNagumo` subnetworks;
- the canonical usage example reproduced from the module docstring of
  :mod:`tvb.simulator.hybrid`.

Output shape convention: ``(time_steps, n_vois, n_nodes, n_modes)``.
"""

import numpy as np
from scipy.sparse import csr_matrix
from tvb.simulator.models import JansenRit, ReducedSetFitzHughNagumo
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.monitors import TemporalAverage
from tvb.simulator.hybrid import (
    Recorder,
    Simulator,
    Subnetwork,
    InterProjection,
    NetworkSet,
)
from .test_base import BaseHybridTest


class TestSimulator(BaseHybridTest):
    """End-to-end tests for :class:`~tvb.simulator.hybrid.Simulator`.

    Exercises the full simulation pipeline: network configuration, time
    stepping, monitor output collection, and output shape validation.
    """

    def test_sim(self):
        """Full simulation run with a global TemporalAverage monitor.

        Runs a 2-subnetwork (cortex + thalamus) simulation for 10 ms.
        Asserts 10 time points are returned and the output array has shape
        ``(10, n_vois=4, n_nodes=76, modes=1)``.
        """
        conn, ix, cortex, thalamus, a, nets = self.setup(nvois=2)
        tavg = TemporalAverage(period=1.0)
        sim = Simulator(
            nets=nets,
            simulation_length=10.0,
            monitors=[tavg],
        )
        sim.configure()
        ((t, y),) = sim.run()
        self.assert_equal(10, len(t))
        self.assert_equal((10, 4, 76, 1), y.shape)

    def test_sim_jrmon(self):
        """Simulation with a per-subnetwork Recorder monitor.

        When the monitor is attached directly to a subnetwork (not passed to
        :class:`~tvb.simulator.hybrid.Simulator`), ``sim.run()`` returns an
        empty list and results must be fetched from ``cortex.monitors[0]``.
        Output shape should be ``(10, n_vois, n_cortex_nodes, modes=1)``.
        """
        jrmon = TemporalAverage(period=1.0)
        conn, ix, cortex, thalamus, a, nets = self.setup(jrmon=jrmon)
        sim = Simulator(nets=nets, simulation_length=10.0)
        sim.configure()
        xs = sim.run()
        self.assert_equal(0, len(xs))
        rec: Recorder = cortex.monitors[0]
        nn = cortex.nnodes
        self.assert_equal((10, 4, nn, 1), rec.shape)
        t, y = rec.to_arrays()
        self.assert_equal((10,), t.shape)
        self.assert_equal((10, 4, nn, 1), y.shape)

    def test_module_example(self):
        """Reproduce the canonical usage example from the hybrid simulator documentation.

        Builds a two-subnetwork coupled system (JansenRit + ReducedSetFitzHughNagumo),
        connects them via an :class:`~tvb.simulator.hybrid.InterProjection`, and runs
        for 100 ms.  Confirms 100 time steps are returned and the simulation
        completes without error.
        """
        # Create subnetworks with different models
        # Specify the same number of variables of interest for both models
        jrkwargs = {
            "variables_of_interest": JansenRit.variables_of_interest.default[:2]
        }
        fhnkwargs = {
            "variables_of_interest": ReducedSetFitzHughNagumo.variables_of_interest.default[
                :2
            ]
        }

        cortex = Subnetwork(
            name="cortex",
            model=JansenRit(**jrkwargs),
            scheme=HeunDeterministic(dt=0.1),
            nnodes=76,
        ).configure()  # Configure the model

        thalamus = Subnetwork(
            name="thalamus",
            model=ReducedSetFitzHughNagumo(**fhnkwargs),
            scheme=HeunDeterministic(dt=0.1),
            nnodes=76,
        ).configure()  # Configure the model

        # Prepare projection parameters
        weights_data = np.random.randn(76, 76)
        weights_matrix = csr_matrix(weights_data)

        # Create lengths matrix with the same sparsity pattern as weights
        lengths_data = np.abs(np.random.randn(76, 76))  # Ensure positive lengths
        lengths_data[weights_data == 0] = 0  # Match sparsity
        lengths_matrix = csr_matrix(lengths_data)

        projection_dt = 0.1  # Match scheme dt
        projection_cv = 1.0

        # Define projections between subnetworks
        nets = NetworkSet(
            subnets=[cortex, thalamus],
            projections=[
                InterProjection(
                    source=cortex,
                    target=thalamus,
                    source_cvar=np.r_[0],
                    target_cvar=np.r_[1],
                    weights=weights_matrix,
                    lengths=lengths_matrix,
                    cv=projection_cv,
                    dt=projection_dt,
                )
            ],
        )

        # Simulate the coupled system
        tavg = TemporalAverage(period=1.0)  # Add a monitor
        sim = Simulator(
            nets=nets,
            simulation_length=100,
            monitors=[tavg],  # Include the monitor
        )
        sim.configure()
        ((t, y),) = sim.run()  # Unpack the first (and only) monitor result

        # Verify the simulation ran successfully
        self.assert_equal(100, len(t))
        # The output shape is (time_steps, variables_of_interest, total_nodes, modes)
        # Total nodes = cortex nodes + thalamus nodes = 76 + 76 = 152
        self.assert_equal((100, 4, 152, 1), y.shape)
