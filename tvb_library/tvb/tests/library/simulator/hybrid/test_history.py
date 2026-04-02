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
Test history-buffer propagation in the hybrid simulator.

This module is the hybrid-simulator analogue of
:mod:`tvb.tests.library.simulator.history_test`.  It re-uses
:class:`~tvb.tests.library.simulator.history_test.TestsExactPropagation`
(the ``IdCoupling`` / ``Sum`` / identity-integrator fixture from that module)
and adapts the exact-propagation test to run through
:class:`~tvb.simulator.hybrid.Simulator` with a single
:class:`~tvb.simulator.hybrid.Subnetwork` and an
:class:`~tvb.simulator.hybrid.IntraProjection`.

The test verifies that the hybrid delay buffer faithfully reproduces the
analytical propagation sequence expected for a small 4-node network with
known connection delays.
"""

import numpy as np
import scipy.sparse as sp

from tvb.simulator.models.base import Model
from tvb.simulator.integrators import Identity
from tvb.simulator.monitors import Raw
from tvb.simulator.hybrid import Subnetwork, IntraProjection, NetworkSet, Simulator
from tvb.tests.library.simulator.history_test import (
    IdCoupling,
    Sum,
    TestsExactPropagation,
)


class TestHybridHistory(TestsExactPropagation):
    """
    Hybrid-simulator equivalent of
    :class:`~tvb.tests.library.simulator.history_test.TestsExactPropagation`.

    :meth:`build_simulator` first calls the parent implementation to create a
    classic TVB simulator (``self.sim``) so that its model, integrator, and
    connectivity parameters are reused.  It then wraps the same components in
    a :class:`~tvb.simulator.hybrid.NetworkSet` / :class:`~tvb.simulator.hybrid.Simulator`
    and stores the result in ``self.hsim``.

    :meth:`test_propagation` checks that the hybrid output matches the
    hard-coded analytical solution from the parent class, step by step.
    """

    def build_simulator(self, n=4):
        """
        Build both a classic and a hybrid simulator sharing the same parameters.

        Calls ``super().build_simulator(n)`` to populate ``self.sim`` with the
        classic TVB simulator (model, integrator, connectivity).  Then mirrors
        that configuration in a hybrid
        :class:`~tvb.simulator.hybrid.Subnetwork` + 
        :class:`~tvb.simulator.hybrid.IntraProjection` topology stored in
        ``self.hsim``.

        The intra-projection uses sparse CSR representations of the classic
        simulator's weight and tract-length matrices, so the delay buffers
        are initialised with the same conduction speed and time step.

        Parameters
        ----------
        n : int, default=4
            Number of nodes in the test network.
        """
        super().build_simulator(n)

        model = self.sim.model
        # Use Identity integrator with dt from the classic simulator
        integrator = self.sim.integrator

        W = self.sim.connectivity.weights
        L = self.sim.connectivity.tract_lengths

        weights_csr = sp.csr_matrix(W)

        rows, cols = W.nonzero()
        sparse_lengths_values = L[rows, cols]
        lengths_csr = sp.csr_matrix(
            (sparse_lengths_values, (rows, cols)), shape=W.shape
        )

        intra_projection = IntraProjection(
            source_cvar=model.cvar,
            target_cvar=model.cvar,
            weights=weights_csr,
            lengths=lengths_csr,
            cv=self.sim.conduction_speed,
            dt=self.sim.integrator.dt,
        )

        subnetwork = Subnetwork(
            name="subnet",
            model=model,
            scheme=integrator,
            nnodes=n,
            projections=[intra_projection],
        )

        network_set = NetworkSet(
            subnets=[subnetwork],
        )

        hybrid_simulator = Simulator(
            nets=network_set,
            simulation_length=self.sim.simulation_length,
            monitors=[self.sim.monitors[0]],
        )

        self.hsim = hybrid_simulator
        self.hsim.configure()

    def test_propagation(self):
        """
        The hybrid delay buffer propagates values according to the analytical
        solution derived in the parent class.

        Sets up a 4-node network, initialises the history buffer with all-ones,
        advances ten steps, and compares the output node-by-node against the
        hard-coded ``xs_expected`` matrix from
        :class:`~tvb.tests.library.simulator.history_test.TestsExactPropagation`.
        Each row of ``xs_expected`` represents the expected state of the 4 nodes
        at one time step; the comparison uses
        :func:`numpy.testing.assert_allclose` with default tolerances.
        """
        self.build_simulator(4)
        # Expected results (from TestsExactPropagation)
        xs_expected = np.array(
            [
                [2.0, 2.0, 2.0, 1.0],
                [3.0, 3.0, 3.0, 1.0],
                [5.0, 4.0, 4.0, 1.0],
                [8.0, 5.0, 5.0, 1.0],
                [12.0, 6.0, 6.0, 1.0],
                [17.0, 7.0, 7.0, 1.0],
                [23.0, 8.0, 8.0, 1.0],
                [30.0, 10.0, 9.0, 1.0],
                [38.0, 13.0, 10.0, 1.0],
                [48.0, 17.0, 11.0, 1.0],
            ]
        )

        model = self.hsim.nets.subnets[0].model
        init = np.ones((model.nvar, 4, model.number_of_modes))
        p: IntraProjection = self.hsim.nets.subnets[0].projections[0]

        # Initialize the history buffer with the initial state
        p.initialize_history_buffer(init)

        ((t, xs),) = self.hsim.run(initial_conditions=[init])
        xs = xs.reshape(xs_expected.shape)
        for t in range(10):
            np.testing.assert_allclose(xs[t], xs_expected[t])
