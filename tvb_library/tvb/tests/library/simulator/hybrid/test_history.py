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
Test history in hybrid simulator, analogous to tvb_library/tvb/tests/library/simulator/history_test.py
"""

import numpy as np
import scipy.sparse as sp

from tvb.simulator.models.base import Model
from tvb.simulator.integrators import Identity
from tvb.simulator.monitors import Raw
from tvb.simulator.hybrid import (
    Subnetwork, IntraProjection, NetworkSet, Simulator
)
from tvb.tests.library.simulator.history_test import IdCoupling, Sum, TestsExactPropagation

class TestHybridHistory(TestsExactPropagation):
    """
    This class tests the history of a hybrid simulator.
    It is based on tvb/tests/library/simulator/history_test.py::TestsExactPropagation, but adapted to use the new hybrid simulator.
    """

    def build_simulator(self, n=4):
        super().build_simulator(n)

        model = self.sim.model
        # Use Identity integrator with dt from the classic simulator
        integrator = self.sim.integrator

        W = self.sim.connectivity.weights
        L = self.sim.connectivity.tract_lengths

        weights_csr = sp.csr_matrix(W)

        rows, cols = W.nonzero()
        sparse_lengths_values = L[rows, cols]
        lengths_csr = sp.csr_matrix((sparse_lengths_values, (rows, cols)), shape=W.shape)

        intra_projection = IntraProjection(
            source_cvar=model.cvar,
            target_cvar=model.cvar,
            weights=weights_csr,
            lengths=lengths_csr,
            cv=self.sim.conduction_speed,
            dt=self.sim.integrator.dt
        )

        subnetwork = Subnetwork(
            name='subnet',
            model=model,
            scheme=integrator,
            nnodes=n,
            projections=[intra_projection]
        )

        network_set = NetworkSet(
            subnets=[subnetwork],
        )

        hybrid_simulator = Simulator(
            nets=network_set,
            simulation_length=self.sim.simulation_length,
            monitors=[self.sim.monitors[0]]
        )

        self.hsim = hybrid_simulator
        self.hsim.configure()

    def test_propagation(self):
        nnodes = 4
        self.build_simulator(nnodes)
        # Expected results (from TestsExactPropagation)
        xs_expected = np.array([[2., 2., 2., 1.],
                                [3., 3., 3., 1.],
                                [5., 4., 4., 1.],
                                [8., 5., 5., 1.],
                                [12., 6., 6., 1.],
                                [17., 7., 7., 1.],
                                [23., 8., 8., 1.],
                                [30., 10., 9., 1.],
                                [38., 13., 10., 1.],
                                [48., 17., 11., 1.]])

        model = self.hsim.nets.subnets[0].model
        init = np.ones((model.nvar,
                        nnodes,
                        model.number_of_modes))
        p: IntraProjection = self.hsim.nets.subnets[0].projections[0]

        # initialize the history buffer, required for exact results here
        # XXX fix this part of the API
        for i in range(p._horizon):
            p.update_buffer(init, i)

        (t, xs), = self.hsim.run(initial_conditions=[init])
        xs = xs.reshape(xs_expected.shape)
        for t in range(10):
            np.testing.assert_allclose(xs[t], xs_expected[t])
