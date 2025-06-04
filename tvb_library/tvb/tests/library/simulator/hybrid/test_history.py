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

from tvb.basic.neotraits.api import List as ListTrait
from tvb.simulator.models.base import Model
from tvb.simulator.integrators import Integrator
from tvb.simulator.monitors import Raw
from tvb.simulator.hybrid import (
    Subnetwork, IntraProjection, InterProjection, NetworkSet, Simulator
)
from tvb.tests.library.base_testcase import BaseTestCase


class SumTestModel(Model):
    """Simplified Sum model for testing, similar to history_test.py."""
    nvar = 1
    _nvar = 1
    state_variable_range = {'x': [0, 100]}
    variables_of_interest = ListTrait(of=str, default=('x',), choices=('x',))
    state_variables = ('x',)
    cvar = np.array([0]) # Indicates that the first state variable is also a coupling variable

    def dfun(self, X, coupling, local_coupling=0.0):
        return X + coupling + local_coupling


class IdentityTestIntegrator(Integrator):
    """Simplified Identity integrator for testing, similar to history_test.py."""
    def __init__(self, dt=1.0): # dt is not used by tvb.integrators.Identity but kept for consistency if needed elsewhere
        super().__init__()
        self.dt = dt

    def scheme(self, X, dfun, coupling, local_coupling, stimulus): # Matches base Integrator signature
        # This should behave like tvb.simulator.integrators.Identity,
        # which directly returns the result of dfun.
        # The stimulus parameter is per the base class, though not used by SumTestModel.dfun.
        return dfun(X, coupling, local_coupling=local_coupling)


class TestHybridPropagation(BaseTestCase):

    def _get_expected_results(self):
        # This is the exact xs_ array from TestsExactPropagation
        return np.array([[2., 2., 2., 1.],
                           [3., 3., 3., 1.],
                           [5., 4., 4., 1.],
                           [8., 5., 5., 1.],
                           [12., 6., 6., 1.],
                           [17., 7., 7., 1.],
                           [23., 8., 8., 1.],
                           [30., 10., 9., 1.],
                           [38., 13., 10., 1.],
                           [48., 17., 11., 1.]])

    def _setup_common_parameters(self, n_nodes=4):
        # Connectivity: node i -> node i+1
        conn_matrix_dense = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes - 1):
            conn_matrix_dense[i, i + 1] = 1.0
        
        # Distances (lengths)
        dist_matrix_dense = np.arange(n_nodes * n_nodes, dtype=float).reshape((n_nodes, n_nodes))
        
        # Simulation parameters
        cv = 1.0
        dt = 1.0
        simulation_length = 10.0

        # Initial conditions: all nodes start at 1.0
        # Shape: (nvar, nnodes, nmodes)
        initial_conditions = np.ones((1, n_nodes, 1), dtype=float)

        return conn_matrix_dense, dist_matrix_dense, cv, dt, simulation_length, initial_conditions

    def test_single_subnetwork_intra_projection(self):
        n_nodes = 4
        conn_dense, dist_dense, cv, dt, sim_len, initial_cond = self._setup_common_parameters(n_nodes)
        
        weights_sparse = sp.csr_matrix(conn_dense)
        # Ensure lengths_sparse has the same sparsity pattern as weights_sparse
        # so that their nnz are identical before eps_matrix addition in BaseProjection.
        # Extract only the length values corresponding to actual connections in weights.
        rows, cols = weights_sparse.nonzero()
        lengths_data = dist_dense[rows, cols]
        lengths_sparse = sp.csr_matrix(
            (lengths_data, (rows, cols)), # Use (rows, cols) for indices
            shape=weights_sparse.shape
        )

        model = SumTestModel()
        # NOTE: Hybrid simulator's Integrator instances are configured by Subnetwork if not already.
        # Here, IdentityTestIntegrator is simple and dt is its only param.
        scheme = IdentityTestIntegrator(dt=dt)

        # Create IntraProjection
        # For IntraProjection, weights and lengths define internal connections.
        # Here, we use the global connectivity directly as if it's all internal.
        intra_proj = IntraProjection(
            source_cvar=np.array([0]),
            target_cvar=np.array([0]),
            weights=weights_sparse,
            lengths=lengths_sparse,
            cv=cv,
            dt=dt 
        )
        
        subnet = Subnetwork(
            name='subnet_A',
            model=model,
            scheme=scheme,
            nnodes=n_nodes,
            projections=[intra_proj]
        )
        # Add a Raw monitor to the subnetwork's recorder list
        raw_monitor = Raw()
        raw_monitor.period = dt # Ensure it samples every dt
        subnet.add_monitor(raw_monitor)

        nets = NetworkSet(subnets=[subnet], projections=[]) # No inter-subnetwork projections

        # IMPORTANT: The hybrid Simulator currently initializes states to zero via nets.zero_states().
        # To use custom initial_conditions, this would need to be handled, e.g., by modifying
        # how 'x' is initialized in the Simulator.run() loop or by allowing
        # NetworkSet.zero_states() to accept initial_conditions.
        # For this test, we proceed with the current API, which will likely lead to
        # all-zero output if SumTestModel().dfun(0,0)=0.
        
        sim = Simulator(
            nets=nets,
            simulation_length=sim_len,
            # Simulator-level monitors are not used here; data comes from Subnetwork's Recorder
        )
        sim.configure() # Configures subnetworks, their projections, and recorders
        
        # Run simulation
        sim.run(initial_conditions=[initial_cond]) # Pass initial conditions here

        # Retrieve data from the subnetwork's recorder
        # Assuming only one monitor (Raw) was added to the subnetwork
        recorder = subnet.monitors[0]
        sim_times, sim_data = recorder.to_arrays()

        # Reshape data: (time_steps, nvar, nnodes, nmodes) -> (time_steps, nnodes)
        # SumTestModel has nvar=1, nmodes=1
        sim_data_squeezed = sim_data.squeeze(axis=(1, 3))

        expected_xs = self._get_expected_results()
        
        # Assertions
        self.assert_equal(sim_data_squeezed.shape, expected_xs.shape)
        # Note: This numerical assertion is expected to fail.
        # The primary reasons are:
        # 1. Delay Handling: BaseProjection in the hybrid simulator imposes a minimum delay of
        #    2 steps (floor(L/cv/dt) + 2), whereas the original simulator's history can
        #    handle delays of 0 or 1 step.
        # 2. Initial History: The original simulator pre-fills its history buffer with initial
        #    conditions, allowing non-zero coupling from the very first step. Hybrid projections
        #    initialize their buffers to zero and update them step-by-step. With a minimum
        #    delay of 2, initial coupling is typically zero.
        # 3. The `expected_xs` array implies specific dynamics (e.g., for Node 0) that are
        #    not straightforwardly reproduced by an `X_new = X_old + C` model given the
        #    connectivity if Node 0 has no inputs and C_node0 is non-zero.
        # The hybrid simulation correctly implements its own model: X_new = X_old + C_total_hybrid_delays.
        np.testing.assert_allclose(sim_data_squeezed, expected_xs, rtol=1e-5, atol=1e-8)


    def test_two_subnetworks_inter_projection(self):
        n_total_nodes = 4
        # Split nodes: SN1 (nodes 0, 1), SN2 (nodes 2, 3)
        n_sn1 = 2
        n_sn2 = 2
        
        conn_global_dense, dist_global_dense, cv, dt, sim_len, _ = \
            self._setup_common_parameters(n_total_nodes)

        # Initial conditions for each subnetwork
        initial_cond_sn1 = np.ones((1, n_sn1, 1), dtype=float)
        initial_cond_sn2 = np.ones((1, n_sn2, 1), dtype=float)

        model_sn1 = SumTestModel()
        scheme_sn1 = IdentityTestIntegrator(dt=dt)
        model_sn2 = SumTestModel()
        scheme_sn2 = IdentityTestIntegrator(dt=dt)

        # --- Projections ---
        # IntraProjection for SN1 (original node 0 -> 1)
        # SN1 nodes are [0, 1]. Connection is SN1_node0 -> SN1_node1
        weights_intra1_dense = np.zeros((n_sn1, n_sn1))
        weights_intra1_dense[0, 1] = conn_global_dense[0, 1]
        lengths_intra1_dense = np.zeros((n_sn1, n_sn1))
        lengths_intra1_dense[0, 1] = dist_global_dense[0, 1]
        proj_intra1 = IntraProjection(
            source_cvar=np.array([0]), target_cvar=np.array([0]),
            weights=sp.csr_matrix(weights_intra1_dense),
            lengths=sp.csr_matrix(lengths_intra1_dense),
            cv=cv, dt=dt
        )

        # IntraProjection for SN2 (original node 2 -> 3)
        # SN2 nodes are [0, 1] (locally). Connection is SN2_node0 -> SN2_node1
        weights_intra2_dense = np.zeros((n_sn2, n_sn2))
        weights_intra2_dense[0, 1] = conn_global_dense[2, 3]
        lengths_intra2_dense = np.zeros((n_sn2, n_sn2))
        lengths_intra2_dense[0, 1] = dist_global_dense[2, 3]
        proj_intra2 = IntraProjection(
            source_cvar=np.array([0]), target_cvar=np.array([0]),
            weights=sp.csr_matrix(weights_intra2_dense),
            lengths=sp.csr_matrix(lengths_intra2_dense),
            cv=cv, dt=dt
        )

        # Subnetworks
        sn1 = Subnetwork(name='sn1', model=model_sn1, scheme=scheme_sn1, nnodes=n_sn1, projections=[proj_intra1])
        raw_mon_sn1 = Raw()
        raw_mon_sn1.period = dt
        sn1.add_monitor(raw_mon_sn1)

        sn2 = Subnetwork(name='sn2', model=model_sn2, scheme=scheme_sn2, nnodes=n_sn2, projections=[proj_intra2])
        raw_mon_sn2 = Raw()
        raw_mon_sn2.period = dt
        sn2.add_monitor(raw_mon_sn2)
        
        # InterProjection SN1 -> SN2 (original node 1 -> 2)
        # Connects SN1_node1 to SN2_node0
        # Weights shape: (target_nodes, source_nodes) = (n_sn2, n_sn1)
        weights_inter_dense = np.zeros((n_sn2, n_sn1))
        weights_inter_dense[0, 1] = conn_global_dense[1, 2] # SN2_node0 from SN1_node1
        lengths_inter_dense = np.zeros((n_sn2, n_sn1))
        lengths_inter_dense[0, 1] = dist_global_dense[1, 2]
        proj_inter_sn1_sn2 = InterProjection(
            source=sn1, target=sn2,
            source_cvar=np.array([0]), target_cvar=np.array([0]),
            weights=sp.csr_matrix(weights_inter_dense),
            lengths=sp.csr_matrix(lengths_inter_dense),
            cv=cv, dt=dt
        )

        nets = NetworkSet(subnets=[sn1, sn2], projections=[proj_inter_sn1_sn2])
        
        # See comment in test_single_subnetwork_intra_projection regarding initial conditions.
        sim = Simulator(
            nets=nets,
            simulation_length=sim_len,
        )
        sim.configure()
        sim.run(initial_conditions=[initial_cond_sn1, initial_cond_sn2]) # Pass initial conditions here

        # Retrieve data
        rec1 = sn1.monitors[0]
        _, data_sn1 = rec1.to_arrays() # (ts, nv, n_sn1, nm)
        data_sn1_squeezed = data_sn1.squeeze(axis=(1,3)) # (ts, n_sn1)

        rec2 = sn2.monitors[0]
        _, data_sn2 = rec2.to_arrays() # (ts, nv, n_sn2, nm)
        data_sn2_squeezed = data_sn2.squeeze(axis=(1,3)) # (ts, n_sn2)

        # Combine data from SN1 and SN2 (SN1 nodes first, then SN2 nodes)
        sim_data_combined = np.concatenate((data_sn1_squeezed, data_sn2_squeezed), axis=1)

        expected_xs = self._get_expected_results()

        # Assertions
        self.assert_equal(sim_data_combined.shape, expected_xs.shape)
        # Note: This numerical assertion is expected to fail for the same reasons
        # as in test_single_subnetwork_intra_projection:
        # 1. Differences in delay calculation (minimum 2 steps in hybrid vs. potentially 0/1).
        # 2. Differences in how initial history affects initial coupling values.
        # 3. The specific generation mechanism of `expected_xs`.
        # The hybrid simulation correctly implements its own model: X_new = X_old + C_total_hybrid_delays.
        np.testing.assert_allclose(sim_data_combined, expected_xs, rtol=1e-5, atol=1e-8)
