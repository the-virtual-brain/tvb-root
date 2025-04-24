# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
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

"""
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import numpy as np
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.simulator.hybrid import NetworkSet, Subnetwork, Projection, Simulator, Recorder
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.models import JansenRit, ReducedSetFitzHughNagumo
from tvb.simulator.integrators import HeunDeterministic
from tvb.datatypes.patterns import StimuliRegion
from tvb.datatypes.equations import DiscreteEquation, TemporalApplicableEquation
from tvb.simulator.monitors import TemporalAverage
import pytest


class TestHybrid1(BaseTestCase):
    
    def setup(self, nvois=None, jrmon=None):
        """Setup test environment with cortex-thalamus network"""
        conn = Connectivity.from_file()
        conn.configure()
        np.random.seed(42)

        # Define brain regions
        class AtlasPart:
            CORTEX = 0
            THALAMUS = 1

        ix = np.random.randint(
            low=0, high=2, size=conn.number_of_regions)

        scheme = HeunDeterministic(dt=0.1)

        jrkwargs = {}
        fhnkwargs = {}
        if nvois:
            jrkwargs['variables_of_interest'] = JansenRit.variables_of_interest.default[:nvois]
            fhnkwargs['variables_of_interest'] = ReducedSetFitzHughNagumo.variables_of_interest.default[:nvois]

        # Create subnetworks with just their size and behavior
        cortex = Subnetwork(
            name='cortex',
            model=JansenRit(**jrkwargs),
            scheme=scheme,
            nnodes=(ix == AtlasPart.CORTEX).sum(),
        ).configure()

        if jrmon:
            cortex.add_monitor(jrmon)

        thalamus = Subnetwork(
            name='thalamus',
            model=ReducedSetFitzHughNagumo(**fhnkwargs),
            scheme=scheme,
            nnodes=(ix == AtlasPart.THALAMUS).sum(),
        ).configure()

        # Create projections with explicit weights from global connectivity
        cortex_indices = np.where(ix == AtlasPart.CORTEX)[0]
        thalamus_indices = np.where(ix == AtlasPart.THALAMUS)[0]

        # Use valid coupling variable indices for each model
        # JansenRit has 2 coupling variables (0,1)
        nets = NetworkSet(
            subnets=[cortex, thalamus],
            projections=[
                Projection(
                    source=cortex, target=thalamus,
                    source_cvar=np.r_[0], target_cvar=np.r_[1],
                    weights=conn.weights[thalamus_indices][:, cortex_indices],
                ),
                Projection(
                    source=cortex, target=thalamus,
                    source_cvar=np.r_[1], target_cvar=np.r_[0],  # Changed from 5 to 1
                    weights=conn.weights[thalamus_indices][:, cortex_indices],
                ),
                Projection(
                    source=thalamus, target=cortex,
                    source_cvar=np.r_[0], target_cvar=np.r_[1],
                    scale=1e-2,
                    weights=conn.weights[cortex_indices][:, thalamus_indices],
                ),
            ]
        )

        return conn, ix, cortex, thalamus, AtlasPart, nets

    def _randn_like_states(self, states):
        """Generate random states with same structure as input states.
        
        Parameters
        ----------
        states : NetworkSet.States
            Template states to match structure of
            
        Returns
        -------
        NetworkSet.States
            New states filled with random numbers
        """
        return states.__class__(*(np.random.randn(*x.shape) for x in states))

    def _randn_like_cvars(self, cvars):
        """Generate random coupling variables with same structure as input cvars.
        
        Parameters
        ----------
        cvars : NetworkSet.States
            Template coupling variables to match structure of
            
        Returns
        -------
        NetworkSet.States
            New coupling variables filled with random numbers
        """
        return cvars.__class__(*(np.random.randn(*x.shape) for x in cvars))

    def test_subnetwork(self):
        """Test subnetwork state and coupling variable shapes"""
        conn, ix, c, t, a, nets = self.setup()
        self.assert_equal((6, (ix == a.CORTEX).sum(), 1), c.zero_states().shape)
        self.assert_equal((2, (ix == a.CORTEX).sum(), 1), c.zero_cvars().shape)
        self.assert_equal((4, (ix == a.THALAMUS).sum(), 3), t.zero_states().shape)

    def test_projection_broadcasting(self):
        """Test different coupling variable broadcasting configurations"""
        conn, ix, cortex, thalamus, a, nets = self.setup()
        
        # Get indices for test setup
        cortex_indices = np.where(ix == a.CORTEX)[0]
        thalamus_indices = np.where(ix == a.THALAMUS)[0]
        test_weights = conn.weights[thalamus_indices][:, cortex_indices]
        
        # Case a: One source to many targets (broadcasting)
        proj_broadcast = Projection(
            source=cortex,
            target=thalamus,
            source_cvar=np.r_[0],  # Single source
            target_cvar=np.r_[0:2],  # Multiple targets [0,1]
            weights=test_weights
        )
        
        # Case b: Many sources to one target (reduction)
        proj_reduce = Projection(
            source=cortex,
            target=thalamus,
            source_cvar=np.r_[0, 1],  # Multiple sources
            target_cvar=np.r_[1],  # Single target
            weights=test_weights
        )
        
        # Case c: Equal number of sources to targets (element-wise)
        proj_elementwise = Projection(
            source=cortex,
            target=thalamus,
            source_cvar=np.r_[0, 1],  # Two sources
            target_cvar=np.r_[0, 1],  # Two targets
            weights=test_weights
        )
        
        # Create random test states
        x = self._randn_like_states(nets.zero_states())
        
        # Test case a: broadcasting
        c_broadcast = nets.zero_cvars()
        proj_broadcast.apply(c_broadcast.thalamus, x.cortex)
        expected_val = test_weights @ x.cortex[0] @ proj_broadcast.mode_map
        assert c_broadcast.thalamus[0].shape == expected_val.shape
        np.testing.assert_allclose(c_broadcast.thalamus[0], expected_val)
        np.testing.assert_allclose(c_broadcast.thalamus[1], expected_val)
        
        # Test case b: reduction
        c_reduce = nets.zero_cvars()
        proj_reduce.apply(c_reduce.thalamus, x.cortex)
        expected_val = (test_weights @ x.cortex[0] + test_weights @ x.cortex[1]) @ proj_reduce.mode_map
        np.testing.assert_allclose(c_reduce.thalamus[1], expected_val)
        
        # Test case c: element-wise
        c_elementwise = nets.zero_cvars()
        proj_elementwise.apply(c_elementwise.thalamus, x.cortex)
        expected_val0 = test_weights @ x.cortex[0] @ proj_elementwise.mode_map
        expected_val1 = test_weights @ x.cortex[1] @ proj_elementwise.mode_map
        np.testing.assert_allclose(c_elementwise.thalamus[0], expected_val0)
        np.testing.assert_allclose(c_elementwise.thalamus[1], expected_val1)
        
        # Test invalid configuration
        with pytest.raises(ValueError):
            Projection(
                source=cortex,
                target=thalamus,
                source_cvar=np.r_[0, 1],
                target_cvar=np.r_[0, 1, 2],
                weights=test_weights
            )

    def test_projection_validation(self):
        """Test projection validation for coupling variables"""
        conn, ix, cortex, thalamus, a, nets = self.setup()
        
        # Setup test weights
        cortex_indices = np.where(ix == a.CORTEX)[0]
        thalamus_indices = np.where(ix == a.THALAMUS)[0]
        test_weights = conn.weights[thalamus_indices][:, cortex_indices]
        
        # Test source index out of bounds
        with pytest.raises(ValueError):  # Use pytest.raises
            Projection(
                source=cortex,
                target=thalamus,
                source_cvar=np.r_[999],  # Invalid index
                target_cvar=np.r_[0],
                weights=test_weights
            )
        
        # Test target index out of bounds
        with pytest.raises(ValueError):  # Use pytest.raises
            Projection(
                source=cortex,
                target=thalamus,
                source_cvar=np.r_[0],
                target_cvar=np.r_[999],  # Invalid index
                weights=test_weights
            )
        
        # Test array of source indices out of bounds
        with pytest.raises(ValueError):  # Use pytest.raises
            Projection(
                source=cortex,
                target=thalamus,
                source_cvar=np.r_[0, 999],  # Second index invalid
                target_cvar=np.r_[0, 1],
                weights=test_weights
            )
        
        # Test array of target indices out of bounds
        with pytest.raises(ValueError):  # Use pytest.raises
            Projection(
                source=cortex,
                target=thalamus,
                source_cvar=np.r_[0, 1],
                target_cvar=np.r_[0, 999],  # Second index invalid
                weights=test_weights
            )
        
        # Test invalid broadcasting configuration - mismatched sizes
        with pytest.raises(ValueError):  # Use pytest.raises
            Projection(
                source=cortex,
                target=thalamus,
                source_cvar=np.r_[0, 1],  # Size 2
                target_cvar=np.r_[0:3],  # Size 3
                weights=test_weights
            )

    def test_networkset(self):
        """Test network coupling computation"""
        conn, ix, cortex, thalamus, a, nets = self.setup()
        
        # Create random test states and coupling variables
        x = self._randn_like_states(nets.zero_states())
        c = self._randn_like_cvars(nets.zero_cvars())  # For testing initial values
        
        # Test coupling computation
        c_new = nets.cfun(x)
        
        # Verify coupling for each projection
        for proj in nets.projections:
            src_state = getattr(x, proj.source.name)
            tgt_coupling = getattr(c_new, proj.target.name)
            
            if len(proj.source_cvar) == 1:  # Broadcasting case
                expected_val = proj.weights @ src_state[proj.source_cvar[0]] @ proj.mode_map
                for tgt_idx in proj.target_cvar:
                    np.testing.assert_allclose(tgt_coupling[tgt_idx], proj.scale * expected_val)
            
            elif len(proj.target_cvar) == 1:  # Reduction case
                expected_val = sum(proj.weights @ src_state[i] @ proj.mode_map 
                                 for i in proj.source_cvar)
                np.testing.assert_allclose(tgt_coupling[proj.target_cvar[0]], 
                                        proj.scale * expected_val)
            
            else:  # Element-wise case
                for src_idx, tgt_idx in zip(proj.source_cvar, proj.target_cvar):
                    expected_val = proj.weights @ src_state[src_idx] @ proj.mode_map
                    np.testing.assert_allclose(tgt_coupling[tgt_idx], 
                                            proj.scale * expected_val)

    def test_netset_step(self):
        """Test network time stepping"""
        conn, ix, cortex, thalamus, a, nets = self.setup()
        x = nets.zero_states()
        nx = nets.step(0, x)
        self.assert_equal(
            [(6, 37, 1), (4, 39, 3)], nx.shape
        )

    def test_sim(self):
        """Test full simulation with temporal averaging"""
        conn, ix, cortex, thalamus, a, nets = self.setup(nvois=2)
        tavg = TemporalAverage(period=1.0)
        sim = Simulator(
            nets=nets,
            simulation_length=10,
            monitors=[tavg],
        )
        sim.configure()
        (t,y), = sim.run()
        self.assert_equal(10, len(t))
        self.assert_equal((10, 2, 76, 1), y.shape)

    def test_sim_jrmon(self):
        """Test simulation with Jansen-Rit monitor"""
        jrmon = TemporalAverage(period=1.0)
        conn, ix, cortex, thalamus, a, nets = self.setup(jrmon=jrmon)
        sim = Simulator(nets=nets, simulation_length=10)
        sim.configure()
        xs = sim.run()
        self.assert_equal(0, len(xs))
        rec: Recorder = cortex.monitors[0]
        nn = cortex.nnodes
        self.assert_equal((10, 4, nn, 1), rec.shape)
        t, y = rec.to_arrays()
        self.assert_equal((10, ), t.shape)
        self.assert_equal((10, 4, nn, 1), y.shape)
        
    def test_module_example(self):
        """Test the example from the module docstring"""
        # Create subnetworks with different models
        # Specify the same number of variables of interest for both models
        jrkwargs = {'variables_of_interest': JansenRit.variables_of_interest.default[:2]}
        fhnkwargs = {'variables_of_interest': ReducedSetFitzHughNagumo.variables_of_interest.default[:2]}
        
        cortex = Subnetwork(
            name='cortex',
            model=JansenRit(**jrkwargs),
            scheme=HeunDeterministic(dt=0.1),
            nnodes=76
        ).configure()  # Configure the model
        
        thalamus = Subnetwork(
            name='thalamus',
            model=ReducedSetFitzHughNagumo(**fhnkwargs),
            scheme=HeunDeterministic(dt=0.1),
            nnodes=76
        ).configure()  # Configure the model
        
        # Define projections between subnetworks
        nets = NetworkSet(
            subnets=[cortex, thalamus],
            projections=[
                Projection(
                    source=cortex, target=thalamus,
                    source_cvar=np.r_[0], target_cvar=np.r_[1],
                    weights=np.random.randn(76, 76)
                )
            ]
        )
        
        # Simulate the coupled system
        tavg = TemporalAverage(period=1.0)  # Add a monitor
        sim = Simulator(
            nets=nets, 
            simulation_length=100,
            monitors=[tavg]  # Include the monitor
        )
        sim.configure()
        (t, y), = sim.run()  # Unpack the first (and only) monitor result
        
        # Verify the simulation ran successfully
        self.assert_equal(100, len(t))
        # The output shape is (time_steps, variables_of_interest, total_nodes, modes)
        # Total nodes = cortex nodes + thalamus nodes = 76 + 76 = 152
        self.assert_equal((100, 2, 152, 1), y.shape)


class TestHybrid2(BaseTestCase):

    def test_stim(self):
        """Test stimulus application to network"""
        conn = Connectivity.from_file()
        nn = conn.weights.shape[0]
        conn.configure()
        class MyStim(StimuliRegion):
            def __call__(self, t):
                return np.random.randn(self.connectivity.weights.shape[0])
        stim = MyStim(connectivity=conn)
        I = stim(5)
        self.assert_equal((nn,), I.shape)
        model = JansenRit()
        model.configure()
        # TODO
