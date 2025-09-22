"""
Tests for the Simulator class.
"""

import numpy as np
from scipy.sparse import csr_matrix
from tvb.simulator.models import JansenRit, ReducedSetFitzHughNagumo, MontbrioPazoRoxin
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.monitors import TemporalAverage
from tvb.simulator.hybrid import Recorder, Simulator, Subnetwork, InterProjection, NetworkSet
from .test_base import BaseHybridTest


class TestSimulator(BaseHybridTest):
    """Tests for the Simulator class."""

    def test_sim(self):
        """Test full simulation with temporal averaging"""
        conn, ix, cortex, thalamus, a, nets = self.setup(nvois=2)
        tavg = TemporalAverage(period=1.0)
        sim = Simulator(
            nets=nets,
            simulation_length=10.0,
            monitors=[tavg],
        )
        sim.configure()
        (t, y), = sim.run()
        self.assert_equal(10, len(t))
        self.assert_equal((10, 2, 76, 1), y.shape)

    def test_sim_jrmon(self):
        """Test simulation with Jansen-Rit monitor"""
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
                    source=cortex, target=thalamus,
                    source_cvar=np.r_[0], target_cvar=np.r_[1],
                    weights=weights_matrix,
                    lengths=lengths_matrix,
                    cv=projection_cv,
                    dt=projection_dt
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

    def test_montbrio_model(self):
        """Test simulation with Montbrio-Pazo-Roxin model"""
        # Create subnetworks with different models
        mpkwargs = {'variables_of_interest': MontbrioPazoRoxin.variables_of_interest.default[:2]}

        cortex = Subnetwork(
            name='cortex',
            model=MontbrioPazoRoxin(**mpkwargs),
            scheme=HeunDeterministic(dt=0.1),
            nnodes=76
        ).configure()  # Configure the model

        thalamus = Subnetwork(
            name='thalamus',
            model=ReducedSetFitzHughNagumo(variables_of_interest=ReducedSetFitzHughNagumo.variables_of_interest.default[:2]),
            scheme=HeunDeterministic(dt=0.1),
            nnodes=76
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
                    source=cortex, target=thalamus,
                    source_cvar=np.r_[0], target_cvar=np.r_[1],
                    weights=weights_matrix,
                    lengths=lengths_matrix,
                    cv=projection_cv,
                    dt=projection_dt
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
