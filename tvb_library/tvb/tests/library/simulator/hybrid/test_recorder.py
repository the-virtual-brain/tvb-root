"""
Tests for the Recorder class.
"""

import numpy as np
from tvb.simulator.monitors import (TemporalAverage, Monitor, Raw, SubSample, Bold,
                                   RawVoi, SpatialAverage, GlobalAverage, AfferentCoupling,
                                   AfferentCouplingTemporalAverage, EEG, MEG, iEEG, BoldRegionROI)
from tvb.simulator.hybrid import Recorder, Subnetwork, Simulator, NetworkSet
from tvb.simulator.models import JansenRit, Generic2dOscillator
from tvb.simulator.integrators import HeunDeterministic
from tvb.datatypes.sensors import SensorsEEG, SensorsMEG, SensorsInternal, Sensors
from tvb.datatypes.projections import ProjectionSurfaceEEG, ProjectionSurfaceMEG, ProjectionSurfaceSEEG
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.surfaces import CorticalSurface
from tvb.datatypes.region_mapping import RegionMapping
from .test_base import BaseHybridTest


class MockMonitor(Monitor):
    """Mock monitor for testing recorder."""
    def __init__(self, period=1.0, sample_shape=(4, 10, 1)):
        super().__init__(period=period)
        self._sample_shape = sample_shape
        self._step_count = 0
        
    def sample(self, step, state):
        """Required abstract method implementation."""
        return self.record(step, state)
        
    def record(self, step, state):
        """Return mock data every other step."""
        if step is not None and step % 2 == 0:
            t = step * 0.1  # arbitrary time step
            y = np.ones(self._sample_shape) * self._step_count
            self._step_count += 1
            return t, y
        return None


class TestRecorder(BaseHybridTest):
    """Tests for the Recorder class."""
    
    def test_lazy_allocation(self):
        """Test that arrays are allocated only after first sample."""
        monitor = MockMonitor()
        recorder = Recorder(monitor=monitor)
        
        # Configure but don't allocate yet
        recorder.configure(simulation_length=10.0)
        assert recorder.num_samples == 10
        assert recorder.samples is None
        assert recorder.times is None
        
        # First sample should trigger allocation
        recorder.record(0, None)
        assert recorder.samples is not None
        assert recorder.times is not None
        assert recorder.shape == (10, 4, 10, 1)
        
    def test_recording(self):
        """Test recording of samples."""
        monitor = MockMonitor(sample_shape=(2, 5, 1))
        recorder = Recorder(monitor=monitor)
        recorder.configure(simulation_length=4.0)
        
        # Record 8 steps (should get 4 samples due to mock monitor)
        for step in range(8):
            recorder.record(step, None)
            
        # Check recorded data
        times, samples = recorder.to_arrays()
        assert len(times) == 4
        assert samples.shape == (4, 2, 5, 1)
        
        # Verify sample values (mock monitor increments counter each sample)
        for i in range(4):
            assert np.all(samples[i, 0, 0, 0] == i)
            
    def test_temporal_average_monitor(self):
        """Test recorder with actual TemporalAverage monitor."""
        tavg = TemporalAverage(period=1.0)
        recorder = Recorder(monitor=tavg)
        recorder.configure(simulation_length=10.0)
        
        # Create some test data
        state = np.random.random((4, 10, 1))  # (vars, nodes, modes)
        
        # Configure the monitor properly with the state shape
        tavg.configure()  # This sets up basic config
        tavg._config_dt(0.1)
        tavg._config_stock(4, 10, 1)  # nvars, nnodes, nmodes
        tavg.voi = slice(None)  # all vars
        
        # Record a few samples
        for step in range(20):
            recorder.record(step, state)
            
        # Verify results
        times, samples = recorder.to_arrays()
        assert len(times) == 10  # Should get 10 samples
        assert samples.shape == (10, 4, 10, 1)
        
    def test_shape_property(self):
        """Test the shape property."""
        monitor = MockMonitor()
        recorder = Recorder(monitor=monitor)
        
        # Before allocation
        assert recorder.shape is None
        
        # After configuration but before first sample
        recorder.configure(simulation_length=5.0)
        assert recorder.shape is None
        
        # After first sample
        recorder.record(0, None)
        assert recorder.shape == (5, 4, 10, 1)

    def _create_hybrid_simulator(self):
        """Create a basic hybrid simulator for testing."""
        # Create a simple model with 4 nodes
        model = Generic2dOscillator()
        scheme = HeunDeterministic(dt=0.1)  # Add integrator scheme
        
        # Create subnetwork
        subnet = Subnetwork(
            name='subnet1',
            model=model,
            scheme=scheme,
            nnodes=4
        ).configure()

        # Create network set
        nets = NetworkSet(
            subnets=[subnet],
            projections=[]  # No projections needed for this test
        )

        # Create simulator
        sim = Simulator(nets=nets)
        sim.configure()
        return sim

    def _test_with_monitor(self, monitor_class=None, monitor_instance=None, period=1.0, simulation_length=5.0, expected_samples=None):
        """Test recorder with a given monitor."""
        sim = self._create_hybrid_simulator()
        
        # Use provided monitor instance or create one from class
        if monitor_instance is not None:
            monitor = monitor_instance
        else:
            monitor = monitor_class(period=period)
            monitor._config_dt(sim.nets.subnets[0].scheme.dt)  # Use dt from first subnet's scheme
        
        sim.monitors.append(monitor)
        recorder = Recorder(sim)
        
        # Run simulation
        for _ in range(int(simulation_length / sim.nets.subnets[0].scheme.dt)):  # Use dt from first subnet's scheme
            sim.run_step()
        
        # Get recorded data
        times = recorder.get_times()
        samples = recorder.get_samples()
        
        # Verify number of samples if expected_samples is provided
        if expected_samples is not None:
            assert len(times) == expected_samples, f"Expected {expected_samples} samples, got {len(times)}"
        
        return times, samples

    def test_with_temporal_average(self):
        """Test recorder with TemporalAverage monitor."""
        self._test_with_monitor(TemporalAverage, period=1.0)

    def test_with_raw(self):
        """Test recorder with Raw monitor."""
        # Raw monitor records at every time step
        dt = 0.1  # simulation dt
        self._test_with_monitor(Raw, period=dt, expected_samples=50)  # 5.0/0.1 = 50 samples

    def test_with_subsample(self):
        """Test recorder with SubSample monitor."""
        self._test_with_monitor(SubSample, period=0.5)  # Should give 10 samples for 5.0s simulation

    def test_with_bold(self):
        """Test recorder with Bold monitor."""
        # Bold typically has longer period (2000ms = 2s default), but we use 2ms for faster testing
        times, samples = self._test_with_monitor(Bold, period=2.0, simulation_length=10.0)
        # Additional Bold-specific checks
        assert len(times) == 5  # 10s / 2ms = 5 samples
        # Check time points are at expected intervals
        assert np.allclose(np.diff(times), 2.0)  # 2ms intervals

    def test_with_hybrid_simulator(self):
        """Test recorder in context of full hybrid simulator setup."""
        # Create two subnetworks
        subnet1 = Subnetwork(
            name='subnet1',
            model=JansenRit(),
            scheme=HeunDeterministic(dt=0.1),
            nnodes=5
        ).configure()
        
        subnet2 = Subnetwork(
            name='subnet2',
            model=JansenRit(),
            scheme=HeunDeterministic(dt=0.1),
            nnodes=5
        ).configure()
        
        # Add monitors to both subnets
        tavg1 = TemporalAverage(period=1.0)
        tavg2 = TemporalAverage(period=0.5)  # Different period to test independence
        subnet1.add_monitor(tavg1)
        subnet2.add_monitor(tavg2)
        
        # Create network set
        nets = NetworkSet(
            subnets=[subnet1, subnet2],
            projections=[]  # No projections needed for this test
        )
        
        # Create and run simulator
        sim = Simulator(
            nets=nets,
            simulation_length=10.0
        )
        sim.configure()
        sim.run()
        
        # Verify recorder data from both subnets
        rec1 = subnet1.monitors[0]
        rec2 = subnet2.monitors[0]
        
        # Check subnet1 recorder (period=1.0)
        t1, y1 = rec1.to_arrays()
        assert len(t1) == 10  # 10 samples (period=1.0, time=10.0)
        assert y1.shape == (10, 4, 5, 1)  # JR has 4 state variables
        
        # Check subnet2 recorder (period=0.5)
        t2, y2 = rec2.to_arrays()
        assert len(t2) == 20  # 20 samples (period=0.5, time=10.0)
        assert y2.shape == (20, 4, 5, 1)  # JR has 4 state variables 

    def test_with_rawvoi(self):
        """Test recorder with RawVoi monitor."""
        times, samples = self._test_with_monitor(RawVoi, period=0.1)  # Same period as dt for raw
        assert len(times) == 50  # 5.0/0.1 = 50 samples
        assert np.allclose(np.diff(times), 0.1)  # dt intervals

    def test_with_spatial_average(self):
        """Test recorder with SpatialAverage monitor."""
        mon = SpatialAverage(period=1.0)
        mon._config_dt(0.1)
        # Shape should be (1, nnodes) for dot product with state[voi, :] which is (nnodes,)
        mon.spatial_mean = np.ones((1, 10)) / 10.0  # Simple averaging matrix
        mon.voi = np.array([0])  # Monitor first state variable
        times, samples = self._test_with_monitor(monitor_instance=mon)
        assert len(times) == 5
        assert np.allclose(np.diff(times), 1.0)

    def test_with_global_average(self):
        """Test recorder with GlobalAverage monitor."""
        mon = GlobalAverage(period=1.0)
        mon._config_dt(0.1)
        # Shape should be (1, nnodes) for dot product with state[voi, :] which is (nnodes,)
        mon.spatial_mean = np.ones((1, 10)) / 10.0  # Simple averaging matrix
        mon.voi = np.array([0])  # Monitor first state variable
        times, samples = self._test_with_monitor(monitor_instance=mon)
        assert len(times) == 5
        assert samples.shape[2] == 1  # Global average reduces spatial dimension to 1

    def test_with_afferent_coupling(self):
        """Test recorder with AfferentCoupling monitor."""
        times, samples = self._test_with_monitor(AfferentCoupling, period=0.1)
        assert len(times) == 50  # Raw-like behavior
        assert np.allclose(np.diff(times), 0.1)

    def test_with_afferent_coupling_temporal_average(self):
        """Test recorder with AfferentCouplingTemporalAverage monitor."""
        times, samples = self._test_with_monitor(AfferentCouplingTemporalAverage, period=1.0)
        assert len(times) == 5
        assert np.allclose(np.diff(times), 1.0)

    def _setup_projection_monitor(self, monitor_class):
        """Helper to set up projection monitors (EEG/MEG/iEEG)."""
        # Create sensors with 2 locations
        sensors = Sensors(
            locations=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            labels=np.array(['Sensor1', 'Sensor2'])
        )
        
        # Create random projection matrix (2 sensors x 4 nodes)
        projection = np.random.rand(2, 4)
        
        # Create and configure monitor
        monitor = monitor_class(period=1.0)
        monitor.sensors = sensors
        monitor.projection = projection
        monitor._config_dt(0.1)
        
        return monitor

    def test_with_eeg(self):
        """Test recording with EEG monitor."""
        monitor = self._setup_projection_monitor(EEG)
        self._test_with_monitor(monitor=monitor)

    def test_with_meg(self):
        """Test recording with MEG monitor."""
        monitor = self._setup_projection_monitor(MEG)
        self._test_with_monitor(monitor=monitor)

    def test_with_ieeg(self):
        """Test recording with iEEG monitor."""
        monitor = self._setup_projection_monitor(iEEG)
        self._test_with_monitor(monitor=monitor)

    def test_with_bold_region_roi(self):
        """Test recording with BoldRegionROI monitor."""
        # Create monitor and configure after initialization
        monitor = BoldRegionROI(period=2.0)
        monitor.roi_indices = np.array([0, 1])
        monitor.compute_hrf()
        monitor._config_dt(0.1)
        
        times, samples = self._test_with_monitor(monitor_instance=monitor, simulation_length=10.0)
        
        # Verify shape matches ROI indices
        assert samples.shape[1] == len(monitor.roi_indices), "Sample shape should match number of ROIs" 