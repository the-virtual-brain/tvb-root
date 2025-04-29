"""
Tests for the Recorder class.
"""

import numpy as np
from tvb.simulator.monitors import (TemporalAverage, Monitor, Raw, SubSample, Bold,
                                   RawVoi, SpatialAverage, GlobalAverage, AfferentCoupling,
                                   AfferentCouplingTemporalAverage, EEG, MEG, iEEG, BoldRegionROI,
                                   Projection) # Added Projection
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
            # Ensure dt is configured if instance is provided externally
            if monitor.dt is None:
                 monitor._config_dt(sim.nets.subnets[0].scheme.dt)
        else:
            monitor = monitor_class(period=period)
            # Configure dt *before* other configurations
            monitor._config_dt(sim.nets.subnets[0].scheme.dt)

        # Manually configure VOIs and Stock based on the first subnetwork
        subnet = sim.nets.subnets[0]
        model = subnet.model
        num_nodes = subnet.nnodes
        num_modes = model.number_of_modes # Assumes model has number_of_modes

        # Configure VOIs
        if monitor.variables_of_interest is None or monitor.variables_of_interest.size == 0:
            if isinstance(monitor, (AfferentCoupling, AfferentCouplingTemporalAverage)):
                 # Use coupling variables for AfferentCoupling monitors
                 monitor.voi = np.r_[:len(model.cvar)]
            else:
                 # Use state variables for others
                 monitor.voi = np.r_[:len(model.variables_of_interest)]
        else:
             monitor.voi = monitor.variables_of_interest

        # Special config for Bold HRF (needs dt, happens before stock)
        # Call compute_hrf *before* _config_stock for Bold monitors
        if isinstance(monitor, (Bold, BoldRegionROI)):
             monitor.compute_hrf() # compute_hrf uses dt (set earlier) and sets _interim_istep

        # Configure Stock (if the monitor requires it, e.g., TemporalAverage, Bold)
        # Now _interim_istep should be set for Bold monitors
        if hasattr(monitor, '_config_stock'):
             num_vars = len(monitor.voi)
             monitor._config_stock(num_vars, num_nodes, num_modes)

        # Minimal setup for SpatialAverage spatial_mean if not configured by config_for_sim
        if isinstance(monitor, SpatialAverage) and not hasattr(monitor, 'spatial_mean'):
             # Create a dummy spatial_mean compatible with the number of nodes
             # This avoids the ValueError, although the averaging isn't meaningful here.
             # Assumes averaging down to a single "region" for simplicity.
             monitor.spatial_mean = np.ones((1, num_nodes)) / num_nodes

        # Minimal setup for Projection monitors (_state, _period_in_steps)
        if isinstance(monitor, Projection):
             # Ensure gain is loaded if needed (already done via projection attr)
             gain = monitor.gain
             # Initialize state array
             monitor._state = np.zeros((gain.shape[0], len(monitor.voi)))
             # Calculate period in steps
             monitor._period_in_steps = int(monitor.period / monitor.dt)
             # Mark gain config as done to prevent issues if config_for_sim were called
             monitor._gain_configuration_done = True
             # Configure EEG specific reference vector if needed
             if isinstance(monitor, EEG):
                 monitor._ref_vec = np.zeros((monitor.sensors.number_of_sensors,))
                 if monitor.reference:
                     if monitor.reference.lower() != 'average':
                         sensor_names = monitor.sensors.labels.tolist()
                         monitor._ref_vec[sensor_names.index(monitor.reference)] = 1.0
                     else:
                         monitor._ref_vec[:] = 1.0 / monitor.sensors.number_of_sensors
                 # Assume all sensors are usable and gain is finite for test
                 monitor._ref_vec_mask = np.ones(monitor.sensors.number_of_sensors, dtype=bool)
                 monitor._ref_vec = monitor._ref_vec[monitor._ref_vec_mask]

             # Configure observation noise (needs dt)
             if monitor.obsnoise is not None:
                 if monitor.obsnoise.ntau > 0.0:
                     # Simplified shape for testing
                     noiseshape = (gain.shape[0], 1)
                     monitor.obsnoise.configure_coloured(dt=monitor.dt, shape=noiseshape)
                 else:
                     monitor.obsnoise.configure_white(dt=monitor.dt)


        # Add monitor to the first subnetwork's recorder list
        # Note: The hybrid Simulator currently doesn't use its own monitors list directly.
        # Monitors are handled within Subnetworks via Recorders.
        # We need a Recorder instance to test.
        recorder = Recorder(monitor=monitor)
        net = sim.nets.subnets[0]
        net.monitors = [recorder] # Replace any existing monitors for this test

        # Configure and run the full simulation
        sim.simulation_length = simulation_length
        sim.configure()
        sim.run()

        # Get recorded data from the recorder instance
        times, samples = recorder.to_arrays()

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
        # Let _test_with_monitor handle the basic configuration
        times, samples = self._test_with_monitor(SpatialAverage, period=1.0)
        # Basic checks after simulation runs
        assert len(times) == 5  # 5s sim / 1s period
        assert np.allclose(np.diff(times), 1.0)
        # Output shape check: (time, vars, averaged_nodes, modes)
        # Since spatial_mean isn't fully configured, check based on voi and modes
        assert samples.shape[0] == 5
        assert samples.shape[1] == 1 # Default VOI is usually 1 var for Generic2dOscillator
        # Spatial dim depends on how spatial_mean is configured - needs fix in helper
        # assert samples.shape[2] == ???
        assert samples.shape[3] == 1 # Generic2dOscillator has 1 mode

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

    def _setup_projection_monitor(self, monitor_class, sensors_class):
        """Helper to set up projection monitors (EEG/MEG/iEEG)."""
        # Create sensors with 2 locations using the correct class
        sensor_kwargs = {
            'locations': np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            'labels': np.array(['Sensor1', 'Sensor2'])
        }
        # Add orientations for MEG sensors
        if sensors_class is SensorsMEG:
            # Dummy orientations (2 sensors x 3D vector)
            sensor_kwargs['orientations'] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        sensors = sensors_class(**sensor_kwargs)
        sensors.configure() # Configure sensors

        # Create a mock projection matrix (2 sensors x 4 nodes)
        # In a real scenario, this would come from ProjectionSurfaceXXX.from_file or similar
        projection_data = np.random.rand(sensors.number_of_sensors, 4)

        # Determine the correct Projection class based on the monitor
        if monitor_class is EEG:
            projection_cls = ProjectionSurfaceEEG
        elif monitor_class is MEG:
            projection_cls = ProjectionSurfaceMEG
        elif monitor_class is iEEG:
            projection_cls = ProjectionSurfaceSEEG
        else:
            raise ValueError(f"Unsupported monitor class for projection: {monitor_class}")

        # Create a mock projection object
        projection = projection_cls(projection_data=projection_data)
        
        # Create and configure monitor
        monitor = monitor_class(period=1.0, sensors=sensors, projection=projection)
        # dt configuration will happen in _test_with_monitor before adding
        # monitor._config_dt(0.1) # Moved to _test_with_monitor

        return monitor

    def test_with_eeg(self):
        """Test recording with EEG monitor."""
        monitor = self._setup_projection_monitor(EEG, SensorsEEG)
        self._test_with_monitor(monitor_instance=monitor)

    def test_with_meg(self):
        """Test recording with MEG monitor."""
        monitor = self._setup_projection_monitor(MEG, SensorsMEG)
        self._test_with_monitor(monitor_instance=monitor)

    def test_with_ieeg(self):
        """Test recording with iEEG monitor."""
        monitor = self._setup_projection_monitor(iEEG, SensorsInternal)
        self._test_with_monitor(monitor_instance=monitor)

    def test_with_bold_region_roi(self):
        """Test recording with BoldRegionROI monitor."""
        # Create monitor
        monitor = BoldRegionROI(period=2.0)
        # Configure dt *before* compute_hrf and adding to simulator
        monitor._config_dt(0.1) # Assuming dt=0.1 from _create_hybrid_simulator
        monitor.compute_hrf()

        # Mock attributes normally set during config_for_sim
        # In a real surface sim, these would be set properly.
        # Here, we need mock values compatible with the test setup (4 nodes).
        mock_region_mapping = np.array([0, 0, 1, 1]) # Map 4 nodes to 2 regions
        monitor.region_mapping = mock_region_mapping
        monitor.no_regions = len(np.unique(mock_region_mapping))
        monitor.voi = slice(None) # Monitor all variables

        times, samples = self._test_with_monitor(monitor_instance=monitor, simulation_length=10.0)

        # Verify number of regions in output samples
        assert samples.shape[2] == monitor.no_regions, \
               f"Sample shape {samples.shape} region dim should match number of ROIs {monitor.no_regions}"
