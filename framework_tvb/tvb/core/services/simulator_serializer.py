import uuid
from tvb.adapters.simulator.simulator_adapter import SimulatorAdapterModel
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.file.simulator import h5_factory
from tvb.core.entities.file.simulator.cortex_h5 import CortexH5
from tvb.core.entities.file.simulator.simulator_h5 import SimulatorH5
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.datatypes.sensors import SensorsEEG, SensorsMEG, SensorsInternal
from tvb.simulator.monitors import Projection, EEG, MEG, iEEG


class SimulatorSerializer(object):

    @staticmethod
    def serialize_simulator(simulator, simulator_gid, simulation_state_gid, storage_path):
        simulator_path = h5.path_for(storage_path, SimulatorH5, simulator_gid)

        with SimulatorH5(simulator_path) as simulator_h5:
            simulator_h5.gid.store(uuid.UUID(simulator_gid))
            simulator_h5.store(simulator)
            simulator_h5.connectivity.store(simulator.connectivity)
            if simulator.stimulus:
                simulator_h5.stimulus.store(simulator.stimulus)
            if simulation_state_gid:
                simulator_h5.simulation_state.store(uuid.UUID(simulation_state_gid))

        return simulator_gid

    @staticmethod
    def deserialize_simulator(simulator_gid, storage_path):
        simulator_in_path = h5.path_for(storage_path, SimulatorH5, simulator_gid)
        simulator_in = SimulatorAdapterModel()

        with SimulatorH5(simulator_in_path) as simulator_in_h5:
            simulator_in_h5.load_into(simulator_in)
            simulator_in.connectivity = simulator_in_h5.connectivity.load()
            simulator_in.stimulus = simulator_in_h5.stimulus.load()
            simulator_in.history_gid = simulator_in_h5.simulation_state.load()

        if isinstance(simulator_in.monitors[0], Projection):
            # TODO: simplify this part
            with SimulatorH5(simulator_in_path) as simulator_in_h5:
                monitor_h5_path = simulator_in_h5.get_reference_path(simulator_in.monitors[0].gid)

            monitor_h5_class = h5_factory.monitor_h5_factory(type(simulator_in.monitors[0]))

            with monitor_h5_class(monitor_h5_path) as monitor_h5:
                sensors_gid = monitor_h5.sensors.load()
                region_mapping_gid = monitor_h5.region_mapping.load()

            sensors_index = ABCAdapter.load_entity_by_gid(sensors_gid.hex)
            sensors = h5.load_from_index(sensors_index)

            if isinstance(simulator_in.monitors[0], EEG):
                sensors = SensorsEEG.build_sensors_subclass(sensors)
            elif isinstance(simulator_in.monitors[0], MEG):
                sensors = SensorsMEG.build_sensors_subclass(sensors)
            elif isinstance(simulator_in.monitors[0], iEEG):
                sensors = SensorsInternal.build_sensors_subclass(sensors)

            simulator_in.monitors[0].sensors = sensors
            region_mapping_index = ABCAdapter.load_entity_by_gid(region_mapping_gid.hex)
            region_mapping = h5.load_from_index(region_mapping_index)
            simulator_in.monitors[0].region_mapping = region_mapping

        if simulator_in.surface:
            cortex_path = h5.path_for(storage_path, CortexH5, simulator_in.surface.gid)
            with CortexH5(cortex_path) as cortex_h5:
                simulator_in.surface.local_connectivity = cortex_h5.local_connectivity.load()
                simulator_in.surface.region_mapping_data = cortex_h5.region_mapping_data.load()
                rm_index = dao.get_datatype_by_gid(simulator_in.surface.region_mapping_data.hex)
                simulator_in.surface.surface_gid = uuid.UUID(rm_index.surface_gid)

        return simulator_in
