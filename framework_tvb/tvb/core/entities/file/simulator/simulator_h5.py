import uuid
from tvb.basic.neotraits.api import Attr
from tvb.simulator.simulator import Simulator
from tvb.core.entities.file.simulator.configurations_h5 import SimulatorConfigurationH5
from tvb.core.neotraits.h5 import Reference, Scalar, Json, DataSet


class SimulatorH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(SimulatorH5, self).__init__(path)
        self.connectivity = Reference(Simulator.connectivity, self)
        self.conduction_speed = Scalar(Simulator.conduction_speed, self)
        self.coupling = Reference(Simulator.coupling, self)
        self.surface = Reference(Simulator.surface, self)
        self.stimulus = Reference(Simulator.stimulus, self)
        self.model = Reference(Simulator.model, self)
        self.integrator = Reference(Simulator.integrator, self)
        self.initial_conditions = DataSet(Simulator.initial_conditions, self)
        self.monitors = Json(Simulator.monitors, self)
        self.simulation_length = Scalar(Simulator.simulation_length, self)
        self.simulation_state = Reference(Attr(field_type=uuid.UUID), self, name='simulation_state')

    def store(self, datatype, scalars_only=False, store_references=False):
        # type: (Simulator) -> None
        # TODO: handle store conn here
        # self.connectivity.store(conn_gid)
        self.conduction_speed.store(datatype.conduction_speed)
        self.initial_conditions.store(datatype.initial_conditions)
        self.simulation_length.store(datatype.simulation_length)

        integrator_gid = self.store_config_as_reference(datatype.integrator)
        self.integrator.store(integrator_gid)

        coupling_gid = self.store_config_as_reference(datatype.coupling)
        self.coupling.store(coupling_gid)

        model_gid = self.store_config_as_reference(datatype.model)
        self.model.store(model_gid)

        # TODO: handle multiple monitors
        monitor_gid = self.store_config_as_reference(datatype.monitors[0])
        self.monitors.store([monitor_gid.hex])

        if datatype.surface:
            cortex_gid = self.store_config_as_reference(datatype.surface)
            self.surface.store(cortex_gid)

        self.type.store(self.get_full_class_name(type(datatype)))

    def load_into(self, datatype):
        # type: (Simulator) -> None
        datatype.conduction_speed = self.conduction_speed.load()
        datatype.initial_conditions = self.initial_conditions.load()
        datatype.simulation_length = self.simulation_length.load()
        datatype.integrator = self.load_from_reference(self.integrator.load())
        datatype.coupling = self.load_from_reference(self.coupling.load())
        datatype.model = self.load_from_reference(self.model.load())
        # TODO: handle multiple monitors
        datatype.monitors = [self.load_from_reference(self.monitors.load()[0])]
        if self.surface.load():
            datatype.surface = self.load_from_reference(self.surface.load())
