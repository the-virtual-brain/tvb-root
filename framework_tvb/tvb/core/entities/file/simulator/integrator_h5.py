from tvb.simulator.integrators import Integrator, IntegratorStochastic

from tvb.core.entities.file.simulator.configurations_h5 import SimulatorConfigurationH5
from tvb.core.neotraits.h5 import Scalar, Reference


class IntegratorH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(IntegratorH5, self).__init__(path)
        self.dt = Scalar(Integrator.dt, self)
        # TODO: store these?
        # self.clamped_state_variable_indices = DataSet(Integrator.clamped_state_variable_indices, self)
        # self.clamped_state_variable_values = DataSet(Integrator.clamped_state_variable_values, self)


class IntegratorStochasticH5(IntegratorH5):

    def __init__(self, path):
        super(IntegratorStochasticH5, self).__init__(path)
        self.noise = Reference(IntegratorStochastic.noise, self)

    def store(self, datatype, scalars_only=False, store_references=False):
        # type: (IntegratorStochastic) -> None
        super(IntegratorStochasticH5, self).store(datatype, scalars_only, store_references)
        noise_gid = self.store_config_as_reference(datatype.noise)
        self.noise.store(noise_gid)

    def load_into(self, datatype):
        # type: (IntegratorStochastic) -> None
        super(IntegratorStochasticH5, self).load_into(datatype)
        datatype.noise = self.load_from_reference(self.noise.load())
