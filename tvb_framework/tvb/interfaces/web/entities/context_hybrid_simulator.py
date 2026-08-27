from tvb.core.entities.file.simulator.hybrid_view_model import HybridSimulatorAdapterModel
from tvb.interfaces.web.controllers import common


class HybridSimulatorContext:
    KEY_CONFIGURATION = "hybrid_simulator_configuration"
    KEY_STEP = "hybrid_simulator_step"

    @property
    def configuration(self):
        return common.get_from_session(self.KEY_CONFIGURATION)

    @property
    def step(self):
        return common.get_from_session(self.KEY_STEP) or 1

    def initialize(self):
        if self.configuration is None:
            common.add2session(self.KEY_CONFIGURATION, HybridSimulatorAdapterModel())
        return self.configuration

    def store(self, configuration, step=None):
        common.add2session(self.KEY_CONFIGURATION, configuration)
        if step is not None:
            common.add2session(self.KEY_STEP, step)

    def reset(self):
        common.remove_from_session(self.KEY_CONFIGURATION)
        common.remove_from_session(self.KEY_STEP)
