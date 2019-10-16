from tvb.simulator.noise import Noise, Additive, Multiplicative
from tvb.core.entities.file.simulator.configurations_h5 import SimulatorConfigurationH5
from tvb.core.neotraits.h5 import Scalar, DataSet, Reference


class NoiseH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(NoiseH5, self).__init__(path)
        self.ntau = Scalar(Noise.ntau, self)
        # TODO: serialize random_stream?
        self.noise_seed = Scalar(Noise.noise_seed, self)


class AdditiveH5(NoiseH5):

    def __init__(self, path):
        super(AdditiveH5, self).__init__(path)
        self.nsig = DataSet(Additive.nsig, self)


class MultiplicativeH5(NoiseH5):

    def __init__(self, path):
        super(MultiplicativeH5, self).__init__(path)
        self.nsig = DataSet(Multiplicative.nsig, self)
        self.b = Reference(Multiplicative.b, self)

    def store(self, datatype, scalars_only=False, store_references=False):
        # type: (Multiplicative) -> None
        super(MultiplicativeH5, self).store(datatype, scalars_only, store_references)
        equation_gid = self.store_config_as_reference(datatype.b)
        self.b.store(equation_gid)

    def load_into(self, datatype):
        # type: (Multiplicative) -> None
        super(MultiplicativeH5, self).load_into(datatype)
        datatype.b = self.load_from_reference(self.b.load())
