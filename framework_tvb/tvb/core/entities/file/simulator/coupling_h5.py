from tvb.simulator.coupling import Linear, Scaling, HyperbolicTangent, Sigmoidal, SigmoidalJansenRit, PreSigmoidal, \
    Difference, Kuramoto

from tvb.core.entities.file.simulator.configurations_h5 import SimulatorConfigurationH5
from tvb.core.neotraits.h5 import DataSet, Scalar


class LinearH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(LinearH5, self).__init__(path)
        self.a = DataSet(Linear.a, self)
        self.b = DataSet(Linear.b, self)


class ScalingH5(SimulatorConfigurationH5):
    def __init__(self, path):
        super(ScalingH5, self).__init__(path)
        self.a = DataSet(Scaling.a, self)


class HyperbolicTangentH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(HyperbolicTangentH5, self).__init__(path)
        self.a = DataSet(HyperbolicTangent.a, self)
        self.b = DataSet(HyperbolicTangent.b, self)
        self.midpoint = DataSet(HyperbolicTangent.midpoint, self)
        self.sigma = DataSet(HyperbolicTangent.sigma, self)


class SigmoidalH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(SigmoidalH5, self).__init__(path)
        self.cmin = DataSet(Sigmoidal.cmin, self)
        self.cmax = DataSet(Sigmoidal.cmax, self)
        self.midpoint = DataSet(Sigmoidal.midpoint, self)
        self.a = DataSet(Sigmoidal.a, self)
        self.sigma = DataSet(Sigmoidal.sigma, self)


class SigmoidalJansenRitH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(SigmoidalJansenRitH5, self).__init__(path)
        self.cmin = DataSet(SigmoidalJansenRit.cmin, self)
        self.cmax = DataSet(SigmoidalJansenRit.cmax, self)
        self.midpoint = DataSet(SigmoidalJansenRit.midpoint, self)
        self.r = DataSet(SigmoidalJansenRit.r, self)
        self.a = DataSet(SigmoidalJansenRit.a, self)


class PreSigmoidalH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(PreSigmoidalH5, self).__init__(path)
        self.H = DataSet(PreSigmoidal.H, self)
        self.Q = DataSet(PreSigmoidal.Q, self)
        self.G = DataSet(PreSigmoidal.G, self)
        self.P = DataSet(PreSigmoidal.P, self)
        self.theta = DataSet(PreSigmoidal.theta, self)
        self.dynamic = Scalar(PreSigmoidal.dynamic, self)
        self.globalT = Scalar(PreSigmoidal.globalT, self)


class DifferenceH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(DifferenceH5, self).__init__(path)
        self.a = DataSet(Difference.a, self)


class KuramotoH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(KuramotoH5, self).__init__(path)
        self.a = DataSet(Kuramoto.a, self)
