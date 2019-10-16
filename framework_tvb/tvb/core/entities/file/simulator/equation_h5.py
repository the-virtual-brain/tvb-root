from tvb.datatypes.equations import Equation, DiscreteEquation, Gaussian, Linear, DoubleGaussian, Sigmoid, \
    GeneralizedSigmoid, Sinusoid, Cosine, Alpha, PulseTrain, Gamma, DoubleExponential, FirstOrderVolterra, \
    MixtureOfGammas

from tvb.core.neotraits.h5 import H5File, Scalar, Json

class EquationH5(H5File):

    def __init__(self, path):
        super(EquationH5, self).__init__(path)

        self.equation = Scalar(Equation.equation, self)
        self.parameters = Json(Equation.parameters, self)

    def load_into(self, datatype):
        # type: (Equation) -> None
        datatype.gid = self.gid.load()
        datatype.parameters = self.parameters.load()


class DiscreteEquationH5(EquationH5):

    def __init__(self, path):
        super(DiscreteEquationH5, self).__init__(path)

        self.equation = Scalar(DiscreteEquation.equation, self)


class LinearH5(EquationH5):

    def __init__(self, path):
        super(LinearH5, self).__init__(path)

        self.equation = Scalar(Linear.equation, self)
        self.parameters = Json(Linear.parameters, self)


class GaussianH5(EquationH5):

    def __init__(self, path):
        super(GaussianH5, self).__init__(path)

        self.equation = Scalar(Gaussian.equation, self)
        self.parameters = Json(Gaussian.parameters, self)


class DoubleGaussianH5(EquationH5):

    def __init__(self, path):
        super(DoubleGaussianH5, self).__init__(path)

        self.equation = Scalar(DoubleGaussian.equation, self)
        self.parameters = Json(DoubleGaussian.parameters, self)


class SigmoidH5(EquationH5):

    def __init__(self, path):
        super(SigmoidH5, self).__init__(path)

        self.equation = Scalar(Sigmoid.equation, self)
        self.parameters = Json(Sigmoid.parameters, self)


class GeneralizedSigmoidH5(EquationH5):

    def __init__(self, path):
        super(GeneralizedSigmoidH5, self).__init__(path)

        self.equation = Scalar(GeneralizedSigmoid.equation, self)
        self.parameters = Json(GeneralizedSigmoid.parameters, self)


class SinusoidH5(EquationH5):

    def __init__(self, path):
        super(SinusoidH5, self).__init__(path)

        self.equation = Scalar(Sinusoid.equation, self)
        self.parameters = Json(Sinusoid.parameters, self)


class CosineH5(EquationH5):

    def __init__(self, path):
        super(CosineH5, self).__init__(path)

        self.equation = Scalar(Cosine.equation, self)
        self.parameters = Json(Cosine.parameters, self)


class AlphaH5(EquationH5):

    def __init__(self, path):
        super(AlphaH5, self).__init__(path)

        self.equation = Scalar(Alpha.equation, self)
        self.parameters = Json(Alpha.parameters, self)


class PulseTrainH5(EquationH5):

    def __init__(self, path):
        super(PulseTrainH5, self).__init__(path)

        self.equation = Scalar(PulseTrain.equation, self)
        self.parameters = Json(PulseTrain.parameters, self)


class GammaH5(EquationH5):

    def __init__(self, path):
        super(GammaH5, self).__init__(path)

        self.equation = Scalar(Gamma.equation, self)
        self.parameters = Json(Gamma.parameters, self)


class DoubleExponentialH5(EquationH5):

    def __init__(self, path):
        super(DoubleExponentialH5, self).__init__(path)

        self.equation = Scalar(DoubleExponential.equation, self)
        self.parameters = Json(DoubleExponential.parameters, self)


class FirstOrderVolterraH5(EquationH5):

    def __init__(self, path):
        super(FirstOrderVolterraH5, self).__init__(path)

        self.equation = Scalar(FirstOrderVolterra.equation, self)
        self.parameters = Json(FirstOrderVolterra.parameters, self)


class MixtureOfGammasH5(EquationH5):

    def __init__(self, path):
        super(MixtureOfGammasH5, self).__init__(path)

        self.equation = Scalar(MixtureOfGammas.equation, self)
        self.parameters = Json(MixtureOfGammas.parameters, self)
