from tvb.core.neotraits.h5 import H5File, DataSet, Scalar, Reference
from tvb.datatypes.spectral import FourierSpectrum, WaveletCoefficients, CoherenceSpectrum, ComplexCoherenceSpectrum


class FourierSpectrumH5(H5File):

    def __init__(self, path):
        super(FourierSpectrumH5, self).__init__(path)
        self.array_data = DataSet(FourierSpectrum.array_data, self, expand_dimension=2)
        self.source = Reference(FourierSpectrum.source, self)
        self.segment_length = Scalar(FourierSpectrum.segment_length, self)
        self.windowing_function = Scalar(FourierSpectrum.windowing_function, self)
        self.amplitude = DataSet(FourierSpectrum.amplitude, self, expand_dimension=2)
        self.phase = DataSet(FourierSpectrum.phase, self, expand_dimension=2)
        self.power = DataSet(FourierSpectrum.power, self, expand_dimension=2)
        self.average_power = DataSet(FourierSpectrum.average_power, self, expand_dimension=2)
        self.normalised_average_power = DataSet(FourierSpectrum.normalised_average_power, self, expand_dimension=2)


class WaveletCoefficientsH5(H5File):

    def __init__(self, path):
        super(WaveletCoefficientsH5, self).__init__(path)
        self.array_data = DataSet(WaveletCoefficients.array_data, self, expand_dimension=2)
        self.source = Reference(WaveletCoefficients.source, self)
        self.mother = Scalar(WaveletCoefficients.mother, self)
        self.sample_period = Scalar(WaveletCoefficients.sample_period, self)
        self.frequencies = DataSet(WaveletCoefficients.frequencies, self)
        self.normalisation = Scalar(WaveletCoefficients.normalisation, self)
        self.q_ratio = Scalar(WaveletCoefficients.q_ratio, self)
        self.amplitude = DataSet(WaveletCoefficients.amplitude, self, expand_dimension=2)
        self.phase = DataSet(WaveletCoefficients.phase, self, expand_dimension=2)
        self.power = DataSet(WaveletCoefficients.power, self, expand_dimension=2)


class CoherenceSpectrumH5(H5File):

    def __init__(self, path):
        super(CoherenceSpectrumH5, self).__init__(path)
        self.array_data = DataSet(CoherenceSpectrum.array_data, self, expand_dimension=3)
        self.source = Reference(CoherenceSpectrum.source, self)
        self.nfft = Scalar(CoherenceSpectrum.nfft, self)
        self.frequency = DataSet(CoherenceSpectrum.frequency, self)


class ComplexCoherenceSpectrumH5(H5File):

    def __init__(self, path):
        super(ComplexCoherenceSpectrumH5, self).__init__(path)
        self.cross_spectrum = DataSet(ComplexCoherenceSpectrum.cross_spectrum, self, expand_dimension=2)
        self.array_data = DataSet(ComplexCoherenceSpectrum.array_data, self, expand_dimension=2)
        self.source = Reference(ComplexCoherenceSpectrum.source, self)
        self.epoch_length = Scalar(ComplexCoherenceSpectrum.epoch_length, self)
        self.segment_length = Scalar(ComplexCoherenceSpectrum.segment_length, self)
        self.windowing_function = Scalar(ComplexCoherenceSpectrum.windowing_function, self)
