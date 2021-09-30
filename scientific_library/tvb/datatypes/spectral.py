# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""

The Spectral datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""
import numpy

from tvb.basic.neotraits.api import HasTraits, Attr, NArray, Int, Float, EnumAttr, TVBEnum
from tvb.datatypes import time_series


class WindowingFunctionsEnum(TVBEnum):
    HAMMING = "hamming"
    BARTLETT = "bartlett"
    BLACKMAN = "blackman"
    HANNING = "hanning"


class FourierSpectrum(HasTraits):
    """
    Result of a Fourier  Analysis.
    """
    # Overwrite attribute from superclass
    array_data = NArray(dtype=numpy.complex128)

    source = Attr(
        field_type=time_series.TimeSeries,
        label="Source time-series",
        doc="Links to the time-series on which the FFT is applied.")

    segment_length = Float(
        label="Segment length",
        doc="""The timeseries was segmented into equally sized blocks
            (overlapping if necessary), prior to the application of the FFT.
            The segement length determines the frequency resolution of the
            resulting spectra.""")

    windowing_function = EnumAttr(
        default=WindowingFunctionsEnum.HAMMING,
        required=False,
        label="Windowing function",
        doc="""The windowing function applied to each time segment prior to
            application of the FFT.""")

    amplitude = NArray(label="Amplitude")

    phase = NArray(label="Phase")

    power = NArray(label="Power")

    average_power = NArray(label="Average Power")

    normalised_average_power = NArray(label="Normalised Power", required=False)

    _frequency = None
    _freq_step = None
    _max_freq = None

    def configure(self):
        """ compute dependent fields like amplitude """
        self.compute_amplitude()
        self.compute_phase()
        self.compute_average_power()
        self.compute_normalised_average_power()

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        return {
            "Spectral type": self.__class__.__name__,
            "Source": self.source.title,
            "Segment length": self.segment_length,
            "Windowing function": self.windowing_function,
            "Frequency step": self.freq_step,
            "Maximum frequency": self.max_freq
        }

    @property
    def freq_step(self):
        """ Frequency step size of the complex Fourier spectrum."""
        if self._freq_step is None:
            self._freq_step = 1.0 / self.segment_length
            msg = "%s: Frequency step size is %s"
            self.log.debug(msg % (str(self), str(self._freq_step)))
        return self._freq_step

    @property
    def max_freq(self):
        """ Amplitude of the complex Fourier spectrum."""
        if self._max_freq is None:
            self._max_freq = 0.5 / self.source.sample_period
            msg = "%s: Max frequency is %s"
            self.log.debug(msg % (str(self), str(self._max_freq)))
        return self._max_freq

    @property
    def frequency(self):
        """ Frequencies represented the complex Fourier spectrum."""
        if self._frequency is None:
            self._frequency = numpy.arange(self.freq_step,
                                           self.max_freq + self.freq_step,
                                           self.freq_step)
        return self._frequency

    def compute_amplitude(self):
        """ Amplitude of the complex Fourier spectrum."""
        self.amplitude = numpy.abs(self.array_data)

    def compute_phase(self):
        """ Phase of the Fourier spectrum."""
        self.phase = numpy.angle(self.array_data)

    def compute_power(self):
        """ Power of the complex Fourier spectrum."""
        self.power = numpy.abs(self.array_data) ** 2

    def compute_average_power(self):
        """ Average-power of the complex Fourier spectrum."""
        self.average_power = numpy.mean(numpy.abs(self.array_data) ** 2, axis=-1)

    def compute_normalised_average_power(self):
        """ Normalised-average-power of the complex Fourier spectrum."""
        self.normalised_average_power = (self.average_power /
                                         numpy.sum(self.average_power, axis=0))


class WaveletCoefficients(HasTraits):
    """
    This class bundles all the elements of a Wavelet Analysis into a single
    object, including the input TimeSeries datatype and the output results as
    arrays (FloatArray)
    """
    # Overwrite attribute from superclass
    array_data = NArray(dtype=numpy.complex128)

    source = Attr(field_type=time_series.TimeSeries, label="Source time-series")

    mother = Attr(
        field_type=str,
        label="Mother wavelet",
        default="morlet",
        doc="""A string specifying the type of mother wavelet to use,
            default is 'morlet'.""")  # default to 'morlet'

    sample_period = Float(label="Sample period")
    # sample_rate = basic.Integer(label = "")  inversely related

    frequencies = NArray(
        label="Frequencies",
        doc="A vector that maps scales to frequencies.")

    normalisation = Attr(field_type=str, label="Normalisation type")
    # 'unit energy' | 'gabor'

    q_ratio = Float(label="Q-ratio", default=5.0)

    amplitude = NArray(label="Amplitude")

    phase = NArray(label="Phase")

    power = NArray(label="Power")

    _frequency = None
    _time = None

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        return {
            "Spectral type": self.__class__.__name__,
            "Source": self.source.title,
            "Wavelet type": self.mother,
            "Normalisation": self.normalisation,
            "Q-ratio": self.q_ratio,
            "Sample period": self.sample_period,
            "Number of scales": self.frequencies.shape[0],
            "Minimum frequency": self.frequencies[0],
            "Maximum frequency": self.frequencies[-1]
        }

    @property
    def frequency(self):
        """ Frequencies represented by the wavelet spectrogram."""
        if self._frequency is None:
            self._frequency = numpy.arange(self.frequencies.lo,
                                           self.frequencies.hi,
                                           self.frequencies.step)
        return self._frequency

    def compute_amplitude(self):
        """ Amplitude of the complex Wavelet coefficients."""
        self.amplitude = numpy.abs(self.array_data)

    def compute_phase(self):
        """ Phase of the Wavelet coefficients."""
        self.phase = numpy.angle(self.array_data)

    def compute_power(self):
        """ Power of the complex Wavelet coefficients."""
        self.power = numpy.abs(self.array_data) ** 2


class CoherenceSpectrum(HasTraits):
    """
    Result of a NodeCoherence Analysis.
    """
    # Overwrite attribute from superclass
    array_data = NArray()

    source = Attr(
        field_type=time_series.TimeSeries,
        label="Source time-series",
        doc="""Links to the time-series on which the node_coherence is
            applied.""")

    nfft = Int(
        label="Data-points per block",
        default=256,
        doc="""NOTE: must be a power of 2""")

    frequency = NArray(label="Frequency")

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        return {
            "Spectral type": self.__class__.__name__,
            "Source": self.source.title,
            "Number of frequencies": self.frequency.shape[0],
            "Minimum frequency": self.frequency[0],
            "Maximum frequency": self.frequency[-1],
            "FFT length (time-points)": self.nfft
        }


class ComplexCoherenceSpectrum(HasTraits):
    """
    Result of a NodeComplexCoherence Analysis.
    """

    cross_spectrum = NArray(
        dtype=numpy.complex128,
        label="The cross spectrum",
        doc=""" A complex ndarray that contains the nodes x nodes cross
                spectrum for every frequency frequency and for every segment.""")

    array_data = NArray(
        dtype=numpy.complex128,
        label="Complex Coherence",
        doc="""The complex coherence coefficients calculated from the cross
                spectrum. The imaginary values of this complex ndarray represent the
                imaginary coherence.""")

    source = Attr(
        field_type=time_series.TimeSeries,
        label="Source time-series",
        doc="""Links to the time-series on which the node_coherence is
                applied.""")

    epoch_length = Float(
        label="Epoch length",
        doc="""The timeseries was segmented into equally sized blocks
                (overlapping if necessary), prior to the application of the FFT.
                The segement length determines the frequency resolution of the
                resulting spectra.""")

    segment_length = Float(
        label="Segment length",
        doc="""The timeseries was segmented into equally sized blocks
                (overlapping if necessary), prior to the application of the FFT.
                The segement length determines the frequency resolution of the
                resulting spectra.""")

    windowing_function = Attr(
        field_type=str,
        label="Windowing function",
        doc="""The windowing function applied to each time segment prior to
                application of the FFT.""")

    _frequency = None
    _freq_step = None
    _max_freq = None
    spectrum_types = ["Imaginary", "Real", "Absolute"]

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        return {
            "Spectral type": self.__class__.__name__,
            "Source": self.source.title,
            "Frequency step": self.freq_step,
            "Maximum frequency": self.max_freq,
            "Epoch length": self.epoch_length,
            "Segment length": self.segment_length,
            "Windowing function": self.windowing_function
        }

    @property
    def freq_step(self):
        """ Frequency step size of the Complex Coherence Spectrum."""
        if self._freq_step is None:
            self._freq_step = 1.0 / self.segment_length
            msg = "%s: Frequency step size is %s"
            self.log.debug(msg % (str(self), str(self._freq_step)))
        return self._freq_step

    @property
    def max_freq(self):
        """ Maximum frequency represented in the Complex Coherence Spectrum."""
        if self._max_freq is None:
            self._max_freq = 0.5 / self.source.sample_period
            msg = "%s: Max frequency is %s"
            self.log.debug(msg % (str(self), str(self._max_freq)))
        return self._max_freq

    @property
    def frequency(self):
        """ Frequencies represented in the Complex Coherence Spectrum."""
        if self._frequency is None:
            self._frequency = numpy.arange(self.freq_step,
                                           self.max_freq + self.freq_step,
                                           self.freq_step)
        return self._frequency
