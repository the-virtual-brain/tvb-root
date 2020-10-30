# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Calculate an FFT on a TimeSeries DataType and return a FourierSpectrum DataType.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import numpy
import scipy.signal
from tvb.datatypes.time_series import TimeSeries
from tvb.datatypes.spectral import FourierSpectrum
from tvb.basic.neotraits.api import HasTraits, Attr, Float, narray_describe

SUPPORTED_WINDOWING_FUNCTIONS = {
    'hamming': numpy.hamming,
    'bartlett': numpy.bartlett,
    'blackman': numpy.blackman,
    'hanning': numpy.hanning
}


class FFT(HasTraits):
    """
    A class for calculating the FFT of a TimeSeries object of TVB and returning
    a FourierSpectrum object. A segment length and windowing function can be
    optionally specified. By default the time series is segmented into 1 second
    blocks and no windowing function is applied.
    """

    time_series = Attr(
        field_type=TimeSeries,
        label="Time Series",
        doc="""The TimeSeries to which the FFT is to be applied.""")

    segment_length = Float(
        label="Segment(window) length (ms)",
        default=1000.0,
        required=False,
        doc="""The TimeSeries can be segmented into equally sized blocks
            (overlapping if necessary). The segment length determines the
            frequency resolution of the resulting power spectra -- longer
            windows produce finer frequency resolution.""")

    window_function = Attr(
        field_type=str,
        label="Windowing function",
        choices=tuple(SUPPORTED_WINDOWING_FUNCTIONS),
        required=False,
        doc="""Windowing functions can be applied before the FFT is performed.
             Default is None, possibilities are: 'hamming'; 'bartlett';
            'blackman'; and 'hanning'. See, numpy.<function_name>.""")

    detrend = Attr(
        field_type=bool,
        label="Detrending",
        default=True,
        required=False,
        doc="""Detrending is not always appropriate.
            Default is True, False means no detrending is performed on the time series""")

    def evaluate(self):
        """
        Calculate the FFT of time_series broken into segments of length
        segment_length and filtered by window_function.
        """

        tpts = self.time_series.data.shape[0]
        time_series_length = tpts * self.time_series.sample_period

        # Segment time-series, overlapping if necessary
        nseg = int(numpy.ceil(time_series_length / self.segment_length))
        if nseg > 1:
            seg_tpts = numpy.ceil(self.segment_length / self.time_series.sample_period)
            overlap = (seg_tpts * nseg - tpts) / (nseg - 1.0)
            starts = [max(seg * (seg_tpts - overlap), 0) for seg in range(nseg)]
            segments = [self.time_series.data[int(start):int(start) + int(seg_tpts)]
                        for start in starts]
            segments = [segment[:, :, :, :, numpy.newaxis] for segment in segments]
            time_series = numpy.concatenate(segments, axis=4)
        else:
            self.segment_length = time_series_length
            time_series = self.time_series.data[:, :, :, :, numpy.newaxis]
            seg_tpts = time_series.shape[0]

        self.log.debug("Segment length being used is: %s" % self.segment_length)

        # Base-line correct the segmented time-series
        if self.detrend:
            time_series = scipy.signal.detrend(time_series, axis=0)
            self.log.debug("time_series " + narray_describe(time_series))

        # Apply windowing function
        if self.window_function is not None:
            window_function = SUPPORTED_WINDOWING_FUNCTIONS[self.window_function]
            window_mask = numpy.reshape(window_function(int(seg_tpts)),
                                        (int(seg_tpts), 1, 1, 1, 1))
            time_series = time_series * window_mask

        # Calculate the FFT
        result = numpy.fft.fft(time_series, axis=0)
        nfreq = result.shape[0] // 2
        result = result[1:nfreq + 1, :]

        self.log.debug("result " + narray_describe(result))

        spectra = FourierSpectrum(
            source=self.time_series,
            segment_length=self.segment_length,
            array_data=result,
            windowing_function=self.window_function
        )
        spectra.configure()

        return spectra

    def result_shape(self, input_shape, segment_length, sample_period):
        """Returns the shape of the main result (complex array) of the FFT."""
        freq_len = (segment_length / sample_period) / 2.0
        freq_len = int(min((input_shape[0], freq_len)))
        nseg = max((1, int(numpy.ceil(input_shape[0] * sample_period / segment_length))))
        result_shape = (freq_len, input_shape[1], input_shape[2], input_shape[3], nseg)
        return result_shape

    def result_size(self, input_shape, segment_length, sample_period):
        """
        Returns the storage size in Bytes of the main result (complex array) of 
        the FFT.
        """
        result_size = numpy.prod(self.result_shape(input_shape, segment_length,
                                                   sample_period)) * 2.0 * 8.0  # complex*Bytes
        return result_size

    def extended_result_size(self, input_shape, segment_length, sample_period):
        """
        Returns the storage size in Bytes of the extended result of the FFT. 
        That is, it includes storage of the evaluated FourierSpectrum attributes
        such as power, phase, amplitude, etc.
        """
        result_shape = self.result_shape(input_shape, segment_length, sample_period)
        result_size = self.result_size(input_shape, segment_length, sample_period)
        extend_size = result_size  # Main array
        extend_size += 0.5 * result_size  # Amplitude
        extend_size += 0.5 * result_size  # Phase
        extend_size += 0.5 * result_size  # Power
        extend_size += 0.5 * result_size / result_shape[4]  # Average power
        extend_size += 0.5 * result_size / result_shape[4]  # Normalised Average power
        extend_size += result_shape[0] * 8.0  # Frequency
        return extend_size
