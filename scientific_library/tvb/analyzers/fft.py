# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
from scipy import signal as sp_signal
from tvb.basic.logger.builder import get_logger
import tvb.datatypes.time_series as time_series
import tvb.datatypes.spectral as spectral
import tvb.basic.traits.core as core
import tvb.basic.traits.types_basic as basic
import tvb.basic.traits.util as util


LOG = get_logger(__name__)
SUPPORTED_WINDOWING_FUNCTIONS = ("hamming", "bartlett", "blackman", "hanning")




class FFT(core.Type):
    """
    A class for calculating the FFT of a TimeSeries object of TVB and returning
    a FourierSpectrum object. A segment length and windowing function can be
    optionally specified. By default the time series is segmented into 1 second
    blocks and no windowing function is applied.
    """

    time_series = time_series.TimeSeries(
        label="Time Series",
        required=True,
        doc="""The TimeSeries to which the FFT is to be applied.""",
        order=1)

    segment_length = basic.Float(
        label="Segment(window) length (ms)",
        default=1000.0,
        required=False,
        doc="""The TimeSeries can be segmented into equally sized blocks
            (overlapping if necessary). The segment length determines the
            frequency resolution of the resulting power spectra -- longer
            windows produce finer frequency resolution.""",
        order=2)

    window_function = basic.Enumerate(
        label="Windowing function",
        options=SUPPORTED_WINDOWING_FUNCTIONS,
        default=None,
        required=False,
        select_multiple=False,
        doc="""Windowing functions can be applied before the FFT is performed.
             Default is None, possibilities are: 'hamming'; 'bartlett';
            'blackman'; and 'hanning'. See, numpy.<function_name>.""",
        order=3)
    
    
    def evaluate(self):
        """
        Calculate the FFT of time_series broken into segments of length
        segment_length and filtered by window_function.
        """
        cls_attr_name = self.__class__.__name__ + ".time_series"
        self.time_series.trait["data"].log_debug(owner=cls_attr_name)
        
        tpts = self.time_series.data.shape[0]
        time_series_length = tpts * self.time_series.sample_period
        
        #Segment time-series, overlapping if necessary
        nseg = int(numpy.ceil(time_series_length / self.segment_length))
        if nseg > 1:
            seg_tpts = numpy.ceil(self.segment_length / self.time_series.sample_period)
            overlap = (seg_tpts * nseg - tpts) / (nseg - 1.0)
            starts = [max(seg * (seg_tpts - overlap), 0) for seg in range(nseg)]
            segments = [self.time_series.data[start:start + seg_tpts]
                        for start in starts]
            segments = [segment[:, :, :, numpy.newaxis] for segment in segments]
            time_series = numpy.concatenate(segments, axis=4)
        else:
            self.segment_length = time_series_length
            time_series = self.time_series.data[:, :, :, numpy.newaxis]
            seg_tpts = time_series.shape[0]
        
        LOG.debug("Segment length being used is: %s" % self.segment_length)
        
        #Base-line correct the segmented time-series  
        time_series = sp_signal.detrend(time_series, axis=0)
        util.log_debug_array(LOG, time_series, "time_series")
        
        #Apply windowing function
        if self.window_function is not None and self.window_function != [None]:
            if self.window_function not in SUPPORTED_WINDOWING_FUNCTIONS:
                LOG.error("Windowing function is: %s" % self.window_function)
                LOG.error("Must be in: %s" % str(SUPPORTED_WINDOWING_FUNCTIONS))
            else:
                window_function = eval("".join(("numpy.", self.window_function[0])))
                window_mask = numpy.reshape(window_function(seg_tpts),
                                            (seg_tpts, 1, 1, 1, 1))
                time_series = time_series * window_mask

        #Calculate the FFT
        result = numpy.fft.fft(time_series, axis=0)
        nfreq = result.shape[0] / 2
        result = result[1:nfreq + 1, :]
        util.log_debug_array(LOG, result, "result")

        spectra = spectral.FourierSpectrum(source=self.time_series,
                                           segment_length=self.segment_length,
                                           array_data=result,
                                           use_storage=False)

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
        extend_size = result_size           # Main array
        extend_size += 0.5 * result_size    # Amplitude
        extend_size += 0.5 * result_size    # Phase
        extend_size += 0.5 * result_size    # Power
        extend_size += 0.5 * result_size / result_shape[4]  # Average power
        extend_size += 0.5 * result_size / result_shape[4]  # Normalised Average power
        extend_size += result_shape[0] * 8.0    # Frequency
        return extend_size


