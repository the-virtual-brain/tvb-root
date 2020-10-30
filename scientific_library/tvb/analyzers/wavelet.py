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
Calculate a wavelet transform on a TimeSeries datatype and return a
WaveletSpectrum datatype.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Andreas Spiegler <anspiegler@googlemail.com>
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import numpy
import scipy.signal as signal
import tvb.datatypes.time_series as time_series
import tvb.datatypes.spectral as spectral
from tvb.basic.neotraits.api import HasTraits, Attr, Range, Float, narray_describe
from tvb.simulator.common import iround


SUPPORTED_WAVELET_FUNCTIONS = ("morlet",)



class ContinuousWaveletTransform(HasTraits):
    """
    A class for calculating the wavelet transform of a TimeSeries object of TVB
    and returning a WaveletSpectrum object. The sampling period and frequency
    range of the result can be specified. The mother wavelet can also be 
    specified... (So far, only Morlet.)
    
    References:
        .. [TBetal_1996] C. Tallon-Baudry et al, *Stimulus Specificity of 
            Phase-Locked and Non-Phase-Locked 40 Hz Visual Responses in Human.*,
            J Neurosci 16(13):4240-4249, 1996. 
        
        .. [Mallat_1999] S. Mallat, *A wavelet tour of signal processing.*,
            book, Academic Press, 1999.
        
    """
    
    time_series = Attr(
        field_type=time_series.TimeSeries,
        label="Time Series",
        doc="""The timeseries to which the wavelet is to be applied.""")
    
    mother = Attr(
        field_type=str,
        label="Wavelet function",
        default="morlet",
        doc="""The mother wavelet function used in the transform. Default is
            'morlet', possibilities are: 'morlet'...""")
    
    sample_period = Float(
        label="Sample period of result (ms)",
        default=7.8125, #7.8125 => 128 Hz
        doc="""The sampling period of the computed wavelet spectrum. NOTE:
            This should be an integral multiple of the of the sampling period 
            of the source time series, otherwise the actual resulting sample
            period will be the first correct value below that requested.""")

    frequencies = Attr(
        field_type=Range,
        label="Frequency range of result (kHz).",
        default=Range(lo=0.008, hi=0.060, step=0.002),
        doc="""The frequency resolution and range returned. Requested
            frequencies are converted internally into appropriate scales.""")
    
    normalisation = Attr(
        field_type=str,
        label="Normalisation",
        default="energy",
        doc="""The type of normalisation for the resulting wavet spectrum.
            Default is 'energy', options are: 'energy'; 'gabor'.""")
    
    q_ratio = Float(
        label="Q-ratio",
        default=5.0,
        doc="""NFC. Must be greater than 5. Ratios of the center frequencies to bandwidths.""")
    
    
    
    def evaluate(self):
        """
        Calculate the continuous wavelet transform of time_series.
        """
        ts_shape = self.time_series.data.shape
        
        if self.frequencies.step == 0:
            self.log.warning("Frequency step can't be 0! Trying default step, 2e-3.")
            self.frequencies.step = 0.002
        
        freqs = numpy.arange(self.frequencies.lo, self.frequencies.hi,
                             self.frequencies.step)
        
        if (freqs.size == 0) or any(freqs <= 0.0):
            # TODO: Maybe should limit number of freqs... ~100 is probably a reasonable upper bound.
            self.log.warning("Invalid frequency range! Falling back to default.")
            self.log.debug("freqs")
            self.log.debug(narray_describe(freqs))
            self.frequencies = Range(lo=0.008, hi=0.060, step=0.002)
            freqs = numpy.arange(self.frequencies.lo, self.frequencies.hi,
                                 self.frequencies.step)

        self.log.debug("freqs")
        self.log.debug(narray_describe(freqs))

        sample_rate = self.time_series.sample_rate
        
        # Duke: code below is as given by Andreas Spiegler, I've just wrapped 
        # some of the original argument names
        nf = len(freqs)
        temporal_step = max((1, iround(self.sample_period / self.time_series.sample_period)))
        nt = int(numpy.ceil(ts_shape[0] /  temporal_step))
        
        if not isinstance(self.q_ratio, numpy.ndarray):
            q_ratio = self.q_ratio * numpy.ones((1, nf))
        
        if numpy.nanmin(q_ratio) < 5:
            msg = "q_ratio must be not lower than 5 !"
            self.log.error(msg)
            raise Exception(msg)
        
        if numpy.nanmax(freqs) > sample_rate / 2.0:
            msg = "Sampling rate is too low for the requested frequency range !"
            self.log.error(msg)
            raise Exception(msg)
        
        # TODO: This isn't used, but min frequency seems like it should be important... Check with A.S.
        #  fmin = 3.0 * numpy.nanmin(q_ratio) * sample_rate / numpy.pi / nt
        sigma_f = freqs / q_ratio
        sigma_t = 1.0 / (2.0 * numpy.pi * sigma_f)
        
        if self.normalisation == 'energy':
            Amp = 1.0 / numpy.sqrt(sample_rate * numpy.sqrt(numpy.pi) * sigma_t)
        elif self.normalisation == 'gabor': 
            Amp = numpy.sqrt(2.0 / numpy.pi) / sample_rate / sigma_t
        
        coef_shape = (nf, nt, ts_shape[1], ts_shape[2], ts_shape[3])
        
        coef = numpy.zeros(coef_shape, dtype = numpy.complex128)
        self.log.debug("coef")
        self.log.debug(narray_describe(coef))

        scales = numpy.arange(0, nf, 1)
        for i in scales:
            f0 = freqs[i]
            SDt = sigma_t[(0, i)]
            A = Amp[(0, i)]
            x = numpy.arange(0, 4.0 * SDt * sample_rate, 1) / sample_rate
            wvlt = A * numpy.exp(-x**2 / (2.0 * SDt**2) ) * numpy.exp(2j * numpy.pi * f0 * x )
            wvlt = numpy.hstack((numpy.conjugate(wvlt[-1:0:-1]), wvlt))
            #util.self.log_debug_array(self.log, wvlt, "wvlt")
            
            for var in range(ts_shape[1]):
                for node in range(ts_shape[2]):
                    for mode in range(ts_shape[3]):
                        data = self.time_series.data[:, var, node, mode]
                        wt = signal.convolve(data, wvlt, 'same')
                        #util.self.log_debug_array(self.log, wt, "wt")
                        res = wt[0::temporal_step]
                        # NOTE: this is a horrible horrible quick hack (alas, a solution) to avoid broadcasting errors
                        # when using dt and sample periods which are not powers of 2.
                        coef[i, :, var, node, mode] = res if len(res) == nt else res[:coef.shape[1]] 
                        

        self.log.debug("coef")
        self.log.debug(narray_describe(coef))

        spectra = spectral.WaveletCoefficients(
            source=self.time_series,
            mother=self.mother,
            sample_period=self.sample_period,
            frequencies=self.frequencies.to_array(),
            normalisation=self.normalisation,
            q_ratio=self.q_ratio,
            array_data=coef)
        
        return spectra


    def result_shape(self, input_shape, input_sample_period):
        """
        Returns the shape of the main result (complex array) of the continuous
        wavelet transform.
        """
        freq_len = int((self.frequencies.hi - self.frequencies.lo) / self.frequencies.step)
        temporal_step = max((1, self.sample_period / input_sample_period))
        nt = int(round(input_shape[0] /  temporal_step))
        result_shape = (freq_len, nt, ) + input_shape[1:]
        return result_shape
    
    
    def result_size(self, input_shape, input_sample_period):
        """
        Returns the storage size in Bytes of the main result (complex array) of
        the continuous wavelet transform.
        """
        result_size = numpy.prod(self.result_shape(input_shape, input_sample_period)) * 2.0 * 8.0 #complex*Bytes
        return result_size
    
    
    def extended_result_size(self, input_shape, input_sample_period):
        """
        Returns the storage size in Bytes of the extended result of the 
        continuous wavelet transform.  That is, it includes storage of the
        evaluated WaveletCoefficients attributes such as power, phase, 
        amplitude, etc.
        """
        result_shape = self.result_shape(input_shape, input_sample_period)
        result_size = self.result_size(input_shape, input_sample_period)
        extend_size = result_size #Main array
        extend_size = extend_size + 0.5 * result_size #Amplitude
        extend_size = extend_size + 0.5 * result_size #Phase
        extend_size = extend_size + 0.5 * result_size #Power
        extend_size = extend_size + result_shape[0] * 8.0 #Frequency
        return extend_size


