# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
Calculate the cross spectrum and complex coherence on a TimeSeries datatype and 
return a ComplexCoherence datatype.

.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import numpy
import tvb.datatypes.spectral as spectral
from scipy import signal as sp_signal
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.info import narray_describe

SUPPORTED_WINDOWING_FUNCTIONS = ("hamming", "bartlett", "blackman", "hanning")

log = get_logger(__name__)

# NOTE: Work only with 2D TimeSeries -- otherwise a MemoryError will raise
# My first attempts made use of itertools on the 4D TimeSeries but they were
# fruitless.
# Nested `for` loops seem a 'good' solution instead of creating enormous ndarrays
# to compute the FFT and derived complex spectra at once


"""
A module for calculating the FFT of a TimeSeries and returning
a ComplexCoherenceSpectrum datatype.
   
  
This algorithm is based on the matlab function data2cs_event.m written by Guido Nolte:
    .. [Freyer_2012] Freyer, F.; Reinacher, M.; Nolte, G.; Dinse, H. R. and
        Ritter, P. *Repetitive tactile stimulation changes resting-state
        functional connectivity-implications for treatment of sensorimotor decline*.
        Front Hum Neurosci, Bernstein Focus State Dependencies of Learning and
        Bernstein Center for Computational Neuroscience Berlin, Germany., 2012, 6, 144
    
Input: 
originally the input could be 2D (tpts x nodes/channels), and it was possible
to give a 3D array (e.g., tpspt x nodes/cahnnels x trials) via the segment_length
attribute. 
Current TVB implementation can handle 4D or 2D TimeSeries datatypes. 
Be warned: the 4D TimeSeries will be averaged and squeezed.
    
Output: (main arrays)
- the cross-spectrum
- the complex coherence, from which the imaginary part can be extracted 
        
By default the time series is segmented into 1 second `epoch` blocks and 0.5
second 50% overlapping `segments` to which a Hanning function is applied. 
    
"""


def calculate_complex_cross_coherence(time_series, epoch_length, segment_length, segment_shift, window_function,
                                      average_segments, subtract_epoch_average, zeropad, detrend_ts, max_freq,
                                      npat):
    """
    # type: (TimeSeries, float, float, float, str, bool, bool, int, bool, float, float)  -> ComplexCoherenceSpectrum
    Calculate the FFT, Cross Coherence and Complex Coherence of time_series
    broken into (possibly) epochs and segments of length `epoch_length` and
    `segment_length` respectively, filtered by `window_function`.

    Parameters
    __________

    time_series : TimeSeries
    The timeseries for which the CrossCoherence and ComplexCoherence is to be computed.

    epoch_length : float
    In general for lengthy EEG recordings (~30 min), the timeseries are divided into equally
    sized segments (~ 20-40s). These contain the  event that is to be characterized by means of the
    cross coherence. Additionally each epoch block will be further divided into segments to  which
    the FFT will be applied.

    segment_length : float
    The segment length determines the frequency resolution of the resulting power spectra --
    longer windows produce finer frequency resolution.

    segment_shift : float
    Time length by which neighboring segments are shifted. e.g.
    `segment shift` = `segment_length` / 2 means 50% overlapping segments.

    window_function : str
    Windowing functions can be applied before the FFT is performed.

    average_segments : bool
    Flag. If `True`, compute the mean Cross Spectrum across  segments.

    subtract_epoch_average: bool
    Flag. If `True` and if the number of epochs is > 1, you can optionally subtract the
    mean across epochs before computing the complex coherence.

    zeropad : int
    Adds `n` zeros at the end of each segment and at the end of window_function. It is not yet functional.

    detrend_ts : bool
    Flag. If `True` removes linear trend along the time dimension before applying FFT.

    max_freq : float
    Maximum frequency points (e.g. 32., 64., 128.) represented in the output. Default is segment_length / 2 + 1.

    npat : float
    This attribute appears to be related to an input projection matrix... Which is not yet implemented.
    """
    # self.time_series.trait["data"].log_debug(owner=cls_attr_name)
    tpts = time_series.data.shape[0]
    time_series_length = tpts * time_series.sample_period

    if len(time_series.data.shape) > 2:
        time_series_data = numpy.squeeze((time_series.data.mean(axis=-1)).mean(axis=1))

    # Divide time-series into epochs, no overlapping
    if epoch_length > 0.0:
        nepochs = int(numpy.floor(time_series_length / epoch_length))
        epoch_tpts = int(epoch_length / time_series.sample_period)
        time_series_length = epoch_length
        tpts = epoch_tpts
    else:
        epoch_length = time_series_length
        nepochs = int(numpy.ceil(time_series_length / epoch_length))

    # Segment time-series, overlapping if necessary
    nseg = int(numpy.floor(time_series_length / segment_length))
    if nseg > 1:
        seg_tpts = int(segment_length / time_series.sample_period)
        seg_shift_tpts = int(segment_shift / time_series.sample_period)
        nseg = int(numpy.floor((tpts - seg_tpts) / seg_shift_tpts) + 1)
    else:
        segment_length = time_series_length
        seg_tpts = time_series_data.shape[0]

    # Frequency
    nfreq = int(numpy.min([max_freq, numpy.floor((seg_tpts + zeropad) / 2.0) + 1]))

    resulted_shape, av_result_shape = complex_coherence_result_shape(time_series.data.shape, max_freq, epoch_length,
                                                                     segment_length, segment_shift,
                                                                     time_series.sample_period, zeropad,
                                                                     average_segments)
    cs = numpy.zeros(resulted_shape, dtype=numpy.complex128)
    av = numpy.zeros(av_result_shape, dtype=numpy.complex128)
    coh = numpy.zeros(resulted_shape, dtype=numpy.complex128)

    # Apply windowing function
    if window_function is not None:
        if window_function not in SUPPORTED_WINDOWING_FUNCTIONS:
            log.error("Windowing function is: %s" % window_function)
            log.error("Must be in: %s" % str(SUPPORTED_WINDOWING_FUNCTIONS))

        window_func = eval("".join(("numpy.", window_function)))
        win = window_func(seg_tpts)
        window_mask = (numpy.kron(numpy.ones((time_series_data.shape[1], 1)), win)).T

    nave = 0

    for j in numpy.arange(nepochs):
        data = time_series_data[j * epoch_tpts:(j + 1) * epoch_tpts, :]

        for i in numpy.arange(nseg):  # average over all segments;
            ts = data[i * seg_shift_tpts: i * seg_shift_tpts + seg_tpts, :]

            if detrend_ts:
                ts = sp_signal.detrend(ts, axis=0)

            datalocfft = numpy.fft.fft(ts * window_mask, axis=0)
            datalocfft = numpy.matrix(datalocfft)

            for f in numpy.arange(nfreq):  # for all frequencies
                if npat == 1:
                    if not average_segments:
                        cs[:, :, f, i] += numpy.conjugate(datalocfft[f, :].conj().T * datalocfft[f, :])
                        av[:, f, i] += numpy.conjugate(datalocfft[f, :].conj().T)
                    else:
                        cs[:, :, f] += numpy.conjugate(datalocfft[f, :].conj().T * datalocfft[f, :])
                        av[:, f] += numpy.conjugate(datalocfft[f, :].conj().T)
                else:
                    if not average_segments:
                        cs[:, :, f, j, i] = numpy.conjugate(datalocfft[f, :].conj().T * datalocfft[f, :])
                        av[:, f, j, i] = numpy.conjugate(datalocfft[f, :].conj().T)
                    else:
                        cs[:, :, f, j] += numpy.conjugate(datalocfft[f, :].conj().T * datalocfft[f, :])
                        av[:, f, j] += numpy.conjugate(datalocfft[f, :].conj().T)
            del datalocfft

        nave += 1.0

    # End of FORs
    if not average_segments:
        cs = cs / nave
        av = av / nave
    else:
        nave = nave * nseg
        cs = cs / nave
        av = av / nave

    # Subtract average
    for f in numpy.arange(nfreq):
        if subtract_epoch_average:
            if npat == 1:
                if not average_segments:
                    for i in numpy.arange(nseg):
                        cs[:, :, f, i] = cs[:, :, f, i] - av[:, f, i] * av[:, f, i].conj().T
                else:
                    cs[:, :, f] = cs[:, :, f] - av[:, f] * av[:, f].conj().T
            else:
                if not average_segments:
                    for i in numpy.arange(nseg):
                        for j in numpy.arange(nepochs):
                            cs[:, :, f, j, i] = cs[:, :, f, j, i] - av[:, f, j, i] * av[:, f, j, i].conj().T

                else:
                    for j in numpy.arange(nepochs):
                        cs[:, :, f, j] = cs[:, :, f, j] - av[:, f, j] * av[:, f, j].conj().T

    # Compute Complex Coherence
    ndim = len(cs.shape)
    if ndim == 3:
        for i in numpy.arange(cs.shape[2]):
            temp = numpy.array(cs[:, :, i])
            coh[:, :, i] = cs[:, :, i] / numpy.sqrt(temp.diagonal().conj().T * temp.diagonal())

    elif ndim == 4:
        for i in numpy.arange(cs.shape[2]):
            for j in numpy.arange(cs.shape[3]):
                temp = numpy.array(numpy.squeeze(cs[:, :, i, j]))
                coh[:, :, i, j] = temp / numpy.sqrt(temp.diagonal().conj().T * temp.diagonal().T)

    log.debug("result")
    log.debug(narray_describe(cs))
    spectra = spectral.ComplexCoherenceSpectrum(source=time_series,
                                                array_data=coh,
                                                cross_spectrum=cs,
                                                epoch_length=epoch_length,
                                                segment_length=segment_length,
                                                windowing_function=window_function)
    return spectra


def complex_coherence_result_shape(input_shape, max_freq, epoch_length, segment_length, segment_shift, sample_period,
                                   zeropad,
                                   average_segments):
    """
    Returns the shape of the main result and the average over epochs
    """
    # this is useless here unless the input could actually be a 2D TimeSeries
    nchan = input_shape[2] if len(input_shape) > 2 else input_shape[1]
    seg_tpts = segment_length / sample_period
    seg_shift_tpts = segment_shift / sample_period
    tpts = (epoch_length / sample_period) if epoch_length > 0.0 else input_shape[0]
    nfreq = int(numpy.min([max_freq, numpy.floor((seg_tpts + zeropad) / 2.0) + 1]))
    nseg = int(numpy.floor((tpts - seg_tpts) / seg_shift_tpts) + 1)

    if not average_segments:
        result_shape = (nchan, nchan, nfreq, nseg)
        av_result_shape = (nchan, nfreq, nseg)
    else:
        result_shape = (nchan, nchan, nfreq)
        av_result_shape = (nchan, nfreq)

    return [result_shape, av_result_shape]
