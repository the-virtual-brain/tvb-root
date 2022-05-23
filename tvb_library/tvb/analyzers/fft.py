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
Calculate an FFT on a TimeSeries DataType and return a FourierSpectrum DataType.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import numpy
import scipy.signal

from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.info import narray_describe
from tvb.datatypes.spectral import FourierSpectrum

log = get_logger(__name__)


SUPPORTED_WINDOWING_FUNCTIONS = {
    'hamming': numpy.hamming,
    'bartlett': numpy.bartlett,
    'blackman': numpy.blackman,
    'hanning': numpy.hanning
}



"""
A module for calculating the FFT of a TimeSeries object of TVB and returning
a FourierSpectrum object. A segment length and windowing function can be
optionally specified. By default the time series is segmented into 1 second
blocks and no windowing function is applied.
"""


def compute_fast_fourier_transform(time_series, segment_length, window_function, detrend):
    """
    # type: (TimeSeries, float, function, bool) -> FourierSpectrum
    Calculate the FFT of time_series broken into segments of length
    segment_length and filtered by window_function.

    Parameters
    __________

    time_series : TimeSeries
    The TimeSeries to which the FFT is to be applied.

    segment_length : float
    The segment length determines the frequency resolution of the resulting power spectra -- longer
    windows produce finer frequency resolution

    window_function : str
    Windowing functions can be applied before the FFT is performed. Default is None, possibilities are: 'hamming';
    'bartlett';'blackman'; and 'hanning'. See, numpy.<function_name>.

    detrend : bool
    Default is True, False means no detrending is performed on the time series.
    """

    tpts = time_series.data.shape[0]
    time_series_length = tpts * time_series.sample_period

    # Segment time-series, overlapping if necessary
    nseg = int(numpy.ceil(time_series_length / segment_length))
    if nseg > 1:
        seg_tpts = numpy.ceil(segment_length / time_series.sample_period)
        overlap = (seg_tpts * nseg - tpts) / (nseg - 1.0)
        starts = [max(seg * (seg_tpts - overlap), 0) for seg in range(nseg)]
        segments = [time_series.data[int(start):int(start) + int(seg_tpts)]
                    for start in starts]
        segments = [segment[:, :, :, :, numpy.newaxis] for segment in segments]
        ts = numpy.concatenate(segments, axis=4)
    else:
        segment_length = time_series_length
        ts = time_series.data[:, :, :, :, numpy.newaxis]
        seg_tpts = ts.shape[0]

    log.debug("Segment length being used is: %s" % segment_length)

    # Base-line correct the segmented time-series
    if detrend:
        ts = scipy.signal.detrend(ts, axis=0)
        log.debug("time_series " + narray_describe(ts))

    # Apply windowing function
    if window_function is not None:
        wf = SUPPORTED_WINDOWING_FUNCTIONS[window_function.value]
        window_mask = numpy.reshape(wf(int(seg_tpts)),
                                    (int(seg_tpts), 1, 1, 1, 1))
        ts = ts * window_mask

    # Calculate the FFT
    result = numpy.fft.fft(ts, axis=0)
    nfreq = result.shape[0] // 2
    result = result[1:nfreq + 1, :]

    log.debug("result " + narray_describe(result))

    spectra = FourierSpectrum(
        source=time_series,
        segment_length=segment_length,
        array_data=result,
        windowing_function=window_function
    )
    spectra.configure()

    return spectra
