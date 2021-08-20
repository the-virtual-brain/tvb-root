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
Calculate a wavelet transform on a TimeSeries datatype and return a
WaveletSpectrum datatype.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Andreas Spiegler <anspiegler@googlemail.com>
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import numpy
import scipy.signal as signal
import tvb.datatypes.spectral as spectral
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import HasTraits, Attr, Range, Float, narray_describe
from tvb.simulator.backend.ref import ReferenceBackend

SUPPORTED_WAVELET_FUNCTIONS = ("morlet",)

log = get_logger(__name__)

"""
A module for calculating the wavelet transform of a TimeSeries object of TVB
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


def compute_continuous_wavelet_transform(time_series, frequencies, sample_period, q_ratio, normalisation, mother):
    """
    # type: (TimeSeries, Range, float, float, str, str)  -> WaveletCoefficients
    Calculate the continuous wavelet transform of time_series.

    Parameters
    __________

    time_series : TimeSeries
    The timeseries to which the wavelet is to be applied.

    frequencies : Range
    The frequency resolution and range returned. Requested frequencies
    are converted internally into appropriate scales.

    sample_period : float
    The sampling period of the computed wavelet spectrum.

    q_ratio : float
    NFC. Must be greater than 5. Ratios of the center frequencies to bandwidths.

    normalisation : str
    The type of normalisation for the resulting wavet spectrum. Default is 'energy', options are: 'energy'; 'gabor'.

    mother : str
    The mother wavelet function used in the transform.
    """
    ts_shape = time_series.data.shape

    if frequencies.step == 0:
        log.warning("Frequency step can't be 0! Trying default step, 2e-3.")
        frequencies.step = 0.002

    freqs = numpy.arange(frequencies.lo, frequencies.hi,
                         frequencies.step)

    if (freqs.size == 0) or any(freqs <= 0.0):
        # TODO: Maybe should limit number of freqs... ~100 is probably a reasonable upper bound.
        log.warning("Invalid frequency range! Falling back to default.")
        log.debug("freqs")
        log.debug(narray_describe(freqs))
        frequencies = Range(lo=0.008, hi=0.060, step=0.002)
        freqs = numpy.arange(frequencies.lo, frequencies.hi,
                             frequencies.step)

    log.debug("freqs")
    log.debug(narray_describe(freqs))

    sample_rate = time_series.sample_rate

    # Duke: code below is as given by Andreas Spiegler, I've just wrapped
    # some of the original argument names
    nf = len(freqs)
    temporal_step = max((1, ReferenceBackend.iround(sample_period / time_series.sample_period)))
    nt = int(numpy.ceil(ts_shape[0] / temporal_step))

    if not isinstance(q_ratio, numpy.ndarray):
        new_q_ratio = q_ratio * numpy.ones((1, nf))

    if numpy.nanmin(new_q_ratio) < 5:
        msg = "q_ratio must be not lower than 5 !"
        log.error(msg)
        raise Exception(msg)

    if numpy.nanmax(freqs) > sample_rate / 2.0:
        msg = "Sampling rate is too low for the requested frequency range !"
        log.error(msg)
        raise Exception(msg)

    # TODO: This isn't used, but min frequency seems like it should be important... Check with A.S.
    #  fmin = 3.0 * numpy.nanmin(q_ratio) * sample_rate / numpy.pi / nt
    sigma_f = freqs / new_q_ratio
    sigma_t = 1.0 / (2.0 * numpy.pi * sigma_f)

    if normalisation == 'energy':
        Amp = 1.0 / numpy.sqrt(sample_rate * numpy.sqrt(numpy.pi) * sigma_t)
    elif normalisation == 'gabor':
        Amp = numpy.sqrt(2.0 / numpy.pi) / sample_rate / sigma_t

    coef_shape = (nf, nt, ts_shape[1], ts_shape[2], ts_shape[3])

    coef = numpy.zeros(coef_shape, dtype=numpy.complex128)
    log.debug("coef")
    log.debug(narray_describe(coef))

    scales = numpy.arange(0, nf, 1)
    for i in scales:
        f0 = freqs[i]
        SDt = sigma_t[(0, i)]
        A = Amp[(0, i)]
        x = numpy.arange(0, 4.0 * SDt * sample_rate, 1) / sample_rate
        wvlt = A * numpy.exp(-x ** 2 / (2.0 * SDt ** 2)) * numpy.exp(2j * numpy.pi * f0 * x)
        wvlt = numpy.hstack((numpy.conjugate(wvlt[-1:0:-1]), wvlt))
        # util.self.log_debug_array(self.log, wvlt, "wvlt")

        for var in range(ts_shape[1]):
            for node in range(ts_shape[2]):
                for mode in range(ts_shape[3]):
                    data = time_series.data[:, var, node, mode]
                    wt = signal.convolve(data, wvlt, 'same')
                    # util.self.log_debug_array(self.log, wt, "wt")
                    res = wt[0::temporal_step]
                    # NOTE: this is a horrible horrible quick hack (alas, a solution) to avoid broadcasting errors
                    # when using dt and sample periods which are not powers of 2.
                    coef[i, :, var, node, mode] = res if len(res) == nt else res[:coef.shape[1]]

    log.debug("coef")
    log.debug(narray_describe(coef))

    spectra = spectral.WaveletCoefficients(
        source=time_series,
        mother=mother,
        sample_period=sample_period,
        frequencies=frequencies.to_array(),
        normalisation=normalisation,
        q_ratio=q_ratio,
        array_data=coef)

    return spectra
