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
Compute cross coherence between all nodes in a time series.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import numpy
import matplotlib.mlab as mlab
from matplotlib.pylab import detrend_linear
import tvb.datatypes.spectral as spectral
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import narray_describe

log = get_logger(__name__)


# TODO: Should do this properly, ie not with mlab, returning both coherence and
#      the complex coherence spectra, then supporting magnitude squared
#      coherence, etc in a similar fashion to the FourierSpectrum datatype...


def _hamming(M, sym=True):
    """
    The M-point Hamming window.
    From scipy.signal
    """
    if M < 1:
        return numpy.array([])
    if M == 1:
        return numpy.ones(1, 'd')
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = numpy.arange(0, M)
    w = 0.54 - 0.46 * numpy.cos(2.0 * numpy.pi * n / (M - 1))
    if not sym and not odd:
        w = w[:-1]
    return w


def coherence_mlab(data, sample_rate, nfft=256):
    _, nsvar, nnode, nmode = data.shape
    # (frequency, nodes, nodes, state-variables, modes)
    coh_shape = nfft/2 + 1, nnode, nnode, nsvar, nmode
    coh = numpy.zeros(coh_shape)
    for mode in range(nmode):
        for var in range(nsvar):
            data = data[:, var, :, mode].copy()
            data -= data.mean(axis=0)[numpy.newaxis, :]
            for n1 in range(nnode):
                for n2 in range(nnode):
                    cxy, freq = mlab.cohere(data[:, n1], data[:, n2],
                                            NFFT=nfft,
                                            Fs=sample_rate,
                                            detrend=detrend_linear,
                                            window=mlab.window_none)
                    coh[:, n1, n2, var, mode] = cxy
    return coh, freq


def _coherence(data, sample_rate, nfft=256, imag=False):
    "Vectorized coherence calculation by windowed FFT"
    nt, ns, nn, nm = data.shape
    nwin = nt // nfft
    if nwin < 1:
        raise ValueError(
            "Not enough time points ({0}) to compute an FFT, given a "
            "window size of nfft={1}.".format(nt, nfft))
    # ignore leftover data; need shape (nn, ... , nwin, nfft)
    wins = data[:int(nwin * nfft)]\
        .copy()\
        .transpose((2, 1, 3, 0))\
        .reshape((nn, ns, nm, nwin, nfft))
    wins *= _hamming(nfft)
    F = numpy.fft.fft(wins)
    fs = numpy.fft.fftfreq(nfft, 1e3 / sample_rate)
    # broadcasts to [node_i, node_j, ..., window, time]
    G = F[:, numpy.newaxis] * F.conj()
    if imag:
        G = G.imag
    dG = numpy.array([G[i, i] for i in range(nn)])
    C = (numpy.abs(G)**2 / (dG[:, numpy.newaxis] * dG)).mean(axis=-2)
    mask = fs > 0.0
    # C_ = numpy.abs(C.mean(axis=0).mean(axis=0))
    return numpy.transpose(C[..., mask], (4, 0, 1, 2, 3)), fs[mask]


def calculate_cross_coherence(time_series, nfft):
    """
    # type: (TimeSeries, int)  -> CoherenceSpectrum
    # Adapter for cross-coherence algorithm(s)
    # Evaluate coherence on time series.

    Parameters
    __________
    time_series : TimeSeries
    The TimeSeries to which the Cross Coherence is to be applied.

    nfft : int
    Data-points per block (should be a power of 2).
    """

    srate = time_series.sample_rate
    coh, freq = _coherence(time_series.data, srate, nfft=nfft)
    log.debug("coherence")
    log.debug(narray_describe(coh))
    log.debug("freq")
    log.debug(narray_describe(freq))

    spec = spectral.CoherenceSpectrum(
        source=time_series,
        nfft=nfft,
        array_data=coh.astype(numpy.float),
        frequency=freq)
    return spec

