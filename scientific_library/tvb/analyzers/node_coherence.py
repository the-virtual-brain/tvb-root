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
Compute cross coherence between all nodes in a time series.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import numpy
import matplotlib.mlab as mlab
from matplotlib.pylab import detrend_linear
import tvb.datatypes.time_series as time_series
import tvb.datatypes.spectral as spectral
from tvb.basic.neotraits.api import HasTraits, Attr, Int, narray_describe


# TODO: Should do this properly, ie not with mlab, returning both coherence and
#      the complex coherence spectra, then supporting magnitude squared
#      coherence, etc in a similar fashion to the FourierSpectrum datatype...


def hamming(M, sym=True):
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


def coherence(data, sample_rate, nfft=256, imag=False):
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
    wins *= hamming(nfft)
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


class NodeCoherence(HasTraits):
    "Adapter for cross-coherence algorithm(s)"

    time_series = Attr(
        field_type=time_series.TimeSeries,
        label="Time Series",
        required=True,
        doc="""The timeseries to which the FFT is to be applied.""")

    nfft = Int(
        label="Data-points per block",
        default=256,
        doc="""Should be a power of 2...""")

    def evaluate(self):
        "Evaluate coherence on time series."
        cls_attr_name = self.__class__.__name__+".time_series"
        # self.time_series.trait["data"].log_debug(owner=cls_attr_name)
        srate = self.time_series.sample_rate
        coh, freq = coherence(self.time_series.data, srate, nfft=self.nfft)
        self.log.debug("coherence")
        self.log.debug(narray_describe(coh))
        self.log.debug("freq")
        self.log.debug(narray_describe(freq))

        spec = spectral.CoherenceSpectrum(
            source=self.time_series,
            nfft=self.nfft,
            array_data=coh,
            frequency=freq)
        return spec

    def result_shape(self, input_shape):
        """Returns the shape of the main result of NodeCoherence."""
        freq_len = self.nfft/2 + 1
        freq_shape = (freq_len,)
        result_shape = (freq_len, input_shape[2], input_shape[2], input_shape[1], input_shape[3])
        return [result_shape, freq_shape]

    def result_size(self, input_shape):
        """
        Returns the storage size in Bytes of the main result of NodeCoherence.
        """
        # TODO This depends on input array dtype!
        result_size = numpy.sum(list(map(numpy.prod, self.result_shape(input_shape)))) * 8.0 #Bytes
        return result_size

    def extended_result_size(self, input_shape):
        """
        Returns the storage size in Bytes of the extended result of the FFT.
        That is, it includes storage of the evaluated FourierSpectrum attributes
        such as power, phase, amplitude, etc.
        """
        extend_size = self.result_size(input_shape) #Currently no derived attributes.
        return extend_size
