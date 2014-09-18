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
Compute cross coherence between all nodes in a time series.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Marmaduke Woodman <maramduke.woodman@univ-amu.fr>

"""

import numpy
import matplotlib.mlab as mlab
from matplotlib.pylab import detrend_linear
#TODO: Currently built around the Simulator's 4D timeseries -- generalise...
import tvb.datatypes.time_series as time_series
import tvb.datatypes.spectral as spectral
import tvb.basic.traits.core as core
import tvb.basic.traits.types_basic as basic
import tvb.basic.traits.util as util
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)

#TODO: Make an appropriate spectral datatype for the output
#TODO: Should do this properly, ie not with mlab, returning both coherence and
#      the complex coherence spectra, then supporting magnitude squared
#      coherence, etc in a similar fashion to the FourierSpectrum datatype...

try:
    from scipy.signal import hamming
except ImportError:
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


class NodeCoherence(core.Type):
    """Compute cross coherence between nodes.
    """
    
    time_series = time_series.TimeSeries(
        label = "Time Series",
        required = True,
        doc = """The timeseries to which the FFT is to be applied.""")
        
    nfft = basic.Integer(
        label="Data-points per block",
        default = 256,
        doc="""Should be a power of 2...""")
    

    def evaluate(self):
        """ 
        Coherence function.  Matplotlib.mlab implementation.
        """
        cls_attr_name = self.__class__.__name__+".time_series"
        self.time_series.trait["data"].log_debug(owner = cls_attr_name)
        
        data_shape = self.time_series.data.shape
        
        #(frequency, nodes, nodes, state-variables, modes)
        result_shape = (self.nfft/2 + 1, data_shape[2], data_shape[2], data_shape[1], data_shape[3])
        LOG.info("result shape will be: %s" % str(result_shape))
        
        result = numpy.zeros(result_shape)
        
        #TODO: For region level, 4s, 2000Hz, this takes ~2min... (which is stupidly slow) 
        #One inter-node coherence, across frequencies for each state-var & mode.
        for mode in range(data_shape[3]):
            for var in range(data_shape[1]):
                data = self.time_series.data[:, var, :, mode]
                data = data - data.mean(axis=0)[numpy.newaxis, :]
                #TODO: Work out a way around the 4 level loop,
                #TODO: coherence isn't directional, so, get rid of redundancy...
                for n1 in range(data_shape[2]):
                    for n2 in range(data_shape[2]):
                        cxy, freq = mlab.cohere(data[:, n1], data[:, n2],
                                                NFFT = self.nfft,
                                                Fs = self.time_series.sample_rate,
                                                detrend = detrend_linear,
                                                window = mlab.window_none)
                        result[:, n1, n2, var, mode] = cxy
        
        util.log_debug_array(LOG, result, "result")
        util.log_debug_array(LOG, freq, "freq")
        
        coherence = spectral.CoherenceSpectrum(source = self.time_series,
                                               nfft = self.nfft,
                                               array_data = result,
                                               frequency = freq,
                                               use_storage = False)
        
        return coherence

    def evaluate_new(self):
        import pydevd
        pydevd.settrace('localhost', port=51234, stdoutToServer=True, stderrToServer=True)
        data = self.time_series.data
        sample_period = self.time_series.sample_period
        nfft = self.nfft

        result_shape = (nfft/2 - 1, data.shape[2], data.shape[2], data.shape[1], data.shape[3])
        LOG.info("result shape will be: %s" % str(result_shape))

        result = numpy.zeros(result_shape)
        for mode in range(data.shape[3]):
            for var in range(data.shape[1]):
                data = self.time_series.data[:, var, :, mode]
                freq, _, cxy = self._new_evaluate_inner(data.T, nfft, sample_period)
                result[..., var, mode] = cxy.transpose(2, 0, 1)

        coherence = spectral.CoherenceSpectrum(source = self.time_series,
                                               nfft = self.nfft,
                                               array_data = result,
                                               frequency = freq,
                                               use_storage = False)

        return coherence

    @staticmethod
    def _new_evaluate_inner(Y, nfft, sample_period):
        "New implementation of cross-coherence w/o for loops"
        # TODO: adapt to tvb timeseries shape
        imag = False
        fs = numpy.fft.fftfreq(nfft, sample_period)
        # shape [ch_i, ch_j, ..., window, time]
        wY = Y.reshape((Y.shape[0], -1, nfft)) * hamming(nfft)
        F = numpy.fft.fft(wY)
        G = F[:, numpy.newaxis]*F
        if imag:
            G = G.imag
        dG = numpy.array([G[i, i] for i in range(G.shape[0])])
        C = (numpy.abs(G)**2 / (dG[:, numpy.newaxis] * dG)).mean(axis=-2)
        mask = fs>0.0
        C_ = numpy.abs(C.mean(axis=0).mean(axis=0))
        return fs[mask], C_[mask], C[..., mask]

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
        result_size = numpy.sum(map(numpy.prod, self.result_shape(input_shape))) * 8.0 #Bytes
        return result_size

    def extended_result_size(self, input_shape):
        """
        Returns the storage size in Bytes of the extended result of the FFT.
        That is, it includes storage of the evaluated FourierSpectrum attributes
        such as power, phase, amplitude, etc.
        """
        extend_size = self.result_size(input_shape) #Currently no derived attributes.
        return extend_size
