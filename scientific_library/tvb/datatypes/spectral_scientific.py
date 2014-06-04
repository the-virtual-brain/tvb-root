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

Scientific methods for the Spectral datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import numpy
import tvb.basic.traits.util as util
import tvb.datatypes.spectral_data as spectral_data
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)


class FourierSpectrumScientific(spectral_data.FourierSpectrumData):
    """ This class exists to add scientific methods to FourierSpectrumData. """
    __tablename__ = None
    
    _frequency = None
    _freq_step = None
    _max_freq = None
    
    
    def configure(self):
        """
        Invoke the compute methods for computable attributes that haven't been
        set during initialisation.
        """
        super(FourierSpectrumScientific, self).configure()
        
        if self.trait.use_storage is False and sum(self.get_data_shape('array_data')) != 0:
            if self.amplitude.size == 0:
                self.compute_amplitude()
            
            if self.phase.size == 0:
                self.compute_phase()
            
            if self.power.size == 0:
                self.compute_power()
            
            if self.average_power.size == 0:
                self.compute_average_power()
            
            if self.normalised_average_power.size == 0:
                self.compute_normalised_average_power()
    
    
    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = {"Spectral type": self.__class__.__name__,
                   "Source": self.source.title,
                   "Segment length": self.segment_length,
                   "Windowing function": self.windowing_function,
                   "Frequency step": self.freq_step,
                   "Maximum frequency": self.max_freq}
        return summary
    
    
    @property
    def freq_step(self):
        """ Frequency step size of the complex Fourier spectrum."""
        if self._freq_step is None:
            self._freq_step = 1.0 / self.segment_length
            msg = "%s: Frequency step size is %s"
            LOG.debug(msg % (str(self), str(self._freq_step)))
        return self._freq_step
    
    
    @property
    def max_freq(self):
        """ Amplitude of the complex Fourier spectrum."""
        if self._max_freq is None:
            self._max_freq = 0.5 / self.source.sample_period
            msg = "%s: Max frequency is %s"
            LOG.debug(msg % (str(self), str(self._max_freq)))
        return self._max_freq
    
    
    @property
    def frequency(self):
        """ Frequencies represented the complex Fourier spectrum."""
        if self._frequency is None:
            self._frequency = numpy.arange(self.freq_step, 
                                           self.max_freq + self.freq_step,
                                           self.freq_step)
            util.log_debug_array(LOG, self._frequency, "frequency")
        return self._frequency
    
    
    def compute_amplitude(self):
        """ Amplitude of the complex Fourier spectrum."""
        self.amplitude = numpy.abs(self.array_data)
        self.trait["amplitude"].log_debug(owner=self.__class__.__name__)
    
    
    def compute_phase(self):
        """ Phase of the Fourier spectrum."""
        self.phase = numpy.angle(self.array_data)
        self.trait["phase"].log_debug(owner=self.__class__.__name__)
    
    
    def compute_power(self):
        """ Power of the complex Fourier spectrum."""
        self.power = numpy.abs(self.array_data) ** 2
        self.trait["power"].log_debug(owner=self.__class__.__name__)
    
    
    def compute_average_power(self):
        """ Average-power of the complex Fourier spectrum."""
        self.average_power = numpy.mean(numpy.abs(self.array_data) ** 2, axis=-1)
        self.trait["average_power"].log_debug(owner=self.__class__.__name__)
    
    
    def compute_normalised_average_power(self):
        """ Normalised-average-power of the complex Fourier spectrum."""
        self.normalised_average_power = (self.average_power / 
                                         numpy.sum(self.average_power, axis=0))
        self.trait["normalised_average_power"].log_debug(owner=self.__class__.__name__)



class WaveletCoefficientsScientific(spectral_data.WaveletCoefficientsData):
    """
    This class exists to add scientific methods to WaveletCoefficientsData.
    """
    __tablename__ = None
    _frequency = None
    _time = None
    
    
    def configure(self):
        """
        Invoke the compute methods for computable attributes that haven't been
        set during initialisation.
        """
        super(WaveletCoefficientsScientific, self).configure()
        
        if self.trait.use_storage is False and sum(self.get_data_shape('array_data')) != 0:
            if self.amplitude.size == 0:
                self.compute_amplitude()
            
            if self.phase.size == 0:
                self.compute_phase()
            
            if self.power.size == 0:
                self.compute_power()
    
    
    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = {"Spectral type": self.__class__.__name__,
                   "Source": self.source.title,
                   "Wavelet type": self.mother,
                   "Normalisation": self.normalisation,
                   "Q-ratio": self.q_ratio,
                   "Sample period": self.sample_period,
                   "Number of scales": self.frequencies.shape[0],
                   "Minimum frequency": self.frequencies[0],
                   "Maximum frequency": self.frequencies[-1]}
        return summary
    
    
    @property
    def frequency(self):
        """ Frequencies represented by the wavelet spectrogram."""
        if self._frequency is None:
            self._frequency = numpy.arange(self.frequencies.lo, 
                                           self.frequencies.hi, 
                                           self.frequencies.step)
            util.log_debug_array(LOG, self._frequency, "frequency")
        return self._frequency
    
    
    def compute_amplitude(self):
        """ Amplitude of the complex Wavelet coefficients."""
        self.amplitude = numpy.abs(self.array_data)
        self.trait["amplitude"].log_debug(owner=self.__class__.__name__)
    
    
    def compute_phase(self):
        """ Phase of the Wavelet coefficients."""
        self.phase = numpy.angle(self.array_data)
        self.trait["phase"].log_debug(owner=self.__class__.__name__)
    
    
    def compute_power(self):
        """ Power of the complex Wavelet coefficients."""
        self.power = numpy.abs(self.array_data) ** 2
        self.trait["power"].log_debug(owner=self.__class__.__name__)



class CoherenceSpectrumScientific(spectral_data.CoherenceSpectrumData):
    """ This class exists to add scientific methods to CoherenceSpectrumData. """
    __tablename__ = None
    
    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = {"Spectral type": self.__class__.__name__,
                   "Source": self.source.title,
                   "Number of frequencies": self.frequency.shape[0],
                   "Minimum frequency": self.frequency[0],
                   "Maximum frequency": self.frequency[-1],
                   "FFT length (time-points)": self.nfft}
        return summary
        
        
class ComplexCoherenceSpectrumScientific(spectral_data.ComplexCoherenceSpectrumData):
    """ This class exists to add scientific methods to ComplexCoherenceSpectrumData. """
    __tablename__ = None
    
    _frequency = None
    _freq_step = None
    _max_freq = None
            
    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = {"Spectral type": self.__class__.__name__,
                   "Source": self.source.title,
                   "Frequency step": self.freq_step,
                   "Maximum frequency": self.max_freq}
        #summary["FFT length (time-points)"] = self.fft_points
        #summary["Number of epochs"] = self.number_of_epochs
        return summary
        
    @property
    def freq_step(self):
        """ Frequency step size of the Complex Coherence Spectrum."""
        if self._freq_step is None:
            self._freq_step = 1.0 / self.segment_length
            msg = "%s: Frequency step size is %s"
            LOG.debug(msg % (str(self), str(self._freq_step)))
        return self._freq_step
    
    
    @property
    def max_freq(self):
        """ Maximum frequency represented in the Complex Coherence Spectrum."""
        if self._max_freq is None:
            self._max_freq = 0.5 / self.source.sample_period
            msg = "%s: Max frequency is %s"
            LOG.debug(msg % (str(self), str(self._max_freq)))
        return self._max_freq
    
    
    @property
    def frequency(self):
        """ Frequencies represented in the Complex Coherence Spectrum."""
        if self._frequency is None:
            self._frequency = numpy.arange(self.freq_step, 
                                           self.max_freq + self.freq_step,
                                           self.freq_step)
        util.log_debug_array(LOG, self._frequency, "frequency")
        return self._frequency
            




