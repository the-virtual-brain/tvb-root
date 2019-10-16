# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Adapter that uses the traits module to generate interfaces for FFT Analyzer.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""
import os
import uuid
import psutil
import numpy
import math
import tvb.analyzers.fft as fft
import tvb.core.adapters.abcadapter as abcadapter
import tvb.basic.filters.chain as entities_filter
import tvb.datatypes.spectral as spectral
from tvb.basic.logger.builder import get_logger
from tvb.datatypes.time_series import TimeSeries

from tvb.core.entities.file.datatypes.spectral_h5 import FourierSpectrumH5
from tvb.core.entities.file.datatypes.time_series import TimeSeriesH5
from tvb.core.entities.model.datatypes.spectral import FourierSpectrumIndex
from tvb.core.entities.model.datatypes.time_series import TimeSeriesIndex
from tvb.core.neotraits._forms import ScalarField, TimeSeriesSelectField
from tvb.interfaces.neocom.h5 import DirLoader

LOG = get_logger(__name__)

class FFTAdapterForm(abcadapter.ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(FFTAdapterForm, self).__init__(prefix)
        self.time_series = TimeSeriesSelectField(fft.FFT.time_series, self.get_required_datatype(), self)
        self.segment_length = ScalarField(fft.FFT.segment_length, self)
        self.window_function = ScalarField(fft.FFT.window_function, self)
        self.detrend = ScalarField(fft.FFT.detrend, self)
        self.project_id = project_id

    @staticmethod
    def get_required_datatype():
        return TimeSeriesIndex

    @staticmethod
    def get_filters():
        return entities_filter.FilterChain(fields=["NArrayIndex.ndim"], operations=["=="], values=[4])

    @staticmethod
    def get_input_name():
        return "time_series"

    def get_traited_datatype(self):
        return fft.FFT()


class FourierAdapter(abcadapter.ABCAsynchronous):
    """ TVB adapter for calling the FFT algorithm. """

    _ui_name = "Fourier Spectral Analysis"
    _ui_description = "Calculate the FFT of a TimeSeries entity."
    _ui_subsection = "fourier"

    form = None

    def get_input_tree(self):
        return None

    def get_form(self):
        if self.form is None:
            return FFTAdapterForm
        return self.form

    def set_form(self, form):
        # TODO: ensure correct form type
        self.form = form

    def get_algorithm(self):
        return self.algorithm

    def get_output(self):
        return [spectral.FourierSpectrum]


    def __init__(self):
        super(FourierAdapter, self).__init__()
        self.algorithm = fft.FFT()
        self.memory_factor = 1


    def configure(self, time_series, segment_length=None, window_function=None, detrend=None):
        """
        Do any configuration needed before launching.

        :param time_series: the input time series to which the fft is to be applied
        :param segment_length: the block size which determines the frequency resolution \
                               of the resulting power spectra
        :param window_function: windowing functions can be applied before the FFT is performed
        :type  window_function: None; ‘hamming’; ‘bartlett’; ‘blackman’; ‘hanning’
        :param detrend: None; specify if detrend is performed on the time series
        """
        self.input_time_series_index = time_series
        self.input_shape = (self.input_time_series_index.data.length_1d,
                            self.input_time_series_index.data.length_2d,
                            self.input_time_series_index.data.length_3d,
                            self.input_time_series_index.data.length_4d)

        LOG.debug("time_series shape is %s" % str(self.input_shape))
        LOG.debug("Provided segment_length is %s" % segment_length)
        LOG.debug("Provided window_function is %s" % window_function)
        LOG.debug("Detrend is %s" % detrend)
        # -------------------- Fill Algorithm for Analysis -------------------
        # The enumerate set function isn't working well. A get around strategy is to create a new algorithm
        if segment_length is not None:
            self.algorithm.segment_length = segment_length

        self.algorithm.window_function = window_function
        self.algorithm.detrend = detrend

        LOG.debug("Using segment_length is %s" % self.algorithm.segment_length)
        LOG.debug("Using window_function  is %s" % self.algorithm.window_function)
        LOG.debug("Using detrend  is %s" % self.algorithm.detrend)


    def get_required_memory_size(self, time_series, segment_length=None, window_function=None, detrend=None):
        """
        Returns the required memory to be able to run the adapter.
        """
        input_size = numpy.prod(self.input_shape) * 8.0
        output_size = self.algorithm.result_size(self.input_shape, self.algorithm.segment_length,
                                                 self.input_time_series_index.sample_period)
        total_free_memory = psutil.virtual_memory().free + psutil.swap_memory().free
        total_required_memory = input_size + output_size
        while total_required_memory / self.memory_factor / total_free_memory > 0.8:
            self.memory_factor += 1
        return total_required_memory / self.memory_factor


    def get_required_disk_size(self, time_series, segment_length=None, window_function=None, detrend=None):
        """
        Returns the required disk size to be able to run the adapter (in kB).
        """
        output_size = self.algorithm.result_size(self.input_shape, self.algorithm.segment_length,
                                                 self.input_time_series_index.sample_period)
        return self.array_size2kb(output_size)


    def launch(self, time_series, segment_length=None, window_function=None, detrend=None):
        """
        Launch algorithm and build results.

        :param time_series: the input time series to which the fft is to be applied
        :param segment_length: the block size which determines the frequency resolution \
                               of the resulting power spectra
        :param window_function: windowing functions can be applied before the FFT is performed
        :type  window_function: None; ‘hamming’; ‘bartlett’; ‘blackman’; ‘hanning’
        :returns: the fourier spectrum for the specified time series
        :rtype: `FourierSpectrum`

        """
        fft_index = FourierSpectrumIndex()
        fft_index.source = self.input_time_series_index

        block_size = int(math.floor(self.input_shape[2] / self.memory_factor))
        blocks = int(math.ceil(self.input_shape[2] / block_size))

        # TODO: handle this in generic manner
        loader = DirLoader(os.path.join(os.path.dirname(self.storage_path), str(self.input_time_series_index.fk_from_operation)))
        input_path = loader.path_for(TimeSeriesH5, self.input_time_series_index.gid)
        input_time_series_h5 = TimeSeriesH5(input_path)

        loader = DirLoader(self.storage_path)
        dest_path = loader.path_for(FourierSpectrumH5, fft_index.gid)
        spectra_file = FourierSpectrumH5(dest_path)
        spectra_file.gid.store(uuid.UUID(fft_index.gid))
        spectra_file.source.store(uuid.UUID(self.input_time_series_index.gid))

        # ------------- NOTE: Assumes 4D, Simulator timeSeries. --------------
        node_slice = [slice(self.input_shape[0]), slice(self.input_shape[1]), None, slice(self.input_shape[3])]

        # ---------- Iterate over slices and compose final result ------------
        small_ts = TimeSeries()
        small_ts.sample_period = input_time_series_h5.sample_period.load()

        for block in xrange(blocks):
            node_slice[2] = slice(block * block_size, min([(block+1) * block_size, self.input_shape[2]]), 1)
            small_ts.data = input_time_series_h5.read_data_slice(tuple(node_slice))
            self.algorithm.time_series = small_ts
            partial_result = self.algorithm.evaluate()

            if blocks <= 1 and len(partial_result.array_data) == 0:
                self.add_operation_additional_info(
                    "Fourier produced empty result (most probably due to a very short input TimeSeries).")
                return None
            spectra_file.write_data_slice(partial_result)
        input_time_series_h5.close()

        fft_index.windowing_function = self.algorithm.window_function
        fft_index.segment_length = self.algorithm.segment_length
        fft_index.detrend = self.algorithm.detrend
        fft_index.frequency_step = partial_result.freq_step
        fft_index.max_frequency = partial_result.max_freq

        spectra_file.segment_length.store(self.algorithm.segment_length)
        spectra_file.windowing_function.store(str(self.algorithm.window_function))
        spectra_file.close()

        LOG.debug("partial segment_length is %s" % (str(partial_result.segment_length)))

        return fft_index

