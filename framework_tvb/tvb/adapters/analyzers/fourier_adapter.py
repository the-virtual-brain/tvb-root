# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
Adapter that uses the traits module to generate interfaces for FFT Analyzer.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""
import math
import uuid

import numpy
import psutil
import tvb.analyzers.fft as fft
from tvb.adapters.datatypes.db.spectral import FourierSpectrumIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.adapters.datatypes.h5.spectral_h5 import FourierSpectrumH5
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField, SelectField, FloatField, BoolField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.time_series import TimeSeries


class FFTAdapterModel(ViewModel, fft.FFT):
    """
    Parameters have the following meaning:
    - time_series: the input time series to which the fft is to be applied
    - segment_length: the block size which determines the frequency resolution of the resulting power spectra
    - window_function: windowing functions can be applied before the FFT is performed
    - detrend: None; specify if detrend is performed on the time series
    """
    time_series = DataTypeGidAttr(
        linked_datatype=TimeSeries,
        label="Time Series",
        doc="""The TimeSeries to which the FFT is to be applied."""
    )


class FFTAdapterForm(ABCAdapterForm):

    def __init__(self, project_id=None):
        super(FFTAdapterForm, self).__init__(project_id)
        self.time_series = TraitDataTypeSelectField(FFTAdapterModel.time_series, self.project_id, name='time_series',
                                                    conditions=self.get_filters(), has_all_option=True)
        self.segment_length = FloatField(FFTAdapterModel.segment_length, self.project_id)
        self.window_function = SelectField(FFTAdapterModel.window_function, self.project_id)
        self.detrend = BoolField(FFTAdapterModel.detrend, self.project_id)

    @staticmethod
    def get_view_model():
        return FFTAdapterModel

    @staticmethod
    def get_required_datatype():
        return TimeSeriesIndex

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.data_ndim'], operations=["=="], values=[4])

    @staticmethod
    def get_input_name():
        return "time_series"

    def get_traited_datatype(self):
        return fft.FFT()


class FourierAdapter(ABCAdapter):
    """ TVB adapter for calling the FFT algorithm. """

    _ui_name = "Fourier Spectral Analysis"
    _ui_description = "Calculate the FFT of a TimeSeries entity."
    _ui_subsection = "fourier"

    def __init__(self):
        super(FourierAdapter, self).__init__()
        self.algorithm = fft.FFT()
        self.memory_factor = 1

    def get_form_class(self):
        return FFTAdapterForm

    def get_output(self):
        return [FourierSpectrumIndex]

    def configure(self, view_model):
        # type: (FFTAdapterModel) -> None
        """
        Do any configuration needed before launching.
        """
        self.input_time_series_index = self.load_entity_by_gid(view_model.time_series)
        self.input_shape = (self.input_time_series_index.data_length_1d,
                            self.input_time_series_index.data_length_2d,
                            self.input_time_series_index.data_length_3d,
                            self.input_time_series_index.data_length_4d)

        self.log.debug("time_series shape is %s" % str(self.input_shape))
        self.log.debug("Provided segment_length is %s" % view_model.segment_length)
        self.log.debug("Provided window_function is %s" % view_model.window_function)
        self.log.debug("Detrend is %s" % view_model.detrend)
        # -------------------- Fill Algorithm for Analysis -------------------
        # The enumerate set function isn't working well. A get around strategy is to create a new algorithm
        if view_model.segment_length is not None:
            self.algorithm.segment_length = view_model.segment_length

        self.algorithm.window_function = view_model.window_function
        self.algorithm.detrend = view_model.detrend

        self.log.debug("Using segment_length is %s" % self.algorithm.segment_length)
        self.log.debug("Using window_function  is %s" % self.algorithm.window_function)
        self.log.debug("Using detrend  is %s" % self.algorithm.detrend)

    def get_required_memory_size(self, view_model):
        # type: (FFTAdapterModel) -> int
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

    def get_required_disk_size(self, view_model):
        # type: (FFTAdapterModel) -> int
        """
        Returns the required disk size to be able to run the adapter (in kB).
        """
        output_size = self.algorithm.result_size(self.input_shape, self.algorithm.segment_length,
                                                 self.input_time_series_index.sample_period)
        return self.array_size2kb(output_size)

    def launch(self, view_model):
        # type: (FFTAdapterModel) -> [FourierSpectrumIndex]
        """
        Launch algorithm and build results.

        :param time_series: the input time series to which the fft is to be applied
        :param segment_length: the block size which determines the frequency resolution \
                               of the resulting power spectra
        :param window_function: windowing functions can be applied before the FFT is performed
        :type  window_function: None; ‘hamming’; ‘bartlett’; ‘blackman’; ‘hanning’
        :returns: the fourier spectrum for the specified time series
        :rtype: `FourierSpectrumIndex`

        """
        fft_index = FourierSpectrumIndex()
        fft_index.fk_source_gid = view_model.time_series.hex

        block_size = int(math.floor(self.input_shape[2] / self.memory_factor))
        blocks = int(math.ceil(self.input_shape[2] / block_size))

        input_time_series_h5 = h5.h5_file_for_index(self.input_time_series_index)

        dest_path = h5.path_for(self.storage_path, FourierSpectrumH5, fft_index.gid)
        spectra_file = FourierSpectrumH5(dest_path)
        spectra_file.gid.store(uuid.UUID(fft_index.gid))
        spectra_file.source.store(uuid.UUID(self.input_time_series_index.gid))

        # ------------- NOTE: Assumes 4D, Simulator timeSeries. --------------
        node_slice = [slice(self.input_shape[0]), slice(self.input_shape[1]), None, slice(self.input_shape[3])]

        # ---------- Iterate over slices and compose final result ------------
        small_ts = TimeSeries()
        small_ts.sample_period = input_time_series_h5.sample_period.load()
        small_ts.sample_period_unit = input_time_series_h5.sample_period_unit.load()

        for block in range(blocks):
            node_slice[2] = slice(block * block_size, min([(block + 1) * block_size, self.input_shape[2]]), 1)
            small_ts.data = input_time_series_h5.read_data_slice(tuple(node_slice))
            self.algorithm.time_series = small_ts
            partial_result = self.algorithm.evaluate()

            if blocks <= 1 and len(partial_result.array_data) == 0:
                self.add_operation_additional_info(
                    "Fourier produced empty result (most probably due to a very short input TimeSeries).")
                return None
            spectra_file.write_data_slice(partial_result)
        fft_index.ndim = len(spectra_file.array_data.shape)
        input_time_series_h5.close()

        fft_index.windowing_function = self.algorithm.window_function
        fft_index.segment_length = self.algorithm.segment_length
        fft_index.detrend = self.algorithm.detrend
        fft_index.frequency_step = partial_result.freq_step
        fft_index.max_frequency = partial_result.max_freq

        spectra_file.segment_length.store(self.algorithm.segment_length)
        spectra_file.windowing_function.store(str(self.algorithm.window_function))
        spectra_file.close()

        self.log.debug("partial segment_length is %s" % (str(partial_result.segment_length)))
        return fft_index
