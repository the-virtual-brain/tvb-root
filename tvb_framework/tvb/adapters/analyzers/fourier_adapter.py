# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
Adapter that uses the traits module to generate interfaces for FFT Analyzer.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""
import math
import uuid
import numpy
import psutil

from tvb.adapters.datatypes.db.spectral import FourierSpectrumIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.adapters.datatypes.h5.spectral_h5 import FourierSpectrumH5
from tvb.analyzers.fft import compute_fast_fourier_transform
from tvb.basic.neotraits.api import Attr, EnumAttr, Float
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField, SelectField, FloatField, BoolField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.spectral import WindowingFunctionsEnum
from tvb.datatypes.time_series import TimeSeries


class FFTAdapterModel(ViewModel):
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

    segment_length = Float(
        label="Segment(window) length (ms)",
        default=1000.0,
        required=False,
        doc="""The TimeSeries can be segmented into equally sized blocks
            (overlapping if necessary). The segment length determines the
            frequency resolution of the resulting power spectra -- longer
            windows produce finer frequency resolution.""")

    window_function = EnumAttr(
        default=WindowingFunctionsEnum.HAMMING,
        label="Windowing function",
        required=False,
        doc="""Windowing functions can be applied before the FFT is performed.
             Default is None, possibilities are: 'hamming'; 'bartlett';
            'blackman'; and 'hanning'. See, numpy.<function_name>.""")

    detrend = Attr(
        field_type=bool,
        label="Detrending",
        default=True,
        required=False,
        doc="""Detrending is not always appropriate.
            Default is True, False means no detrending is performed on the time series""")


class FFTAdapterForm(ABCAdapterForm):

    def __init__(self):
        super(FFTAdapterForm, self).__init__()
        self.time_series = TraitDataTypeSelectField(FFTAdapterModel.time_series, name='time_series',
                                                    conditions=self.get_filters(), has_all_option=True)
        self.segment_length = FloatField(FFTAdapterModel.segment_length)
        self.window_function = SelectField(FFTAdapterModel.window_function)
        self.detrend = BoolField(FFTAdapterModel.detrend)

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


class FourierAdapter(ABCAdapter):
    """ TVB adapter for calling the FFT algorithm. """

    _ui_name = "Fourier Spectral Analysis"
    _ui_description = "Calculate the FFT of a TimeSeries entity."
    _ui_subsection = "fourier"

    def __init__(self):
        super(FourierAdapter, self).__init__()
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

    def get_required_memory_size(self, view_model):
        # type: (FFTAdapterModel) -> int
        """
        Returns the required memory to be able to run the adapter.
        """
        input_size = numpy.prod(self.input_shape) * 8.0
        output_size = self.result_size(self.input_shape, view_model.segment_length,
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
        output_size = self.result_size(self.input_shape, view_model.segment_length,
                                       self.input_time_series_index.sample_period)
        return self.array_size2kb(output_size)

    def launch(self, view_model):
        # type: (FFTAdapterModel) -> [FourierSpectrumIndex]
        """
        Launch algorithm and build results.
        :param view_model: the ViewModel keeping the algorithm inputs
        :return: the fourier spectrum for the specified time series
        """
        block_size = int(math.floor(self.input_shape[2] / self.memory_factor))
        blocks = int(math.ceil(self.input_shape[2] / block_size))

        input_time_series_h5 = h5.h5_file_for_index(self.input_time_series_index)

        # --------------------- Prepare result entities ----------------------
        fft_index = FourierSpectrumIndex()
        dest_path = self.path_for(FourierSpectrumH5, fft_index.gid)
        spectra_file = FourierSpectrumH5(dest_path)

        # ------------- NOTE: Assumes 4D, Simulator timeSeries. --------------
        node_slice = [slice(self.input_shape[0]), slice(self.input_shape[1]), None, slice(self.input_shape[3])]

        # ---------- Iterate over slices and compose final result ------------
        small_ts = TimeSeries()
        small_ts.sample_period = input_time_series_h5.sample_period.load()
        small_ts.sample_period_unit = input_time_series_h5.sample_period_unit.load()

        for block in range(blocks):
            node_slice[2] = slice(block * block_size, min([(block + 1) * block_size, self.input_shape[2]]), 1)
            small_ts.data = input_time_series_h5.read_data_slice(tuple(node_slice))

            partial_result = compute_fast_fourier_transform(small_ts, view_model.segment_length,
                                                            view_model.window_function, view_model.detrend)

            if blocks <= 1 and len(partial_result.array_data) == 0:
                self.add_operation_additional_info(
                    "Fourier produced empty result (most probably due to a very short input TimeSeries).")
                return None
            spectra_file.write_data_slice(partial_result)

        input_time_series_h5.close()

        # ---------------------------- Fill results ----------------------------
        partial_result.source.gid = view_model.time_series
        partial_result.gid = uuid.UUID(fft_index.gid)

        fft_index.fill_from_has_traits(partial_result)
        self.fill_index_from_h5(fft_index, spectra_file)

        spectra_file.store(partial_result, scalars_only=True)
        spectra_file.windowing_function.store(view_model.window_function)
        spectra_file.close()

        self.log.debug("partial segment_length is %s" % (str(partial_result.segment_length)))
        return fft_index

    @staticmethod
    def result_shape(input_shape, segment_length, sample_period):
        """Returns the shape of the main result (complex array) of the FFT."""
        freq_len = (segment_length / sample_period) / 2.0
        freq_len = int(min((input_shape[0], freq_len)))
        nseg = max((1, int(numpy.ceil(input_shape[0] * sample_period / segment_length))))
        result_shape = (freq_len, input_shape[1], input_shape[2], input_shape[3], nseg)
        return result_shape

    def result_size(self, input_shape, segment_length, sample_period):
        """
        Returns the storage size in Bytes of the main result (complex array) of
        the FFT.
        """
        result_size = numpy.prod(self.result_shape(input_shape, segment_length,
                                                   sample_period)) * 2.0 * 8.0  # complex*Bytes
        return result_size
